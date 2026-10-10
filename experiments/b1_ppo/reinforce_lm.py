"""B1, item 1: A4's REINFORCE on a language model, against a learned reward.

The policy continues the opening words of an IMDB review; a sentiment
classifier (the proxy reward model) scores the full text once at the end.
There is no KL term by default: nothing ties the policy to the pretrained
model, which is the point. A second, larger sentiment classifier that the
policy never trains against is the held-out "gold" judge, so a proxy gain
that does not carry over is visible as such.

    python experiments/b1_ppo/reinforce_lm.py --seed 0                    # one run
    python experiments/b1_ppo/reinforce_lm.py --seed 0 --kl-coef 0.05     # with a KL penalty
    python experiments/b1_ppo/reinforce_lm.py --smoke                     # CPU check, tiny model

The estimator is A4's: per-token returns from `discounted_returns` (γ = 1,
one terminal reward, so every token's return is the sequence reward), minus
a baseline, times the token log-probabilities. The policy is the base model
with LoRA adapters; the reference is the same model with the adapters
switched off, so no second copy is loaded.

Logged every step: proxy reward, gold P(positive), the KL to the reference
(sum over completion tokens of log π − log π_ref), the reference model's
per-token NLL of the samples (fluency as the pretrained model sees it),
distinct-1/2 and the repeated-trigram rate (diversity), completion length.
Every `--eval-every` steps the same 64 held-out prompts are sampled with a
fixed seed and a few texts are saved, so degeneration can be read, not only
plotted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "experiments" / "a4_policy_gradient")]

from pg import discounted_returns  # noqa: E402  (A4's estimator, reused)
from rl_ladder import Run, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--reward-model", default="lvwerra/distilbert-imdb")
    p.add_argument("--gold-model", default="siebert/sentiment-roberta-large-english")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--kl-coef", type=float, default=0.0, help="β in r − β·KL; 0 = no anchor")
    p.add_argument("--baseline", choices=("none", "batch-mean"), default="batch-mean")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-prompts", type=int, default=64)
    p.add_argument("--name", default=None)
    p.add_argument("--smoke", action="store_true", help="tiny model, 3 steps, CPU")
    a = p.parse_args()
    if a.smoke:
        a.policy, a.steps, a.batch, a.max_new_tokens = "HuggingFaceTB/SmolLM2-135M", 3, 4, 12
        a.eval_every, a.eval_prompts, a.gold_model = 2, 4, "lvwerra/distilbert-imdb"
    return a


# ------------------------------------------------------------------ prompts

def imdb_prompts(split: str, n: int, tokenizer, rng: np.random.Generator) -> list[str]:
    """The opening 4-8 tokens of IMDB reviews, as text."""
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/imdb", split=split).shuffle(seed=int(rng.integers(1 << 31)))
    out = []
    for text in ds["text"]:
        text = re.sub(r"<br\s*/?>", " ", text).strip()
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        k = int(rng.integers(4, 9))
        if len(ids) >= k:
            out.append(tokenizer.decode(ids[:k]))
        if len(out) == n:
            return out
    return out


# ------------------------------------------------------------------ scoring

class Classifier:
    """A sentiment classifier as a reward: log-odds of positive, and P(positive)."""

    def __init__(self, name: str, device: str):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.pos, self.neg = labels["positive"], labels["negative"]
        self.device = device

    @torch.no_grad()
    def __call__(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        enc = self.tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
        logits = self.model(**enc).logits.float()
        log_odds = (logits[:, self.pos] - logits[:, self.neg]).cpu().numpy()
        p_pos = torch.softmax(logits, -1)[:, self.pos].cpu().numpy()
        return log_odds, p_pos


def diversity(token_lists: list[list[int]]) -> dict[str, float]:
    uni, bi, n_uni, n_bi, rep, n_tri = set(), set(), 0, 0, 0, 0
    for t in token_lists:
        uni.update(t)
        n_uni += len(t)
        b = list(zip(t, t[1:]))
        bi.update(b)
        n_bi += len(b)
        tri = list(zip(t, t[1:], t[2:]))
        rep += len(tri) - len(set(tri))
        n_tri += len(tri)
    return {"distinct1": len(uni) / max(n_uni, 1), "distinct2": len(bi) / max(n_bi, 1),
            "repeat_trigram": rep / max(n_tri, 1)}


# ------------------------------------------------------------------ the loop

def completion_logprobs(model, seq: torch.Tensor, attn: torch.Tensor, n_prompt: int) -> torch.Tensor:
    """Per-token log π(completion token) for each sequence, shape (batch, new tokens)."""
    logits = model(input_ids=seq, attention_mask=attn).logits[:, n_prompt - 1:-1].float()
    return torch.log_softmax(logits, -1).gather(-1, seq[:, n_prompt:, None]).squeeze(-1)


def sample(model, tok, prompts: list[str], max_new: int, device: str, generator_seed: int | None = None):
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    if generator_seed is not None:
        torch.manual_seed(generator_seed)
    with torch.no_grad():
        seq = model.generate(**enc, do_sample=True, top_k=0, top_p=1.0, max_new_tokens=max_new,
                             pad_token_id=tok.pad_token_id)
    n_prompt = enc["input_ids"].shape[1]
    new = seq[:, n_prompt:]
    # Everything after the first EOS is padding: mask it out of the loss and the metrics.
    is_eos = new == tok.eos_token_id
    after = (is_eos.cumsum(1) - is_eos.long()) > 0
    mask = (~after).long()
    attn = torch.cat([enc["attention_mask"], mask], 1)
    texts = tok.batch_decode(new * mask + tok.pad_token_id * (1 - mask), skip_special_tokens=True)
    tokens = [row[m.bool()].tolist() for row, m in zip(new, mask)]
    return seq, attn, mask, n_prompt, texts, tokens


def main() -> None:
    args = parse_args()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = args.name or f"reinforce-kl{args.kl_coef:g}-s{args.seed}"
    run = Run(phase="b1", name=name, config=vars(args))

    tok = AutoTokenizer.from_pretrained(args.policy, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # fp32 weights with fp16 autocast on the GPU: the LoRA parameters stay fp32,
    # which the grad scaler needs (T4 has no bf16).
    base = AutoModelForCausalLM.from_pretrained(args.policy, dtype=torch.float32, attn_implementation="sdpa")
    model = get_peft_model(base, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                                            target_modules="all-linear", task_type="CAUSAL_LM")).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    proxy, gold = Classifier(args.reward_model, device), Classifier(args.gold_model, device)

    train_prompts = imdb_prompts("train", args.steps * args.batch, tok, rng)
    eval_prompts = imdb_prompts("test", args.eval_prompts, tok, np.random.default_rng(12345))
    samples_path = run.path("samples.jsonl")
    t0 = time.time()

    def score(prompts, seq, attn, mask, n_prompt, texts, tokens) -> dict[str, np.ndarray | float]:
        full = [p + t for p, t in zip(prompts, texts)]
        r_proxy, p_proxy = proxy(full)
        _, p_gold = gold(full)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            with model.disable_adapter():
                ref_lp = completion_logprobs(model, seq, attn, n_prompt)
            pol_lp = completion_logprobs(model, seq, attn, n_prompt)
        m = mask.float()
        kl = ((pol_lp - ref_lp) * m).sum(1).cpu().numpy()
        ref_nll = float(-(ref_lp * m).sum() / m.sum().clamp(min=1))
        return {"r_proxy": r_proxy, "p_proxy": p_proxy, "p_gold": p_gold, "kl": kl, "ref_nll": ref_nll,
                "length": m.sum(1).cpu().numpy(), **diversity(tokens)}

    def summarise(prefix: str, s: dict) -> dict[str, float]:
        return {f"{prefix}{k}": float(np.mean(v)) for k, v in s.items()}

    def evaluate(step: int) -> None:
        model.eval()
        out = sample(model, tok, eval_prompts, args.max_new_tokens, device, generator_seed=999)
        s = score(eval_prompts, *out)
        run.log(step=step, **summarise("eval_", s))
        with open(samples_path, "a") as f:
            for p, t, r, g in list(zip(eval_prompts, out[4], s["r_proxy"], s["p_gold"]))[:6]:
                f.write(json.dumps({"step": step, "prompt": p, "completion": t,
                                    "proxy_log_odds": float(r), "gold_p_pos": float(g)}) + "\n")
        model.train()

    evaluate(0)
    for step in range(1, args.steps + 1):
        prompts = train_prompts[(step - 1) * args.batch: step * args.batch]
        model.eval()
        seq, attn, mask, n_prompt, texts, tokens = sample(model, tok, prompts, args.max_new_tokens, device)
        s = score(prompts, seq, attn, mask, n_prompt, texts, tokens)
        model.train()

        # Sequence reward, optionally shaped by the KL to the reference (RLHF's r − β·KL).
        reward = s["r_proxy"] - args.kl_coef * s["kl"]
        # A4's per-token returns: zero reward until the last token, γ = 1.
        returns = [discounted_returns([0.0] * (int(n) - 1) + [float(r)], 1.0) if n > 0 else np.zeros(0)
                   for r, n in zip(reward, s["length"])]
        b = float(np.mean(reward)) if args.baseline == "batch-mean" else 0.0
        adv = torch.zeros_like(mask, dtype=torch.float32)
        for i, g in enumerate(returns):
            adv[i, : len(g)] = torch.as_tensor(g - b, dtype=torch.float32)

        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            lp = completion_logprobs(model, seq, attn, n_prompt)
        loss = -(adv.to(device) * lp * mask).sum() / len(prompts)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        scaler.step(opt)
        scaler.update()

        run.log(step=step, loss=loss.item(), grad_norm=grad_norm, reward=float(np.mean(reward)),
                seconds=time.time() - t0, **summarise("train_", s))
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step)
        if step % 10 == 0:
            print(f"step {step} proxy {np.mean(s['r_proxy']):+.2f} gold {np.mean(s['p_gold']):.2f} "
                  f"kl {np.mean(s['kl']):.1f} d2 {s['distinct2']:.2f} {time.time() - t0:.0f}s", flush=True)

    run.finish({"final_eval": summarise("", score(eval_prompts, *sample(
        model, tok, eval_prompts, args.max_new_tokens, device, generator_seed=999)))})


if __name__ == "__main__":
    main()
