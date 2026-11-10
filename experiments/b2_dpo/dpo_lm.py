"""B2: DPO and IPO on B1's task, written out (no TRL), with every log-probability logged.

Policy and reference as in B1: Qwen2.5-0.5B with LoRA adapters (r = 16, all
linear layers); the reference is the same model with the adapters off. The
pairs come from `make_prefs.py`: K reference samples per prompt scored by
B1's reward model; by default chosen = the best-scored sample, rejected =
the worst.

With h = [log π(y_w|x) − log π_ref(y_w|x)] − [log π(y_l|x) − log π_ref(y_l|x)]
(sequence log-probabilities, summed over completion tokens):

    DPO   −log σ(β·h)                 unbounded: the loss keeps falling as h → ∞
    IPO   (h − 1/(2τ))²               bounded: the optimum is h = 1/(2τ); --beta is τ

    python experiments/b2_dpo/dpo_lm.py --prefs runs/b2/prefs.jsonl.gz --rm $RM_DIR --loss dpo --beta 0.1
    python experiments/b2_dpo/dpo_lm.py ... --pairs 64 --epochs 100        # item 2: overfit a small set
    python experiments/b2_dpo/dpo_lm.py ... --loss ipo --beta 0.1          # item 3
    python experiments/b2_dpo/dpo_lm.py --smoke --prefs /tmp/prefs.jsonl.gz --rm /tmp/rm

Logged every step on the training batch, and every `--eval-every` steps on
up to 256 held-out pairs (test-split prompts): the loss, the preference
accuracy (h > 0), h itself, the chosen and rejected log-ratios to the
reference, and the chosen and rejected log-probabilities per token. Every
`--eval-every` steps, B1's generation evaluation on its 64 held-out
prompts: reward-model score, gold P(positive), KL to the reference
(sampled), distinct-2, reference NLL, the share of samples containing
Chinese, with six texts saved (`samples.jsonl`).
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "experiments" / "b1_ppo"),
                str(REPO / "experiments" / "a4_policy_gradient")]

from ppo_lm import RewardModel  # noqa: E402
from reinforce_lm import Classifier, completion_logprobs, diversity, imdb_prompts, sample  # noqa: E402
from rl_ladder import Run, seed_everything  # noqa: E402

CJK = re.compile(r"[　-〿一-鿿＀-￯]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefs", required=True, help="file written by make_prefs.py")
    p.add_argument("--rm", required=True, help="B1's reward model (for evaluation only)")
    p.add_argument("--policy", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--gold-model", default="siebert/sentiment-roberta-large-english")
    p.add_argument("--loss", choices=("dpo", "ipo"), default="dpo")
    p.add_argument("--beta", type=float, default=0.1, help="DPO's β, or IPO's τ")
    p.add_argument("--pairs", type=int, default=0, help="training pairs to use (0 = all)")
    p.add_argument("--pair-rule", choices=("best-worst", "random"), default="best-worst")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=16, help="pairs per update")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-prompts", type=int, default=64)
    p.add_argument("--eval-pairs", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    p.add_argument("--smoke", action="store_true", help="tiny model, a few steps, CPU")
    a = p.parse_args()
    if a.smoke:
        a.policy, a.gold_model, a.max_new_tokens = "HuggingFaceTB/SmolLM2-135M", "lvwerra/distilbert-imdb", 8
        a.batch, a.eval_every, a.eval_prompts, a.eval_pairs = 2, 2, 4, 4
    return a


# ------------------------------------------------------------------ data

def load_pairs(path: str, rule: str, seed: int) -> dict[str, list[tuple[list[int], list[int], list[int], float]]]:
    """split -> [(prompt ids, chosen ids, rejected ids, reward-model margin)], ties dropped."""
    rng = np.random.default_rng(seed)
    out: dict[str, list] = {"train": [], "test": []}
    with gzip.open(path, "rt") as f:
        for line in f:
            o = json.loads(line)
            r = np.asarray(o["rm"])
            if rule == "best-worst":
                w, l_ = int(r.argmax()), int(r.argmin())
            else:
                w, l_ = rng.choice(len(r), 2, replace=False)
                w, l_ = (w, l_) if r[w] >= r[l_] else (l_, w)
            if r[w] > r[l_]:
                out[o["split"]].append((o["prompt_ids"], o["completions"][w], o["completions"][l_],
                                        float(r[w] - r[l_])))
    return out


def batch_tensors(rows, pad_id: int, device: str):
    """Right-padded prompt+completion ids, attention mask, and a mask over completion tokens.

    Rows are [(prompt ids, completion ids)]; the label mask marks positions whose *next* token is a
    completion token, aligned with logits[:, :-1].
    """
    n = max(len(p) + len(c) for p, c in rows)
    ids = torch.full((len(rows), n), pad_id, dtype=torch.long)
    attn = torch.zeros((len(rows), n), dtype=torch.long)
    lab = torch.zeros((len(rows), n - 1), dtype=torch.float32)
    for i, (p, c) in enumerate(rows):
        seq = p + c
        ids[i, : len(seq)] = torch.as_tensor(seq)
        attn[i, : len(seq)] = 1
        lab[i, len(p) - 1: len(seq) - 1] = 1
    return ids.to(device), attn.to(device), lab.to(device)


def seq_logprobs(model, ids, attn, lab) -> tuple[torch.Tensor, torch.Tensor]:
    """(sum, count) of log π over completion tokens, per row."""
    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
    tgt = ids[:, 1:]
    lp = logits.gather(-1, tgt[..., None]).squeeze(-1) - torch.logsumexp(logits, -1)
    return (lp * lab).sum(1), lab.sum(1)


def preference_loss(kind: str, beta: float, pi_w, pi_l, ref_w, ref_l) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pair loss and h = (log-ratio of chosen) − (log-ratio of rejected)."""
    h = (pi_w - ref_w) - (pi_l - ref_l)
    if kind == "dpo":
        return -F.logsigmoid(beta * h), h
    return (h - 1.0 / (2.0 * beta)) ** 2, h


# ------------------------------------------------------------------ main

def main() -> None:
    args = parse_args()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    half = torch.float16 if use_amp else torch.float32
    pairs = load_pairs(args.prefs, args.pair_rule, args.seed)
    train = pairs["train"]
    if args.pairs:
        # A fixed subset (the first N prompts), the same for every seed and loss.
        train = train[: args.pairs]
    held = pairs["test"][: args.eval_pairs]
    name = args.name or (f"{args.loss}-b{args.beta:g}" + (f"-n{args.pairs}" if args.pairs else "")
                         + (f"-e{args.epochs:g}" if args.epochs != 1 else "") + f"-s{args.seed}")
    run = Run(phase="b2", name=name, config={**vars(args), "train_pairs": len(train), "held_pairs": len(held)})

    tok = AutoTokenizer.from_pretrained(args.policy, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.policy, dtype=torch.float32, attn_implementation="sdpa")
    model = get_peft_model(base, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                                            target_modules="all-linear", task_type="CAUSAL_LM")).to(device)
    steps = int(np.ceil(args.epochs * len(train) / args.batch))
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(args.warmup, 1)))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def logprobs(rows, chunk: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        sums, counts = [], []
        for i in range(0, len(rows), chunk):
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                s, c = seq_logprobs(model, *batch_tensors(rows[i:i + chunk], tok.pad_token_id, device))
            sums.append(s)
            counts.append(c)
        return torch.cat(sums), torch.cat(counts)

    # The reference log-probabilities never change: compute them once, adapters off.
    t0 = time.time()
    model.eval()
    with torch.no_grad(), model.disable_adapter():
        ref = {}
        for split, rows in (("train", train), ("held", held)):
            w, nw = logprobs([(p, c) for p, c, _, _ in rows])
            l_, nl = logprobs([(p, r) for p, _, r, _ in rows])
            ref[split] = (w.cpu(), l_.cpu(), nw.cpu(), nl.cpu())
    print(f"reference log-probs for {len(train)} + {len(held)} pairs in {time.time() - t0:.0f}s", flush=True)

    def pair_stats(prefix: str, loss, h, pi_w, pi_l, ref_w, ref_l, nw, nl) -> dict[str, float]:
        return {f"{prefix}loss": float(loss.mean()), f"{prefix}acc": float((h > 0).float().mean()),
                f"{prefix}margin": float(h.mean()),
                f"{prefix}chosen_logratio": float((pi_w - ref_w).mean()),
                f"{prefix}rejected_logratio": float((pi_l - ref_l).mean()),
                f"{prefix}chosen_logp_tok": float((pi_w / nw).mean()),
                f"{prefix}rejected_logp_tok": float((pi_l / nl).mean()),
                f"{prefix}ref_chosen_logp_tok": float((ref_w / nw).mean()),
                f"{prefix}ref_rejected_logp_tok": float((ref_l / nl).mean())}

    rm_model = AutoModelForSequenceClassification.from_pretrained(args.rm, num_labels=1, dtype=half).to(device).eval()
    rm_model.config.pad_token_id = tok.pad_token_id
    rm = RewardModel(rm_model, tok, device)
    gold = Classifier(args.gold_model, device, half)
    eval_prompts = imdb_prompts("test", args.eval_prompts, tok, np.random.default_rng(12345))
    samples_path = run.path("samples.jsonl")

    @torch.no_grad()
    def evaluate(step: int) -> dict[str, float]:
        model.eval()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            seq, attn, mask, n_prompt, texts, tokens = sample(model, tok, eval_prompts, args.max_new_tokens,
                                                              device, generator_seed=999)
            pol = completion_logprobs(model, seq, attn, n_prompt)
            with model.disable_adapter():
                refl = completion_logprobs(model, seq, attn, n_prompt)
        full = [p + t for p, t in zip(eval_prompts, texts)]
        r, (_, p_gold) = rm(full), gold(full)
        m = mask.float()
        out = {"eval_score": float(r.mean()), "eval_p_gold": float(p_gold.mean()),
               "eval_kl": float(((pol - refl) * m).sum(1).mean()),
               "eval_ref_nll": float(-(refl * m).sum() / m.sum().clamp(min=1)),
               "eval_length": float(m.sum(1).mean()),
               "eval_chinese": float(np.mean([bool(CJK.search(t)) for t in texts])),
               **{f"eval_{k}": v for k, v in diversity(tokens).items()}}
        # The held-out pairs: is the preference learned, or only memorised?
        pw, nw = logprobs([(p, c) for p, c, _, _ in held])
        pl, nl = logprobs([(p, r_) for p, _, r_, _ in held])
        rw, rl, _, _ = (x.to(device) for x in ref["held"])
        loss, h = preference_loss(args.loss, args.beta, pw, pl, rw, rl)
        out.update(pair_stats("held_", loss, h, pw, pl, rw, rl, nw, nl))
        run.log(step=step, seconds=time.time() - t0, **out)
        with open(samples_path, "a") as f:
            for p, t, s, g in list(zip(eval_prompts, texts, r, p_gold))[:6]:
                f.write(json.dumps({"step": step, "prompt": p, "completion": t,
                                    "rm_score": float(s), "gold_p_pos": float(g)}) + "\n")
        model.train()
        return out

    evaluate(0)
    rng = np.random.default_rng(args.seed)
    order: list[int] = []
    model.train()
    for step in range(1, steps + 1):
        if len(order) < args.batch:
            order += rng.permutation(len(train)).tolist()
        idx, order = order[: args.batch], order[args.batch:]
        rows = [train[i] for i in idx]
        rw, rl, nw, nl = (x[idx].to(device) for x in ref["train"])
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            # Chosen and rejected in one forward pass.
            s, _ = seq_logprobs(model, *batch_tensors([(p, c) for p, c, _, _ in rows]
                                                      + [(p, r) for p, _, r, _ in rows], tok.pad_token_id, device))
        pw, pl = s[: len(rows)], s[len(rows):]
        loss, h = preference_loss(args.loss, args.beta, pw, pl, rw, rl)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss.mean()).backward()
        scaler.unscale_(opt)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        scaler.step(opt)
        scaler.update()
        sched.step()
        with torch.no_grad():
            run.log(step=step, epoch=step * args.batch / len(train), grad_norm=grad_norm,
                    seconds=time.time() - t0, **pair_stats("train_", loss, h, pw, pl, rw, rl, nw, nl))
        if step % args.eval_every == 0 or step == steps:
            e = evaluate(step)
            print(f"step {step}/{steps} loss {loss.mean():.3f} held acc {e['held_acc']:.2f} "
                  f"chosen {e['held_chosen_logratio']:+.1f} rejected {e['held_rejected_logratio']:+.1f} "
                  f"score {e['eval_score']:+.2f} gold {e['eval_p_gold']:.2f} kl {e['eval_kl']:.1f} "
                  f"d2 {e['eval_distinct2']:.2f} zh {e['eval_chinese']:.2f} {time.time() - t0:.0f}s", flush=True)
    run.finish({"final_eval": e, "steps": steps, "seconds": time.time() - t0})


if __name__ == "__main__":
    main()
