"""B1, items 2-3: TRL's PPOTrainer on the same task, sweeping clip range and KL coefficient.

Same task as `reinforce_lm.py`: continue the opening words of an IMDB review.
The reward is `train_rm.py`'s Qwen-0.5B sentiment head (TRL's PPOTrainer needs
a reward model on the policy's tokenizer); the value model starts from the
same head, as in TRL's own examples. The policy is Qwen2.5-0.5B with LoRA;
the reference is the same model with the adapters off.

    python experiments/b1_ppo/ppo_lm.py --rm /kaggle/input/b1-reward-model/rm --kl-coef 0.05 --cliprange 0.2 --seed 0

Needs TRL 1.11 (`pip install trl==1.11.0`): TRL removed PPOTrainer on
2026-09-04 (huggingface/trl#7020), and the version this repo pins no longer
has it.

Logged: every PPO update's TRL metrics (score, KL to the reference, the
RLHF reward r − β·KL, entropy, clip fractions, value loss), and every
`--eval-every` updates the held-out evaluation `reinforce_lm.py` uses —
the same 64 test prompts sampled with a fixed seed, scored by the reward
model, by the gold judge it never trains against, and for KL to the
reference, reference NLL and diversity — with a few texts saved.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src"), str(HERE), str(REPO / "experiments" / "a4_policy_gradient")]

from reinforce_lm import Classifier, completion_logprobs, diversity, imdb_prompts, sample  # noqa: E402
from rl_ladder import Run, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--rm", required=True, help="directory written by train_rm.py")
    p.add_argument("--gold-model", default="siebert/sentiment-roberta-large-english")
    p.add_argument("--kl-coef", type=float, default=0.05)
    p.add_argument("--cliprange", type=float, default=0.2)
    p.add_argument("--updates", type=int, default=150)
    p.add_argument("--batch", type=int, default=32, help="rollouts per PPO update")
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--value-layers", type=int, default=2, help="trainable top layers of the value model")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-prompts", type=int, default=64)
    p.add_argument("--name", default=None)
    p.add_argument("--smoke", action="store_true", help="tiny model, 2 updates, CPU")
    a = p.parse_args()
    if a.smoke:
        a.policy, a.updates, a.batch, a.max_new_tokens, a.ppo_epochs = "HuggingFaceTB/SmolLM2-135M", 2, 4, 8, 1
        a.eval_every, a.eval_prompts, a.gold_model = 1, 4, "lvwerra/distilbert-imdb"
    return a


class RewardModel:
    """The PPO reward model as a scorer of text (its logit is the log-odds of positive).

    Shares the model object TRL trains against, so only one copy is in GPU memory.
    """

    def __init__(self, model, tok, device: str):
        from transformers import AutoTokenizer

        self.model, self.device = model, device
        self.tok = AutoTokenizer.from_pretrained(tok.name_or_path, padding_side="right")
        self.tok.pad_token = tok.pad_token

    @torch.no_grad()
    def __call__(self, texts: list[str]) -> np.ndarray:
        enc = self.tok(texts, padding=True, return_tensors="pt").to(self.device)
        return self.model(**enc).logits.squeeze(-1).float().cpu().numpy()


def main() -> None:
    args = parse_args()
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, TrainerCallback
    from trl.experimental.ppo import PPOConfig, PPOTrainer

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = args.name or f"ppo-kl{args.kl_coef:g}-clip{args.cliprange:g}-s{args.seed}"
    run = Run(phase="b1", name=name, config=vars(args))

    tok = AutoTokenizer.from_pretrained(args.policy, padding_side="left")
    # PPOTrainer finds the end of a response by its first pad token, so pad must not be EOS.
    tok.pad_token = "<|fim_pad|>" if "<|fim_pad|>" in tok.get_vocab() else tok.unk_token or tok.eos_token
    policy = AutoModelForCausalLM.from_pretrained(args.policy, dtype=torch.float32, attn_implementation="sdpa")
    value = AutoModelForSequenceClassification.from_pretrained(args.rm, num_labels=1, dtype=torch.float32)
    reward = AutoModelForSequenceClassification.from_pretrained(args.rm, num_labels=1,
                                                                dtype=torch.float16 if device == "cuda" else torch.float32)
    for m in (value, reward):
        m.config.pad_token_id = tok.pad_token_id
    # Memory on a 16 GB T4: a fully trainable fp32 0.5B value model with Adam states is ~8 GB on its
    # own, so the value model trains only its last `--value-layers` transformer layers and its head.
    for n, q in value.named_parameters():
        q.requires_grad = n.startswith("score") or any(f"layers.{i}." in n for i in
                                                      range(value.config.num_hidden_layers - args.value_layers,
                                                            value.config.num_hidden_layers))

    rng = np.random.default_rng(args.seed)
    prompts = imdb_prompts("train", args.updates * args.batch, tok, rng)
    train_ds = Dataset.from_dict({"input_ids": [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]})
    eval_prompts = imdb_prompts("test", args.eval_prompts, tok, np.random.default_rng(12345))

    cfg = PPOConfig(
        output_dir=str(run.path("trainer")), seed=args.seed, report_to="none", save_strategy="no",
        logging_steps=1, num_sample_generations=0, learning_rate=args.lr,
        per_device_train_batch_size=args.batch // 4, gradient_accumulation_steps=4, num_mini_batches=1,
        num_ppo_epochs=args.ppo_epochs, total_episodes=args.updates * args.batch,
        response_length=args.max_new_tokens, stop_token="eos", temperature=1.0,
        kl_coef=args.kl_coef, cliprange=args.cliprange,
        local_rollout_forward_batch_size=args.batch // 4, fp16=device == "cuda", bf16=False,  # T4: no bf16
        use_cpu=device == "cpu",
    )
    trainer = PPOTrainer(
        args=cfg, processing_class=tok, model=policy, ref_model=None, reward_model=reward, value_model=value,
        train_dataset=train_ds,
        peft_config=LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                               target_modules="all-linear", task_type="CAUSAL_LM"),
    )

    rm_score = RewardModel(reward, tok, device)
    gold = Classifier(args.gold_model, device, torch.float16 if device == "cuda" else torch.float32)
    samples_path = run.path("samples.jsonl")
    t0 = time.time()
    use_amp = device == "cuda"

    def evaluate(step: int) -> dict[str, float]:
        model = trainer.accelerator.unwrap_model(trainer.model).policy
        was_training = model.training
        model.eval()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            seq, attn, mask, n_prompt, texts, tokens = sample(model, tok, eval_prompts, args.max_new_tokens,
                                                              device, generator_seed=999)
            with torch.no_grad():
                pol = completion_logprobs(model, seq, attn, n_prompt)
                with model.disable_adapter():
                    ref = completion_logprobs(model, seq, attn, n_prompt)
        model.train(was_training)
        full = [p + t for p, t in zip(eval_prompts, texts)]
        r, (_, p_gold) = rm_score(full), gold(full)
        m = mask.float()
        out = {"eval_score": float(r.mean()), "eval_p_gold": float(p_gold.mean()),
               "eval_kl": float(((pol - ref) * m).sum(1).mean()),
               "eval_ref_nll": float(-(ref * m).sum() / m.sum().clamp(min=1)),
               "eval_length": float(m.sum(1).mean()), **{f"eval_{k}": v for k, v in diversity(tokens).items()}}
        run.log(step=step, **out)
        with open(samples_path, "a") as f:
            for p, t, s, g in list(zip(eval_prompts, texts, r, p_gold))[:6]:
                f.write(json.dumps({"step": step, "prompt": p, "completion": t,
                                    "rm_score": float(s), "gold_p_pos": float(g)}) + "\n")
        return out

    class Logger(TrainerCallback):
        def on_log(self, _args, state, control, logs=None, **kw):
            if logs and "objective/scores" in logs:
                run.log(step=state.global_step, seconds=time.time() - t0,
                        **{k.replace("/", "_"): float(v) for k, v in logs.items() if isinstance(v, (int, float))})
                if state.global_step % 10 == 0:
                    print(f"update {state.global_step} score {logs['objective/scores']:+.2f} "
                          f"kl {logs['objective/kl']:.2f} {time.time() - t0:.0f}s", flush=True)

        def on_step_end(self, _args, state, control, **kw):
            if state.global_step % args.eval_every == 0:
                evaluate(state.global_step)

    trainer.add_callback(Logger())
    evaluate(0)
    trainer.train()
    last = trainer.state.global_step
    run.finish({"final_eval": evaluate(last) if last % args.eval_every else None})


if __name__ == "__main__":
    main()
