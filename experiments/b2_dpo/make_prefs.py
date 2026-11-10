"""B2, item 1: a synthetic preference set for B1's task, labelled by B1's reward model.

DPO needs pairs (prompt, chosen, rejected). This builds them the way an RLHF
pipeline would if it had a reward model and no human labellers: sample K
continuations of each prompt from the *reference* policy (Qwen2.5-0.5B, the
model PPO and DPO both start from), score every one with B1's reward model
(`experiments/b1_ppo/train_rm.py`, the reward PPO trained against), and
keep all K with their scores. `dpo_lm.py` turns them into pairs.

Labelling with PPO's own reward model makes the comparison with PPO about
the optimiser and not about the preference signal: both see the same r(x, y),
PPO through online samples, DPO through K fixed samples per prompt. The gold
judge (`siebert/sentiment-roberta-large-english`) scores every sample too, so
the best-of-K reference set is a third point on the same scale.

    python experiments/b2_dpo/make_prefs.py --rm $RM_DIR --out runs/b2/prefs.jsonl.gz
    python experiments/b2_dpo/make_prefs.py --smoke --rm /tmp/rm --out /tmp/prefs.jsonl.gz

One line per prompt: {"split", "prompt", "prompt_ids", "completions": [ids, ...], "texts", "rm", "gold"},
with the samples' token ids cut at the first EOS (inclusive). Train-split prompts are fresh IMDB
openings; test-split prompts give held-out pairs. They are drawn from the same test split as B1's
64 evaluation prompts with a different seed, so a few can coincide by chance; the count is reported.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "experiments" / "b1_ppo"),
                str(REPO / "experiments" / "a4_policy_gradient")]

from ppo_lm import RewardModel  # noqa: E402
from reinforce_lm import Classifier, imdb_prompts, sample  # noqa: E402
from rl_ladder import Run, seed_everything  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--rm", required=True, help="directory written by experiments/b1_ppo/train_rm.py")
    p.add_argument("--gold-model", default="siebert/sentiment-roberta-large-english")
    p.add_argument("--train-prompts", type=int, default=4096)
    p.add_argument("--test-prompts", type=int, default=512)
    p.add_argument("--k", type=int, default=4, help="samples per prompt")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--batch", type=int, default=64, help="prompts per generation batch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.policy, a.gold_model = "HuggingFaceTB/SmolLM2-135M", "lvwerra/distilbert-imdb"
        a.train_prompts, a.test_prompts, a.k, a.max_new_tokens, a.batch = 8, 4, 2, 8, 4

    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    seed_everything(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = torch.float16 if device == "cuda" else torch.float32
    run = Run(phase="b2", name=f"prefs-k{a.k}-s{a.seed}", config=vars(a))
    tok = AutoTokenizer.from_pretrained(a.policy, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    policy = AutoModelForCausalLM.from_pretrained(a.policy, dtype=half, attn_implementation="sdpa").to(device).eval()
    rm_model = AutoModelForSequenceClassification.from_pretrained(a.rm, num_labels=1, dtype=half).to(device).eval()
    rm_model.config.pad_token_id = tok.pad_token_id
    rm, gold = RewardModel(rm_model, tok, device), Classifier(a.gold_model, device, half)

    rng = np.random.default_rng(a.seed)
    splits = {"train": imdb_prompts("train", a.train_prompts, tok, rng),
              "test": imdb_prompts("test", a.test_prompts, tok, np.random.default_rng(a.seed + 777))}
    b1_eval = set(imdb_prompts("test", 64, tok, np.random.default_rng(12345)))
    overlap = sum(x in b1_eval for x in splits["test"])

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, n_done, stats = time.time(), 0, {}
    with gzip.open(out, "wt") as f:
        for split, prompts in splits.items():
            rm_all, gold_all = [], []
            for i in range(0, len(prompts), a.batch):
                chunk = prompts[i:i + a.batch]
                rep = [x for x in chunk for _ in range(a.k)]
                seq, attn, mask, n_prompt, texts, tokens = sample(policy, tok, rep, a.max_new_tokens, device)
                full = [x + t for x, t in zip(rep, texts)]
                r = rm(full)
                _, g = gold(full)
                # `sample`'s mask already includes the EOS a sample ended on (its probability is part of
                # the sequence's) and excludes the padding after it.
                ids = [row[: int(m.sum())].tolist()
                       for row, m in zip(seq[:, n_prompt:].cpu(), mask.cpu())]
                for j, x in enumerate(chunk):
                    s = slice(j * a.k, (j + 1) * a.k)
                    f.write(json.dumps({"split": split, "prompt": x,
                                        "prompt_ids": tok(x, add_special_tokens=False)["input_ids"],
                                        "completions": ids[s], "texts": texts[s],
                                        "rm": [round(float(v), 4) for v in r[s]],
                                        "gold": [round(float(v), 4) for v in g[s]]}) + "\n")
                rm_all.append(r.reshape(-1, a.k))
                gold_all.append(g.reshape(-1, a.k))
                n_done += len(chunk)
                if (i // a.batch) % 10 == 0:
                    print(f"{split} {i + len(chunk)}/{len(prompts)} {time.time() - t0:.0f}s", flush=True)
            r, g = np.concatenate(rm_all), np.concatenate(gold_all)
            best, worst = r.argmax(1), r.argmin(1)
            rows = np.arange(len(r))
            stats[split] = {
                "prompts": len(r), "rm_mean": float(r.mean()), "gold_mean": float(g.mean()),
                "best_of_k_rm": float(r[rows, best].mean()), "best_of_k_gold": float(g[rows, best].mean()),
                "worst_of_k_rm": float(r[rows, worst].mean()), "worst_of_k_gold": float(g[rows, worst].mean()),
                "margin_median": float(np.median(r[rows, best] - r[rows, worst])),
                # Does the gold judge agree with the reward model on which of best/worst is better?
                "gold_agrees_best_vs_worst": float((g[rows, best] >= g[rows, worst]).mean()),
            }
            run.log(step=0, **{f"{split}_{k}": v for k, v in stats[split].items()})
    summary = {**{f"{s}_{k}": v for s, d in stats.items() for k, v in d.items()},
               "test_prompts_in_b1_eval_set": overlap, "seconds": time.time() - t0, "file": str(out)}
    print(json.dumps(summary, indent=1), flush=True)
    run.finish(summary)


if __name__ == "__main__":
    main()
