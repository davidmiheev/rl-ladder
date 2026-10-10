"""B1: a sentiment reward model on the policy's own tokenizer, for TRL's PPOTrainer.

TRL's PPOTrainer scores completions with a sequence-classification model that
reads the policy's token ids, so the DistilBERT proxy used by
`reinforce_lm.py` cannot be plugged in. This trains the standard RLHF
replacement: Qwen2.5-0.5B with a scalar head, fitted to IMDB's positive /
negative labels on the opening 64 tokens of each review (the length PPO's
texts have), with LoRA, then merged and saved as a plain model.

    python experiments/b1_ppo/train_rm.py --out /kaggle/working/rm        # on a GPU
    python experiments/b1_ppo/train_rm.py --smoke --out /tmp/rm            # CPU check

Reported: held-out accuracy on IMDB test openings, and agreement with the
gold judge (`siebert/sentiment-roberta-large-english`) on the same texts.
The reward is the head's logit, i.e. log-odds of positive.
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
sys.path[:0] = [str(REPO / "src")]

from rl_ladder import Run, seed_everything  # noqa: E402


def openings(split: str, n: int, tok, max_tokens: int, seed: int) -> tuple[list[str], np.ndarray]:
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/imdb", split=split).shuffle(seed=seed).select(range(n))
    texts = []
    for t in ds["text"]:
        ids = tok(re.sub(r"<br\s*/?>", " ", t).strip(), add_special_tokens=False)["input_ids"][:max_tokens]
        texts.append(tok.decode(ids))
    return texts, np.asarray(ds["label"], dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--out", required=True)
    p.add_argument("--train", type=int, default=20000)
    p.add_argument("--test", type=int, default=2000)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gold-model", default="siebert/sentiment-roberta-large-english")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.base, a.train, a.test, a.batch, a.gold_model = "HuggingFaceTB/SmolLM2-135M", 64, 32, 8, "lvwerra/distilbert-imdb"

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    seed_everything(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = Run(phase="b1", name=f"rm-s{a.seed}", config=vars(a))
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(a.base, num_labels=1, dtype=torch.float32)
    base.config.pad_token_id = tok.pad_token_id
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear",
                                            modules_to_save=["score"], task_type="SEQ_CLS")).to(device)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    x_tr, y_tr = openings("train", a.train, tok, a.max_tokens, a.seed)
    x_te, y_te = openings("test", a.test, tok, a.max_tokens, 12345)

    def logits(texts: list[str]) -> torch.Tensor:
        enc = tok(texts, padding=True, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            return model(**enc).logits.squeeze(-1).float()

    t0, step = time.time(), 0
    model.train()
    for _ in range(a.epochs):
        order = np.random.default_rng(a.seed).permutation(len(x_tr))
        for i in range(0, len(order), a.batch):
            idx = order[i:i + a.batch]
            out = logits([x_tr[j] for j in idx])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, torch.as_tensor(y_tr[idx], device=device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % 20 == 0:
                run.log(step=step, loss=loss.item(), seconds=time.time() - t0)

    model.eval()
    with torch.no_grad():
        scores = np.concatenate([logits(x_te[i:i + 64]).cpu().numpy() for i in range(0, len(x_te), 64)])
    acc = float(((scores > 0) == (y_te > 0.5)).mean())

    from transformers import pipeline

    gold = pipeline("text-classification", model=a.gold_model, device=0 if use_amp else -1, truncation=True)
    g = np.asarray([r["label"].lower() == "positive" for r in gold(x_te, batch_size=64)])
    agree = float(((scores > 0) == g).mean())
    gold_acc = float((g == (y_te > 0.5)).mean())

    merged = model.merge_and_unload()
    merged.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    summary = {"test_accuracy": acc, "agreement_with_gold": agree, "gold_test_accuracy": gold_acc,
               "score_mean": float(scores.mean()), "score_std": float(scores.std()), "seconds": time.time() - t0}
    (Path(a.out) / "rm_eval.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)
    run.finish(summary)


if __name__ == "__main__":
    main()
