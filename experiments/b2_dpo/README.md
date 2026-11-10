# B2 — Direct Alignment: DPO / IPO

**Goal.** Why a closed-form preference objective can replace an RL loop, and where that shortcut breaks down.

**Prerequisites.** B1
**Companion lab.** TRL DPOTrainer docs
**Compute.** Kaggle's free 2×T4 notebooks; Qwen2.5-0.5B with LoRA, as B1

The checklist items, reading list and lab links for this phase live in
[`docs/ladder.md`](../../docs/ladder.md#b2) — that file is the single
source of truth for what "done" means. This README is the lab notebook: how the
code here is run, and what the runs actually showed. Writeup:
[`docs/writeups/b2-dpo.md`](../../docs/writeups/b2-dpo.md).

## The task

B1's task, policy and judges, unchanged: continue the opening 4-8 tokens of
an IMDB review with up to 48 sampled tokens, positively; Qwen2.5-0.5B (base)
+ LoRA (r = 16, all linear layers), the reference is the same model with the
adapters off; evaluated on B1's 64 held-out prompts with B1's fixed sampling
seed by B1's reward model (`experiments/b1_ppo/train_rm.py`, the reward PPO
trained against), the gold judge (`siebert/sentiment-roberta-large-english`,
never trained against), KL to the reference on the policy's samples,
distinct-2, reference NLL and the share of samples containing Chinese.

**Preference set** (`make_prefs.py`): four reference samples (temperature 1)
for each of 4,096 train-split and 512 test-split openings, every one scored
by B1's reward model and by the gold judge. A pair is the best- and the
worst-scored of the four (`--pair-rule random`: two at random, the better
one chosen). Same reward model as PPO, so DPO and PPO differ in the
optimiser, not in the preference signal.

**Training** (`dpo_lm.py`): DPO or IPO written out, sequence log-probabilities
summed over completion tokens, 16 pairs per update, AdamW, 10 warm-up
updates, fp16 autocast. Logged every update (training batch) and every 25
updates (256 held-out pairs plus B1's generation evaluation): loss,
preference accuracy, margin $h$, chosen and rejected log-ratios to the
reference, chosen and rejected log-probabilities per token.

## How to run

```bash
# the preference set (B1's reward-model notebook mounted; ≈5 min on a T4):
python scripts/kaggle/gen_b.py "experiments/b2_dpo/make_prefs.py --rm \$RM_DIR --out /kaggle/working/prefs.jsonl.gz" > nb.py
python scripts/kaggle/kpush.py b2-prefs nb.py gpu source=davidmiheev/b1-reward-model
# DPO / IPO runs (both notebooks mounted; $PREFS and $RM_DIR are set by the notebook):
python scripts/kaggle/gen_b.py "experiments/b2_dpo/dpo_lm.py --prefs \$PREFS --rm \$RM_DIR --beta 0.1 --lr 5e-5 --seed 0" \
    "experiments/b2_dpo/dpo_lm.py --prefs \$PREFS --rm \$RM_DIR --loss ipo --beta 0.01 --lr 5e-5 --pairs 64 --epochs 200 --seed 0" > nb.py
python scripts/kaggle/kpush.py b2-dpo-a nb.py gpu source=davidmiheev/b1-reward-model source=davidmiheev/b2-prefs
# collect, then tables and figures (B1's runs under runs/b1 give the PPO rows):
kaggle kernels output davidmiheev/<slug> -p out && tar xzf out/b_runs.tgz --skip-old-files
python experiments/b2_dpo/report.py
```

Every script has `--smoke` (tiny model, CPU). `tests/test_b2_dpo.py`
checks the sequence log-probabilities under right padding against an
unpadded computation, and both losses at known points.

Notebooks: `b2-prefs` (the preference set), `b2-dpo-a` (full set),
`b2-overfit` (64 pairs), `b2-followup` (seeds and the β/τ checks below).

<!-- RESULTS -->
