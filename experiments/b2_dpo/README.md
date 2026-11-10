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

## Results

Tables: `python experiments/b2_dpo/report.py --tables`. Figures in
`docs/assets/b2/`. "Before training" on B1's 64 evaluation prompts: RM
−0.10, gold 0.669, KL 0, distinct-2 0.88, reference NLL 3.32, 2 of 64
samples (3%) containing Chinese (every run's step 0).

### The preference set

| split | prompts | RM score, all samples | gold, all samples | RM, best of 4 | gold, best of 4 | RM, worst of 4 | gold, worst of 4 | median RM margin best-worst | gold agrees best ≥ worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 4096 | -0.21 | 0.592 | 1.20 | 0.734 | -1.54 | 0.435 | 2.37 | 0.786 |
| test | 512 | -0.13 | 0.603 | 1.28 | 0.735 | -1.50 | 0.453 | 2.30 | 0.783 |

Wall clock 5.3 min on one T4; 0 test prompts coincide with B1's 64 evaluation prompts.

Chinese in the labels: 6.5% of chosen and 2.6% of rejected training
responses contain Chinese (3.6% of all samples); the reward model scores
them +0.86 on average against −0.25 for the rest. B1's hack is in the
preference data, weakly.

### Item 1 — DPO (and IPO) on the full set, against B1's best PPO

4,096 best-vs-worst pairs (no ties; 4,094 under the random rule, which
drops two tied pairs), one epoch = 256 updates of 16 pairs, unless stated. Final held-out
evaluation, mean ± std over seeds:

| method | seeds | RM score | gold P(positive) | KL to reference | distinct-2 | reference NLL | samples in Chinese | held-out pair accuracy | chosen log-ratio (held out) | rejected log-ratio (held out) | minutes per run (T4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DPO β=0.05, lr 5e-5 | 1 | 5.05 | 0.999 | 40.32 | 0.87 | 4.11 | 0.25 | 0.83 | -84.96 | -129.11 | 5 |
| DPO β=0.1, lr 1e-5 | 1 | 2.50 | 0.905 | 6.76 | 0.86 | 3.27 | 0.11 | 0.80 | -6.47 | -12.60 | 6 |
| DPO β=0.1, lr 5e-5 | 3 | 4.07 ± 0.39 | 0.968 ± 0.012 | 20.70 ± 2.15 | 0.88 ± 0.02 | 3.76 ± 0.12 | 0.15 ± 0.03 | 0.83 ± 0.00 | -30.84 ± 3.37 | -51.45 ± 4.50 | 5 |
| DPO β=0.1, lr 5e-5, random pairs | 1 | 3.14 | 0.921 | 11.58 | 0.85 | 3.44 | 0.09 | 0.70 | -15.12 | -20.33 | 5 |
| DPO β=0.1, lr 5e-5, 3 ep | 1 | 4.71 | 0.998 | 38.49 | 0.87 | 3.35 | 0.19 | 0.85 | -86.36 | -136.79 | 12 |
| DPO β=0.2, lr 5e-5 | 3 | 2.96 ± 0.02 | 0.957 ± 0.007 | 9.59 ± 0.72 | 0.89 ± 0.00 | 3.65 ± 0.00 | 0.12 ± 0.01 | 0.84 ± 0.01 | -10.41 ± 0.50 | -20.51 ± 0.80 | 5 |
| DPO β=0.5, lr 5e-5 | 1 | 2.10 | 0.899 | 3.59 | 0.88 | 3.53 | 0.09 | 0.84 | -2.12 | -6.21 | 5 |
| IPO τ=0.01, lr 5e-5 | 3 | 4.30 ± 0.40 | 0.978 ± 0.007 | 24.86 ± 2.64 | 0.86 ± 0.01 | 3.77 ± 0.13 | 0.10 ± 0.04 | 0.82 ± 0.01 | -43.77 ± 5.09 | -69.20 ± 7.58 | 5 |
| IPO τ=0.03, lr 5e-5 | 1 | 2.73 | 0.952 | 8.08 | 0.88 | 3.63 | 0.14 | 0.86 | -8.35 | -16.96 | 5 |
| IPO τ=0.1, lr 5e-5 | 1 | 1.35 | 0.829 | 1.51 | 0.88 | 3.51 | 0.08 | 0.81 | -0.72 | -3.10 | 5 |
| B1 PPO β=0.2, clip 0.05, lr 1e-4 | 3 | 5.78 ± 0.11 | 0.983 ± 0.013 | 13.84 ± 0.75 | 0.77 ± 0.01 | 3.19 ± 0.01 | – | – | – | – | 54 |
| B1 PPO β=0.2, clip 0.2, lr 1e-5 | 3 | 4.68 ± 0.05 | 0.973 ± 0.007 | 11.11 ± 0.33 | 0.80 ± 0.01 | 3.17 ± 0.02 | – | – | – | – | 56 |

Run directories: `runs/b2/{dpo,ipo}-b{β}-lr{lr}[-e3][-random]-s{seed}__*`.
Figures: `docs/assets/b2/frontier.png`, `docs/assets/b2/trajectory.png`.
The minutes column is the whole run including 11 evaluations; the
preference set (5 min) and B1's reward model (4.5 min) are one-off costs
shared by every DPO run.

### Items 2-3 — 64 pairs, 200 epochs (800 updates): DPO against IPO

The first 64 training pairs, the same for every run and seed; lr 5e-5.

| method | seeds | train loss | train accuracy | train margin h | chosen log-ratio (train) | lowest chosen log-ratio (train) | rejected log-ratio (train) | held-out accuracy | KL to reference | peak KL | gold P(positive) | distinct-2 | samples in Chinese |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DPO β=0.1, lr 5e-5, 64 pairs, 200 ep | 3 | 1.22e-06 ± 1.02e-07 | 1.00 ± 0.00 | 153.3 ± 5.2 | -13.9 ± 3.9 | -31.17 ± 1.92 | -167.2 ± 4.1 | 0.57 ± 0.01 | 41.9 ± 1.5 | 41.98 ± 1.52 | 0.837 ± 0.029 | 0.85 ± 0.01 | 0.32 ± 0.04 |
| DPO β=0.5, lr 5e-5, 64 pairs, 200 ep | 1 | 3.47e-07 | 1.00 | 32.5 | 11.5 | 0.00 | -21.0 | 0.57 | 1.6 | 1.56 | 0.661 | 0.89 | 0.05 |
| DPO β=1, lr 5e-5, 64 pairs, 200 ep | 1 | 2.97e-09 | 1.00 | 24.5 | 9.1 | 0.00 | -15.4 | 0.59 | 0.8 | 0.89 | 0.760 | 0.89 | 0.03 |
| IPO τ=0.01, lr 5e-5, 64 pairs, 200 ep | 3 | 1.72 ± 1.17 | 1.00 ± 0.00 | 50.2 ± 0.3 | 20.6 ± 0.6 | -0.02 ± 0.02 | -29.6 ± 0.9 | 0.59 ± 0.01 | 2.2 ± 0.1 | 2.53 ± 0.21 | 0.672 ± 0.013 | 0.90 ± 0.00 | 0.03 ± 0.01 |
| IPO τ=0.1, lr 5e-5, 64 pairs, 200 ep | 3 | 1.55 ± 0.914 | 1.00 ± 0.00 | 5.1 ± 0.2 | 1.5 ± 0.3 | -0.09 ± 0.09 | -3.7 ± 0.3 | 0.50 ± 0.01 | 2.7 ± 0.4 | 2.75 ± 0.37 | 0.631 ± 0.009 | 0.91 ± 0.00 | 0.07 ± 0.02 |
| IPO τ=1, lr 5e-5, 64 pairs, 200 ep | 1 | 1.12 | 0.69 | 0.2 | -0.7 | -1.89 | -0.9 | 0.50 | 2.1 | 2.13 | 0.579 | 0.91 | 0.03 |

Run directories: `runs/b2/{dpo,ipo}-b{β}-lr5e-05-n64-e200-s{seed}__*`.
Figures: `docs/assets/b2/overfit.png`, `docs/assets/b2/coefficient.png`.

## Findings

- **DPO reaches PPO's gold score at PPO's KL, for a tenth of the compute,
  but buys less reward-model score per nat.** DPO β = 0.2 (3 seeds): gold
  0.957 ± 0.007 at 9.6 nats, distinct-2 0.89; B1's PPO β = 0.2: 0.973 ±
  0.007 at 11.1 nats (lr 1e-5) and 0.983 ± 0.013 at 13.8 (lr 1e-4),
  distinct-2 0.77-0.80. Along the training trajectories gold rises with KL at
  the same rate for both (`trajectory.png`); the reward model's score does
  not: 3.0 for DPO against 4.7 for PPO at ≈10 nats. PPO's extra reward
  model score is not credited by the gold judge. One DPO run is ≈5 minutes on
  a T4 against ≈55 for PPO, needs no value model and no reward model in the
  training loop, and has two hyperparameters that mattered (β, lr) against
  PPO's β, clip range, lr and the value-model setup.
- **β sets the KL budget as the derivation says**: β = 0.5, 0.2, 0.1, 0.05
  end at 3.6, 9.6, 20.7, 40.3 nats; gold climbs 0.90 → 0.999 with it.
- **DPO goes far past its own data.** The best of the four reference
  samples per prompt scores gold 0.735 (KL ≤ log 4 − 3/4 = 0.64 nats); DPO
  at β = 0.2 reaches 0.957. The pairs give a direction, and the loss keeps
  following it.
- **DPO imports the reward model's exploit from the labels, partially.**
  12-25% of DPO's samples contain Chinese (3% before training), rising as β
  falls; PPO at β = 0.2 had none, and PPO at β ≤ 0.05 switched completely.
  Offline, the hack arrives only at the strength it has in the data.
- **The chosen log-probability falls even on the full set, in one epoch.**
  Held-out chosen log-ratio: −10 at β = 0.2, −31 at β = 0.1, −85 at β =
  0.05; the rejected one falls faster (−21, −51, −129), so the preference
  is learned (held-out accuracy 0.83-0.86) while both responses become less
  likely. A third epoch at β = 0.1 takes the chosen side to −86 and KL to
  38 nats for +0.02 held-out accuracy.
- **Item 2, DPO on 64 pairs (β = 0.1, 3 seeds): chosen-logprob collapse and
  KL blowup.** The loss falls to 1e-6 and the training margin to 153 nats,
  bought almost entirely on the rejected side (−167). The chosen log-ratio
  first rises to +14-16 (update ≈30), then falls through zero by update
  255-315 and ends at −14 (−0.2 nats per token): the model ends *less*
  likely to produce the responses it was told to prefer than before
  training. On held-out chosen responses it falls monotonically to −62.
  Sampled KL rises to 42 nats and 32% of samples contain Chinese; held-out
  pair accuracy is 0.57. The aggregate loss shows none of it.
- **The runaway is a small-β effect, not a property of DPO at every β.**
  The derivation says the loss keeps falling as h → ∞ whatever β is; in
  practice the gradient (≈ e^(−βh)) vanishes once βh ≈ 15-25, so DPO stops
  at h ≈ 153, 32.5, 24.5 for β = 0.1, 0.5, 1 (`coefficient.png`). At β = 0.5
  and 1 (one seed each) the chosen side goes *up* (+11.5, +9.1), KL is 1.6
  and 0.8 nats, and nothing collapses. The bound exists, but it sits at a
  margin of roughly 20/β nats, and at β = 0.1 that is far enough to wreck
  the policy.
- **Item 3, IPO stops exactly where τ says.** Final margins 50.2 ± 0.3
  (τ = 0.01, target 50), 5.1 ± 0.2 (τ = 0.1, target 5), 0.2 (τ = 1, target
  0.5, still climbing). At τ = 0.01 IPO separates the pairs by 50 nats with
  the chosen side *up* 20.6 and KL 2.2, where DPO β = 0.1 needed −14 and 42
  nats for 153; no IPO run collapses, on any seed, and the Chinese share
  stays at the reference's 3-7%.
- **But on the full set IPO and DPO trace the same frontier.** IPO τ = 0.03
  (target 16.7) ends at 8.1 nats, gold 0.952; τ = 0.01 at 24.9 nats, gold
  0.978 ± 0.007 with a held-out chosen log-ratio of −44: in one epoch over
  4,096 pairs nothing reaches its target margin, so the bound never binds
  and IPO behaves like DPO with a different β. IPO's bound is a fix for
  repeated passes over few pairs, which is where DPO breaks.
- **Random pairs teach less than best-vs-worst**: at β = 0.1, 11.6 nats and
  gold 0.92 against 20.7 and 0.97 (one seed); the median reward-model gap of a
  random pair is 1.10 against 2.37 for best-vs-worst, so the pairs carry
  less signal and the same β moves the policy less.
