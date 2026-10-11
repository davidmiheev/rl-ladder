# B1 — Foundations: REINFORCE -> PPO

**Goal.** PPO on a language model end to end, and what the KL-to-reference term is actually buying you.

**Prerequisites.** A4 (policy gradients)
**Companion lab.** nlp_course week07_finetuning
**Compute.** Kaggle's free 2×T4 notebooks (see `docs/free-compute.md`); Qwen2.5-0.5B with LoRA

The checklist items, reading list and lab links for this phase live in
[`docs/ladder.md`](../../docs/ladder.md#b1-foundations-reinforce->-ppo) — that file is the single
source of truth for what "done" means. This README is the lab notebook: how the
code here is run, and what the runs actually showed.

## The task

Every run here continues the opening 4-8 tokens of an IMDB review (train
split for training, 64 fixed test-split openings for evaluation) with up to
48 sampled tokens, and is rewarded once, at the end, by a sentiment
classifier: "write a positive continuation". Policy: Qwen2.5-0.5B (base) with
LoRA (r = 16, all linear layers); the reference is the same model with the
adapters switched off.

Three judges:

| role | model | used by |
| --- | --- | --- |
| proxy reward | `lvwerra/distilbert-imdb`, log-odds of positive | REINFORCE (`reinforce_lm.py`) |
| proxy reward | Qwen2.5-0.5B + scalar head trained on IMDB labels (`train_rm.py`), its logit | PPO (`ppo_lm.py`): TRL's PPOTrainer needs a reward model on the policy's tokenizer |
| gold judge, never trained against | `siebert/sentiment-roberta-large-english`, P(positive) | every evaluation |

The PPO reward model reaches 85.8% accuracy on 2,000 held-out 64-token test
openings; the gold judge reaches 85.5% on the same texts, and the two agree
on 87.5% of them (`train_rm.py`, 5.5 min on a T4).

Measured on the held-out prompts every 25 updates: proxy reward, gold
P(positive), KL to the reference (sum over completion tokens of
log π − log π_ref, nats per sequence), distinct-2 over the 64 completions,
and the reference model's per-token NLL of the completions; six texts are
saved each time (`samples.jsonl` in the run directory).

## How to run

Runs go to Kaggle GPU notebooks built by `scripts/kaggle/gen_b.py` (code
embedded; one run per GPU at a time):

```bash
python scripts/kaggle/gen_b.py "experiments/b1_ppo/reinforce_lm.py --seed 0" "experiments/b1_ppo/reinforce_lm.py --seed 1" > nb.py
python scripts/kaggle/kpost.py b1-reinforce-a nb.py gpu
# PPO needs the reward model notebook's output mounted, and TRL 1.11 (PPOTrainer was removed after it):
python scripts/kaggle/gen_b.py --pip trl==1.11.0 "experiments/b1_ppo/ppo_lm.py --rm \$RM_DIR --kl-coef 0.05 --seed 0" > nb.py
python scripts/kaggle/kpush.py b1-ppo-a nb.py gpu source=davidmiheev/b1-reward-model
# collect, then tables and figures:
kaggle kernels output davidmiheev/<slug> -p out && tar xzf out/b_runs.tgz --skip-old-files
python experiments/b1_ppo/report.py
```

Every script has `--smoke` (tiny model, a few steps, CPU).

## Results

### Item 1 — REINFORCE on a learned reward, no KL term

A4's estimator (`pg.discounted_returns` with γ = 1 and one terminal reward,
batch-mean baseline), 200 updates of 32 samples, lr 1e-4, ≈14 min per run on
a T4. Final held-out evaluation, mean ± std over seeds:

| run | seeds | proxy reward (log-odds) | gold P(positive) | KL to reference (nats/seq) | distinct-2 | reference NLL (nats/token) |
| --- | --- | --- | --- | --- | --- | --- |
| before training (update 0) | 8 | 0.75 | 0.68 | 0 | 0.88 | 3.32 |
| no KL penalty, after 200 updates | 5 | 5.69 ± 0.03 | 0.96 ± 0.01 | 100.1 ± 8.3 | 0.05 ± 0.01 | 2.76 ± 0.26 |
| KL penalty β = 0.05 (r − β·KL), after 200 updates | 3 | 5.22 ± 0.14 | 0.97 ± 0.02 | 14.6 ± 1.8 | 0.79 ± 0.03 | 3.44 ± 0.07 |

Run directories: `runs/b1/reinforce-kl0-s{0..4}__*`, `runs/b1/reinforce-kl0.05-s{0..2}__*`.
Figure: `docs/assets/b1/reinforce.png`.

What the policy writes (seed 0, the same held-out prompts):

| update | no KL penalty | β = 0.05 |
| --- | --- | --- |
| 0 | "This movie is stunningly so sad, as the lightweight fabric gets rendered toxic to all" | (same model) |
| 100 | "This movie is stunningly and beautifully crafted, beautifully crafted and beautifully and gracefully and beautifully, a, a superb, a master, beautifully and bea…" | |
| 200 | "I really like this movie. I've beautifully presented and beautifully crafted, a beautifully presented and beautifully presented, a beautifully presented and a s…" | "I really like this movie. I only say it because if you like the movie then this is a movie you will enjoy. It's simply brilliant and you'll love it, this is one of the ye…" |

### Items 2-3 — PPO (TRL 1.11 PPOTrainer), KL coefficient and clip range

Qwen2.5-0.5B + LoRA, value model initialised from the reward model (last two
layers and head trainable), reward = `train_rm.py`'s logit; 150 updates of 32
rollouts, 4 PPO epochs; ≈55 min per run on a T4. Final held-out evaluation
(`python experiments/b1_ppo/report.py --tables`):

| lr | KL coefficient β | clip range | seeds | RM score | gold P(positive) | KL to reference | distinct-2 | reference NLL | final samples in Chinese | clipped fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-5 | 0 | 0.2 | 3 | 14.20 ± 0.22 | 0.90 ± 0.01 | 90.76 ± 2.35 | 0.26 ± 0.05 | 2.95 ± 0.18 | 1.00 | 0.0019 |
| 1e-5 | 0.01 | 0.2 | 3 | 13.97 ± 0.06 | 0.89 ± 0.04 | 78.55 ± 3.12 | 0.32 ± 0.03 | 2.89 ± 0.06 | 1.00 | 0.0017 |
| 1e-5 | 0.05 | 0.2 | 3 | 12.57 ± 0.16 | 0.85 ± 0.01 | 53.56 ± 0.86 | 0.54 ± 0.02 | 2.88 ± 0.07 | 1.00 | 0.0007 |
| 1e-5 | 0.2 | 0.05 | 1 | 4.50 | 0.97 | 11.03 | 0.79 | 3.17 | 0.00 | 0.0051 |
| 1e-5 | 0.2 | 0.2 | 3 | 4.68 ± 0.05 | 0.97 ± 0.01 | 11.11 ± 0.33 | 0.80 ± 0.01 | 3.17 ± 0.02 | 0.00 | 0.0003 |
| 1e-5 | 0.2 | 0.5 | 1 | 4.79 | 0.95 | 10.91 | 0.80 | 3.21 | 0.00 | 0.0000 |
| 1e-5 | 0.5 | 0.2 | 3 | 1.40 ± 0.06 | 0.82 ± 0.02 | 1.24 ± 0.02 | 0.88 ± 0.01 | 3.36 ± 0.06 | 0.00 | 0.0003 |
| 1e-4 | 0.2 | 0.05 | 3 | 5.78 ± 0.11 | 0.98 ± 0.01 | 13.84 ± 0.75 | 0.77 ± 0.01 | 3.19 ± 0.01 | 0.00 | 0.1274 |
| 1e-4 | 0.2 | 0.2 | 3 | 6.96 ± 2.08 | 0.96 ± 0.03 | 18.24 ± 7.21 | 0.79 ± 0.01 | 3.21 ± 0.10 | 0.33 | 0.0241 |
| 1e-4 | 0.2 | 0.5 | 3 | 5.65 ± 0.00 | 0.96 ± 0.01 | 13.89 ± 0.34 | 0.78 ± 0.01 | 3.17 ± 0.01 | 0.06 | 0.0050 |

Run directories: `runs/b1/ppo-kl{β}-clip{ε}-s{seed}__*` (lr 1e-5) and
`runs/b1/ppo-lr1e-4-kl0.2-clip{ε}-s{seed}__*`. Figures: `docs/assets/b1/ppo_kl.png`,
`docs/assets/b1/ppo_clip.png`. Writeup: [`docs/writeups/b1-ppo.md`](../../docs/writeups/b1-ppo.md).

## Findings

- **Without a KL term REINFORCE finds the degenerate optimum in under 100
  updates, on every seed.** The proxy reaches its ceiling (log-odds ≈ 5.7)
  by update 75; after that the reward cannot rise, and the policy keeps
  drifting anyway: KL to the reference goes from 20 to ≈100 nats per
  48-token completion and distinct-2 from 0.8 to 0.05, as every
  continuation collapses into the same chant of "beautifully crafted,
  beautifully presented". Five seeds land in the same place (KL 94-109,
  distinct-2 0.04-0.09).
- **The gold judge does not catch it.** A larger sentiment model the policy
  never trained against scores the chant P(positive) ≈ 0.96. A held-out judge
  of the *same property* cannot see a hack that games the property itself;
  what exposes it here is diversity and the distance from the pretrained
  model, neither of which the reward looks at.
- **Reference NLL is a misleading fluency measure for this failure.**
  Repetition is cheap for a language model to predict, so the pretrained
  model's NLL of the degenerate text *falls* (3.32 → 2.76 nats/token) while
  the text becomes unreadable. Distinct-n and KL are the signals to watch.
- **A KL penalty of β = 0.05 keeps nearly all of the gain and none of the
  collapse**: gold P(positive) 0.97 (the same), proxy 5.22 instead of 5.69,
  7× less KL (15 nats) and distinct-2 0.79 against 0.88 before training. The
  half-point of proxy reward that the penalty gives up is exactly the part
  that was bought by degenerating.
- **PPO's near-zero-KL failure is a language switch (item 3).** With β ≤
  0.05 every seed starts writing Chinese praise after an English opening
  clause, abruptly (none to all saved samples within 25 updates, by update
  75 for β ≤ 0.01, 100 for β = 0.05). The reward model's Qwen backbone reads
  Chinese and its head was fitted only on English reviews; it pays +14 to
  +16 for Chinese praise against +3 to +5 for strong English praise. Here
  the gold judge does diverge from the proxy: it peaks at ≈0.99 around 40
  nats of KL and falls to 0.85-0.90 as the switch sets in.
- **β = 0.2 is the operating point (item 2):** gold 0.97 ± 0.01, 11 nats,
  distinct-2 0.80, English, on all three seeds; β = 0.5 barely moves
  (KL 1.2, gold 0.82). The same β = 0.05 that held REINFORCE to 15 nats lets
  PPO reach 54, because the DistilBERT proxy saturates and the reward model's
  logit does not: β only means something relative to the reward's range.
- **The clip range matters only when updates are large.** At lr 1e-5 fewer
  than 0.5% of tokens are clipped and ε = 0.05/0.2/0.5 end in the same place.
  At lr 1e-4, ε = 0.05 clips 13% of tokens and gives the tightest runs (peak
  KL 16.7-17.1 on all seeds, gold 0.98, no switch); at ε = 0.2 one seed of
  three took a large step (approximate KL 0.38 in one update) and escaped
  into Chinese despite β = 0.2; ε = 0.5 did not. Three seeds do not make that
  a trend in ε.
- **Deviations:** TRL removed PPOTrainer after 1.11, so the PPO runs pin
  `trl==1.11.0`; the value model trains only its last two layers to fit a
  16 GB T4.

