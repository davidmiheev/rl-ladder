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
