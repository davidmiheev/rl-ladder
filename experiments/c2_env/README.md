# C2 — Framing Market Making as an RL Problem

**Goal.** RL adjusts the coefficients of a known-good analytic model rather than
replacing the model outright — keeps the policy auditable.

**Prerequisites.** C1 (its calibration, its split, and most of its design conclusions)
**Companion lab.** Gymnasium custom-environment tutorial
**Compute.** CPU. About 90 s per episode, dominated by replaying 60 million
market events — roughly 40 episodes an hour on one core.

Checklist items live in [`docs/ladder.md`](../../docs/ladder.md#c2); the
reasoning and the full argument are in
[`docs/writeups/c2-env.md`](../../docs/writeups/c2-env.md).

## What is here

| file | what it is |
| --- | --- |
| [`env.py`](env.py) | `MarketMakingEnv`, a Gymnasium env: one session per episode, one coefficient decision per second, one quote refresh per 100 ms |
| [`run_checks.py`](run_checks.py) | the four checks against answers known in advance |
| [`run_random.py`](run_random.py) | does the action space do anything, and what scale is the reward on |

## How to run

```bash
python experiments/c2_env/run_checks.py                 # ~4 min (replays a session twice)
python experiments/c2_env/run_checks.py --skip-baseline # ~1 min
python experiments/c2_env/run_random.py --date 2025-08-26
```

```python
from env import MarketMakingEnv
env = MarketMakingEnv(date="2025-08-26")          # held-out days need allow_held_out=True
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.baseline_action())
```

## The formulation, in one screen

**State** (10 features, fixed scalings, no running statistics): inventory,
volatility in bp/√s, touch spread, touch imbalance, depth-5 imbalance, signed
aggressive flow over 1 s and 10 s, log k, the model's own half spread in bp,
time of day.

**Action** (4 continuous, log-mapped from [−1, 1]): γ ∈ [0.005, 1], c1 scale
∈ [0.1, 4], half-spread scale ∈ [0.125, 8], skew scale ∈ [0.01, 2]. Quotes are

```
half spread = (c1_scale * c1(γ,k) + ½ σ c2(γ,A,k)) * half_scale
skew        = σ c2(γ,A,k) * skew_scale
```

**Reward**: mark-to-market PnL in bp of a 100k book, minus `η q²` with
η = 0.015 and q normalised by the 20-unit position cap.

**Episode**: one session, preceded by a 10-minute calibration warm-up under
baseline coefficients that earns no reward.

## Results

### The four checks

```
2. no lookahead                901 observations compared, max abs diff 0
3. the reward telescopes       summed -4.949953400 bp, direct -4.949953400 bp
4. gymnasium conformance       passed
1. baseline is a special case  balance diff 1.5e-11, position/fee/trades identical
                               (2,694 trades, both)
```

### Does the action space do anything

ETHUSDC 2025-08-26, one session per row:

| policy | PnL (% of book) | fills | median half spread | max &#124;q&#124; |
| --- | --- | --- | --- | --- |
| widest (half-spread scale at max) | **0.000** | **0** | 605 ticks | 0.00 |
| baseline (C1's `glft-grid`) | −0.93 | 2,694 | 76 ticks | 0.60 |
| uniform random, seed 0/1/2 | −7.26 / −9.02 / −6.57 | ~41,000 | 116 ticks | 1.8–2.0 |
| tightest (half-spread scale at min) | −33.84 | 221,046 | 10 ticks | 1.90 |

Range: 33.8% of book.

## Findings

**Doing nothing is reachable, and on this day it wins.** The widest corner of
the action space takes zero fills and returns exactly zero, beating the
baseline's −0.93%. C1's degeneracy is now a property of the environment rather
than a surprise waiting for C3 — and it is the single most likely way C3 goes
wrong. Log fill counts alongside returns from the first run.

**Random is 7× worse than the baseline, not slightly worse.** Most of the
action space is bad. That matters for exploration: this is not a
neighbourhood of decent policies with a few bad corners.

**The micro-price is not a separate feature from imbalance.** At the touch,
`micro − mid = (spread/2) × imbalance` exactly, so the ladder's "order-book
imbalance / micro-price features" are one feature and a scaling. The second
slot carries depth-5 imbalance instead. I found this by chasing a feature that
was saturating the observation clip; the saturation turned out to be correct
(wide-spread moments) and the identity fell out of checking it properly.

**Ten features, about six directions.** `imbalance` and `imbalance_depth`
correlate at 0.974 on this asset; `volatility`, `log_k` and
`model_half_spread` at 0.91–0.97 (the last is a function of the other two).
So replacing the micro-price fixed an exact duplicate and left a near
duplicate. It is kept because AVAXUSDC and SUIUSDC — the wide-spread assets in
C1's cross-asset study — are where depth beyond the touch differs, and because
dropping a feature to improve a correlation with no downstream metric to judge
it by is unmotivated tuning. C3 decides.

**η cannot be calibrated against the reward's per-step noise.** The obvious
approach — set the penalty to a fraction of the PnL term's per-step standard
deviation — gives η = 0.38, which is two orders of magnitude too large. The
PnL term is near zero-mean and its total grows like √T; the penalty is
one-signed and grows like T. Only the episode totals are commensurable:
Σq² = 1545 against −93 bp, so η = 0.015 is a quarter of the PnL term.

**A 1 s policy step is C1's finding cashed in.** Acting every second rather
than every 100 ms is a 10× shorter episode, and C1's interval study is the
evidence that nothing is lost: post-fill drift, which dominates the loss, is
identical at every reaction interval from 100 ms to 2 ms. This is a prediction
C3 can falsify by running one configuration at 100 ms.

**What I would do differently.** The env was written before `run_random.py`
existed, so the observation scalings were guesses that a measurement later
had to correct. Writing the probe first — even a crude one — would have set
the feature scalings and η from data instead of from taste, and would have
caught the duplicate feature before it was in the observation space.
