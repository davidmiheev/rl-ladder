# C8 — Risk controls, denoising and new quoting ideas, on a month split

Not a ladder phase. It comes from the Part C follow-up list after C4: a
circuit breaker, a daily loss limit, stress tests of the one configuration
that made money, the breadth assets on more held-out sessions, and RL on top
of the markout rule — plus denoising of the signals and a set of new quoting
ideas, each tested rather than argued.

Writeup: [`docs/writeups/c8-risk-controls.md`](../../docs/writeups/c8-risk-controls.md).

## The split

The whole month, with more of it for training: **24 training sessions**
(2025-08-26 to 09-18) and **7 held-out sessions** (09-19 to 09-25). The
held-out week keeps a weekend and the 2025-09-22 crash. Every choice is made
on training sessions; held-out sessions are only reported. C4 and C7 used
20 / 11 over the same month, so their held-out numbers are recomputed on the
last seven sessions wherever they are compared here.

## How to run

```bash
# The pipeline, per asset (every stage reads and writes the same cache,
# runs/c8/days/<config hash>/<session>.json, and skips what exists):
C8_SYMBOLS=ethusdc python experiments/c8_risk_controls/c8.py run --stage A --shard 0/3
python experiments/c8_risk_controls/c8.py status

# Stages, each on top of the previous stage's choice per (strategy, asset):
#   A        floor x markout kappa x skew grid (C4's grid, kappa 1 added)
#   B        circuit breaker and loss limits
#   C        denoising and new quoting ideas
#   D        greedy combination of the families' best variants (training days)
#   D-test   the choices on all sessions
#   E        stress tests of the choices
#   breadth  the seven other assets, floor x kappa per strategy

# RL on top of the markout rule (recipe v2-mk), from a stage-A choice:
python experiments/c6_strategy_zoo/train.py --strategy obi-wide --symbol ethusdc \
    --recipe v2-mk --train-set c8 --passes 1 --seed 0 \
    --set half_spread_bp=6.0 --set markout_kappa=4.0 --set skew_bp=14.8
python experiments/c6_strategy_zoo/evaluate.py --policy runs/c6/<run>/policy.pt --days test

# Tables and figures (docs/assets/c8/):
python experiments/c8_risk_controls/report.py
```

`C8_DEADLINE=<epoch seconds>` makes a worker stop taking new sessions, so a
time-limited host finishes cleanly. XRPUSDC and the breadth tier ran on Kaggle
CPU sessions (four cores each), ETHUSDC on this machine; their caches merge
because a configuration's key is a hash of the configuration alone.

## The controls and ideas

All live in C6's quoting kernel (`experiments/c6_strategy_zoo/zoo.py`) and are
off by default; an old C7 markout run replays to the identical return and fill
count with them compiled in.

| family | parameter(s) | what it does |
| --- | --- | --- |
| circuit breaker | `cb_mode`, `cb_vol`, `cb_flow`, `cb_fast_hl`, `cb_cooldown` | pause quoting when fast/slow volatility exceeds `cb_vol`, or when signed aggressive flow over its fast average is one-sided beyond `cb_flow` while activity is above normal; both sides (`cb_mode` 1) or only the side being hit (2), for `cb_cooldown` steps |
| loss limit | `loss_limit_bp`, `loss_mode`, `loss_cooldown` | reduce-only after the session loses `loss_limit_bp` of the book (1), or after a drawdown of that size from the session's peak, for `loss_cooldown` steps (2) |
| signal smoothing | `signal_hl` | exponential average of the strategy's fair-price signal |
| alpha clipping | `alpha_clip` | clip OBI's standardised imbalance at ±`alpha_clip` |
| markout half-life | `markout_halflife` | how many fills the markout average spans |
| momentum | `mom_coef`, `mom_hl` | lean the fair price along the recent mid drift (C7 measured positive lag-1 autocorrelation) |
| per-side markout | `markout_sided` | one toxicity average per side; widen each side by its own |
| requote hysteresis | `requote_steps` | keep a resting quote while the new one is within that many grid steps, to keep queue position |
| toxicity sizing | `tox_size` | shrink order size as toxicity rises |
| inventory flatten | `inv_flatten` | above that fraction of the cap, cut back to half of it with an IOC order at the touch (taker fee) |
| time of day | `tod0`..`tod5` | spread-floor multiplier per four-hour UTC block |

Stress variants (stage E): maker fee 0, +1 bp and +2 bp instead of the −0.5 bp
rebate; order latency ×2 and ×4; the risk-averse and log queue models; twice
the order size; the floor ×0.8 and ×1.25; κ ×0.5 and ×2.

## Results

To be filled in from `report.py` as the stages finish.

## Findings

To be filled in.
