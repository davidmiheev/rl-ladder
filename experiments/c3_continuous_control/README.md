# C3 — Continuous Control on the Coefficient Set

**Goal.** A real answer to whether learned, state-conditioned coefficients beat
a well-calibrated fixed formula, and which coefficients are worth learning at all.

**Prerequisites.** A5 (the algorithms), C1 (the baseline and the split), C2 (the environment)
**Companion lab.** Stable-Baselines3 TD3 / SAC docs
**Compute.** CPU. About 140 s per training session; the full grid is 18 runs of
10 sessions, roughly 4 hours at two workers (a session holds ~4.5 GB).

Checklist items live in [`docs/ladder.md`](../../docs/ladder.md#c3); the
argument, the tables and the caveats are in
[`docs/writeups/c3-continuous-control.md`](../../docs/writeups/c3-continuous-control.md).

## What is here

| file | what it is |
| --- | --- |
| [`train.py`](train.py) | the outer loop: cycles over training sessions, one coefficient decision per second |
| [`evaluate.py`](evaluate.py) | replays a policy over a day set and scores it with C1's `LinearAssetRecord` call |
| [`summarise.py`](summarise.py) | collects the evaluations into one table, and the two-panel figure |

The algorithms themselves are **A5's**, imported not copied: `build_agent`,
`select_action` and `update_step` come from
[`../a5_continuous_control/continuous.py`](../a5_continuous_control/continuous.py).

## How to run

```bash
python experiments/c3_continuous_control/train.py --algo td3 --seed 0 --participation forced
python experiments/c3_continuous_control/evaluate.py --policy runs/c3/<run>/policy.pt --days test
python experiments/c3_continuous_control/evaluate.py --policy baseline --days test
python experiments/c3_continuous_control/summarise.py --days train --plot docs/assets/c3/held-out.png
```

`--participation free` uses C2's action ranges; `forced` caps the half-spread
scale so leaving the market costs more. `--learnable no-gamma` pins γ at its
calibrated 0.05 and removes it from the action space for item 3.

## Results

![held out](../../docs/assets/c3/held-out.png)

Median over five held-out sessions; the baseline row is C1's `glft-grid`
re-scored through this code path, which reproduces its published +0.37 exactly.

| configuration | SR | Sortino | Return | RoMDD | MaxPos | fills/day | days + |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **C1 baseline** | **+0.37** | **+0.43** | **+0.0001** | **+0.018** | 89,816 | 3,868 | **3/5** |
| `td3-forced` (3 seeds) | −0.80 / −4.21 / −16.95 | −1.0 / −6.0 / −19.5 | −0.001 | −0.06 / −0.30 / −0.67 | 88–145k | 1,526–8,016 | 2 / 2 / 0 |
| `ddpg-forced` (3 seeds) | −3.61 / −20.79 / −25.65 | −4.9 / −23.9 / −32.7 | −0.002 to −0.036 | −0.17 / −0.66 / −0.80 | 126–185k | 945–90,515 | 2 / 0 / 0 |
| `ddpg-free` (2 seeds) | −3.44 / −21.25 | −4.7 / −27.3 | −0.002 / −0.004 | −0.15 / −0.77 | 104–170k | 5,419–11,806 | 1 / 1 |
| `td3-free` (2 seeds) | −14.10 / −31.04 | −19.7 / −34.6 | −0.003 / −0.002 | −0.63 / −0.90 | 54–77k | 911–2,556 | 1 / 1 |
| `sac-free` (2 seeds) | −23.16 / −24.41 | −25.6 / −27.3 | −0.003 | −0.83 / −0.90 | 72–83k | 2,566–3,052 | 0 / 0 |
| `sac-forced` (3 seeds) | −27.19 to −30.38 | −29.5 to −34.1 | −0.004 | −0.89 to −0.91 | 61–83k | 3,690–4,115 | 0 / 0 / 0 |
| `td3-forced`, γ pinned | *no trades* / −27.03 / *no trades* | — | 0.0000 / −0.024 | — | 0 / 168k / 0 | **0 / 79,332 / 0** | 1 / 0 / 1 |

Training sessions, same policies, same scorer:

| configuration | SR (train) | SR (held out) | vs baseline, train | vs baseline, held out |
| --- | --- | --- | --- | --- |
| C1 baseline | −32.34 | +0.37 | — | — |
| `ddpg-forced` s2 | **+6.88** (4/5 days +) | −3.61 | **+39.2** | −4.0 |
| `ddpg-forced` s0 | −1.75 | −20.79 | **+30.6** | −21.2 |
| `td3-forced` s0 | −9.72 | −4.21 | **+22.6** | −4.6 |
| `td3-forced` s2 | −30.06 | −0.80 | **+2.3** | −1.2 |

## Findings

**No learned configuration beats the fixed formula out of sample.** Best single
seed −0.80 against the baseline's +0.37; best median over seeds −4.21. That is
item 2's answer, on this asset and this split.

**Four of fourteen forced runs beat the baseline on the days they trained on,
and none beats it on the days they did not.** DDPG seed 2 is the sharpest: a
39-Sharpe-point improvement on training sessions with four of five profitable,
and four points below the baseline on held-out sessions.

Measured carefully, because the naive version of this comparison is wrong.
Raw returns are higher for *everything* on held-out days — those days were
quieter (C1: median σ 31.2 against 35.2 bp/√s) — so comparing a policy's
held-out return to its own training return would make several of these look
like they generalise. The comparison that survives is against the baseline on
the same days, which is what the tables and panel (b) show.

**They lose and carry more risk.** Most configurations hold more inventory than
the baseline (110–185k against 89.8k) while earning less, so this is not a
risk-return trade, just a worse point. MaxPositionValue is in the item's list
for exactly this reason.

**Pinning γ made things dramatically less stable — the opposite of item 3's
prior.** Two of three seeds take *zero* fills on every held-out session; the
third takes 79,332. With γ free the same algorithm stayed within a factor of
five on fills. The reason is visible in the coefficients: with γ free a policy
widens by raising γ, which lifts c1 and c2 together and holds the
spread-to-skew ratio the model intends; with γ pinned the only routes are
`c1_scale` and `skew_scale` moving independently, and seeds drove them to
opposite corners (3.74/1.82 and 3.74/0.011). γ is the coefficient the others
are defined in terms of, and removing it removed the parameterisation's own
coordination rather than usefully shrinking the hypothesis class.

**SAC is the most stable and the worst.** Its three seeds agree to two decimals
on every coefficient, and it trades four times the baseline's volume while
losing twice as much. The entropy term rewards keeping the action distribution
wide, which in this action space means continuing to quote.

**My "forced participation" constraint leaked, and the ablation found it.** The
half spread is `(c1_scale·c1 + ½σc2)·half_scale`; capping `half_scale` at 2×
leaves `c1_scale` (ceiling 4) as another route to the same corner. Twelve of
fourteen forced runs traded normally; the two γ-pinned seeds used the leak to
reach zero fills. A constraint written in the parameterisation rather than in
the quantity it bounds is usually leaky. C4 should cap the computed half
spread.

**What I would do differently.** The grid was sized before I knew an episode's
cost, so the first launch used A5's (256, 256) network and took 429 s per
session — 4× what was needed for a 10-dimensional observation. Ten minutes of
benchmarking up front would have bought the compute for two more seeds, which
is what this phase is actually short of: every conclusion rests on two or
three seeds over five training sessions, and "no configuration beat the
baseline" honestly reads as "not with this much data and no hyperparameter
search".
