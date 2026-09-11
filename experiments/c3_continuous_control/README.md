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

> **Correction.** The first version of this README reported that no learned
> configuration beat the fixed baseline, and that four runs beat it on
> training days and none on held-out days. Both statements came from
> summarising with the median of per-session Sharpes, which is the wrong
> statistic for a strategy whose losses sit in the tail, and both reverse
> under the period metric. The analysis below is the corrected one and the
> mistake is dissected in the writeup.

![held out](../../docs/assets/c3/held-out.png)

The five held-out sessions as one track record, ranked by cumulative return.
The baseline row is C1's `glft-grid` re-scored through this code path.

| configuration | seed | cumulative % | Sharpe (ann.) | worst session % | fills | up |
| --- | --- | --- | --- | --- | --- | --- |
| `td3-forced` | 2 | **+0.60** | **+4.97** | −0.20 | 33,859 | 2/5 |
| `td3-forced`, γ pinned | 0, 2 | +0.01 | — | 0.00 | **1** | 1/5 |
| `ddpg-forced` | 2 | −0.23 | −1.01 | −0.92 | 4,439 | 2/5 |
| `td3-free` | 0, 1 | −2.96, −3.16 | −12.6, −7.5 | −1.66, −2.99 | 12,702, 7,003 | 1/5 |
| `td3-forced` | 0, 1 | −3.19, −4.00 | −6.6, −9.7 | −2.88, −3.14 | 8,444, 17,607 | 2/5, 0/5 |
| `ddpg-free` | 0, 1 | −5.19, −3.88 | −11.3, −11.4 | −3.22, −2.41 | 58,109, 29,921 | 1/5 |
| `sac-free` | 0, 1 | −4.93, −4.76 | −15.3, −15.1 | −2.37, −2.43 | 17,189, 14,758 | 0/5 |
| `ddpg-forced` | 0 | −4.82 | −17.46 | −1.93 | 14,438 | 0/5 |
| `sac-forced` | 0, 1, 2 | −5.32, −6.60, −6.37 | −16.0, −15.1, −14.4 | −2.35, −3.17, −3.27 | 21–26k | 0/5 |
| **C1 baseline** | — | **−8.44** | **−11.19** | **−4.96** | 23,897 | 3/5 |
| `ddpg-forced` | 1 | −18.19 | −19.67 | −7.26 | 402,209 | 0/5 |
| `td3-forced`, γ pinned | 1 | −20.88 | −20.84 | −8.38 | 418,116 | 0/5 |

Per family, median seed: `td3-free` −3.06, `td3-forced` −3.19, `ddpg-free`
−4.54, `ddpg-forced` −4.82, `sac-free` −4.85, `sac-forced` −6.37, **baseline
−8.44**.

## Findings

**The fixed baseline finishes seventeenth of nineteen.** Fifteen of eighteen
learned configurations beat it over the held-out period, and every family
beats it on its median seed. That is item 2's answer, and it is the opposite
of what the per-session medians said.

**One configuration is genuinely good, and not by declining to play.** TD3
forced seed 2 returns +0.60% against the baseline's −8.44%, with an
annualised Sharpe of +4.97 against −11.19 and a worst session of −0.20%
against −4.96% — while taking **42% more fills** than the baseline. The two
γ-pinned seeds just below it, on a combined total of one fill, are the
degeneracy C2 warned about; the fills column is what separates them.

**But the win is risk management, not profit.** Exactly one of eighteen
configurations ends the period positive while actually trading; the median
TD3-forced seed is −3.19%. Learned coefficients lose *less* than the
calibrated formula, mostly by not blowing up in the volatile regime.

**Why the medians said the opposite.** The baseline's five held-out sessions
are +0.01, +0.02, +0.04, **−4.96**, **−3.55** percent — it wins three by
hundredths and loses two by whole percents, so its median session is the best
in the table and its period is second-worst. σ on those two sessions was 47.0
and 49.4 bp/√s against 12.8–31.2 on the other three. C1 had already measured
that the gross value of a fill is `-0.72·σ - 16.2` ticks; the volatile days
are where the money goes, and a median over five sessions averages exactly
that away.

**There is no generalisation gap.** Every configuration that beats the
baseline beats it by *more* out of sample than in (panel b). TD3 seed 2 is
4.08 points worse than the baseline on training sessions and 9.04 better on
held-out ones. The edge is tail-avoidance and the held-out set has more tail —
two sessions above σ 47 against one in training. C4 gets no positive control
for its reward-hacking item from this phase.

**Pinning γ destabilises — withdrawn pending a re-run.** The γ-pinned runs
were trained with a confound I had claimed to remove: commit `c15a515` says
pinned coefficients left the agent's action space, but the edit silently
failed to apply, so every one of them used a four-output actor with a column
the critic never saw vary. The re-run is in progress. What was observed, and
may or may not survive it: Two of three seeds take a combined
*one fill* across five sessions; the third takes 418,116 and loses 20.88%.
With γ free all three seeds trade 8,000–34,000 fills and land within four
points of each other. With γ free a policy widens by raising γ, lifting c1
and c2 together and holding the spread-to-skew ratio the model intends;
pinned, the only routes are `c1_scale` and `skew_scale` moving independently,
and seeds drove them to opposite corners (3.74/1.82 and 3.74/0.011). γ is the
coefficient the others are defined in terms of.

**SAC is the most stable and the worst.** Three seeds agreeing to two
decimals on every coefficient, and the three worst non-degenerate rows in the
table. The entropy term rewards keeping the action distribution wide, which
here means continuing to quote into the volatile sessions.

**My "forced participation" constraint leaked, and the ablation found it.**
The half spread is `(c1_scale·c1 + ½σc2)·half_scale`; capping `half_scale` at
2× leaves `c1_scale` (ceiling 4) as another route to the same corner. Twelve
of fourteen forced runs traded normally; the two γ-pinned seeds used the leak
to reach one fill. A constraint written in the parameterisation rather than
in the quantity it bounds is usually leaky. C4 should cap the computed half
spread.

**What I would do differently.** Two things, and the second is the real one.

The grid was sized before I knew an episode's cost, so the first launch used
A5's (256, 256) network and took 429 s per session — 4× what a 10-dimensional
observation needs. Ten minutes of benchmarking up front would have bought two
more seeds, which is what this phase is short of.

More importantly: I chose the summary statistic before looking at the
distribution it would summarise. The median was chosen for a good reason — to
stop one session dominating a mean — and it was the wrong tool for a
distribution whose whole character is two bad sessions out of five. C1 had
already measured that the loss scales with σ and that σ varies three-fold
across these days, so the shape was known before the first policy was
trained. The rule that survives: decide what a strategy's risk looks like
*before* choosing how to average it, and when the answer is "the tail",
report the tail.
