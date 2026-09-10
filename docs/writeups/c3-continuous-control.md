# C3 — Continuous Control on the Coefficient Set: task, goal, and how it was solved

The first real test in Part C: do learned, state-conditioned coefficients beat
well-calibrated fixed ones? Lab notebook:
[`experiments/c3_continuous_control/README.md`](../../experiments/c3_continuous_control/README.md).

---

## 1. The task

[`ladder.md`](../ladder.md#c3) states three items:

1. Apply DDPG/TD3/SAC to the C2 environment: policy outputs (γ, c1,
   half-spread-scale) per step, conditioned on state.
2. Compare against the C1 fixed-calibration baseline on held-out days: Sharpe,
   Sortino, MaxPositionValue, ReturnOverMDD.
3. Ablate: fix γ at its calibrated value and let RL learn only c1/skew, vs.
   letting RL learn all three jointly — compare sample efficiency and
   stability.

## 2. The goal, and the question that had to be fixed first

The stated goal is "a real answer to whether learned, state-conditioned
coefficients beat a well-calibrated fixed formula". C2 ended by showing that
question, asked of this environment as designed, has a degenerate answer.

The action space contains a corner — the half-spread scale at its maximum —
that quotes six hundred ticks from the mid, takes **zero fills**, and returns
**exactly zero**. C1 established that the fixed baseline loses 0.93% of book
on a training session. So "beat the baseline" is achieved by not trading, and
an agent that finds that corner has answered the question truthfully and
uselessly.

A five-episode smoke test confirmed it is not hypothetical: TD3 drove the
half-spread scale to 7.1 and fills to zero.

So C3 runs two experiments rather than one.

- **Free participation** — the action space as C2 defined it. This asks what
  RL does when leaving the market is available, and the answer is a finding
  about the reward, not about the algorithms.
- **Forced participation** — the half-spread scale capped at 2× the
  calibrated value, so quoting far enough away to stop trading is not
  representable. The baseline remains interior to the range. This asks the
  question the phase is actually about: *given that you must make a market,
  do learned coefficients beat fixed ones?*

Reporting only the first would be a non-result dressed as a result. Reporting
only the second would hide that the unconstrained problem has a trivial
optimum. Both are reported.

## 3. What is reused, and what is not

**The algorithms are A5's, imported rather than reimplemented.** `build_agent`,
`select_action` and `update_step` were split out of A5's `train()` so that
both phases drive one implementation of DDPG, TD3 and SAC. Only the outer loop
is new, and it has to be: an episode here is a trading session that costs
about 140 seconds, so the loop cycles over the five training days and does one
gradient update every eight environment steps rather than one per step. With
85,800 transitions per session, data is not the scarce resource; wall clock
is.

**Two deviations from "reuse A5 directly", both deliberate.**

*The network is (64, 64), not A5's (256, 256).* A5's width was its default for
Pendulum and LunarLander. Here the observation is 10-dimensional and the
action at most 4-dimensional, so 256 means 140k critic parameters and 9.6 ms
per update against 10k and 3.9 ms. The first launch at 256 took 429 seconds
per episode, essentially all of it gradient updates.

*A pinned coefficient is removed from the action space, not overwritten after
the actor emits it.* These look equivalent and are not. If the actor keeps
emitting a coefficient that is then replaced by a constant, the critic sees no
variance in that column, its gradient there is unidentified, and the actor
follows that arbitrary gradient through a shared trunk into the coefficients
that do matter. That would confound precisely the comparison item 3 exists to
make, so the ablation's agent has a three-dimensional action space rather than
a four-dimensional one with a dead column.

**The action space has four coefficients, not the three the item names.** C2
split the ladder's single "half-spread/skew scaling term" in two, because C1
showed the coupling between spread and skew is what breaks the fixed model:
both scale with $\sigma c_2$, so 0.67 units of inventory cancelled the whole
half spread, and the source notebook introduces `adj2` by hand for exactly
that reason. The ablation is therefore run over four.

## 4. What is measured

Every policy is scored by replaying whole sessions and passing the equity
record to `hftbacktest.stats.LinearAssetRecord` — the same call C1 used for
the fixed baseline, on the same record type — so the two phases' tables are
directly comparable rather than merely similar. The baseline itself is
re-scored through C3's code path, so that no difference between the rows can
be an artefact of two different scorers.

Reported per configuration: Sharpe, Sortino, MaxPositionValue and
ReturnOverMDD as the item asks, and **fills** next to them. A risk-adjusted
measure computed over a strategy that took eleven fills is not a risk-adjusted
measure, and the fills column is the one that says whether a row is a market
maker at all.

Seeds are reported individually and then as a median, never as a mean: five
held-out sessions and three seeds are few enough that one session decides a
mean, which is the mistake Part A kept making.

## 5. Results

_(pending: the grid is 18 runs of 10 sessions each)_

## 6. What this sets up

_(pending)_

## 7. Where these items come from

| item | primary source | why it was picked |
| --- | --- | --- |
| 1 — DDPG/TD3/SAC on the coefficient set | the ladder's own; the implementations are A5's, validated there against [Stable-Baselines3's TD3](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html) and [SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html) documentation | The action is a low-dimensional continuous vector, which is the setting these algorithms were designed for, so the novelty is entirely in the environment. Running three rather than one is what separates "RL does not help here" from "this particular optimiser did not help here". |
| 2 — compare on held-out days | the ladder's own; the split is C1's, fixed in `config.py` before any policy existed | A backtest comparison on days used for calibration or training measures memorisation. The risk-adjusted measures matter because a strategy that earns more by carrying more inventory has not improved anything — which is why MaxPositionValue is in the list. |
| 3 — ablate γ | the ladder's own | Learning fewer coefficients is a smaller hypothesis class and harder to overfit, and the comparison says which coefficients genuinely benefit from being state-conditioned. C1 gives a prior worth testing: γ barely moves the half spread near its lower range, so it may be the least useful of the four to learn. |

The two-experiment structure in §2, the action-space treatment of pinned
coefficients in §3, and the fills-before-Sharpe rule in §4 are this repo's,
derived from C1's and C2's measurements.
