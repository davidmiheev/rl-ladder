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

**Two summaries, because they disagree, and the disagreement turned out to be
the result.** Per-session medians describe the typical day; the period —
cumulative return, the annualised Sharpe of the daily returns, the worst
session — describes the track record. For a strategy whose losses are
concentrated in two of five sessions these rank the field almost inversely,
and §5 shows the period is the summary that answers the item's question.
Seeds are always listed individually as well. (The first version of this
phase reported only the medians and drew the opposite conclusion; see §5.)

## 5. Results

### Item 1 — the three algorithms

Eighteen runs: three algorithms × three seeds under forced participation,
three × two under free, and TD3 with γ pinned × three seeds. Ten sessions
each, cycling twice over the five training days.

On **training** sessions the three algorithms behave quite differently.

| configuration | fills (final session) | PnL (% of book) | γ | half scale | skew scale |
| --- | --- | --- | --- | --- | --- |
| `sac-forced` (3 seeds) | ~10,000 | −1.82 to −2.07 | 0.19 | 1.09 | 0.39 |
| `sac-free` (2 seeds) | ~23,000 | −4.38 to −4.55 | 0.20 | 1.97 | 0.39 |
| `ddpg-forced` (3 seeds) | 2,508–3,740 | −0.67 to +0.26 | 0.35–0.53 | 1.14–1.30 | 0.43–0.55 |
| `td3-forced` (3 seeds) | 904–1,792 | −0.35 to +0.25 | 0.29–0.56 | 1.11–1.18 | 0.18–0.48 |
| `td3-forced`, γ pinned (3 seeds) | **42 / 71 / 6,793** | +0.38 / +0.04 / −0.67 | pinned | 0.73–1.95 | 0.01–1.82 |

SAC is by far the most *stable* — its three seeds agree to two decimal places
on every coefficient — and by far the worst: it quotes tight, trades four
times the baseline's volume and loses twice as much. Its entropy term rewards
keeping the action distribution wide, which in this action space means
continuing to quote. TD3 and DDPG both drift towards higher γ and a wider
spread floor, and several seeds reach a positive training session.

None of that survives contact with held-out data.

### Item 2 — held-out days

> **Correction.** The first version of this section reported that no learned
> configuration beat the fixed baseline, on the strength of the median Sharpe
> across held-out sessions. That summary statistic was the wrong one, and it
> inverted the conclusion. What follows is the corrected analysis; the
> mistake is dissected below rather than quietly fixed, because it is the
> seventh instance in this repo of a headline metric hiding the effect it was
> chosen to measure, and the first one to reverse a phase's answer.

The five held-out sessions as **one track record**, which is what an
allocator sees. Ranked by cumulative return; the baseline row is C1's
`glft-grid` re-scored through C3's code path, reproducing its published
numbers exactly.

| configuration | seed | cumulative % | Sharpe (ann.) | worst session % | fills | sessions up |
| --- | --- | --- | --- | --- | --- | --- |
| `td3-forced` | 2 | **+0.60** | **+4.97** | −0.20 | 33,859 | 2/5 |
| `td3-forced`, γ pinned | 0, 2 | +0.01 | — | 0.00 | **1** | 1/5 |
| `ddpg-forced` | 2 | −0.23 | −1.01 | −0.92 | 4,439 | 2/5 |
| `td3-free` | 0 | −2.96 | −12.60 | −1.66 | 12,702 | 1/5 |
| `td3-free` | 1 | −3.16 | −7.52 | −2.99 | 7,003 | 1/5 |
| `td3-forced` | 0 | −3.19 | −6.60 | −2.88 | 8,444 | 2/5 |
| `ddpg-free` | 1 | −3.88 | −11.40 | −2.41 | 29,921 | 1/5 |
| `td3-forced` | 1 | −4.00 | −9.68 | −3.14 | 17,607 | 0/5 |
| `sac-free` | 1 | −4.76 | −15.08 | −2.43 | 14,758 | 0/5 |
| `ddpg-forced` | 0 | −4.82 | −17.46 | −1.93 | 14,438 | 0/5 |
| `sac-free` | 0 | −4.93 | −15.28 | −2.37 | 17,189 | 0/5 |
| `ddpg-free` | 0 | −5.19 | −11.31 | −3.22 | 58,109 | 1/5 |
| `sac-forced` | 0 | −5.32 | −15.98 | −2.35 | 21,777 | 0/5 |
| `sac-forced` | 2 | −6.37 | −14.39 | −3.27 | 24,489 | 0/5 |
| `sac-forced` | 1 | −6.60 | −15.11 | −3.17 | 26,379 | 0/5 |
| **C1 baseline** | — | **−8.44** | **−11.19** | **−4.96** | 23,897 | 3/5 |
| `ddpg-forced` | 1 | −18.19 | −19.67 | −7.26 | 402,209 | 0/5 |
| `td3-forced`, γ pinned | 1 | −20.88 | −20.84 | −8.38 | 418,116 | 0/5 |

**The fixed baseline finishes seventeenth of nineteen.** Fifteen of the
eighteen learned configurations beat it, and every configuration *family*
beats it on its median seed:

| family | median cumulative % | seeds |
| --- | --- | --- |
| `td3-free` | −3.06 | −2.96, −3.16 |
| `td3-forced` | −3.19 | −3.19, −4.00, **+0.60** |
| `ddpg-free` | −4.54 | −5.19, −3.88 |
| `ddpg-forced` | −4.82 | −4.82, −18.19, −0.23 |
| `sac-free` | −4.85 | −4.93, −4.76 |
| `sac-forced` | −6.37 | −5.32, −6.60, −6.37 |
| **C1 baseline** | **−8.44** | — |

**One configuration is genuinely good, and it is not good by refusing to
play.** TD3 forced, seed 2: +0.60% of book over the period against the
baseline's −8.44%, annualised Sharpe +4.97 against −11.19, worst session
−0.20% against −4.96% — and it does this while taking **42% more fills than
the baseline** (33,859 against 23,897). That rules out the degeneracy C2
warned about. The two γ-pinned seeds sitting just below it, on a *combined
total of one fill*, are the degeneracy, and the fills column is what separates
them.

**But the win is risk management, not profit.** Exactly one of eighteen
configurations ends the period positive while actually trading. The median
TD3-forced seed is −3.19%. The correct summary is that learned coefficients
lose *less* than the calibrated formula on held-out days, mostly by not
blowing up in the volatile regime — and that one seed of three found something
better than that.

![held out](../assets/c3/held-out.png)

### Why the first version of this concluded the opposite

The two summaries disagree because of what the baseline's held-out
distribution looks like:

| | 09-15 | 09-18 | 09-20 | 09-22 | 09-25 |
| --- | --- | --- | --- | --- | --- |
| σ (bp/√s), from C1 | 31.2 | 24.5 | **12.8** | **47.0** | **49.4** |
| C1 baseline, % of book | +0.01 | +0.02 | +0.04 | **−4.96** | **−3.55** |
| `td3-forced` seed 2 | +0.66 | +0.38 | −0.15 | **−0.10** | **−0.20** |

The baseline wins three sessions by hundredths of a percent and loses two by
whole percents. Its **median session is the best in the table and its period
is second-worst**. TD3 seed 2 gives up the small wins and caps the two
volatile sessions at −0.10% and −0.20%.

So a median over five sessions is blind to precisely the tail that defines a
market maker's risk — and C1 had already measured that tail and said where it
lives: the gross value of a fill is $-0.72\sigma - 16.2$ ticks, so the
volatile days are where the money goes. I had that number in hand and still
chose a statistic that averages it away.

The reasoning that produced the mistake is in §4 of the first draft, and it
was not careless: *"five held-out sessions are few enough that one session
decides a mean, which is the mistake Part A kept making"*. That is true. The
error was treating the median as the safe alternative. A median protects
against one session distorting a typical-day estimate; it does not protect
against anything when the question is a period's total, and for a strategy
whose losses are concentrated in two of five sessions it actively removes the
answer. The right response to a heavy tail is to report the tail — worst
session, cumulative, drawdown — not to summarise it away.

Both tables are now printed by `summarise.py`, the period one by default.

### The item's own metrics, per session

For completeness, the median-over-sessions figures the item's wording asks
for. These are the numbers the first version led with, and they rank the
baseline first:

| configuration | SR | Sortino | Return | RoMDD | MaxPos | fills/day | days + |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **C1 baseline** | **+0.37** | **+0.43** | **+0.0001** | **+0.018** | 89,816 | 3,868 | **3/5** |
| `td3-forced` s2 | −0.80 | −1.01 | −0.0010 | −0.055 | 144,523 | 8,016 | 2/5 |
| `ddpg-forced` s2 | −3.61 | −4.90 | −0.0019 | −0.165 | 125,656 | 945 | 2/5 |
| `td3-forced` s0 | −4.21 | −6.01 | −0.0007 | −0.302 | 109,719 | 1,526 | 2/5 |
| `sac-forced` (3 seeds) | −27.2 to −30.4 | −29.5 to −34.1 | −0.004 | −0.89 to −0.91 | 61–83k | 3,690–4,115 | 0/5 |

MaxPositionValue is the one per-session figure that still says something the
period table does not: most learned policies hold more inventory than the
baseline (110–185k against 89.8k). They are taking more position risk and, on
the period, being paid for it — which is a trade, not a free lunch.

### Is any of it overfitting?

> **Second correction.** The first version of this section reported that four
> of fourteen runs beat the baseline on training days and none beat it on
> held-out days, and called that a textbook generalisation gap. That was the
> same median-of-per-session-Sharpes artefact. On the period metric the gap
> does not exist.

The same deterministic policies over the five **training** sessions, measured
against the baseline on those same sessions, next to the held-out figure
(cumulative % of book; the baseline is −4.53% on training sessions and −8.44%
on held-out ones):

| configuration | vs baseline, train | vs baseline, held out |
| --- | --- | --- |
| `td3-forced` s2 | −4.08 | **+9.04** |
| `ddpg-forced` s2 | +6.36 | **+8.21** |
| `td3-forced` s0 | +2.98 | **+5.26** |
| `td3-forced` s1 | +0.92 | **+4.45** |
| `ddpg-forced` s0 | +3.90 | **+3.63** |
| `sac-forced` s0, s1, s2 | +0.93, +0.99, +0.92 | **+3.12, +1.84, +2.07** |
| `ddpg-forced` s1 | −47.13 | −9.74 |
| `td3-forced`, γ pinned s1 | −41.81 | −12.44 |

**Every configuration that beats the baseline beats it by more on the
held-out sessions than on the sessions it was fitted to** — in panel (b),
every point sits above the dotted line. TD3 seed 2 is the extreme case: 4.08
points *worse* than the baseline on training sessions and 9.04 points better
on held-out ones.

That is not overfitting; it is close to the opposite, and it has a mechanical
explanation rather than being luck. The learned policies' entire advantage is
avoiding the baseline's tail, and the held-out set contains proportionally
more tail — two of its five sessions have σ above 47 bp/√s against one of
five in the training set. An edge that consists of not blowing up shows up
most where there is most opportunity to blow up.

It also means C3 supplies no positive control for C4's reward-hacking item.

### Item 3 — the ablation

> **Item 3's result is currently withdrawn.** The ablation below was run with
> a confound I had claimed to remove. Commit `c15a515` says pinned
> coefficients were taken out of the agent's action space; they were not —
> three `str.replace` calls silently failed to match and I asserted on none of
> them, so every γ-pinned run used a four-output actor whose fourth column the
> critic never saw vary. That is exactly the unidentified-gradient confound
> §3 describes, and it is a live candidate for the instability reported here.
> The arm is being re-run with the fix verified rather than assumed (the
> checkpoint's actor now has three outputs); items 1 and 2 are unaffected,
> because with all four coefficients learnable the overwrite is the identity.

The ladder's prior is that learning fewer coefficients should be more stable:
a smaller hypothesis class is harder to overfit. **The measurement says the
opposite.**

| configuration | cumulative % by seed | total held-out fills by seed |
| --- | --- | --- |
| `td3-forced`, all four | −3.19, −4.00, **+0.60** | 8,444 / 17,607 / 33,859 |
| `td3-forced`, γ pinned | +0.01, **−20.88**, +0.01 | **1** / 418,116 / **1** |

Pinning γ at its calibrated 0.05 produced the two most extreme outcomes in the
whole phase — two seeds that take a combined *one fill* across five sessions,
and one that takes 418,116 and loses 20.88% of book — where the same algorithm
with γ free kept all three seeds trading between 8,000 and 34,000 fills and
within four points of each other.

This is the one conclusion that does not change under the corrected metric.
Its median cumulative return (+0.01%) is nominally the best in the table, and
it is worthless: two of the three seeds achieved it by not quoting. This is
precisely why the fills column is printed first.

The mechanism is visible in the coefficients. With γ free, the policy reaches
a wide spread by *raising γ*, which lifts `c1` and `c2` together and keeps the
half spread and the skew in the ratio the model intends. With γ pinned, the
only route to a wide spread is `c1_scale`, and the only route to inventory
control is `skew_scale` — two knobs that move the half spread and the skew
independently, and that the policy pushes to opposite corners (seed 0 ended at
`c1_scale` 3.74 with `skew_scale` 1.82; seed 2 at 3.74 and 0.011). Removing
the coefficient that *couples* the others did not shrink the hypothesis class
in a useful direction; it removed the parameterisation's own coordination and
left two unconstrained knobs.

That is a real answer to "which coefficients are worth learning at all", and
it is not the one the item anticipated: γ is the one worth keeping learnable,
because it is the coefficient the others are defined in terms of.

### A constraint that leaked

`forced` participation caps the half-spread *scale* at 2×, but the half spread
is $(s_{c_1} c_1 + \tfrac{1}{2}\sigma c_2) s_{\text{half}}$ — so pushing
$s_{c_1}$ to its ceiling of 4 reaches the same "quote two hundred ticks out"
corner by another route. Twelve of the fourteen forced runs still traded
normally, but the two γ-pinned seeds found the leak and used it, which is
exactly how they reached zero fills.

So "forced participation" does not mean what its name claims. It raises the
cost of leaving the market rather than forbidding it. A constraint that
actually bound would have to cap the *computed half spread*, not one of its
factors — which is the version C4 should use, and a reminder that a constraint
expressed in the parameterisation rather than in the quantity it is meant to
bound is usually leaky.

## 6. What this sets up

**C4 does not get a positive control from here.** Its third item hunts for
RL-tuned coefficients that overfit to the training days' microstructure, and
C3 found none: every configuration that beats the baseline beats it by more
out of sample than in. C4 will have to construct its own positive control —
the obvious one being a policy deliberately trained on a single session — and
should not expect its diagnostics to fire on C3's outputs.

**Three concrete things C4 should change.**

- *Cap the computed half spread, not one of its factors.* C3's forced
  participation capped `half_spread_scale` and the γ-pinned seeds simply
  widened through `c1_scale` instead. A constraint expressed in the
  parameterisation rather than in the quantity it is meant to bound is
  usually leaky, and this one was.
- *Report fills before Sharpe, and treat a zero-fill policy as a failed run
  rather than an undefined one.* Two seeds here score approximately the best
  return in the table by never quoting.
- *Give the comparison more than five training sessions.* Every result in
  this phase is a policy fitted to five days, and the generalisation gap is
  consistent with that being simply too few. C3 cannot separate "RL does not
  help on this problem" from "RL cannot be fitted on five sessions", and
  saying which would need more data rather than more algorithms.

**C5's premise gets some support.** The one thing every policy here lacked is
a directional signal: the state carries volatility, imbalance and flow, and
C1 measured that flow predicts the next 100 ms with a 1.9σ mean, but nothing
in the state predicts *direction over the horizon the inventory is held*. If
learned coefficients are ever going to beat the formula, the alpha signals C5
adds are the most likely reason — which makes C5 a test of the state rather
than of the algorithm.

### Limits worth stating

- **Five training sessions, ten episodes, three seeds.** That is a small
  experiment by RL standards, and the honest reading of "no configuration
  beat the baseline" is "not with this much data and this much tuning".
  Nothing here licenses "RL does not work for market making".
- **No hyperparameter search.** Learning rates, γ_RL, τ, batch size and the
  exploration noise are A5's defaults throughout. A negative result under
  default hyperparameters is weaker than one under tuned ones, and tuning
  them on five days would have been its own overfitting problem.
- **The reward's risk penalty η = 0.015 was calibrated once, in C2, against
  the baseline's own numbers.** No sensitivity was run, so "the policy
  carries more inventory than the baseline" is partly a statement about η.
- **The episode is a whole session and the discount is 0.99**, i.e. an
  effective horizon of about one hundred seconds against an episode of
  85,800 steps. Whether that is the right horizon for an inventory problem
  was not tested, and it is the first hyperparameter I would question.
- **The forced/free distinction is weaker than its name**, as above.

## 7. Where these items come from

| item | primary source | why it was picked |
| --- | --- | --- |
| 1 — DDPG/TD3/SAC on the coefficient set | the ladder's own; the implementations are A5's, validated there against [Stable-Baselines3's TD3](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html) and [SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html) documentation | The action is a low-dimensional continuous vector, which is the setting these algorithms were designed for, so the novelty is entirely in the environment. Running three rather than one is what separates "RL does not help here" from "this particular optimiser did not help here". |
| 2 — compare on held-out days | the ladder's own; the split is C1's, fixed in `config.py` before any policy existed | A backtest comparison on days used for calibration or training measures memorisation. The risk-adjusted measures matter because a strategy that earns more by carrying more inventory has not improved anything — which is why MaxPositionValue is in the list. |
| 3 — ablate γ | the ladder's own | Learning fewer coefficients is a smaller hypothesis class and harder to overfit, and the comparison says which coefficients genuinely benefit from being state-conditioned. C1 gives a prior worth testing: γ barely moves the half spread near its lower range, so it may be the least useful of the four to learn. |

The two-experiment structure in §2, the action-space treatment of pinned
coefficients in §3, and the fills-before-Sharpe rule in §4 are this repo's,
derived from C1's and C2's measurements.
