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

Every policy replayed over the five held-out sessions and scored with the same
`LinearAssetRecord` call C1 used. The baseline row is C1's `glft-grid`,
re-scored through C3's code path — it reproduces C1's published +0.37 exactly,
which is the check that the two tables are comparable.

| configuration | seed | SR | Sortino | Return | RoMDD | MaxPos | fills/day | days + |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **C1 baseline** | — | **+0.37** | **+0.43** | **+0.0001** | **+0.018** | 89,816 | 3,868 | **3/5** |
| `td3-forced` | 2 | −0.80 | −1.01 | −0.0010 | −0.055 | 144,523 | 8,016 | 2/5 |
| `ddpg-forced` | 2 | −3.61 | −4.90 | −0.0019 | −0.165 | 125,656 | 945 | 2/5 |
| `ddpg-free` | 0 | −3.44 | −4.73 | −0.0018 | −0.147 | 169,932 | 11,806 | 1/5 |
| `td3-forced` | 0 | −4.21 | −6.01 | −0.0007 | −0.302 | 109,719 | 1,526 | 2/5 |
| `td3-free` | 0 | −14.10 | −19.68 | −0.0032 | −0.628 | 76,652 | 2,556 | 1/5 |
| `td3-forced` | 1 | −16.95 | −19.51 | −0.0026 | −0.667 | 87,873 | 3,670 | 0/5 |
| `ddpg-forced` | 0 | −20.79 | −23.90 | −0.0054 | −0.659 | 147,500 | 3,150 | 0/5 |
| `ddpg-free` | 1 | −21.25 | −27.30 | −0.0036 | −0.772 | 103,500 | 5,419 | 1/5 |
| `sac-free` | 0, 1 | −23.16, −24.41 | −25.6, −27.3 | −0.0026 | −0.83, −0.90 | 71–83k | 2,566–3,052 | 0/5 |
| `ddpg-forced` | 1 | −25.65 | −32.70 | −0.0359 | −0.802 | 185,156 | **90,515** | 0/5 |
| `td3-forced`, γ pinned | 1 | −27.03 | −34.03 | −0.0236 | −0.923 | 168,374 | **79,332** | 0/5 |
| `sac-forced` | 0, 1, 2 | −27.2 to −30.4 | −29.5 to −34.1 | −0.004 | −0.89 to −0.91 | 61–83k | 3,690–4,115 | 0/5 |
| `td3-forced`, γ pinned | 0, 2 | *no trades* | — | 0.0000 | — | 0 | **0** | 1/5 |

**Not one of the eighteen beats the fixed formula.** The best single seed is
TD3 at −0.80 against the baseline's +0.37; the best median over seeds is TD3
forced at −4.21. The answer to the item's question, on this asset and this
split, is no.

![held out](../assets/c3/held-out.png)

Three details matter more than the headline.

**They lose *and* carry more risk.** The item asks for MaxPositionValue for a
reason — a policy that earns more by holding more inventory has not improved
anything. Here it runs the other way: most configurations hold *more*
inventory than the baseline (110k–185k against 89.8k) while earning less.
There is no risk-return trade being made, just a worse point.

**Two seeds behave completely differently out of sample.** `ddpg-forced` seed
1 takes 90,515 fills per held-out session against roughly 3,600 per training
session, and `td3-forced` with γ pinned seed 1 takes 79,332. A twenty-five-fold
change in trading volume between training and held-out days is not a policy
generalising badly; it is a policy whose behaviour is not defined by anything
stable in the state.

**Two seeds stop trading entirely.** The γ-pinned seeds 0 and 2 take *zero*
fills across all five held-out sessions. Their Sharpe is undefined and their
return is exactly 0.0000 — which, on a held-out set where the baseline returns
+0.0001, is very nearly the best score in the table. This is the degeneracy C2
predicted, and it is why the fills column is printed before the Sharpe column
and why the summariser prints "no trades" rather than `nan`.

### The same policies on the days they trained on

Running the identical deterministic policies over the five *training*
sessions, scored the same way:

| configuration | seed | SR (train) | SR (held out) | vs baseline, train | vs baseline, held out |
| --- | --- | --- | --- | --- | --- |
| **C1 baseline** | — | **−32.34** | **+0.37** | — | — |
| `ddpg-forced` | 2 | **+6.88** (4/5 days +) | −3.61 | **+39.2** | −4.0 |
| `ddpg-forced` | 0 | −1.75 | −20.79 | **+30.6** | −21.2 |
| `td3-forced` | 0 | −9.72 | −4.21 | **+22.6** | −4.6 |
| `td3-forced` | 2 | −30.06 | −0.80 | **+2.3** | −1.2 |
| `td3-forced` | 1 | −38.58 | −16.95 | −6.2 | −17.3 |
| `sac-forced` | 0, 1, 2 | −48.6, −56.9, −59.5 | −27.2 to −30.4 | −16 to −27 | −28 to −31 |
| `ddpg-forced` | 1 | −58.58 | −25.65 | −26.2 | −26.0 |
| `td3-forced`, γ pinned | 1 | −72.86 | −27.03 | −40.5 | −27.4 |
| `td3-forced`, γ pinned | 0, 2 | *no trades* | *no trades* | — | — |

**Four of the fourteen forced runs beat the calibrated baseline on the days
they trained on. None of them beats it on the days they did not.** DDPG seed 2
is the sharpest case: +6.88 Sharpe on training sessions with four of five
profitable — a 39-point improvement on the baseline — and −3.61 on held-out
sessions, four points *below* it.

A caution about how this is measured. Raw returns are higher for everything on
held-out days, because those days were quieter: C1 measured their median σ at
31.2 bp/√s against the training set's 35.2. Comparing a policy's held-out
return against its training return would therefore conflate overfitting with
an easier regime, and would make several of these policies look like they
generalise. The comparison that survives is against **the baseline on the same
days**, which is what the table's last two columns and panel (b) show: the
regime cancels, and what is left is that the edge exists only where it was
fitted.

### Item 3 — the ablation

The ladder's prior is that learning fewer coefficients should be more stable:
a smaller hypothesis class is harder to overfit. **The measurement says the
opposite.**

| configuration | held-out SR by seed | fills/day by seed |
| --- | --- | --- |
| `td3-forced`, all four | −4.21, −16.95, −0.80 | 1,526 / 3,670 / 8,016 |
| `td3-forced`, γ pinned | *no trades*, −27.03, *no trades* | 0 / 79,332 / 0 |

Pinning γ at its calibrated 0.05 produced the two most extreme outcomes in the
whole phase — two seeds that never quote and one that quotes 79,332 times a
session — where the same algorithm with γ free stayed within a factor of five
on fills and a factor of twenty on Sharpe.

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

**C4 inherits a sharper question than "does PPO beat TD3".** The phase it
bridges to asks whether GRPO's group-relative baseline survives outside
language modelling, and whether RL-tuned coefficients overfit to the training
days' microstructure. C3 has already answered the second in the affirmative
for three off-policy algorithms, with the specific signature to look for:
an edge of 20–40 Sharpe points on training sessions and none at all on
held-out ones. C4's reward-hacking item should therefore be run as a
*confirmation with a known positive*, not as an open search — if its
diagnostics do not flag DDPG seed 2, they are not working.

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
