# C2 — Framing Market Making as an RL Problem: task, goal, and how it was solved

The environment every later Part C result passes through, and the four checks
that make it trustworthy. Lab notebook:
[`experiments/c2_env/README.md`](../../experiments/c2_env/README.md).

---

## 1. The task

[`ladder.md`](../ladder.md#c2) states four items:

1. Define state: inventory $q$, short-horizon volatility estimate, order-book
   imbalance / micro-price features, basis & APT alpha signals.
2. Define the action space as the GLFT coefficient set itself — $\gamma$, $c_1$,
   and a half-spread/skew scaling term — rather than raw quote prices.
3. Define reward as mark-to-market PnL minus an inventory-risk penalty
   $-\eta q^2$.
4. Wrap hftbacktest as a Gym-style environment: one episode per trading
   session, one step per grid-refresh interval.

## 2. The goal: an environment whose failures are not silent

An environment is the one component every later result depends on, and its
characteristic bugs do not raise exceptions. A leaked future, a reward that
does not sum to the PnL it claims to measure, an action space that cannot
express the baseline it is supposed to beat — each of those produces a
*plausible* learning curve. The first two produce a suspiciously good one.

So the deliverable is not "an environment" but "an environment plus the
evidence that it is one", and §5 is four checks against answers known before
the code ran.

The design decision that defines Part C is item 2: **the policy emits
coefficients, not quotes**. Three consequences follow, and C3 and C4 depend on
all of them.

- The fixed-coefficient baseline is a *strict special case* of the policy, so
  "did RL help" has a clean answer rather than a comparison between two
  differently-parameterised strategies.
- Every action is interpretable in the language the model already uses — risk
  aversion, spread floor, spread scale, skew scale — rather than as an opaque
  price.
- The policy cannot emit a nonsense quote. The formula bounds what it can say.

## 3. What C1 forced into the design

C2 would have been mostly a matter of taste if C1 had not measured what it did.
Four choices below are consequences, not preferences.

**The action space must contain "do not quote."** Every parameter sweep in C1 —
$\gamma$, `adj1`, the quote percentile — was monotone in how little the
strategy traded. A policy that cannot decline to quote is being asked to choose
the least bad way to lose. The half-spread scale therefore reaches $8\times$,
which on these assets puts both quotes far outside anything that trades: the
probe in §6 confirms the corner is reachable and produces exactly zero fills.

**The state must carry order flow.** C1's entire loss is adverse selection, and
the one thing found to predict it is aggressive trade flow — a single at-touch
trade calls the next 100 ms with a $1.9\sigma$ mean. A state of (inventory,
volatility, spread) is blind to that. This one carries signed flow over one
second and ten.

**The reward must be marked to market every step.** Almost none of C1's loss
appears in the execution price; nearly all of it accrues while the position is
held. A reward that credited only closed round trips would give no gradient on
the thing that costs the money.

**The policy can act every second even though the quoter re-posts every
100 ms.** C1's interval study separated execution quality — entirely a function
of reaction speed — from post-fill drift, which is not: the 0.5 s mark is
$-44$ ticks whether the quoter reacts every 100 ms or every 2 ms. Since the
drift dominates, the expensive fast loop buys the small term, and the agent can
act at the interval its coefficients are actually recalibrated at. That is a
$10\times$ saving in episode length and it is a prediction C3 can falsify.

## 4. The formulation

### State

Ten features, all scaled by fixed constants rather than running statistics —
running normalisation leaks state across episodes and makes a run impossible to
reproduce from its seed alone.

| feature | what it is | why |
| --- | --- | --- |
| `inventory` | $q / q_{\max}$ | risk is a function of position; the reward's penalty is measured against the same cap |
| `volatility` | $\sigma$ in bp per $\sqrt{s}$, trailing 10 min | C1: the cost per fill is $-0.72\sigma$ |
| `spread` | touch spread in ticks | says whether there is any room to quote inside |
| `imbalance` | $(q_b - q_a)/(q_b + q_a)$ at the touch | the standard adverse-selection predictor |
| `imbalance_depth` | the same over the best 5 levels a side | what the touch imbalance cannot see |
| `flow_1s`, `flow_10s` | signed aggressive volume, squashed | C1's measured predictor of the loss |
| `log_k` | the fitted intensity decay | the model's own state, so the policy knows what it is scaling |
| `model_half_spread` | where the unmodified formula wants to quote, in bp | ditto |
| `time_of_day` | fraction of session elapsed | liquidity is not stationary within a day |

**The micro-price is not in this list, and that is a measurement, not an
omission.** The ladder's item asks for "order-book imbalance / micro-price
features". At the touch those are the same number. With
$m = (P_b + P_a)/2$ and $\mu = (P_b q_a + P_a q_b)/(q_a + q_b)$,

$$\mu - m = \frac{P_b q_a + P_a q_b}{q_a + q_b} - \frac{P_b + P_a}{2}
= \frac{(P_a - P_b)(q_b - q_a)}{2(q_a + q_b)}
= \frac{\text{spread}}{2}\cdot\text{imbalance},$$

exactly and always. A micro-price feature next to a touch-imbalance feature is
the same number scaled by the spread, which is already its own feature. The
second slot carries depth beyond the touch instead — which is what a
micro-price *would* add if it were computed over more than the best level.

I found this the wrong way round: the micro-price feature was in the
observation first, and it was saturating the observation clip at $\pm 10$. The
saturation turned out to be correct behaviour (wide-spread moments, where
$\mu - m$ legitimately reaches several ticks), and chasing it produced the
identity above. Worth recording because C1 had *just* been corrected for
explaining a suspicious number instead of measuring it.

**Basis and APT alpha are deferred to C5**, which is the phase for them
("Fold the basis/APT alpha signals into the state … and re-run C3/C4"). They
need a second instrument in the feed, which this dataset does not carry for a
single-asset backtest. `FEATURES` is a tuple so C5 extends it rather than
rewrites it.

### How much of the state is actually distinct

Correlations between features over a baseline session, worth printing because
two features that always agree are one feature and a maintenance cost:

| pair | \|r\| |
| --- | --- |
| `imbalance` / `imbalance_depth` | 0.974 |
| `log_k` / `model_half_spread` | 0.969 |
| `volatility` / `model_half_spread` | 0.950 |
| `volatility` / `log_k` | 0.912 |
| everything else | ≤ 0.264 |

Two things to be honest about. **The depth-5 imbalance barely differs from the
touch imbalance** (r = 0.974) on this asset, where the book's size sits at the
touch and the spread is one tick — so replacing the micro-price with it fixed
an exact duplicate and left a near-duplicate. It is kept because it is
genuinely different on the wide-spread assets in C1's cross-asset study
(AVAXUSDC and SUIUSDC sit at one tick only 66% and 78% of the time), and
because dropping a feature to improve a correlation number, with no downstream
metric to judge it by, is the kind of unmotivated tuning this repo's protocol
warns about. C3 is where it earns its place or does not.

**The three model-state features are close to one factor**: `volatility`,
`log_k` and `model_half_spread` are mutually correlated at 0.91–0.97, which is
unsurprising since the half spread is a function of the other two. So the ten
features carry perhaps six independent directions. That is not a defect, but
it is worth knowing before concluding anything from a feature-importance plot
in C3.

### Action

Four continuous actions in $[-1, 1]$, mapped logarithmically so that a
policy's uniform prior is uniform over orders of magnitude:

| action | range | meaning |
| --- | --- | --- |
| $\gamma$ | 0.005 – 1.0 | risk aversion, which also sets $\xi$ |
| $c_1$ scale | 0.1 – 4.0 | weight on the spread floor |
| half-spread scale | 0.125 – 8.0 | overall spread multiplier (C1's `adj1`) |
| skew scale | 0.01 – 2.0 | inventory skew multiplier (C1's `adj2`) |

The quotes are then

$$\text{half spread} = \left(s_{c_1} c_1(\gamma, k) + \tfrac{1}{2}\,\sigma\, c_2(\gamma, A, k)\right) s_{\text{half}},
\qquad
\text{skew} = \sigma\, c_2(\gamma, A, k)\, s_{\text{skew}}.$$

**The ladder names three knobs and this has four.** Its "a half-spread/skew
scaling term" is split in two because C1 showed the coupling between them is
precisely what breaks the fixed model: the half spread and the skew both scale
with $\sigma c_2$, so 0.67 units of inventory cancelled the entire spread, and
the source notebook introduces `adj2` by hand for exactly that reason. Giving
the policy one knob for both would hand it the pathology and no way out. The
baseline remains a strict special case at $(0.05, 1, 1, 0.05)$.

### Reward

$$r_t = \underbrace{\frac{\Delta(\text{balance} + q\,m - \text{fees})_t}{\text{book}}\times 10^4}_{\text{mark-to-market PnL, in bp}} \;-\; \eta\left(\frac{q_t}{q_{\max}}\right)^2$$

$\eta$ is measured, not guessed. A baseline session accumulates
$\sum_t q_t^2 = 1545$ against a PnL term of $-93$ bp, so $\eta = 0.015$ makes
the penalty a quarter of the PnL term over a session.

Calibrating $\eta$ against the PnL term's *per-step standard deviation* — the
obvious thing, and what I did first — is wrong by two orders of magnitude
($\eta = 0.38$). The PnL term is near zero-mean noise whose total grows like
$\sqrt{T}$; the penalty is one-signed and grows like $T$. Only their episode
totals are commensurable.

### Episode

One session per episode. The environment replays one calibration window
(10 minutes, 6,000 quoting iterations) under the baseline coefficients before
the first observation, because the analytic model cannot quote until $A$, $k$
and $\sigma$ exist. No reward is attributed to the warm-up, and the policy is
not asked for decisions it could not inform.

## 5. Checks with known answers

`python experiments/c2_env/run_checks.py`. The first replays a full session
twice and takes a few minutes; the rest take seconds.

| check | how | result |
| --- | --- | --- |
| the baseline is a strict special case | drive the env with the constant baseline action and compare against C1's `glft-grid` backtest | balance differs by $1.5\times10^{-11}$; position, fees and **trade count (2,694) identical** |
| no lookahead | replay a truncated copy of the same session and compare observations step for step | 901 observations, max absolute difference **0** |
| the reward telescopes | sum the PnL term and compare with the change in equity | $-4.949953400$ bp both ways |
| Gymnasium conformance | `gymnasium.utils.env_checker` | passes |

The first is the one the rest of Part C rests on. "The baseline is a special
case of the policy" is the sentence that makes C3's comparison meaningful, and
it is the kind of claim that is easy to state and easy to have quietly broken
by an off-by-one in a re-implemented loop. Reproducing the trade count exactly
— 2,694 fills, not 2,693 — is a much stronger statement than reproducing the
PnL approximately.

The action-to-coefficient round trip is exact only to $8\times10^{-10}$,
because Gymnasium action spaces are `float32` by convention and a policy emits
`float32`. Demanding more would be testing the dtype. The backtest agreeing
exactly *despite* that is the more informative result.

## 6. Does the action space do anything?

An action space the policy cannot move the outcome with produces a flat
learning curve that looks like a hard problem. One session, ETHUSDC
2025-08-26:

| policy | PnL (% of book) | fills | median half spread | max &#124;q&#124; |
| --- | --- | --- | --- | --- |
| widest (half-spread scale at max) | **0.000** | **0** | 605 ticks | 0.00 |
| baseline (C1's `glft-grid`) | −0.93 | 2,694 | 76 ticks | 0.60 |
| uniform random, 3 seeds | −7.26, −9.02, −6.57 | ~41,000 | 116 ticks | 1.8–2.0 |
| tightest (half-spread scale at min) | −33.84 | 221,046 | 10 ticks | 1.90 |

The action space moves a session's PnL over a range of **33.8% of book**, so
there is something to choose between. Two things in that table matter more than
the range.

**Doing nothing is reachable, and on this day it wins.** The widest corner
takes zero fills and returns exactly zero, which beats the baseline's −0.93%.
This is C1's degeneracy showing up in the environment as designed rather than
as a surprise — and it is the single most likely way C3 goes wrong. An agent
maximising this reward on training days has an available policy that scores
zero, and any policy that trades has to beat it.

**Random is much worse than the baseline, not slightly.** A uniformly random
coefficient policy loses 7× what the baseline does. The action space is not a
neighbourhood of good policies with a few bad corners; most of it is bad. That
is worth knowing before choosing an exploration schedule.

## 7. What this sets up

**C3** inherits a benchmark triple rather than a single number: beat the
baseline (−0.93% on this day), beat random, and beat *zero* — the last being
the one that bites. It also inherits a concrete failure mode to instrument
for: a policy that converges to the widest corner has learned to stop trading,
which will look like success on every PnL-based metric. C3 should log fill
counts alongside returns from the first run, not after the first surprise.

**C4** gets a sharper version of its own item. "Check for overfitting" here
means checking three things: the held-out split (fixed in C1 and read from
`config.py`), the fill-count floor above, and whether the learned coefficients
are stable across seeds or merely fitted to a session's volatility path.

**C5** extends `FEATURES`. The basis and APT signals belong there by the
ladder's own structure, and the observation vector is a tuple of named
features so adding them is an append rather than a rewrite.

### Limits worth stating

- **One episode is one session, and there are five training sessions.** That is
  a very small number of episodes for on-policy RL. C3 will need either
  sub-session episodes (which breaks the "one episode per trading session"
  framing the ladder asks for) or many passes over the same five days (which
  invites memorising them). Neither is free and the choice should be explicit.
- **An episode costs about 90 seconds** of CPU, dominated by replaying 60
  million market events, not by the policy. That sets the scale of what C3 can
  afford: roughly 40 episodes an hour on one core.
- **The reward's risk penalty is a modelling choice, not a measurement.**
  $\eta$ is calibrated to a fraction of the baseline's PnL, but which fraction
  is a preference. C3 should report at least one sensitivity.
- **The environment is single-asset.** Multi-asset market making, which is
  where hftbacktest's own examples end up, is out of scope for Part C as the
  ladder defines it.

## 8. Where these items come from

| item | primary source | why it was picked |
| --- | --- | --- |
| 1 — state | the ladder's own list; the imbalance and micro-price features follow hftbacktest's [Market Making with Alpha — Order Book Imbalance](https://github.com/nkaz001/hftbacktest/blob/master/examples/Market%20Making%20with%20Alpha%20-%20Order%20Book%20Imbalance.ipynb) | The state is the only thing that decides what the policy *can* learn, and C1 measured exactly one predictor of the loss worth having in it. The rest of the list is what the analytic model already conditions on, so that the policy can at least reproduce it. |
| 2 — coefficients as actions | the ladder's own, described there as "the central design decision of Part C" | It makes the baseline a strict special case, which turns "did RL help" from a rhetorical question into a checkable one. Every other formulation compares two different strategies and attributes the difference to learning. |
| 3 — reward with $-\eta q^2$ | the ladder's own; the quadratic mirrors the risk term in [Guéant, Lehalle & Fernandez-Tapia](https://huggingface.co/papers/1605.01862) | PnL alone rewards inventory risk that happened not to be punished on the training days. A quadratic penalty makes the learned objective and the analytic model's objective the same shape, so the comparison is between solutions rather than between objectives. |
| 4 — Gym-style wrapper | [Gymnasium's custom environment tutorial](https://gymnasium.farama.org/introduction/create_custom_env/) | The standard interface is what lets C3 use an off-the-shelf PPO rather than write one, and `gymnasium.utils.env_checker` is a free correctness check. |

The four design consequences in §3, the micro-price/imbalance identity in §4,
and the $\eta$ calibration argument are this repo's, derived from C1's
measurements rather than taken from a source.
