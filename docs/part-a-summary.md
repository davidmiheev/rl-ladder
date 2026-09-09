# Part A — what eight phases actually taught

Part A is complete: **26 of 26 items**, A1 through A8. This is not a recap of
each phase (the [writeups](writeups/) do that) but the things that turned out to
be true across all of them — including several that only became visible by
getting them wrong first.

| phase | items | the finding that mattered most |
| --- | --- | --- |
| [A1](writeups/a1-bandits.md) bandits | 3/3 | constant $\alpha$ provably *fails* Robbins–Monro — and that failure is why it can track |
| [A2](writeups/a2-tabular.md) tabular | 4/4 | Q-learning learns the better policy and earns the worse return, in the same run |
| [A3](writeups/a3-dqn.md) DQN | 4/4 | removing the target network is not degradation but collapse to a constant action |
| [A4](writeups/a4-policy-gradient.md) policy gradients | 3/3 | λ=0 scores 20.8 *despite* having the lowest variance of any arm |
| [A5](writeups/a5-continuous-control.md) continuous control | 4/4 | of TD3's three fixes, twin critics do the work and the delayed actor does none |
| [A6](writeups/a6-mcts.md) MCTS/AlphaZero | 3/3 | the learned prior beats the learned value 8:1 at small search budgets |
| [A7](writeups/a7-multi-agent.md) multi-agent | 2/2 | a policy optimal in 97.2% of positions loses *every* game to a chosen line |
| [A8](writeups/a8-imitation.md) imitation & IRL | 3/3 | recovered reward correlates only +0.556 with the true one, yet plans near-optimally |

---

## 1. Every headline metric saturated or misled at least once

This is the strongest pattern in Part A, and it cost real time before it became
a habit.

| phase | the misleading metric | what it hid | the fix |
| --- | --- | --- | --- |
| A3 item 2 | mean overestimation | one seed at $+841$ carrying the mean | median + every seed listed |
| A3 item 3 | median steps-to-threshold | dropped seeds that never got there, flattering the arm that failed more | treat non-reaching seeds as censored |
| A4 item 3 | equal *environment steps* | a 100× difference in gradient updates taken from them | report both, add a matched-update arm |
| A5 items 1–2 | final return | all five arms within 5 points; the arms differed in value bias | report the bias |
| A7 | exploitability | saturates at 1.00 — one mistake anywhere scores the same as total failure | add graded regret and optimal-move rate |
| A8 | success rate | a random walk still reaches the goal, just 9× slower | report return |

The general rule that emerged: **a metric with a ceiling or a floor cannot show
you an effect near it**, and a single aggregate over a heavy-tailed quantity is
not a summary, it is a hiding place. Two habits fixed all six cases — report a
second metric that cannot saturate where the first does, and list per-seed
values whenever the spread might exceed the effect.

A related one, from A6: **report a quantity whose correct value you know in
advance.** Perfect play in Tic-Tac-Toe is a draw, so an agent facing a perfect
opponent can never win. Printing the win rate cost nothing and immediately
exposed a sign bug that had been reporting 35% wins against an opponent that
cannot lose.

## 2. The environment was repeatedly too easy to show the phenomenon

Three phases produced null results that were facts about the testbed, not the
algorithm:

- **A3.** CartPole has two actions, so the max-operator bias $\sigma\sqrt{2\ln|\mathcal{A}|}$
  had almost nothing to bite on and dueling's shared value stream had nothing to
  disentangle. Both items came out inconclusive.
- **A5.** Pendulum is forgiving enough that DDPG's critic could overestimate by
  $+26.3$ without that ever becoming a policy failure — its *return* stability
  was the best of five arms.
- **A3 again, resolved.** Running Pong on a GPU produced exactly the separation
  CartPole could not: value bias positive at 10 of 10 late checkpoints for DQN,
  negative at 8 of 10 for Double DQN.

The correction is to pick the testbed from the phenomenon rather than from
convenience — and, when compute forbids that, to say the result is about the
testbed rather than the method. Both A3 and A5 now say so explicitly.

## 3. Ablations attribute; aggregate comparisons do not

Four phases ran component ablations, and **three of them overturned the obvious
reading**:

- **A3**: replay and the target network are not interchangeable. Removing replay
  costs a lot (80 vs 250); removing the target network collapses the agent to a
  constant action, scoring 9.2 where *random* scores 22.7. Inefficiency
  degrades; a missing fixed point diverges.
- **A5**: of TD3's three fixes, twin critics move the value bias by $+17.4$ and
  are the only one whose removal flips its sign, while the delayed actor update
  moves it by $1.4$ — nothing. A fix can be worth having and still not be the
  one doing the work on a given metric.
- **A6**: the item asks to ablate the *value* network, which presumes the value
  is the interesting half. The full 2×2 says the policy prior contributes
  $+0.162$ and the value head $+0.019$.
- **A7**: the random-starts arm was added purely as a control, and it showed that
  much of self-play's advantage is state-action coverage (0.684 vs 0.32–0.36 —
  a shared table gets updates from both seats) rather than the curriculum effect
  usually credited.

## 4. Where ground truth was available, it was worth the detour

Four phases used environments small enough to solve exactly, and every one of
them paid for it:

- **A2**: value iteration checked against BFS shortest paths — an independent
  computation that never touches the Bellman equation.
- **A6**: the exact 5478-position minimax solution turned "converges to
  near-optimal play" into a measurement, and its impossible-win invariant caught
  a bug.
- **A7**: exact exploitability, which is the only honest measure in a two-player
  game — a win rate says the two policies are mismatched, not that either is
  good.
- **A8**: the exact optimal policy, so BC and IRL are both scored absolutely
  rather than against each other.

## 5. Four technical threads run the length of Part A

**The max operator's optimism.** $\mathbb{E}[\max] \ge \max \mathbb{E}$ appears
in A3 as DQN's overestimation, in A5 amplified because the actor is *trained* to
find the maximiser (gradient ascent is a better maximiser than an argmax over
columns), and in A7 in disguise: zero-initialised $Q$ is optimistic in a
zero-sum game where nearly every true value is $\le 0$, so a greedy policy walks
into untried actions — which is exactly where a best-response opponent steers.

**Distribution shift.** A3's replay buffer holds transitions the policy has
outgrown; A4 cannot use replay at all, because a policy gradient is an
expectation under the *current* policy, which is why A2C parallelises
environments instead; A8's behavioural cloning fails for the same reason with
the roles reversed, and Ross & Bagnell's $O(\varepsilon T^2)$ bound names the
compounding. B5's on-policy distillation is the same fix again.

**Identifiability.** A6's dueling head needs mean-subtraction because $Q = V + A$
has a gauge freedom nothing in the loss constrains. A8's recovered reward
correlates only $+0.556$ with the truth while planning near-optimally, because
potential shaping leaves the optimal policy unchanged. Same shape, and the
second one is the caveat that carries into every reward model in Part B.

**Bias against variance.** A1's step-size choice, A2's $\lambda$-free TD, A4's
GAE sweep where $\lambda = 0$ has the lowest variance *and* the worst
performance, A5's twin critics buying underestimation to avoid overestimation.
The recurring lesson is that minimum variance is worthless when the estimand is
wrong.

## 6. What Part A hands to Part B

- **A4's policy gradient** is PPO's estimator; the clipping exists to reuse a
  batch that A4 could only use once.
- **A3's overestimation probe** becomes B4's proxy-versus-gold plot; the shape
  of the question is identical.
- **A6's search-as-policy-improvement** is the contrast for B3's GRPO, which
  averages *independent* rollouts because language has no cheap simulator to
  build a tree over.
- **A8's identifiability caveat** is the one to carry hardest: a reward model
  identifies an ordering on the distribution its preferences covered, and
  Part B spends most of its time optimising far outside that distribution.
- **A7's exploitability lesson** — average-case competence and worst-case
  exploitability are different quantities — is what B4's reward hacking is,
  with the optimiser playing the exploiter.

## 7. Honest limits of Part A as run

- **Seed counts.** Most phases ran 5 seeds; A3's Atari items ran **one**, below
  this repo's own floor, and produced two contradictory orderings at the same
  nominal seed. That is recorded as an illustration rather than a measurement.
- **Compute ceilings shaped conclusions.** A3's Atari runs are 1M steps against
  the papers' 50M; A6's AlphaZero is 900 self-play games with a two-layer net,
  so its "worth 2× the search budget" is a lower bound.
- **Predictions that failed are kept.** A5 predicted SAC would degrade least
  under evaluation noise; it degraded most, and the phase records both that and
  the design flaw behind it — the entropy argument concerns *action*
  perturbations, and the experiment perturbed observations.
- **Untested predictions are labelled as such.** A6 predicts the value head's
  contribution grows with simulation budget and the prior's shrinks; that was
  not run.
