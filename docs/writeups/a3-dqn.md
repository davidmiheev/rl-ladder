# A3 — Value-Function Approximation: task, goal, and how it was solved

What phase A3 asks, why the table has to go, and what the ablations showed. The
lab notebook with the numbers is
[`experiments/a3_dqn/README.md`](../../experiments/a3_dqn/README.md).

---

## 1. The task

[`ladder.md`](../ladder.md#a3) states four items:

1. DQN on CartPole/LunarLander; disable the replay buffer and target network
   independently, document the instability.
2. Add Double DQN; measure whether Q-value overestimation drops.
3. Add a dueling architecture; compare sample efficiency vs. vanilla DQN.
4. Optional: DQN on a small Atari environment.

Item 4 is skipped for now — it is the only piece that genuinely wants a GPU, and
it adds no concept the other three do not already cover.

## 2. The goal

A2 could store one number per state. That ends here: CartPole's state is four
real numbers, so the table would need an entry per point in $\mathbb{R}^4$. The
fix is to *approximate* $Q$ with a parameterised function $Q_\theta(s,a)$ and
learn $\theta$ by regression on TD targets:

$$\mathcal{L}(\theta) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q_{\theta^-}(s',a') - Q_\theta(s,a)\right)^2\right]$$

That single change breaks every guarantee A2 relied on. Value iteration
converged because the Bellman operator is a contraction *in the space of exact
value functions*; compose it with a function approximator's projection and the
composition need not contract at all. The result is the **deadly triad**:

| ingredient | where it appears here |
| --- | --- |
| bootstrapping | the target contains $Q$ itself |
| off-policy learning | the $\max$ evaluates a policy the agent is not following |
| function approximation | one gradient step changes $Q$ at states it never visited |

Any two are usually fine. All three together can diverge. **The goal of A3 is to
find out which stabiliser addresses which ingredient** — which is exactly why
item 1 removes them one at a time rather than reporting that DQN works.

## 3. Why the two stabilisers exist

**Replay** attacks correlation and distribution shift. Consecutive transitions
from one episode are heavily dependent, and gradient descent on a stream of
correlated samples is not the i.i.d. setting SGD assumes. Worse, without a
buffer the training distribution moves with the policy, so the network
continually overwrites what it learned on states it no longer visits. Sampling
uniformly from a buffer of the last $N$ transitions restores something closer to
i.i.d. data over a slowly-changing distribution.

**The target network** attacks the moving target. In the loss above, $\theta^-$
is a periodically-frozen copy. Without it ($\theta^- = \theta$) the regression
target moves the moment the parameters do, and because the approximator
generalises, updating $Q(s,a)$ also shifts $Q(s',a')$ in the target — the
network chases a value it is itself pushing away.

Note these two are answers to *different* ingredients: replay to the data
distribution, the target network to bootstrapping. That is the prediction item 1
tests.

## 4. Double DQN (item 2)

The max operator is biased upward. For any noisy estimates,

$$\mathbb{E}\left[\max_a \hat{Q}(s,a)\right] \ge \max_a \mathbb{E}\left[\hat{Q}(s,a)\right]$$

by Jensen's inequality, since $\max$ is convex. Selecting *and* evaluating with
the same noisy estimates means any action whose value is overestimated by noise
is the one the max picks — the error is selected for, not averaged out.

Double DQN splits the two roles:

$$
\text{DQN:} \quad r + \gamma\, Q_{\theta^-}\!\left(s', \arg\max_{a'} Q_{\theta^-}(s',a')\right)
$$
$$
\text{Double:} \quad r + \gamma\, Q_{\theta^-}\!\left(s', \arg\max_{a'} Q_{\theta}(s',a')\right)
$$

The online network chooses, the target network scores. Two independent estimates
are unlikely to have correlated noise on the same action, so the selected action
is no longer systematically the over-valued one.

**Measuring the bias.** The probe is the difference between what the network
believes at an episode's start state and what the greedy policy actually
collects from it:

$$\text{overestimation} = \max_a Q_\theta(s_0, a) - \mathbb{E}\left[\textstyle\sum_t \gamma^t r_t\right]$$

**A design decision worth stating:** deterministic CartPole supplies almost no
estimate noise for the max to amplify, so the effect may be small there by
construction. The comparison is therefore run twice — once as-is, and once with
zero-mean Gaussian reward noise, which injects the variance the bias feeds on
without changing the optimal policy. If Double DQN only helps in the noisy
variant, that is a result about *when* the fix matters, not a failed experiment.

## 5. Dueling (item 3)

The dueling head splits $Q$ into a state value and an advantage:

$$Q(s,a) = V(s) + \left(A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a')\right)$$

Subtracting the mean advantage is not cosmetic — without it the decomposition is
unidentifiable, since any constant can move between $V$ and $A$ leaving $Q$
unchanged, and the two streams drift.

The argument for it: in most states the choice of action barely matters, and the
useful signal is "is this state good at all". A shared $V(s)$ stream is updated
by *every* transition through $s$ regardless of the action taken, whereas a plain
head only updates the column of the action actually tried. So it should learn
more per sample — a sample-efficiency claim, which is why the item asks for
efficiency rather than final score.

## 6. What was done

`dqn.py` holds one agent with four switches — `replay`, `target_network`,
`double`, `dueling` — so every arm in every item is the same code path with
different flags, and no comparison can be contaminated by an incidental
implementation difference.

Decisions worth recording:

- **Truncation is not termination.** CartPole ends at 500 steps by time limit.
  Bootstrapping is suppressed only on true termination; treating a time-limit
  truncation as terminal would teach the agent that the world stops at step 500
  and cap every value estimate. Gymnasium returns these as separate flags and
  this is the most common place to get it wrong.
- **The no-replay ablation trains on the single latest transition**, not on a
  batch of one sampled from a buffer — the whole point is the loss of both
  decorrelation and the ability to revisit old data.
- **The no-target ablation bootstraps from the online network** rather than
  updating the target every step, which is the same thing expressed without a
  redundant copy.
- **Runs are parallelised across processes**, with `torch.set_num_threads(1)`
  inside each, since four single-threaded runs beat one four-threaded run on
  networks this small.
- Evaluation is **greedy** ($\varepsilon = 0$) and separate from the training
  environment, so exploration cost does not contaminate the score — the A2
  lesson about online return versus policy quality.

## 7. Results

Filled in from the runs; see the phase README for the tables.

## 8. Where these items come from

| item | primary source | why it was picked |
| --- | --- | --- |
| 1 — DQN, ablating replay and the target network | [Mnih et al. 2013](https://arxiv.org/abs/1312.5602) (the NIPS workshop paper: replay, **no** target network) and [Mnih et al. 2015](https://www.nature.com/articles/nature14236) (Nature: adds the target network). The triad itself is S&B §11.3 "The Deadly Triad" | The two papers *are* the ablation. The "no target net" arm is not a strawman someone invented — it is literally the 2013 configuration, and the gap between the arms is the gap between the two publications. |
| 2 — Double DQN | [van Hasselt, Guez & Silver 2016](https://arxiv.org/abs/1509.06461); the tabular original is van Hasselt 2010, and S&B §6.7 "Maximization Bias and Double Learning" gives the bandit-sized version of the same bias | It isolates one specific defect — the max operator's upward bias — with a fix small enough that nothing else changes. The measurement, predicted value against realised return, is the paper's own diagnostic. |
| 3 — Dueling architecture | [Wang et al. 2016](https://arxiv.org/abs/1511.06581) | It is the one improvement in this group that changes the *network*, not the update rule, so it tests whether architecture can buy sample efficiency at a fixed algorithm. |
| 4 — Atari (optional, skipped) | [Mnih et al. 2015](https://www.nature.com/articles/nature14236) | Skipped deliberately: it is a scale demonstration, not a new concept, and the only item here that needs a GPU. |

The **lab** pairing (`Practical_RL`
[week04_approx_rl](https://github.com/yandexdataschool/Practical_RL/tree/master/week04_approx_rl))
is the ladder's own and covers experience replay, target networks and
double/dueling/bootstrap DQN — matching items 1–3.

The **goal statement** in section 2 is quoted from
[`ladder.md`](../ladder.md#a3). The mapping in section 3 from each stabiliser to
the triad ingredient it addresses is this repo's framing of the papers'
arguments, and item 1 is what tests it — not a claim either paper states in
those terms.

## 9. What this sets up

- **A4** abandons the value-only approach: policy gradients parameterise
  $\pi_\theta$ directly, which removes the max-operator bias measured in item 2
  entirely, at the cost of the variance A1 measured.
- **A5** needs continuous actions, where $\max_a Q(s,a)$ is no longer a lookup
  over a handful of columns but an optimisation problem — the reason DDPG has an
  actor at all.
- **B1** meets the replay question again as the on-policy/off-policy tradeoff in
  PPO: how much can you reuse a batch collected under an older policy before the
  updates stop being valid.
