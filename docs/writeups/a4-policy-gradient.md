# A4 — Policy Gradient & Actor-Critic: task, goal, and how it was solved

What phase A4 asks, why parameterising the policy directly is a different kind
of answer from everything before it, and what the runs showed. The lab notebook
with the numbers is
[`experiments/a4_policy_gradient/README.md`](../../experiments/a4_policy_gradient/README.md).

---

## 1. The task

[`ladder.md`](../ladder.md#a4) states three items:

1. REINFORCE on CartPole; compare variance and efficiency against the A3 DQN run.
2. Add a learned baseline and GAE; sweep $\lambda$ and observe the bias/variance
   tradeoff.
3. A2C on LunarLander; compare stability against REINFORCE-with-baseline.

## 2. The goal

Everything up to here learned a *value* and recovered a policy from it by taking
an argmax. This phase parameterises the policy itself, $\pi_\theta(a \mid s)$,
and improves it by gradient ascent on expected return.

The **policy-gradient theorem** says that gradient is computable from samples
without differentiating through the environment:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t\right]$$

Read it as an instruction: push up the log-probability of an action in
proportion to the return that followed it. Nothing about the environment's
dynamics appears, which is why it works on a black box.

Three things change relative to A3, and it is worth being precise about which
are gains:

- **The max operator is gone.** A3 item 2 measured the bias it induces; here the
  policy is improved directly, so there is nothing to overestimate.
- **The action space stops mattering.** $\arg\max_a Q(s,a)$ needs a finite
  enumerable set; $\pi_\theta$ does not. This is what makes A5 (continuous
  control) and all of Part B possible.
- **The policy is stochastic by construction**, so exploration is built into the
  object being learned rather than bolted on as $\varepsilon$.

The cost is **variance**. $G_t$ is a single sampled return, and the estimator
credits every action in an episode with all of it — including the reward that
arrived before the action was taken and had nothing to do with it. Items 2 and 3
are both about reducing that variance without introducing so much bias that the
gradient points somewhere else.

## 3. Why a baseline is free (item 2)

Subtract any function of state from the return:

$$\nabla_\theta J(\theta) = \mathbb{E}\!\left[\nabla_\theta \log \pi_\theta(a \mid s)\,\big(G_t - b(s_t)\big)\right]$$

This leaves the gradient unchanged, because for any $b$ depending only on state,

$$\mathbb{E}_{a \sim \pi}\!\left[\nabla_\theta \log \pi_\theta(a \mid s)\, b(s)\right]
= b(s) \nabla_\theta \sum_a \pi_\theta(a \mid s) = b(s)\, \nabla_\theta 1 = 0$$

So the baseline is *free* in expectation and can only change the variance. The
natural choice is $b(s) = V(s)$, which turns the return into an advantage
$A(s,a) = Q(s,a) - V(s)$: how much better this action was than the state's
average, rather than how good the state happened to be.

**GAE** then interpolates between two ways of estimating that advantage. With
$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$,

$$\hat{A}^{\text{GAE}(\lambda)}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l\, \delta_{t+l}$$

At $\lambda = 0$ this is the single residual $\delta_t$: minimum variance, but
biased by whatever error $V$ carries. At $\lambda = 1$ the sum telescopes into
$G_t - V(s_t)$: the Monte Carlo advantage, unbiased but carrying the full
variance of the sampled return. The claim to test is that neither endpoint is
best.

## 4. Why A2C needs parallel environments (item 3)

A3 used a replay buffer to decorrelate its training batches. Policy gradients
cannot: the estimator above is an expectation **under the current policy**, so
data from an older policy is simply the wrong distribution. Replay is
unavailable by construction.

A2C's answer is to decorrelate *across environments instead of across time* —
step $N$ independent copies simultaneously and take one transition from each.
The batch is decorrelated, and every sample is on-policy. This is why on-policy
methods are described as needing parallelism where off-policy methods need
memory, and it is the same tradeoff that reappears in B1: PPO's clipping exists
to squeeze a few extra gradient steps out of each batch precisely because a
fresh batch is expensive.

## 5. What was done

`pg.py` implements the family as one estimator with flags — `baseline=False` is
REINFORCE, `baseline=True` adds a critic, `gae_lambda` selects the advantage
estimator — so the arms differ only where the phase says they differ.

Decisions worth recording:

- **Item 1 reuses A3's exact evaluation protocol**: CartPole-v1, 30k environment
  steps, greedy evaluation every 2000, five seeds. "Comparable" should mean
  numerically comparable, not comparable in spirit. A3's reference numbers are
  recorded as constants in the script so the comparison does not depend on a
  gitignored run directory.
- **The critic gets its own optimizer and many gradient steps per batch.** With a
  single shared update the value function never fits — the first attempt here
  reached an explained variance of $0.0$, meaning the "baseline" was subtracting
  noise and item 2 would have measured nothing. A policy gradient takes one
  on-policy step per batch; a critic is a supervised regression and needs more.
- **Advantage spread is recorded before normalisation.** Normalising advantages
  is standard and helps optimisation, but it rescales exactly the quantity
  $\lambda$ controls. Measuring after normalisation would erase the effect.
- **Truncation is bootstrapped, termination is not** — the same distinction as
  A3, and equally easy to get wrong here.
- **Greedy evaluation takes the mode of the policy**, so the evaluation number
  is not inflated by exploration and is comparable with A3's.

## 6. Results

Filled in from the runs; see the phase README.

## 7. What this sets up

- **A5** needs the action space to be continuous, which the argmax in A3 could
  not handle and the policy parameterisation here can. DDPG's actor is exactly
  the "learn the maximising action" idea made explicit.
- **B1** is this phase at scale: PPO is a policy gradient with a trust region,
  and its KL term is a baseline-like correction that keeps the estimator valid
  against a reference distribution rather than against a value function.
- **B3's GRPO** replaces the learned $V(s)$ derived here with the mean score of a
  group of completions — a Monte Carlo baseline instead of a fitted one, which
  is why the group-size sweep is a variance measurement.

## 8. Where these items come from

| item | primary source | why it was picked |
| --- | --- | --- |
| 1 — REINFORCE | [Williams 1992](https://doi.org/10.1007/BF00992696), the paper that named it; S&B §13.3 | It is the policy-gradient theorem with nothing added, so it isolates the estimator before any variance reduction obscures it. Comparing against A3 on the same task is what makes the "high variance" claim a measurement rather than folklore. |
| 2 — baseline and GAE | S&B §13.4 for the baseline; [Schulman et al. 2016](https://arxiv.org/abs/1506.02438) for GAE | The baseline is the one variance reduction that is provably free, and GAE is the standard knob that spans from TD to Monte Carlo — one sweep covers the whole bias/variance axis instead of two separate algorithms. |
| 3 — A2C | [Mnih et al. 2016](https://arxiv.org/abs/1602.01783), which introduced A3C; A2C is its synchronous variant | It is the smallest change that makes policy gradients practical on a harder task, and the mechanism — parallel environments in place of replay — is the on-policy/off-policy distinction from A2 showing up as a systems decision. |

The **lab** pairing (`Practical_RL`
[week06_policy_based](https://github.com/yandexdataschool/Practical_RL/tree/master/week06_policy_based))
is the ladder's own and covers REINFORCE and advantage actor-critic, matching
items 1 and 3.

The **goal statement** in section 2 is quoted from
[`ladder.md`](../ladder.md#a4). Section 4's framing of parallel environments as
"the on-policy substitute for replay" is this repo's, not a claim from the A3C
paper.
