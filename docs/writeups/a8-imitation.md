# A8 — Imitation Learning & Inverse RL: task, goal, and how it was solved

Why copying an expert's actions and recovering an expert's objective are
different things, and what the difference costs. Lab notebook:
[`experiments/a8_imitation/README.md`](../../experiments/a8_imitation/README.md).

---

## 1. The task

[`ladder.md`](../ladder.md#a8) states three items:

1. Behavioural cloning from a handful of demonstrations; observe compounding
   error away from the expert's state distribution.
2. A simple inverse RL method (max-entropy IRL) to recover a reward, then run
   RL against it.
3. Compare the two on states the demonstrations never visited.

## 2. The goal: the reward was always given, until now

Every phase so far received a reward function. Here it is withheld, and the
only signal is a set of expert trajectories. Two things can be done with them,
and the gap between them is the phase.

**Behavioural cloning** is supervised learning: fit
$\pi_\theta(a \mid s)$ to the demonstration pairs. It is the obvious approach
and it has a specific, well-understood failure.

**Inverse RL** infers a reward $r$ such that the demonstrations look optimal
under it, then plans. The extra step buys generalisation, because a reward plus
a model *constructs* behaviour at states never seen, while a cloned policy can
only look one up.

### Why BC fails away from the data

BC's training and deployment distributions differ, and the difference is caused
by BC itself. Training samples come from $d^{\pi^*}$, the expert's state
distribution; at deployment the agent visits $d^{\pi_\theta}$, its own. The
moment it errs it moves somewhere the expert never was, where it has no training
signal, so it errs again.

The standard result (Ross & Bagnell 2010) makes the compounding precise. If the
cloned policy errs with probability at most $\varepsilon$ under the expert's own
distribution,

$$\mathbb{E}_{s \sim d^{\pi^*}}\left[\mathbb{1}\{\pi_\theta(s) \ne \pi^*(s)\}\right] \le \varepsilon
\quad\Longrightarrow\quad
J(\pi^*) - J(\pi_\theta) \le O\!\left(\varepsilon T^2\right)$$

over a horizon $T$ — quadratic, not the $O(\varepsilon T)$ that supervised
intuition suggests. The extra factor of $T$ is the
distribution shift — an error at step $t$ does not just cost one step's reward,
it changes every state that follows. This is why DAgger and friends exist, and
why item 1 asks to *observe* the compounding rather than take it on faith.

### What MaxEnt IRL actually models

Early IRL matched feature expectations, which is under-determined — many
rewards produce the same expectations. Ziebart's maximum-entropy formulation
resolves the ambiguity by choosing the distribution over trajectories with
highest entropy subject to matching them, which turns out to be

$$P(\tau) \propto \exp\left(\sum_{t} r_\theta(s_t)\right)$$

Trajectories are exponentially preferred by return rather than strictly optimal,
which has a practical consequence worth stating: **suboptimal demonstrations
become information rather than contradictions.** Under a hard-optimality model, a
single mistake by the expert makes the data inconsistent; here it is merely
lower-probability.

The gradient of the log-likelihood is a difference of expected feature counts,

$$\nabla_\theta \mathcal{L} = \boldsymbol{\phi}_{\text{expert}} - \sum_s D_\theta(s)\,\boldsymbol{\phi}(s)$$

where $D_\theta$ is the expected state-visitation frequency under the current
reward's soft-optimal policy. Computing $D_\theta$ needs a forward pass over a
discrete state space, which is why this phase uses a gridworld rather than the
CartPole the ladder suggests — and why item 3's "states never visited" is an
exactly enumerable set here instead of a fuzzy region.

## 3. The identifiability problem

A reward is not determined by behaviour. Ng, Harada and Russell (1999) showed
that potential-based shaping

$$r'(s,a,s') = r(s,a,s') + \gamma\Phi(s') - \Phi(s)$$

leaves the optimal policy unchanged for *any* $\Phi$, because the shaping term
telescopes along every trajectory:

$$\sum_{t=0}^{T-1} \gamma^t\left[\gamma\Phi(s_{t+1}) - \Phi(s_t)\right] = \gamma^T \Phi(s_T) - \Phi(s_0)$$

which depends only on the endpoints and so shifts every trajectory's return by
the same amount, leaving the ordering — and hence the argmax — untouched. An
infinite family of rewards therefore explains any set of demonstrations equally
well, and IRL can only ever
return a member of that class. Scaling is free too, and a constant offset
changes nothing.

This matters beyond the phase. A learned reward model in Part B is an IRL-like
object fitted to human preferences, and its *numbers* are routinely read as
though they measured something absolute — "the reward went up" — when what is
identified is at best an ordering, and only on the distribution the preferences
covered. Section 5's measurement puts a figure on the gap.

## 4. What was done

An 8×8 gridworld with walls and 10% slip, a solvable MDP so every policy can be
scored against the exact optimum rather than against the other learner. Six
demonstrations from a single start cover 49.5% of free states, leaving 21 never
visited.

Three learners, deliberately including a strong form of BC:

- **tabular BC** — counts per state, uniform where unseen. Not a strawman: with
  no data at a state there is nothing to imitate, and a uniform policy is the
  honest representation of that.
- **feature-based BC** — logistic regression on coordinates and local wall
  structure, so that BC is also represented by a version that *does* answer
  everywhere.
- **MaxEnt IRL then value iteration** on the recovered reward.

Measurements: action agreement with the expert, split by whether a state
appeared in the demonstrations; return from demonstrated, all, and
never-demonstrated starts; the fraction of a rollout spent off the demonstrated
states (item 1's compounding, made numeric); and the correlation between the
recovered and true rewards.

## 5. Results

Tables in the [phase README](../../experiments/a8_imitation/README.md).

**BC is exactly as good as its data and no better.** Tabular BC agrees with the
expert on 100% of demonstrated states and returns $-12.3$ from the demonstrated
start, indistinguishable from the expert's $-11.7$. From never-demonstrated
states it returns $-70.6$ against the expert's $-8.1$.

**Generalising is not generalising correctly.** The feature-based BC answers
everywhere and is worse than the tabular version *at every measurement*,
including $-70.1$ from the demonstrated start where tabular BC is optimal. A
smooth function fitted to a policy that depends on local wall structure produces
confident wrong answers in place of absent ones, and confident wrong answers
steer: its off-distribution fraction is $0.283$ against the expert's $0.012$.
That number is item 1's compounding error made visible — drift into unfamiliar
states, then further drift.

**Recovering the objective transfers.** IRL-then-RL matches the expert
essentially everywhere ($-11.5 / -4.7 / -8.2$ against $-11.7 / -4.7 / -8.1$) and
agrees with it on $90.6\%$ of the 21 never-demonstrated states. It saw those
states no more often than BC did. The difference is that a reward and a model
can *construct* an action there.

**And the recovered reward is not the true one.** Correlation $+0.556$, while
the policy planned against it is near-optimal — section 3's identifiability
result, measured. What IRL returns is a member of an equivalence class: fine for
planning, unreliable as a description of what the expert wanted.

**A measurement note, the third of its kind in Part A.** Success rate reported
$0.869$ for tabular BC from unseen starts, which reads as respectable; return
reported $-70.6$ against $-8.1$. Both are correct — a random walk in a small
connected gridworld does reach the goal within the step cap, it merely takes
forever — but a metric that asks only *whether* the goal was reached cannot see
it. Same shape as A3's mean-vs-median and A7's saturated exploitability.

## 6. What this sets up

- **B2's preference learning** is IRL's cousin: fit a scalar to human choices,
  then optimise it. The identifiability caveat transfers directly — a reward
  model identifies an ordering on the distribution its preferences covered, and
  little else.
- **B4's overoptimisation** is what happens when the equivalence-class member
  you recovered is optimised far past the region where it was fitted. The gap
  measured here as a correlation of $+0.556$ is the same gap that opens between
  proxy and gold metric there.
- **The distribution-shift argument reappears in B5**: offline distillation
  trains a student on the teacher's distribution and deploys it on its own,
  which is behavioural cloning's failure with a different name, and on-policy
  distillation is the DAgger-shaped fix.

## 7. Where these items come from

| item | primary source | why it was picked |
| --- | --- | --- |
| 1 — behavioural cloning | the ladder's own framing; the compounding-error result is [Ross & Bagnell 2010](https://proceedings.mlr.press/v9/ross10a.html), with DAgger in [Ross et al. 2011](https://arxiv.org/abs/1011.0686) | It is the simplest thing that could work, and its failure mode is specific, provable and reappears throughout Part B, so it earns its place as a baseline rather than a curiosity. |
| 2 — MaxEnt IRL | [Ziebart et al. 2008](https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf); the identifiability result is [Ng, Harada & Russell 1999](https://people.eecs.berkeley.edu/~russell/papers/ml99-shaping.pdf) | Maximum entropy is the formulation that made IRL well-posed, and its gradient is simple enough to implement from the equations, which is where the assumptions become visible. |
| 3 — compare off-distribution | the ladder's own | It is the only comparison that separates the two methods: on the demonstrated distribution they are both fine, so measuring there would show nothing. |

The **lab** pairing (`Practical_RL` week10_planning and yet_another_week) is the
ladder's own, which treats this as bonus material beyond the core syllabus.

The **goal statement** in section 2 is quoted from
[`ladder.md`](../ladder.md#a8). The framing in section 3 of a reward model as an
IRL-like object inheriting the identifiability problem, and the forward links in
section 6, are this repo's rather than claims from the sources.
