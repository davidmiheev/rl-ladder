# The ladder

Single source of truth for what this repo is working through. Ported from the
original HTML tracker so the curriculum lives in git rather than in one
browser's local storage.

The original is archived verbatim at
[`assets/rl-ladder-tracker.html`](assets/rl-ladder-tracker.html) — open it in a
browser for the visual ladder view. Its checkboxes persist through an artifact
runtime API (`window.storage`) that is absent when the file is opened locally,
so ticking there is not saved. **This markdown file is the one that tracks
progress**; treat the HTML as a read-only rendering of the same content.

Three parts, 19 phases, 59 checklist items:

- **Part A — Classical, games & robotics RL** (A1–A8, 26 items). Builds the
  intuition. CPU or one cheap vast.ai box is enough.
- **Part B — LLM post-training RL** (B1–B6, 18 items). Applies it to language
  models. Needs a GPU pod and a small (0.5–3B) HF model.
- **Part C — Market-making parameter RL** (C1–C5, 15 items). Applies the same
  algorithms to tuning the GLFT coefficient set on replayed order-book data.

Tick a box here when the phase README in `experiments/<phase>/` records the run
that backs it. See [`experiment-protocol.md`](experiment-protocol.md) for what
counts as done.

## How to read this file

Every phase opens with **what it is about** — the problem it solves and why it
sits where it does in the sequence — and every checklist item carries an
explanation of what is actually being asked, how to approach it, and what you
should expect to see. The intent is that this file is self-contained: you should
not have to open a paper or a course notebook to know what an item means or why
it is worth doing. The reading and lab links are there to go *deeper*, not to
make the item comprehensible.

**The checkbox lines themselves are verbatim** from the original tracker
(archived at [`assets/rl-ladder-tracker.html`](assets/rl-ladder-tracker.html)) —
only the surrounding explanations are this repo's. That is checked
mechanically, so a tick always refers to the item as originally written.

Per-phase writeups in [`writeups/`](writeups/) go further still: the derivations,
the design decisions behind each experiment, what the runs actually showed, and
a table citing the source of every item.

## Cadence

Work Part A roughly in order; **A7 and A8 are side branches**. Part B assumes
Part A's policy-gradient (A4) and rollout-averaging (A6) intuition, but only
strictly needs familiarity with policy gradients and value functions to start.
B1–B4 build on each other directly; **B5 can run in parallel with B4**; B6 is
optional, once B1–B3 feel mechanical. Part C can start any time after A5
(continuous control) — C4 additionally reuses B1/B3's PPO/GRPO implementations
directly, so it is the natural place to do Part C after touching Part B. In
practice this repo ran Part C first, after Part A: only C4 needs B1/B3, and
everything before it needs A5 at most.

Where a phase lists a **Practical lab**, that is a hands-on notebook from
Yandex School of Data Analysis's [Practical_RL](https://github.com/yandexdataschool/Practical_RL)
or [nlp_course](https://github.com/yandexdataschool/nlp_course) covering that
exact topic — worth running *alongside* your own implementation rather than
instead of it. Forks of every lab repo are pinned in
[`companion-repos.md`](companion-repos.md).

---

# Part A — Classical, Games & Robotics RL

CPU or one cheap vast.ai instance is enough here — no need to reserve serious
GPU compute.

<a id="a1"></a>

## A1. Multi-Armed Bandits

**What this phase is about.** A bandit is a Markov decision process with the
state removed: one situation, `k` actions, a fixed reward distribution behind
each. Everything that makes RL hard — credit assignment across time,
bootstrapping, function approximation — is deliberately absent, leaving one
difficulty standing alone: you can only learn about an action by taking it, and
every pull spent learning is a pull not spent earning. That is the
exploration/exploitation tradeoff, and this phase exists to make it concrete
before anything is layered on top.

**Goal.** The exploration/exploitation tradeoff in its purest form, before state
or function approximation get added.

**Reading.** [Sutton & Barto ch. 2](http://incompleteideas.net/book/the-book.html) ·
[UCB1 paper](https://link.springer.com/article/10.1023/A:1013689704352) ·
[Thompson sampling tutorial](https://arxiv.org/abs/1707.02038)

**Practical lab.** [Practical_RL week05_explore](https://github.com/yandexdataschool/Practical_RL/tree/master/week05_explore) (bandits, UCB, Thompson sampling)

- [x] Implement ε-greedy, UCB1, and Thompson sampling on a synthetic k-armed bandit; plot regret curves

  Three different answers to one question: how do you spend pulls on *learning*
  rather than *earning*? Build a synthetic testbed — each arm has a hidden true
  value `q*(a) ~ N(0,1)` and returns `reward ~ N(q*(a), 1)` — and implement all
  three. Score them by cumulative regret, the reward given up against always
  pulling the best arm, computed from the hidden `q*` rather than realised
  rewards so the curve measures decisions and not luck. Expect Thompson ≲ UCB1
  < ε-greedy, and note that a single run tells you nothing: average over many
  seeds.

- [x] Sweep ε and the UCB exploration constant; observe the exploration/exploitation tradeoff

  Both parameters buy the same thing — exploration — at different prices. Sweep
  each over a log-spaced grid and plot final performance against the parameter.
  The shape you are looking for is an inverted U: too little exploration never
  identifies the best arm, too much keeps paying the exploration tax after the
  answer is known. The optimum is horizon-dependent, so record the budget you
  swept at; a constant that looks wrong at 1000 pulls may be right at 100 000.

- [x] Make the bandit non-stationary; show fixed-step-size ε-greedy tracking drift where sample-averaging fails

  Let the arms drift: start them equal and give each an independent random
  walk. Now run two ε-greedy agents that differ *only* in the update rule —
  sample-average (`α = 1/N`) against a constant `α`. Unrolling the incremental
  update shows why one tracks and one cannot: constant `α` is an exponential
  recency average, while `1/N` weights the reward from pull 1 exactly as
  heavily as pull 10 000. Expect the sample-average agent to lock onto
  information that has gone stale, and look at the spread across seeds rather
  than the mean — the failure is a bias toward old data, not extra noise.

<a id="a2"></a>

## A2. Tabular RL: Q-Learning & SARSA

**What this phase is about.** State comes back, and nothing else does: finitely
many states and actions, values held in a table, no approximation anywhere.
That makes this the last rung where the exact answer is computable, so it is
where you get a reference to measure against. The phase solves the same problem
twice — once with dynamic programming, which is *given* the transition model,
and once with temporal-difference control, which only ever sees sampled
transitions — and the gap between them is the cost of not having a model.

**Goal.** The on-policy/off-policy distinction concretely, and why TD
bootstrapping trades bias for variance reduction.

**Reading.** [Sutton & Barto ch. 4–6](http://incompleteideas.net/book/the-book.html)

**Practical lab.** [Practical_RL week02_value_based](https://github.com/yandexdataschool/Practical_RL/tree/master/week02_value_based)
and [week03_model_free](https://github.com/yandexdataschool/Practical_RL/tree/master/week03_model_free)
(value/policy iteration, Q-learning, SARSA, TD(λ))

- [x] Implement value iteration & policy iteration on a small GridWorld; confirm convergence to the known optimum

  Both compute the optimal value function *given the model*, which is the
  luxury every later phase loses. Value iteration repeatedly applies the
  Bellman optimality backup until it stops moving; policy iteration alternates
  exact evaluation of the current policy (a linear solve) with greedy
  improvement. "Known optimum" needs a check that does not come from the same
  equations: on a deterministic grid with −1 per step and γ = 1, the optimal
  value of a state is exactly minus its shortest-path distance to the goal, so
  compare against a breadth-first search. Watch for the trap that exact policy
  evaluation needs γ < 1 — at γ = 1 a policy that walks into a wall forever
  makes the linear system singular.

- [x] Implement Q-learning and SARSA from scratch; compare policies near a hazard (cliff-walking style)

  Identical update shape, one difference: the bootstrap target. Q-learning
  takes the max over next actions and so evaluates a greedy policy it is not
  following (off-policy); SARSA bootstraps from the action its ε-greedy policy
  actually took (on-policy), and therefore prices in its own exploration. A
  hazard is what makes the difference visible — use a cliff-walking grid where
  the shortest route runs along a large penalty. Expect Q-learning to learn the
  optimal edge-hugging path yet earn *less* online, because one exploratory
  step in ten sends it over; SARSA earns more online while learning a detour.
  Report both the online return and the value of the learned greedy policy, or
  you will draw the opposite conclusion from whichever one you picked.

- [x] Sweep discount factor γ and learning rate α

  These are different kinds of object and should be swept with different tools.
  **γ defines the objective**: build a small MDP with a near small reward and a
  far larger one, derive by hand the γ at which the optimal action flips, and
  confirm dynamic programming agrees — sweeping γ does not find a better
  solution, it changes which solution is correct. **α controls the estimator**:
  sweep it with TD control and look for instability at the top end. Run it on
  both a deterministic and a stochastic version of the task, because that tells
  you whether the noise α has to average comes from the environment or from the
  agent's own exploration.

- [x] Add a Dyna-Q planning step and measure the sample-efficiency gain over plain Q-learning

  Record the transitions you have already seen as a model, and after every real
  step replay `n` of them as extra Q-learning updates. Every arm takes the same
  number of *real* environment steps, so any improvement is pure reuse of
  experience rather than more of it. Expect a large drop in episodes-to-
  convergence, because one-step Q-learning propagates value backwards only one
  state per real transition and planning does that propagation in memory for
  free. Note the assumption you are making: the model is deterministic and
  trusted absolutely, which is the same staleness risk a replay buffer carries
  in A3.

<a id="a3"></a>

## A3. Value-Function Approximation: DQN

**What this phase is about.** The table stops scaling the moment the state is
continuous, so `Q` becomes a neural network trained by regression on TD
targets. That one substitution destroys A2's guarantees: value iteration
converged because the Bellman operator contracts in the space of exact value
functions, and composing it with a function approximator's projection need not
contract at all. The result is the *deadly triad* — bootstrapping, off-policy
learning, and function approximation — which is fine in any pair and can
diverge as a trio. This phase is about which stabiliser addresses which leg of
it.

**Goal.** The "deadly triad" (bootstrapping + off-policy + function
approximation) and which fix addresses which part of it.

**Reading.** [DQN (Atari) paper](https://arxiv.org/abs/1312.5602) ·
[Double DQN](https://arxiv.org/abs/1509.06461) ·
[Dueling DQN](https://arxiv.org/abs/1511.06581)

**Practical lab.** [Practical_RL week04_approx_rl](https://github.com/yandexdataschool/Practical_RL/tree/master/week04_approx_rl)
(experience replay, target networks, double/dueling/bootstrap DQN)

- [x] Implement DQN on CartPole/LunarLander; disable replay buffer and target network independently, document the instability

  The table is gone: `Q` is now a network trained by regression onto TD
  targets. That single change breaks A2's convergence guarantee, because the
  contraction argument assumed exact backups — bootstrapping, off-policy
  learning and function approximation together are the "deadly triad". The two
  stabilisers address different legs of it: **replay** attacks correlated data
  and a training distribution that moves with the policy, and the **target
  network** attacks the fact that the regression target shifts the moment the
  parameters do. Remove them one at a time and you can see which does which.
  Evaluate greedily and separately from training, record peak as well as final
  (instability looks like a high peak followed by collapse), and measure
  baselines — random, and each constant action — so you can distinguish
  "learned badly" from "diverged".

- [x] Add Double DQN; measure whether Q-value overestimation drops

  A max over noisy estimates is biased upward, since `E[max] ≥ max E`, and
  because the same noisy values both *select* and *evaluate* the action, the
  error is selected for rather than averaged away. Double DQN splits those
  roles: the online network picks the action, the target network scores it.
  Measure the bias directly — the value the network predicts at a start state
  minus the discounted return the greedy policy actually collects from it.
  Train long enough to reach the regime you are measuring: early on the network
  *under*estimates because it has not finished learning, and the overestimation
  only appears once the estimates are good enough for the max bias to dominate.

- [x] Add a dueling architecture; compare sample efficiency vs. vanilla DQN

  Split the head into a state-value stream and an advantage stream and
  recombine as `Q(s,a) = V(s) + (A(s,a) − mean_a A(s,a))`; subtracting the mean
  is what makes the decomposition identifiable, since otherwise a constant can
  slide between the two streams. The claim is *sample efficiency*, not final
  score: `V(s)` is updated by every transition through `s` whatever action was
  taken, while a plain head only updates the column of the action tried.
  Compare at equal environment steps. Be aware the benefit should scale with
  the number of actions, so a two-action task is close to the weakest possible
  test of it.

- [x] Optional: run DQN on a small Atari environment (e.g. Pong)

  The scale test: pixel observations, a convolutional trunk, frame stacking,
  and millions of steps rather than tens of thousands. It is also the
  environment where the previous two items actually resolve — Atari supplies
  both the value noise Double DQN targets and the large action space dueling
  targets, which is exactly why the original papers used it.

<a id="a4"></a>

## A4. Policy Gradient & Actor-Critic

**What this phase is about.** Everything so far learned *values* and derived a
policy from them by taking an argmax. This phase parameterises the policy
itself and improves it by gradient ascent on expected return. That removes the
max operator and its bias entirely, works directly with continuous or
structured action spaces, and gives a smooth, stochastic policy — at the cost
of variance, since the gradient must be estimated from sampled returns. Almost
every algorithm in Part B is a descendant of what is derived here, which makes
this the most load-bearing phase in Part A.

**Goal.** The policy-gradient theorem — the throughline connecting almost every
algorithm in Part B.

**Reading.** [REINFORCE (Williams 1992)](https://doi.org/10.1007/BF00992696) ·
[A3C paper](https://arxiv.org/abs/1602.01783) ·
[GAE paper](https://arxiv.org/abs/1506.02438)

**Practical lab.** [Practical_RL week06_policy_based](https://github.com/yandexdataschool/Practical_RL/tree/master/week06_policy_based)
(REINFORCE, advantage actor-critic)

- [x] Implement REINFORCE on CartPole; compare variance/efficiency against your A3 DQN run

  A different strategy entirely: instead of learning values and acting
  greedily, parameterise the policy `π_θ(a|s)` and adjust it directly. The
  policy-gradient theorem gives `∇J(θ) = E[∇ log π_θ(a|s) · G]` — push up the
  log-probability of an action in proportion to the return that followed it. It
  is unbiased and needs no value function, but the variance is severe, because
  every action in an episode is credited with the whole episode's return
  regardless of whether it helped. Compare against your A3 DQN on the same task
  on sample efficiency, spread across seeds, and stability, and use A3's honest
  final number rather than its peak.

- [x] Add a learned baseline + GAE; sweep λ and observe the bias/variance tradeoff

  Subtracting any state-dependent baseline `b(s)` from the return leaves the
  gradient unbiased — it has zero expectation against `∇ log π` — while
  potentially cutting its variance a lot. The natural choice is a learned
  `V(s)`, which turns the raw return into an advantage: how much better was
  this action than the state's average. GAE then interpolates between a one-
  step TD residual (low variance, biased by whatever error `V` has) and the
  full Monte Carlo return (unbiased, high variance) with a single `λ`. Sweep it
  and you should see both failure modes at the two ends rather than a monotone
  curve.

- [x] Implement A2C on LunarLander; compare stability against REINFORCE-with-baseline

  Actor and critic trained together, with several environments stepped in
  parallel so that a batch is decorrelated without needing a replay buffer —
  the on-policy answer to the problem replay solved in A3. Compare stability
  against REINFORCE-with-baseline on a harder task than CartPole, where the
  credit-assignment horizon is longer and the reward is shaped.

<a id="a5"></a>

## A5. Continuous Control (Robotics)

**What this phase is about.** In continuous action spaces `max_a Q(s,a)` is no
longer a lookup over a handful of columns but an optimisation problem in its
own right, solved afresh at every step. The algorithms here answer that with an
actor: a learned function whose output *is* the maximising action. The phase is
really about the failure modes that creates — an actor and critic that move
together and can chase each other — and about three progressively better
answers to them.

**Goal.** What changes with continuous actions, and why SAC's entropy
maximization tends to produce more robust policies.

**Reading.** [DDPG paper](https://arxiv.org/abs/1509.02971) ·
[TD3 paper](https://arxiv.org/abs/1802.09477) ·
[SAC paper](https://arxiv.org/abs/1801.01290)

**Practical lab.** [Practical_RL week09_policy_II](https://github.com/yandexdataschool/Practical_RL/tree/master/week09_policy_II)
(TRPO, PPO, DDPG for robot control)

- [x] Implement DDPG on Pendulum-v1; note training instability

  With continuous actions the `max_a Q(s,a)` at the heart of Q-learning becomes
  an optimisation problem rather than a lookup over a few columns. DDPG's
  answer: learn a deterministic actor whose output *is* the maximising action,
  and train it by gradient ascent through the critic. Expect instability, and
  understand its source — the critic is being queried at actions the actor is
  simultaneously moving, so the two can chase each other.

- [x] Implement TD3 (twin critics, delayed updates, target smoothing); compare stability to DDPG

  Three targeted fixes to the specific ways DDPG fails. **Twin critics** with
  the minimum taken, against the overestimation you measured in A3; **delayed
  actor updates**, so the critic is reasonably fitted before it is used to
  improve the policy; and **target policy smoothing**, noise added to the
  target action so that a sharp error in the critic cannot be exploited by a
  knife-edge action. Add them and compare — ideally one at a time, since the
  paper's contribution is the combination and it is worth knowing which part
  carries it.

- [x] Run/implement SAC on a harder continuous task; sweep the entropy-temperature coefficient

  SAC maximises reward *plus* policy entropy, which makes exploration part of
  the objective rather than noise injected on top of a deterministic policy.
  Sweep the temperature that trades the two and watch what a too-high value
  does (a policy that stays deliberately random) against a too-low one
  (premature collapse to a narrow behaviour).

- [x] Optional: add action/observation noise at eval time; compare degradation across DDPG/TD3/SAC

  Perturb actions or observations at evaluation time only, and compare how much
  each algorithm degrades. The entropy-maximised policy should be flatter
  around its optimum — it was trained to keep a distribution of good actions
  rather than one — so it should lose less, which is the practical argument for
  SAC beyond its learning curves.

<a id="a6"></a>

## A6. Planning & Self-Play: MCTS / AlphaZero

**What this phase is about.** Everything so far learned a reactive policy: see
state, emit action. This phase adds *deliberation* — spending computation at
decision time to search ahead through a model of the game. The interesting part
is the combination: search produces better decisions than the current network,
those decisions become training targets, and the improved network makes the
next search better. It is also a useful contrast for B3, where GRPO averages
over a group of rollouts instead of searching through them.

**Goal.** How planning (search) and model-free learning combine — a useful
contrast to GRPO's rollout-averaging in B3.

**Reading.** [MCTS survey (Browne et al.)](https://doi.org/10.1109/TCIAIG.2012.2186810) ·
[AlphaGo Zero](https://doi.org/10.1038/nature24270) / [AlphaZero](https://arxiv.org/abs/1712.01815)

**Practical lab.** [Practical_RL week10_planning](https://github.com/yandexdataschool/Practical_RL/tree/master/week10_planning)
(MCTS for toy tasks) and [week05_explore](https://github.com/yandexdataschool/Practical_RL/tree/master/week05_explore)
(UCB for MCTS)

- [x] Implement plain MCTS (UCT) on Tic-Tac-Toe/Connect Four; verify convergence to near-optimal play

  Build a search tree by repeating four steps: select down the tree with a UCB
  rule (A1's confidence bonus, applied to child nodes), expand a new node, roll
  out to a result, and back the result up the path. No learning is involved —
  the strength comes entirely from the simulations spent. Verify that with
  enough of them it converges to near-optimal play on a game small enough that
  you can check.

- [x] Add a learned policy/value net guiding search, trained via self-play; compare sample efficiency to plain MCTS

  Replace the random rollout with a network that supplies a prior over moves
  and an evaluation of the position, trained on the outcomes of the search's
  own self-play games. Search improves the network's targets, and the improved
  network makes search better — the loop that AlphaZero is. Compare the
  simulations needed for equal playing strength.

- [x] Ablate the value network (random/uniform rollouts) to isolate search vs. learned-prior strength

  Swap the learned evaluation back out for random rollouts while keeping
  everything else. It separates how much of the strength is search and how much
  is the learned prior, which is the question the previous item's improvement
  leaves open.

<a id="a7"></a>

## A7. Multi-Agent RL (optional)

**What this phase is about.** Every convergence result so far assumed a
stationary environment. Put two learning agents in one environment and each
becomes part of the other's dynamics, so from either point of view the world
changes as the opponent improves. This phase is about seeing that break, and
about self-play as the standard response to it.

**Goal.** Why multi-agent RL breaks single-agent convergence guarantees, and why
self-play sidesteps rather than solves that.

**Reading.** [Independent Q-Learning (Tan 1993)](http://web.media.mit.edu/~cynthiab/Readings/tan-MAS-reinfLearn.pdf) ·
[MADDPG paper](https://arxiv.org/abs/1706.02275) ·
[OpenAI self-play blog](https://openai.com/index/competitive-self-play/)

**Practical lab.** [PettingZoo tutorials](https://pettingzoo.farama.org/tutorials/)
(Farama Foundation) — independent-learner and self-play training on standard
multi-agent environments; not covered by Practical_RL.

- [x] Train two independent Q-learners in a simple competitive game; observe non-stationarity from each agent's view

  Each agent treats the other as part of the environment. But the other is
  learning, so from either agent's point of view the environment's dynamics
  change over time — and every single-agent convergence result assumed they do
  not. Watch a policy that was good become bad because the opponent moved,
  which is what breaks the guarantee.

- [x] Try self-play (one agent playing both sides); compare policy quality/stability to independent learners

  One policy plays both sides, so improvement automatically produces a stronger
  opponent and the curriculum is generated for free. Compare quality and
  stability against the independent learners, and note what it does and does
  not fix: self-play *sidesteps* non-stationarity by making both sides the same
  object, it does not solve the underlying problem.

<a id="a8"></a>

## A8. Imitation Learning & Inverse RL (optional)

**What this phase is about.** So far the reward has been given. Here it is not:
you have demonstrations of good behaviour and must either copy the behaviour
(imitation) or recover the objective behind it (inverse RL). The distinction
matters well beyond this phase — a reward model in Part B is exactly an attempt
to recover an objective from human judgements rather than to imitate them.

**Goal.** The difference between copying actions (imitation) and recovering the
objective behind them (inverse RL) — relevant background for any
reward-model-adjacent work in Part B.

**Reading / lab.** [Practical_RL week10_planning](https://github.com/yandexdataschool/Practical_RL/tree/master/week10_planning)
and [yet_another_week](https://github.com/yandexdataschool/Practical_RL/tree/master/yet_another_week)
(imitation learning and inverse RL). This phase is lab-first: the course treats
it as bonus material beyond the core syllabus.

- [x] Implement behavioral cloning on a small control task (e.g. CartPole) from a handful of expert demonstrations; observe compounding error away from the expert's state distribution

  Pure supervised learning: fit a policy to expert state-action pairs, no
  reward involved. The failure mode is the interesting part — a small error
  moves the agent to a state the expert never visited, where the policy has no
  training data, so errors compound rather than correct. Measure that drift
  rather than just the training accuracy.

- [x] Implement a simple inverse RL method (e.g. max-entropy IRL) to recover a reward function from the same demonstrations, then run Q-learning/PPO against the recovered reward

  Instead of copying the actions, recover a reward function that would make the
  demonstrations look optimal, then run ordinary RL against it. The motivation
  is generalisation: a reward is a compact statement of the *objective*, and it
  stays valid in states the demonstrations never reached, where a copied action
  does not.

- [x] Compare the behavioral-cloning policy against the IRL-then-RL policy on states the expert demonstrations never visited

  Evaluate both specifically on states outside the demonstration distribution.
  That is where the two approaches should come apart, and it is the whole
  reason to prefer recovering an objective over imitating a behaviour — the
  same argument that motivates reward models in Part B.

---

> **Part A is complete (26/26).** The cross-cutting lessons — including the six
> times a headline metric hid the effect being measured — are collected in
> [`part-a-summary.md`](part-a-summary.md).
>
> **Part C is being worked before Part B**, on the reasoning that Part A is the
> prerequisite for both and Part C needs nothing from Part B until C4. The
> ladder below is left in its original order; work jumps to
> [Part C](#part-c--quant-research-market-making-parameter-rl) and comes back here.

# Part B — LLM Post-Training RL

This is where you want a vast.ai (or RunPod) GPU pod. Small HF model (0.5–3B)
via TRL for fast iteration.

<a id="b1"></a>

## B1. Foundations: REINFORCE → PPO

**What this phase is about.** The same policy-gradient machinery from A4,
applied to a language model. The action space becomes a vocabulary of tens of
thousands of tokens, an episode is a generated sequence, and the reward arrives
once at the end from a model or a program rather than from an environment. Two
things are genuinely new: the policy starts from a pretrained model worth
preserving, and the reward is a learned proxy that can be exploited. PPO's
clipping addresses the first problem, the KL-to-reference term the second.

**Reading.** [Learning to Summarize w/ Human Feedback](https://arxiv.org/abs/2009.01325) ·
[InstructGPT](https://arxiv.org/abs/2203.02155) ·
[PPO paper](https://arxiv.org/abs/1707.06347)

**Practical lab.** [nlp_course week07_finetuning](https://github.com/yandexdataschool/nlp_course/tree/2025/week07_finetuning)
(PEFT & RLHF)

- [x] Reuse your A4 REINFORCE implementation on a language-modeling reward; observe reward hacking in a generative action space

  Same estimator, a radically different action space: the "actions" are tokens
  from a vocabulary of tens of thousands, and an "episode" is a generated
  sequence scored once at the end. Reuse A4's REINFORCE against a reward model
  or a programmatic reward and watch how fast it finds degenerate outputs that
  score well without being good — with no constraint tying the policy to the
  pretrained model, nothing stops it.

- [x] Train full PPO (TRL PPOTrainer); sweep clip range and KL coefficient; plot reward vs. KL-to-reference

  PPO makes several gradient steps per batch of generations safe by clipping
  the probability ratio between the new policy and the one that produced the
  data, so an update cannot move far from the distribution it was estimated on.
  RLHF adds a second leash: a KL penalty against the frozen reference model,
  which keeps the fluency the pretraining bought. Sweep the clip range and the
  KL coefficient and plot reward against KL-to-reference — that curve *is* the
  tradeoff, and reading it is the point of the item.

- [x] Set the KL penalty near zero and document the resulting failure mode

  Remove the anchor and document what breaks. Reward keeps climbing while the
  text degenerates, because the reward model is a proxy trained on a
  distribution the policy has now left. This is the cheapest possible
  demonstration that the KL term is not a regulariser bolted on for stability —
  it is what makes the proxy meaningful.

<a id="b2"></a>

## B2. Direct Alignment: DPO / IPO

**What this phase is about.** RLHF is an expensive pipeline: train a reward
model, then run an RL loop that generates, scores and updates. DPO observes
that the objective has a closed-form optimal policy, and that you can therefore
fit the policy *directly* from preference pairs with a supervised-looking loss
— no reward model, no sampling. This phase is about what that buys and what it
quietly gives up.

**Reading.** [DPO paper](https://arxiv.org/abs/2305.18290) ·
[IPO paper](https://arxiv.org/abs/2310.12036)

**Practical lab.** [TRL DPOTrainer docs](https://huggingface.co/docs/trl/dpo_trainer)
(supports both DPO and IPO loss via one config flag)

- [x] Build/synthesize a preference dataset; train DPO; compare against the best PPO run

  DPO's insight is that the RLHF objective has a closed-form optimal policy, so
  the reward model can be substituted out and the policy fitted *directly* from
  preference pairs with a classification-style loss — no reward model, no
  sampling loop, no RL. Build a preference set, train it, and compare against
  your best PPO run on both quality and the engineering cost of getting there.

- [x] Overfit DPO on a small preference set; show chosen-logprob collapse and KL blowup

  Train too long on too little and watch a specific pathology: the log-
  probability of the *chosen* response falls. The loss only cares about the
  margin between chosen and rejected, and pushing the rejected one down far
  enough achieves that while dragging both down. Track chosen and rejected log-
  probabilities separately, plus KL to the reference — the aggregate loss looks
  fine while this happens.

- [x] Swap in IPO's bounded objective; compare stability

  IPO replaces the unbounded log-sigmoid with an objective that saturates, so
  an easy preference pair cannot be driven to an infinite margin. That is aimed
  exactly at the failure above; check whether it actually fixes it in your
  setup rather than assuming it does.

<a id="b3"></a>

## B3. RLVR & GRPO

**What this phase is about.** Two ideas that reshaped post-training. **RLVR**
replaces the learned reward model with a verifier — a test suite, an exact
answer match — so the reward cannot be flattered, only circumvented. **GRPO**
removes the value network by sampling a group of completions per prompt and
using the group's mean score as the baseline, which matters at a scale where
the critic is another model as large as the policy. The phase asks you to
derive GRPO rather than import it, because the derivation is where its
assumptions are visible.

**Reading.** [DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300) ·
[DeepSeek-R1](https://arxiv.org/abs/2501.12948) ·
[RL-for-LLMs survey](https://arxiv.org/abs/2509.08827) ·
[first-principles GRPO derivation](https://arxiv.org/abs/2606.16733)

**Practical lab.** [HuggingFace Open R1](https://github.com/huggingface/open-r1)
(full open reproduction of DeepSeek-R1's GRPO pipeline) and
[TRL GRPOTrainer docs](https://huggingface.co/docs/trl/grpo_trainer)

- [ ] Implement GRPO from the derivation yourself — group rollouts, group-relative advantages, no value net

  Sample a group of `G` completions for the same prompt, score them, and use
  the group's mean score as the baseline. The advantage of a completion becomes
  how much better it was than its own siblings — a Monte Carlo baseline
  estimated from the group, which removes the need for a learned value network
  entirely. That matters at LLM scale, where the critic is another model of
  comparable size. Derive it before you implement it; the derivation is where
  the assumptions live.

- [ ] Cross-check against TRL's GRPOTrainer

  Run yours and the library's on the same data and reconcile the differences.
  They almost always hide in the same few places: whether advantages are
  normalised and over what, whether the loss is averaged per token or per
  sequence, and whether the KL term enters the reward or the loss. Assume yours
  is wrong until shown otherwise, and record the discrepancy even after fixing
  it.

- [ ] Sweep group size; measure advantage-estimate variance

  The group mean is an estimate of the baseline from `G` samples, so its
  variance falls roughly as `1/G` while generation cost rises linearly with it.
  Measure both sides of that trade rather than accepting a default, and
  remember the cost is multiplied by every rollout you make.

- [ ] Run RLVR against a gameable verifier and document reward hacking

  Reinforcement learning from *verifiable* rewards replaces a learned reward
  model with a checker — a test suite, a numeric answer match — which cannot be
  flattered the way a reward model can. Then deliberately use a checker with a
  loophole (matching the final number without the reasoning, say) and document
  what the policy finds. Verifiable does not mean unhackable; it means the hack
  has to be in the checker.

<a id="b4"></a>

## B4. Reward Hacking & Overoptimization

**What this phase is about.** Every reward here is a proxy for something you
actually want, and optimising a proxy hard enough eventually moves away from
the target. This phase makes that quantitative: track the proxy and a held-out
gold metric together, find where they diverge, and measure how much a
mitigation moves that point. It is the phase that turns "reward hacking" from a
worry into a number.

**Reading.** [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565) ·
[Scaling Laws for RM Overoptimization](https://arxiv.org/abs/2210.10760)

**Practical lab.** [RewardBench](https://github.com/allenai/reward-bench) (AI2) —
evaluate your own trained reward model against a standard suite before
deliberately overoptimizing against it

- [ ] Overoptimize PPO against a learned RM; plot proxy reward vs. held-out gold metric

  Train hard against a learned reward model while tracking a held-out gold
  metric that the model never saw. The expected shape is proxy reward rising
  monotonically while the gold metric peaks and then declines — Goodhart's law
  with an x-axis. The useful output is the *location* of that turn, in KL,
  because that is the budget you actually have.

- [ ] Repeat with GRPO on a verifiable reward; compare hackability

  Same experiment with a checker instead of a learned model, to see how much of
  the overoptimisation was the reward model's error surface rather than the
  optimiser's strength.

- [ ] Try one mitigation (RM ensembling, tighter KL); quantify the recovered budget

  Pick one — an ensemble of reward models so that idiosyncratic errors
  disagree, or simply a tighter KL — and quantify how much further you can
  optimise before the gold metric turns. A mitigation that moves the turning
  point is worth something; one that only lowers the whole curve is not.

<a id="b5"></a>

## B5. On-Policy Distillation

**What this phase is about.** Getting a small model to behave like a large one.
The naive approach trains the student on the teacher's outputs, but the student
will be evaluated on *its own* generations, and those diverge from the
teacher's the moment it makes a mistake — the compounding-error problem from
A8, in a new setting. On-policy distillation fixes the mismatch by having the
student generate and the teacher label.

**Reading.** [MiniLLM](https://arxiv.org/abs/2306.08543) ·
[GKD](https://arxiv.org/abs/2306.13649) ·
[2026 OPD survey](https://arxiv.org/abs/2604.00626)

**Practical lab.** [nlp_course week08_efficiency](https://github.com/yandexdataschool/nlp_course/tree/2025/week08_efficiency)
(quantization, distillation, pruning, speculative decoding)

- [ ] Offline-KD baseline: distill the best B3 model into a smaller student

  Distil your best model into a smaller student on a fixed dataset of teacher
  outputs. The flaw to keep in view: the student is trained on the teacher's
  distribution but will be *evaluated* on its own, and those diverge as soon as
  it makes its first mistake — the same compounding-error argument as
  behavioural cloning in A8.

- [ ] GKD-style on-policy distillation: student rollouts, teacher relabels

  Let the student generate, and have the teacher label the student's own
  sequences. The training distribution becomes the one the student will
  actually be scored on, which is what "on-policy" means here. Compare against
  the offline baseline, especially on longer generations where the drift has
  room to accumulate.

- [ ] Run both students through your quantized inference path; compare robustness

  Quantisation perturbs the weights, so this asks whether the two distillation
  methods produced students that are equally robust to that perturbation, not
  just equally good in fp16.

<a id="b6"></a>

## B6. Unifying View + Systems Port

**What this phase is about.** Consolidation. Every method in Part B differs in
a small number of dimensions — what serves as the baseline, how far off-policy
the data may be, what limits the size of an update — and seeing them on those
axes is worth more than remembering five algorithms. The Rust port is a
comprehension test: an implementation forces every shape, reduction and
normalisation to be explicit.

**Reading.** [2026 unified off-policy/on-policy post-training survey](https://arxiv.org/abs/2604.07941) · optional

**Practical lab.** [TRL source](https://github.com/huggingface/trl) — one codebase
with PPO, DPO, GRPO, and GKD implementations side by side; reading the shared
abstractions is a good proxy for the survey's unifying framework

- [ ] Map B1–B5 onto the survey's unifying framework

  Put PPO, DPO, GRPO and GKD in one picture and answer the same questions of
  each: what plays the role of the baseline, how far off-policy is the data
  allowed to be, and what constrains the size of an update. Most of the
  apparent variety collapses once those three axes are fixed.

- [ ] Port GRPO's core update into Rust against your Candle fork — a comprehension check, not a throughput play

  Reimplement the update against your Candle fork. It is a comprehension check
  rather than a throughput play — a port makes every implicit shape, reduction
  and normalisation explicit, and you cannot port what you have not understood.

---

# Part C — Quant Research: Market-Making Parameter RL

Results so far, in one page: [`writeups/c0-part-c-overview.md`](writeups/c0-part-c-overview.md).

hftbacktest as the environment (Rust-backed, replays real order-book data). CPU
is enough for calibration; a small GPU helps once you are running many parallel
episodes for PPO/GRPO-style rollouts.

<a id="c1"></a>

## C1. Baseline: Calibrated GLFT (no RL)

**What this phase is about.** Before any learning, establish what good already
looks like. The Guéant–Lehalle–Fernandez-Tapia model derives a market maker's
optimal quotes in closed form from a few measurable quantities — order-arrival
intensity and volatility — and its calibrated performance is the benchmark
every RL variant in Part C must beat. Reproducing it exactly also validates
your data pipeline and fill assumptions, which is where backtests usually go
wrong.

**Goal.** This fixed-calibration run is the benchmark every RL variant below has
to beat — on the same held-out days, not just the training days.

**Reading.** [Guéant–Lehalle–Fernandez-Tapia paper](https://arxiv.org/abs/1605.01862) ·
your GLFT + grid-trading notes

**Writeup.** [`writeups/c1-glft-baseline.md`](writeups/c1-glft-baseline.md) ·
lab notebook [`experiments/c1_glft_baseline/README.md`](../experiments/c1_glft_baseline/README.md)

**Practical lab.** [hftbacktest's official GLFT + grid-trading notebook](https://github.com/nkaz001/hftbacktest/blob/master/examples/GLFT%20Market%20Making%20Model%20and%20Grid%20Trading.ipynb)
— likely the direct source of your own notes; run it as-is before modifying anything

- [x] Calibrate A, k (order-arrival intensity) and σ (volatility) from market data; derive c1, c2 and reproduce the GLFT half-spread/skew formulas

  The GLFT model prices a market maker's quotes from a small number of
  measurable quantities: how order arrivals fall off with distance from the mid
  (the intensity parameters `A` and `k`), and volatility `σ`. Estimate them
  from data, derive the model's coefficients, and reproduce its closed-form
  half-spread and inventory skew. This is the analytic model that Part C's RL
  will later adjust rather than replace.

- [x] Run the calibrated GLFT grid market maker in hftbacktest on the sample data; reproduce the reported Sharpe/Sortino/MaxPositionValue stats

  Replay real order-book data through the calibrated quoter and reproduce the
  published statistics. Getting the same numbers is the check that your
  calibration and your fill assumptions are right, before anything is learned
  on top of them.

- [x] Reproduce the simplified non-parametric HF-grid variant (percentile-based intensity, micro-price) and compare against the parametric GLFT baseline

  The same strategy with the parametric intensity fit replaced by empirical
  percentiles and the mid replaced by a micro-price. Comparing them tells you
  how much of the baseline's performance comes from the model's functional form
  versus from the market microstructure it is reading.

<a id="c2"></a>

## C2. Framing Market Making as an RL Problem

**What this phase is about.** Turning market making into an RL problem, with
one deliberate constraint: the policy adjusts the *coefficients* of the
analytic model rather than emitting raw quotes. That keeps the known-good
functional form in the loop, makes every action interpretable in the language
the desk already uses, and makes the fixed-coefficient baseline a strict
special case of the learned policy — so "did RL help" has a clean answer.

**Goal.** The key design choice: RL adjusts the coefficients of a known-good
analytic model rather than replacing the model outright — keeps the policy
auditable.

**Reading.** Your order-book-imbalance, basis, and APT alpha notes.

**Writeup.** [`writeups/c2-env.md`](writeups/c2-env.md) ·
lab notebook [`experiments/c2_env/README.md`](../experiments/c2_env/README.md)

**Practical lab.** [Gymnasium custom environment tutorial](https://gymnasium.farama.org/introduction/create_custom_env/)
— the standard pattern for wrapping hftbacktest as a `step()`/`reset()` Gym-style env

- [x] Define state: inventory q, short-horizon volatility estimate, order-book imbalance / micro-price features, basis & APT alpha signals

  What the policy is allowed to see. Inventory matters because risk is a
  function of position; short-horizon volatility and order-book imbalance
  because they predict adverse selection; basis and alpha signals because they
  predict direction. Anything you leave out, the policy cannot condition on —
  and anything you put in, it can overfit to.

- [x] Define the action space as the GLFT coefficient set itself — γ, c1, and a half-spread/skew scaling term — rather than raw quote prices, so the policy stays interpretable inside the existing formula

  The central design decision of Part C. Rather than emitting raw quote prices,
  the policy outputs the *coefficients* of a model that is already known to be
  reasonable. The analytic form stays in the loop, so the policy cannot produce
  nonsense quotes, every action remains interpretable in terms the desk already
  uses, and the fixed-coefficient baseline is a strict special case — which
  makes "did RL help" a well-posed question.

- [x] Define reward as mark-to-market PnL minus an inventory-risk penalty (e.g. −η·q²), mirroring the risk term the GLFT model already encodes analytically

  PnL alone would reward taking unbounded inventory risk that happens not to
  have been punished on the training days. Subtracting a penalty in the squared
  position mirrors the risk aversion the GLFT model already encodes
  analytically, so the learned objective and the baseline's objective are
  comparable.

- [x] Wrap hftbacktest as a Gym-style environment: one episode per trading session, one step per grid-refresh interval

  One episode per trading session, one step per grid refresh. The engineering
  here is mostly making sure the environment's step boundary matches the
  strategy's real decision points, and that nothing from the future leaks into
  the observation.

<a id="c3"></a>

## C3. Continuous Control on the Coefficient Set

**What this phase is about.** The first real test: do learned, state-
conditioned coefficients beat well-calibrated fixed ones? The algorithms are
A5's, unchanged; what is new is an environment where the reward is money, the
episodes are trading sessions, and the evaluation must be on days the policy
never saw.

**Goal.** A real answer to whether learned, state-conditioned coefficients beat a
well-calibrated fixed formula, and which coefficients are worth learning at all.

**Reading.** Reuse your A5 DDPG/TD3/SAC implementations.

**Writeup.** [`writeups/c3-continuous-control.md`](writeups/c3-continuous-control.md) ·
lab notebook [`experiments/c3_continuous_control/README.md`](../experiments/c3_continuous_control/README.md)

**Practical lab.** [Stable-Baselines3 TD3](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html)
and [SAC docs](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
— battle-tested reference implementations to validate your own against

- [x] Apply DDPG/TD3/SAC to the C2 environment: policy outputs (γ, c1, half-spread-scale) per step, conditioned on state

  Reuse the A5 implementations directly. The action is a low-dimensional
  continuous vector, which is the setting those algorithms were designed for —
  the novelty is entirely in the environment.

- [x] Compare against the C1 fixed-calibration baseline on held-out days: Sharpe, Sortino, MaxPositionValue, ReturnOverMDD

  On days never used for calibration or training. Report the risk-adjusted
  measures, not just PnL: a strategy that earns more by carrying more inventory
  has not improved anything.

- [x] Ablate: fix γ at its calibrated value and let RL learn only c1/skew, vs. letting RL learn all three jointly — compare sample efficiency and stability

  Learning fewer coefficients is a smaller hypothesis class and therefore
  harder to overfit. Comparing "learn one" against "learn all three" tells you
  which coefficients genuinely benefit from being state-conditioned, and which
  were fine as constants.

<a id="c4"></a>

## C4. PPO/GRPO for Parameter Search

**What this phase is about.** The bridge between Part B and Part C. The same
PPO and GRPO implementations used for language models are applied to a three-
dimensional continuous action, which is a direct test of whether GRPO's value-
network-free baseline is a general idea or an artefact of the LLM setting. It
is also where Part B's reward-hacking lens gets pointed at a backtest, which is
one of the easiest places in this ladder to fool yourself.

**Goal.** The direct bridge between the LLM post-training RL work and this quant
research — same algorithms, a much smaller and more interpretable action space.

**Reading.** Reuse your B1 PPO and B3 GRPO implementations.

**Practical lab.** [Stable-Baselines3 PPO docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
for the low-dimensional continuous-control side; no off-the-shelf library does
GRPO outside LM tooling, so the group-relative-advantage update here is genuinely
from-scratch code

- [x] Since the action here is a low-dimensional continuous vector, compare PPO (B1) directly against a GRPO-style approach: sample K coefficient sets per episode, rank by realized PnL, apply a group-relative-advantage update as in B3

  A direct test of whether GRPO's central trick survives outside language
  modelling. Sample `K` coefficient sets per episode, rank them by realised
  PnL, and apply a group-relative advantage update — the group mean replaces
  the value network exactly as it does in B3.

- [x] Check whether GRPO's group-baseline idea (no value network) transfers outside language modeling to this much lower-dimensional control problem

  The question behind the previous item. GRPO's baseline works partly because
  LLM value functions are expensive and hard to fit; in a three-dimensional
  continuous action space a critic is cheap and easy, so the trade may simply
  not pay. A negative result here is informative.

- [x] Apply the B4 reward-hacking lens: check whether RL-tuned coefficients overfit to microstructure quirks of the training days (e.g. only widening spread around a few known volatility spikes) rather than learning a generalizable policy — validate strictly out-of-sample

  Backtests are the easiest place in this whole ladder to fool yourself. The
  specific failure to look for: coefficients that encode "widen the spread at
  14:30 on the third Tuesday" rather than a policy — memorising the training
  days' microstructure instead of learning to respond to state. Validate
  strictly out of sample, and treat an implausibly good result as a bug report.

<a id="c5"></a>

Done in [`writeups/c4-ppo-grpo.md`](writeups/c4-ppo-grpo.md). The PPO and GRPO
updates there are written from scratch for a three-parameter Gaussian policy,
not reused from B1/B3, which are not built yet; the comparison is still like
for like, since both share the same clipped update and differ only in the
baseline.

## C5. Multi-Signal Extension

**What this phase is about.** Whether RL earns its complexity. Fold directional
alpha signals into the state and see whether the policy discovers the
interaction between signal and quoting coefficients by itself — or whether it
merely rediscovers what you would have written by hand in an afternoon. A
negative answer here is a genuine finding, not a failure.

**Goal.** Tests whether RL buys you anything over manually engineering the alpha
signal into the analytic formula.

**Reading.** Your basis and APT alpha notes.

**Practical lab.** [hftbacktest examples directory](https://github.com/nkaz001/hftbacktest/tree/master/examples)
— includes the official "Market Making with Alpha" notebooks (imbalance, basis,
APT) your notes are based on

- [x] Fold the basis/APT alpha signals into the state (or into a directly-predicted fair-price adjustment) and re-run C3/C4

  Give the policy directional information and re-run. Either it uses it or it
  does not, and both answers are useful.

- [x] Check whether the RL-tuned coefficients learn to widen/narrow skew in response to the alpha signal automatically, versus the manual extension a fixed-form model would need

  The real question of Part C: does RL discover the *interaction* between an
  alpha signal and the quoting coefficients on its own, or does it merely
  reproduce what you would have written by hand? If it is the latter, the
  analytic model with a manual extension is the better engineering choice, and
  knowing that is worth the experiment.

---

## C6. The strategy zoo (added, not from the original ladder)

Not one of the 59 items. C6 came from a request after C3 landed: run all five
of the strategies hftbacktest publishes rather than only GLFT, give each one a
wider asset-appropriate spread, use more days on both sides of the split, and
put the lot in one table with PnL and inventory plots.

It exists because C3's answer -- learned parameters do not beat fixed ones --
was open to the objection that GLFT is a losing baseline on this venue, so
beating it proves little. C6 removes that objection and asks again.

Writeup: [`writeups/c6-strategy-zoo.md`](writeups/c6-strategy-zoo.md).
Lab notebook: [`../experiments/c6_strategy_zoo/README.md`](../experiments/c6_strategy_zoo/README.md).
Tables: [`writeups/c6-compare.md`](writeups/c6-compare.md) (paired) and
[`writeups/c6-table.md`](writeups/c6-table.md) (per configuration).

C6 is also where C5's two items get answered, since folding the basis and APT
alpha signals into the state and re-running C3 is exactly what the `apt` and
`basis` strategies plus the `alpha_mult` action do. They are ticked there, not
here, only once the evidence for each is in the writeup.


<a id="c7"></a>
## C7. Signals and adaptivity (added, not from the original ladder)

Not one of the 59 items. C7 came from a request after C6's RL results: test
other volatility estimators, imbalance measured in basis points, deeper book
levels and BTCUSDT's book as possible hidden information, and a spread rule
driven by the markouts of our own fills.

Its answer: C6's σ is biased 1.45× low and correcting it pays; the deeper book
and BTCUSDT carry too little at one-second resolution to change a quote; and
the markout rule beats C6's widened baselines on held-out data in every pair
tried.

Writeup: [`writeups/c7-signals.md`](writeups/c7-signals.md).
Lab notebook: [`../experiments/c7_signals/README.md`](../experiments/c7_signals/README.md).


<a id="c8"></a>
## C8. Risk controls, denoising and stress tests (added, not from the original ladder)

Not one of the 59 items. C8 did the Part C follow-ups after C4 except the
forward test: a circuit breaker, daily and trailing loss limits, stress tests
of every configuration that made money, the breadth assets on seven held-out
sessions, and RL on top of the markout rule — plus three denoisers and six
new quoting ideas, each run through the full backtest. The month is re-split
into 24 training and 7 held-out sessions.

Its answer: OBI with the markout rule makes money held out on five of nine
assets, and on SUIUSDC it survives removing the crash, the best session and
the maker rebate, a 1 bp fee and 4× latency (+2.52% held out, +1.13% at a
1 bp fee). XRP's gain is the crash session and ETH's the rebate. A circuit
breaker helps the losing ETH strategies; no denoiser, combination of controls
or RL policy improves OBI.

Writeup: [`writeups/c8-risk-controls.md`](writeups/c8-risk-controls.md).
Lab notebook: [`../experiments/c8_risk_controls/README.md`](../experiments/c8_risk_controls/README.md).


<a id="c9"></a>
## C9. The fair price, forecasts of the mid, and denoising on every asset (added, not from the original ladder)

Not one of the 59 items. C9 asked whether the price the spread is quoted
around can be forecast about a second ahead and whether quoting around the
future mid would pay; then tested denoising on all nine assets, acting only
on large moves, and DAgger with a teacher that sees the future mid.

Its answer: foresight of 1–3 s would be worth a lot on seven of eight
configurations (ETH GLFT −3.25% to +4.30% at 1 s), but the best causal
forecast explains 2–10% of the 1 s move against the ~40% needed; a
micro-price or forecast fair price helps only AVAXUSDC; a perfect
large-move detector would suffice but the book does not anticipate those
moves; the DAgger students halve ETH GLFT's loss without making it pay; and
no denoiser helps on any asset.

Why the learning methods of Part C keep losing to rules, what could make RL
tractable, whether the strategies' parameters are too rigid, and the data
from outside the order book that could raise the information limit (with
free sources) are in the Part C overview, §6–§8
([`writeups/c0-part-c-overview.md`](writeups/c0-part-c-overview.md)).

Writeup: [`writeups/c9-fair-price.md`](writeups/c9-fair-price.md).
Lab notebook: [`../experiments/c9_fair_price/README.md`](../experiments/c9_fair_price/README.md).

## C10. A year forward on Bybit, the crash, and data from outside the book (added, not from the original ladder)

Not one of the 59 items. C10 ran the frozen C8 configurations, unchanged, on
a year of Bybit's free order books (one session a week from 2025-09-26, and
the study month's held-out week as a venue control), tested risk controls on
the 2025-10-10 crash, and measured whether the same asset's other markets
(Binance perpetual and spot trades, Bybit's perpetual) forecast the traded
book's next second — first as forecasts, then through the backtest.

Its answer: without the crash only one of six configurations makes money on
Bybit (XRPUSDC → XRPPERP, +3.96%, +0.21% at the standard maker fee), and the
same held-out week already costs most of the edge on the new venue; none of
C8's controls bounds the crash, while a loss limit that exits at the touch
(stop 50 bp) holds every configuration's crash drawdown to at most 6.4% under
every stress; outside data roughly doubles the 1 s forecast's R² and makes
almost every forecast rule better held out, but the genuine lead (ETH OBI
+0.85% → +1.78%) is gone at +100–250 ms of delay, and XRP's gain is the
crash session.

Writeup: [`writeups/c10-forward.md`](writeups/c10-forward.md).
Lab notebook: [`../experiments/c10_forward/README.md`](../experiments/c10_forward/README.md).

## C13. C8's selection procedure on a year of Bybit, as a heuristic contest (added, not from the original ladder)

Not one of the 59 items. C13 reran C8's whole procedure — a 54-setting
screen, every risk-control family, greedy forward selection, validation and
the stress set — on Bybit with six months of training (48 sessions), a
validation block (27) and a locked test (27, Jun-Oct 2026), on C10's five
contracts and six more, under C8's selection rule and under C10's crash
guard.

Its answer: C8's rule picks the configuration that won the 2025-10-10 crash
(positions past the cap on the right side of the rebound) and loses on all
ten tests with a usable result (−32.45 points); the crash-guarded rule makes
+9.08 points and beats C8's choices run unchanged over C10's five (+3.33 vs
+0.64). Three picks hold up, each a slow quoter with a volatility breaker:
LTCUSDT +9.66% (positive under every stress, +4.62% at Bybit's standard
+2 bp fee), SUIUSDT +4.60% and BNBUSDT +1.06%. ENAUSDT's pick earned
+11,024% on validation from the rebate on a spread floor under one tick,
and lost on every test session.

Writeup: [`writeups/c13-bybit-contest.md`](writeups/c13-bybit-contest.md).
Lab notebook: [`../experiments/c13_bybit_contest/README.md`](../experiments/c13_bybit_contest/README.md).
