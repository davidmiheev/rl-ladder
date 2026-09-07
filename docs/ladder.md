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

## Cadence

Work Part A roughly in order; **A7 and A8 are side branches**. Part B assumes
Part A's policy-gradient (A4) and rollout-averaging (A6) intuition, but only
strictly needs familiarity with policy gradients and value functions to start.
B1–B4 build on each other directly; **B5 can run in parallel with B4**; B6 is
optional, once B1–B3 feel mechanical. Part C can start any time after A5
(continuous control) — C4 additionally reuses B1/B3's PPO/GRPO implementations
directly, so it is the natural place to do Part C after touching Part B.

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

**Goal.** The exploration/exploitation tradeoff in its purest form, before state
or function approximation get added.

**Reading.** [Sutton & Barto ch. 2](http://incompleteideas.net/book/the-book.html) ·
[UCB1 paper](https://link.springer.com/article/10.1023/A:1013689704352) ·
[Thompson sampling tutorial](https://arxiv.org/abs/1707.02038)

**Practical lab.** [Practical_RL week05_explore](https://github.com/yandexdataschool/Practical_RL/tree/master/week05_explore) (bandits, UCB, Thompson sampling)

- [x] Implement ε-greedy, UCB1, and Thompson sampling on a synthetic k-armed bandit; plot regret curves
- [x] Sweep ε and the UCB exploration constant; observe the exploration/exploitation tradeoff
- [x] Make the bandit non-stationary; show fixed-step-size ε-greedy tracking drift where sample-averaging fails

<a id="a2"></a>

## A2. Tabular RL: Q-Learning & SARSA

**Goal.** The on-policy/off-policy distinction concretely, and why TD
bootstrapping trades bias for variance reduction.

**Reading.** [Sutton & Barto ch. 4–6](http://incompleteideas.net/book/the-book.html)

**Practical lab.** [Practical_RL week02_value_based](https://github.com/yandexdataschool/Practical_RL/tree/master/week02_value_based)
and [week03_model_free](https://github.com/yandexdataschool/Practical_RL/tree/master/week03_model_free)
(value/policy iteration, Q-learning, SARSA, TD(λ))

- [x] Implement value iteration & policy iteration on a small GridWorld; confirm convergence to the known optimum
- [x] Implement Q-learning and SARSA from scratch; compare policies near a hazard (cliff-walking style)
- [x] Sweep discount factor γ and learning rate α
- [x] Add a Dyna-Q planning step and measure the sample-efficiency gain over plain Q-learning

<a id="a3"></a>

## A3. Value-Function Approximation: DQN

**Goal.** The "deadly triad" (bootstrapping + off-policy + function
approximation) and which fix addresses which part of it.

**Reading.** [DQN (Atari) paper](https://arxiv.org/abs/1312.5602) ·
[Double DQN](https://arxiv.org/abs/1509.06461) ·
[Dueling DQN](https://arxiv.org/abs/1511.06581)

**Practical lab.** [Practical_RL week04_approx_rl](https://github.com/yandexdataschool/Practical_RL/tree/master/week04_approx_rl)
(experience replay, target networks, double/dueling/bootstrap DQN)

- [x] Implement DQN on CartPole/LunarLander; disable replay buffer and target network independently, document the instability
- [x] Add Double DQN; measure whether Q-value overestimation drops
- [x] Add a dueling architecture; compare sample efficiency vs. vanilla DQN
- [ ] Optional: run DQN on a small Atari environment (e.g. Pong)

<a id="a4"></a>

## A4. Policy Gradient & Actor-Critic

**Goal.** The policy-gradient theorem — the throughline connecting almost every
algorithm in Part B.

**Reading.** [REINFORCE (Williams 1992)](https://doi.org/10.1007/BF00992696) ·
[A3C paper](https://arxiv.org/abs/1602.01783) ·
[GAE paper](https://arxiv.org/abs/1506.02438)

**Practical lab.** [Practical_RL week06_policy_based](https://github.com/yandexdataschool/Practical_RL/tree/master/week06_policy_based)
(REINFORCE, advantage actor-critic)

- [ ] Implement REINFORCE on CartPole; compare variance/efficiency against your A3 DQN run
- [ ] Add a learned baseline + GAE; sweep λ and observe the bias/variance tradeoff
- [ ] Implement A2C on LunarLander; compare stability against REINFORCE-with-baseline

<a id="a5"></a>

## A5. Continuous Control (Robotics)

**Goal.** What changes with continuous actions, and why SAC's entropy
maximization tends to produce more robust policies.

**Reading.** [DDPG paper](https://arxiv.org/abs/1509.02971) ·
[TD3 paper](https://arxiv.org/abs/1802.09477) ·
[SAC paper](https://arxiv.org/abs/1801.01290)

**Practical lab.** [Practical_RL week09_policy_II](https://github.com/yandexdataschool/Practical_RL/tree/master/week09_policy_II)
(TRPO, PPO, DDPG for robot control)

- [ ] Implement DDPG on Pendulum-v1; note training instability
- [ ] Implement TD3 (twin critics, delayed updates, target smoothing); compare stability to DDPG
- [ ] Run/implement SAC on a harder continuous task; sweep the entropy-temperature coefficient
- [ ] Optional: add action/observation noise at eval time; compare degradation across DDPG/TD3/SAC

<a id="a6"></a>

## A6. Planning & Self-Play: MCTS / AlphaZero

**Goal.** How planning (search) and model-free learning combine — a useful
contrast to GRPO's rollout-averaging in B3.

**Reading.** [MCTS survey (Browne et al.)](https://doi.org/10.1109/TCIAIG.2012.2186810) ·
[AlphaGo Zero](https://doi.org/10.1038/nature24270) / [AlphaZero](https://arxiv.org/abs/1712.01815)

**Practical lab.** [Practical_RL week10_planning](https://github.com/yandexdataschool/Practical_RL/tree/master/week10_planning)
(MCTS for toy tasks) and [week05_explore](https://github.com/yandexdataschool/Practical_RL/tree/master/week05_explore)
(UCB for MCTS)

- [ ] Implement plain MCTS (UCT) on Tic-Tac-Toe/Connect Four; verify convergence to near-optimal play
- [ ] Add a learned policy/value net guiding search, trained via self-play; compare sample efficiency to plain MCTS
- [ ] Ablate the value network (random/uniform rollouts) to isolate search vs. learned-prior strength

<a id="a7"></a>

## A7. Multi-Agent RL (optional)

**Goal.** Why multi-agent RL breaks single-agent convergence guarantees, and why
self-play sidesteps rather than solves that.

**Reading.** [Independent Q-Learning (Tan 1993)](http://web.media.mit.edu/~cynthiab/Readings/tan-MAS-reinfLearn.pdf) ·
[MADDPG paper](https://arxiv.org/abs/1706.02275) ·
[OpenAI self-play blog](https://openai.com/index/competitive-self-play/)

**Practical lab.** [PettingZoo tutorials](https://pettingzoo.farama.org/tutorials/)
(Farama Foundation) — independent-learner and self-play training on standard
multi-agent environments; not covered by Practical_RL.

- [ ] Train two independent Q-learners in a simple competitive game; observe non-stationarity from each agent's view
- [ ] Try self-play (one agent playing both sides); compare policy quality/stability to independent learners

<a id="a8"></a>

## A8. Imitation Learning & Inverse RL (optional)

**Goal.** The difference between copying actions (imitation) and recovering the
objective behind them (inverse RL) — relevant background for any
reward-model-adjacent work in Part B.

**Reading / lab.** [Practical_RL week10_planning](https://github.com/yandexdataschool/Practical_RL/tree/master/week10_planning)
and [yet_another_week](https://github.com/yandexdataschool/Practical_RL/tree/master/yet_another_week)
(imitation learning and inverse RL). This phase is lab-first: the course treats
it as bonus material beyond the core syllabus.

- [ ] Implement behavioral cloning on a small control task (e.g. CartPole) from a handful of expert demonstrations; observe compounding error away from the expert's state distribution
- [ ] Implement a simple inverse RL method (e.g. max-entropy IRL) to recover a reward function from the same demonstrations, then run Q-learning/PPO against the recovered reward
- [ ] Compare the behavioral-cloning policy against the IRL-then-RL policy on states the expert demonstrations never visited

---

# Part B — LLM Post-Training RL

This is where you want a vast.ai (or RunPod) GPU pod. Small HF model (0.5–3B)
via TRL for fast iteration.

<a id="b1"></a>

## B1. Foundations: REINFORCE → PPO

**Reading.** [Learning to Summarize w/ Human Feedback](https://arxiv.org/abs/2009.01325) ·
[InstructGPT](https://arxiv.org/abs/2203.02155) ·
[PPO paper](https://arxiv.org/abs/1707.06347)

**Practical lab.** [nlp_course week07_finetuning](https://github.com/yandexdataschool/nlp_course/tree/2025/week07_finetuning)
(PEFT & RLHF)

- [ ] Reuse your A4 REINFORCE implementation on a language-modeling reward; observe reward hacking in a generative action space
- [ ] Train full PPO (TRL PPOTrainer); sweep clip range and KL coefficient; plot reward vs. KL-to-reference
- [ ] Set the KL penalty near zero and document the resulting failure mode

<a id="b2"></a>

## B2. Direct Alignment: DPO / IPO

**Reading.** [DPO paper](https://arxiv.org/abs/2305.18290) ·
[IPO paper](https://arxiv.org/abs/2310.12036)

**Practical lab.** [TRL DPOTrainer docs](https://huggingface.co/docs/trl/dpo_trainer)
(supports both DPO and IPO loss via one config flag)

- [ ] Build/synthesize a preference dataset; train DPO; compare against the best PPO run
- [ ] Overfit DPO on a small preference set; show chosen-logprob collapse and KL blowup
- [ ] Swap in IPO's bounded objective; compare stability

<a id="b3"></a>

## B3. RLVR & GRPO

**Reading.** [DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300) ·
[DeepSeek-R1](https://arxiv.org/abs/2501.12948) ·
[RL-for-LLMs survey](https://arxiv.org/abs/2509.08827) ·
[first-principles GRPO derivation](https://arxiv.org/abs/2606.16733)

**Practical lab.** [HuggingFace Open R1](https://github.com/huggingface/open-r1)
(full open reproduction of DeepSeek-R1's GRPO pipeline) and
[TRL GRPOTrainer docs](https://huggingface.co/docs/trl/grpo_trainer)

- [ ] Implement GRPO from the derivation yourself — group rollouts, group-relative advantages, no value net
- [ ] Cross-check against TRL's GRPOTrainer
- [ ] Sweep group size; measure advantage-estimate variance
- [ ] Run RLVR against a gameable verifier and document reward hacking

<a id="b4"></a>

## B4. Reward Hacking & Overoptimization

**Reading.** [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565) ·
[Scaling Laws for RM Overoptimization](https://arxiv.org/abs/2210.10760)

**Practical lab.** [RewardBench](https://github.com/allenai/reward-bench) (AI2) —
evaluate your own trained reward model against a standard suite before
deliberately overoptimizing against it

- [ ] Overoptimize PPO against a learned RM; plot proxy reward vs. held-out gold metric
- [ ] Repeat with GRPO on a verifiable reward; compare hackability
- [ ] Try one mitigation (RM ensembling, tighter KL); quantify the recovered budget

<a id="b5"></a>

## B5. On-Policy Distillation

**Reading.** [MiniLLM](https://arxiv.org/abs/2306.08543) ·
[GKD](https://arxiv.org/abs/2306.13649) ·
[2026 OPD survey](https://arxiv.org/abs/2604.00626)

**Practical lab.** [nlp_course week08_efficiency](https://github.com/yandexdataschool/nlp_course/tree/2025/week08_efficiency)
(quantization, distillation, pruning, speculative decoding)

- [ ] Offline-KD baseline: distill the best B3 model into a smaller student
- [ ] GKD-style on-policy distillation: student rollouts, teacher relabels
- [ ] Run both students through your quantized inference path; compare robustness

<a id="b6"></a>

## B6. Unifying View + Systems Port

**Reading.** [2026 unified off-policy/on-policy post-training survey](https://arxiv.org/abs/2604.07941) · optional

**Practical lab.** [TRL source](https://github.com/huggingface/trl) — one codebase
with PPO, DPO, GRPO, and GKD implementations side by side; reading the shared
abstractions is a good proxy for the survey's unifying framework

- [ ] Map B1–B5 onto the survey's unifying framework
- [ ] Port GRPO's core update into Rust against your Candle fork — a comprehension check, not a throughput play

---

# Part C — Quant Research: Market-Making Parameter RL

hftbacktest as the environment (Rust-backed, replays real order-book data). CPU
is enough for calibration; a small GPU helps once you are running many parallel
episodes for PPO/GRPO-style rollouts.

<a id="c1"></a>

## C1. Baseline: Calibrated GLFT (no RL)

**Goal.** This fixed-calibration run is the benchmark every RL variant below has
to beat — on the same held-out days, not just the training days.

**Reading.** [Guéant–Lehalle–Fernandez-Tapia paper](https://arxiv.org/abs/1605.01862) ·
your GLFT + grid-trading notes

**Practical lab.** [hftbacktest's official GLFT + grid-trading notebook](https://github.com/nkaz001/hftbacktest/blob/master/examples/GLFT%20Market%20Making%20Model%20and%20Grid%20Trading.ipynb)
— likely the direct source of your own notes; run it as-is before modifying anything

- [ ] Calibrate A, k (order-arrival intensity) and σ (volatility) from market data; derive c1, c2 and reproduce the GLFT half-spread/skew formulas
- [ ] Run the calibrated GLFT grid market maker in hftbacktest on the sample data; reproduce the reported Sharpe/Sortino/MaxPositionValue stats
- [ ] Reproduce the simplified non-parametric HF-grid variant (percentile-based intensity, micro-price) and compare against the parametric GLFT baseline

<a id="c2"></a>

## C2. Framing Market Making as an RL Problem

**Goal.** The key design choice: RL adjusts the coefficients of a known-good
analytic model rather than replacing the model outright — keeps the policy
auditable.

**Reading.** Your order-book-imbalance, basis, and APT alpha notes.

**Practical lab.** [Gymnasium custom environment tutorial](https://gymnasium.farama.org/introduction/create_custom_env/)
— the standard pattern for wrapping hftbacktest as a `step()`/`reset()` Gym-style env

- [ ] Define state: inventory q, short-horizon volatility estimate, order-book imbalance / micro-price features, basis & APT alpha signals
- [ ] Define the action space as the GLFT coefficient set itself — γ, c1, and a half-spread/skew scaling term — rather than raw quote prices, so the policy stays interpretable inside the existing formula
- [ ] Define reward as mark-to-market PnL minus an inventory-risk penalty (e.g. −η·q²), mirroring the risk term the GLFT model already encodes analytically
- [ ] Wrap hftbacktest as a Gym-style environment: one episode per trading session, one step per grid-refresh interval

<a id="c3"></a>

## C3. Continuous Control on the Coefficient Set

**Goal.** A real answer to whether learned, state-conditioned coefficients beat a
well-calibrated fixed formula, and which coefficients are worth learning at all.

**Reading.** Reuse your A5 DDPG/TD3/SAC implementations.

**Practical lab.** [Stable-Baselines3 TD3](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html)
and [SAC docs](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
— battle-tested reference implementations to validate your own against

- [ ] Apply DDPG/TD3/SAC to the C2 environment: policy outputs (γ, c1, half-spread-scale) per step, conditioned on state
- [ ] Compare against the C1 fixed-calibration baseline on held-out days: Sharpe, Sortino, MaxPositionValue, ReturnOverMDD
- [ ] Ablate: fix γ at its calibrated value and let RL learn only c1/skew, vs. letting RL learn all three jointly — compare sample efficiency and stability

<a id="c4"></a>

## C4. PPO/GRPO for Parameter Search

**Goal.** The direct bridge between the LLM post-training RL work and this quant
research — same algorithms, a much smaller and more interpretable action space.

**Reading.** Reuse your B1 PPO and B3 GRPO implementations.

**Practical lab.** [Stable-Baselines3 PPO docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
for the low-dimensional continuous-control side; no off-the-shelf library does
GRPO outside LM tooling, so the group-relative-advantage update here is genuinely
from-scratch code

- [ ] Since the action here is a low-dimensional continuous vector, compare PPO (B1) directly against a GRPO-style approach: sample K coefficient sets per episode, rank by realized PnL, apply a group-relative-advantage update as in B3
- [ ] Check whether GRPO's group-baseline idea (no value network) transfers outside language modeling to this much lower-dimensional control problem
- [ ] Apply the B4 reward-hacking lens: check whether RL-tuned coefficients overfit to microstructure quirks of the training days (e.g. only widening spread around a few known volatility spikes) rather than learning a generalizable policy — validate strictly out-of-sample

<a id="c5"></a>

## C5. Multi-Signal Extension

**Goal.** Tests whether RL buys you anything over manually engineering the alpha
signal into the analytic formula.

**Reading.** Your basis and APT alpha notes.

**Practical lab.** [hftbacktest examples directory](https://github.com/nkaz001/hftbacktest/tree/master/examples)
— includes the official "Market Making with Alpha" notebooks (imbalance, basis,
APT) your notes are based on

- [ ] Fold the basis/APT alpha signals into the state (or into a directly-predicted fair-price adjustment) and re-run C3/C4
- [ ] Check whether the RL-tuned coefficients learn to widen/narrow skew in response to the alpha signal automatically, versus the manual extension a fixed-form model would need
