# A7 — Multi-Agent RL: task, goal, and how it was solved

Why every convergence guarantee from A2 onward assumed something that stops
being true the moment a second learner enters the environment. The lab notebook
is [`experiments/a7_multi_agent/README.md`](../../experiments/a7_multi_agent/README.md).

---

## 1. The task

[`ladder.md`](../ladder.md#a7) states two items:

1. Two independent Q-learners in a competitive game; observe non-stationarity
   from each agent's view.
2. Self-play (one agent playing both sides); compare policy quality and
   stability to independent learners.

## 2. The goal: what exactly breaks

A2 proved Q-learning converges to $Q^*$ given Robbins–Monro step sizes,
infinite visitation, and — silently — a **fixed MDP**. The stochastic
approximation argument needs the operator being iterated to stand still.

With two learners it does not. From agent 1's perspective the transition kernel
is

$$P_1(s' \mid s, a) = \sum_{a_2} \pi_2(a_2 \mid s)\, P(s' \mid s, a, a_2)$$

which contains $\pi_2$. As agent 2 learns, $P_1$ changes, so agent 1 is solving
a **different MDP at every moment**. There is no fixed point for it to converge
to, and no contraction argument to make.

Worse, the pair need not be heading anywhere at all. The natural solution
concept becomes a Nash equilibrium rather than an optimum, and independent
Q-learning has no mechanism aiming at one: each agent computes a best response
to the other's *current* policy, and best-response dynamics can cycle
indefinitely. Convergence of the pair is not merely unproven, it is not
generally expected.

**Why this phase uses a solvable game.** In a two-player game the obvious
metric — win rate against the other agent — is nearly uninformative: it says the
two are mismatched, not whether either is good. Two agents can trade wins
forever while both play badly. Tic-Tac-Toe is small enough to solve exactly
(4520 reachable non-terminal positions), which allows an absolute measure:

$$\text{exploitability}(\pi) = \max_{\pi'} V^{\pi', \pi} \;-\; V^{*}$$

the value a *perfect* opponent extracts. Zero means unexploitable. This is the
standard measure in game-solving work for exactly this reason, and it is
computable here rather than estimated.

## 3. Self-play, and what it does and does not fix

Self-play uses one policy for both seats. Two things follow immediately:

- **The asymmetry disappears.** There is no possibility of one seat outrunning
  the other, because there is only one thing to improve.
- **The opponent's strength tracks yours automatically**, which is the
  curriculum argument usually given for it.

Neither of these makes the problem stationary. The shared policy still
bootstraps from values produced by an earlier version of itself, so the target
still moves — self-play changes *who* the moving target is, not that there is
one. It is a symmetrisation, not a solution, and the honest test is whether the
result is close to equilibrium, which exploitability answers directly.

There is also a third effect, rarely stated and — as it turns out here — large:
**a shared table receives updates from both seats**, so at equal episode counts
self-play collects roughly twice the data per parameter as two independent
tables. Any comparison that does not separate this from the curriculum effect is
attributing to strategy what is partly just sample count.

## 4. What was done

Tabular Q-learning on Tic-Tac-Toe, three arms — independent, self-play, and
self-play with random starting positions. The third exists as a **control for
coverage**: it improves the state distribution by a mechanism that has nothing
to do with opponent quality, so if quality tracks it, the curriculum story is
incomplete.

Measurement decisions:

- **Exploitability is exact**, by dynamic programming over the game tree, and
  validated on known cases (a perfect policy scores $0.000$; uniform random
  $+0.925$ first, $+0.995$ second).
- **Graded metrics alongside it** — optimal-move rate over all 4520 positions,
  and regret against perfect play from random starts — because exploitability
  is all-or-nothing on a deterministic policy and saturates.
- **Regret rather than raw value** from random starts, since some starting
  positions are simply better than others and the raw number would measure the
  start distribution.
- **Non-stationarity as a cross-checkpoint matrix**: freeze seat-1 policies from
  every checkpoint, play them against seat-2 policies from every checkpoint,
  compute the value exactly. If each agent's environment were stationary, a row
  would be flat.

## 5. Results

Tables in the [phase README](../../experiments/a7_multi_agent/README.md).

**Non-stationarity, measured.** Every seat-1 checkpoint scores better against
its contemporary opponent than against later ones — mean drop $+0.465$, positive
in 6 of 7 rows. The 18k policy goes from $+0.80$ against the opponent it trained
alongside to $-0.10$ against later ones, without changing.

The row-wise form of the comparison is doing real work. The raw diagonal
($+0.575$) versus off-diagonal ($+0.254$) is confounded: column means fall from
$+0.79$ to $+0.13$ across training, so later opponents are genuinely stronger.
Comparing each policy to *its own* future opponents controls for that, and the
effect survives — which is the difference between "opponents got better" and
"this policy was tuned to a specific opponent".

**Self-play wins, substantially for a reason that is not the usual story.**
Optimal-move rate $0.936$ against $0.773/0.749$; regret $0.099$ against $0.383$.
But state-action coverage is $0.684$ against $0.320$–$0.364$ — almost exactly
double, which is what a shared table updated from both seats should produce. The
random-starts arm confirms the direction: it raises coverage to $0.813$ by a
mechanism unrelated to opponent quality, and quality follows ($0.972$, regret
$0.038$). Across all three arms, quality tracks coverage.

So self-play's advantage here decomposes into at least three parts — twice the
updates, a symmetric curriculum, and no seat specialisation — and the experiment
separates the first from the others rather than reporting a single number.

**Independent learners are seat-specialists.** The seat-2 table, made to play
first, scores $+0.92$ exploitability against $+0.00$ for the seat-1 table. Each
learned one side of the game only. Self-play cannot have this failure at all,
which is a genuine advantage independent of data volume.

**And nothing reaches equilibrium.** Second-player exploitability is $+1.00$ for
every arm, including the best one at $97.2\%$ optimal moves. The reason is worth
stating precisely because it is not the obvious one: it is not a coverage
failure — exploitability stayed pinned at $1.00$ while coverage rose
$0.509 \to 0.947$ in a separate sweep. Walking the exploiter's best line to the
first mistake shows that at the critical position the never-tried actions sit at
their zero initialisation while every tried action is *negative*, because those
moves lose. **Zero-initialised $Q$ is optimistic in a zero-sum game**, where
nearly every true value is $\le 0$, so a greedy policy is systematically drawn
toward whatever it has not tried — exactly where a best-response opponent
steers. Exploitability demands correctness along the whole line, so one such
state is fatal.

That is the sharpest statement of this phase's lesson: a policy that is optimal
in $97.2\%$ of positions loses *every* game to an opponent allowed to choose the
line. Average-case competence and worst-case exploitability are different
quantities, and in adversarial settings only the second one is the score.

## 6. What this sets up

- **B4's reward hacking** is the same structure with the roles renamed: a policy
  optimised against a proxy, then evaluated by something that searches for where
  the proxy is wrong. The exploiter here is a best response; there it is the
  optimiser itself finding the reward model's blind spots.
- **A8's inverse RL** shares the identifiability problem — many reward functions
  explain the same behaviour, just as many policies share the same win rate
  against a fixed opponent while differing wildly in exploitability.

## 7. Where these items come from

| item | primary source | why it was picked |
| --- | --- | --- |
| 1 — independent Q-learners | [Tan 1993](http://web.media.mit.edu/~cynthiab/Readings/tan-MAS-reinfLearn.pdf), the standard reference for independent learners in a multi-agent setting | It is the minimal way to break A2's stationarity assumption — change nothing about the algorithm, add a second learner — so the failure is attributable to the setting rather than to any design choice. |
| 2 — self-play | [OpenAI's competitive self-play work](https://openai.com/index/competitive-self-play/); [MADDPG](https://arxiv.org/abs/1706.02275) is the ladder's reading for the centralised-critic alternative | Self-play is the standard response, and the interesting question is not whether it helps but *why* — which requires separating curriculum from sample count, and asking whether it reaches equilibrium at all. |

The **lab** pairing is the ladder's suggested PettingZoo tutorials. They are not
used: an exactly solvable game buys absolute measurement, which in a two-player
setting is worth more than environment variety, and the reasoning is recorded in
the phase README.

The **goal statement** in section 2 is quoted from
[`ladder.md`](../ladder.md#a7). Section 3's decomposition of self-play's
advantage into three separable effects, and the random-starts control that tests
it, are this repo's rather than claims from the sources.
