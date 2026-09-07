# A3 — Value-Function Approximation: DQN

**Goal.** The "deadly triad" (bootstrapping + off-policy + function
approximation) and which fix addresses which part of it.

**Prerequisites.** A2
**Companion lab.** Practical_RL week04_approx_rl
**Compute.** CPU. ~70 s per 30k-step run; runs parallelise across processes.

Checklist items, reading and lab links live in
[`docs/ladder.md`](../../docs/ladder.md#a3). This README is the lab notebook; the
reasoning — why the table has to go, what each stabiliser attacks, and the
Jensen argument behind the max operator's bias — is in
[`docs/writeups/a3-dqn.md`](../../docs/writeups/a3-dqn.md).

## Code

| file | what it holds |
| --- | --- |
| `dqn.py` | `QNetwork` (optionally dueling), `ReplayBuffer`, and `train` with four switches |
| `run_ablation.py` | item 1 — replay and target network removed independently |
| `run_double.py` | item 2 — Double DQN, on stock and reward-noised CartPole |
| `run_dueling.py` | item 3 — dueling vs vanilla at equal environment steps |

One agent, four flags (`replay`, `target_network`, `double`, `dueling`). Every
arm of every item is the same code path, so no comparison can be contaminated by
an incidental implementation difference.

## How to run

```bash
cd experiments/a3_dqn
python run_ablation.py --seeds 5 --steps 30000    # ~13 min on 4 cores
python run_double.py   --seeds 5 --steps 90000    # ~40 min
python run_dueling.py  --seeds 5 --steps 30000    # ~7 min
```

## Reference points for CartPole-v1

Measured, not assumed — they are what makes the item 1 result readable:

| policy | return |
| --- | --- |
| random | 22.7 ± 11.4 |
| always push left | 9.4 |
| always push right | 9.2 |
| time limit (maximum) | 500 |

## Results

### Item 1 — ablating replay and the target network

`ablation__20260907-210256`, 5 seeds, 30k steps, greedy evaluation.
*Retention* is final ÷ peak: the signature of instability is a high peak
followed by collapse.

| arm | final eval return | peak | retention |
| --- | --- | --- | --- |
| DQN (replay + target) | **250.1 ± 151.2** | 469.4 | 0.56 |
| no replay | 80.0 ± 60.8 | 179.5 | 0.50 |
| no target net | 9.2 ± 0.2 | 9.2 | 1.00 |
| neither | 9.1 ± 0.2 | 9.1 | 1.00 |

![ablation](../../docs/assets/a3/ablation.png)

### Item 2 — Double DQN

_Running (90k steps); results to follow._

### Item 3 — dueling architecture

_Running; results to follow._

### Item 4 — Atari

Deliberately skipped: a scale demonstration rather than a new concept, and the
only item here that wants a GPU. See the writeup's provenance table.

## Findings

**Removing the target network is not a degradation, it is a collapse.** The
no-target arm scores 9.2 — and "always push right" scores exactly 9.2, while
random scores 22.7. The agent is not playing badly; its Q-function has
degenerated so that one action's value dominates in every state, making the
greedy policy constant. That is what divergence looks like from the outside, and
it is *worse than not learning at all*: a random policy would have scored 2.5×
higher.

**Removing replay is bad but not fatal** (80.0, against 250.1 with everything).
The no-replay agent does learn — it reaches 179 at its peak — it just learns
slowly and erratically, because each gradient step sees one correlated sample
and immediately forgets it. The ordering matters: **the two stabilisers are not
interchangeable**, and the one that prevents catastrophe is the one that
addresses bootstrapping, not the one that addresses data correlation.

**Even full DQN is unstable.** It peaks at 469 and ends at 250, retaining 56% of
its own best, with a final standard deviation of 151 across seeds — larger than
the mean-to-peak gap. Reporting only the final number would call this a 250-point
agent; reporting only the peak would call it a solved environment. Neither is
true, and the retention column exists so the summary cannot hide it.

**A note on what the ablation is.** The "no target net" arm is not a strawman: it
is the configuration of the 2013 NIPS workshop DQN, which had experience replay
but no target network. The gap between the two arms is the gap between that
paper and the 2015 Nature one.
