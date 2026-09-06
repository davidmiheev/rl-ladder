# Compute

## What each part needs

| part | hardware | why |
| --- | --- | --- |
| A1–A2 | any laptop CPU | tabular; runs are seconds |
| A3–A4 | CPU | CartPole/LunarLander DQN and PG fit comfortably; a GPU barely helps at these network sizes |
| A5 | CPU, more cores helps | Pendulum is trivial; MuJoCo tasks are env-step-bound, so cores beat GPU |
| A6 | CPU, many cores | self-play rollouts parallelize; the net is tiny |
| A3 optional Atari, A7 | CPU fine, GPU pleasant | frame stacks make a small GPU worthwhile |
| B1–B5 | 1 GPU, 24–80 GB | 0.5–3B model + reference model + optimizer state |
| B6 | CPU | reading and a Rust port |
| C1–C2 | CPU | backtest replay is single-threaded Rust, I/O bound |
| C3–C5 | CPU, small GPU optional | the policy is tiny; the environment is the bottleneck until you run many parallel episodes |

The honest summary: **only Part B actually needs a rented GPU.** Everything else
runs on whatever you already have, and renting for Part A mostly buys you the
overhead of syncing code to a pod.

## Sizing Part B

Rough VRAM for full-parameter training with a reference model in memory
(PPO/GRPO hold both; DPO can share a frozen ref):

| model | LoRA/PEFT | full fine-tune |
| --- | --- | --- |
| 0.5B | 8–12 GB | 24 GB |
| 1.5B | 16–24 GB | 40 GB |
| 3B | 24–40 GB | 80 GB |

Start at **0.5B**. The point of Part B is watching KL-vs-reward tradeoffs and
reward hacking, and both show up just as clearly on a small model with a 10×
faster iteration loop. Move up only when a phenomenon plausibly needs capacity.

GRPO multiplies rollout cost by the group size — a group of 8 means 8× the
generation per update. Budget for that before sweeping group size in B3, and
prefer shorter completions over a bigger model when you run out of room.

## Pods

The ladder assumes vast.ai or RunPod:

- Pick a **persistent volume** and put `HF_HOME` on it. Re-downloading a 3B model
  and a dataset on every pod start wastes more money than the volume costs.
- Prefer **one long-lived pod** over many short ones for Part B: environment
  setup is the dominant fixed cost, and TRL + flash-attn installs are slow.
- `runs/` is gitignored, so anything you want to keep from a pod must be pushed
  as a phase-README results row (and any plot worth keeping copied into
  `docs/assets/`) before the pod dies.
- Sanity-check throughput before a long run: tokens/sec on a 100-step run tells
  you whether an 8-hour sweep is actually an 80-hour sweep.

## Cost discipline

Every Part B item is a question with a yes/no answer ("does the KL penalty at
zero break it?"). Run the smallest configuration that answers the question,
record it, and stop. The ladder's value is the documented findings, not the
checkpoints.
