# Companion repositories

Seven repositories back the ladder's "Practical lab" entries and serve as
reference implementations. Six are forks under `davidmiheev/`; hftbacktest is
tracked upstream (no fork yet — Part C uses it read-only for now).

Clone them all, pinned to the commits below:

```bash
make companions          # -> ./companions/<name>, each checked out at its pin
```

`companions/` is gitignored — the pins in `scripts/clone_companions.sh` are the
committed record, not the checkouts themselves.

## Pins

| repo | upstream | pin | tracks | used by |
| --- | --- | --- | --- | --- |
| [Practical_RL](https://github.com/yandexdataschool/Practical_RL) | yandexdataschool | `6f7fa8bc` | `master` | A1–A6, A8 |
| [nlp_course](https://github.com/yandexdataschool/nlp_course) | yandexdataschool | `5cbabc4d` | `2025` | B1, B5 |
| [trl](https://github.com/huggingface/trl) | huggingface | `ac518401` | `main` | B1–B3, B5, B6 |
| [open-r1](https://github.com/huggingface/open-r1) | huggingface | `1416fa0c` | `main` | B3 |
| [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) | DLR-RM | `3246f506` | `master` | A3–A5, C3, C4 |
| [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | Farama-Foundation | `c956926f` | `main` | A1–A8, C2 |
| [hftbacktest](https://github.com/nkaz001/hftbacktest) | nkaz001 | `5f3ec40b` | `master` | C1–C5 |

## What each is for

**Practical_RL** — the Part A labs, one week per phase:

| phase | week directory |
| --- | --- |
| A1 | `week05_explore` (bandits, UCB, Thompson sampling) |
| A2 | `week02_value_based`, `week03_model_free` |
| A3 | `week04_approx_rl` |
| A4 | `week06_policy_based` |
| A5 | `week09_policy_II` (TRPO, PPO, DDPG) |
| A6 | `week10_planning`, `week05_explore` |
| A8 | `week10_planning`, `yet_another_week` |

A7 has no Practical_RL week — use the [PettingZoo tutorials](https://pettingzoo.farama.org/tutorials/)
instead.

**nlp_course** — Part B labs. Note the default branch is `2025`, not `main`:
`week07_finetuning` (PEFT & RLHF) for B1, `week08_efficiency` (quantization,
distillation, pruning, speculative decoding) for B5.

**trl** — the trainers you cross-check your own implementations against:
`PPOTrainer` (B1), `DPOTrainer` with the IPO loss flag (B2), `GRPOTrainer` (B3),
`GKDTrainer` (B5). For B6, reading PPO/DPO/GRPO/GKD side by side in one codebase
is the practical substitute for the unifying survey. Its `CLAUDE.md`/`AGENTS.md`
is worth reading before touching the fork: trainers there are deliberately
self-contained and duplicated, and consistency across copies is mandatory.

**open-r1** — HuggingFace's full open reproduction of DeepSeek-R1's GRPO
pipeline. For B3, use it as the "does my GRPO look sane at scale" reference:
`recipes/` holds the training configs, `src/open_r1/` the GRPO entry points.

**stable-baselines3** — battle-tested DQN/TD3/SAC/PPO. Not a substitute for
writing your own (A3–A5 explicitly ask for from-scratch implementations), but the
right thing to diff against when yours underperforms, and the reference for C3/C4
where the point is the environment, not the algorithm.

**Gymnasium** — the environment API everything else speaks. Its
[custom-environment tutorial](https://gymnasium.farama.org/introduction/create_custom_env/)
is the pattern C2 follows to wrap hftbacktest.

**hftbacktest** — the Part C environment: a Rust-backed backtester that replays
real order-book data, with a Python binding (`pip install hftbacktest`, package
version 2.4.x at the pin). The notebooks in `examples/` are the Part C labs:

| notebook | phase |
| --- | --- |
| `GLFT Market Making Model and Grid Trading.ipynb` | C1 (the baseline to reproduce first) |
| `High-Frequency Grid Trading - Simplified from GLFT.ipynb` | C1 (non-parametric variant) |
| `Market Making with Alpha - Order Book Imbalance.ipynb` | C2 state features, C5 |
| `Market Making with Alpha - Basis.ipynb` | C5 |
| `Market Making with Alpha - APT.ipynb` | C5 |
| `Data Preparation.ipynb` | prerequisite for all of Part C |
| `Impact of Order Latency.ipynb`, `Probability Queue Models.ipynb` | fill-model realism; read before trusting any C3/C4 result |

Also useful: `docs/data.rst` (the 8-field normalized event format hftbacktest
digests) and `collector/` (feed collection). Data itself is **not** in the repo —
see [`data.md`](data.md).

## Access note

The six forks are attached to this workspace with push access on branch
`claude/rl-ladder-setup-docs-d43b31`. hftbacktest is a public upstream clone:
readable and fetchable, but not pushable from here. If Part C ends up needing
patches to hftbacktest itself, fork it first.
