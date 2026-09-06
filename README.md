# rl-ladder

A hands-on ladder through reinforcement learning, in three parts:

- **Part A — Classical, games & robotics RL.** Bandits → tabular → DQN → policy
  gradients → continuous control → MCTS/self-play, plus optional multi-agent and
  imitation/inverse-RL branches. CPU is enough.
- **Part B — LLM post-training RL.** REINFORCE → PPO → DPO/IPO → GRPO/RLVR,
  reward hacking, on-policy distillation, and a unifying view. TRL on a 0.5–3B
  model, on a rented GPU pod.
- **Part C — Market-making parameter RL.** The same algorithms applied to tuning
  the GLFT coefficient set on replayed order-book data in hftbacktest, with a
  calibrated no-RL baseline as the thing to beat.

19 phases, 59 checklist items. The curriculum is in
[`docs/ladder.md`](docs/ladder.md) — that is the single source of truth for what
is done and what is next.

## Quick start

```bash
uv venv --python 3.11 && source .venv/bin/activate
make install                                  # core deps, no torch
make test                                     # plumbing smoke tests
python experiments/_template/run_example.py   # writes a run dir + a plot
make companions                               # clone the 7 pinned lab repos
```

Then `make install-a` (or `-b` / `-c`) when you reach that part.
Full instructions: [`docs/setup.md`](docs/setup.md).

## Layout

```
docs/          the ladder, setup, protocol, companion pins   <- read these first
src/rl_ladder/ shared plumbing only: seeding, run logging, plotting
experiments/   one directory per phase; _template shows the conventions
runs/          generated, gitignored
```

`src/rl_ladder/` is intentionally ~200 lines. The point of the ladder is writing
the algorithms, so each phase is self-contained and readable top to bottom;
reuse across phases happens by copying, which keeps the diffs visible. See
[`docs/repo-layout.md`](docs/repo-layout.md).

## How a phase gets done

1. Copy `experiments/_template/` to `experiments/<phase>/`.
2. Implement the item; run it over ≥3 seeds with `rl_ladder.Run`.
3. Add a results row and a finding to the phase README.
4. Tick the box in [`docs/ladder.md`](docs/ladder.md).

Several items are designed to fail — the near-zero KL penalty in B1, the DPO
overfit in B2, the gameable verifier in B3. For those the documented failure is
the deliverable. See [`docs/experiment-protocol.md`](docs/experiment-protocol.md).

## Companion repos

Seven pinned repositories provide the labs and reference implementations:
Practical_RL and nlp_course (Yandex SDA course notebooks), TRL and open-r1
(Part B reference trainers), Stable-Baselines3 and Gymnasium (Part A/C
reference), hftbacktest (the Part C environment). `make companions` clones them
all at their pins — see [`docs/companion-repos.md`](docs/companion-repos.md).
