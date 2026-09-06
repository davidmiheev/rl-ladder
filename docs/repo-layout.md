# Repo layout

```
rl-ladder/
├── docs/
│   ├── README.md                docs index
│   ├── ladder.md                the curriculum — 19 phases, 59 items, canonical
│   ├── setup.md                 environments and per-part extras
│   ├── compute.md               what hardware each part needs, pod sizing
│   ├── experiment-protocol.md   seeds, runs, plots, what "done" means
│   ├── companion-repos.md       the 7 lab/reference repos and their pins
│   ├── data.md                  market data for Part C
│   └── assets/                  plots worth committing, plus the original
│                                 HTML tracker (rl-ladder-tracker.html)
├── src/rl_ladder/               shared plumbing only (~200 lines)
│   ├── seeding.py               seed_everything
│   ├── run.py                   Run: run dirs, config/env snapshot, JSONL metrics
│   └── plotting.py              multi-seed mean±std curves
├── experiments/
│   ├── _template/               conventions demo — copy this to start a phase
│   ├── a1_bandits/ … a8_imitation/
│   ├── b1_ppo/ … b6_unifying_view/
│   └── c1_glft_baseline/ … c5_alpha_signals/
├── scripts/clone_companions.sh  pinned clones of the lab repos
├── tests/test_smoke.py          plumbing tests, no torch, no gym
├── runs/                        generated, gitignored
├── companions/                  cloned labs, gitignored
├── data/                        market data, gitignored
├── Makefile
└── pyproject.toml
```

## Where things go

**`src/rl_ladder/` is not an RL library.** It holds seeding, run logging, and
plotting, and it should stay that way. A shared `BaseAgent` would defeat the
purpose: the ladder is about implementing algorithms, and each one is supposed to
be readable in isolation. If you find yourself wanting to factor a DQN out of
A3 so C3 can import it — copy it instead, and let the diff show what changed.

**One directory per phase.** The phase id prefix (`a3_`, `b1_`, `c4_`) keeps the
listing in ladder order and makes the mapping to
[`ladder.md`](ladder.md) unambiguous. Directory names use underscores so scripts
inside them stay importable.

**Phase READMEs are lab notebooks.** Goal, prerequisites, how to run, a results
table, and findings. The checklist items themselves live in `ladder.md` — one
source of truth, so a tick and a result never drift apart.

**Nothing generated is committed.** `runs/`, `companions/`, `data/`,
checkpoints and model weights are all gitignored. What survives is code, docs,
results rows, and any plot deliberately copied into `docs/assets/`.

## Naming

- Phase directories: `<phase-id>_<slug>` (`a5_continuous_control`).
- Run names: what the arm *is*, not when it ran — `dqn-no-replay`, `grpo-group8`,
  `glft-baseline-heldout`. Timestamps are appended automatically.
- Scripts: `train_<thing>.py`, `eval_<thing>.py`, `sweep_<param>.py`.
