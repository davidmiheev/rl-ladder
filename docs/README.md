# Docs

Start here.

| doc | what it covers |
| --- | --- |
| [`ladder.md`](ladder.md) | **The curriculum.** 19 phases, 59 checklist items, reading and labs. Canonical — tick boxes here. |
| [`setup.md`](setup.md) | Environments, per-part extras, verifying the install |
| [`compute.md`](compute.md) | What hardware each part needs; GPU pod sizing for Part B |
| [`free-compute.md`](free-compute.md) | Free GPU/TPU tiers (Colab, Kaggle, TRC): what fits, T4 config, where they break |
| [`experiment-protocol.md`](experiment-protocol.md) | Seeds, run logging, plots, what counts as "done" |
| [`repo-layout.md`](repo-layout.md) | Where code goes and why the shared library stays tiny |
| [`companion-repos.md`](companion-repos.md) | The 7 lab/reference repos, their pins, and which phase uses which |
| [`data.md`](data.md) | Part C market data: format, sources, the train/held-out split rule |
| [`assets/rl-ladder-tracker.html`](assets/rl-ladder-tracker.html) | The original HTML tracker, archived verbatim. Read-only here — progress is tracked in `ladder.md`. |

## The short version

Three parts. **A** builds RL intuition on classical, games and robotics problems
(CPU). **B** applies it to LLM post-training with TRL on a small model (GPU pod).
**C** applies the same algorithms to tuning the GLFT market-making coefficient
set on replayed order-book data (CPU, small GPU optional).

Work A roughly in order — A7 and A8 are side branches. B needs A4's policy
gradients; B3 is easier after A6. C can start any time after A5, but C4 reuses
B1's PPO and B3's GRPO directly, so after touching B is the natural time.

## The one rule worth repeating

Each phase's checklist is ticked in `ladder.md` only when the phase README in
`experiments/<phase>/` carries the run that backs it — multi-seed, with the
finding written down, including when the finding is "no effect". Several items
are designed to fail; the documented failure is the deliverable.
