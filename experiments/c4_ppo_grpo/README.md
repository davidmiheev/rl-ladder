# C4 — PPO/GRPO for Parameter Search

**Goal.** Whether GRPO's group-baseline idea transfers outside language modeling to a low-dimensional control problem.

**Prerequisites.** C3, plus B1 (PPO) and B3 (GRPO)
**Companion lab.** Stable-Baselines3 PPO docs; GRPO is from scratch here
**Compute.** CPU or small GPU

The checklist items, reading list and lab links for this phase live in
[`docs/ladder.md`](../../docs/ladder.md#c4) — that file is the single
source of truth for what "done" means. This README is the lab notebook: how the
code here is run, and what the runs actually showed.

Writeup: [`docs/writeups/c4-ppo-grpo.md`](../../docs/writeups/c4-ppo-grpo.md).

## How to run

```bash
# 0. (once) every pre-split day for ETHUSDC and XRPUSDC, for --train-set train20
HF_TOKEN=... python experiments/c4_ppo_grpo/fetch_train20.py

# 1. The grid: 18 points x 8 pairs on training sessions, then the choices held out
python experiments/c4_ppo_grpo/grid_search.py --kind grid --train-set train20 --shard 0/3
python experiments/c4_ppo_grpo/grid_search.py --kind select --train-set train20
python experiments/c4_ppo_grpo/grid_search.py --kind test --train-set train20

# 2. GRPO-style and PPO-style search (120 session replays each), then held out
python experiments/c4_ppo_grpo/param_search.py --method grpo --strategy obi --symbol ethusdc --seed 0
python experiments/c4_ppo_grpo/param_search.py --method ppo  --strategy obi --symbol ethusdc --seed 0
python experiments/c4_ppo_grpo/param_search.py --method grpo --strategy obi --symbol ethusdc \
    --seed 0 --fill-target 1500          # Lagrangian fill target
python experiments/c4_ppo_grpo/grid_search.py --kind search-test

# 3. RL with the trading/holding reward split (recipe v2-edge, C6's queue)
python experiments/c6_strategy_zoo/run_queue.py --kind trace
python experiments/c6_strategy_zoo/run_queue.py --kind train
python experiments/c6_strategy_zoo/run_queue.py --kind eval

# 4. Figures and the comparison tables
python experiments/c4_ppo_grpo/plot_c4.py
python experiments/c4_ppo_grpo/plot_c4.py --table
```

Everything is resumable: each queue skips runs that already have a
`summary.json` for the right sessions.

## Results

| run | config | metric | value | run dir |
| --- | --- | --- | --- | --- |
| grid, 20-day choice | floor × κ × skew, 8 pairs | held out, positive with real trading | OBI XRP only (+3.64%) | `runs/c4/*-test__*` |
| GRPO vs PPO | 120 replays, 2 seeds, glft/obi ETH | held-out median | glft −1.89 / −2.75, obi −2.91 / −2.54 | `runs/c4-search/` |
| fill target | GRPO, obi ETH, F = 1,500 / 3,000 | held out | worse at every target; λ 0.02–0.08 bp/fill | `runs/c4-search/grpo-F*` |
| RL v2-edge | holding PnL × 0.25, 5 seeds | held-out median vs v2 | glft −9.27 vs −6.30, obi −9.40 vs −8.00 | `runs/c6/*-v2-edge-*` |
| best overall | OBI XRPUSDC, C7 markout κ = 1 | held out / without 09-22 / days up | +4.91% / +1.38% / 9 of 11 | `runs/c7/obi-wide-xrpusdc-mk1__*` |

## Findings

- The markout rule is what matters in the grid; choices are identical from 10
  and from 20 ETH training days.
- GRPO's group baseline works (toy check) but does not beat a per-session
  critic here, and neither search beats the grid on OBI.
- Forcing volume costs money; so does a reward that discounts holding PnL,
  because adverse selection *is* holding PnL.
- XRP selection on four days was unreliable, and several positive XRP numbers
  are one crash session (2025-09-22) or no trading at all.
- Operational: a trace built inside a training job doubles its memory; build
  traces as their own jobs (`--kind trace`).
