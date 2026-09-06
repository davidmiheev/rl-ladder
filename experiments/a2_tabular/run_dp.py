"""A2 item 1: value iteration and policy iteration on a GridWorld.

Two independent checks that the answer is the true optimum, not just a fixed
point of the code:

1. Deterministic grid, `step_reward = -1`, `gamma = 1`: the optimal value of a
   state is exactly minus its BFS distance to the goal.
2. Stochastic grid (slip): VI and PI must agree on both values and policy, and
   the Bellman optimality residual must be ~0.
"""

from __future__ import annotations

import argparse

import numpy as np
from gridworld import SIMPLE_GRID, GridWorld
from mdp import policy_iteration, value_iteration

from rl_ladder import Run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slip", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()

    run = Run(phase="a2", name="dp", config=vars(args))
    summary = {}

    # --- check 1: deterministic grid against BFS ------------------------------
    grid = GridWorld(SIMPLE_GRID)
    v_det, policy_det, vi_iters = value_iteration(grid.to_mdp(), gamma=1.0)
    distance = grid.shortest_paths()
    reachable = np.isfinite(distance) & ~grid.terminal
    matches_bfs = bool(np.allclose(v_det[reachable], -distance[reachable]))

    print(f"deterministic, gamma=1: VI converged in {vi_iters} sweeps")
    print(f"  V* == -BFS distance : {matches_bfs}")
    print(grid.render_policy(policy_det))
    summary["deterministic"] = {"vi_iterations": vi_iters, "matches_bfs": matches_bfs}

    # --- check 2: stochastic grid, VI vs PI -----------------------------------
    slippery = GridWorld(SIMPLE_GRID, slip=args.slip)
    mdp = slippery.to_mdp()
    v_vi, policy_vi, vi_iters = value_iteration(mdp, args.gamma)
    v_pi, policy_pi, pi_iters = policy_iteration(mdp, args.gamma)

    value_gap = float(np.abs(v_vi - v_pi).max())
    policies_agree = bool(np.array_equal(policy_vi, policy_pi))
    residual = mdp.bellman_residual(v_vi, args.gamma)

    print(f"\nslip={args.slip}, gamma={args.gamma}")
    print(f"  VI {vi_iters} sweeps | PI {pi_iters} improvement steps")
    print(f"  max |V_VI - V_PI|   : {value_gap:.2e}")
    print(f"  policies identical  : {policies_agree}")
    print(f"  Bellman residual    : {residual:.2e}")
    print(slippery.render_policy(policy_vi))
    summary["stochastic"] = {
        "vi_iterations": vi_iters,
        "pi_iterations": pi_iters,
        "value_gap": value_gap,
        "policies_agree": policies_agree,
        "bellman_residual": residual,
    }

    run.log(step=0, **summary["deterministic"])
    run.log(step=1, **summary["stochastic"])
    print(f"\nrun -> {run.finish(summary)}")


if __name__ == "__main__":
    main()
