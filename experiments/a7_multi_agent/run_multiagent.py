"""A7 items 1 and 2: independent learners vs self-play, measured exactly.

Tic-Tac-Toe is solvable, so nothing here is a win rate against an arbitrary
opponent — every number is computed against the exact optimum:

* **optimal-move rate** over all 4520 reachable positions (graded),
* **mean value against a perfect opponent** from random starts (graded),
* **exploitability** from the empty board (all-or-nothing, and its saturation
  is itself a finding),
* **state-action coverage**, since a Q-value never updated is still at its
  optimistic zero initialisation.

Item 1's non-stationarity is measured as a cross-checkpoint matrix: how a
frozen policy from iteration t fares against opponents from other iterations.
If the environment each agent faced were stationary, that row would be flat.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool

import game
import numpy as np
from learners import match_value, reachable_positions, train

from rl_ladder import Run

ARMS = {
    "independent Q-learners": {"shared": False},
    "self-play (shared table)": {"shared": True},
    "self-play + random starts": {"shared": True, "random_starts": 0.5},
}


def perfect_policy(cache):
    def policy(board):
        probs = np.zeros(game.N_ACTIONS)
        best = game.solve(board, cache)[1]
        for m in best:
            probs[m] = 1.0 / len(best)
        return probs
    return policy


def evaluate_table(table, positions, cache, rng, n_starts: int = 300) -> dict:
    """Everything measurable exactly about one learned table."""
    policy = table.greedy_policy()

    correct = 0
    for board in positions:
        best = game.solve(board, cache)[1]
        if int(policy(board).argmax()) in best:
            correct += 1

    tried = total = 0
    for board in positions:
        q = table.q.get(board.tobytes())
        moves = game.legal_moves(board)
        total += len(moves)
        if q is not None:
            tried += int((q[moves] != 0).sum())

    # Graded stand-in for exploitability: the exact value a perfect opponent
    # extracts, averaged over random starts rather than only the empty board,
    # so a single bad line does not saturate the measure. Reported as *regret*
    # against what a perfect policy would score from the same position, since
    # the raw value depends on how favourable the random starts happen to be.
    idx = rng.choice(len(positions), size=min(n_starts, len(positions)), replace=False)
    values = []
    for i in idx:
        board = positions[i]
        local: dict = {}

        def value(b, pi_turn, local=local):
            key = (b.tobytes(), pi_turn)
            if key in local:
                return local[key]
            result = game.outcome(b)
            if result is not None:
                local[key] = -result if pi_turn else result
                return local[key]
            moves = game.legal_moves(b)
            if pi_turn:
                probs = policy(b)[moves]
                probs = probs / probs.sum() if probs.sum() > 0 else np.full(len(moves), 1 / len(moves))
                out = float(sum(p * value(game.play(b, int(m)), False)
                                for m, p in zip(moves, probs, strict=True)))
            else:
                out = max(value(game.play(b, int(m)), True) for m in moves)
            local[key] = out
            return out

        # regret = what the exploiter gets against us, minus what it would get
        # against perfect play from the same start. Zero iff we play optimally.
        values.append(value(board, True) + game.solve(board, cache)[0])

    return {
        "optimal_move_rate": correct / len(positions),
        "state_action_coverage": tried / total,
        "mean_regret_vs_perfect": float(np.mean(values)),
        "exploitability_first": game.exploitability(policy, True, {}),
        "exploitability_second": game.exploitability(policy, False, {}),
    }


def one_run(job):
    label, overrides, seed, episodes, checkpoint_every = job
    tables, checkpoints = train(episodes=episodes, seed=seed,
                                checkpoint_every=checkpoint_every, **overrides)
    cache: dict = {}
    positions = reachable_positions()
    rng = np.random.default_rng(seed)
    # Seat 0 always moves first, seat 1 always second, so each table is scored
    # in the seat it actually trained in.
    first = evaluate_table(tables[0], positions, cache, rng)
    second = evaluate_table(tables[1], positions, cache, rng)
    return label, seed, first, second, checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=150_000)
    parser.add_argument("--checkpoints", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    run = Run(phase="a7", name="multiagent", config=vars(args))
    every = args.episodes // args.checkpoints
    jobs = [(label, o, seed, args.episodes, every)
            for label, o in ARMS.items() for seed in range(args.seeds)]
    with Pool(args.workers) as pool:
        results = pool.map(one_run, jobs)

    summary = {}
    for label in ARMS:
        rows = [r for r in results if r[0] == label]
        for seat, name in ((2, "seat 1 (moves first)"), (3, "seat 2 (moves second)")):
            metrics = [r[seat] for r in rows]
            agg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
            agg["optimal_move_rate_std"] = float(np.std([m["optimal_move_rate"] for m in metrics]))
            summary[f"{label} | {name}"] = agg
            run.log(step=len(summary), arm=label, seat=name, **agg)
            print(f"{label:26s} {name:22s} optimal-move {agg['optimal_move_rate']:.3f}"
                  f" +/- {agg['optimal_move_rate_std']:.3f} | (s,a) cov {agg['state_action_coverage']:.3f}"
                  f" | regret {agg['mean_regret_vs_perfect']:.3f}"
                  f" | exploitability {agg['exploitability_first']:+.2f}/{agg['exploitability_second']:+.2f}")

    # Item 1: is the environment each agent faces stationary? Freeze seat-1
    # policies from every checkpoint and play them against seat-2 policies from
    # every checkpoint. A flat row means the opponent stopped mattering.
    print("\n=== non-stationarity: independent learners, value to seat 1 ===")
    all_checkpoints = [r[4] for r in results if r[0] == "independent Q-learners"]
    episodes = [c[0] for c in all_checkpoints[0]]
    matrices = []
    for checkpoints in all_checkpoints:
        m = np.zeros((len(checkpoints), len(checkpoints)))
        for i, (_, first_table, _) in enumerate(checkpoints):
            for j, (_, _, second_table) in enumerate(checkpoints):
                m[i, j] = match_value(first_table.greedy_policy(), second_table.greedy_policy(), {})
        matrices.append(m)
    matrix = np.mean(matrices, axis=0)
    header = "        " + "".join(f"{e // 1000:>7d}k" for e in episodes)
    print(header)
    for i, e in enumerate(episodes):
        print(f"{e // 1000:>6d}k " + "".join(f"{matrix[i, j]:+7.2f} " for j in range(len(episodes))))
    diag = float(np.mean(np.diag(matrix)))
    off = float((matrix.sum() - np.trace(matrix)) / (matrix.size - len(matrix)))
    print(f"\ncontemporary (diagonal) mean {diag:+.3f} | cross-checkpoint mean {off:+.3f}")
    summary["nonstationarity"] = {"episodes": episodes, "matrix": matrix.tolist(),
                                  "diagonal_mean": diag, "offdiagonal_mean": off}
    run.log(step=999, diagonal_mean=diag, offdiagonal_mean=off)
    print(f"\nrun -> {run.finish(summary)}")


if __name__ == "__main__":
    main()
