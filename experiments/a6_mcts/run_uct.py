"""A6 item 1: does plain UCT converge to optimal play?

Tic-Tac-Toe is solved exactly, so this is checkable rather than assertable. Two
measurements, because either alone is misleading:

1. **Move accuracy** on random reachable positions where optimal play is a
   *strict* subset of legal moves. Positions where everything draws (the empty
   board, for one) would inflate the score for free.
2. **Results against a perfect player**, as both first and second player.
   Perfect play in Tic-Tac-Toe draws, so a converged agent must draw every
   game and can never win. A loss is proof of a mistake.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool

import numpy as np
from games import TicTacToe, solve
from mcts import search, uniform_evaluator

from rl_ladder import Run, plot_curves

BUDGETS = [10, 25, 50, 100, 200, 400, 800]


def reachable_positions(rng: np.random.Generator, n: int, cache: dict) -> list[np.ndarray]:
    """Random non-terminal positions whose optimal move set is a strict subset."""
    out = []
    while len(out) < n:
        board = TicTacToe.initial()
        for _ in range(int(rng.integers(0, 6))):
            if TicTacToe.outcome(board) is not None:
                break
            board = TicTacToe.play(board, int(rng.choice(TicTacToe.legal_moves(board))))
        if TicTacToe.outcome(board) is not None:
            continue
        _, best = solve(TicTacToe, board, cache)
        if 0 < len(best) < len(TicTacToe.legal_moves(board)):
            out.append(board)
    return out


def play_vs_perfect(budget: int, seed: int, mcts_first: bool, cache: dict) -> float:
    """One game against a perfect player. Returns +1 win / 0 draw / -1 loss for MCTS."""
    rng = np.random.default_rng(seed)
    evaluate = uniform_evaluator(TicTacToe, rng)
    board = TicTacToe.initial()
    mcts_turn = mcts_first
    while TicTacToe.outcome(board) is None:
        if mcts_turn:
            counts = search(TicTacToe, board, evaluate, budget, rng, c=1.4, selection="uct")
            move = int(counts.argmax())
        else:
            move = int(rng.choice(solve(TicTacToe, board, cache)[1]))
        board = TicTacToe.play(board, move)
        mcts_turn = not mcts_turn
    # `outcome` is from the perspective of the player to move, so it is already
    # MCTS's result when it is MCTS to move, and the opponent's otherwise.
    result = TicTacToe.outcome(board)
    return result if mcts_turn else -result


def one_budget(job):
    budget, seed, n_positions, n_games = job
    cache: dict = {}
    rng = np.random.default_rng(seed)
    evaluate = uniform_evaluator(TicTacToe, rng)

    correct = 0
    for board in reachable_positions(rng, n_positions, cache):
        best = solve(TicTacToe, board, cache)[1]
        counts = search(TicTacToe, board, evaluate, budget, rng, c=1.4, selection="uct")
        correct += int(counts.argmax()) in best

    results = [play_vs_perfect(budget, seed * 977 + g, g % 2 == 0, cache) for g in range(n_games)]
    results = np.asarray(results)
    # Wins are reported too: against a perfect player they must be exactly zero,
    # so a non-zero win rate is a bug in the harness rather than a strong agent.
    return (budget, seed, correct / n_positions, float((results == 0).mean()),
            float((results < 0).mean()), float((results > 0).mean()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--positions", type=int, default=120)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    run = Run(phase="a6", name="uct", config=vars(args))
    jobs = [(b, s, args.positions, args.games) for b in BUDGETS for s in range(args.seeds)]
    with Pool(args.workers) as pool:
        results = pool.map(one_budget, jobs)

    accuracy, draws, summary = {}, {}, {}
    for budget in BUDGETS:
        rows = [r for r in results if r[0] == budget]
        acc = np.array([r[2] for r in rows])
        drw = np.array([r[3] for r in rows])
        los = np.array([r[4] for r in rows])
        win = np.array([r[5] for r in rows])
        accuracy[str(budget)] = acc.tolist()
        draws[str(budget)] = drw.tolist()
        summary[str(budget)] = {
            "move_accuracy": float(acc.mean()), "move_accuracy_std": float(acc.std()),
            "draw_rate_vs_perfect": float(drw.mean()), "loss_rate_vs_perfect": float(los.mean()),
            "win_rate_vs_perfect": float(win.mean()),
        }
        run.log(step=budget, simulations=budget, **summary[str(budget)])
        print(f"sims {budget:4d} | optimal-move rate {acc.mean():.3f} +/- {acc.std():.3f}"
              f" | vs perfect: draws {drw.mean():.3f} losses {los.mean():.3f}"
              f" wins {win.mean():.3f}")

    plot_curves({"optimal-move rate": [[summary[str(b)]["move_accuracy"] for b in BUDGETS]],
                 "draw rate vs perfect play": [[summary[str(b)]["draw_rate_vs_perfect"] for b in BUDGETS]]},
                title=f"Tic-Tac-Toe: plain UCT vs simulation budget ({args.seeds} seeds)",
                xlabel=f"budget index (sims = {BUDGETS})", ylabel="rate",
                out_path=run.path("uct.png"))
    print(f"run -> {run.finish(summary)}")


if __name__ == "__main__":
    main()
