"""A6 item 3: isolate what the network contributes — prior, value, or both.

The trained network supplies two things to the search: a prior over moves at
expansion, and a value at the leaf in place of a random rollout. Crossing them
gives a 2x2, with the search itself (PUCT, same budget, same opponent) held
fixed, so each cell attributes strength to one component rather than to
"the network".

The uniform-prior + rollout-value cell is plain MCTS, which makes it the
control rather than a separate baseline.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool

import numpy as np
import torch
from alphazero import play_match
from games import Connect4
from mcts import uniform_evaluator
from net import PolicyValueNet, net_evaluator

from rl_ladder import Run

GAME = Connect4
ARMS = {
    "uniform prior + rollout value (plain MCTS)": (False, False),
    "net prior + rollout value": (True, False),
    "uniform prior + net value": (False, True),
    "net prior + net value (full AZ)": (True, True),
}


def _match(job):
    (use_prior, use_value), sims, opponent_sims, seed, a_first = job
    torch.set_num_threads(1)
    net = PolicyValueNet(GAME.rows, GAME.cols, GAME.n_actions)
    net.load_state_dict(torch.load("az_net.pt", map_location="cpu"))
    net.eval()
    rng = np.random.default_rng(seed)
    evaluate_a = net_evaluator(GAME, net, use_value=use_value, use_prior=use_prior, rng=rng)
    evaluate_b = uniform_evaluator(GAME, rng)
    # Both sides search; only the evaluator differs, so PUCT is used for the
    # arm under test and plain UCT for the fixed reference opponent.
    return play_match(GAME, evaluate_a, evaluate_b, sims, opponent_sims, seed, a_first)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=50)
    parser.add_argument("--opponent-sims", type=int, default=200)
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    run = Run(phase="a6", name="ablation", config=vars(args))
    summary = {}
    with Pool(args.workers) as pool:
        for label, flags in ARMS.items():
            jobs = [(flags, args.sims, args.opponent_sims, hash(label) % 10_000 + g, g % 2 == 0)
                    for g in range(args.games)]
            scores = np.asarray(pool.map(_match, jobs))
            score = float((scores > 0).mean() + 0.5 * (scores == 0).mean())
            summary[label] = {
                "score": score, "wins": float((scores > 0).mean()),
                "draws": float((scores == 0).mean()), "losses": float((scores < 0).mean()),
            }
            run.log(step=len(summary), arm=label, **summary[label])
            print(f"{label:45s} score {score:.3f}  (W {(scores > 0).mean():.2f} "
                  f"D {(scores == 0).mean():.2f} L {(scores < 0).mean():.2f})")

    print(f"\nrun -> {run.finish(summary)}")


if __name__ == "__main__":
    main()
