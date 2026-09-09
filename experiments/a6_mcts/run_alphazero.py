"""A6 item 2: train a policy/value net by self-play, compare with plain MCTS.

"Sample efficiency" for a searcher is best read as *search* efficiency: how
many simulations does plain MCTS need to match the trained agent at a fixed
budget? That number is what the learned prior and value are worth, expressed
in the currency the search itself spends.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool

import numpy as np
import torch
from alphazero import AZConfig, play_match, self_play_game, train_step
from games import Connect4
from mcts import uniform_evaluator
from net import PolicyValueNet, net_evaluator

from rl_ladder import Run, plot_curves

GAME = Connect4
OPPONENT_BUDGETS = [50, 100, 200, 400]


def _self_play(job):
    state_dict, config, seed = job
    return self_play_game(GAME, state_dict, config, seed)


def _match(job):
    state_dict, sims_a, sims_b, seed, a_first = job
    torch.set_num_threads(1)
    net = PolicyValueNet(GAME.rows, GAME.cols, GAME.n_actions)
    net.load_state_dict(state_dict)
    net.eval()
    rng = np.random.default_rng(seed)
    evaluate_a = net_evaluator(GAME, net, use_value=True, use_prior=True, rng=rng)
    evaluate_b = uniform_evaluator(GAME, rng)
    return play_match(GAME, evaluate_a, evaluate_b, sims_a, sims_b, seed, a_first)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--sims", type=int, default=50)
    parser.add_argument("--match-games", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    config = AZConfig(simulations=args.sims, games_per_iter=args.games, iterations=args.iterations)
    run = Run(phase="a6", name="alphazero", config=vars(args))
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    net = PolicyValueNet(GAME.rows, GAME.cols, GAME.n_actions)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    boards_buf, policies_buf, outcomes_buf = [], [], []
    progress = []

    with Pool(args.workers) as pool:
        for iteration in range(1, args.iterations + 1):
            state = {k: v.clone() for k, v in net.state_dict().items()}
            jobs = [(state, config, iteration * 10_000 + g) for g in range(args.games)]
            for b, p, o in pool.map(_self_play, jobs):
                boards_buf.append(b)
                policies_buf.append(p)
                outcomes_buf.append(o)
            boards_buf = boards_buf[-config.buffer_games:]
            policies_buf = policies_buf[-config.buffer_games:]
            outcomes_buf = outcomes_buf[-config.buffer_games:]

            boards = np.concatenate(boards_buf)
            policies = np.concatenate(policies_buf)
            outcomes = np.concatenate(outcomes_buf)
            net.train()
            losses = [train_step(net, optimizer, boards, policies, outcomes, config, rng)
                      for _ in range(config.train_steps)]
            net.eval()

            # Strength check against plain MCTS at the same budget, so any gain
            # is the network's rather than extra search.
            state = {k: v.clone() for k, v in net.state_dict().items()}
            match_jobs = [(state, args.sims, args.sims, iteration * 7919 + g, g % 2 == 0)
                          for g in range(20)]
            scores = np.asarray(pool.map(_match, match_jobs))
            progress.append(float((scores > 0).mean() + 0.5 * (scores == 0).mean()))
            run.log(step=iteration, iteration=iteration, loss=float(np.mean(losses)),
                    score_vs_plain_mcts=progress[-1], positions=len(boards))
            print(f"iter {iteration:2d} | loss {np.mean(losses):.4f} | positions {len(boards):6d} "
                  f"| score vs plain MCTS@{args.sims}: {progress[-1]:.3f}", flush=True)

        # Final sweep: how much search does plain MCTS need to match AZ@sims?
        print(f"\nAZ@{args.sims} against plain MCTS at increasing budgets:")
        state = {k: v.clone() for k, v in net.state_dict().items()}
        summary = {"training_progress": progress}
        for budget in OPPONENT_BUDGETS:
            jobs = [(state, args.sims, budget, budget * 31 + g, g % 2 == 0)
                    for g in range(args.match_games)]
            scores = np.asarray(pool.map(_match, jobs))
            score = float((scores > 0).mean() + 0.5 * (scores == 0).mean())
            summary[f"vs_mcts_{budget}"] = {
                "score": score, "wins": float((scores > 0).mean()),
                "draws": float((scores == 0).mean()), "losses": float((scores < 0).mean()),
            }
            run.log(step=budget, opponent_sims=budget, **summary[f"vs_mcts_{budget}"])
            print(f"  vs MCTS@{budget:4d}: score {score:.3f} "
                  f"(W {(scores > 0).mean():.2f} D {(scores == 0).mean():.2f} L {(scores < 0).mean():.2f})")

    torch.save(net.state_dict(), "az_net.pt")
    plot_curves({f"AZ@{args.sims} score vs plain MCTS@{args.sims}": [progress]},
                title=f"Connect Four: self-play training ({args.games} games/iteration)",
                xlabel="iteration", ylabel="score (win + half draw)", out_path=run.path("alphazero.png"))
    print(f"\nsaved az_net.pt | run -> {run.finish(summary)}")


if __name__ == "__main__":
    main()
