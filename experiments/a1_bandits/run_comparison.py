"""A1 item 1: epsilon-greedy vs UCB1 vs Thompson sampling on a stationary bandit."""

from __future__ import annotations

import argparse

import numpy as np
from bandits import UCB1, EpsilonGreedy, GaussianBandit, ThompsonGaussian, run_episode

from rl_ladder import Run, plot_curves, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--arms", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--ucb-c", type=float, default=2.0)
    args = parser.parse_args()

    agents = {
        f"eps-greedy (eps={args.eps})": lambda g: EpsilonGreedy(args.arms, g, eps=args.eps),
        f"UCB1 (c={args.ucb_c})": lambda g: UCB1(args.arms, g, c=args.ucb_c),
        "Thompson sampling": lambda g: ThompsonGaussian(args.arms, g),
    }

    run = Run(phase="a1", name="comparison", config=vars(args))
    regret, optimal, summary = {}, {}, {}

    for label, make_agent in agents.items():
        seed_everything(0)
        results = [
            run_episode(make_agent, lambda g: GaussianBandit(args.arms, g), args.steps, seed)
            for seed in range(args.seeds)
        ]
        regret[label] = [r["cumulative_regret"] for r in results]
        optimal[label] = [r["optimal_action"] for r in results]

        finals = np.array([r["cumulative_regret"][-1] for r in results])
        last_opt = np.array([r["optimal_action"][-100:].mean() for r in results])
        summary[label] = {
            "final_regret_mean": float(finals.mean()),
            "final_regret_std": float(finals.std()),
            "optimal_action_last100": float(last_opt.mean()),
        }
        run.log(step=len(summary), agent=label, **summary[label])
        print(f"{label:28s} regret {finals.mean():7.1f} +/- {finals.std():5.1f}   opt% {last_opt.mean():.3f}")

    plot_curves(
        regret,
        title=f"Cumulative regret, {args.arms}-armed Gaussian bandit ({args.seeds} seeds)",
        xlabel="pull",
        ylabel="cumulative regret",
        out_path=run.path("regret.png"),
    )
    plot_curves(
        optimal,
        title=f"Optimal-action rate ({args.seeds} seeds, 50-pull moving average)",
        xlabel="pull",
        ylabel="P(optimal arm)",
        out_path=run.path("optimal_action.png"),
        smooth_window=50,
    )
    print(f"run -> {run.finish(summary)}")


if __name__ == "__main__":
    main()
