"""A1 item 3: sample-average vs constant step size on a drifting bandit.

Both agents are epsilon-greedy with the same epsilon; the only difference is how
Q is updated. Sample averaging weights every observation equally, so its
estimates are dominated by stale rewards once the arms drift. A constant step
size keeps an exponentially weighted recent average and tracks the drift.
"""

from __future__ import annotations

import argparse

import numpy as np
from bandits import EpsilonGreedy, RandomWalkBandit, run_episode

from rl_ladder import Run, plot_curves, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--arms", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--walk-std", type=float, default=0.01)
    args = parser.parse_args()

    agents = {
        "sample average": lambda g: EpsilonGreedy(args.arms, g, eps=args.eps),
        f"constant alpha={args.alpha}": lambda g: EpsilonGreedy(
            args.arms, g, eps=args.eps, step_size=args.alpha
        ),
    }

    run = Run(phase="a1", name="nonstationary", config=vars(args))
    regret, optimal, summary = {}, {}, {}

    for label, make_agent in agents.items():
        seed_everything(0)
        results = [
            run_episode(
                make_agent,
                lambda g: RandomWalkBandit(args.arms, g, walk_std=args.walk_std),
                args.steps,
                seed,
            )
            for seed in range(args.seeds)
        ]
        regret[label] = [r["cumulative_regret"] for r in results]
        optimal[label] = [r["optimal_action"] for r in results]

        finals = np.array([r["cumulative_regret"][-1] for r in results])
        first = np.array([r["optimal_action"][: args.steps // 10].mean() for r in results])
        last = np.array([r["optimal_action"][-args.steps // 10 :].mean() for r in results])
        summary[label] = {
            "final_regret_mean": float(finals.mean()),
            "final_regret_std": float(finals.std()),
            "optimal_action_first10pct": float(first.mean()),
            "optimal_action_last10pct": float(last.mean()),
        }
        run.log(step=len(summary), agent=label, **summary[label])
        print(
            f"{label:22s} regret {finals.mean():8.1f} +/- {finals.std():5.1f}"
            f"   opt% first10 {first.mean():.3f} -> last10 {last.mean():.3f}"
        )

    plot_curves(
        regret,
        title=f"Non-stationary bandit: cumulative regret ({args.seeds} seeds)",
        xlabel="pull",
        ylabel="cumulative regret",
        out_path=run.path("regret.png"),
    )
    plot_curves(
        optimal,
        title="Non-stationary bandit: optimal-action rate (500-pull moving average)",
        xlabel="pull",
        ylabel="P(optimal arm)",
        out_path=run.path("optimal_action.png"),
        smooth_window=500,
    )
    print(f"run -> {run.finish(summary)}")


if __name__ == "__main__":
    main()
