"""Plumbing demo: a uniform-random baseline on a Gaussian bandit.

This is NOT an A1 deliverable — A1 asks for epsilon-greedy, UCB1 and Thompson
sampling. This script only shows the run/log/plot conventions the rest of the
ladder uses. Copy its structure, replace the agent.
"""

from __future__ import annotations

import argparse

import numpy as np

from rl_ladder import Run, plot_curves, seed_everything


def rollout(n_arms: int, steps: int, rng: np.random.Generator) -> list[float]:
    """Run a uniform-random policy, returning the cumulative-regret curve."""
    true_means = rng.normal(size=n_arms)
    best = true_means.max()

    regret = 0.0
    curve = []
    for _ in range(steps):
        arm = rng.integers(n_arms)
        reward = rng.normal(true_means[arm])
        regret += best - true_means[arm]
        curve.append(regret)
        del reward  # the random policy ignores feedback; a real agent would not
    return curve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--arms", type=int, default=10)
    args = parser.parse_args()

    run = Run(phase="template", name="random-baseline", config=vars(args))

    curves = []
    for seed in range(args.seeds):
        seed_everything(seed)
        rng = np.random.default_rng(seed)
        curve = rollout(args.arms, args.steps, rng)
        curves.append(curve)
        run.log(step=seed, seed=seed, final_regret=curve[-1])

    finals = [c[-1] for c in curves]
    plot_curves(
        {"uniform-random": curves},
        title=f"Cumulative regret ({args.arms}-armed Gaussian bandit)",
        xlabel="pull",
        ylabel="cumulative regret",
        out_path=run.path("regret.png"),
    )
    out = run.finish({"mean_final_regret": float(np.mean(finals)), "std": float(np.std(finals))})
    print(f"mean final regret {np.mean(finals):.1f} +/- {np.std(finals):.1f}")
    print(f"run -> {out}")


if __name__ == "__main__":
    main()
