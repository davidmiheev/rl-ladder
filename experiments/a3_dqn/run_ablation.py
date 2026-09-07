"""A3 item 1: disable the replay buffer and the target network, independently.

Four arms on the same environment and seeds. The deadly triad is bootstrapping +
off-policy + function approximation; replay and the target network attack
different parts of the resulting instability, so removing them one at a time is
the only way to see which does what.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool

import numpy as np
from dqn import Config, train

from rl_ladder import Run, plot_curves

ARMS = {
    "DQN (replay + target)": {"replay": True, "target_network": True},
    "no replay": {"replay": False, "target_network": True},
    "no target net": {"replay": True, "target_network": False},
    "neither": {"replay": False, "target_network": False},
}


def one_run(job):
    label, overrides, seed, steps = job
    metrics = train(Config(total_steps=steps, **overrides), seed)
    return label, seed, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    run = Run(phase="a3", name="ablation", config=vars(args))
    jobs = [(label, overrides, seed, args.steps)
            for label, overrides in ARMS.items() for seed in range(args.seeds)]

    with Pool(args.workers) as pool:
        results = pool.map(one_run, jobs)

    curves, summary = {}, {}
    for label in ARMS:
        runs = [m for lab, _, m in results if lab == label]
        curves[label] = [m.eval_returns for m in runs]
        final = np.array([m.eval_returns[-1] for m in runs])
        best = np.array([max(m.eval_returns) for m in runs])
        summary[label] = {
            "final_eval_return": float(final.mean()),
            "final_std": float(final.std()),
            "best_eval_return": float(best.mean()),
            # How much of its own peak the agent still holds at the end: the
            # signature of instability is a high peak followed by collapse.
            "retention": float((final / np.maximum(best, 1e-9)).mean()),
        }
        run.log(step=len(summary), arm=label, **summary[label])
        print(f"{label:24s} final {final.mean():6.1f} +/- {final.std():5.1f}"
              f"   peak {best.mean():6.1f}   retention {summary[label]['retention']:.2f}")

    steps = results[0][2].eval_steps
    plot_curves(curves, title=f"CartPole-v1: greedy eval return ({args.seeds} seeds)",
                xlabel=f"evaluation (every {steps[0]} env steps)", ylabel="return",
                out_path=run.path("ablation.png"))
    print(f"run -> {run.finish(summary)}")


if __name__ == "__main__":
    main()
