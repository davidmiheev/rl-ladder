"""A3 item 3: dueling architecture vs vanilla DQN, at equal environment steps.

Both arms use replay, a target network and identical hyperparameters; only the
head differs, so any gap is the architecture. Sample efficiency is read as the
number of environment steps to first reach a return threshold.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool

import numpy as np
from dqn import Config, train

from rl_ladder import Run, plot_curves

THRESHOLD = 400.0


def one_run(job):
    label, overrides, seed, steps = job
    return label, seed, train(Config(total_steps=steps, **overrides), seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    run = Run(phase="a3", name="dueling", config=vars(args))
    arms = {"DQN": {"dueling": False}, "Dueling DQN": {"dueling": True}}
    jobs = [(label, o, seed, args.steps) for label, o in arms.items() for seed in range(args.seeds)]

    with Pool(args.workers) as pool:
        results = pool.map(one_run, jobs)

    curves, summary = {}, {}
    for label in arms:
        runs = [m for lab, _, m in results if lab == label]
        curves[label] = [m.eval_returns for m in runs]
        final = np.array([m.eval_returns[-1] for m in runs])
        area = np.array([np.mean(m.eval_returns) for m in runs])

        reached = []
        for m in runs:
            hits = [s for s, r in zip(m.eval_steps, m.eval_returns, strict=True) if r >= THRESHOLD]
            reached.append(hits[0] if hits else np.nan)
        summary[label] = {
            "final_eval_return": float(final.mean()),
            "mean_eval_return": float(area.mean()),
            "steps_to_threshold_median": float(np.nanmedian(reached)),
            "seeds_reaching_threshold": int(np.sum(~np.isnan(reached))),
        }
        run.log(step=len(summary), arm=label, **summary[label])
        print(f"{label:14s} final {final.mean():6.1f}   area-under-curve {area.mean():6.1f}"
              f"   median steps to {THRESHOLD:.0f}: {np.nanmedian(reached):.0f}"
              f"   ({summary[label]['seeds_reaching_threshold']}/{args.seeds} seeds)")

    plot_curves(curves, title=f"CartPole-v1: dueling vs vanilla ({args.seeds} seeds)",
                xlabel="evaluation", ylabel="greedy eval return", out_path=run.path("dueling.png"))
    print(f"run -> {run.finish(summary)}")


if __name__ == "__main__":
    main()
