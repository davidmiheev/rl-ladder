"""A4 item 3: A2C against REINFORCE-with-baseline on LunarLander.

LunarLander is the harder task: eight-dimensional state, shaped reward, and a
credit-assignment horizon long enough that the difference between updating from
whole episodes and updating every few steps across parallel environments
actually matters. The comparison is stability, not just final score.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool

import numpy as np
from pg import Config, train_a2c, train_reinforce

from rl_ladder import Run, plot_curves

ENV = "LunarLander-v3"


# At equal environment steps the two differ in how many gradient steps they
# extract: A2C updates every n_envs * n_steps = 40 steps, episodic REINFORCE
# every batch_steps. The third arm is a control for that confound — the same
# REINFORCE with a 4x smaller batch, and so 4x more updates.
BATCH_STEPS = {"REINFORCE + baseline + GAE": 4_000, "REINFORCE, small batch": 1_000}


def one_run(job):
    label, seed, steps = job
    if label.startswith("A2C"):
        config = Config(env_id=ENV, total_steps=steps, n_envs=8, n_steps=5,
                        eval_every=steps // 15, lr=1e-3, gae_lambda=0.95)
        return label, seed, train_a2c(config, seed)
    config = Config(env_id=ENV, total_steps=steps, baseline=True, gae_lambda=0.95,
                    batch_steps=BATCH_STEPS[label], eval_every=steps // 15, lr=1e-3)
    return label, seed, train_reinforce(config, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    run = Run(phase="a4", name="a2c", config=vars(args))
    labels = ["A2C (8 envs)", "REINFORCE + baseline + GAE", "REINFORCE, small batch"]
    jobs = [(label, seed, args.steps) for label in labels for seed in range(args.seeds)]
    with Pool(args.workers) as pool:
        results = pool.map(one_run, jobs)

    curves, summary = {}, {}
    for label in labels:
        runs = [m for lab, _, m in results if lab == label]
        curves[label] = [m.eval_returns for m in runs]
        final = np.array([m.eval_returns[-1] for m in runs])
        peak = np.array([max(m.eval_returns) for m in runs])
        # Stability, three ways: spread across seeds, how much of its own peak
        # each run still holds, and how much the curve moves between evaluations.
        swing = np.array([np.mean(np.abs(np.diff(m.eval_returns))) for m in runs])
        updates = args.steps // (40 if label.startswith("A2C") else BATCH_STEPS[label])
        summary[label] = {
            "gradient_updates": updates,
            "final_eval_return": float(final.mean()),
            "final_std": float(final.std()),
            "peak": float(peak.mean()),
            "retention": float((final / np.where(peak > 0, peak, np.nan)).mean()),
            "mean_abs_change_between_evals": float(swing.mean()),
            "final_per_seed": [round(float(f), 1) for f in final],
        }
        run.log(step=len(summary), arm=label, **summary[label])
        print(f"{label:28s} final {final.mean():8.1f} +/- {final.std():6.1f}  peak {peak.mean():7.1f}"
              f"  swing {swing.mean():6.1f}  updates {updates:5d}"
              f"  per-seed {summary[label]['final_per_seed']}")

    plot_curves(curves, title=f"{ENV}: A2C vs REINFORCE+baseline ({args.seeds} seeds)",
                xlabel="evaluation", ylabel="greedy eval return", out_path=run.path("a2c.png"))
    print(f"run -> {run.finish(summary)}")


if __name__ == "__main__":
    main()
