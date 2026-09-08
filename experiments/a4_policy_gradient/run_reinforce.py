"""A4 item 1: REINFORCE against the A3 DQN baseline, same task and protocol.

Identical evaluation to A3 — CartPole-v1, 30k environment steps, greedy eval
every 2000 steps, 5 seeds — so the numbers are directly comparable rather than
comparable in spirit.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool

import numpy as np
from pg import Config, train_reinforce

from rl_ladder import Run, plot_curves

# From A3's ablation run (runs/a3/ablation__20260907-210256, 5 seeds, 30k steps),
# recorded here so this script prints the comparison without depending on a
# gitignored run directory. See experiments/a3_dqn/README.md.
A3_DQN = {"final": 250.1, "final_std": 151.2, "peak": 469.4, "retention": 0.56}

ARMS = {
    "REINFORCE": {"baseline": False},
    "REINFORCE + baseline": {"baseline": True},
}


def one_run(job):
    label, overrides, seed, steps = job
    return label, seed, train_reinforce(Config(total_steps=steps, **overrides), seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    run = Run(phase="a4", name="reinforce", config=vars(args))
    jobs = [(label, o, seed, args.steps) for label, o in ARMS.items() for seed in range(args.seeds)]
    with Pool(args.workers) as pool:
        results = pool.map(one_run, jobs)

    curves, summary = {}, {"a3_dqn_reference": A3_DQN}
    for label in ARMS:
        runs = [m for lab, _, m in results if lab == label]
        curves[label] = [m.eval_returns for m in runs]
        final = np.array([m.eval_returns[-1] for m in runs])
        peak = np.array([max(m.eval_returns) for m in runs])
        summary[label] = {
            "final_eval_return": float(final.mean()),
            "final_std": float(final.std()),
            "peak": float(peak.mean()),
            "retention": float((final / np.maximum(peak, 1e-9)).mean()),
            "final_per_seed": [round(float(f), 1) for f in final],
            "mean_advantage_std": float(np.mean([np.mean(m.advantage_std) for m in runs])),
        }
        run.log(step=len(summary), arm=label, **summary[label])
        print(f"{label:22s} final {final.mean():6.1f} +/- {final.std():5.1f}   peak {peak.mean():6.1f}"
              f"   retention {summary[label]['retention']:.2f}"
              f"   adv std {summary[label]['mean_advantage_std']:5.1f}")

    print(f"{'DQN (A3 reference)':22s} final {A3_DQN['final']:6.1f} +/- {A3_DQN['final_std']:5.1f}"
          f"   peak {A3_DQN['peak']:6.1f}   retention {A3_DQN['retention']:.2f}")

    plot_curves(curves, title=f"CartPole-v1: policy gradient ({args.seeds} seeds, same protocol as A3)",
                xlabel="evaluation (every 2000 env steps)", ylabel="greedy eval return",
                out_path=run.path("reinforce.png"))
    print(f"run -> {run.finish(summary)}")


if __name__ == "__main__":
    main()
