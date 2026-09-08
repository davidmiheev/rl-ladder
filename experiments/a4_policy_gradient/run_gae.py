"""A4 item 2: sweep the GAE lambda and watch the bias/variance tradeoff.

lambda = 0 is the one-step TD residual: lowest variance, biased by whatever
error the critic carries. lambda = 1 telescopes to the Monte Carlo return minus
the baseline: unbiased, highest variance. The claim is that neither end is best,
which only shows up if both the variance and the resulting performance are
measured.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool

import numpy as np
from pg import Config, train_reinforce

from rl_ladder import Run, plot_curves

LAMBDAS = [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]


def one_run(job):
    label, overrides, seed, steps = job
    return label, seed, train_reinforce(Config(total_steps=steps, **overrides), seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    run = Run(phase="a4", name="gae", config=vars(args))
    arms = {"no baseline": {"baseline": False}}
    arms |= {f"GAE lambda={lam:g}": {"baseline": True, "gae_lambda": lam} for lam in LAMBDAS}

    jobs = [(label, o, seed, args.steps) for label, o in arms.items() for seed in range(args.seeds)]
    with Pool(args.workers) as pool:
        results = pool.map(one_run, jobs)

    curves, summary, table = {}, {}, []
    for label in arms:
        runs = [m for lab, _, m in results if lab == label]
        curves[label] = [m.eval_returns for m in runs]
        final = np.array([m.eval_returns[-1] for m in runs])
        # Advantage spread is measured before normalisation, so it reflects what
        # lambda does to the estimator rather than what the optimiser rescales.
        adv = float(np.mean([np.mean(m.advantage_std) for m in runs]))
        ev = (
            float(np.mean([m.explained_variance[-1] for m in runs]))
            if runs[0].explained_variance
            else float("nan")
        )
        summary[label] = {
            "final_eval_return": float(final.mean()),
            "final_std": float(final.std()),
            "advantage_std": adv,
            "explained_variance": ev,
            "final_per_seed": [round(float(f), 1) for f in final],
        }
        run.log(step=len(summary), arm=label, **summary[label])
        table.append((label, final.mean(), final.std(), adv, ev))
        print(f"{label:20s} final {final.mean():6.1f} +/- {final.std():5.1f}"
              f"   advantage std {adv:6.2f}   explained var {ev:5.2f}")

    plot_curves(curves, title=f"CartPole-v1: GAE lambda sweep ({args.seeds} seeds)",
                xlabel="evaluation", ylabel="greedy eval return", out_path=run.path("gae.png"))
    print(f"run -> {run.finish(summary)}")


if __name__ == "__main__":
    main()
