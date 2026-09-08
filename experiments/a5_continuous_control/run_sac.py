"""A5 item 3: SAC on a harder task, sweeping the entropy temperature.

SAC maximises reward plus alpha * entropy, so alpha sets the exchange rate
between the two. Too high and the policy stays deliberately random; too low and
it collapses early to whatever it found first. The sweep should show both ends.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool

import numpy as np
from continuous import Config, train

from rl_ladder import Run, plot_curves

ALPHAS = [0.01, 0.05, 0.2, 0.5]
ENV = "LunarLanderContinuous-v3"


def one_run(job):
    alpha, seed, steps = job
    config = Config(env_id=ENV, algo="sac", twin_critics=True, alpha=alpha,
                    total_steps=steps, eval_every=steps // 15)
    metrics, *_ = train(config, seed)
    return alpha, seed, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    run = Run(phase="a5", name="sac_alpha", config=vars(args))
    jobs = [(alpha, seed, args.steps) for alpha in ALPHAS for seed in range(args.seeds)]
    with Pool(args.workers) as pool:
        results = pool.map(one_run, jobs)

    curves, summary = {}, {}
    for alpha in ALPHAS:
        runs = [m for a, _, m in results if a == alpha]
        label = f"alpha={alpha:g}"
        curves[label] = [m.eval_returns for m in runs]
        final = np.array([m.eval_returns[-1] for m in runs])
        peak = np.array([max(m.eval_returns) for m in runs])
        summary[label] = {
            "final_return": float(final.mean()), "final_std": float(final.std()),
            "peak_return": float(peak.mean()),
            "final_per_seed": [round(float(f)) for f in final],
        }
        run.log(step=len(summary), alpha=alpha, **summary[label])
        print(f"{label:12s} final {final.mean():8.1f} +/- {final.std():6.1f} | peak {peak.mean():8.1f}"
              f" | per-seed {summary[label]['final_per_seed']}")

    plot_curves(curves, title=f"{ENV}: SAC entropy temperature sweep ({args.seeds} seeds)",
                xlabel="evaluation", ylabel="eval return", out_path=run.path("sac_alpha.png"))
    print(f"run -> {run.finish(summary)}")


if __name__ == "__main__":
    main()
