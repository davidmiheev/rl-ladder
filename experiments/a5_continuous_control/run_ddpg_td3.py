"""A5 items 1 and 2: DDPG's instability, and which of TD3's three fixes matters.

TD3 is DDPG plus twin critics, delayed actor updates and target policy
smoothing. Reporting only "TD3 beats DDPG" says nothing about which fix carried
it, so each is also removed individually from the full TD3.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool

import numpy as np
from continuous import Config, train

from rl_ladder import Run, plot_curves

TD3 = {"twin_critics": True, "policy_delay": 2, "target_noise": 0.2}
ARMS = {
    "DDPG": {"algo": "ddpg"},
    "TD3 (all three)": {"algo": "td3", **TD3},
    "TD3 minus twin critics": {"algo": "td3", **TD3, "twin_critics": False},
    "TD3 minus delayed actor": {"algo": "td3", **TD3, "policy_delay": 1},
    "TD3 minus target smoothing": {"algo": "td3", **TD3, "target_noise": 0.0},
}


def one_run(job):
    label, overrides, seed, steps = job
    metrics, *_ = train(Config(total_steps=steps, **overrides), seed)
    return label, seed, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    run = Run(phase="a5", name="ddpg_td3", config=vars(args))
    jobs = [(label, o, seed, args.steps) for label, o in ARMS.items() for seed in range(args.seeds)]
    with Pool(args.workers) as pool:
        results = pool.map(one_run, jobs)

    curves, summary = {}, {}
    for label in ARMS:
        runs = [m for lab, _, m in results if lab == label]
        curves[label] = [m.eval_returns for m in runs]
        final = np.array([m.eval_returns[-1] for m in runs])
        peak = np.array([max(m.eval_returns) for m in runs])
        # Value bias, the same probe as A3: what the critic predicts at s0 minus
        # what the policy actually collects. Twin critics target exactly this.
        bias = np.array([np.mean([p - a for p, a in
                                  zip(m.predicted_q[len(m.predicted_q) // 2:],
                                      m.actual_return[len(m.actual_return) // 2:], strict=True)])
                         for m in runs])
        # Instability: how far the curve moves between consecutive evaluations.
        swing = np.array([np.mean(np.abs(np.diff(m.eval_returns))) for m in runs])
        summary[label] = {
            "final_return": float(final.mean()), "final_std": float(final.std()),
            "peak_return": float(peak.mean()),
            "retention": float((final / peak).mean()),
            "value_bias_late": float(bias.mean()), "value_bias_std": float(bias.std()),
            "swing_per_eval": float(swing.mean()),
            "final_per_seed": [round(float(f)) for f in final],
        }
        run.log(step=len(summary), arm=label, **summary[label])
        print(f"{label:28s} final {final.mean():8.1f} +/- {final.std():6.1f} | peak {peak.mean():8.1f}"
              f" | bias {bias.mean():+7.1f} | swing {swing.mean():6.1f}"
              f" | per-seed {summary[label]['final_per_seed']}")

    plot_curves(curves, title=f"Pendulum-v1: DDPG vs TD3 and its ablations ({args.seeds} seeds)",
                xlabel="evaluation (every 2000 steps)", ylabel="eval return",
                out_path=run.path("ddpg_td3.png"))
    print(f"run -> {run.finish(summary)}")


if __name__ == "__main__":
    main()
