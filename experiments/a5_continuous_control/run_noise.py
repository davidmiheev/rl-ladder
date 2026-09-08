"""A5 item 4: robustness to action and observation noise at evaluation time.

The argument for SAC is that maximising entropy produces a policy that keeps a
*distribution* of good actions rather than balancing on one, so it should sit
in a flatter region of the return surface and lose less when perturbed. The
agents are trained clean and only perturbed at evaluation, so this measures the
shape of the learned policy rather than robustness training.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool

import numpy as np
from continuous import Config, evaluate, train

from rl_ladder import Run

LEVELS = [0.0, 0.1, 0.2, 0.4]
ARMS = {
    "DDPG": {"algo": "ddpg"},
    "TD3": {"algo": "td3", "twin_critics": True, "policy_delay": 2, "target_noise": 0.2},
    "SAC": {"algo": "sac", "twin_critics": True, "alpha": 0.2},
}


def one_run(job):
    label, overrides, seed, steps = job
    config = Config(total_steps=steps, **overrides)
    _, actor, critic, eval_env = train(config, seed)
    out = {}
    for kind in ("action", "observation"):
        for level in LEVELS:
            kwargs = {"action_noise": level} if kind == "action" else {"obs_noise": level}
            ret, _, _ = evaluate(eval_env, actor, critic, config, seed + 500, **kwargs)
            out[(kind, level)] = ret
    return label, seed, out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    run = Run(phase="a5", name="noise", config=vars(args))
    jobs = [(label, o, seed, args.steps) for label, o in ARMS.items() for seed in range(args.seeds)]
    with Pool(args.workers) as pool:
        results = pool.map(one_run, jobs)

    summary = {}
    for kind in ("action", "observation"):
        print(f"\n=== {kind} noise ===")
        for label in ARMS:
            runs = [out for lab, _, out in results if lab == label]
            means = [float(np.mean([r[(kind, lvl)] for r in runs])) for lvl in LEVELS]
            clean = means[0]
            # Degradation relative to each algorithm's own clean score, so an
            # algorithm is not rewarded for simply being better to begin with.
            retained = [m / clean if clean < 0 else float("nan") for m in means]
            summary[f"{label}-{kind}"] = {
                "returns": means,
                "clean": clean,
                "loss_at_max_noise": float(means[0] - means[-1]),
                "relative_at_max_noise": float(means[-1] / clean) if clean else float("nan"),
            }
            run.log(step=len(summary), arm=label, kind=kind, **summary[f"{label}-{kind}"])
            cells = "  ".join(f"{lvl:g}:{m:8.1f}" for lvl, m in zip(LEVELS, means, strict=True))
            print(f"{label:5s} {cells}   drop {means[0] - means[-1]:8.1f}")
            del retained

    print(f"\nrun -> {run.finish(summary)}")


if __name__ == "__main__":
    main()
