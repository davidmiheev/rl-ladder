"""A1 item 2: sweep epsilon and the UCB exploration constant.

Produces regret curves per parameter value plus a Sutton & Barto style
parameter study: average reward over the second half of the run against the
parameter, on a log axis.
"""

from __future__ import annotations

import argparse

import numpy as np
from bandits import UCB1, EpsilonGreedy, GaussianBandit, run_episode

from rl_ladder import Run, plot_curves, seed_everything

EPS_VALUES = [0.0, 0.01, 0.04, 0.1, 0.25, 0.5]
UCB_C_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def sweep(make_agent, values, args, run, tag):
    """Run one parameter sweep, returning (regret curves, mean tail reward per value)."""
    curves, tail_reward, finals = {}, [], {}
    for value in values:
        seed_everything(0)
        results = [
            run_episode(make_agent(value), lambda g: GaussianBandit(args.arms, g), args.steps, seed)
            for seed in range(args.seeds)
        ]
        label = f"{tag}={value:g}"
        curves[label] = [r["cumulative_regret"] for r in results]
        tail_reward.append(float(np.mean([r["reward"][args.steps // 2 :].mean() for r in results])))
        finals[label] = float(np.mean([r["cumulative_regret"][-1] for r in results]))
        run.log(
            step=len(finals), param=tag, value=value,
            final_regret=finals[label], tail_reward=tail_reward[-1],
        )
        print(f"  {label:12s} regret {finals[label]:7.1f}   tail reward {tail_reward[-1]:.3f}")
    return curves, tail_reward, finals


def parameter_study(eps_reward, ucb_reward, out_path):
    """Average reward over the second half vs. parameter value, both series on one log axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(EPS_VALUES[1:], eps_reward[1:], marker="o", label="eps-greedy (eps)")
    ax.plot(UCB_C_VALUES, ucb_reward, marker="s", label="UCB1 (c)")
    ax.axhline(eps_reward[0], linestyle="--", linewidth=1, color="grey", label="greedy (eps=0)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("parameter value (log scale)")
    ax.set_ylabel("average reward, second half of run")
    ax.set_title("Parameter study")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--arms", type=int, default=10)
    args = parser.parse_args()

    run = Run(phase="a1", name="sweeps", config=vars(args))

    print("epsilon sweep:")
    eps_curves, eps_reward, eps_finals = sweep(
        lambda v: (lambda g: EpsilonGreedy(args.arms, g, eps=v)), EPS_VALUES, args, run, "eps"
    )
    print("UCB c sweep:")
    ucb_curves, ucb_reward, ucb_finals = sweep(
        lambda v: (lambda g: UCB1(args.arms, g, c=v)), UCB_C_VALUES, args, run, "c"
    )

    plot_curves(eps_curves, title="eps-greedy: cumulative regret by eps", xlabel="pull",
                ylabel="cumulative regret", out_path=run.path("sweep_eps.png"))
    plot_curves(ucb_curves, title="UCB1: cumulative regret by c", xlabel="pull",
                ylabel="cumulative regret", out_path=run.path("sweep_ucb.png"))
    parameter_study(eps_reward, ucb_reward, run.path("parameter_study.png"))

    best_eps = EPS_VALUES[int(np.argmax(eps_reward))]
    best_c = UCB_C_VALUES[int(np.argmax(ucb_reward))]
    print(f"best eps={best_eps:g}  best c={best_c:g}")
    out = run.finish(
        {
            "eps_final_regret": eps_finals,
            "ucb_final_regret": ucb_finals,
            "best_eps": best_eps,
            "best_c": best_c,
        }
    )
    print(f"run -> {out}")


if __name__ == "__main__":
    main()
