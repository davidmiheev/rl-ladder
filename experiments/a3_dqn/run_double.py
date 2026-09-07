"""A3 item 2: does Double DQN reduce Q-value overestimation?

Overestimation comes from taking a max over *noisy* estimates: E[max] >= max[E].
Deterministic CartPole has little estimate noise to amplify, so the comparison is
run twice — once as-is, and once with Gaussian reward noise, which supplies the
noise the bias needs without changing the optimal policy.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool

import gymnasium as gym
import numpy as np
from dqn import Config, train

from rl_ladder import Run, plot_curves


class NoisyReward(gym.RewardWrapper):
    """Add zero-mean Gaussian noise to the reward. Optimal policy is unchanged."""

    def __init__(self, env, sigma: float, seed: int):
        super().__init__(env)
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def reward(self, reward):
        return float(reward + self.rng.normal(0.0, self.sigma))


def register_noisy(sigma: float) -> str:
    env_id = f"NoisyCartPole-sigma{sigma:g}-v0"
    gym.register(
        id=env_id,
        entry_point=lambda sigma=sigma, **kw: NoisyReward(gym.make("CartPole-v1"), sigma, seed=0),
        max_episode_steps=500,
    )
    return env_id


def one_run(job):
    label, overrides, seed, steps, env_id, sigma = job
    if sigma > 0:
        env_id = register_noisy(sigma)
    return label, seed, train(Config(total_steps=steps, env_id=env_id, **overrides), seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    run = Run(phase="a3", name="double", config=vars(args))
    arms = {"DQN": {"double": False}, "Double DQN": {"double": True}}

    jobs, summary, bias_curves = [], {}, {}
    for sigma in (0.0, args.sigma):
        for label, overrides in arms.items():
            for seed in range(args.seeds):
                tag = f"{label}{'' if sigma == 0 else f' (noise {sigma:g})'}"
                jobs.append((tag, overrides, seed, args.steps, "CartPole-v1", sigma))

    with Pool(args.workers) as pool:
        results = pool.map(one_run, jobs)

    for tag in dict.fromkeys(j[0] for j in jobs):
        runs = [m for lab, _, m in results if lab == tag]
        # Overestimation = what the net predicts at the start state, minus what
        # the greedy policy actually collects from there.
        bias = [np.asarray(m.predicted_values) - np.asarray(m.actual_returns) for m in runs]
        bias_curves[tag] = bias
        final_bias = np.array([b[-1] for b in bias])
        final_return = np.array([m.eval_returns[-1] for m in runs])
        summary[tag] = {
            "final_overestimation": float(final_bias.mean()),
            "final_overestimation_std": float(final_bias.std()),
            "final_eval_return": float(final_return.mean()),
        }
        run.log(step=len(summary), arm=tag, **summary[tag])
        print(f"{tag:26s} overestimation {final_bias.mean():+7.2f} +/- {final_bias.std():5.2f}"
              f"   eval return {final_return.mean():6.1f}")

    plot_curves(bias_curves, title="Q-value overestimation: predicted V(s0) - actual return",
                xlabel="evaluation", ylabel="overestimation", out_path=run.path("overestimation.png"))
    print(f"run -> {run.finish(summary)}")


if __name__ == "__main__":
    main()
