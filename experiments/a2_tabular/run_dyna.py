"""A2 item 4: Dyna-Q's planning steps against plain Q-learning.

Sutton & Barto figure 8.3. `planning_steps=0` is exactly Q-learning, so the
comparison isolates the value of replaying a learned model — every arm sees the
same number of *real* environment steps per episode.
"""

from __future__ import annotations

import argparse

import numpy as np
from gridworld import DYNA_MAZE, GridWorld
from td import DynaQ, train

from rl_ladder import Run, plot_curves, seed_everything

PLANNING_STEPS = [0, 5, 50]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=0.1)
    args = parser.parse_args()

    run = Run(phase="a2", name="dyna", config=vars(args))
    optimal = 14  # shortest path through the S&B maze
    curves, summary = {}, {"optimal_path_length": optimal}

    for n in PLANNING_STEPS:
        seed_everything(0)
        lengths = []
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            env = GridWorld(DYNA_MAZE, step_reward=0.0, goal_reward=1.0)
            agent = DynaQ(env.n_states, env.n_actions, alpha=args.alpha, gamma=args.gamma,
                          eps=args.eps, planning_steps=n)
            _, episode_lengths = train(env, agent, args.episodes, rng, max_steps=10_000)
            lengths.append(episode_lengths.astype(float))

        label = "Q-learning (n=0)" if n == 0 else f"Dyna-Q (n={n})"
        curves[label] = lengths
        stacked = np.asarray(lengths)
        # Episodes until the policy is consistently within 20% of the optimal path.
        good = stacked.mean(axis=0) <= optimal * 1.2
        to_converge = int(np.argmax(good)) + 1 if good.any() else -1
        summary[label] = {
            "final_episode_length": float(stacked[:, -1].mean()),
            "second_episode_length": float(stacked[:, 1].mean()),
            "episodes_to_within_20pct": to_converge,
        }
        run.log(step=len(summary), agent=label, **summary[label])
        print(f"{label:18s} ep2 {stacked[:, 1].mean():8.1f} steps | "
              f"final {stacked[:, -1].mean():6.1f} | reaches 1.2x optimal at episode {to_converge}")

    plot_curves(
        curves,
        title=f"Dyna maze: steps per episode ({args.seeds} seeds, optimal = {optimal})",
        xlabel="episode",
        ylabel="steps to goal",
        out_path=run.path("dyna_steps.png"),
    )
    print(f"run -> {run.finish(summary)}")


if __name__ == "__main__":
    main()
