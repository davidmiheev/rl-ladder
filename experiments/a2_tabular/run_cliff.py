"""A2 item 2: Q-learning vs SARSA near a hazard (Sutton & Barto example 6.6).

Both agents are epsilon-greedy with the same epsilon, alpha and gamma. The only
difference is the bootstrap target, so any divergence in behaviour is the
on-policy/off-policy distinction and nothing else.
"""

from __future__ import annotations

import argparse

import numpy as np
from gridworld import CLIFF, GridWorld
from mdp import value_iteration
from td import QLearning, Sarsa, train

from rl_ladder import Run, plot_curves, seed_everything


def greedy_return(env: GridWorld, agent, max_steps: int = 200) -> tuple[float, bool]:
    """Roll out the greedy (epsilon=0) policy once.

    Returns `(return, reached_goal)`. The flag matters: a greedy policy extracted
    from on-policy values can contain an absorbing wall-bump cycle that the
    epsilon-greedy behaviour policy would have escaped by exploring, in which
    case the return is just the step cap and says nothing about path quality.
    """
    rng = np.random.default_rng(0)
    policy = agent.greedy_policy()
    state, total = env.reset(), 0.0
    for _ in range(max_steps):
        state, reward, done = env.step(int(policy[state]), rng)
        total += reward
        if done:
            return total, True
    return total, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args()

    run = Run(phase="a2", name="cliff", config=vars(args))

    optimal_v, _, _ = value_iteration(GridWorld(CLIFF).to_mdp(), gamma=args.gamma)
    optimal_return = float(optimal_v[GridWorld(CLIFF).start])
    print(f"DP optimal return from start: {optimal_return:.1f}\n")

    curves, summary = {}, {"dp_optimal_return": optimal_return}
    for cls in (QLearning, Sarsa):
        seed_everything(0)
        returns, policies, greedy = [], [], []
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            env = GridWorld(CLIFF)
            agent = cls(env.n_states, env.n_actions, alpha=args.alpha, gamma=args.gamma, eps=args.eps)
            episode_returns, _ = train(env, agent, args.episodes, rng)
            returns.append(episode_returns)
            policies.append(agent.greedy_policy())
            greedy.append(greedy_return(GridWorld(CLIFF), agent))

        label = cls.__name__
        curves[label] = returns
        final = np.array([r[-100:].mean() for r in returns])
        values = np.array([g[0] for g in greedy])
        reached = np.array([g[1] for g in greedy])
        summary[label] = {
            "online_return_last100": float(final.mean()),
            "std": float(final.std()),
            "greedy_return_mean": float(values.mean()),
            "greedy_return_median": float(np.median(values)),
            "greedy_reaches_goal_frac": float(reached.mean()),
            "greedy_optimal_frac": float(np.mean(values == optimal_return)),
        }
        run.log(step=len(summary), agent=label, **summary[label])
        print(f"{label:10s} online {final.mean():7.1f} +/- {final.std():4.1f}"
              f"   | greedy return median {np.median(values):6.1f} mean {values.mean():7.1f}"
              f"   | reaches goal {100 * reached.mean():3.0f}%"
              f" | optimal {100 * np.mean(values == optimal_return):3.0f}%")
        print(GridWorld(CLIFF).render_policy(policies[0]))
        print()

    plot_curves(
        curves,
        title=f"Cliff walking: online return per episode ({args.seeds} seeds, 20-ep moving average)",
        xlabel="episode",
        ylabel="sum of rewards",
        out_path=run.path("cliff_return.png"),
        smooth_window=20,
    )
    print(f"run -> {run.finish(summary)}")


if __name__ == "__main__":
    main()
