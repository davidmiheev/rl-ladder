"""A8: behavioural cloning vs inverse RL, measured on and off the demonstrations.

Everything is scored against the exact optimal policy, which the gridworld's
solver provides, so "how good is this policy" is answered absolutely rather
than by comparing the two learners to each other.

Item 1's compounding error is measured directly: the fraction of a rollout
spent in states the demonstrations never covered. An agent that drifts off the
expert's distribution and then has no idea what to do will show a high
off-distribution fraction *and* a low success rate, and the two should move
together.
"""

from __future__ import annotations

import argparse

import numpy as np
from gridworld import GridWorld
from methods import (
    behavioural_cloning_features,
    behavioural_cloning_tabular,
    collect_demonstrations,
    maxent_irl,
)

from rl_ladder import Run


def evaluate(env, policy, starts, seen_mask, rng, episodes: int = 30):
    """Success rate, mean return, and time spent off the demonstrated states."""
    successes, returns, off_fraction = [], [], []
    for start in starts:
        for _ in range(episodes):
            states, _, total, reached = env.rollout(policy, start, rng)
            successes.append(float(reached))
            returns.append(total)
            if states:
                off_fraction.append(float(np.mean([not seen_mask[s] for s in states])))
    return (float(np.mean(successes)), float(np.mean(returns)),
            float(np.mean(off_fraction)) if off_fraction else 0.0)


def one_seed(seed: int, args):
    rng = np.random.default_rng(seed)
    env = GridWorld()
    _, expert = env.value_iteration()
    starts = [env.start]
    pairs = collect_demonstrations(env, expert, starts, args.demos, rng)
    seen_mask = np.zeros(env.n_states, dtype=bool)
    for state, _ in pairs:
        seen_mask[state] = True

    free = env.free_states()
    unseen = np.array([s for s in free if not seen_mask[s]])

    bc_tab, _ = behavioural_cloning_tabular(env, pairs, rng)
    bc_feat = behavioural_cloning_features(env, pairs, rng)
    theta = maxent_irl(env, pairs, starts, n_trajectories=len(starts) * args.demos,
                       iterations=args.irl_iters)
    _, irl_policy = env.value_iteration(rewards=theta)

    policies = {
        "expert (reference)": expert,
        "BC tabular": bc_tab,
        "BC features": bc_feat,
        "IRL then RL": irl_policy,
    }

    out = {}
    for name, policy in policies.items():
        greedy = policy if policy.ndim == 1 else policy.argmax(axis=1)
        row = {
            "agree_all": float(np.mean([greedy[s] == expert[s] for s in free])),
            "agree_seen": float(np.mean([greedy[s] == expert[s] for s in free if seen_mask[s]])),
            "agree_unseen": float(np.mean([greedy[s] == expert[s] for s in unseen])),
        }
        for label, start_set in (("demo_start", starts), ("all_free", free), ("unseen", unseen)):
            success, ret, off = evaluate(env, policy, start_set, seen_mask, rng,
                                         episodes=args.episodes)
            row[f"{label}_success"] = success
            row[f"{label}_return"] = ret
            row[f"{label}_off_distribution"] = off
        out[name] = row

    nonwall = np.flatnonzero(~env.walls)
    out["_meta"] = {
        "coverage": float(seen_mask[free].mean()),
        "n_unseen": int(len(unseen)),
        "reward_correlation": float(np.corrcoef(theta[nonwall], env.rewards[nonwall])[0, 1]),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--demos", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--irl-iters", type=int, default=150)
    args = parser.parse_args()

    run = Run(phase="a8", name="imitation", config=vars(args))
    results = [one_seed(seed, args) for seed in range(args.seeds)]

    meta = {k: float(np.mean([r["_meta"][k] for r in results])) for k in results[0]["_meta"]}
    print(f"demonstrations cover {meta['coverage']:.1%} of free states "
          f"({meta['n_unseen']:.0f} never visited) | "
          f"recovered-vs-true reward correlation {meta['reward_correlation']:+.3f}\n")

    summary = {"meta": meta}
    names = [n for n in results[0] if n != "_meta"]
    print(f"{'policy':20s} {'agree:seen':>11s} {'agree:unseen':>13s} "
          f"{'succ@demo':>10s} {'succ@all':>9s} {'succ@unseen':>12s} {'off-dist':>9s}")
    for name in names:
        agg = {k: float(np.mean([r[name][k] for r in results])) for k in results[0][name]}
        agg["agree_unseen_std"] = float(np.std([r[name]["agree_unseen"] for r in results]))
        summary[name] = agg
        run.log(step=len(summary), policy=name, **agg)
        print(f"{name:20s} {agg['agree_seen']:11.3f} {agg['agree_unseen']:13.3f} "
              f"{agg['demo_start_success']:10.3f} {agg['all_free_success']:9.3f} "
              f"{agg['unseen_success']:12.3f} {agg['demo_start_off_distribution']:9.3f}")

    print(f"\nrun -> {run.finish(summary)}")


if __name__ == "__main__":
    main()
