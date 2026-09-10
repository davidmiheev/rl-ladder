"""C3 item 1 -- DDPG, TD3 and SAC on the C2 coefficient policy.

    python experiments/c3_continuous_control/train.py --algo td3 --seed 0

The algorithms are A5's, imported rather than reimplemented: `build_agent`,
`select_action` and `update_step` come straight from
`experiments/a5_continuous_control/continuous.py`, and only the outer loop is
new. It has to be, for two reasons A5's does not face.

- **An episode is a trading session and costs about ninety seconds**, so the
  loop cycles over the five training days rather than resetting a cheap
  simulator, and it does one gradient update every `update_every` environment
  steps rather than one per step.
- **A session is 85,800 steps**, so a single pass over the training days
  already yields more than four hundred thousand transitions. Sample
  efficiency is not the constraint here; wall-clock time is.

What is logged, and why: C2's probe found that the best available policy on a
training day is the corner of the action space that takes zero fills and
returns exactly zero. A policy that finds it will look like a success on every
PnL metric. So fill counts and the mean coefficients are logged next to the
return from the first episode, not added after the first surprise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "a5_continuous_control"))
sys.path.insert(0, str(HERE.parent / "c1_glft_baseline"))
sys.path.insert(0, str(HERE.parent / "c2_env"))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import config as market_config  # noqa: E402
from continuous import Config, ReplayBuffer, build_agent, select_action, update_step  # noqa: E402
from env import MarketMakingEnv  # noqa: E402

from rl_ladder.run import Run  # noqa: E402

#: TD3 is DDPG plus these three, exactly as in A5.
TD3 = {"twin_critics": True, "policy_delay": 2, "target_noise": 0.2}
ALGOS = {
    "ddpg": {"algo": "ddpg"},
    "td3": {"algo": "td3", **TD3},
    "sac": {"algo": "sac", "twin_critics": True},
}

#: Which action components the policy is allowed to move.  C3's third item
#: asks whether every coefficient benefits from being state-conditioned, so
#: the ones held fixed are pinned at C1's calibrated values.
LEARNABLE = {
    "all": (0, 1, 2, 3),
    "no-gamma": (1, 2, 3),          # gamma fixed at its calibrated 0.05
    "spread-only": (2,),            # only the half-spread scale
}


def episode(env, agent, config, buffer, rng, act_dim, baseline, mask, explore,
            updates_from, update_every):
    """One trading session. Returns its summary; fills the buffer as it goes."""
    obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
    total_reward, pnl_bp, actions, inventories = 0.0, 0.0, [], []
    step = 0
    truncated = False
    while not truncated:
        step += 1
        if explore:
            free = rng.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
        else:
            free = select_action(agent, obs, config, rng, act_dim).astype(np.float32)
        action = baseline.copy()
        action[mask] = np.clip(free[mask], -1.0, 1.0)

        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs

        total_reward += reward
        pnl_bp += info["pnl_bp"]
        actions.append(action)
        inventories.append(info["inventory"])

        if buffer.size >= updates_from and step % update_every == 0:
            update_step(config, agent, buffer.sample(config.batch_size, rng), step)

    coefficients = np.mean([env.to_coefficients(a) for a in actions], axis=0)
    return {
        "steps": step,
        "return": float(total_reward),
        "pnl_bp": float(pnl_bp),
        "pnl_pct_of_book": float(info["equity"] / env.book_size * 100),
        "fills": int(info["num_trades"]),
        "mean_abs_inventory": float(np.mean(np.abs(inventories))),
        "gamma": float(coefficients[0]),
        "c1_scale": float(coefficients[1]),
        "half_spread_scale": float(coefficients[2]),
        "skew_scale": float(coefficients[3]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", default="td3", choices=sorted(ALGOS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--passes", type=int, default=2, help="passes over the training days")
    parser.add_argument("--learnable", default="all", choices=sorted(LEARNABLE))
    parser.add_argument(
        "--update-every",
        type=int,
        default=8,
        help="environment steps between gradient updates; a session yields 85,800 "
             "transitions, so data is not the scarce resource here, wall clock is",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=64,
        help="width of both hidden layers. A5 used 256 for Pendulum and "
             "LunarLander; this observation is 10-dimensional and the action "
             "4-dimensional, so 64 is 10k critic parameters instead of 140k and "
             "2.5x faster per update",
    )
    parser.add_argument("--explore-episodes", type=int, default=1,
                        help="episodes of uniform-random actions before the policy takes over")
    parser.add_argument("--symbol", default="ethusdc")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="cap the episode length, for smoke tests only")
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    config = Config(buffer_size=500_000, learning_starts=0,
                    hidden=(args.hidden, args.hidden), **ALGOS[args.algo])
    env = MarketMakingEnv(date=market_config.TRAIN_DAYS[0], symbol=args.symbol,
                          max_steps=args.max_steps)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    baseline = env.baseline_action()
    mask = list(LEARNABLE[args.learnable])
    env.close()

    agent = build_agent(config, obs_dim, act_dim, 1.0)
    buffer = ReplayBuffer(config.buffer_size, obs_dim, act_dim)

    name = f"{args.algo}-{args.learnable}-s{args.seed}"
    run = Run(phase="c3", name=name, config={
        "algo": args.algo, "seed": args.seed, "passes": args.passes,
        "learnable": args.learnable, "learnable_indices": mask,
        "update_every": args.update_every, "explore_episodes": args.explore_episodes,
        "symbol": args.symbol, "train_days": market_config.TRAIN_DAYS,
        "max_steps": args.max_steps,
        "gamma_rl": config.gamma, "batch_size": config.batch_size,
        "hidden": config.hidden,
        "eta": env.eta, "algo_overrides": ALGOS[args.algo],
    })

    print(f"{name}: {args.passes} passes over {len(market_config.TRAIN_DAYS)} training days")
    print(f"{'ep':>3} {'day':>12} {'return':>9} {'PnL bp':>9} {'% book':>8} {'fills':>8} "
          f"{'|q|':>6} {'gamma':>7} {'c1':>6} {'half':>6} {'skew':>6} {'sec':>6}")

    history = []
    index = 0
    for _ in range(args.passes):
        for date in market_config.TRAIN_DAYS:
            started = time.time()
            env = MarketMakingEnv(date=date, symbol=args.symbol, max_steps=args.max_steps)
            summary = episode(
                env, agent, config, buffer, rng, act_dim, baseline, mask,
                explore=index < args.explore_episodes, updates_from=config.batch_size,
                update_every=args.update_every,
            )
            env.close()
            summary |= {"episode": index, "date": date, "seconds": time.time() - started}
            history.append(summary)
            run.log(step=index, **summary)
            print(f"{index:3d} {date:>12} {summary['return']:9.1f} {summary['pnl_bp']:9.1f} "
                  f"{summary['pnl_pct_of_book']:8.3f} {summary['fills']:8d} "
                  f"{summary['mean_abs_inventory']:6.3f} {summary['gamma']:7.4f} "
                  f"{summary['c1_scale']:6.2f} {summary['half_spread_scale']:6.2f} "
                  f"{summary['skew_scale']:6.3f} {summary['seconds']:6.0f}", flush=True)
            index += 1

    torch.save({"actor": agent.actor.state_dict(), "config": vars(args),
                "baseline": baseline, "mask": mask, "algo": args.algo},
               run.path("policy.pt"))
    run.finish({"history": history, "final": history[-1]})
    print(f"\nrun dir: {run.dir}")
    print(json.dumps({k: history[-1][k] for k in
                      ("pnl_pct_of_book", "fills", "half_spread_scale")}, indent=2))


if __name__ == "__main__":
    main()
