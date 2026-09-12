"""C6 -- TD3 on the four multipliers, for each of the five strategies.

    python experiments/c6_strategy_zoo/train.py --strategy hfgrid --seed 0

The algorithm is A5's, imported rather than reimplemented, exactly as C3 did:
`build_agent`, `select_action` and `update_step` come from
`experiments/a5_continuous_control/continuous.py`.  Only the outer loop is
C6's, and it is C3's loop with the environment swapped.

Two decisions that are compute, not science, and are recorded as such:

- **TD3 only.**  C3 ran DDPG, TD3 and SAC across three seeds on one strategy
  and TD3 won on held-out cumulative return (family medians -2.19 against
  -4.54 and -4.85).  Repeating the three-way comparison for five strategies
  would be fifteen times the work to re-answer a question C3 answered, on a
  machine with four cores.  So C6 takes C3's winner and spends the budget on
  breadth instead.
- **Two seeds, two passes.**  C3 found the seed spread on this problem is
  large (3.74 points of return between the best and worst seed of a family
  with a free gamma), so two seeds is *not* enough to rank two configurations
  that are close.  It is enough to notice when a result is seed-dependent,
  which is what the master table reports.  Any C6 claim that rests on a gap
  smaller than the seed spread is flagged rather than made.

`--participation forced` is the default here, unlike C3, because C3 settled
that question: with a free spread range the highest-scoring policy available
on a training day is the corner that quotes so far out it takes no fills.
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
sys.path.insert(0, str(HERE.parents[1] / "src"))

import config as market_config  # noqa: E402
import strategies  # noqa: E402
from continuous import Config, ReplayBuffer, build_agent, select_action, update_step  # noqa: E402
from env import ZooEnv  # noqa: E402

from rl_ladder.run import Run  # noqa: E402

#: TD3 is DDPG plus these three, exactly as in A5.
TD3 = {"twin_critics": True, "policy_delay": 2, "target_noise": 0.2}
ALGOS = {
    "ddpg": {"algo": "ddpg"},
    "td3": {"algo": "td3", **TD3},
    "sac": {"algo": "sac", "twin_critics": True},
}

#: Which multipliers the policy may move.  `no-alpha` pins the strategy's own
#: signal at its published weight, which isolates "did RL improve the quoting"
#: from "did RL re-weight the signal" -- C6's counterpart of C3's gamma
#: ablation, and the question C5's second item asks.
LEARNABLE = {
    "all": (0, 1, 2, 3),
    "no-alpha": (0, 1, 3),
    "spread-only": (0,),
}

ACTION_NAMES = ("spread_mult", "skew_mult", "alpha_mult", "grid_mult")


def training_days(symbol: str) -> list[str]:
    pair = symbol in market_config.REFERENCE_LEG
    return market_config.PAIR_TRAIN_DAYS if pair else market_config.TRAIN_DAYS_EXT


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
            learned = rng.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
        else:
            learned = select_action(agent, obs, config, rng, act_dim).astype(np.float32)
        learned = np.clip(learned, -1.0, 1.0)
        action = baseline.copy()
        action[mask] = learned

        next_obs, reward, terminated, truncated, info = env.step(action)
        # The buffer stores the *learnable* action, which is the agent's whole
        # action space; the pinned multipliers are constants of the
        # environment, not a column of zero variance for the critic to have an
        # unidentified gradient in.
        buffer.add(obs, learned, reward, next_obs, terminated)
        obs = next_obs

        total_reward += reward
        pnl_bp += info["pnl_bp"]
        actions.append(action)
        inventories.append(info["inventory"])

        if buffer.size >= updates_from and step % update_every == 0:
            update_step(config, agent, buffer.sample(config.batch_size, rng), step)

    multipliers = np.mean([env.to_multipliers(a) for a in actions], axis=0)
    summary = {
        "steps": step,
        "return": float(total_reward),
        "pnl_bp": float(pnl_bp),
        "pnl_pct_of_book": float(info["equity"] / env.book_size * 100),
        "fills": int(info["num_trades"]),
        "mean_abs_inventory": float(np.mean(np.abs(inventories))),
        "half_spread_bp": float(info["half_spread_bp"]),
    }
    summary |= {name: float(value) for name, value in zip(ACTION_NAMES, multipliers, strict=True)}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="glft", choices=sorted(strategies.NAMES))
    parser.add_argument("--algo", default="td3", choices=sorted(ALGOS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--passes", type=int, default=2, help="passes over the training days")
    parser.add_argument("--learnable", default="all", choices=sorted(LEARNABLE))
    parser.add_argument("--update-every", type=int, default=8,
                        help="environment steps between gradient updates")
    parser.add_argument("--hidden", type=int, default=64, help="width of both hidden layers")
    parser.add_argument("--explore-episodes", type=int, default=1)
    parser.add_argument("--symbol", default="ethusdc")
    parser.add_argument("--beta", type=float, default=1.0, help="APT's reference-leg beta")
    parser.add_argument("--grid-num", type=int, default=None)
    parser.add_argument("--participation", default="forced", choices=["free", "forced"])
    parser.add_argument("--max-steps", type=int, default=0, help="smoke tests only")
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    days = training_days(args.symbol)
    overrides = {"grid_num": args.grid_num} if args.grid_num else {}
    ranges = ZooEnv.FORCED_PARTICIPATION if args.participation == "forced" else None
    env_kwargs = dict(strategy=args.strategy, symbol=args.symbol, max_steps=args.max_steps,
                      action_ranges=ranges, overrides=overrides, beta=args.beta)

    config = Config(buffer_size=500_000, learning_starts=0,
                    hidden=(args.hidden, args.hidden), **ALGOS[args.algo])
    env = ZooEnv(date=days[0], **env_kwargs)
    obs_dim = int(np.prod(env.observation_space.shape))
    baseline = env.baseline_action()
    env.close()

    mask = list(LEARNABLE[args.learnable])
    act_dim = len(mask)
    agent = build_agent(config, obs_dim, act_dim, 1.0)
    buffer = ReplayBuffer(config.buffer_size, obs_dim, act_dim)

    name = f"{args.strategy}-{args.symbol}-{args.algo}-{args.participation}-{args.learnable}-s{args.seed}"
    run = Run(phase="c6", name=name, config={
        "strategy": args.strategy, "symbol": args.symbol, "algo": args.algo, "seed": args.seed,
        "passes": args.passes, "learnable": args.learnable, "learnable_indices": mask,
        "update_every": args.update_every, "explore_episodes": args.explore_episodes,
        "train_days": days, "max_steps": args.max_steps, "participation": args.participation,
        "grid_num": args.grid_num, "beta": args.beta, "hidden": config.hidden,
        "gamma_rl": config.gamma, "batch_size": config.batch_size, "eta": 0.015,
        "algo_overrides": ALGOS[args.algo],
    })

    print(f"{name}: {args.passes} passes over {len(days)} training days")
    print(f"{'ep':>3} {'day':>12} {'return':>9} {'PnL bp':>9} {'% book':>8} {'fills':>8} "
          f"{'|q|':>6} {'spread':>7} {'skew':>7} {'alpha':>7} {'grid':>6} {'bp':>6} {'sec':>6}")

    history = []
    index = 0
    for _ in range(args.passes):
        for date in days:
            started = time.time()
            env = ZooEnv(date=date, **env_kwargs)
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
                  f"{summary['mean_abs_inventory']:6.3f} {summary['spread_mult']:7.3f} "
                  f"{summary['skew_mult']:7.3f} {summary['alpha_mult']:7.3f} "
                  f"{summary['grid_mult']:6.3f} {summary['half_spread_bp']:6.2f} "
                  f"{summary['seconds']:6.0f}", flush=True)
            index += 1

    torch.save({"actor": agent.actor.state_dict(), "config": vars(args),
                "baseline": baseline, "mask": mask, "algo": args.algo,
                "act_dim": act_dim, "participation": args.participation,
                "strategy": args.strategy, "symbol": args.symbol},
               run.path("policy.pt"))
    run.finish({"history": history, "final": history[-1]})
    print(f"\nrun dir: {run.dir}")
    print(json.dumps({k: history[-1][k] for k in ("pnl_pct_of_book", "fills", "spread_mult")}, indent=2))


if __name__ == "__main__":
    main()
