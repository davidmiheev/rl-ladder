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

import assets  # noqa: E402
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

#: Training recipes.  `v1` is exactly what C6 first ran, kept so its extra
#: seeds are the same experiment.  `v2` bundles the three changes made after
#: v1 lost to its own baseline: a policy that starts *on* the baseline and can
#: only move away from it (`residual`), explored with small noise around it
#: rather than a first episode of uniformly random parameters, deciding every
#: five seconds so a discount of 0.99 spans minutes rather than a hundred
#: seconds, and rewarded against the same session replayed at the baseline
#: (`reward`, `cf_coef`) so that most of the market's own noise cancels.  The
#: two `v2-no-*` recipes each remove one of those, so their effects can be told
#: apart.
#:
#: `cf_coef = 0.5`, not 1.  Subtracting the *whole* baseline reward was the
#: first design and it was wrong: measured on a held-back training session,
#: it raised the per-step reward variance 1.1-2.3x on XRPUSDC, because near the
#: spread floor fills are sparse and a policy that widens slightly stops
#: trading while the baseline keeps going -- subtracting an uncorrelated
#: series only adds its noise.  The variance-optimal constant is
#: Cov(r, b)/Var(b), which came out between 0.31 and 0.77 across six
#: probes on two assets; 0.5 recovers most of the optimum in every one of them
#: and never increases variance (x0.45 to x0.89).
#:
#: `deviation_penalty = 0.001` bp per step on the mean squared distance from
#: the baseline action: holding one multiplier at 2^0.5 for a whole session
#: of five-second steps costs about 1 bp, the size of the effects in the
#: results table -- enough to keep the policy off the no-fill corner without
#: pinning it to the baseline.
RECIPES = {
    "v1": dict(step_seconds=1, residual=0.0, reward="absolute", cf_coef=0.0,
               deviation_penalty=0.0, explore_episodes=1, participation="forced"),
    "v2": dict(step_seconds=5, residual=2.0, reward="counterfactual", cf_coef=0.5,
               deviation_penalty=0.001, explore_episodes=0, participation="residual"),
    "v2-no-cf": dict(step_seconds=5, residual=2.0, reward="absolute", cf_coef=0.0,
                     deviation_penalty=0.001, explore_episodes=0, participation="residual"),
    "v2-no-residual": dict(step_seconds=5, residual=0.0, reward="counterfactual", cf_coef=0.5,
                           deviation_penalty=0.0, explore_episodes=1, participation="forced"),
    # C7: v2 with fourteen more observations -- deeper book, OFI, own returns
    # and BTCUSDT's book (`env.EXTRA_FEATURES`). The one question it asks is
    # whether the policy can use information C7's regressions found to be
    # small but real; everything else is v2's.
    "v2-deep": dict(step_seconds=5, residual=2.0, reward="counterfactual", cf_coef=0.5,
                    deviation_penalty=0.001, explore_episodes=0, participation="residual",
                    extra_features=True),
    # v2 with the step reward split into trading and holding PnL and the
    # holding part counted at a quarter: credit for market making, little for
    # direction. Only the reward differs from v2.
    "v2-edge": dict(step_seconds=5, residual=2.0, reward="counterfactual", cf_coef=0.5,
                    deviation_penalty=0.001, explore_episodes=0, participation="residual",
                    inventory_weight=0.25),
    # C4's network-scaling probes: v2 with a wider actor and critic, then the
    # same with four times as many gradient updates per environment step
    # (update-to-data 1/2 instead of 1/8), then a residual LayerNorm critic
    # (GPU only: ~70 ms an update on one CPU thread against ~7 ms on a T4).
    "v2-w256": dict(step_seconds=5, residual=2.0, reward="counterfactual", cf_coef=0.5,
                    deviation_penalty=0.001, explore_episodes=0, participation="residual",
                    actor_hidden=(256, 256), critic_hidden=(256, 256)),
    "v2-w256-utd4": dict(step_seconds=5, residual=2.0, reward="counterfactual", cf_coef=0.5,
                         deviation_penalty=0.001, explore_episodes=0, participation="residual",
                         actor_hidden=(256, 256), critic_hidden=(256, 256), update_every=2),
    # C8: RL on top of the markout rule. The baseline is whatever `--set`
    # makes it (a C8 choice: floor, kappa, skew, vol scale), and the policy
    # also sees the rule's running toxicity.
    "v2-mk": dict(step_seconds=5, residual=2.0, reward="counterfactual", cf_coef=0.5,
                  deviation_penalty=0.001, explore_episodes=0, participation="residual",
                  toxicity_feature=True),
    "v2-resln": dict(step_seconds=5, residual=2.0, reward="counterfactual", cf_coef=0.5,
                     deviation_penalty=0.001, explore_episodes=0, participation="residual",
                     actor_hidden=(256, 256), critic_hidden=(256, 256), critic_arch="resln",
                     update_every=2),
}


def zero_final_layer(actor) -> None:
    """Make an untrained deterministic actor output exactly zero.

    With residual action ranges, zero *is* the baseline, so the policy begins
    on the published-plus-floor parameters instead of at a random point, and
    only the last layer's weights start at zero -- the hidden layers keep their
    random initialisation, so the gradient into that layer is not zero.
    """
    final = [m for m in actor.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        final.weight.zero_()
        final.bias.zero_()


def training_days(symbol: str, train_set: str = "c6") -> list[str]:
    assets.ensure_registered(symbol)
    if train_set == "c8":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "c8_risk_controls"))
        import c8

        return c8.TRAIN
    return assets.train_days(symbol)


def episode(env, agent, config, buffer, rng, act_dim, baseline, mask, explore,
            updates_from, update_every):
    """One trading session. Returns its summary; fills the buffer as it goes."""
    obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
    total_reward, pnl_bp, actions, inventories, losses = 0.0, 0.0, [], [], []
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
            batch = buffer.sample(config.batch_size, rng, config.device)
            losses.append(update_step(config, agent, batch, step))

    multipliers = np.mean([env.to_multipliers(a) for a in actions], axis=0)
    summary = {
        "steps": step,
        "critic_loss": float(np.mean(losses)) if losses else float("nan"),
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
    parser.add_argument("--recipe", default="v1", choices=sorted(RECIPES))
    parser.add_argument("--device", default="cpu", help="cpu or cuda; cuda pays off for v2-resln")
    parser.add_argument("--train-set", default="c6", choices=["c6", "c8"],
                        help="c8: C8's 24 training sessions (08-26 to 09-18)")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="quoter parameter overrides for the baseline, e.g. markout_kappa=1")
    args = parser.parse_args()
    recipe = dict(RECIPES[args.recipe])
    args.explore_episodes = recipe["explore_episodes"]
    args.participation = recipe["participation"]
    args.update_every = recipe.get("update_every", args.update_every)

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    days = training_days(args.symbol, args.train_set)
    overrides = {"grid_num": args.grid_num} if args.grid_num else {}
    for item in args.set:
        name, _, value = item.partition("=")
        overrides[name] = float(value)
    ranges = ZooEnv.FORCED_PARTICIPATION if args.participation == "forced" else None
    env_kwargs = dict(strategy=args.strategy, symbol=args.symbol, max_steps=args.max_steps,
                      action_ranges=ranges, overrides=overrides, beta=args.beta,
                      step_interval_ns=int(recipe["step_seconds"] * 1e9),
                      residual=recipe["residual"], reward_mode=recipe["reward"],
                      cf_coef=recipe["cf_coef"],
                      deviation_penalty=recipe["deviation_penalty"],
                      extra_features=recipe.get("extra_features", False),
                      inventory_weight=recipe.get("inventory_weight", 1.0),
                      toxicity_feature=recipe.get("toxicity_feature", False),
                      # C8's training set includes 09-15..09-18, held out under C6's split.
                      allow_held_out=args.train_set == "c8")

    config = Config(buffer_size=500_000, learning_starts=0,
                    hidden=(args.hidden, args.hidden), **ALGOS[args.algo],
                    actor_hidden=recipe.get("actor_hidden"),
                    critic_hidden=recipe.get("critic_hidden"),
                    critic_arch=recipe.get("critic_arch", "mlp"), device=args.device)
    env = ZooEnv(date=days[0], **env_kwargs)
    obs_dim = int(np.prod(env.observation_space.shape))
    baseline = env.baseline_action()
    env.close()

    mask = list(LEARNABLE[args.learnable])
    act_dim = len(mask)
    agent = build_agent(config, obs_dim, act_dim, 1.0)
    if recipe["residual"]:
        zero_final_layer(agent.actor)
        zero_final_layer(agent.actor_target)
    buffer = ReplayBuffer(config.buffer_size, obs_dim, act_dim)

    tag = f"{args.participation}-{args.learnable}" if args.recipe == "v1" else args.recipe
    tag += "-c8" if args.train_set == "c8" else ""
    name = f"{args.strategy}-{args.symbol}-{args.algo}-{tag}-s{args.seed}"
    run = Run(phase="c6", name=name, config={
        "strategy": args.strategy, "symbol": args.symbol, "algo": args.algo, "seed": args.seed,
        "passes": args.passes, "learnable": args.learnable, "learnable_indices": mask,
        "update_every": args.update_every, "explore_episodes": args.explore_episodes,
        "train_days": days, "max_steps": args.max_steps, "participation": args.participation,
        "grid_num": args.grid_num, "beta": args.beta, "hidden": config.hidden,
        "gamma_rl": config.gamma, "batch_size": config.batch_size, "eta": 0.015,
        "recipe": args.recipe, **{f"recipe_{k}": v for k, v in recipe.items()},
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

    torch.save({"actor": {k: v.cpu() for k, v in agent.actor.state_dict().items()},
                "config": vars(args),
                "baseline": baseline, "mask": mask, "algo": args.algo,
                "act_dim": act_dim, "participation": args.participation,
                "strategy": args.strategy, "symbol": args.symbol,
                "recipe": args.recipe, "recipe_params": recipe},
               run.path("policy.pt"))
    run.finish({"history": history, "final": history[-1]})
    print(f"\nrun dir: {run.dir}")
    print(json.dumps({k: history[-1][k] for k in ("pnl_pct_of_book", "fills", "spread_mult")}, indent=2))


if __name__ == "__main__":
    main()
