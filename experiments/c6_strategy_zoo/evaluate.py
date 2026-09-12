"""C6 -- score a learned policy, or its own baseline, on held-out sessions.

    python experiments/c6_strategy_zoo/evaluate.py --policy runs/c6/hfgrid-...__/policy.pt
    python experiments/c6_strategy_zoo/evaluate.py --policy baseline --strategy hfgrid --days test

Each session is scored with `hftbacktest.stats.LinearAssetRecord` on the
equity record the environment itself produced -- the same call `run_baseline.py`
uses -- so a learned row and a baseline row in the master table mean the same
thing.

Both the per-session medians and the pooled period are reported.  C3 learned
the hard way that these rank strategies almost inversely here: a policy with
ten flat sessions and one catastrophe has an excellent median session and a
terrible track record, and the track record is the thing an operator gets.
The period numbers are the headline; the medians are printed next to them so
the gap is visible rather than a choice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from hftbacktest.stats import LinearAssetRecord

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "a5_continuous_control"))
sys.path.insert(0, str(HERE.parent / "c1_glft_baseline"))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import assets  # noqa: E402
from continuous import Config, build_agent  # noqa: E402
from env import ZooEnv  # noqa: E402
from train import ACTION_NAMES, ALGOS  # noqa: E402

from rl_ladder.run import Run  # noqa: E402

REPORTED = ["SR", "Sortino", "Return", "MaxDrawdown", "ReturnOverMDD",
            "ReturnOverTrade", "MaxPositionValue", "DailyNumberOfTrades"]


def held_out_days(symbol: str) -> list[str]:
    assets.ensure_registered(symbol)
    return assets.test_days(symbol)


def train_days(symbol: str) -> list[str]:
    assets.ensure_registered(symbol)
    return assets.train_days(symbol)


def load_policy(path: Path):
    """Rebuild the actor from a checkpoint, plus everything it was trained with."""
    blob = torch.load(path, weights_only=False)
    cfg = blob["config"]
    width = cfg.get("hidden", 64)
    config = Config(hidden=(width, width), **ALGOS[blob["algo"]])
    strategy = blob.get("strategy", cfg["strategy"])
    symbol = blob.get("symbol", cfg["symbol"])
    overrides = {"grid_num": cfg["grid_num"]} if cfg.get("grid_num") else {}
    env = ZooEnv(date=train_days(symbol)[0], strategy=strategy, symbol=symbol,
                 overrides=overrides, beta=cfg.get("beta", 1.0))
    obs_dim = int(np.prod(env.observation_space.shape))
    env.close()
    # The actor's action space is the learnable multipliers only; pinned ones
    # are constants of the environment, not columns of the action.
    act_dim = int(blob["act_dim"])
    agent = build_agent(config, obs_dim, act_dim, 1.0)
    agent.actor.load_state_dict(blob["actor"])
    agent.actor.eval()
    # Evaluating a forced-participation policy under free ranges replays it
    # through a different action mapping than it was trained with, which
    # silently reports a different policy.  C3 shipped that bug once.
    return {
        "agent": agent,
        "baseline": np.asarray(blob["baseline"], dtype=np.float32),
        "mask": list(blob["mask"]),
        "is_sac": blob["algo"] == "sac",
        "participation": blob["participation"],
        "strategy": strategy,
        "symbol": symbol,
        "overrides": overrides,
        "beta": cfg.get("beta", 1.0),
    }


@torch.no_grad()
def act(agent, obs, is_sac: bool) -> np.ndarray:
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    if is_sac:
        action, _ = agent.actor(obs_t, deterministic=True)
    else:
        action = agent.actor(obs_t)
    return action.squeeze(0).numpy().astype(np.float32)


def run_session(date: str, policy: dict | None, strategy: str, symbol: str,
                overrides: dict, beta: float, participation: str) -> dict:
    ranges = ZooEnv.FORCED_PARTICIPATION if participation == "forced" else None
    env = ZooEnv(date=date, strategy=strategy, symbol=symbol, allow_held_out=True,
                 action_ranges=ranges, overrides=overrides, beta=beta)
    obs, _ = env.reset(seed=0)
    baseline_action = env.baseline_action()
    multipliers, inventories = [], []
    truncated = False
    while not truncated:
        if policy is None:
            action = baseline_action
        else:
            learned = np.clip(act(policy["agent"], obs, policy["is_sac"]), -1.0, 1.0)
            action = policy["baseline"].copy()
            action[policy["mask"]] = learned
        obs, _, _, truncated, info = env.step(action)
        multipliers.append([info[name] for name in ACTION_NAMES])
        inventories.append(info["inventory"])

    record = env.equity_record()
    log = env.quoter_log()
    summary = LinearAssetRecord(record).resample("10s").stats(book_size=env.book_size).summary()
    row = summary.to_dicts()[0]
    stats = {key: float(row[key]) for key in REPORTED if key in row}
    stats["date"] = date
    stats["fills"] = int(info["num_trades"])
    stats["mean_abs_inventory"] = float(np.mean(np.abs(inventories)))
    stats["median_half_spread_bp"] = float(np.nanmedian(log["half_spread_bp"]))
    stats["median_abs_signal_bp"] = float(np.nanmedian(np.abs(log["signal_bp"])))
    stats["median_A"] = float(np.nanmedian(log["A"]))
    stats["median_k"] = float(np.nanmedian(log["k"]))
    stats["hours"] = float(len(log) * env.quote_interval_ns / 3.6e12)
    stats["fills_per_hour"] = stats["fills"] / max(stats["hours"], 1e-9)
    mean_mult = np.mean(multipliers, axis=0)
    stats |= {name: float(value) for name, value in zip(ACTION_NAMES, mean_mult, strict=True)}
    # The record carries balance, position, price and fee, not equity: mark
    # it to market here the same way the environment and the baseline plots
    # do, so all three equity curves are the same quantity.
    marked = record["balance"] + record["position"] * record["price"] - record["fee"]
    # Inventory is stored normalised by the notional cap, not in units of the
    # asset, so that an ETH panel and an XRP panel mean the same thing.
    norm = log["position"] * log["mid"] / env.max_position_dollar
    stats["equity_path"] = marked[:: max(1, len(marked) // 4000)].tolist()
    stats["position_path"] = norm[:: max(1, len(norm) // 4000)].tolist()
    env.close()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, help="path to policy.pt, or 'baseline'")
    parser.add_argument("--strategy", default="glft", help="only used with --policy baseline")
    parser.add_argument("--symbol", default="ethusdc", help="only used with --policy baseline")
    parser.add_argument("--participation", default="forced", help="only used with --policy baseline")
    parser.add_argument("--grid-num", type=int, default=None)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--days", default="test", choices=["test", "train"])
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    if args.policy == "baseline":
        policy = None
        strategy, symbol = args.strategy, args.symbol
        overrides = {"grid_num": args.grid_num} if args.grid_num else {}
        beta, participation = args.beta, args.participation
        name = f"baseline-{strategy}-{symbol}"
    else:
        policy = load_policy(Path(args.policy))
        strategy, symbol = policy["strategy"], policy["symbol"]
        overrides, beta = policy["overrides"], policy["beta"]
        participation = policy["participation"]
        name = Path(args.policy).parent.name

    days = held_out_days(symbol) if args.days == "test" else train_days(symbol)
    run = Run(phase="c6-eval", name=args.tag or name, config={
        "policy": args.policy, "strategy": strategy, "symbol": symbol,
        "days": days, "day_set": args.days, "participation": participation,
        "grid_num": overrides.get("grid_num"), "beta": beta,
    })

    print(f"{name}  {strategy} on {symbol}, {args.days} ({len(days)} sessions)")
    print(f"{'date':>12} {'SR':>8} {'Return':>9} {'MaxDD':>8} {'fills':>8} {'|q|':>6} "
          f"{'bp':>7} {'spread':>7} {'skew':>7} {'alpha':>7} {'grid':>6}")
    rows = []
    for i, date in enumerate(days):
        stats = run_session(date, policy, strategy, symbol, overrides, beta, participation)
        rows.append(stats)
        run.log(step=i, **{k: v for k, v in stats.items() if not k.endswith("_path")})
        print(f"{date:>12} {stats['SR']:8.2f} {stats['Return']:9.4f} "
              f"{stats['MaxDrawdown']:8.4f} {stats['fills']:8d} "
              f"{stats['mean_abs_inventory']:6.3f} {stats['median_half_spread_bp']:7.2f} "
              f"{stats['spread_mult']:7.3f} {stats['skew_mult']:7.3f} "
              f"{stats['alpha_mult']:7.3f} {stats['grid_mult']:6.3f}", flush=True)

    # The period: the sessions chained as one track record, which is what an
    # operator running this every day would have got.
    returns = np.array([r["Return"] for r in rows])
    cumulative = float(np.prod(1.0 + returns) - 1.0)
    period = {
        "cumulative_return": cumulative,
        "sharpe_of_daily": float(np.mean(returns) / np.std(returns) * np.sqrt(365))
        if np.std(returns) > 0 else float("nan"),
        "worst_session": float(returns.min()),
        "sessions_profitable": int((returns > 0).sum()),
        "total_fills": int(sum(r["fills"] for r in rows)),
        "median_session_SR": float(np.median([r["SR"] for r in rows])),
        "median_half_spread_bp": float(np.median([r["median_half_spread_bp"] for r in rows])),
        "median_fills_per_hour": float(np.median([r["fills_per_hour"] for r in rows])),
        "median_abs_inventory": float(np.median([r["mean_abs_inventory"] for r in rows])),
    }
    period |= {name: float(np.mean([r[name] for r in rows])) for name in ACTION_NAMES}

    print(f"\nperiod: cumulative {cumulative * 100:+.2f}% of book over {len(days)} sessions, "
          f"worst {period['worst_session'] * 100:+.2f}%, "
          f"{period['sessions_profitable']}/{len(days)} profitable, "
          f"{period['total_fills']} fills")
    run.finish({"per_day": [{k: v for k, v in r.items() if not k.endswith("_path")} for r in rows],
                "period": period, "strategy": strategy, "symbol": symbol})
    np.savez_compressed(
        run.dir / "paths.npz",
        **{f"equity_{r['date']}": np.asarray(r["equity_path"]) for r in rows},
        **{f"position_{r['date']}": np.asarray(r["position_path"]) for r in rows},
    )
    print(json.dumps(period, indent=2))


if __name__ == "__main__":
    main()
