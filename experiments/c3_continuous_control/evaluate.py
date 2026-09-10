"""C3 item 2 -- score a learned policy against C1's baseline on held-out days.

    python experiments/c3_continuous_control/evaluate.py --policy runs/c3/td3-all-s0__.../policy.pt
    python experiments/c3_continuous_control/evaluate.py --policy baseline --days test

Runs the policy deterministically over a set of sessions and scores each with
`hftbacktest.stats.LinearAssetRecord` -- the same call C1 used for the fixed
baseline, on the same equity record -- so the two result tables mean the same
thing and can be put in one column.

Reports Sharpe, Sortino, MaxPositionValue and ReturnOverMDD as the item asks,
and fills alongside them, because C2's probe showed the highest-scoring policy
available on a training day is the one that never trades. A risk-adjusted
measure computed over a strategy that took eleven fills is not a
risk-adjusted measure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from hftbacktest.stats import LinearAssetRecord

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "a5_continuous_control"))
sys.path.insert(0, str(HERE.parent / "c1_glft_baseline"))
sys.path.insert(0, str(HERE.parent / "c2_env"))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import config as market_config  # noqa: E402
from continuous import Config, build_agent  # noqa: E402
from env import MarketMakingEnv  # noqa: E402
from train import ALGOS  # noqa: E402

from rl_ladder.run import Run  # noqa: E402

BOOK_SIZE = 100_000.0
REPORTED = ["SR", "Sortino", "Return", "MaxDrawdown", "ReturnOverMDD",
            "ReturnOverTrade", "MaxPositionValue", "DailyNumberOfTrades"]


def load_policy(path: Path):
    """Rebuild the actor from a checkpoint, plus the mask it was trained with."""
    blob = torch.load(path, weights_only=False)
    width = blob["config"].get("hidden", 64)
    config = Config(hidden=(width, width), **ALGOS[blob["algo"]])
    env = MarketMakingEnv(date=market_config.TRAIN_DAYS[0], symbol=blob["config"]["symbol"])
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    env.close()
    agent = build_agent(config, obs_dim, act_dim, 1.0)
    agent.actor.load_state_dict(blob["actor"])
    agent.actor.eval()
    return agent, np.asarray(blob["baseline"], dtype=np.float32), list(blob["mask"]), config


@torch.no_grad()
def act(agent, obs, is_sac: bool) -> np.ndarray:
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    if is_sac:
        action, _ = agent.actor(obs_t, deterministic=True)
    else:
        action = agent.actor(obs_t)
    return action.squeeze(0).numpy().astype(np.float32)


def run_day(date: str, symbol: str, agent, baseline, mask, is_sac: bool) -> dict:
    env = MarketMakingEnv(date=date, symbol=symbol, allow_held_out=True)
    obs, _ = env.reset(seed=0)
    coefficients = []
    truncated = False
    while not truncated:
        action = baseline.copy()
        if agent is not None:
            action[mask] = np.clip(act(agent, obs, is_sac)[mask], -1.0, 1.0)
        obs, _, _, truncated, info = env.step(action)
        coefficients.append(env.to_coefficients(action))
    record = env.equity_record()
    env.close()

    summary = LinearAssetRecord(record).resample("10s").stats(book_size=BOOK_SIZE).summary()
    row = summary.to_dicts()[0]
    stats = {key: float(row[key]) for key in REPORTED if key in row}
    mean = np.mean(coefficients, axis=0)
    stats |= {
        "date": date,
        "fills": int(info["num_trades"]),
        "gamma": float(mean[0]),
        "c1_scale": float(mean[1]),
        "half_spread_scale": float(mean[2]),
        "skew_scale": float(mean[3]),
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True,
                        help="path to a policy.pt, or the literal 'baseline'")
    parser.add_argument("--days", default="test", choices=["train", "test", "all"])
    parser.add_argument("--symbol", default="ethusdc")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    days = {"train": market_config.TRAIN_DAYS, "test": market_config.TEST_DAYS,
            "all": market_config.TRAIN_DAYS + market_config.TEST_DAYS}[args.days]

    if args.policy == "baseline":
        env = MarketMakingEnv(date=market_config.TRAIN_DAYS[0], symbol=args.symbol)
        agent, baseline, mask, is_sac = None, env.baseline_action(), [], False
        env.close()
        label = args.label or "baseline"
    else:
        agent, baseline, mask, config = load_policy(Path(args.policy))
        is_sac = config.algo == "sac"
        label = args.label or Path(args.policy).parent.name

    run = Run(phase="c3", name=f"eval-{label}-{args.days}",
              config={"policy": args.policy, "days": days, "symbol": args.symbol})

    print(f"{label} on {args.days} days")
    print(f"{'date':>12} {'SR':>9} {'Sortino':>9} {'Return':>9} {'RoMDD':>8} {'MaxPos':>10} "
          f"{'fills':>8} {'half':>6} {'skew':>6}")
    rows = []
    for i, date in enumerate(days):
        stats = run_day(date, args.symbol, agent, baseline, mask, is_sac)
        rows.append(stats)
        run.log(step=i, **stats)
        print(f"{date:>12} {stats['SR']:9.2f} {stats['Sortino']:9.2f} {stats['Return']:9.4f} "
              f"{stats['ReturnOverMDD']:8.3f} {stats['MaxPositionValue']:10.0f} "
              f"{stats['fills']:8d} {stats['half_spread_scale']:6.2f} "
              f"{stats['skew_scale']:6.3f}", flush=True)

    summary = {}
    print(f"\n{'metric':>22} {'median':>10} {'min':>10} {'max':>10}")
    for key in [*REPORTED, "fills"]:
        values = np.array([row[key] for row in rows], dtype=float)
        summary[f"{key}_median"] = float(np.median(values))
        summary[f"{key}_per_day"] = values.tolist()
        print(f"{key:>22} {np.median(values):10.3f} {values.min():10.3f} {values.max():10.3f}")
    profitable = int(sum(row["Return"] > 0 for row in rows))
    print(f"{'days with Return > 0':>22} {profitable} / {len(rows)}")
    summary |= {"days_profitable": profitable, "n_days": len(rows), "per_day": rows,
                "label": label}
    run.finish(summary)
    print(f"\nrun dir: {run.dir}")


if __name__ == "__main__":
    main()
