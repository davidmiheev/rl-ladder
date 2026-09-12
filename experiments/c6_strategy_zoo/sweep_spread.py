"""Sweep a strategy parameter on training days, and set the asset's spread floor.

    python experiments/c6_strategy_zoo/sweep_spread.py --strategy glft --param half_spread_bp --write
    python experiments/c6_strategy_zoo/sweep_spread.py --strategy glft --param grid_num

Two sweeps, one script, because they ask the same question of the same
machinery:

- **`half_spread_bp`** -- how wide does this asset have to be quoted?  C1
  found the calibrated GLFT half spread (about 1.7 bp on ETHUSDC) loses to
  adverse selection, and `measure_adverse_selection.py` puts a number on the
  cost per fill.  But the tape cannot say which fills a resting quote wins:
  queue position and latency decide that, and they are exactly what the
  simulator models.  So the floor is set here, by running the strategy at each
  width and reading off the best, rather than from the tape estimate -- which
  disagrees by about 0.9 bp per fill and would put the floor near zero.

- **`grid_num`** -- how many orders deep should the grid be?  The notebooks
  use 20 (glft, hfgrid) or 1-5 (the alpha three) and none of them say why.
  Order size is held fixed, so a deeper grid does not take more risk: the
  notional cap still binds. What it buys is queue position at more price
  levels, and what it costs is cancel-replace traffic and fills further from
  the fair price.

Training days only, and a subset of them: the sweep is a search, so it is the
part of C6 most able to overfit, and it is deliberately given the least data
it can be run on rather than the most.  The chosen value is then applied
unchanged to the held-out set, where it is tested once.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "c1_glft_baseline"))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import config as market_config  # noqa: E402
import run_baseline  # noqa: E402
import strategies  # noqa: E402

from rl_ladder.run import Run  # noqa: E402

OUT = HERE / "spread_floor.json"

#: Half-spread floors to try, in bp.  0 means "whatever the strategy's own
#: model says", which for glft and hfgrid is the published behaviour and for
#: the alpha three is their published constant.  The rest span two orders of
#: magnitude around it, because C1's sweeps were all monotone and a grid that
#: does not contain a turning point cannot show one.
HALF_SPREAD_BP = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)

#: Grid depths to try.  1 and 5 are the alpha notebooks' values, 20 is the
#: grid notebooks', 40 and 60 go past all of them.
GRID_NUM = (1, 5, 20, 40, 60)

VALUES = {"half_spread_bp": HALF_SPREAD_BP, "grid_num": GRID_NUM}

#: Sweep days: the first four training days, which is the least data the
#: search can run on.  Named here rather than inline so the writeup can say
#: exactly which sessions chose the number.
def sweep_days(symbol: str, n: int) -> list[str]:
    days = (market_config.PAIR_TRAIN_DAYS if symbol in market_config.REFERENCE_LEG
            else market_config.TRAIN_DAYS_EXT)
    return days[:n]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--strategy", default="glft", choices=sorted(strategies.NAMES))
    parser.add_argument("--param", default="half_spread_bp", choices=sorted(VALUES))
    parser.add_argument("--symbol", default="ethusdc")
    parser.add_argument("--days", type=int, default=4, help="how many training days to sweep over")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--write", action="store_true",
                        help="record the chosen half spread in spread_floor.json")
    args = parser.parse_args()

    days = sweep_days(args.symbol, args.days)
    values = VALUES[args.param]
    run = Run(phase="c6-sweep", name=f"{args.param}-{args.strategy}-{args.symbol}", config={
        "strategy": args.strategy, "symbol": args.symbol, "param": args.param,
        "values": list(values), "days": days,
    })

    print(f"sweeping {args.param} for {args.strategy} on {args.symbol} "
          f"over {len(days)} training days: {days}")
    print(f"{args.param:>16} {'cumulative':>11} {'worst day':>10} {'SR median':>10} "
          f"{'fills':>9} {'half bp':>9} {'|q|':>7}")

    rows = []
    for i, value in enumerate(values):
        overrides = {"beta": args.beta, args.param: float(value)}
        returns, stats = [], []
        for date in days:
            day_stats, _, out = run_baseline.run_day(args.strategy, args.symbol, date, overrides)
            day_stats |= run_baseline.quoter_stats(out, strategies.INTERVAL_NS)
            returns.append(day_stats["Return"])
            stats.append(day_stats)
        returns = np.array(returns)
        row = {
            args.param: float(value),
            "cumulative_return": float(np.prod(1.0 + returns) - 1.0),
            "worst_day": float(returns.min()),
            "median_SR": float(np.median([s["SR"] for s in stats])),
            "total_fills": int(sum(s["DailyNumberOfTrades"] for s in stats)),
            "median_half_spread_bp": float(np.median([s["median_half_spread_bp"] for s in stats])),
            "median_abs_position": float(np.median([s["median_abs_position"] for s in stats])),
            "per_day_return": [float(r) for r in returns],
        }
        rows.append(row)
        run.log(step=i, **row)
        print(f"{value:16g} {row['cumulative_return'] * 100:10.3f}% "
              f"{row['worst_day'] * 100:9.3f}% {row['median_SR']:10.2f} "
              f"{row['total_fills']:9d} {row['median_half_spread_bp']:9.2f} "
              f"{row['median_abs_position']:7.2f}", flush=True)

    best = max(rows, key=lambda r: r["cumulative_return"])
    print(f"\nbest {args.param} = {best[args.param]:g}  "
          f"({best['cumulative_return'] * 100:+.3f}% over the sweep days, "
          f"{best['total_fills']} fills)")
    interior = 0 < values.index(type(values[0])(best[args.param])) < len(values) - 1
    if not interior:
        print("  WARNING: the best value is at an end of the grid, so the sweep has not\n"
              "  found a turning point -- the number to report is 'at least this', not 'this'.")

    run.finish({"rows": rows, "best": best, "interior": interior})
    if args.write and args.param == "half_spread_bp":
        existing = json.loads(OUT.read_text()) if OUT.exists() else {}
        existing[args.symbol] = float(best["half_spread_bp"])
        OUT.write_text(json.dumps(dict(sorted(existing.items())), indent=2) + "\n")
        print(f"wrote {OUT}: {args.symbol} -> {best['half_spread_bp']:g} bp")
    print(f"run dir: {run.dir}")


if __name__ == "__main__":
    main()
