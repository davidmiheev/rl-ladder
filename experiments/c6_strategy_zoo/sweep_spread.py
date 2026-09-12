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

- **`grid_interval_bp`** -- how far apart?  This turns out to be the knob
  `grid_num` is usually mistaken for.  With the notebooks' default step (the
  half spread itself) the sweep finds 20, 40 and 60 orders give *identical*
  results down to the fill count, because the 21st order sits twenty half
  spreads out and nothing reaches it.  Adding orders only does something if
  they land somewhere that trades, which means shrinking the step.

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

import assets  # noqa: E402
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

#: Fills per day, below which a sweep value is disqualified from being chosen.
#: Without this the sweep walks into the trap C2 found and C3 had to work
#: around: quoting far enough out to take no fills returns exactly zero, which
#: beats every losing width, so the search's optimum is "stop making a
#: market".  The first XRPUSDC sweep, run before this guard existed, duly
#: chose 16 bp on **two fills in four days**.
#:
#: The bar is set below every published configuration's own fill rate: the
#: thinnest of the five on these days is `hfgrid` on ETHUSDC at 119 fills a
#: day, so 50 excludes only what trades less than anything published.  It is
#: a judgement, so the sensitivity is reported rather than buried -- at a
#: 100/day bar the XRPUSDC floor would be 4 bp instead of 8.  Disqualified
#: rows are still printed: the shape of the curve past the turning point is
#: the evidence that the turning point is real.
MIN_FILLS_PER_DAY = 50

#: Grid steps to try, in bp.  0 means "follow the half spread", which is what
#: every grid notebook does and what makes `grid_num` inert past about 20: at
#: a step equal to the half spread, the 21st order sits 20 half spreads from
#: the fair price and nothing ever reaches it.  The smaller values pack the
#: same number of orders into a span that trades.
GRID_INTERVAL_BP = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)

VALUES = {
    "half_spread_bp": HALF_SPREAD_BP,
    "grid_num": GRID_NUM,
    "grid_interval_bp": GRID_INTERVAL_BP,
}

#: Sweep days: the first four training days, which is the least data the
#: search can run on.  Named here rather than inline so the writeup can say
#: exactly which sessions chose the number.
def sweep_days(symbol: str, n: int) -> list[str]:
    assets.ensure_registered(symbol)
    return assets.train_days(symbol)[:n]


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
    parser.add_argument("--recompute", default=None,
                        help="re-score a finished sweep run directory instead of sweeping")
    args = parser.parse_args()

    if args.recompute:
        recompute(Path(args.recompute), args.write)
        return

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

    best, interior = choose(rows, args.param, values, len(days))
    run.finish({"rows": rows, "best": best, "interior": interior})
    if args.write and args.param == "half_spread_bp":
        write_floor(args.symbol, best["half_spread_bp"])
    print(f"run dir: {run.dir}")


def choose(rows: list[dict], param: str, values, n_days: int) -> tuple[dict, bool]:
    """Pick the best swept value among those that actually make a market."""
    floor = MIN_FILLS_PER_DAY * n_days
    trading = [r for r in rows if r["total_fills"] >= floor]
    skipped = [r for r in rows if r["total_fills"] < floor]
    if not trading:
        raise SystemExit(
            f"no swept value takes {MIN_FILLS_PER_DAY} fills a day; nothing here makes a market"
        )
    best = max(trading, key=lambda r: r["cumulative_return"])
    if skipped:
        print(f"\ndisqualified for taking under {MIN_FILLS_PER_DAY} fills a day: "
              + ", ".join(f"{r[param]:g} ({r['total_fills']} fills)" for r in skipped))
    print(f"best {param} = {best[param]:g}  "
          f"({best['cumulative_return'] * 100:+.3f}% over the sweep days, "
          f"{best['total_fills']} fills)")
    index = list(values).index(type(values[0])(best[param]))
    bracketed = index + 1 < len(values) and any(
        type(values[0])(r[param]) == values[index + 1] for r in trading
    )
    interior = index > 0 and bracketed
    if not interior:
        print("  WARNING: the best value is at an edge of the *trading* range, so the sweep\n"
              "  has not bracketed a turning point -- the number to report is 'at least\n"
              "  this', not 'this'.")
    return best, interior


def write_floor(symbol: str, value: float) -> None:
    existing = json.loads(OUT.read_text()) if OUT.exists() else {}
    existing[symbol] = float(value)
    OUT.write_text(json.dumps(dict(sorted(existing.items())), indent=2) + "\n")
    print(f"wrote {OUT}: {symbol} -> {value:g} bp")


def recompute(run_dir: Path, write: bool) -> None:
    """Re-score a finished sweep without re-running it.

    The participation guard was added after the first sweeps had already been
    launched, so their stored rows are re-scored under the current rule rather
    than spending another half hour of replays to reproduce them.
    """
    summary = json.loads((run_dir / "summary.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    best, _ = choose(summary["rows"], config["param"], VALUES[config["param"]],
                     len(config["days"]))
    if write and config["param"] == "half_spread_bp":
        write_floor(config["symbol"], best["half_spread_bp"])


if __name__ == "__main__":
    main()
