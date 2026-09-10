"""C1 -- pick the baseline's free parameters, on training days only.

The faithful reproduction of the notebook loses money on this data (see the
phase README).  Two of its numbers are not calibrated from anything -- `gamma`
is "arbitrarily chosen" in the paper and `adj1`/`adj2` are fudge factors the
notebook introduces by hand -- so leaving them at the published values is a
choice, not a measurement.  This sweeps them, and the non-parametric variant's
quote percentile, on the five training days.  Held-out days are never touched
here; `run_backtest.py --days test` reports the winner once.

    python experiments/c1_glft_baseline/run_sweeps.py --sweep adj1
    python experiments/c1_glft_baseline/run_sweeps.py --sweep percentile

Selection metric is the *median* daily Sharpe across training days, not the
mean: five days is few enough that one blow-up decides a mean, and Part A's
recurring lesson was that the metric which cannot be dominated by one sample is
the one worth selecting on.  Every day's value is written out regardless.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import config  # noqa: E402
import run_backtest as rb  # noqa: E402

SWEEPS: dict[str, tuple[str, str, list[float]]] = {
    # name: (strategy, knob, values)
    "gamma": ("glft-grid", "gamma", [0.005, 0.02, 0.05, 0.2, 1.0]),
    "adj1": ("glft-grid", "adj1", [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]),
    "adj2": ("glft-grid", "adj2", [0.01, 0.05, 0.2, 1.0]),
    "percentile": ("simplified", "percentile", [40.0, 55.0, 65.0, 75.0, 85.0]),
    "skew_factor": ("simplified", "skew_factor", [0.25, 0.5, 1.0, 2.0]),
    "grid_num": ("simplified", "grid_num", [1, 5, 20]),
    "interval_ms": ("simplified", "interval_ms", [100.0, 25.0, 10.0, 2.0]),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True, choices=sorted(SWEEPS))
    parser.add_argument("--strategy", help="override the strategy the sweep runs on")
    parser.add_argument(
        "--fix",
        action="append",
        default=[],
        metavar="KNOB=VALUE",
        help="hold another knob at a value, e.g. --fix adj1=0.05",
    )
    args = parser.parse_args()

    strategy, knob, values = SWEEPS[args.sweep]
    if args.strategy:
        strategy = args.strategy

    base = {
        "gamma": 0.05,
        "adj1": 1.0,
        "adj2": rb.ADJ2,
        "percentile": rb.PERCENTILE,
        "skew_factor": rb.SKEW_FACTOR,
        "grid_num": rb.GRID_NUM,
        "interval_ms": rb.INTERVAL_MS,
        "order_qty": rb.ORDER_QTY,
        "max_position": rb.MAX_POSITION,
    }
    for item in args.fix:
        name, _, value = item.partition("=")
        if name not in base:
            raise SystemExit(f"unknown knob {name!r}")
        base[name] = float(value)

    rows = []
    for value in values:
        knobs = dict(base, **{knob: value})
        print(f"\n===== {strategy}  {knob}={value}  (train days only)")
        summary = rb.evaluate(
            strategy, config.TRAIN_DAYS, knobs, tag=f"{knob}{value}", keep_records=False
        )
        rows.append(
            {
                knob: value,
                "SR_median": summary["SR_median"],
                "SR_per_day": summary["SR_per_day"],
                "Return_median": summary["Return_median"],
                "days_profitable": summary["days_profitable"],
                "half_spread": summary["median_half_spread_tick_median"],
                "trades_per_day": summary["DailyNumberOfTrades_median"],
                "max_position_value": summary["MaxPositionValue_median"],
                "run_dir": summary["run_dir"],
            }
        )

    print(f"\n===== {strategy}: {knob} sweep over {len(config.TRAIN_DAYS)} training days")
    print(
        f"{knob:>12} {'SR median':>11} {'Return med':>11} {'days +':>7} "
        f"{'half-spr':>9} {'trades/d':>9} {'maxpos':>10}"
    )
    for row in rows:
        print(
            f"{row[knob]:12g} {row['SR_median']:11.2f} {row['Return_median']:11.4f} "
            f"{row['days_profitable']:5d}/{len(config.TRAIN_DAYS)} {row['half_spread']:9.1f} "
            f"{row['trades_per_day']:9.0f} {row['max_position_value']:10.0f}"
        )
    best = max(rows, key=lambda r: r["SR_median"])
    print(f"\nbest on training days: {knob}={best[knob]:g}  (median SR {best['SR_median']:.2f})")
    print("per-day SR at that setting: " + ", ".join(f"{v:.1f}" for v in best["SR_per_day"]))

    out = Path(rows[0]["run_dir"]).parent / f"sweep-{strategy}-{knob}.json"
    out.write_text(json.dumps(rows, indent=2, default=str) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
