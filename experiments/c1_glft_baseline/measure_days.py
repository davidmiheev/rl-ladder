"""C1 -- describe every day in the split: volatility, spread, trade activity.

    python experiments/c1_glft_baseline/measure_days.py

Replays each day once with no orders and caches the 100 ms series under
`data/measured/`, then reports the properties that turn out to decide C1's
results.  Held-out days are included: these are descriptions of the data, not
of any strategy, and they are used to *explain* results already computed, never
to choose anything.  Nothing here reads a PnL.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import config  # noqa: E402
import glft  # noqa: E402
from run_calibrate import measure  # noqa: E402

from rl_ladder.run import Run  # noqa: E402


def main() -> None:
    run = Run(phase="c1", name="measure-days", config={"symbol": config.SYMBOL})
    print(f"{'date':>12} {'set':>6} {'sigma':>8} {'spread':>7} {'1-tick':>7} "
          f"{'steps':>9} {'w/trade':>8} {'p50 depth':>10} {'p90 depth':>10} {'open':>9}")
    rows = []
    for date in config.TRAIN_DAYS + config.TEST_DAYS:
        measured = measure(date)
        chg = measured["mid_price_chg"]
        spread = measured["spread"][np.isfinite(measured["spread"])]
        depth = measured["arrival_depth"][np.isfinite(measured["arrival_depth"])]
        mid = measured["mid_price"]
        row = {
            "date": date,
            "set": "test" if date in config.TEST_DAYS else "train",
            "sigma_ticks_per_sqrt_s": float(glft.realised_volatility(chg, glft.INTERVAL_NS)),
            "median_spread_tick": float(np.median(spread)),
            "pct_spread_1_tick": float((spread == 1).mean() * 100),
            "steps": int(len(chg)),
            "steps_with_trade": int(len(depth)),
            "arrival_depth_p50": float(np.percentile(depth, 50)),
            "arrival_depth_p90": float(np.percentile(depth, 90)),
            "open_price": float(np.nanmin(mid) * config.TICK_SIZE),
        }
        rows.append(row)
        run.log(step=len(rows) - 1, **row)
        print(
            f"{row['date']:>12} {row['set']:>6} {row['sigma_ticks_per_sqrt_s']:8.1f} "
            f"{row['median_spread_tick']:7.1f} {row['pct_spread_1_tick']:6.1f}% "
            f"{row['steps']:9d} {row['steps_with_trade']:8d} "
            f"{row['arrival_depth_p50']:10.1f} {row['arrival_depth_p90']:10.1f} "
            f"{row['open_price']:9.1f}",
            flush=True,
        )

    train = [r["sigma_ticks_per_sqrt_s"] for r in rows if r["set"] == "train"]
    test = [r["sigma_ticks_per_sqrt_s"] for r in rows if r["set"] == "test"]
    print(f"\nsigma, ticks per sqrt(second):  train median {np.median(train):.1f}   "
          f"held out median {np.median(test):.1f}")
    run.finish({"rows": rows, "sigma_train_median": float(np.median(train)),
                "sigma_test_median": float(np.median(test))})
    print(f"run dir: {run.dir}")


if __name__ == "__main__":
    main()
