"""C1 items 2 and 3 -- run the calibrated quoters over whole days and score them.

    python experiments/c1_glft_baseline/run_backtest.py --strategy glft-adj --days train
    python experiments/c1_glft_baseline/run_backtest.py --strategy simplified --days test

Strategies:

  touch           no model at all: one lot at the best bid and the best ask.
                  The floor every other row has to clear.
  glft            equations (4.6)/(4.7) with no adjustment at all: one bid, one
                  ask, `adj1 = adj2 = 1`.  The notebook's first run, and the one
                  that loses money.
  glft-adj        the same with `adj2 = 0.05`, the notebook's fix for a skew so
                  strong the quoter refuses to hold inventory.
  glft-grid       `glft-adj` quotes laid out as a 20-deep grid, which is the
                  configuration the published Sharpe/Sortino come from.
  simplified      item 3: percentile-based (non-parametric) half spread and a
                  micro-price fair value, otherwise the same grid.
  simplified-mid  `simplified` with the mid-price instead of the micro-price, so
                  that the two substitutions in item 3 can be told apart.

Each day is replayed independently -- position starts and is marked at zero --
and reported separately as well as pooled, because a single day can dominate any
average of daily Sharpe ratios.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from hftbacktest import Recorder
from hftbacktest.stats import LinearAssetRecord

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import config  # noqa: E402
import glft  # noqa: E402

from rl_ladder.run import Run  # noqa: E402

#: The notebook quotes 1 ETH a side up to 20 ETH of inventory against a 30,000
#: book, at ETH ~ 1330: capital is 1.13x the largest position it can hold.  ETH
#: traded near 4400 over our days, so 100,000 keeps that ratio at 1.14x and
#: Sharpe stays comparable with the published figure.
ORDER_QTY = 1.0
MAX_POSITION = 20.0
BOOK_SIZE = 100_000.0

#: Grid depth, from the notebook.
GRID_NUM = 20

#: Reaction interval in milliseconds.  100 is the notebook's; C1's interval
#: study (`run_interval_study.py`) is what justifies anything else.
INTERVAL_MS = 100.0

#: Equity is recorded every 100 ms of simulated time whatever the reaction
#: interval, so the statistics are computed on the same sampling grid and
#: Sharpe numbers stay comparable across intervals.
RECORD_EVERY_MS = 100.0

#: Default quote depth for the non-parametric variant: the depth that ~30% of
#: trading steps reach.  `run_sweeps.py` chooses it properly, on training days.
PERCENTILE = 70.0
SKEW_FACTOR = 1.0

REPORTED = [
    "SR",
    "Sortino",
    "Return",
    "MaxDrawdown",
    "ReturnOverMDD",
    "ReturnOverTrade",
    "MaxPositionValue",
    "DailyNumberOfTrades",
    "DailyTurnover",
]


#: `adj2` for every GLFT variant except the untouched `glft`.  0.05 is the
#: notebook's own value, chosen there to match `gamma`.
ADJ2 = 0.05

STRATEGIES = ["touch", "glft", "glft-adj", "glft-grid", "simplified", "simplified-mid"]

DAY_SETS = {
    # A single training day, for quick diagnostics only -- never for a claim.
    "one": config.TRAIN_DAYS[:1],
    "train": config.TRAIN_DAYS,
    "test": config.TEST_DAYS,
    "all": config.TRAIN_DAYS + config.TEST_DAYS,
}


def run_day(strategy: str, date: str, knobs: dict[str, float]) -> tuple[dict, np.ndarray, np.ndarray]:
    """Replay one day and return its stats, its record, and the quoter's own log."""
    latency_mul = knobs.get("latency_mul", 1.0)
    config.ensure_latency(date, config.MUL_ENTRY * latency_mul, config.MUL_RESP * latency_mul)
    hbt = config.build_backtest(
        date,
        maker_fee=knobs.get("maker_fee", config.MAKER_FEE),
        taker_fee=knobs.get("taker_fee", config.TAKER_FEE),
        latency_mul=latency_mul,
        queue_model=knobs.get("queue_model", config.DEFAULT_QUEUE_MODEL),
    )
    recorder = Recorder(1, 1_000_000)
    qty, cap, grid = knobs["order_qty"], knobs["max_position"], int(knobs["grid_num"])
    interval_ns = int(knobs.get("interval_ms", INTERVAL_MS) * 1e6)
    record_every = max(1, round(RECORD_EVERY_MS * 1e6 / interval_ns))

    if strategy == "touch":
        out = np.zeros(1, glft.out_dtype)
        out["half_spread_tick"] = 0.5
        glft.touch_quoter(
            hbt, recorder.recorder, qty, cap, interval_ns, record_every,
            np.zeros((1, 3)), np.zeros(1), 0,
        )
    elif strategy == "glft":
        out = glft.glft_market_maker(
            hbt, recorder.recorder, knobs["gamma"], 1.0, 1.0, 1.0, qty, cap, interval_ns, record_every
        )
    elif strategy == "glft-adj":
        out = glft.glft_market_maker(
            hbt, recorder.recorder, knobs["gamma"], 1.0, knobs["adj1"], knobs["adj2"], qty, cap,
            interval_ns, record_every,
        )
    elif strategy == "glft-grid":
        out = glft.gridtrading_glft_mm(
            hbt, recorder.recorder, knobs["gamma"], 1.0, knobs["adj1"], knobs["adj2"], qty, cap, grid,
            interval_ns, record_every,
        )
    elif strategy in ("simplified", "simplified-mid"):
        out = glft.simplified_grid_mm(
            hbt,
            recorder.recorder,
            knobs["percentile"],
            knobs["skew_factor"],
            qty,
            cap,
            grid,
            strategy == "simplified",
            interval_ns,
            record_every,
        )
    else:
        raise SystemExit(f"unknown strategy {strategy!r}")

    hbt.close()
    record = recorder.get(0)
    summary = LinearAssetRecord(record).resample("10s").stats(book_size=BOOK_SIZE).summary()
    row = summary.to_dicts()[0]
    stats = {key: float(row[key]) for key in REPORTED if key in row}
    return stats, record, out


def describe(values: np.ndarray) -> str:
    return f"median {np.median(values):9.3f}   min {values.min():9.3f}   max {values.max():9.3f}"


def evaluate(
    strategy: str,
    days: list[str],
    knobs: dict[str, float],
    tag: str = "",
    keep_records: bool = False,
) -> dict:
    """Run `strategy` over `days`, print and return per-day and pooled stats."""
    name = f"{strategy}-{tag}" if tag else strategy
    run = Run(
        phase="c1",
        name=f"{name}-{'-'.join(sorted({d[:7] for d in days}))}",
        config={"strategy": strategy, "days": days, "symbol": config.SYMBOL,
                "book_size": BOOK_SIZE, "split_date": config.SPLIT_DATE, **knobs},
    )

    per_day: list[dict] = []
    for i, date in enumerate(days):
        stats, record, out = run_day(strategy, date, knobs)
        quoted = out["half_spread_tick"][np.isfinite(out["half_spread_tick"])]
        stats["median_half_spread_tick"] = float(np.median(quoted)) if len(quoted) else float("nan")
        stats["date"] = date
        per_day.append(stats)
        run.log(step=i, **stats)
        if keep_records:
            np.savez_compressed(run.path(f"record_{date}.npz"), record=record)
        print(
            f"{date}  SR {stats['SR']:8.2f}  Sortino {stats['Sortino']:8.2f}  "
            f"Return {stats['Return']:8.4f}  MaxPos {stats['MaxPositionValue']:10.1f}  "
            f"trades/day {stats['DailyNumberOfTrades']:8.0f}  "
            f"half-spread {stats['median_half_spread_tick']:7.1f}t",
            flush=True,
        )
        del record, out

    print(f"\nacross {len(days)} days:")
    summary: dict = {}
    for key in [*REPORTED, "median_half_spread_tick"]:
        values = np.array([day[key] for day in per_day])
        print(f"  {key:23s} {describe(values)}")
        summary[f"{key}_median"] = float(np.median(values))
        summary[f"{key}_min"] = float(values.min())
        summary[f"{key}_max"] = float(values.max())
        summary[f"{key}_per_day"] = [float(v) for v in values]

    profitable = int(sum(day["Return"] > 0 for day in per_day))
    print(f"  {'days with Return > 0':23s} {profitable} / {len(days)}")
    summary["days_profitable"] = profitable
    summary["n_days"] = len(days)
    summary["per_day"] = per_day
    summary["knobs"] = knobs
    run.finish(summary)
    print(f"\nrun dir: {run.dir}")
    summary["run_dir"] = str(run.dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="glft-adj", choices=STRATEGIES)
    parser.add_argument(
        "--symbol",
        default=config.SYMBOL,
        choices=sorted(config.SYMBOLS),
        help="cross-check on another asset; Part C's own asset is the default",
    )
    parser.add_argument("--days", default="train", choices=sorted(DAY_SETS))
    parser.add_argument("--gamma", type=float, default=0.05, help="risk aversion; also xi")
    parser.add_argument("--adj1", type=float, default=1.0, help="multiplier on the half spread")
    parser.add_argument("--adj2", type=float, default=ADJ2, help="multiplier on the skew")
    parser.add_argument("--percentile", type=float, default=PERCENTILE)
    parser.add_argument("--skew-factor", type=float, default=SKEW_FACTOR)
    parser.add_argument("--grid-num", type=int, default=GRID_NUM)
    parser.add_argument(
        "--interval-ms",
        type=float,
        default=INTERVAL_MS,
        help="how often the quoter wakes up, re-prices and re-posts",
    )
    parser.add_argument("--order-qty", type=float, default=ORDER_QTY)
    parser.add_argument("--max-position", type=float, default=MAX_POSITION)
    parser.add_argument(
        "--maker-fee",
        type=float,
        default=config.MAKER_FEE,
        help="maker fee as a fraction of trade value; negative is a rebate",
    )
    parser.add_argument("--taker-fee", type=float, default=config.TAKER_FEE)
    parser.add_argument(
        "--latency-mul",
        type=float,
        default=1.0,
        help="scale the synthetic order latency, for sensitivity checks",
    )
    parser.add_argument(
        "--queue-model",
        default=config.DEFAULT_QUEUE_MODEL,
        choices=sorted(config.QUEUE_MODELS),
        help="how the simulator guesses our queue position; see config.QUEUE_MODELS",
    )
    parser.add_argument("--tag", default="", help="suffix for the run directory name")
    parser.add_argument("--keep-records", action="store_true", help="save each day's equity record")
    args = parser.parse_args()

    config.use_symbol(args.symbol)
    days = DAY_SETS[args.days]
    knobs: dict = {
        "gamma": args.gamma,
        "adj1": args.adj1,
        "adj2": args.adj2,
        "percentile": args.percentile,
        "skew_factor": args.skew_factor,
        "grid_num": args.grid_num,
        "interval_ms": args.interval_ms,
        "order_qty": args.order_qty,
        "max_position": args.max_position,
        "maker_fee": args.maker_fee,
        "taker_fee": args.taker_fee,
        "latency_mul": args.latency_mul,
        "queue_model": args.queue_model,
    }
    summary = evaluate(args.strategy, days, knobs, tag=args.tag, keep_records=args.keep_records)
    print(json.dumps({k: v for k, v in summary.items() if k.endswith("_median")}, indent=2))


if __name__ == "__main__":
    main()
