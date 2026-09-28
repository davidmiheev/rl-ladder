"""C6 baselines -- the five strategies at their published parameters.

    python experiments/c6_strategy_zoo/run_baseline.py --strategy glft --days train
    python experiments/c6_strategy_zoo/run_baseline.py --strategy obi-wide --days test
    python experiments/c6_strategy_zoo/run_baseline.py --strategy apt --symbol xrpusdc --days pair-test

Every run writes its per-day statistics, its equity record and its inventory
path under `runs/`, which is what `master_table.py` and `plot_zoo.py` read.
Days are replayed independently and reported both per day and pooled as one
track record, because C3 learned the hard way that the median of per-session
Sharpes and the Sharpe of the pooled period rank strategies almost inversely
here: a strategy with four flat days and one catastrophe has an excellent
median session.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
from hftbacktest import Recorder
from hftbacktest.stats import LinearAssetRecord

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "c1_glft_baseline"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import assets  # noqa: E402
import config  # noqa: E402
import precompute  # noqa: E402
import strategies  # noqa: E402
import zoo  # noqa: E402

from rl_ladder.run import Run  # noqa: E402

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

RUNS = Path(__file__).resolve().parents[2] / "runs"

#: An empty table for the three strategies that do not read a reference leg.
#: Numba needs the argument to have a fixed type, so it is the same shape as a
#: real one rather than `None`.
NO_PRECOMPUTE = np.zeros((1, 3), np.float64)


def day_sets(symbol: str) -> dict[str, list[str]]:
    assets.ensure_registered(symbol)
    sets = assets.day_sets(symbol)
    # C1's original five-day lists, so a C6 run can be put beside a C1 one.
    sets |= {"train-c1": config.TRAIN_DAYS, "test-c1": config.TEST_DAYS}
    # Retained spellings from before `assets.py` existed.
    sets |= {"pair-train": sets["train"], "pair-test": sets["test"],
             "pair-all": sets["all"], "pair-one": sets["one"]}
    return sets


def table_for(strategy: str, symbol: str, date: str) -> np.ndarray:
    base = strategy.split("-")[0]
    if base not in strategies.PAIR_STRATEGIES:
        return NO_PRECOMPUTE
    table = precompute.load(base, symbol, date)
    config.use_symbol(symbol)  # `precompute` repoints the symbol while building
    return np.ascontiguousarray(table, dtype=np.float64)


def run_day(
    strategy: str,
    symbol: str,
    date: str,
    overrides: dict[str, float],
    multipliers: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    fair: np.ndarray | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Replay one session; return its statistics, its record and the quoter's log.

    `fair` is C9's external series, one row per quoting step: a 1-D fair-price
    shift in bp of the mid (`fair_mode`, `fair_coef` say how it is used), or
    the full `zoo.FAIR_COLS` columns (the DAgger student's market scores too).
    """
    assets.ensure_registered(symbol)
    config.use_symbol(symbol)
    kwargs = {"beta": overrides["beta"]} if strategy.split("-")[0] == "apt" else {}
    params = strategies.build(strategy, symbol, **kwargs)
    for name, value in overrides.items():
        if name in zoo.PARAM_NAMES:
            params[zoo.PARAM_NAMES.index(name)] = value

    table = table_for(strategy, symbol, date)
    # C8's stress tests: order latency as a multiple of the calibrated
    # model, and the maker fee (negative is the rebate the default assumes).
    latency_mul = float(overrides.get("latency_mul", 1.0))
    config.ensure_latency(date, config.MUL_ENTRY * latency_mul, config.MUL_RESP * latency_mul)
    hbt = config.build_backtest(
        date, queue_model=overrides.get("queue_model", config.DEFAULT_QUEUE_MODEL),
        maker_fee=float(overrides.get("maker_fee", config.MAKER_FEE)), latency_mul=latency_mul,
    )
    recorder = Recorder(1, 1_000_000)
    interval_ns = int(overrides.get("interval_ns", strategies.INTERVAL_NS))
    record_every = max(1, round(strategies.RECORD_EVERY_NS / interval_ns))

    if fair is None:
        fair = zoo.NO_FAIR
    else:
        fair = np.asarray(fair, dtype=np.float64)
        if fair.ndim == 1:  # a fair-price shift alone: column 0
            fair = np.column_stack([fair, np.full((len(fair), zoo.FAIR_COLS - 1), np.nan)])
        fair = np.ascontiguousarray(fair)
    out = zoo.run_day(
        hbt, recorder.recorder, params, table, fair, interval_ns, record_every, *multipliers
    )
    hbt.close()

    record = recorder.get(0)
    summary = LinearAssetRecord(record).resample("10s").stats(book_size=strategies.BOOK_SIZE).summary()
    row = summary.to_dicts()[0]
    stats = {key: float(row[key]) for key in REPORTED if key in row}
    return stats, record, out


def quoter_stats(out: np.ndarray, interval_ns: int, max_position_dollar: float) -> dict[str, float]:
    """What the quoter itself did, as opposed to what it earned.

    `A` and `k` are the order-arrival intensity `lambda(d) = A exp(-k d)` that
    C1 calibrates; they are measured identically for all five strategies and
    depend only on the market, so they belong in the master table as the
    *market's* trade intensity, against which each strategy's realised fill
    rate can be read.  `1/k` is the depth, in *half*-tick buckets (the notebook's
    unit; `master_table.py` converts), at which arrivals fall by a
    factor of e -- C1's proof that no risk aversion lets GLFT quote inside it.
    """
    finite = np.isfinite(out["A"]) & (out["k"] > 0)
    k = out["k"][finite]
    return {
        "median_half_spread_bp": float(np.nanmedian(out["half_spread_bp"])),
        "median_skew_bp": float(np.nanmedian(out["skew_bp"])),
        "median_abs_signal_bp": float(np.nanmedian(np.abs(out["signal_bp"]))),
        "median_volatility_tick": float(np.nanmedian(out["volatility"])),
        "median_A": float(np.median(out["A"][finite])) if finite.any() else float("nan"),
        "median_k": float(np.median(k)) if finite.any() else float("nan"),
        "spread_floor_tick": float(np.median(1.0 / k)) if finite.any() else float("nan"),
        "median_abs_position": float(np.nanmedian(np.abs(out["position"]))),
        "max_abs_position": float(np.nanmax(np.abs(out["position"]))),
        # Normalised by the notional cap, which is the only inventory number
        # comparable across assets -- and the one the environment reports, so
        # a baseline row and a learned row mean the same thing.
        "median_abs_norm_pos": float(
            np.nanmedian(np.abs(out["position"] * out["mid"])) / max_position_dollar
        ),
        "max_abs_norm_pos": float(
            np.nanmax(np.abs(out["position"] * out["mid"])) / max_position_dollar
        ),
        # C8: share of the session a circuit breaker or loss limit held
        # quotes back (either side).
        "paused_frac": float(np.mean(out["paused"] > 0)),
        "steps": int(len(out)),
        "hours": float(len(out) * interval_ns / 3.6e12),
    }


def evaluate(
    strategy: str,
    symbol: str,
    days: list[str],
    overrides: dict[str, float],
    tag: str | None = None,
    keep_records: bool = True,
    phase: str = "c6",
) -> dict:
    run = Run(phase, tag or f"{strategy}-{symbol}", config={
        "strategy": strategy, "symbol": symbol, "days": days, **overrides,
    })
    interval_ns = int(overrides.get("interval_ns", strategies.INTERVAL_NS))
    per_day = []
    for i, date in enumerate(days):
        stats, record, out = run_day(strategy, symbol, date, overrides)
        stats.update(quoter_stats(out, interval_ns, strategies.MAX_POSITION_DOLLAR))
        stats["fills_per_hour"] = stats["DailyNumberOfTrades"] / max(stats["hours"], 1e-9)
        stats["date"] = date
        per_day.append(stats)
        run.log(step=i, **stats)
        if keep_records:
            np.savez_compressed(
                run.dir / f"record_{date}.npz",
                record=record,
                position=out["position"][:: max(1, len(out) // 20_000)],
                mid=out["mid"][:: max(1, len(out) // 20_000)],
                half_spread_bp=out["half_spread_bp"][:: max(1, len(out) // 20_000)],
            )
        # A session's record is ~55 MB and its quoter log ~50 MB, and a run
        # over twenty-one of them is a single process.  Without this the
        # process grows until the memory cap kills it -- which it did, on day
        # twenty of twenty-one, losing the whole run.
        del record, out
        gc.collect()
        print(
            f"{date}  SR {stats['SR']:8.2f}  Return {stats['Return']:9.4f}  "
            f"MaxDD {stats['MaxDrawdown']:8.4f}  trades {stats['DailyNumberOfTrades']:8.0f}  "
            f"half-spread {stats['median_half_spread_bp']:7.2f} bp  "
            f"|q| {stats['median_abs_position']:8.2f}",
            flush=True,
        )

    summary: dict = {"strategy": strategy, "symbol": symbol, "n_days": len(days)}
    for key in per_day[0]:
        if key == "date":
            continue
        values = np.array([d[key] for d in per_day], float)
        summary[f"{key}_median"] = float(np.median(values))
        summary[f"{key}_per_day"] = [float(v) for v in values]
    summary["per_day"] = per_day
    kwargs = {"beta": overrides["beta"]} if strategy.split("-")[0] == "apt" else {}
    summary["params"] = strategies.describe(strategies.build(strategy, symbol, **kwargs))
    run.finish(summary)
    summary["run_dir"] = str(run.dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--strategy", required=True, choices=sorted(strategies.NAMES))
    parser.add_argument("--symbol", default="ethusdc")
    parser.add_argument("--days", default="train")
    parser.add_argument("--dates", nargs="*", default=None,
                        help="explicit session dates; overrides --days")
    parser.add_argument("--grid-num", type=float, default=None)
    parser.add_argument("--half-spread-bp", type=float, default=None)
    parser.add_argument("--interval-ms", type=float, default=None)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--queue-model", default=config.DEFAULT_QUEUE_MODEL)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--no-records", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="override any quoter parameter by name, e.g. vol_scale=1.5")
    parser.add_argument("--phase", default="c6",
                        help="runs/<phase>/ to write under; C7's variants use c7")
    args = parser.parse_args()

    sets = day_sets(args.symbol)
    if args.days not in sets:
        raise SystemExit(f"unknown day set {args.days!r}; known: {sorted(sets)}")
    days = args.dates or sets[args.days]

    overrides: dict[str, float] = {"beta": args.beta, "queue_model": args.queue_model}
    if args.grid_num is not None:
        overrides["grid_num"] = args.grid_num
    if args.half_spread_bp is not None:
        overrides["half_spread_bp"] = args.half_spread_bp
    if args.interval_ms is not None:
        overrides["interval_ns"] = args.interval_ms * 1e6
    for item in args.set:
        name, _, value = item.partition("=")
        if name not in zoo.PARAM_NAMES:
            raise SystemExit(f"unknown parameter {name!r}")
        overrides[name] = float(value)

    summary = evaluate(
        args.strategy, args.symbol, days, overrides,
        tag=args.tag, keep_records=not args.no_records, phase=args.phase,
    )
    print(json.dumps({k: v for k, v in summary.items() if k.endswith("_median")}, indent=2))


if __name__ == "__main__":
    main()
