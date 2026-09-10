"""C1 -- collect the finished runs into one table, and decompose the loss.

    python experiments/c1_glft_baseline/summarise.py --tag final

Reads the `summary.json` of every run in `runs/c1/` matching a tag and prints
per-day and pooled statistics, plus the only accounting that makes the phase's
five strategies comparable to each other: what one fill was worth.

A passive fill earns at most half the touch spread plus the maker rebate, and
loses whatever the price does against the new position afterwards.  In ticks,

    net per fill  =  Return x book_size / fills / tick_size
    rebate per fill  =  -maker_fee x price / tick_size
    gross per fill  =  net per fill - rebate per fill

Strategies that quote in completely different places turn out to have nearly
the same gross figure, which is the phase's main finding: they differ in how
many fills they take, not in what a fill is worth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import glft  # noqa: E402

ORDER = ["touch", "glft", "glft-adj", "glft-grid", "simplified", "simplified-mid"]


def load(tag: str, days: str) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    root = Path(__file__).resolve().parents[2] / "runs" / "c1"
    for path in sorted(root.glob(f"*-{tag}-*/summary.json")):
        blob = json.loads(path.read_text())
        if "per_day" not in blob:
            continue
        dates = [row["date"] for row in blob["per_day"]]
        which = "test" if dates[0] in config.TEST_DAYS else "train"
        if which != days:
            continue
        strategy = blob["knobs"].get("strategy") or json.loads(
            (path.parent / "config.json").read_text()
        )["strategy"]
        runs[strategy] = blob
    return runs


def per_fill_ticks(blob: dict, price: float) -> tuple[float, float, float]:
    """Net, rebate and gross value of one fill, in ticks, medianed over days."""
    nets, grosses = [], []
    rebate = -config.MAKER_FEE * price / config.TICK_SIZE
    for row in blob["per_day"]:
        if row["DailyNumberOfTrades"] <= 0:
            continue
        net = row["Return"] * blob["knobs"].get("book_size", 100_000.0)
        net = net / row["DailyNumberOfTrades"] / config.TICK_SIZE
        nets.append(net)
        grosses.append(net - rebate)
    return float(np.median(nets)), rebate, float(np.median(grosses))


def day_sigma(date: str) -> float:
    """That day's mid-price volatility in ticks per sqrt(second).

    Read from the cache `measure_days.py` writes.  This is a property of the
    market, measured with no orders and no PnL in sight, so reading it for a
    held-out day describes the data rather than selecting on it.
    """
    path = Path(__file__).resolve().parents[2] / "data" / "measured" / f"{config.SYMBOL}_{date}.npz"
    if not path.exists():
        return float("nan")
    return float(glft.realised_volatility(np.load(path)["mid_price_chg"], glft.INTERVAL_NS))


def volatility_regression(runs: dict[str, dict], price: float) -> list[tuple[float, float]]:
    """One (day volatility, gross value of a fill) point per strategy-day."""
    points = []
    rebate = -config.MAKER_FEE * price / config.TICK_SIZE
    for blob in runs.values():
        book = blob["knobs"].get("book_size", 100_000.0)
        for row in blob["per_day"]:
            sigma = day_sigma(row["date"])
            if row["DailyNumberOfTrades"] <= 0 or not np.isfinite(sigma):
                continue
            net = row["Return"] * book / row["DailyNumberOfTrades"] / config.TICK_SIZE
            points.append((sigma, net - rebate))
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="final")
    parser.add_argument("--price", type=float, default=4400.0, help="reference price for the rebate")
    args = parser.parse_args()

    pooled: list[tuple[float, float]] = []
    for days in ("train", "test"):
        runs = load(args.tag, days)
        if not runs:
            continue
        print(f"\n=== {days} days ({len(next(iter(runs.values()))['per_day'])}) ===")
        print(
            f"{'strategy':>15} {'SR med':>9} {'Sortino':>9} {'Ret med':>9} {'days +':>7} "
            f"{'fills/day':>10} {'half-spr':>9} {'MaxPos':>10} {'net/fill':>9} {'gross/fill':>11}"
        )
        for strategy in ORDER:
            blob = runs.get(strategy)
            if blob is None:
                continue
            net, _, gross = per_fill_ticks(blob, args.price)
            print(
                f"{strategy:>15} {blob['SR_median']:9.2f} {blob['Sortino_median']:9.2f} "
                f"{blob['Return_median']:9.4f} {blob['days_profitable']:4d}/{blob['n_days']:<2d} "
                f"{blob['DailyNumberOfTrades_median']:10.0f} "
                f"{blob['median_half_spread_tick_median']:9.1f} "
                f"{blob['MaxPositionValue_median']:10.0f} {net:9.1f} {gross:11.1f}"
            )
        rebate = -config.MAKER_FEE * args.price / config.TICK_SIZE
        print(f"  (rebate is +{rebate:.1f} ticks a fill at a price of {args.price:g})")

        pooled += volatility_regression(runs, args.price)

        print("\n  per-day Sharpe:")
        for strategy in ORDER:
            blob = runs.get(strategy)
            if blob is None:
                continue
            values = ", ".join(f"{v:.1f}" for v in blob["SR_per_day"])
            print(f"    {strategy:>15}  {values}")

    report_regression(pooled)


def report_regression(pooled: list[tuple[float, float]]) -> None:
    x = np.array([p[0] for p in pooled])
    y = np.array([p[1] for p in pooled])
    slope, intercept = np.polyfit(x, y, 1)
    corr = float(np.corrcoef(x, y)[0, 1])
    print("\n=== what a fill is worth, against how volatile the day was ===")
    print("(sigma in ticks per sqrt(second), measured with no orders placed;")
    print(" run measure_days.py first or these rows are skipped)")
    print(f"  gross ticks per fill = {slope:.2f} x sigma {intercept:+.1f}")
    print(f"  r = {corr:.2f} over n = {len(x)} strategy-days")
    rebate = -config.MAKER_FEE * 4400 / config.TICK_SIZE
    print(f"  break-even needs gross > -{rebate:.0f} ticks (the rebate), i.e. sigma below "
          f"{(-rebate - intercept) / slope:.1f} ticks/sqrt(s)")
    print("  quietest day in the split: 12.8.  hftbacktest's published ETHUSDT day: 10.7.")


if __name__ == "__main__":
    main()
