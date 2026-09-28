"""Per-step (100 ms) market features, on exactly the strategy's quoting grid.

The fair-price questions in C9 need, for every quoting step, what the book
looked like at that step and where the mid went afterwards. This replays one
session with no orders through the same backtest the strategies run on
(`config.build_backtest`), elapsing the same 100 ms per step and reading the
book after each elapse as `zoo.quote_once` does, so step `t` here is step `t`
of every strategy run on that session: the oracle's future mid and a
predictor's features line up with the quotes without any timestamp matching.

Written to `data/c9/<symbol>_<date>.npz`, one row per step:

    mid2        int32    best bid + best ask, in ticks (-1 when one-sided)
    spread      int16    best ask - best bid, in ticks
    bq1, aq1    float32  quantity at the best bid / best ask
    bb_2, ab_2  float32  cumulative bid / ask quantity within 2.5 bp of the mid
    bb_10, ab_10 float32 the same within 10 bp (OBI's default band)
    ofi         float32  top-of-book order-flow imbalance over the step
                         (Cont, Kukanov and Stoikov), in base units
    buy, sell   float32  aggressive volume by side over the step

    python experiments/c9_fair_price/features.py --symbol ethusdc --days all
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from hftbacktest import BUY_EVENT
from numba import njit

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for sub in ("c1_glft_baseline", "c6_strategy_zoo", "c7_signals", "c8_risk_controls"):
    sys.path.insert(0, str(REPO / "experiments" / sub))

import assets  # noqa: E402
import config  # noqa: E402
import snapshots  # noqa: E402
import strategies  # noqa: E402

OUT = REPO / "data" / "c9"
BANDS_BP = np.array([2.5, 10.0])
LEVELS = np.array([1], np.int64)
INVALID_MIN = -9223372036854775808
INVALID_MAX = 9223372036854775807


@njit
def _replay(hbt, interval_ns, mid2, spread, bq1, aq1, bb, ab, ofi, buy, sell, bands_bp, levels):
    tick = hbt.depth(0).tick_size
    lvl = np.zeros(len(levels))
    band = np.zeros(len(bands_bp))
    prev_bid = INVALID_MIN
    prev_ask = INVALID_MAX
    prev_bq = 0.0
    prev_aq = 0.0
    t = 0
    while t < len(mid2) and hbt.elapse(interval_ns) == 0:
        for trade in hbt.last_trades(0):
            if trade.ev & BUY_EVENT == BUY_EVENT:
                buy[t] += trade.qty
            else:
                sell[t] += trade.qty
        hbt.clear_last_trades(0)
        depth = hbt.depth(0)
        b = depth.best_bid_tick
        a = depth.best_ask_tick
        if b != INVALID_MIN and a != INVALID_MAX and a > b:
            mid2[t] = b + a
            spread[t] = a - b
            qb = depth.bid_qty_at_tick(b)
            qa = depth.ask_qty_at_tick(a)
            bq1[t] = qb
            aq1[t] = qa
            mid = (b + a) / 2.0 * tick
            snapshots._walk(depth, b, -1, mid, tick, levels, bands_bp, lvl, band)
            for k in range(len(bands_bp)):
                bb[t, k] = band[k]
            snapshots._walk(depth, a, 1, mid, tick, levels, bands_bp, lvl, band)
            for k in range(len(bands_bp)):
                ab[t, k] = band[k]
            if prev_bid != INVALID_MIN:
                e = 0.0
                if b >= prev_bid:
                    e += qb
                if b <= prev_bid:
                    e -= prev_bq
                if a <= prev_ask:
                    e -= qa
                if a >= prev_ask:
                    e += prev_aq
                ofi[t] = e
            prev_bid, prev_ask, prev_bq, prev_aq = b, a, qb, qa
        else:
            mid2[t] = -1
        t += 1
    return t


def path_for(symbol: str, date: str) -> Path:
    return OUT / f"{symbol}_{date}.npz"


def extract(symbol: str, date: str) -> Path:
    assets.ensure_registered(symbol)
    config.use_symbol(symbol)
    config.ensure_latency(date, config.MUL_ENTRY, config.MUL_RESP)
    hbt = config.build_backtest(date)
    n = int(86_400e9 / strategies.INTERVAL_NS) + 36_000  # a day plus an hour
    mid2 = np.full(n, -1, np.int64)
    spread = np.zeros(n, np.int64)
    bq1, aq1 = np.zeros(n), np.zeros(n)
    bb, ab = np.zeros((n, len(BANDS_BP))), np.zeros((n, len(BANDS_BP)))
    ofi, buy, sell = np.zeros(n), np.zeros(n), np.zeros(n)
    steps = _replay(hbt, strategies.INTERVAL_NS, mid2, spread, bq1, aq1, bb, ab, ofi, buy, sell,
                    BANDS_BP, LEVELS)
    hbt.close()
    out = path_for(symbol, date)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.stem}.tmp{os.getpid()}.npz")
    f32 = lambda x: x[:steps].astype(np.float32)  # noqa: E731
    np.savez_compressed(
        tmp, mid2=mid2[:steps].astype(np.int32), spread=spread[:steps].astype(np.int16),
        bq1=f32(bq1), aq1=f32(aq1), bb_2=f32(bb[:, 0]), ab_2=f32(ab[:, 0]),
        bb_10=f32(bb[:, 1]), ab_10=f32(ab[:, 1]), ofi=f32(ofi), buy=f32(buy), sell=f32(sell),
        tick=config.TICK_SIZE,
    )
    os.replace(tmp, out)
    return out


def load(symbol: str, date: str) -> dict[str, np.ndarray]:
    path = path_for(symbol, date)
    if not path.exists():
        extract(symbol, date)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def main() -> None:
    import c8

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--days", default="all", help="all, train, test or one date")
    parser.add_argument("--shard", default="0/1")
    args = parser.parse_args()
    days = {"all": c8.DAYS, "train": c8.TRAIN, "test": c8.TEST}.get(args.days, [args.days])
    index, count = (int(x) for x in args.shard.split("/"))
    for date in days[index::count]:
        if not path_for(args.symbol, date).exists():
            print(extract(args.symbol, date), flush=True)


if __name__ == "__main__":
    main()
