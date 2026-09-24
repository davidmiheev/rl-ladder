"""Per-second order-book snapshots, for asking what the book knows.

C7 asks three questions that C6 could not: is there information deeper in the
book than the five ticks C2's features look at, is there information in
BTCUSDT's book that the traded asset's own book does not already carry, and is
there a better volatility estimate than a ten-minute standard deviation. All
three are questions about the *market*, not about a strategy, so they are
answered offline from the data before anything is trained on them.

This script replays one symbol-day with no orders and writes, per day:

    mid100   int32  the mid on the 100 ms grid, in half-ticks (-1 when one-sided)
    ts       int64  local timestamp at the end of each second
    bid_lvl, ask_lvl   float32 (seconds, len(LEVELS))
             resting quantity from the touch out to `LEVELS` ticks, cumulative
    bid_bp, ask_bp     float32 (seconds, len(BANDS_BP))
             the same out to `BANDS_BP` basis points from the mid, cumulative
    ofi      float32  order-flow imbalance (Cont, Kukanov & Stoikov) summed
                      over the second's ten 100 ms steps, in base units
    buy_qty, sell_qty  float32  aggressive volume by side over the second
    n_trades           int32
    max_trade          float32  largest single print over the second

Bands in basis points as well as levels in ticks, because five ticks is 0.11 bp
on ETHUSDC and 8 bp on ENAUSDC: a feature defined in ticks means a different
thing on every asset, and on a fine-tick one "five levels deep" is the touch.

Days are replayed with a constant zero order latency: no order is ever sent, so
the synthesised latency the strategy runs need is irrelevant here, and BTCUSDT
has none built.

BTCUSDT is ~0.8 GB a day and the disk allowance does not hold three weeks of
it, so `--stream` downloads a day, extracts it, and deletes the raw file. Only
the ~10 MB of features is kept.

    python experiments/c7_signals/snapshots.py --symbol ethusdc
    HF_TOKEN=... python experiments/c7_signals/snapshots.py --symbol btcusdt --stream
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from hftbacktest import BUY_EVENT, BacktestAsset, ROIVectorMarketDepthBacktest
from numba import njit

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "c1_glft_baseline"))

import config  # noqa: E402

DATA = REPO / "data" / "npz"
FEATURES = REPO / "data" / "features"
DATASET = "cardinalai/backtest_data"

#: Cumulative depth out to this many ticks. 5 is C2's "deep" feature.
LEVELS = np.array([1, 5, 20, 50, 200], np.int64)
#: Cumulative depth out to this many basis points from the mid.
BANDS_BP = np.array([1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0])

STEP_NS = 100_000_000
STEPS_PER_SECOND = 10
SECONDS = 86_400 + 60

#: Tick, lot and the ROI ceiling for each symbol C7 reads. The two traded
#: assets are C6's; BTCUSDT is Binance's published perp contract spec, checked
#: against the data by `_check_tick` on every day it reads.
SPECS = {
    "ethusdc": (0.01, 0.001, 15_000.0),
    "xrpusdc": (0.0001, 0.1, 30.0),
    "btcusdt": (0.1, 0.001, 250_000.0),
}

INVALID_MIN = -9223372036854775808
INVALID_MAX = 9223372036854775807


@njit
def _walk(depth, touch, direction, mid, tick, levels, bands_bp, out_lvl, out_bp):
    """Cumulative resting quantity from the touch outward, one side.

    `direction` is -1 for bids (walking down) and +1 for asks. A tick is
    inside band `b` when it is no further than `b` bp from the mid; the sum is
    recorded for a band just before the first tick outside it is added, and
    for a level just after its last tick is.
    """
    lo = depth.roi_lb_tick
    hi = depth.roi_ub_tick
    grid = depth.bid_depth if direction < 0 else depth.ask_depth
    n_lvl = len(levels)
    n_bp = len(bands_bp)
    edges = np.empty(n_bp, np.int64)
    for i in range(n_bp):
        if direction < 0:
            edges[i] = int(np.ceil(mid * (1.0 - bands_bp[i] * 1e-4) / tick))
        else:
            edges[i] = int(np.floor(mid * (1.0 + bands_bp[i] * 1e-4) / tick))
    reach = max(levels[-1] - 1, direction * (edges[-1] - touch))
    total = 0.0
    li = 0
    bi = 0
    for j in range(reach + 1):
        t = touch + direction * j
        while bi < n_bp and direction * (t - edges[bi]) > 0:
            out_bp[bi] = total
            bi += 1
        if lo <= t <= hi:
            total += grid[t - lo]
        while li < n_lvl and j + 1 == levels[li]:
            out_lvl[li] = total
            li += 1
    for i in range(li, n_lvl):
        out_lvl[i] = total
    for i in range(bi, n_bp):
        out_bp[i] = total


@njit
def _replay(hbt, levels, bands_bp, mid100, ts, bid_lvl, ask_lvl, bid_bp, ask_bp, ofi,
            buy_qty, sell_qty, n_trades, max_trade):
    tick = hbt.depth(0).tick_size
    step = 0
    sec = 0
    prev_bid = INVALID_MIN
    prev_ask = INVALID_MAX
    prev_bq = 0.0
    prev_aq = 0.0
    while hbt.elapse(100_000_000) == 0:
        for trade in hbt.last_trades(0):
            if trade.ev & BUY_EVENT == BUY_EVENT:
                buy_qty[sec] += trade.qty
            else:
                sell_qty[sec] += trade.qty
            n_trades[sec] += 1
            if trade.qty > max_trade[sec]:
                max_trade[sec] = trade.qty
        hbt.clear_last_trades(0)

        depth = hbt.depth(0)
        bb = depth.best_bid_tick
        ba = depth.best_ask_tick
        two_sided = bb != INVALID_MIN and ba != INVALID_MAX and ba > bb
        if two_sided:
            mid100[step] = bb + ba
            bq = depth.bid_qty_at_tick(bb)
            aq = depth.ask_qty_at_tick(ba)
            if prev_bid != INVALID_MIN and prev_ask != INVALID_MAX:
                e = 0.0
                if bb >= prev_bid:
                    e += bq
                if bb <= prev_bid:
                    e -= prev_bq
                if ba <= prev_ask:
                    e -= aq
                if ba >= prev_ask:
                    e += prev_aq
                ofi[sec] += e
            prev_bid, prev_ask, prev_bq, prev_aq = bb, ba, bq, aq
        else:
            mid100[step] = -1

        step += 1
        if step % 10 == 0:
            ts[sec] = hbt.current_timestamp
            if two_sided:
                mid = (bb + ba) / 2.0 * tick
                _walk(depth, bb, -1, mid, tick, levels, bands_bp, bid_lvl[sec], bid_bp[sec])
                _walk(depth, ba, 1, mid, tick, levels, bands_bp, ask_lvl[sec], ask_bp[sec])
            else:
                for k in range(len(levels)):
                    bid_lvl[sec, k] = np.nan
                    ask_lvl[sec, k] = np.nan
                for k in range(len(bands_bp)):
                    bid_bp[sec, k] = np.nan
                    ask_bp[sec, k] = np.nan
            sec += 1
            if sec >= len(ts):
                break
        if step >= len(mid100):
            break
    return step, sec


def feature_path(symbol: str, date: str) -> Path:
    return FEATURES / f"{symbol}_{date}.npz"


def _check_tick(path: Path, tick: float) -> None:
    """Every price in the file sits on the tick grid, or the spec is wrong."""
    px = np.load(path, mmap_mode="r")["data"]["px"][:2_000_000]
    px = px[np.isfinite(px) & (px > 0)]
    off = np.abs(px / tick - np.round(px / tick))
    if off.max() > 1e-6:
        raise SystemExit(f"{path.name}: prices off the {tick} grid by {off.max():.3g} ticks")


def extract(symbol: str, date: str, raw: Path) -> Path:
    tick, lot, roi_ub = SPECS[symbol]
    _check_tick(raw, tick)
    asset = (
        BacktestAsset()
        .data([str(raw)])
        .linear_asset(1.0)
        .constant_order_latency(0, 0)
        .risk_adverse_queue_model()
        .no_partial_fill_exchange()
        .trading_value_fee_model(0.0, 0.0)
        .tick_size(tick)
        .lot_size(lot)
        .roi_lb(0.0)
        .roi_ub(roi_ub)
        .last_trades_capacity(100_000)
    )
    hbt = ROIVectorMarketDepthBacktest([asset])
    n_steps = SECONDS * STEPS_PER_SECOND
    mid100 = np.full(n_steps, -1, np.int64)
    ts = np.zeros(SECONDS, np.int64)
    bid_lvl = np.zeros((SECONDS, len(LEVELS)))
    ask_lvl = np.zeros((SECONDS, len(LEVELS)))
    bid_bp = np.zeros((SECONDS, len(BANDS_BP)))
    ask_bp = np.zeros((SECONDS, len(BANDS_BP)))
    ofi = np.zeros(SECONDS)
    buy_qty = np.zeros(SECONDS)
    sell_qty = np.zeros(SECONDS)
    n_trades = np.zeros(SECONDS, np.int64)
    max_trade = np.zeros(SECONDS)
    steps, secs = _replay(hbt, LEVELS, BANDS_BP, mid100, ts, bid_lvl, ask_lvl, bid_bp, ask_bp,
                          ofi, buy_qty, sell_qty, n_trades, max_trade)
    hbt.close()

    out = feature_path(symbol, date)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp, mid100=mid100[:steps].astype(np.int32), ts=ts[:secs], tick=tick,
        levels=LEVELS, bands_bp=BANDS_BP,
        bid_lvl=bid_lvl[:secs].astype(np.float32), ask_lvl=ask_lvl[:secs].astype(np.float32),
        bid_bp=bid_bp[:secs].astype(np.float32), ask_bp=ask_bp[:secs].astype(np.float32),
        ofi=ofi[:secs].astype(np.float32), buy_qty=buy_qty[:secs].astype(np.float32),
        sell_qty=sell_qty[:secs].astype(np.float32), n_trades=n_trades[:secs].astype(np.int32),
        max_trade=max_trade[:secs].astype(np.float32),
    )
    os.replace(tmp, out)
    return out


def download(symbol: str, date: str) -> Path:
    from huggingface_hub import hf_hub_download

    name = f"{symbol}_{date}.npz"
    return Path(hf_hub_download(DATASET, name, repo_type="dataset", local_dir=DATA))


def days_for(symbol: str) -> list[str]:
    if symbol == "xrpusdc":
        return config.PAIR_TRAIN_DAYS + config.PAIR_TEST_DAYS_EXT
    # ETHUSDC's days are a superset of XRPUSDC's, so BTCUSDT uses them too.
    return config.TRAIN_DAYS_EXT + config.TEST_DAYS_EXT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", required=True, choices=sorted(SPECS))
    parser.add_argument("--days", nargs="*", default=None)
    parser.add_argument("--stream", action="store_true",
                        help="download each missing raw day, extract it, then delete it")
    args = parser.parse_args()

    for date in args.days or days_for(args.symbol):
        if feature_path(args.symbol, date).exists():
            print(f"{args.symbol} {date}: done")
            continue
        raw = DATA / f"{args.symbol}_{date}.npz"
        fetched = False
        if not raw.exists():
            if not args.stream:
                raise SystemExit(f"{raw} missing; pass --stream to download it")
            raw = download(args.symbol, date)
            fetched = True
        out = extract(args.symbol, date, raw)
        size = out.stat().st_size / 1e6
        if fetched:
            raw.unlink()
        print(f"{args.symbol} {date}: {size:.1f} MB{' (raw deleted)' if fetched else ''}",
              flush=True)


if __name__ == "__main__":
    main()
