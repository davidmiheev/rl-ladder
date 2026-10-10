"""Data, day split and backtest configuration for Part C.

Everything in Part C -- C1's baseline and C3-C5's learned policies -- reads its
days from here, so that the train/held-out split is decided once and cannot be
quietly re-picked after seeing a result.
"""

import os
from pathlib import Path

import numpy as np
from hftbacktest import (
    EXCH_EVENT, LOCAL_EVENT, BacktestAsset, HashMapMarketDepthBacktest, ROIVectorMarketDepthBacktest,
)
from numba import njit

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "npz"
LATENCY = REPO / "data" / "latency"

#: Binance USDⓈ-M perpetuals from the `cardinalai/backtest_data` Hugging Face
#: dataset, with the venue's tick and lot sizes.  ETHUSDC is Part C's asset: it
#: is the closest liquid analogue of the ETHUSDT day hftbacktest's published
#: notebook uses.  XRPUSDT is here only as a cross-check, because the
#: "Simplified from GLFT" notebook reports its results on altcoins and a
#: negative result on one asset should not be read as a result about a method.
#:
#: Note that ETHUSDC's tick is tiny in relative terms (0.01 on a ~4400 price,
#: 0.023 bp) and yet its touch spread is one tick 98% of the time -- measured
#: in C1, not assumed.  That makes it a *large*-tick asset in the sense that
#: matters: there is no room inside the spread, so GLFT's continuous quote-depth
#: derivation has nothing to resolve.
SYMBOLS = {
    "ethusdc": {"tick_size": 0.01, "lot_size": 0.001, "roi_ub": 15_000.0},
    "xrpusdt": {"tick_size": 0.0001, "lot_size": 0.1, "roi_ub": 30.0},
    "xrpusdc": {"tick_size": 0.0001, "lot_size": 0.1, "roi_ub": 30.0},
}

#: Reference leg for the two pair strategies in C6.  The published notebooks
#: use *spot* against the perpetual; this dataset has no spot feed, so the
#: USDT-quoted perpetual stands in as the reference for the USDC-quoted one.
#: The pair is cointegrated for a different reason -- the stablecoin basis and
#: a venue lead-lag, not cash-and-carry -- which changes what the signal means
#: but not the shape of the strategy.  See docs/writeups/c6-strategy-zoo.md.
REFERENCE_LEG = {"xrpusdc": "xrpusdt"}

SYMBOL = "ethusdc"
TICK_SIZE = SYMBOLS[SYMBOL]["tick_size"]
LOT_SIZE = SYMBOLS[SYMBOL]["lot_size"]


def register_symbol(symbol: str, tick_size: float, lot_size: float, roi_ub: float) -> None:
    """Add a symbol at run time, for the cross-asset study.

    Part C's own asset is declared above; the study measures tick and lot sizes
    from the data instead of trusting a table, and registers what it finds.
    """
    SYMBOLS[symbol] = {"tick_size": tick_size, "lot_size": lot_size, "roi_ub": roi_ub}


def use_symbol(symbol: str) -> None:
    """Point every path and every backtest at a different symbol.

    Part C runs on ETHUSDC; this exists so a cross-check on another asset does
    not need a parallel copy of the scripts.
    """
    global SYMBOL, TICK_SIZE, LOT_SIZE
    if symbol not in SYMBOLS:
        raise SystemExit(f"unknown symbol {symbol!r}; known: {sorted(SYMBOLS)}")
    SYMBOL = symbol
    TICK_SIZE = SYMBOLS[symbol]["tick_size"]
    LOT_SIZE = SYMBOLS[symbol]["lot_size"]

#: Binance's highest USDⓈ-M market-maker rebate (-0.005%) and the standard
#: taker fee (0.07%), as used by the reference notebooks.  Sign convention:
#: negative is a rebate paid to us.
MAKER_FEE = -0.00005
TAKER_FEE = 0.0007

#: THE SPLIT.  Chronological, fixed in C1, reused unchanged by C3, C4 and C5.
#: The dataset covers 2025-08-26 to 2025-09-25; everything up to and including
#: 2025-09-14 may be looked at, everything from 2025-09-15 may not be, except
#: to report a final number.
SPLIT_DATE = "2025-09-15"

TRAIN_DAYS = ["2025-08-26", "2025-08-27", "2025-08-28", "2025-09-01", "2025-09-08"]
TEST_DAYS = ["2025-09-15", "2025-09-18", "2025-09-20", "2025-09-22", "2025-09-25"]

#: The wider day sets C6 uses.  These are *supersets* of the five-day lists
#: above, on the same side of the same `SPLIT_DATE`: C1, C2 and C3 keep running
#: on exactly the days they published on, and nothing already reported moves.
#: Widening rather than re-picking is the point -- re-picking a split after
#: seeing results is the failure C4's third item exists to catch, and doing it
#: to ourselves would be worse than doing it by accident.
#:
#: Training days are ten of the twenty available before the split, spaced to
#: cover the window rather than clustered at its start.  The held-out set is
#: *every* day from the split to the end of the data, with no selection at all,
#: which is the only held-out set that cannot be accused of cherry-picking.
TRAIN_DAYS_EXT = [
    "2025-08-26", "2025-08-27", "2025-08-28", "2025-08-29", "2025-09-01",
    "2025-09-03", "2025-09-05", "2025-09-08", "2025-09-10", "2025-09-12",
]
TEST_DAYS_EXT = [
    "2025-09-15", "2025-09-16", "2025-09-17", "2025-09-18", "2025-09-19",
    "2025-09-20", "2025-09-21", "2025-09-22", "2025-09-23", "2025-09-24",
    "2025-09-25",
]

#: The days the USDC/USDT pair strategies (C6's APT and basis) run on.  Two
#: whole symbols per day is roughly three times the bytes of one, so this is a
#: four-plus-four subset of the lists above rather than the full ten-plus-eleven.
PAIR_TRAIN_DAYS = ["2025-08-26", "2025-08-29", "2025-09-03", "2025-09-08"]
PAIR_TEST_DAYS = ["2025-09-15", "2025-09-18", "2025-09-22", "2025-09-25"]

#: The pair's held-out set, widened to every day from the split onward -- the
#: same eleven sessions ETHUSDC is held out on.  Four sessions turned out to be
#: too few to tell a learned policy from its baseline once the seed spread was
#: measured, so this is widened the way the rest of the split was: a superset,
#: on the same side of the same date, with no day chosen.
PAIR_TEST_DAYS_EXT = list(TEST_DAYS_EXT)

#: The single day A, k and sigma are first calibrated on by hand in item 1.
CALIBRATION_DAY = TRAIN_DAYS[0]

#: Queue-position models, by the name the scripts use.  hftbacktest cannot see
#: individual orders in a market-by-price feed, so it has to guess where in the
#: queue a resting order sits; the guess decides which fills a passive strategy
#: gets, and therefore how adversely selected it is.  The reference GLFT
#: notebook uses `power2`; the newer grid-trading notebooks use `power3-3`.
#: C1 reports across all of them because on this venue the choice moves the
#: answer more than anything the strategy does.
QUEUE_MODELS = {
    "risk-adverse": ("risk_adverse_queue_model", ()),
    "power2": ("power_prob_queue_model", (2.0,)),
    "power2-2": ("power_prob_queue_model2", (2.0,)),
    "power3-3": ("power_prob_queue_model3", (3.0,)),
    "log": ("log_prob_queue_model", ()),
    "log2": ("log_prob_queue_model2", ()),
    # C11: the exact queue, from a Market-By-Order (L3) file; needs the
    # order-keyed depth (`HashMapMarketDepthBacktest`).
    "l3": ("l3_fifo_queue_model", ()),
}
DEFAULT_QUEUE_MODEL = "power2"

#: Feed latency multipliers for the synthetic order-latency model.  We have no
#: recorded order latency for this venue, so order latency is modelled as a
#: multiple of the observed feed latency, following hftbacktest's
#: "Order Latency Data" tutorial and using its multipliers unchanged.
MUL_ENTRY = 4
MUL_RESP = 3
#: The feed latency Bybit's order latency is synthesised from (C10): the
#: Binance days' typical value, so both venues are tested at the same order
#: latency. See `ensure_latency`.
BYBIT_ORDER_FEED_NS = 3_000_000


def day_path(date: str) -> Path:
    return DATA / f"{SYMBOL}_{date}.npz"


def latency_path(date: str, mul_entry: float = MUL_ENTRY, mul_resp: float = MUL_RESP) -> Path:
    if (mul_entry, mul_resp) == (MUL_ENTRY, MUL_RESP):
        return LATENCY / f"{SYMBOL}_{date}.npz"
    return LATENCY / f"{SYMBOL}_{date}_x{mul_entry:g}-{mul_resp:g}.npz"


def build_backtest(
    date: str,
    roi_ub: float | None = None,
    maker_fee: float = MAKER_FEE,
    taker_fee: float = TAKER_FEE,
    latency_mul: float = 1.0,
    queue_model: str = DEFAULT_QUEUE_MODEL,
) -> ROIVectorMarketDepthBacktest:
    """One asset, one day, hftbacktest's standard perp configuration.

    Each day's file opens with a depth-clear plus full snapshot within its first
    second, so no external initial snapshot is needed and days are independent.
    """
    roi_ub = roi_ub if roi_ub is not None else SYMBOLS[SYMBOL]["roi_ub"]
    method, params = QUEUE_MODELS[queue_model]
    asset = BacktestAsset()
    getattr(asset, method)(*params)
    asset = (
        asset.data([str(day_path(date))])
        .linear_asset(1.0)
        .intp_order_latency([str(latency_path(date, MUL_ENTRY * latency_mul, MUL_RESP * latency_mul))])
        .no_partial_fill_exchange()
        .trading_value_fee_model(maker_fee, taker_fee)
        .tick_size(TICK_SIZE)
        .lot_size(LOT_SIZE)
        .roi_lb(0.0)
        .roi_ub(roi_ub)
        .last_trades_capacity(10_000)
    )
    if queue_model == "l3":
        return HashMapMarketDepthBacktest([asset])
    return ROIVectorMarketDepthBacktest([asset])


ORDER_LATENCY_DTYPE = np.dtype(
    [("req_ts", "i8"), ("exch_ts", "i8"), ("resp_ts", "i8"), ("_padding", "i8")], align=True
)


@njit
def _resample_feed_latency(ev, exch_ts, local_ts, bucket_ns, out_exch, out_local):
    """Last (exch_ts, local_ts) of each `bucket_ns` window, by local timestamp.

    A day is ~60M events and the rows are only needed once a second, so this
    walks the arrays in place.  hftbacktest ships an equivalent helper that goes
    through polars, but that copies the whole day twice and does not fit in
    memory here; `--verify-latency` checks the two agree.
    """
    n = 0
    current = -1
    for i in range(len(ev)):
        e = ev[i]
        if (e & EXCH_EVENT) != EXCH_EVENT or (e & LOCAL_EVENT) != LOCAL_EVENT:
            continue
        bucket = local_ts[i] // bucket_ns
        if bucket != current:
            if current >= 0:
                n += 1
            current = bucket
        # numba does not bounds-check, so an undersized buffer here corrupts
        # the heap rather than raising.  The caller sizes it from the day's
        # time span; this is the backstop that turns a silent memory bug into
        # a truncated result.
        if n >= len(out_exch):
            return n
        out_exch[n] = exch_ts[i]
        out_local[n] = local_ts[i]
    if current >= 0:
        n += 1
    return n


def ensure_latency(date: str, mul_entry: float = MUL_ENTRY, mul_resp: float = MUL_RESP) -> Path:
    """Synthesise order latency from this day's feed latency, once.

    We have no recorded order latency for this venue, so order latency is
    modelled as a multiple of the feed latency the data itself reveals
    (`local_ts - exch_ts`), following hftbacktest's "Order Latency Data"
    tutorial.  Entry and response latency get separate multipliers because they
    are separate hops.
    """
    out = latency_path(date, mul_entry, mul_resp)
    if out.exists():
        return out

    data = load_day(date)
    if SYMBOL.startswith("bitstamp-"):
        # C11's L3 files carry each event's exchange and local copies
        # separately (`l3data._ordered`), and their feed latency is a US
        # cloud host's to London, not a quoter's: order latency is pinned as
        # for Bybit, on a one-second grid over the session.
        grid = np.arange(int(data["local_ts"].min()), int(data["local_ts"].max()) + 1_000_000_000,
                         1_000_000_000, dtype=np.int64)
        del data
        latency = np.zeros(len(grid), dtype=ORDER_LATENCY_DTYPE)
        latency["req_ts"] = grid
        latency["exch_ts"] = grid + int(mul_entry * BYBIT_ORDER_FEED_NS)
        latency["resp_ts"] = latency["exch_ts"] + int(mul_resp * BYBIT_ORDER_FEED_NS)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f"{out.stem}.tmp{os.getpid()}.npz")
        np.savez_compressed(tmp, data=latency)
        os.replace(tmp, out)
        return out
    # One row per bucket at most, and never more than one per event.  Sizing
    # this from the event count alone is wrong for a thin symbol: a day with
    # 5 million events still has 86,400 one-second buckets.
    span = int(data["local_ts"][-1] - data["local_ts"][0])
    cap = min(span // 1_000_000_000 + 2, len(data) + 1) + 1024
    out_exch = np.zeros(cap, np.int64)
    out_local = np.zeros(cap, np.int64)
    n = _resample_feed_latency(
        data["ev"], data["exch_ts"], data["local_ts"], 1_000_000_000, out_exch, out_local
    )
    del data

    feed = out_local[:n] - out_exch[:n]
    if SYMBOL.startswith("bybit-"):
        # Bybit's 200-level book is pushed in 10-100 ms batches, so its
        # feed latency is mostly batching, which delays the data and not our
        # orders. Order latency is pinned to the Binance days' typical feed
        # latency instead (~3 ms, a ~20 ms round trip at the multipliers),
        # so the two venues are compared at the same order latency (C10).
        feed = np.full(n, BYBIT_ORDER_FEED_NS, np.int64)
    latency = np.zeros(n, dtype=ORDER_LATENCY_DTYPE)
    latency["req_ts"] = out_local[:n]
    latency["exch_ts"] = out_local[:n] + (mul_entry * feed).astype(np.int64)
    latency["resp_ts"] = latency["exch_ts"] + (mul_resp * feed).astype(np.int64)

    out.parent.mkdir(parents=True, exist_ok=True)
    # Written under a temporary name and renamed: parallel workers that need
    # the same day's latency at once otherwise read each other's half-written
    # file, which hftbacktest rejects as an invalid zip and aborts on (C8).
    tmp = out.with_name(f"{out.stem}.tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, data=latency)
    os.replace(tmp, out)
    return out


def load_day(date: str) -> np.ndarray:
    return np.load(day_path(date))["data"]
