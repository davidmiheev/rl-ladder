"""Which assets C6 runs on, and how a new one gets registered.

C1's cross-asset study established that a negative result on one asset is not
a result about a method, and measured ten assets on one day each to show it.
C6 needs the same breadth, but with a train/held-out split so the widened
spread and the learned policy can each be tested out of sample.

Two tiers, because compute is the binding constraint on four cores:

- **deep** -- ETHUSDC and XRPUSDC, on the full ten training and eleven (or
  four, for the pair) held-out sessions. This is where the RL comparison
  lives and where any headline number comes from.
- **breadth** -- seven assets on three training and three held-out sessions.
  Enough to say whether a finding travels; not enough to be a track record,
  and labelled as such everywhere it appears.

The breadth days are chosen to be a subset of the deep split, on the correct
sides of it, so no new day is introduced and nothing has to be re-decided.

Tick and lot sizes are **measured from each day's data** rather than taken
from a table, following `run_cross_asset.py`: a venue's published tick is a
fact about the venue's API docs, and what the backtest needs is a fact about
the file. The measured values are cached in `assets.json` so a second run
does not re-read a 400 MB day to rediscover them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "c1_glft_baseline"))

import config  # noqa: E402

CACHE = Path(__file__).resolve().parent / "assets.json"

#: Breadth sessions. A subset of `TRAIN_DAYS_EXT` and `TEST_DAYS_EXT`, so the
#: split is the one C1 fixed and C6 widened, not a third one.
CROSS_TRAIN_DAYS = ["2025-08-26", "2025-08-29", "2025-09-08"]
CROSS_TEST_DAYS = ["2025-09-15", "2025-09-22", "2025-09-25"]

#: Assets the three single-leg strategies run on. BTC is excluded: its days
#: are three to five times the bytes of any other symbol and it would cost
#: more than the rest of the breadth study combined.
SINGLE_LEG = [
    "ethusdc", "xrpusdc", "xrpusdt", "suiusdc", "suiusdt",
    "enausdc", "enausdt", "avaxusdc", "bnbusdc",
]

#: Traded leg -> reference leg, for the two pair strategies. Every pair is
#: the same underlying quoted in two stablecoins on the same venue, so the
#: residual is the USDC/USDT basis plus a venue lead-lag.
PAIRS = {"xrpusdc": "xrpusdt", "suiusdc": "suiusdt", "enausdc": "enausdt"}


#: The two assets with the full split. Everything else gets the breadth
#: sessions, and the table labels it by the session count so the two are
#: never read as the same kind of evidence.
DEEP = {"ethusdc", "xrpusdc"}


def train_days(symbol: str) -> list[str]:
    if symbol == "ethusdc":
        return config.TRAIN_DAYS_EXT
    if symbol == "xrpusdc":
        return config.PAIR_TRAIN_DAYS
    return CROSS_TRAIN_DAYS


def test_days(symbol: str) -> list[str]:
    if symbol == "ethusdc":
        return config.TEST_DAYS_EXT
    if symbol == "xrpusdc":
        return config.PAIR_TEST_DAYS_EXT
    return CROSS_TEST_DAYS


def day_sets(symbol: str) -> dict[str, list[str]]:
    """Every day set a script may ask for, for this symbol."""
    train, test = train_days(symbol), test_days(symbol)
    return {
        "one": train[:1],
        "train": train,
        "test": test,
        "all": train + test,
        "cross-train": CROSS_TRAIN_DAYS,
        "cross-test": CROSS_TEST_DAYS,
        "cross-all": CROSS_TRAIN_DAYS + CROSS_TEST_DAYS,
    }


def detect_step(values: np.ndarray) -> float:
    """The smallest non-zero gap between sorted distinct values.

    Which is the tick (or lot) size as the *file* expresses it. Reading it off
    the data rather than a table is what lets a symbol be added without anyone
    looking anything up, and it is how `run_cross_asset.py` does it.
    """
    distinct = np.unique(values)
    if len(distinct) < 2:
        return float("nan")
    gaps = np.diff(distinct)
    gaps = gaps[gaps > 0]
    return float(np.min(gaps)) if len(gaps) else float("nan")


#: Series already written by earlier phases that carry a mid price, checked
#: before falling back to opening a whole day.  A day file is 40-460 MB
#: compressed and several gigabytes once expanded, and all that is wanted
#: from it here is a median.
MEASURED = Path(__file__).resolve().parents[2] / "data" / "measured"
PRECOMPUTED = Path(__file__).resolve().parents[2] / "data" / "precompute"


def _price(symbol: str, date: str, tick: float | None) -> float | None:
    """Median mid price in currency units, from a series an earlier phase wrote.

    The two sources disagree on units and nothing in either file says so:
    C1's `measure_market` records the mid in **ticks**, while C6's own
    `precompute` records it in currency.  Multiplying one and not the other
    is the whole of the difference between $4,442 and $444,231 for ETH.
    """
    path = MEASURED / f"{symbol}_{date}.npz"
    if path.exists() and tick is not None:
        return float(np.nanmedian(np.load(path)["mid_price"])) * tick
    path = PRECOMPUTED / f"mid_{symbol}_{date}.npy"
    if path.exists():
        return float(np.nanmedian(np.load(path)))
    return None


def _measure(symbol: str, date: str) -> dict[str, float]:
    """Tick size, lot size and price level, from the file rather than a table.

    The price comes from trade prints, not the book: the book's outer levels
    run to absurd prices and would put the ROI bound -- and the depth
    vector's memory with it -- somewhere silly.
    """
    known = config.SYMBOLS.get(symbol)
    price = _price(symbol, date, known["tick_size"] if known else None)
    if known is not None and price is not None:
        # Tick, lot and ROI bound were fixed for this symbol in C1 and are
        # load-bearing for results already published; only the price level is
        # missing, and it is wanted for the tick-in-bp column alone.
        return {**known, "price": price}

    day = np.load(config.DATA / f"{symbol}_{date}.npz")["data"]
    tick = detect_step(day["px"])
    lot = detect_step(day["qty"])
    if price is None:
        trades = day["px"][(day["ev"] & 0xFF) == 2]
        price = float(np.median(trades)) if len(trades) else float(np.median(day["px"]))
    # A price that is not within a factor of a few of the traded range means
    # a unit slipped somewhere upstream; better to stop than to size a
    # position off it.
    sample = day["px"][(day["ev"] & 0xFF) == 2]
    if len(sample):
        reference = float(np.median(sample))
        if not 0.2 < price / reference < 5.0:
            raise SystemExit(
                f"{symbol}: price {price:g} disagrees with trade prints {reference:g}"
            )
    del day
    if known is not None:
        return {**known, "price": price}
    return {"tick_size": tick, "lot_size": lot, "price": price, "roi_ub": 4.0 * price}


def ensure_registered(symbol: str, date: str | None = None) -> dict[str, float]:
    """Make `symbol` usable by `config`, measuring it once if need be."""
    cached = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    if symbol not in cached:
        cached[symbol] = _measure(symbol, date or CROSS_TRAIN_DAYS[0])
        CACHE.write_text(json.dumps(dict(sorted(cached.items())), indent=2) + "\n")
    entry = cached[symbol]
    config.register_symbol(symbol, entry["tick_size"], entry["lot_size"], entry["roi_ub"])
    if symbol in PAIRS:
        config.REFERENCE_LEG[symbol] = PAIRS[symbol]
    return entry


def register_all() -> dict[str, dict[str, float]]:
    """Register every asset C6 knows about, and return what was measured."""
    return {symbol: ensure_registered(symbol) for symbol in SINGLE_LEG}


def main() -> None:
    print(f"{'symbol':>10} {'tick':>10} {'lot':>8} {'price':>11} {'tick bp':>9} {'reference':>10}")
    for symbol, entry in register_all().items():
        tick_bp = entry["tick_size"] / entry["price"] * 1e4
        print(f"{symbol:>10} {entry['tick_size']:10g} {entry['lot_size']:8g} "
              f"{entry['price']:11.4f} {tick_bp:9.4f} {PAIRS.get(symbol, '-'):>10}")


if __name__ == "__main__":
    main()
