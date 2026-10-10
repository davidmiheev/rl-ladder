"""Market data from outside the traded contract's own book, reduced per day.

C9 found that the value of a better fair price is in rare 1-3 s moves the
traded contract's own book does not anticipate. Large moves usually start
somewhere else; these are the free sources that could see them first, for
the same asset (`ETH` for ETHUSDC, ...):

| source        | what                                   | from                                  |
| binance-perp  | the USDT-margined perpetual's trades   | data.binance.vision, futures/um trades |
| binance-spot  | the USDT spot pair's trades            | data.binance.vision, spot trades       |
| bybit-perp    | Bybit's USDT perpetual's trades        | public.bybit.com/trading               |
| bybit-book    | Bybit's USDT perpetual's best bid/ask  | quote-saver.bycsi.com ob200 books      |

Each is reduced to `data/c10_ext/<source>_<asset>_<date>.npz` with `t` (the
source's own timestamp, ns), `px` (trade price or book mid) and `flow`
(signed notional, + for a buyer-initiated trade; 0 for the book), and the
raw file is deleted.

**When could we have seen it?** The traded books were recorded near Binance
(Tokyo); an outside event is usable `LATENCY_NS[source]` after its own
timestamp: 5 ms for Binance's other markets, 80 ms for Bybit (Singapore to
Tokyo, ~70 ms, plus its publishing delay). `leadlag.py` adds +100 ms and
+250 ms on top as stresses, because a lead that disappears under a quarter
second of extra delay is not one a quoter can use.
"""

from __future__ import annotations

import csv
import gzip
import io
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import orjson

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import bybit  # noqa: E402

OUT = REPO / "data" / "c10_ext"
RAW = REPO / "data" / "c10_raw"
SOURCES = ("binance-perp", "binance-spot", "bybit-perp", "bybit-book")
LATENCY_NS = {"binance-perp": 5_000_000, "binance-spot": 5_000_000,
              "bybit-perp": 80_000_000, "bybit-book": 80_000_000}
BV = "https://data.binance.vision/data"


def asset_of(symbol: str) -> str:
    """ethusdc -> ETH, bybit-xrpperp -> XRP."""
    s = symbol.removeprefix("bybit-").upper()
    for suffix in ("USDC", "USDT", "PERP"):
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s


def path(source: str, asset: str, date: str) -> Path:
    return OUT / f"{source}_{asset}_{date}.npz"


def _binance_trades(url: str, raw: Path, spot: bool):
    bybit._fetch(url, raw)
    with zipfile.ZipFile(raw) as z, z.open(z.namelist()[0]) as f:
        text = io.TextIOWrapper(f)
        first = text.readline()
        rows = [] if first.startswith("id,") else [first]
        rows += text.readlines()
    # Columns: id, price, qty, quote_qty, time, is_buyer_maker[, is_best_match]
    data = [r.split(",") for r in rows if r.strip()]
    t = np.array([int(r[4]) for r in data], np.int64)
    # Binance's spot archive switched from ms to us timestamps in 2025.
    t = t * (1_000 if t[0] > 10**15 else 1_000_000)
    px = np.array([float(r[1]) for r in data])
    quote = np.array([float(r[3]) for r in data])
    maker_buy = np.array([r[5].strip().lower() == "true" for r in data])
    flow = np.where(maker_buy, -quote, quote)  # buyer is maker -> the taker sold
    return t, px, flow


def _bybit_trades(asset: str, date: str, raw: Path):
    bybit._fetch(bybit.TRADE_URL.format(sym=f"{asset}USDT", date=date), raw)
    with gzip.open(raw, "rt") as f:
        reader = csv.reader(f)
        header = next(reader)
        i_t, i_side, i_px, i_qty = (header.index(c) for c in ("timestamp", "side", "price", "size"))
        rows = list(reader)
    t = np.array([round(float(r[i_t]) * 1e9) for r in rows], np.int64)
    px = np.array([float(r[i_px]) for r in rows])
    qty = np.array([float(r[i_qty]) for r in rows])
    flow = np.where(np.array([r[i_side] == "Buy" for r in rows]), px * qty, -px * qty)
    order = np.argsort(t, kind="stable")
    return t[order], px[order], flow[order]


def _bybit_book(asset: str, date: str, raw: Path):
    """Best bid/ask mid after every message, timed by Bybit's publish time."""
    bybit._fetch(bybit.BOOK_URL.format(cat="linear", sym=f"{asset}USDT", date=date), raw)
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    best_b = best_a = None
    ts_out, mid_out = [], []
    with zipfile.ZipFile(raw) as z, z.open(z.namelist()[0]) as f:
        for line in f:
            o = orjson.loads(line)
            d = o["data"]
            if o["type"] == "snapshot":
                bids.clear()
                asks.clear()
            for side, book in (("b", bids), ("a", asks)):
                for p, q in d.get(side, ()):
                    p = float(p)
                    if float(q) == 0.0:
                        book.pop(p, None)
                    else:
                        book[p] = 1.0
            if not bids or not asks:
                continue
            # Recomputing the extremes of <=200 levels is cheap next to parsing.
            best_b, best_a = max(bids), min(asks)
            if best_a > best_b:
                ts_out.append(int(o["ts"]) * 1_000_000)
                mid_out.append((best_b + best_a) / 2.0)
    t = np.array(ts_out, np.int64)
    return t, np.array(mid_out), np.zeros(len(t))


def ensure(source: str, asset: str, date: str) -> Path:
    out = path(source, asset, date)
    if out.exists():
        return out
    raw = RAW / f"{source}_{asset}_{date}.raw"
    sym = f"{asset}USDT"
    if source == "binance-perp":
        t, px, flow = _binance_trades(f"{BV}/futures/um/daily/trades/{sym}/{sym}-trades-{date}.zip",
                                      raw, spot=False)
    elif source == "binance-spot":
        t, px, flow = _binance_trades(f"{BV}/spot/daily/trades/{sym}/{sym}-trades-{date}.zip",
                                      raw, spot=True)
    elif source == "bybit-perp":
        t, px, flow = _bybit_trades(asset, date, raw)
    elif source == "bybit-book":
        t, px, flow = _bybit_book(asset, date, raw)
    else:
        raise ValueError(source)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(f"{out.stem}.tmp{os.getpid()}.npz")
    np.savez_compressed(part, t=t, px=px, flow=flow.astype(np.float64))
    os.replace(part, out)
    raw.unlink(missing_ok=True)
    return out


def load(source: str, asset: str, date: str) -> dict[str, np.ndarray]:
    with np.load(ensure(source, asset, date)) as z:
        return {k: z[k] for k in z.files}
