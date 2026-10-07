"""Bybit's free historical order books and trades, converted for hftbacktest.

Bybit publishes, free and without an account:

* **order books** -- `quote-saver.bycsi.com/orderbook/{linear,spot}/<SYMBOL>/
  <date>_<SYMBOL>_ob200.data.zip`: one JSON message per line, the top 200
  levels, a snapshot at the start of the day and deltas after it, pushed in
  batches every 10-100 ms; each message carries `ts` (when Bybit published
  it) and, for perpetuals, `cts` (the matching engine's time);
* **trades** -- `public.bybit.com/trading/<SYMBOL>/<SYMBOL><date>.csv.gz`:
  every print, the taker's side, a 0.1 ms timestamp, and an `RPI` flag.

This writes the same `data/npz/<symbol>_<date>.npz` the Binance days use, so
every C6-C9 script runs on a Bybit day unchanged. The symbol is
`bybit-<contract>` (e.g. `bybit-xrpperp`, Bybit's USDC-settled XRP perpetual).

Three choices, each stated where it is made:

1. **Timestamps.** A book message's exchange time is its matching-engine time
   (`cts`) and its local time is its publish time plus `NETWORK_NS`, so the
   feed latency the backtest sees includes Bybit's batching delay (median
   10 ms, p99 ~95 ms on the first day checked), which is real for anyone
   reading this feed. A trade's exchange time is its print time and its
   local time adds the day's median publish delay and `NETWORK_NS`.
2. **Order latency** is *not* derived from that feed latency (see
   `config.ensure_latency`): batching delays the market data, not our orders.
3. **RPI trades are dropped.** Retail Price Improvement orders trade only
   with non-API takers and are not in the public book, so a print against
   one could never have filled our quote; keeping it would credit our queue
   position with volume that was not ours to share.

    python experiments/c10_forward/bybit.py --symbol XRPUSDT --date 2025-09-26
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import orjson

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "c1_glft_baseline"))

import config  # noqa: E402
from hftbacktest.data import correct_event_order, validate_event_order  # noqa: E402
from hftbacktest.data.validation import correct_local_timestamp  # noqa: E402
from hftbacktest.types import (  # noqa: E402
    BUY_EVENT, DEPTH_CLEAR_EVENT, DEPTH_EVENT, DEPTH_SNAPSHOT_EVENT, SELL_EVENT, TRADE_EVENT,
    event_dtype,
)

RAW = REPO / "data" / "bybit_raw"
BOOK_URL = "https://quote-saver.bycsi.com/orderbook/{cat}/{sym}/{date}_{sym}_ob200.data.zip"
TRADE_URL = "https://public.bybit.com/trading/{sym}/{sym}{date}.csv.gz"
#: One-way network time from Bybit's servers to a host near them.
NETWORK_NS = 2_000_000


def name(contract: str) -> str:
    return f"bybit-{contract.lower()}"


def contract_of(symbol: str) -> str:
    return symbol.removeprefix("bybit-").upper()


def _fetch(url: str, path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".part{os.getpid()}")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as out:
                while chunk := r.read(1 << 22):
                    out.write(chunk)
            os.replace(tmp, path)
            return path
        except Exception:  # transient network errors: back off and retry
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))
    return path


def download(contract: str, date: str) -> tuple[Path, Path]:
    book = _fetch(BOOK_URL.format(cat="linear", sym=contract, date=date),
                  RAW / f"{contract}_{date}_ob200.zip")
    trades = _fetch(TRADE_URL.format(sym=contract, date=date), RAW / f"{contract}_{date}_trades.csv.gz")
    return book, trades


class _Columns:
    """Event columns built in chunks: Python lists are flushed to numpy every
    `CHUNK` rows, so a 30-million-row day never holds its rows as objects."""

    CHUNK = 2_000_000

    def __init__(self):
        self.parts = []
        self._new()

    def _new(self):
        self.ev, self.exch, self.local, self.px, self.qty = [], [], [], [], []

    def flush(self):
        if self.ev:
            self.parts.append((np.asarray(self.ev, np.uint64), np.asarray(self.exch, np.int64),
                               np.asarray(self.local, np.int64), np.asarray(self.px, np.float64),
                               np.asarray(self.qty, np.float64)))
            self._new()

    def add(self, ev, exch, local, px, qty):
        self.ev.append(ev)
        self.exch.append(exch)
        self.local.append(local)
        self.px.append(px)
        self.qty.append(qty)
        if len(self.ev) >= self.CHUNK:
            self.flush()

    def array(self):
        self.flush()
        n = sum(len(p[0]) for p in self.parts)
        out = np.zeros(n, event_dtype)
        i = 0
        for ev, exch, local, px, qty in self.parts:
            j = i + len(ev)
            out["ev"][i:j], out["exch_ts"][i:j], out["local_ts"][i:j] = ev, exch, local
            out["px"][i:j], out["qty"][i:j] = px, qty
            i = j
        self.parts = []
        return out


def _book_rows(path: Path, cols: _Columns) -> int:
    """Append the book's events to `cols`; return the median publish delay (ns)."""
    delays = []
    with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as f:
        for line in f:
            o = orjson.loads(line)
            ts = int(o["ts"]) * 1_000_000
            cts = int(o.get("cts", o["ts"])) * 1_000_000
            if len(delays) < 200_000:
                delays.append(ts - cts)
            lt = ts + NETWORK_NS
            snapshot = o["type"] == "snapshot"
            d = o["data"]
            for key, side in (("b", BUY_EVENT), ("a", SELL_EVENT)):
                levels = d.get(key)
                if not levels:
                    continue
                if snapshot:
                    # Clear this side out to the snapshot's farthest level,
                    # then lay the snapshot down: hftbacktest's convention.
                    cols.add(DEPTH_CLEAR_EVENT | side, cts, lt, levels[-1][0], "0")
                kind = (DEPTH_SNAPSHOT_EVENT if snapshot else DEPTH_EVENT) | side
                for p, q in levels:
                    cols.add(kind, cts, lt, p, q)
    return int(np.median(delays)) if delays else 0


def convert(contract: str, date: str, keep_raw: bool = False) -> Path:
    """Download (if needed) and convert one day; returns the npz path."""
    out = config.DATA / f"{name(contract)}_{date}.npz"
    if out.exists():
        return out
    book, trades = download(contract, date)
    cols = _Columns()
    delay = _book_rows(book, cols)
    n_rpi = n_trades = 0
    with gzip.open(trades, "rt") as f:
        reader = csv.reader(f)
        header = next(reader)
        i_ts, i_side, i_px, i_qty = (header.index(c) for c in ("timestamp", "side", "price", "size"))
        i_rpi = header.index("RPI") if "RPI" in header else None
        for row in reader:
            n_trades += 1
            if i_rpi is not None and row[i_rpi] not in ("0", "", "false", "False"):
                n_rpi += 1
                continue
            t = int(round(float(row[i_ts]) * 1e9))
            cols.add(TRADE_EVENT | (BUY_EVENT if row[i_side] == "Buy" else SELL_EVENT),
                     t, t + delay + NETWORK_NS, row[i_px], row[i_qty])
    tmp = cols.array()
    del cols
    tmp = correct_local_timestamp(tmp, 0)
    data = correct_event_order(tmp, np.argsort(tmp["exch_ts"], kind="mergesort"),
                               np.argsort(tmp["local_ts"], kind="mergesort"))
    validate_event_order(data)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(f"{out.stem}.tmp{os.getpid()}.npz")
    np.savez_compressed(part, data=data)
    os.replace(part, out)
    meta = out.with_suffix(".json")
    meta.write_text(orjson.dumps({"rows": len(data), "trades": n_trades, "rpi_dropped": n_rpi,
                                  "median_publish_delay_ms": delay / 1e6}).decode())
    if not keep_raw:
        book.unlink(missing_ok=True)
        trades.unlink(missing_ok=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", required=True, help="Bybit contract, e.g. XRPUSDT or XRPPERP")
    parser.add_argument("--date", nargs="+", required=True)
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()
    for date in args.date:
        t0 = time.time()
        print(convert(args.symbol, date, args.keep_raw), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
