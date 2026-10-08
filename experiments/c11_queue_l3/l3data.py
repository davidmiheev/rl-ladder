"""Recorded Bitstamp L3 feeds -> hftbacktest data, order by order and by level.

`record.py` writes Bitstamp's raw messages. For one pair and one recording
session this replays them into a book of individual orders and writes two
files that describe the same market:

    data/npz/bitstamp-<pair>-l3_<session>.npz   Market-By-Order: every order's
        add, modify and cancel by id, a FILL on the resting order of each
        trade, and the trades -- for hftbacktest's L3 FIFO queue model;
    data/npz/bitstamp-<pair>_<session>.npz      Market-By-Price: the same book
        summed by price level (a DEPTH event whenever a level's total
        changes) and the same trades -- what Binance and Bybit publish, for
        the L2 queue models every Part C result used.

Replaying a strategy on both asks the question C11 is for: which L2 queue
model reproduces the fills the exact queue gives?

Choices, each where it is made:

* exchange time is Bitstamp's `microtimestamp`; local time is the recorder's
  receive time;
* the book starts from the session's first REST snapshot (whole book, by
  order id); websocket events up to the snapshot's time are skipped;
* after a recorder disconnect the book is rebuilt from the next snapshot
  (clear, then re-add), so a gap costs only the time to that snapshot;
* a trade's FILL names the maker order (the sell order when the taker
  bought); Bitstamp then sends that order's `order_changed` (partial fill)
  or `order_deleted` (filled), which become MODIFY and CANCEL;
* events are ordered by exchange time, a FILL before the order update at the
  same microsecond.

    python experiments/c11_queue_l3/l3data.py --raw data/l3 --pair ethusd
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "c1_glft_baseline"))

import config  # noqa: E402
from hftbacktest.data import validate_event_order  # noqa: E402
from hftbacktest.data.validation import correct_local_timestamp  # noqa: E402
from hftbacktest.types import (  # noqa: E402
    ADD_ORDER_EVENT, BUY_EVENT, CANCEL_ORDER_EVENT, DEPTH_CLEAR_EVENT, DEPTH_EVENT,
    DEPTH_SNAPSHOT_EVENT, EXCH_EVENT, FILL_EVENT, LOCAL_EVENT, MODIFY_ORDER_EVENT, SELL_EVENT,
    TRADE_EVENT, event_dtype,
)

BOTH = EXCH_EVENT | LOCAL_EVENT


def name(pair: str, l3: bool) -> str:
    return f"bitstamp-{pair}" + ("-l3" if l3 else "")


def _lines(raw: Path, pattern: str):
    for path in sorted(raw.glob(pattern)):
        # A recorder stopped mid-hour leaves a gzip stream without its end:
        # keep the complete lines before it.
        with gzip.open(path, "rt") as f:
            try:
                for line in f:
                    if line.endswith("\n"):
                        local, _, msg = line.partition(" ")
                        yield int(local), msg
            except EOFError:
                pass


def _messages(raw: Path, pair: str):
    """Websocket and snapshot messages for `pair`, by local receive time."""
    import heapq

    def ws():
        tag_o, tag_t = f'"live_orders_{pair}"', f'"live_trades_{pair}"'
        for local, msg in _lines(raw, "bitstamp_ws_*.jsonl.gz"):
            if tag_o in msg or tag_t in msg or '"recorder"' in msg:
                yield local, 1, msg

    def snaps():
        for local, msg in _lines(raw, f"bitstamp_snap_{pair}_*.jsonl.gz"):
            yield local, 0, msg

    return heapq.merge(ws(), snaps())


class _Out:
    def __init__(self):
        self.rows: list[tuple] = []

    def add(self, ev, exch, local, px, qty, oid=0):
        self.rows.append((ev, exch, local, px, qty, oid))

    def array(self) -> np.ndarray:
        a = np.zeros(len(self.rows), event_dtype)
        if self.rows:
            ev, exch, local, px, qty, oid = zip(*self.rows)
            a["ev"], a["exch_ts"], a["local_ts"] = ev, exch, local
            a["px"], a["qty"], a["order_id"] = px, qty, oid
        return a


def _ordered(a: np.ndarray) -> np.ndarray:
    """Every event twice: an exchange copy in exchange-time order and a local
    copy in local-time order, interleaved by their own timestamps.

    Bitstamp's order and trade channels arrive with their own delays, so the
    two orders disagree often; hftbacktest's `correct_event_order` did not
    produce a valid sequence on these files. Splitting is valid by
    construction, at twice the rows.
    """
    ex = a[np.argsort(a["exch_ts"], kind="stable")].copy()
    lo = a[np.argsort(a["local_ts"], kind="stable")].copy()
    ex["ev"] &= ~np.uint64(LOCAL_EVENT)
    lo["ev"] &= ~np.uint64(EXCH_EVENT)
    both = np.concatenate([ex, lo])
    key = np.concatenate([ex["exch_ts"], lo["local_ts"]])
    return both[np.argsort(key, kind="stable")]


def convert(raw: Path, pair: str, session: str) -> dict:
    """Write both files for one recorded session; return diagnostics."""
    l3, l2 = _Out(), _Out()
    orders: dict[int, tuple[int, float, float]] = {}  # id -> (side +1/-1, px, qty)
    levels: dict[int, dict[float, float]] = {1: defaultdict(float), -1: defaultdict(float)}
    live = False  # False until a snapshot has been laid down (and after a disconnect)
    snap_ts = 0
    stats = defaultdict(int)

    def level(side, px, delta, exch, local):
        book = levels[side]
        # Rounded: Bitstamp sizes have at most 8 decimals, and float sums
        # would otherwise leave residues that look like a tiny lot size.
        book[px] = round(book[px] + delta, 8)
        if book[px] <= 0.0:
            book.pop(px, None)
            q = 0.0
        else:
            q = book[px]
        l2.add(DEPTH_EVENT | BOTH | (BUY_EVENT if side > 0 else SELL_EVENT), exch, local, px, q)

    def lay_snapshot(book, local):
        nonlocal snap_ts
        exch = int(book["microtimestamp"]) * 1000
        orders.clear()
        for side, key in ((1, "bids"), (-1, "asks")):
            flag = BUY_EVENT if side > 0 else SELL_EVENT
            rows = book[key]
            far = float(rows[-1][0]) if rows else 0.0
            l3.add(DEPTH_CLEAR_EVENT | BOTH | flag, exch, local, far, 0.0)
            l2.add(DEPTH_CLEAR_EVENT | BOTH | flag, exch, local, far, 0.0)
            levels[side].clear()
            for p, q, oid in rows:
                p, q, oid = float(p), float(q), int(oid)
                orders[oid] = (side, p, q)
                levels[side][p] = round(levels[side][p] + q, 8)
                l3.add(ADD_ORDER_EVENT | BOTH | flag, exch, local, p, q, oid)
            for p, q in levels[side].items():
                l2.add(DEPTH_SNAPSHOT_EVENT | BOTH | flag, exch, local, p, q)
        snap_ts = exch
        stats["snapshots_laid"] += 1

    # Exchange-time order across the two channels (they arrive with their own
    # delays): at one microsecond, trades first (their FILL must precede the
    # maker's update), then order updates, then a snapshot (which already
    # includes them). A disconnect stops the book until the next snapshot.
    msgs = []
    for local, kind, msg in _messages(raw, pair):
        o = json.loads(msg)
        if kind == 0:
            if "book" in o:
                msgs.append((int(o["book"]["microtimestamp"]) * 1000, 2, local, "snapshot", o))
        elif "recorder" in o:
            msgs.append((local, 1, local, "disconnect", o))
        elif "microtimestamp" in o.get("data", {}):
            prio = 0 if o.get("event") == "trade" else 1
            msgs.append((int(o["data"]["microtimestamp"]) * 1000, prio, local, "event", o))
    msgs.sort(key=lambda m: (m[0], m[1]))

    for _, _, local, kind, o in msgs:
        if kind == "snapshot":
            if not live:
                lay_snapshot(o["book"], local)
                live = True
            continue
        if kind == "disconnect":
            live = False
            stats["disconnects"] += 1
            continue
        if not live:
            continue
        if "data" not in o or "microtimestamp" not in o.get("data", {}):
            continue
        d, ev = o["data"], o.get("event")
        exch = int(d["microtimestamp"]) * 1000
        if exch <= snap_ts:
            continue
        if ev == "trade":
            taker_buy = int(d["type"]) == 0
            maker = int(d["sell_order_id"] if taker_buy else d["buy_order_id"])
            px, qty = float(d["price"]), float(d["amount"])
            for out in (l3, l2):
                out.add(TRADE_EVENT | BOTH | (BUY_EVENT if taker_buy else SELL_EVENT), exch, local, px, qty)
            if maker in orders:
                side = orders[maker][0]
                # Ordered before the maker's own update at the same time (below).
                l3.add(FILL_EVENT | BOTH | (BUY_EVENT if side > 0 else SELL_EVENT),
                       exch - 1, local, px, qty, maker)
                stats["fills"] += 1
            else:
                stats["fills_unknown_maker"] += 1
            continue
        oid = int(d["id"])
        side = 1 if int(d["order_type"]) == 0 else -1
        flag = BUY_EVENT if side > 0 else SELL_EVENT
        px, qty = float(d["price"]), float(d["amount"])
        if ev == "order_created":
            if oid in orders or qty <= 0:
                continue
            orders[oid] = (side, px, qty)
            l3.add(ADD_ORDER_EVENT | BOTH | flag, exch, local, px, qty, oid)
            level(side, px, qty, exch, local)
            stats["adds"] += 1
        elif ev == "order_changed":
            if oid not in orders:
                stats["changes_unknown"] += 1
                continue
            s0, p0, q0 = orders[oid]
            orders[oid] = (side, px, qty)
            l3.add(MODIFY_ORDER_EVENT | BOTH | flag, exch, local, px, qty, oid)
            level(s0, p0, -q0, exch, local)
            level(side, px, qty, exch, local)
            stats["modifies"] += 1
        elif ev == "order_deleted":
            if oid not in orders:
                stats["deletes_unknown"] += 1
                continue
            s0, p0, q0 = orders.pop(oid)
            l3.add(CANCEL_ORDER_EVENT | BOTH | (BUY_EVENT if s0 > 0 else SELL_EVENT), exch, local,
                   p0, 0.0, oid)
            level(s0, p0, -q0, exch, local)
            stats["cancels"] += 1

    result = {"pair": pair, "session": session, **stats}
    for l3flag, out in ((True, l3), (False, l2)):
        a = _ordered(correct_local_timestamp(out.array(), 0))
        validate_event_order(a)
        path = config.DATA / f"{name(pair, l3flag)}_{session}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.tmp{os.getpid()}.npz")
        np.savez_compressed(tmp, data=a)
        os.replace(tmp, path)
        result["rows_l3" if l3flag else "rows_l2"] = len(a)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", required=True, help="a directory of record.py files (one session)")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--session", default=None, help="session label (default: the directory name)")
    args = parser.parse_args()
    raw = Path(args.raw)
    print(json.dumps(convert(raw, args.pair, args.session or raw.name)))


if __name__ == "__main__":
    main()
