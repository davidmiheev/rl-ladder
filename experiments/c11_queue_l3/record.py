"""Record free order-by-order (L3) feeds: Bitstamp and Bitfinex.

Binance and Bybit publish only price-level books, so every Part C backtest
guessed where our order sat in the queue (`queue_model`), and C10 found the
guess moves results by several points. Two venues publish every order:

* **Bitstamp** -- `live_orders_<pair>` (each order's creation, change and
  deletion, with its id, price, size and a microsecond timestamp) and
  `live_trades_<pair>` (each trade, with both orders' ids), plus a REST
  snapshot of the whole book with order ids (`order_book/<pair>/?group=2`),
  taken at the start and every `SNAPSHOT_S` so a gap can be healed;
* **Bitfinex** -- the raw book (`book`, precision `R0`): the best 250 orders
  a side by id, and its trades.

Every message is written as one line `<local receive time, ns> <raw JSON>`
to `<out>/<venue>_<channel>_<YYYYmmddHH>.jsonl.gz`, a new file every hour.
Nothing is parsed here beyond what reconnection needs.

    python experiments/c11_queue_l3/record.py --out data/l3 --hours 11.5
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import ssl
import time
import urllib.request
from pathlib import Path

PAIRS = {"bitstamp": ("ethusd", "xrpusd", "suiusd"), "bitfinex": ("tETHUSD", "tXRPUSD", "tSUIUSD")}
SNAPSHOT_S = 600


def _ssl() -> ssl.SSLContext:
    ca = os.environ.get("SSL_CERT_FILE")
    return ssl.create_default_context(cafile=ca) if ca else ssl.create_default_context()


class Sink:
    """Hourly-rotated gzip files, one per (venue, stream)."""

    def __init__(self, out: Path):
        self.out = out
        self.files: dict[str, tuple[str, gzip.GzipFile]] = {}
        self.flushed = time.time()
        out.mkdir(parents=True, exist_ok=True)

    def write(self, stream: str, raw: str) -> None:
        hour = time.strftime("%Y%m%d%H", time.gmtime())
        cur = self.files.get(stream)
        if cur is None or cur[0] != hour:
            if cur is not None:
                cur[1].close()
            f = gzip.open(self.out / f"{stream}_{hour}.jsonl.gz", "at", compresslevel=6)
            self.files[stream] = (hour, f)
        self.files[stream][1].write(f"{time.time_ns()} {raw}\n")
        # Flush every 30 s, so a session stopped without closing its files
        # keeps all but the last half minute.
        now = time.time()
        if now - self.flushed > 30:
            for _, f in self.files.values():
                f.flush()
            self.flushed = now

    def close(self) -> None:
        for _, f in self.files.values():
            f.close()


async def bitstamp(sink: Sink, deadline: float) -> None:
    import websockets

    while time.time() < deadline:
        try:
            async with websockets.connect("wss://ws.bitstamp.net", ssl=_ssl(), max_size=None,
                                          ping_interval=20) as ws:
                for p in PAIRS["bitstamp"]:
                    for ch in ("live_orders", "live_trades"):
                        await ws.send(json.dumps({"event": "bts:subscribe",
                                                  "data": {"channel": f"{ch}_{p}"}}))
                while time.time() < deadline:
                    raw = await asyncio.wait_for(ws.recv(), timeout=60)
                    sink.write("bitstamp_ws", raw)
                    if '"bts:request_reconnect"' in raw:
                        break
        except Exception as exc:  # reconnect after any network error
            sink.write("bitstamp_ws", json.dumps({"recorder": "disconnect", "error": repr(exc)}))
            await asyncio.sleep(2)


async def bitstamp_snapshots(sink: Sink, deadline: float) -> None:
    loop = asyncio.get_running_loop()
    while time.time() < deadline:
        for p in PAIRS["bitstamp"]:
            url = f"https://www.bitstamp.net/api/v2/order_book/{p}/?group=2"
            try:
                body = await loop.run_in_executor(
                    None, lambda u=url: urllib.request.urlopen(u, timeout=30, context=_ssl()).read())
                sink.write(f"bitstamp_snap_{p}", json.dumps({"pair": p, "book": json.loads(body)}))
            except Exception as exc:
                sink.write(f"bitstamp_snap_{p}", json.dumps({"pair": p, "error": repr(exc)}))
        await asyncio.sleep(max(0.0, min(SNAPSHOT_S, deadline - time.time())))


async def bitfinex(sink: Sink, deadline: float) -> None:
    import websockets

    while time.time() < deadline:
        try:
            async with websockets.connect("wss://api-pub.bitfinex.com/ws/2", ssl=_ssl(), max_size=None,
                                          ping_interval=20) as ws:
                # Sequence numbers and timestamps on every message.
                await ws.send(json.dumps({"event": "conf", "flags": 65536 + 32768}))
                for s in PAIRS["bitfinex"]:
                    await ws.send(json.dumps({"event": "subscribe", "channel": "book", "symbol": s,
                                              "prec": "R0", "len": "250"}))
                    await ws.send(json.dumps({"event": "subscribe", "channel": "trades", "symbol": s}))
                while time.time() < deadline:
                    raw = await asyncio.wait_for(ws.recv(), timeout=60)
                    sink.write("bitfinex_ws", raw)
        except Exception as exc:
            sink.write("bitfinex_ws", json.dumps({"recorder": "disconnect", "error": repr(exc)}))
            await asyncio.sleep(2)


async def main_async(out: Path, hours: float) -> None:
    sink = Sink(out)
    deadline = time.time() + hours * 3600
    try:
        await asyncio.gather(bitstamp(sink, deadline), bitstamp_snapshots(sink, deadline),
                             bitfinex(sink, deadline))
    finally:
        sink.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/l3")
    parser.add_argument("--hours", type=float, default=11.5)
    args = parser.parse_args()
    asyncio.run(main_async(Path(args.out), args.hours))


if __name__ == "__main__":
    main()
