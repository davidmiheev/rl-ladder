"""The frozen C8 configurations on their own Binance month, every session.

C10 compares each configuration on Binance (where C8 chose it) with the same
configuration a year forward on Bybit. This replays each one, as C8 chose it,
on all 31 Binance sessions of the study month (24 training, 7 held out), so
the "before" side has a per-session path too, not only C8's totals.

    python experiments/c10_forward/binance_month.py --shard 0/3     # needs HF_TOKEN
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "experiments" / "c7_signals")]

import forward  # noqa: E402  (puts c8, c6 and c1 on the path)
import c8  # noqa: E402


def binance_symbol(label: str) -> str:
    """'XRPUSDT OBI → XRPUSDT' -> 'xrpusdt' (the Binance contract C8 chose on)."""
    return label.split()[0].lower()


def configs() -> dict[str, dict]:
    return {label: c8.config("obi", binance_symbol(label), sets)
            for label, (_, sets) in forward.FROZEN.items()}


def main() -> None:
    import config
    import snapshots

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shard", default="0/1")
    args = parser.parse_args()
    index, count = (int(x) for x in args.shard.split("/"))
    cfgs = {c8.key(c): c for c in configs().values()}.values()  # SUIUSDC appears twice
    units = [(c, d) for d in c8.DAYS for c in cfgs if c8.result(c, d) is None][index::count]
    print(f"### Binance month shard {args.shard}: {len(units)} replays", flush=True)
    for cfg, date in units:
        t0 = time.time()
        path = config.DATA / f"{cfg['symbol']}_{date}.npz"
        try:
            fetched = not path.exists()
            if fetched:
                snapshots.download(cfg["symbol"], date)
            c8.run_one(cfg, date)
            if fetched:
                path.unlink(missing_ok=True)
            status = "ok"
        except Exception as exc:
            status = f"FAILED {exc!r}"
        print(cfg["symbol"], date, status, f"{time.time() - t0:.0f}s", flush=True)
    print(f"### SHARD DONE {args.shard}", flush=True)


if __name__ == "__main__":
    main()
