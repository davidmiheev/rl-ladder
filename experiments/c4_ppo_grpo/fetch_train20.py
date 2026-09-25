"""Fetch every pre-split day for ETHUSDC and XRPUSDC: the `train20` set.

C6 trained on ten of ETHUSDC's twenty pre-split days and four of XRPUSDC's,
and C4's grid chose its XRP parameters on those four -- which is thin enough
that several chosen points barely traded held out. This fetches the rest, so
selection can be repeated on all twenty. The held-out set does not change: it
is already every day from the split to the end of the data.

Disk is the constraint (a few GB free), so:

- XRPUSDT, the reference leg APT and basis need, is only ever read to build
  their precomputed tables. Each new day is downloaded, tabled, and deleted,
  and the existing raw XRPUSDT days -- whose tables are already built -- are
  deleted first. Re-running C6's breadth tier on XRPUSDT would re-download.
- ETHUSDC and XRPUSDC days are kept, since every replay reads them.

    HF_TOKEN=... python experiments/c4_ppo_grpo/fetch_train20.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "c6_strategy_zoo"))
sys.path.insert(0, str(REPO / "experiments" / "c1_glft_baseline"))
sys.path.insert(0, str(REPO / "experiments" / "c7_signals"))

import precompute  # noqa: E402
import snapshots  # noqa: E402

DATA = REPO / "data" / "npz"
SPLIT = dt.date(2025, 9, 15)
TRAIN20 = [(dt.date(2025, 8, 26) + dt.timedelta(i)).isoformat()
           for i in range((SPLIT - dt.date(2025, 8, 26)).days)]


def tabled(date: str) -> bool:
    return all((precompute.PRECOMP / f"{kind}_xrpusdc_{date}.npy").exists()
               for kind in ("apt", "basis"))


def main() -> None:
    for path in sorted(DATA.glob("xrpusdt_2025-*.npz")):
        date = path.stem.split("_")[1]
        if tabled(date):
            path.unlink()
            print(f"deleted {path.name} (tables built)", flush=True)

    for symbol in ("ethusdc", "xrpusdc"):
        for date in TRAIN20:
            if not (DATA / f"{symbol}_{date}.npz").exists():
                snapshots.download(symbol, date)
                print(f"fetched {symbol} {date}", flush=True)

    for date in TRAIN20:
        if tabled(date):
            continue
        reference = DATA / f"xrpusdt_{date}.npz"
        fetched = not reference.exists()
        if fetched:
            snapshots.download("xrpusdt", date)
        precompute.build("xrpusdc", date)
        if fetched:
            reference.unlink()
        print(f"tabled xrpusdc {date}{' (reference deleted)' if fetched else ''}", flush=True)


if __name__ == "__main__":
    main()
