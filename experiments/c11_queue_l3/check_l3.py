"""Check an L3 file the way hftbacktest will read it: on each clock, every
modify, cancel and fill names an order that clock has seen added (and not
yet cancelled). Prints the first violation per clock, or OK.

    python experiments/c11_queue_l3/check_l3.py data/npz/bitstamp-ethusd-l3_<session>.npz
"""

import sys

import numpy as np
from hftbacktest.types import (ADD_ORDER_EVENT, BUY_EVENT, CANCEL_ORDER_EVENT, DEPTH_CLEAR_EVENT,
                               EXCH_EVENT, FILL_EVENT, LOCAL_EVENT, MODIFY_ORDER_EVENT)


def check(path: str) -> bool:
    d = np.load(path)["data"]
    ok = True
    for flag, ts, name in ((EXCH_EVENT, "exch_ts", "exchange"), (LOCAL_EVENT, "local_ts", "local")):
        rows = d[(d["ev"] & flag) == flag]
        live: dict[int, bool] = {}
        for i, r in enumerate(rows):
            k = int(r["ev"]) & 0xFF
            oid = int(r["order_id"])
            buy = bool(r["ev"] & BUY_EVENT)
            if k == DEPTH_CLEAR_EVENT:
                live = {o: b for o, b in live.items() if b != buy}
            elif k == ADD_ORDER_EVENT:
                if oid in live:
                    print(f"{name}: duplicate add of {oid} at row {i}, {r[ts]}")
                    ok = False
                    break
                live[oid] = buy
            elif k in (MODIFY_ORDER_EVENT, CANCEL_ORDER_EVENT) or (k == FILL_EVENT and flag == EXCH_EVENT):
                if oid not in live:
                    print(f"{name}: {k} on unknown order {oid} at row {i}, {r[ts]}")
                    ok = False
                    break
                if k == CANCEL_ORDER_EVENT:
                    del live[oid]
        else:
            print(f"{name}: OK ({len(rows)} events)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if all(check(p) for p in sys.argv[1:]) else 1)
