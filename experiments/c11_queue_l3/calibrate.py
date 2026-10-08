"""C11: which L2 queue model gives the fills the exact (L3) queue gives?

Every Part C result came from a price-level (L2) book, where our order's place
in the queue is a guess: hftbacktest's probabilistic queue models
(`config.QUEUE_MODELS`, `power2` by default). C10's stress tests showed the
guess can move a year's result by several points. Bitstamp publishes every
order (`record.py`), so on its books the place is known exactly
(`l3_fifo_queue_model`). For each recorded session, pair and OBI floor, the
same strategy is replayed twice over the same market:

    truth   the L3 file, FIFO queue (`queue_model` "l3")
    guess   the L2 file built from the same messages (`l3data.py`), under each
            L2 queue model

and the guesses are scored against the truth on fills and on return.

    python experiments/c11_queue_l3/calibrate.py run --sessions s1 s2 --shard 0/4
    python experiments/c11_queue_l3/calibrate.py report --sessions s1 s2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "experiments" / "c8_risk_controls"),
                str(REPO / "experiments" / "c6_strategy_zoo"),
                str(REPO / "experiments" / "c1_glft_baseline")]

import c8  # noqa: E402
import l3data  # noqa: E402

PAIRS = ("ethusd", "xrpusd", "suiusd")
FLOORS_BP = (1.0, 2.0, 4.0, 8.0, 16.0)
KAPPA = 2.0
L2_MODELS = ("power2", "power2-2", "power3-3", "log", "log2", "risk-adverse")


def configs(pair: str) -> list[dict]:
    out = []
    for f in FLOORS_BP:
        sets = {"half_spread_bp": f, "markout_kappa": KAPPA}
        out.append(c8.config("obi", l3data.name(pair, True), {**sets, "queue_model": "l3"}))
        out += [c8.config("obi", l3data.name(pair, False), {**sets, "queue_model": m})
                for m in L2_MODELS]
    return out


def cmd_run(sessions: list[str], shard: str) -> None:
    import assets

    index, count = (int(x) for x in shard.split("/"))
    units = [(p, s) for s in sessions for p in PAIRS][index::count]
    for pair, session in units:
        for l3 in (True, False):
            assets.ensure_registered(l3data.name(pair, l3), session)
        for cfg in configs(pair):
            if c8.result(cfg, session) is not None:
                continue
            t0 = time.time()
            try:
                c8.run_one(cfg, session)
                status = "ok"
            except Exception as exc:
                status = f"FAILED {exc!r}"
            print(pair, session, cfg["sets"]["half_spread_bp"], cfg["sets"]["queue_model"], status,
                  f"{time.time() - t0:.0f}s", flush=True)
    print(f"### SHARD DONE {shard}", flush=True)


def table(sessions: list[str]) -> dict:
    """Per (pair, floor): the truth and each guess, summed over sessions."""
    out = {}
    for pair in PAIRS:
        cfgs = configs(pair)
        for f in FLOORS_BP:
            row = {}
            for cfg in cfgs:
                if cfg["sets"]["half_spread_bp"] != f:
                    continue
                rs = [c8.result(cfg, s) for s in sessions]
                if any(r is None for r in rs):
                    continue
                ret = float(np.prod([1 + r["Return"] for r in rs]) - 1) * 100
                fills = float(sum(r["DailyNumberOfTrades"] for r in rs))
                row[cfg["sets"]["queue_model"]] = {"return": ret, "fills": fills}
            if "l3" in row:
                out[f"{pair}|{f:g}"] = row
    return out


def cmd_report(sessions: list[str]) -> None:
    t = table(sessions)
    print("| pair | floor, bp | L3 (truth): return % / fills | " + " | ".join(L2_MODELS) + " |")
    print("| --- | --- | --- |" + " --- |" * len(L2_MODELS))
    err = {m: {"fills": [], "ret": []} for m in L2_MODELS}
    for key, row in t.items():
        pair, f = key.split("|")
        truth = row["l3"]
        cells = []
        for m in L2_MODELS:
            g = row.get(m)
            if g is None:
                cells.append("–")
                continue
            cells.append(f"{g['return']:+.3f} / {g['fills']:.0f}")
            if truth["fills"] > 0:
                err[m]["fills"].append(np.log((g["fills"] + 1) / (truth["fills"] + 1)))
            err[m]["ret"].append(g["return"] - truth["return"])
        print(f"| {pair} | {f} | {truth['return']:+.3f} / {truth['fills']:.0f} | " + " | ".join(cells) + " |")
    print("\n| L2 model | fills vs truth (geometric mean ratio) | return error, mean / mean abs (points) |")
    print("| --- | --- | --- |")
    for m, e in err.items():
        if e["ret"]:
            print(f"| {m} | ×{np.exp(np.mean(e['fills'])):.2f} | {np.mean(e['ret']):+.3f} / "
                  f"{np.mean(np.abs(e['ret'])):.3f} |")
    (REPO / "runs" / "c11").mkdir(parents=True, exist_ok=True)
    (REPO / "runs" / "c11" / "calibrate.json").write_text(json.dumps(t, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--sessions", nargs="+", required=True)
        if name == "run":
            p.add_argument("--shard", default="0/1")
    args = parser.parse_args()
    if args.cmd == "run":
        cmd_run(args.sessions, args.shard)
    else:
        cmd_report(args.sessions)


if __name__ == "__main__":
    main()
