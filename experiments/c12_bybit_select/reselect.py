"""C12: choose OBI's parameters on Bybit itself, then test them a year forward.

C10 ran the C8 configurations, chosen on Binance, unchanged on Bybit, and
only one of six made money. That mixes two questions: did the edge not
survive *the venue's* differences (the floors and kappa were tuned to
Binance's books), or did it not survive *time*? Here every parameter is
chosen again on Bybit, on the same 24 training dates C8 used (2025-08-26 to
09-18), and tested on what C10 tested: the study month's held-out week
(09-19 to 09-25) and one session a week from 2025-09-26 to 2026-10-02.

    grid    OBI, half-spread floor 4, 6, 8, 12, 16, 24 bp x markout kappa
            0, 1, 2, 4 (skew at the preset), every point with C10's stop
            (loss limit 50 bp that exits at the touch)
    pick    the best chained return on the training dates among points
            that trade (`c8.choose`)
    report  the pick on the held-out week and the forward year, with C8's
            stress set (fees, latency, queue, size, the parameter
            neighbourhood), fills beside returns, and without 2025-10-10

    python experiments/c12_bybit_select/reselect.py run --phase train --shard 0/12
    python experiments/c12_bybit_select/reselect.py run --phase test --shard 0/12
    python experiments/c12_bybit_select/reselect.py status
    python experiments/c12_bybit_select/reselect.py report
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "experiments" / "c10_forward")]

import forward  # noqa: E402  (puts c8, c6 and c1 on the path)
import bybit  # noqa: E402
import c8  # noqa: E402

CONTRACTS = ("XRPPERP", "XRPUSDT", "SUIPERP", "SUIUSDT", "AVAXUSDT")
FLOORS_BP = (4.0, 6.0, 8.0, 12.0, 16.0, 24.0)
KAPPAS = c8.KAPPAS
STOP = forward.STOPS[forward.STOP_PICK]
TEST_DATES = forward.CONTROL + forward.FORWARD


def grid(contract: str) -> list[dict]:
    return [c8.config("obi", bybit.name(contract), {"half_spread_bp": f, "markout_kappa": k, **STOP})
            for f in FLOORS_BP for k in KAPPAS]


def pick(contract: str) -> dict | None:
    return c8.choose(grid(contract), c8.TRAIN)


def test_configs(contract: str) -> list[dict]:
    p = pick(contract)
    return [] if p is None else [p, *c8.stress_configs(p).values()]


def units(phase: str) -> list[tuple[str, str, list[dict]]]:
    out = []
    only = os.environ.get("C12_ONLY")
    keep = {tuple(u.split(":")) for u in only.split(",")} if only else None
    for contract in CONTRACTS:
        if contract not in os.environ.get("C12_CONTRACTS", contract).split(","):
            continue
        dates, cfgs = (c8.TRAIN, grid(contract)) if phase == "train" else (TEST_DATES, test_configs(contract))
        for date in dates:
            if keep is None or (contract, date) in keep:
                out.append((contract, date, cfgs))
    return out


def run_unit(contract: str, date: str, cfgs: list[dict]) -> str:
    todo = [c for c in cfgs if c8.result(c, date) is None]
    if not todo:
        return "cached"
    import assets
    import config

    symbol = bybit.name(contract)
    path = bybit.convert(contract, date)
    assets.ensure_registered(symbol, date)
    for c in todo:
        c8.run_one(c, date)
    path.unlink(missing_ok=True)
    config.use_symbol(symbol)
    for f in config.LATENCY.glob(f"{symbol}_{date}*.npz"):
        f.unlink()
    gc.collect()
    return f"{len(todo)} runs"


def cmd_run(phase: str, shard: str) -> None:
    index, count = (int(x) for x in shard.split("/"))
    todo = units(phase)[index::count]
    deadline = float(os.environ.get("C8_DEADLINE", "inf"))
    print(f"### C12 {phase} shard {shard}: {len(todo)} (contract, session) units", flush=True)
    for i, (contract, date, cfgs) in enumerate(todo):
        if time.time() > deadline:
            print("### deadline reached", flush=True)
            break
        t0 = time.time()
        try:
            status = run_unit(contract, date, cfgs)
        except Exception as exc:  # one broken day must not stop a shard
            status = f"FAILED {exc!r}"
        print(f"{i + 1}/{len(todo)} {contract} {date} {status} {time.time() - t0:.0f}s", flush=True)
    print(f"### SHARD DONE {phase} {shard}", flush=True)


def cmd_status() -> None:
    for phase in ("train", "test"):
        us = units(phase)
        done = sum(bool(cfgs) and all(c8.result(c, d) is not None for c in cfgs) for _, d, cfgs in us)
        print(f"{phase}: {done}/{len(us)} units done")


def summary(cfg: dict, dates: list[str]) -> dict | None:
    rows = c8.results(cfg, dates)
    if rows is None:
        return None
    r = np.array([x["Return"] for x in rows])
    return {"chained": float(np.prod(1 + r) - 1) * 100, "up": int((r > 0).sum()), "n": len(r),
            "fills": float(np.median([x["DailyNumberOfTrades"] for x in rows]))}


def describe(cfg: dict) -> str:
    s = cfg["sets"]
    return f"floor {s['half_spread_bp']:g} bp, κ {s['markout_kappa']:g}"


def cmd_report() -> None:
    fw_ex = [d for d in forward.FORWARD if d != forward.CRASH]
    print("| contract | pick (training) | training | held-out week | forward, all | forward without 10-10 "
          "| fills/day | sessions up |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for contract in CONTRACTS:
        p = pick(contract)
        if p is None:
            print(f"| {contract} | (training incomplete) | | | | | | |")
            continue
        tr, ct = summary(p, c8.TRAIN), summary(p, forward.CONTROL)
        fa, fe = summary(p, forward.FORWARD), summary(p, fw_ex)
        f = lambda x: "–" if x is None else f"{x['chained']:+.2f}"  # noqa: E731
        print(f"| {contract} | {describe(p)} | {f(tr)} | {f(ct)} | {f(fa)} | {f(fe)} | "
              + ("– | – |" if fe is None else f"{fe['fills']:.0f} | {fe['up']}/{fe['n']} |"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--phase", required=True, choices=("train", "test"))
    r.add_argument("--shard", default="0/1")
    sub.add_parser("status")
    sub.add_parser("report")
    args = parser.parse_args()
    if args.cmd == "run":
        cmd_run(args.phase, args.shard)
    elif args.cmd == "report":
        cmd_report()
    else:
        cmd_status()


if __name__ == "__main__":
    main()
