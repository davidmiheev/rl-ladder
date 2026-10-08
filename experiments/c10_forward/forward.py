"""C10 forward test: the frozen C8 choices, unchanged, on Bybit sessions.

Every Part C conclusion so far rests on seven held-out Binance sessions. This
runs the configurations that made money there, with every parameter frozen
as C8 reported it (nothing is re-chosen), on Bybit's free order books:

* **forward** -- one session a week from 2025-09-26 (the day after the study
  month) to the latest Bybit day, fixed in advance as every 7th day so no
  session is picked;
* **venue control** -- the study month's held-out week (2025-09-19 to 09-25)
  on Bybit, which separates "the edge does not survive time" from "the edge
  does not survive the venue".

Each Binance configuration runs on the Bybit contract that matches it:

| Binance (C8)            | Bybit                                  |
| XRPUSDC OBI (A)         | XRPPERP (USDC-settled)                 |
| XRPUSDT OBI             | XRPUSDT                                |
| SUIUSDC OBI             | SUIPERP (USDC-settled), and SUIUSDT    |
| SUIUSDT OBI             | SUIUSDT                                |
| AVAXUSDC OBI            | AVAXUSDT (Bybit has no USDC AVAX perp) |

Fees: the Binance results assumed a 0.5 bp maker rebate. Bybit's standard
(VIP 0) maker fee is +2 bp, and its market-maker programme pays rebates, so
every session runs at -0.5, 0, +1 and +2 bp, plus 4x order latency and the
risk-averse queue model. Fills are reported beside returns.

    python experiments/c10_forward/forward.py run --shard 0/3
    python experiments/c10_forward/forward.py status
    python experiments/c10_forward/forward.py report
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "experiments" / "c8_risk_controls"),
                str(REPO / "experiments" / "c6_strategy_zoo"),
                str(REPO / "experiments" / "c1_glft_baseline")]

import bybit  # noqa: E402
import c8  # noqa: E402

#: label -> (Bybit contract, the frozen C8 parameter overrides)
FROZEN = {
    "XRPUSDC OBI → XRPPERP": ("XRPPERP", {"half_spread_bp": 12.0, "markout_kappa": 2.0,
                                           "skew_bp": 29.6}),
    "XRPUSDT OBI → XRPUSDT": ("XRPUSDT", {"half_spread_bp": 8.0, "markout_kappa": 2.0}),
    "SUIUSDC OBI → SUIPERP": ("SUIPERP", {"half_spread_bp": 8.0, "markout_kappa": 4.0}),
    "SUIUSDC OBI → SUIUSDT": ("SUIUSDT", {"half_spread_bp": 8.0, "markout_kappa": 4.0}),
    "SUIUSDT OBI → SUIUSDT": ("SUIUSDT", {"half_spread_bp": 16.0, "markout_kappa": 4.0}),
    "AVAXUSDC OBI → AVAXUSDT": ("AVAXUSDT", {"half_spread_bp": 16.0, "markout_kappa": 0.0}),
}
CONTRACTS = sorted({c for c, _ in FROZEN.values()})

#: Stress variants run on every session (label -> overrides).
VARIANTS = {
    "as frozen (-0.5 bp)": {},
    "fee 0": {"maker_fee": 0.0},
    "fee +1 bp": {"maker_fee": 0.0001},
    "fee +2 bp (Bybit VIP 0)": {"maker_fee": 0.0002},
    "latency x4": {"latency_mul": 4.0},
    "queue risk-averse": {"queue_model": "risk-adverse"},
}

CONTROL = c8.TEST  # 2025-09-19 .. 09-25
FORWARD_START = dt.date(2025, 9, 26)
#: The last Bybit day checked to exist when the forward set was fixed.
FORWARD_END = dt.date(2026, 10, 5)
FORWARD = [str(FORWARD_START + dt.timedelta(days=7 * k))
           for k in range((FORWARD_END - FORWARD_START).days // 7 + 1)]
DATES = CONTROL + FORWARD


def configs_for(contract: str) -> list[dict]:
    out = []
    for label, (c, sets) in FROZEN.items():
        if c != contract:
            continue
        for extra in VARIANTS.values():
            out.append(c8.config("obi", bybit.name(c), {**sets, **extra}))
    return out


def cfg(label: str, variant: str) -> dict:
    contract, sets = FROZEN[label]
    return c8.config("obi", bybit.name(contract), {**sets, **VARIANTS[variant]})


def pending(contract: str, date: str) -> list[dict]:
    return [c for c in configs_for(contract) if c8.result(c, date) is None]


def units() -> list[tuple[str, str]]:
    out = [(c, d) for d in DATES for c in CONTRACTS
           if c in os.environ.get("C10_CONTRACTS", c).split(",")]
    # `C10_ONLY=CONTRACT:DATE,...` restricts a host to named units, so a
    # host without the cache does not redo what another host finished.
    only = os.environ.get("C10_ONLY")
    if only:
        keep = {tuple(u.split(":")) for u in only.split(",")}
        out = [u for u in out if u in keep]
    return out


def run_unit(contract: str, date: str) -> str:
    todo = pending(contract, date)
    if not todo:
        return "cached"
    import assets
    import config

    symbol = bybit.name(contract)
    path = bybit.convert(contract, date)
    assets.ensure_registered(symbol, date)
    for c in todo:
        c8.run_one(c, date)
    # Stream: nothing but the cached statistics is kept.
    path.unlink(missing_ok=True)
    config.use_symbol(symbol)
    for f in config.LATENCY.glob(f"{symbol}_{date}*.npz"):
        f.unlink()
    gc.collect()
    return f"{len(todo)} runs"


def cmd_run(shard: str) -> None:
    index, count = (int(x) for x in shard.split("/"))
    todo = units()[index::count]
    deadline = float(os.environ.get("C8_DEADLINE", "inf"))
    print(f"### C10 shard {shard}: {len(todo)} (contract, session) units", flush=True)
    for i, (contract, date) in enumerate(todo):
        if time.time() > deadline:
            print("### deadline reached", flush=True)
            break
        t0 = time.time()
        try:
            status = run_unit(contract, date)
        except Exception as exc:  # one broken day must not stop a shard
            status = f"FAILED {exc!r}"
        print(f"{i + 1}/{len(todo)} {contract} {date} {status} {time.time() - t0:.0f}s", flush=True)
    print(f"### SHARD DONE {shard}", flush=True)


def cmd_status() -> None:
    done = sum(not pending(c, d) for c, d in units())
    print(f"{done}/{len(units())} (contract, session) units done")


def summary(label: str, variant: str, dates: list[str]) -> dict | None:
    rows = [r for r in (c8.result(cfg(label, variant), d) for d in dates) if r is not None]
    if not rows:
        return None
    r = np.array([x["Return"] for x in rows])
    return {"n": len(rows), "chained": float(np.prod(1 + r) - 1) * 100,
            "mean_bp": float(r.mean()) * 1e4, "up": int((r > 0).sum()),
            "fills": float(np.median([x["DailyNumberOfTrades"] for x in rows]))}


def cmd_report() -> None:
    for period, dates in (("control (held-out week)", CONTROL), ("forward (weekly)", FORWARD)):
        print(f"\n## {period}\n")
        print("| configuration | sessions | " + " | ".join(VARIANTS) + " | fills/day | sessions up |")
        print("| --- | --- |" + " --- |" * len(VARIANTS) + " --- | --- |")
        for label in FROZEN:
            cells = [summary(label, v, dates) for v in VARIANTS]
            base = cells[0]
            if base is None:
                continue
            print(f"| {label} | {base['n']} | " + " | ".join(
                "–" if c is None else f"{c['chained']:+.2f}" for c in cells)
                + f" | {base['fills']:.0f} | {base['up']}/{base['n']} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--shard", default="0/1")
    sub.add_parser("status")
    sub.add_parser("report")
    args = parser.parse_args()
    if args.cmd == "run":
        cmd_run(args.shard)
    elif args.cmd == "report":
        cmd_report()
    else:
        cmd_status()


if __name__ == "__main__":
    main()
