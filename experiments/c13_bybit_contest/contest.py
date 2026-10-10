"""C13: C8's full selection procedure on Bybit, over a year, as a heuristic contest.

C8 chose its configurations on 24 Binance sessions and checked them on 7;
C10 found the result did not travel. Bybit's free books cover a year, so the
whole procedure is rerun on Bybit with far more sessions, and every heuristic
the project has is a contestant:

    split     sessions every 4th day, in three blocks that never mix:
              train  2025-08-26 .. 2026-02-28 (plus the 2025-10-10 crash)
              valid  2026-03-02 .. 2026-06-14
              test   2026-06-18 .. 2026-10-02, locked: scored once, at the end
    screen    stage A on every 8th training day: OBI, GLFT and hfgrid over the
              spread floor x markout kappa (x skew for OBI), absolute floors
              in bp because Bybit has no calibrated per-asset floor
    families  stage B on all training days: on top of each contract's
              screen pick, every C8 family (breaker, loss limit, smoothing,
              markout half-life, momentum, per-side markout, requote
              hysteresis, toxicity sizing, inventory flatten, time of day,
              alpha clipping) and C10's (the 50 bp stop, the units cap)
    greedy    stage C: forward selection over the families' training-best
              variants, as C8 (`c8.greedy`), one round per run
    rules     stages B and C run under two scoring rules (`RULES`): C8's
              chained return over all training sessions, and C10's
              crash-guarded rule (crash drawdown <= 10%, then the best
              without the crash)
    valid     the screen's top 3, and each rule's family bests and greedy
              result, on the validation block; each rule picks the best on
              train+valid
    test      both picks and their stress sets (C8's) on the locked test block

Selection uses training (and, for the final pick, validation) sessions only;
the test block is reported, never chosen on.

    python experiments/c13_bybit_contest/contest.py run --phase screen --shard 0/12
    python experiments/c13_bybit_contest/contest.py units --phase families > units.json
    C13_UNITS=units.json python experiments/c13_bybit_contest/contest.py run --phase families --shard 0/12
    python experiments/c13_bybit_contest/contest.py status
    python experiments/c13_bybit_contest/contest.py report
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
sys.path[:0] = [str(HERE), str(REPO / "experiments" / "c10_forward")]

import forward  # noqa: E402  (puts c8, c6 and c1 on the path)
import bybit  # noqa: E402
import c8  # noqa: E402

#: The five contracts C10 forward-tested C8's choices on, then six more
#: mid-cap USDT perpetuals with a year of Bybit's free books (ENA and BNB were
#: in C8's breadth set on Binance). ETH and SOL are left out: their books are
#: several times larger than XRP's, which already strains a Kaggle session.
CONTRACTS = ("XRPPERP", "SUIPERP", "XRPUSDT", "SUIUSDT", "AVAXUSDT",
             "ENAUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "LTCUSDT")


def _every(start: str, end: str, step: int) -> list[str]:
    d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    return [str(d0 + dt.timedelta(days=k)) for k in range(0, (d1 - d0).days + 1, step)]


TRAIN = sorted(set(_every("2025-08-26", "2026-02-28", 4)) | {forward.CRASH})
SCREEN = _every("2025-08-26", "2026-02-28", 8)
VALID = _every("2026-03-02", "2026-06-14", 4)
TEST = _every("2026-06-18", "2026-10-02", 4)

#: Stage A grids: absolute floors (bp), kappa, OBI's skew multiple.
OBI_FLOORS = (4.0, 6.0, 8.0, 12.0, 16.0, 24.0)
OBI_SKEW = (1.0, 2.0)
#: OBI's preset skew (`strategies.obi`), the base C8's skew multiples scale.
OBI_PRESET_SKEW_BP = 14.8
MM_FLOORS = (4.0, 8.0, 16.0)
KAPPAS = (0.0, 2.0, 4.0)

C10_FAMILIES = {
    "stop": [forward.STOPS["stop 50 bp"], forward.STOPS["stop 100 bp"],
             forward.STOPS["trailing stop 100 bp, 30 min"]],
    "units cap": [forward.STOPS["units cap + inventory flatten 0.5"],
                  {"cap_ref": 1.0, "cap_resting": 1.0}],
}


def screen_configs(contract: str) -> list[dict]:
    sym = bybit.name(contract)
    out = []
    obi_skew = OBI_PRESET_SKEW_BP
    for f in OBI_FLOORS:
        for k in KAPPAS:
            for s in OBI_SKEW:
                out.append(c8.config("obi", sym, {"half_spread_bp": f, "markout_kappa": k,
                                                  "skew_bp": obi_skew * s}))
    for strategy in ("glft", "hfgrid"):
        for f in MM_FLOORS:
            for k in KAPPAS:
                out.append(c8.config(strategy, sym, {"half_spread_bp": f, "markout_kappa": k,
                                                     "vol_scale": c8.TRANSFER_VOL_SCALE}))
    return out


#: How training sessions are scored. "all sessions" is C8's rule: the chained
#: return over every session. The 2025-10-10 crash is one of the 48, and on
#: XRPPERP it alone decided the families' ranking (the best breaker: +46.6% on
#: the crash, -8.9% over the other 47), so "crash-guarded" is C10's stop rule:
#: a crash-session drawdown of at most CRASH_MDD, then the best chained return
#: without the crash. Both rules' picks go to validation and to the test.
RULES = ("all sessions", "crash-guarded")
CRASH_MDD = 0.10


def rule_score(rows: list[dict], days: list[str], rule: str) -> dict:
    """c8.score under `rule`, plus `ok`: whether the config may be chosen at all."""
    s = c8.score(rows)
    ok = s["median_fills"] >= c8.MIN_FILLS
    if rule == "crash-guarded":
        crash = [r["MaxDrawdown"] for d, r in zip(days, rows) if d == forward.CRASH]
        s = c8.score([r for d, r in zip(days, rows) if d != forward.CRASH])
        ok = ok and max(crash, default=0.0) <= CRASH_MDD
    return {**s, "ok": ok}


def ranked(cfgs: list[dict], days: list[str], rule: str = "all sessions") -> list[tuple[dict, dict]] | None:
    """(config, score) best first among those that may be chosen; None until all are in."""
    scored = []
    for cfg in cfgs:
        rows = c8.results(cfg, days)
        if rows is None:
            return None
        scored.append((cfg, rule_score(rows, days, rule)))
    eligible = [x for x in scored if x[1]["ok"]] or scored
    return sorted(eligible, key=lambda x: -x[1]["chained"])


def screen_pick(contract: str, n: int = 1) -> list[dict] | None:
    r = ranked(screen_configs(contract), SCREEN)
    return None if r is None else [c for c, _ in r[:n]]


def families(strategy: str) -> dict[str, list[dict]]:
    return {**c8.families(strategy), **C10_FAMILIES}


def family_configs(contract: str) -> dict[str, list[dict]] | None:
    picks = screen_pick(contract)
    if picks is None:
        return None
    base = picks[0]
    return {name: [c8.on_top(base, v) for v in vs] for name, vs in families(base["strategy"]).items()}


def family_best(contract: str, rule: str = "all sessions") -> dict[str, dict] | None:
    fams = family_configs(contract)
    if fams is None:
        return None
    best = {}
    for name, cfgs in fams.items():
        r = ranked(cfgs, TRAIN, rule)
        if r is None:
            return None
        best[name] = r[0][0]
    return best


def greedy(contract: str, rule: str = "all sessions") -> tuple[dict | None, list[dict], list[str]]:
    """C8's forward selection on TRAIN: (current, pending configs, families added)."""
    picks = screen_pick(contract)
    best = family_best(contract, rule)
    if picks is None or best is None:
        return None, [], []
    base = current = picks[0]
    added: list[str] = []
    while True:
        cur = c8.results(current, TRAIN)
        if cur is None:
            return current, [current], added
        cur_score = rule_score(cur, TRAIN, rule)["chained"]
        trials = []
        for name, pick in best.items():
            if name in added:
                continue
            extra = {k: v for k, v in pick["sets"].items() if base["sets"].get(k) != v}
            trials.append((name, c8.on_top(current, extra)))
        pending = [c for _, c in trials if c8.results(c, TRAIN) is None]
        if pending:
            return current, pending, added
        scored = []
        for name, c in trials:
            s = rule_score(c8.results(c, TRAIN), TRAIN, rule)
            if s["ok"]:
                scored.append((name, c, s["chained"]))
        if not scored:
            return current, [], added
        name, c, value = max(scored, key=lambda x: x[2])
        if value <= cur_score:
            return current, [], added
        current = c
        added.append(name)


def candidates(contract: str) -> dict[str, dict] | None:
    """What goes to validation: the screen's top 3, and under each rule each
    family's best and the greedy result (each config once, under its first name)."""
    top = screen_pick(contract, 3)
    if top is None:
        return None
    out = {f"screen #{i + 1}": c for i, c in enumerate(top)}
    for rule in RULES:
        best = family_best(contract, rule)
        cur, pending, added = greedy(contract, rule)
        if best is None or cur is None or pending:
            return None
        tag = "" if rule == RULES[0] else f" [{rule}]"
        out.update({f"family: {k}{tag}": v for k, v in best.items()})
        out[f"greedy{tag} (" + (", ".join(added) or "nothing added") + ")"] = cur
    unique: dict[str, dict] = {}
    for name, c in out.items():
        if all(c8.key(c) != c8.key(v) for v in unique.values()):
            unique[name] = c
    return unique


def final_pick(contract: str, rule: str = "all sessions") -> tuple[str, dict] | None:
    cands = candidates(contract)
    if cands is None:
        return None
    r = ranked(list(cands.values()), TRAIN + VALID, rule)
    if r is None:
        return None
    pick = r[0][0]
    return next(k for k, v in cands.items() if c8.key(v) == c8.key(pick)), pick


def units(phase: str) -> list[tuple[str, str, list[dict]]]:
    out = []
    only = os.environ.get("C13_ONLY")
    keep = {tuple(u.split(":")) for u in only.split(",")} if only else None
    for contract in CONTRACTS:
        if contract not in os.environ.get("C13_CONTRACTS", contract).split(","):
            continue
        if phase == "screen":
            dates, cfgs = SCREEN, screen_configs(contract)
        elif phase == "families":
            fams = family_configs(contract)
            dates, cfgs = TRAIN, ([] if fams is None else
                                  [c for vs in fams.values() for c in vs] + screen_pick(contract))
        elif phase == "greedy":
            pending = {c8.key(c): c for rule in RULES for c in greedy(contract, rule)[1]}
            dates, cfgs = TRAIN, list(pending.values())
        elif phase == "valid":
            cands = candidates(contract)
            dates, cfgs = TRAIN + VALID, ([] if cands is None else list(cands.values()))
        elif phase == "test":
            picks = [fp[1] for fp in (final_pick(contract, rule) for rule in RULES) if fp is not None]
            stress = {c8.key(c): c for p in picks for c in (p, *c8.stress_configs(p).values())}
            dates, cfgs = TEST, list(stress.values()) if len(picks) == len(RULES) else []
        else:
            raise ValueError(phase)
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
    # Later phases are built from earlier results; a machine without them (a
    # fresh Kaggle session) is handed the units, as dumped by `units`.
    us = json.loads(Path(os.environ["C13_UNITS"]).read_text()) if os.environ.get("C13_UNITS") else units(phase)
    todo = [u for u in us if u[2]][index::count]
    deadline = float(os.environ.get("C8_DEADLINE", "inf"))
    print(f"### C13 {phase} shard {shard}: {len(todo)} (contract, session) units", flush=True)
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
    for phase in ("screen", "families", "greedy", "valid", "test"):
        us = units(phase)
        ready = [u for u in us if u[2]]
        done = sum(all(c8.result(c, d) is not None for c in cfgs) for _, d, cfgs in ready)
        print(f"{phase}: {done}/{len(ready)} units done"
              + ("" if len(ready) == len(us) else f" ({len(us) - len(ready)} waiting on the stage before)"))


def summary(cfg: dict, dates: list[str]) -> dict | None:
    rows = c8.results(cfg, dates)
    if rows is None:
        return None
    r = np.array([x["Return"] for x in rows])
    return {"chained": float(np.prod(1 + r) - 1) * 100, "up": int((r > 0).sum()), "n": len(r),
            "fills": float(np.median([x["DailyNumberOfTrades"] for x in rows]))}


def describe(cfg: dict) -> str:
    s = cfg["sets"]
    core = f"{cfg['strategy']} floor {s['half_spread_bp']:g} bp, κ {s['markout_kappa']:g}"
    extra = {k: v for k, v in s.items() if k not in ("half_spread_bp", "markout_kappa", "skew_bp",
                                                     "vol_scale")}
    return core + ("" if not extra else " + " + ", ".join(f"{k} {v:g}" for k, v in extra.items()))


def cmd_report() -> None:
    f = lambda x: "–" if x is None else f"{x['chained']:+.2f} ({x['fills']:.0f})"  # noqa: E731
    for contract in CONTRACTS:
        print(f"\n## {contract}")
        top = screen_pick(contract, 3)
        if top is None:
            print("screen incomplete")
            continue
        for i, c in enumerate(top):
            print(f"screen #{i + 1}: {describe(c)} | screen {f(summary(c, SCREEN))} | train {f(summary(c, TRAIN))}")
        cands = candidates(contract)
        if cands is None:
            continue
        print("| candidate | config | train % (fills) | valid % (fills) |")
        print("| --- | --- | --- | --- |")
        for name, c in cands.items():
            print(f"| {name} | {describe(c)} | {f(summary(c, TRAIN))} | {f(summary(c, VALID))} |")
        for rule in RULES:
            fp = final_pick(contract, rule)
            if fp is not None:
                name, c = fp
                print(f"\npick ({rule}): {name}: {describe(c)} | test {f(summary(c, TEST))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--phase", required=True, choices=("screen", "families", "greedy", "valid", "test"))
    r.add_argument("--shard", default="0/1")
    u = sub.add_parser("units")
    u.add_argument("--phase", required=True, choices=("screen", "families", "greedy", "valid", "test"))
    sub.add_parser("status")
    sub.add_parser("report")
    args = parser.parse_args()
    if args.cmd == "run":
        cmd_run(args.phase, args.shard)
    elif args.cmd == "units":
        todo = [(k, d, [c for c in cfgs if c8.result(c, d) is None]) for k, d, cfgs in units(args.phase)]
        print(json.dumps([u for u in todo if u[2]]))
    elif args.cmd == "report":
        cmd_report()
    else:
        cmd_status()


if __name__ == "__main__":
    main()
