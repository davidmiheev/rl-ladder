"""C8 -- risk controls, denoising and new quoting ideas, on a month split.

Everything here is one cache of **(configuration, session)** results, so a
configuration is replayed once per session however many analyses read it:
the train/held-out split, the walk-forward selection and the stress tests all
come from the same files.

    runs/c8/days/<key>/config.json      the configuration
    runs/c8/days/<key>/<date>.json      its statistics on one session

The split uses the whole month: 24 training sessions (2025-08-26 to 09-18)
and 7 held-out (09-19 to 09-25). Every choice is made on training sessions.

Stages, each on top of the previous stage's choice per (strategy, asset):

    A  base grid        floor x markout kappa x skew (C4's grid, kappa 1 added)
    B  risk controls    circuit breaker (volatility or one-sided flow; both
                        sides or one) and loss limits (daily stop, trailing)
    C  denoising, ideas signal smoothing, alpha clipping, markout half-life,
                        momentum, per-side markout, requote hysteresis,
                        toxicity-scaled size, inventory flattening,
                        time-of-day floors
    D  combination      greedy forward selection over the best of each family
    E  stress           maker fee without the rebate, latency, queue model,
                        order size, parameter neighbourhood
    breadth             the seven other assets on the same month split:
                        floor x markout kappa per strategy (skew at preset)

    python experiments/c8_risk_controls/c8.py run --stage A --shard 0/3
    python experiments/c8_risk_controls/c8.py status
    python experiments/c8_risk_controls/c8.py report
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments" / "c6_strategy_zoo"))
sys.path.insert(0, str(REPO / "experiments" / "c4_ppo_grpo"))

import assets  # noqa: E402
import grid_search  # noqa: E402
import strategies  # noqa: E402

CACHE = REPO / "runs" / "c8" / "days"


def _days(first: str, last: str) -> list[str]:
    start, end = dt.date.fromisoformat(first), dt.date.fromisoformat(last)
    return [str(start + dt.timedelta(days=i)) for i in range((end - start).days + 1)]


#: The month split. Training ends on the Thursday before the last full week;
#: the held-out week keeps a weekend and the 09-22 crash.
TRAIN = _days("2025-08-26", "2025-09-18")
TEST = _days("2025-09-19", "2025-09-25")
DAYS = TRAIN + TEST
CRASH = "2025-09-22"

#: `C8_SYMBOLS=xrpusdc` restricts every stage to one asset, so two machines
#: can each run the whole pipeline for their own assets: an asset's choices
#: only ever read that asset's results.
PAIRS = tuple(p for p in grid_search.PAIRS
              if p[1] in os.environ.get("C8_SYMBOLS", p[1]).split(","))
FLOOR_MULTS = (1.0, 1.5, 2.0)
KAPPAS = (0.0, 1.0, 2.0, 4.0)
SKEW_MULTS = (1.0, 2.0)
MIN_FILLS = grid_search.MIN_FILLS

#: The breadth tier: every other asset C6 measured, on the full month split.
BREADTH = ("xrpusdt", "suiusdc", "suiusdt", "enausdc", "enausdt", "avaxusdc", "bnbusdc")
BREADTH_PAIRS = tuple(
    (strategy, symbol)
    for symbol in BREADTH if symbol in os.environ.get("C8_SYMBOLS", symbol).split(",")
    for strategy in ("glft", "hfgrid", "obi")
    + (("apt", "basis") if symbol in ("suiusdc", "enausdc") else ())
)
#: C7 measured the sigma estimator's variance bias on ETHUSDC and XRPUSDC
#: only (training days, x1.45 and x1.44); the breadth assets get their mean,
#: transferred rather than measured, and labelled so.
TRANSFER_VOL_SCALE = 1.445

#: Stress variants: what each changes, applied to a chosen configuration.
STRESS = {
    "fee 0": {"maker_fee": 0.0},
    "fee +1bp": {"maker_fee": 0.0001},
    "fee +2bp": {"maker_fee": 0.0002},
    "latency x2": {"latency_mul": 2.0},
    "latency x4": {"latency_mul": 4.0},
    "queue risk-averse": {"queue_model": "risk-adverse"},
    "queue log": {"queue_model": "log"},
    "size x2": {"order_qty_dollar": 10_000.0},
}


# ------------------------------------------------------------------ configs

def config(strategy: str, symbol: str, sets: dict) -> dict:
    return {"strategy": strategy, "symbol": symbol, "sets": dict(sorted(sets.items()))}


def key(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:14]


def result(cfg: dict, date: str) -> dict | None:
    path = CACHE / key(cfg) / f"{date}.json"
    return json.loads(path.read_text()) if path.exists() else None


def results(cfg: dict, days: list[str]) -> list[dict] | None:
    out = [result(cfg, d) for d in days]
    return None if any(r is None for r in out) else out


def score(per_day: list[dict]) -> dict:
    r = np.array([d["Return"] for d in per_day])
    fills = np.array([d["DailyNumberOfTrades"] for d in per_day])
    return {"chained": float(np.prod(1 + r) - 1) * 100, "median_fills": float(np.median(fills)),
            "days_up": int((r > 0).sum()), "n": len(r),
            "paused": float(np.mean([d.get("paused_frac", 0.0) for d in per_day]))}


def stage_a(strategy: str, symbol: str) -> list[dict]:
    return [config(f"{strategy}-wide", symbol, grid_search.point_sets(strategy, symbol, f, k, s))
            for f, k, s in itertools.product(FLOOR_MULTS, KAPPAS, SKEW_MULTS)]


def breadth_grid(strategy: str, symbol: str) -> list[dict]:
    assets.ensure_registered(symbol)
    kwargs = {"beta": 1.0} if strategy == "apt" else {}
    base = strategies.describe(strategies.build(f"{strategy}-wide", symbol, **kwargs))
    out = []
    for f, k in itertools.product(FLOOR_MULTS, KAPPAS):
        sets = {"half_spread_bp": base["half_spread_bp"] * f, "markout_kappa": k}
        if grid_search.sigma_users(strategy):
            sets["vol_scale"] = TRANSFER_VOL_SCALE
        out.append(config(f"{strategy}-wide", symbol, sets))
    return out


def choose(cfgs: list[dict], days: list[str] = TRAIN) -> dict | None:
    """Best chained return on `days` among configurations that trade; None until all are in."""
    scored = []
    for cfg in cfgs:
        per_day = results(cfg, days)
        if per_day is None:
            return None
        scored.append((cfg, score(per_day)))
    eligible = [x for x in scored if x[1]["median_fills"] >= MIN_FILLS] or scored
    return max(eligible, key=lambda x: x[1]["chained"])[0]


#: Stage B and C families: name -> list of parameter overrides.
def families(strategy: str) -> dict[str, list[dict]]:
    breaker = [
        {"cb_mode": m, trig: v, "cb_fast_hl": 100.0, "cb_cooldown": c}
        for m in (1.0, 2.0) for trig, v in (("cb_vol", 2.5), ("cb_vol", 4.0),
                                           ("cb_flow", 0.5), ("cb_flow", 0.7))
        for c in (300.0, 3000.0)
    ]
    loss = ([{"loss_limit_bp": v, "loss_mode": 1.0} for v in (25.0, 50.0, 100.0)]
            + [{"loss_limit_bp": v, "loss_mode": 2.0, "loss_cooldown": c}
               for v in (15.0, 30.0, 60.0) for c in (6000.0, 36000.0)])
    fams = {
        "breaker": breaker,
        "loss limit": loss,
        "signal smoothing": [{"signal_hl": h} for h in (10.0, 50.0, 200.0)],
        "markout half-life": [{"markout_halflife": h} for h in (5.0, 50.0)],
        "momentum": [{"mom_coef": c, "mom_hl": h} for c in (0.5, 1.0, 2.0) for h in (10.0, 50.0)],
        "per-side markout": [{"markout_sided": 1.0}],
        "requote hysteresis": [{"requote_steps": s} for s in (1.0, 2.0)],
        "toxicity sizing": [{"tox_size": v} for v in (0.5, 1.0)],
        "inventory flatten": [{"inv_flatten": v} for v in (0.5, 0.8)],
        "time of day": [{f"tod{b}": 2.0} for b in range(6)],
    }
    if strategy == "obi":
        fams["alpha clipping"] = [{"alpha_clip": c} for c in (1.5, 2.5)]
    return fams


def on_top(cfg: dict, extra: dict) -> dict:
    return config(cfg["strategy"], cfg["symbol"], {**cfg["sets"], **extra})


def choice_a(strategy: str, symbol: str) -> dict | None:
    return choose(stage_a(strategy, symbol))


def family_configs(strategy: str, symbol: str) -> dict[str, list[dict]] | None:
    base = choice_a(strategy, symbol)
    if base is None:
        return None
    return {name: [on_top(base, x) for x in variants]
            for name, variants in families(strategy).items()}


def family_best(strategy: str, symbol: str) -> dict[str, tuple[dict, float]] | None:
    """Each family's best variant on training days, and its gain over the stage-A choice."""
    base = choice_a(strategy, symbol)
    fams = family_configs(strategy, symbol)
    if fams is None:
        return None
    base_score = score(results(base, TRAIN))["chained"]
    best = {}
    for name, cfgs in fams.items():
        pick = choose(cfgs)
        if pick is None:
            return None
        best[name] = (pick, score(results(pick, TRAIN))["chained"] - base_score)
    # The time-of-day family is used as a schedule: widen every four-hour
    # block whose own widening helped on training days.
    helped = {}
    for pick in fams["time of day"]:
        extra = {k: v for k, v in pick["sets"].items() if k.startswith("tod")}
        if score(results(pick, TRAIN))["chained"] > base_score:
            helped.update(extra)
    if helped:
        best["time of day"] = (on_top(base, helped), None)
    else:
        del best["time of day"]
    return best


def greedy(strategy: str, symbol: str) -> tuple[dict | None, list[dict], list[str]]:
    """Forward selection over the families' best variants, on training days.

    Returns (current choice, pending configurations, families added). A round
    tries adding each unused family's best variant to the current choice; the
    best improvement is kept; it stops when nothing improves.
    """
    best = family_best(strategy, symbol)
    if best is None:
        return None, [], []
    base = current = choice_a(strategy, symbol)
    added: list[str] = []
    while True:
        cur_score = score(results(current, TRAIN))["chained"]
        trials = []
        for name, (pick, _) in best.items():
            if name in added:
                continue
            # Only what the family changed relative to the stage-A choice,
            # so a trial never undoes an earlier addition.
            extra = {k: v for k, v in pick["sets"].items() if base["sets"].get(k) != v}
            trials.append((name, on_top(current, extra)))
        pending = [c for _, c in trials if results(c, TRAIN) is None]
        if pending:
            return current, pending, added
        scored = [(name, c, score(results(c, TRAIN))["chained"]) for name, c in trials]
        scored = [x for x in scored
                  if score(results(x[1], TRAIN))["median_fills"] >= MIN_FILLS]
        if not scored:
            return current, [], added
        name, c, value = max(scored, key=lambda x: x[2])
        if value <= cur_score:
            return current, [], added
        current, added = c, added + [name]


def stress_configs(cfg: dict) -> dict[str, dict]:
    out = {name: on_top(cfg, extra) for name, extra in STRESS.items()}
    sets = cfg["sets"]
    out["floor x0.8"] = on_top(cfg, {"half_spread_bp": sets["half_spread_bp"] * 0.8})
    out["floor x1.25"] = on_top(cfg, {"half_spread_bp": sets["half_spread_bp"] * 1.25})
    if sets.get("markout_kappa", 0.0) > 0:
        out["kappa x0.5"] = on_top(cfg, {"markout_kappa": sets["markout_kappa"] * 0.5})
        out["kappa x2"] = on_top(cfg, {"markout_kappa": sets["markout_kappa"] * 2})
    return out


def final_choices(strategy: str, symbol: str) -> dict[str, dict]:
    out = {}
    a = choice_a(strategy, symbol)
    if a is not None:
        out["A"] = a
    current, pending, _ = greedy(strategy, symbol)
    if current is not None and not pending:
        out["D"] = current
    return out


# ------------------------------------------------------------------- stages

def jobs_for(stage: str) -> list[tuple[dict, list[str]]]:
    jobs = []
    if stage == "breadth":
        return [(c, DAYS) for strategy, symbol in BREADTH_PAIRS
                for c in breadth_grid(strategy, symbol)]
    for strategy, symbol in PAIRS:
        if stage == "A":
            jobs += [(c, DAYS) for c in stage_a(strategy, symbol)]
        elif stage in ("B", "C"):
            fams = family_configs(strategy, symbol)
            if fams is None:
                continue
            names = ("breaker", "loss limit") if stage == "B" else \
                [n for n in fams if n not in ("breaker", "loss limit")]
            jobs += [(c, DAYS) for n in names for c in fams[n]]
        elif stage == "D":
            _, pending, _ = greedy(strategy, symbol)
            jobs += [(c, TRAIN) for c in pending]
            best = family_best(strategy, symbol)
            if best and "time of day" in best:
                jobs += [(best["time of day"][0], DAYS)]
        elif stage == "D-test":
            jobs += [(c, DAYS) for c in final_choices(strategy, symbol).values()]
        elif stage == "E":
            for cfg in final_choices(strategy, symbol).values():
                jobs += [(c, DAYS) for c in stress_configs(cfg).values()]
    return jobs


def run_one(cfg: dict, date: str) -> None:
    import run_baseline
    import strategies

    overrides = {"beta": 1.0, "queue_model": "power2", **cfg["sets"]}
    stats, _, out = run_baseline.run_day(cfg["strategy"], cfg["symbol"], date, overrides)
    stats.update(run_baseline.quoter_stats(out, strategies.INTERVAL_NS,
                                           strategies.MAX_POSITION_DOLLAR))
    stats["date"] = date
    folder = CACHE / key(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(cfg, indent=1))
    tmp = folder / f"{date}.json.tmp"
    tmp.write_text(json.dumps(stats))
    tmp.rename(folder / f"{date}.json")
    del out
    gc.collect()


def pending_units(stage: str) -> list[tuple[dict, str]]:
    seen, units = set(), []
    for cfg, days in jobs_for(stage):
        for date in days:
            unit = (key(cfg), date)
            if unit in seen or result(cfg, date) is not None:
                continue
            seen.add(unit)
            units.append((cfg, date))
    # Group by session so a worker's page cache keeps the day it just read.
    units.sort(key=lambda u: (u[0]["symbol"], u[1], key(u[0])))
    return units


def cmd_run(stage: str, shard: str, units_file: str | None = None) -> None:
    """Replay the stage's pending sessions (or an exported list), one shard of them.

    `C8_REVERSE=1` walks the list from the end, so a second host can take the
    same list from the other side; every unit is re-checked against the cache
    just before it runs, so results merged in from elsewhere are never redone.
    """
    index, count = (int(x) for x in shard.split("/"))
    if units_file:
        units = [(cfg, date) for cfg, date in json.loads(Path(units_file).read_text())]
    else:
        units = pending_units(stage)
    units = units[index::count]
    if os.environ.get("C8_REVERSE") == "1":
        units = units[::-1]
    print(f"### stage {stage} shard {shard}: {len(units)} session replays", flush=True)
    # `C8_DEADLINE` (epoch seconds): take no new session after it, so a
    # time-limited host (a Kaggle session) finishes cleanly and keeps its cache.
    deadline = float(os.environ.get("C8_DEADLINE", "inf"))
    for i, (cfg, date) in enumerate(units):
        if time.time() > deadline:
            print("### deadline reached", flush=True)
            break
        if result(cfg, date) is not None:
            continue
        start = time.time()
        try:
            run_one(cfg, date)
            status = "ok"
        except Exception as exc:  # one broken unit must not stop a shard
            status = f"FAILED {exc!r}"
        print(f"{i + 1}/{len(units)} {cfg['strategy']} {cfg['symbol']} {date} {key(cfg)} "
              f"{status} {time.time() - start:.0f}s", flush=True)
    print(f"### SHARD DONE {stage} {shard}", flush=True)


def cmd_export(stages: list[str], out: str) -> None:
    """Every pending (configuration, session) of these stages, for another host."""
    units, seen = [], set()
    for stage in stages:
        for cfg, date in pending_units(stage):
            if (key(cfg), date) not in seen:
                seen.add((key(cfg), date))
                units.append([cfg, date])
    Path(out).write_text(json.dumps(units))
    print(f"{len(units)} units -> {out}")


def cmd_status() -> None:
    for stage in ("A", "B", "C", "D", "D-test", "E", "breadth"):
        print(f"{stage:7s} pending {len(pending_units(stage))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--stage", required=True,
                     choices=["A", "B", "C", "D", "D-test", "E", "breadth"])
    run.add_argument("--shard", default="0/1")
    run.add_argument("--units", default=None, help="a JSON list exported by `export`")
    export = sub.add_parser("export")
    export.add_argument("--stage", nargs="+", required=True)
    export.add_argument("--out", required=True)
    sub.add_parser("status")
    args = parser.parse_args()
    if args.cmd == "run":
        cmd_run(args.stage, args.shard, args.units)
    elif args.cmd == "export":
        cmd_export(args.stage, args.out)
    else:
        cmd_status()


if __name__ == "__main__":
    main()
