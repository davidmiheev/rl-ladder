"""C10 lead-lag through the backtest: do the outside-data forecasts pay?

`leadlag.py` measures forecasts (R^2, AUC). Whether a forecast is worth
anything is decided here, the way C9 tested its own: quoting with it in the
full backtest. The C8 OBI configurations of the three assets with outside
data (ETHUSDC, XRPUSDC, SUIUSDC; parameters as C8 chose them on training
sessions) quote with each forecast in C9's ways:

    shift   the 1 s forecast added to the fair price (`fair_mode` 1)
    pull    the side a forecast says is about to be run over is pulled for
            the step, when the forecast is in its top 5% / 1% / 0.3% on the
            training sessions (`fair_mode` 4)

each with the outside data (`r10`, `cls`) and, for the comparison that
isolates them, with the own book alone (`r10b`, `clsb`) -- the same model
and settings, fitted on the same sessions.

Selection follows CLAUDE.md: the best of the configuration as C8 chose it
and every variant of a family, on training sessions; reported held out;
stress-tested (fees, latency, queue, the parameter neighbourhood) and, for
outside data, with the forecast rebuilt from data arriving +100 and +250 ms
later (models refitted at that delay). The forecast models are fitted on the
training sessions, so a variant's training-session result is in sample for
the model -- the held-out row is the one to read.

    python experiments/c10_forward/llbacktest.py run --symbol ethusdc --phase main --shard 0/4
    python experiments/c10_forward/llbacktest.py run --symbol ethusdc --phase stress --shard 0/4
    python experiments/c10_forward/llbacktest.py report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "experiments" / "c9_fair_price"),
                str(REPO / "experiments" / "c8_risk_controls"),
                str(REPO / "experiments" / "c6_strategy_zoo"),
                str(REPO / "experiments" / "c7_signals"),
                str(REPO / "experiments" / "c1_glft_baseline")]

import c8  # noqa: E402

#: C8's training-session choices for OBI (the floor, the markout kappa, the skew).
BASE = {
    "ethusdc": {"half_spread_bp": 6.0, "markout_kappa": 4.0},
    "xrpusdc": {"half_spread_bp": 12.0, "markout_kappa": 2.0, "skew_bp": 29.6},
    "suiusdc": {"half_spread_bp": 8.0, "markout_kappa": 4.0},
}
DELAYS_STRESS = (100, 250)


def base(symbol: str) -> dict:
    return c8.config("obi", symbol, BASE[symbol])


def family(outside: bool) -> dict[str, dict]:
    r, c = ("r10", "cls") if outside else ("r10b", "clsb")
    return {
        "shift by the 1 s forecast": {"fair_src": f"ll:{r}:0", "fair_mode": 1.0, "fair_coef": 1.0},
        "pull, top 5% of 1 s forecasts": {"fair_src": f"gate:0.95:ll:{r}:0", "fair_mode": 4.0,
                                          "fair_coef": 1.0},
        "pull, top 1% of 1 s forecasts": {"fair_src": f"gate:0.99:ll:{r}:0", "fair_mode": 4.0,
                                          "fair_coef": 1.0},
        "pull, top 1% of large-move scores": {"fair_src": f"gate:0.99:ll:{c}:0", "fair_mode": 4.0,
                                              "fair_coef": 1.0},
        "pull, top 0.3% of large-move scores": {"fair_src": f"gate:0.997:ll:{c}:0",
                                                "fair_mode": 4.0, "fair_coef": 1.0},
    }


FAMILIES = {"own book": False, "own book + outside": True}


def variants(symbol: str, outside: bool) -> dict[str, dict]:
    return {k: c8.on_top(base(symbol), v) for k, v in family(outside).items()}


def choice(symbol: str, outside: bool) -> dict | None:
    """The training-session best of the base and the family (None until all are in)."""
    return c8.choose([base(symbol), *variants(symbol, outside).values()])


def delayed(cfg: dict, d_ms: int) -> dict:
    """The same configuration with the forecast built from outside data `d_ms` later."""
    src = cfg["sets"]["fair_src"]
    assert src.endswith(":0") and (":r10:" in src or ":cls:" in src), src
    return c8.on_top(cfg, {"fair_src": f"{src[:-2]}:{d_ms}"})


def stress_set(cfg: dict, outside: bool) -> dict[str, dict]:
    out = dict(c8.stress_configs(cfg))
    if outside and "fair_src" in cfg["sets"]:
        for d in DELAYS_STRESS:
            out[f"outside data +{d} ms"] = delayed(cfg, d)
    return out


def jobs(symbol: str, phase: str) -> list[tuple[dict, list[str]]]:
    if phase == "main":
        cfgs = [base(symbol)] + [c for o in FAMILIES.values() for c in variants(symbol, o).values()]
        return [(c, c8.DAYS) for c in cfgs]
    out = []
    for outside in FAMILIES.values():
        pick = choice(symbol, outside)
        if pick is None:
            continue
        out += [(c, c8.TEST) for c in stress_set(pick, outside).values()]
    return out


def pending(symbol: str, phase: str) -> list[tuple[dict, str]]:
    seen, units = set(), []
    for cfg, days in jobs(symbol, phase):
        for date in days:
            u = (c8.key(cfg), date)
            if u not in seen and c8.result(cfg, date) is None:
                seen.add(u)
                units.append((cfg, date))
    # Session-major, so a worker reuses a downloaded L2 day across configurations.
    units.sort(key=lambda u: (u[1], c8.key(u[0])))
    return units


def cmd_run(symbol: str, phase: str, shard: str) -> None:
    import c9

    index, count = (int(x) for x in shard.split("/"))
    units = pending(symbol, phase)
    dates = sorted({d for _, d in units})[index::count]
    units = [u for u in units if u[1] in dates]
    print(f"### C10 lead-lag backtest {symbol} {phase} {shard}: {len(units)} replays", flush=True)
    import config
    import snapshots

    current = None
    for i, (cfg, date) in enumerate(units):
        # One L2 day on disk at a time: fetched for its first replay, deleted
        # after its last (the units are session-major).
        if date != current:
            if current is not None:
                (config.DATA / f"{symbol}_{current}.npz").unlink(missing_ok=True)
            current = date
            if not (config.DATA / f"{symbol}_{date}.npz").exists():
                snapshots.download(symbol, date)
        t0 = time.time()
        try:
            c9.run_one(cfg, date)
            status = "ok"
        except Exception as exc:  # one broken unit must not stop a shard
            status = f"FAILED {exc!r}"
        print(f"{i + 1}/{len(units)} {date} {c8.key(cfg)} {cfg['sets'].get('fair_src', 'base')} "
              f"{status} {time.time() - t0:.0f}s", flush=True)
    print(f"### SHARD DONE {symbol} {phase} {shard}", flush=True)


def summary(cfg: dict, days: list[str]) -> dict | None:
    rows = c8.results(cfg, days)
    if rows is None:
        return None
    r = np.array([x["Return"] for x in rows])
    return {"chained": float(np.prod(1 + r) - 1) * 100, "up": int((r > 0).sum()), "n": len(r),
            "fills": float(np.median([x["DailyNumberOfTrades"] for x in rows]))}


def table() -> dict:
    """Every number the writeup reports, as one dict (also printed by `report`)."""
    out = {}
    for symbol in BASE:
        rows = {"as C8 chose it": base(symbol)}
        for fam, outside in FAMILIES.items():
            for label, cfg in variants(symbol, outside).items():
                rows[f"{fam}: {label}"] = cfg
        sym = {"variants": {}, "choices": {}}
        for label, cfg in rows.items():
            sym["variants"][label] = {"train": summary(cfg, c8.TRAIN), "test": summary(cfg, c8.TEST)}
        for fam, outside in FAMILIES.items():
            pick = choice(symbol, outside)
            if pick is None:
                continue
            label = next(k for k, v in rows.items() if c8.key(v) == c8.key(pick))
            stress = {name: summary(c, c8.TEST) for name, c in stress_set(pick, outside).items()}
            sym["choices"][fam] = {"pick": label, "test": summary(pick, c8.TEST), "stress": stress}
        out[symbol] = sym
    return out


def _fmt(s: dict | None, key: str = "chained", spec: str = "+.2f") -> str:
    return "–" if s is None else format(s[key], spec)


def cmd_report() -> None:
    t = table()
    for symbol, sym in t.items():
        print(f"\n## {symbol} OBI\n")
        print("| variant | training (chained %) | held out (chained %) | held-out fills/day |")
        print("| --- | --- | --- | --- |")
        for label, v in sym["variants"].items():
            print(f"| {label} | {_fmt(v['train'])} | {_fmt(v['test'])} | "
                  f"{_fmt(v['test'], 'fills', '.0f')} |")
        for fam, ch in sym["choices"].items():
            print(f"\n{fam}: training pick **{ch['pick']}**, held out {_fmt(ch['test'])}%")
            for name, s in ch["stress"].items():
                print(f"  {name}: {_fmt(s)}% ({_fmt(s, 'fills', '.0f')} fills)")
    out = REPO / "runs" / "c10" / "llbacktest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(t, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--symbol", required=True, choices=sorted(BASE))
    r.add_argument("--phase", required=True, choices=("main", "stress"))
    r.add_argument("--shard", default="0/1")
    s = sub.add_parser("status")
    s.add_argument("--symbol", required=True, choices=sorted(BASE))
    sub.add_parser("report")
    args = parser.parse_args()
    if args.cmd == "run":
        cmd_run(args.symbol, args.phase, args.shard)
    elif args.cmd == "status":
        for phase in ("main", "stress"):
            print(phase, len(pending(args.symbol, phase)), "replays pending")
    else:
        cmd_report()


if __name__ == "__main__":
    main()
