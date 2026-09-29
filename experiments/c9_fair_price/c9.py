"""C9 -- the fair price the spread is quoted around, and denoising on every asset.

Two questions, both on C8's month split (24 training sessions, 7 held out) and
C8's per-session cache (`runs/c8/days/<config hash>/<date>.json`), so a
configuration C8 already replayed is never replayed again:

1. **Denoising on every asset.** C8 tried three denoisers on ETHUSDC and
   XRPUSDC only. Here every denoiser runs on the strategy that makes money
   held out -- OBI with the markout rule -- on all nine assets, plus BNBUSDC
   GLFT, on top of each asset's C8 choice. Each family's best variant on
   training sessions is reported held out and stress-tested.

2. **The fair price** (stages `oracle` and `predict`, see `fair.py`): how much
   return a better centre for the quotes is worth, first with the future mid
   (an oracle), then with causal predictions of it.

    python experiments/c9_fair_price/c9.py run --stage denoise --shard 0/3
    python experiments/c9_fair_price/c9.py export --stage denoise --out units.json
    python experiments/c9_fair_price/c9.py status
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments" / "c8_risk_controls"))

import c8  # noqa: E402

TRAIN, TEST, DAYS = c8.TRAIN, c8.TEST, c8.DAYS
ALL_SYMBOLS = ("ethusdc", "xrpusdc") + c8.BREADTH

#: The configurations denoising is tried on: OBI everywhere (the only
#: strategy that makes money held out while trading, on five of nine assets)
#: and BNBUSDC GLFT (the one other positive, trading row in C8).
TARGETS = tuple(
    (strategy, symbol)
    for strategy, symbol in [("obi", s) for s in ALL_SYMBOLS] + [("glft", "bnbusdc")]
    if symbol in os.environ.get("C8_SYMBOLS", symbol).split(",")
)


#: The oracle test runs on the configurations that make money held out (OBI
#: on its five positive assets, BNBUSDC GLFT) and on the two ETH strategies
#: that lose, to see whether a better centre would rescue them.
ORACLE_TARGETS = tuple(
    (strategy, symbol)
    for strategy, symbol in (("obi", "ethusdc"), ("obi", "xrpusdc"), ("glft", "ethusdc"),
                             ("hfgrid", "ethusdc"), ("obi", "suiusdc"), ("obi", "xrpusdt"),
                             ("obi", "avaxusdc"), ("glft", "bnbusdc"))
    if symbol in os.environ.get("C8_SYMBOLS", symbol).split(",")
)


def oracle_variants(strategy: str) -> dict[str, dict]:
    """Label -> fair-price overrides. k is in 100 ms steps.

    The full set (horizons, the strategy's own signal switched off, scaled
    and noisy oracles) on OBI; the horizons alone on GLFT and hfgrid, plus
    hfgrid's micro-price signal switched off.
    """
    out = {}
    for k in (5, 10, 15, 30):
        out[f"oracle +{k / 10:g}s"] = {"fair_src": f"oracle:{k}", "fair_mode": 1.0, "fair_coef": 1.0}
    if strategy in ("obi", "hfgrid"):
        out["oracle +1s, own signal off"] = {"fair_src": "oracle:10", "fair_mode": 2.0,
                                             "fair_coef": 1.0}
    if strategy == "obi":
        for coef in (0.5, 2.0):
            out[f"oracle +1s x{coef:g}"] = {"fair_src": "oracle:10", "fair_mode": 1.0,
                                            "fair_coef": coef}
        for r2 in (0.05, 0.1, 0.2, 0.4):
            out[f"noisy +1s R2={r2:g}"] = {"fair_src": f"noisy:10:{r2:g}", "fair_mode": 1.0,
                                            "fair_coef": 1.0}
    return out


#: Forecast horizon, in 100 ms steps, the traded predictions are fitted for.
PRED_K = int(os.environ.get("C9_PRED_K", "10"))


#: The large-move check: ETH GLFT and OBI (the largest oracle gaps) and XRP OBI.
LARGE_TARGETS = tuple(t for t in ORACLE_TARGETS
                      if t in (("glft", "ethusdc"), ("obi", "ethusdc"), ("obi", "xrpusdc")))


def large_oracle_variants() -> dict[str, dict]:
    """Oracle versions of 'act only on the large moves' (diagnostic, all sessions).

    Mode 3 widens only the side the coming move would run over, by the move;
    mode 4 pulls that side. `gate:q` keeps the top (1-q) of moves.
    """
    v = {"widen endangered side by the 1 s move": {"fair_src": "oracle:10", "fair_mode": 3.0}}
    for q in (0.9, 0.99):
        v[f"widen, top {100 * (1 - q):g}% of 1 s moves"] = {
            "fair_src": f"gate:{q:g}:oracle:10", "fair_mode": 3.0}
        v[f"pull, top {100 * (1 - q):g}% of 1 s moves"] = {
            "fair_src": f"gate:{q:g}:oracle:10", "fair_mode": 4.0}
    v["pull, top 1% of 3 s moves"] = {"fair_src": "gate:0.99:oracle:30", "fair_mode": 4.0}
    return {k: {**x, "fair_coef": 1.0} for k, x in v.items()}


def large_forecast_variants() -> list[dict]:
    """The same with the boosted-tree forecast: chosen on training sessions."""
    out = []
    for k in (10, 30):
        for q in (0.95, 0.99):
            src = f"gate:{q:g}:pred:gbm:{k}"
            out.append({"fair_src": src, "fair_mode": 4.0, "fair_coef": 1.0})
            if k == 10:
                out.append({"fair_src": src, "fair_mode": 3.0, "fair_coef": 2.0})
    out.append({"fair_src": "gate:0.99:pred:gbm:10", "fair_mode": 1.0, "fair_coef": 1.0})
    return out


def large_choice(strategy: str, symbol: str) -> dict | None:
    base = base_choice(strategy, symbol)
    return None if base is None else c8.choose(
        [c8.on_top(base, v) for v in large_forecast_variants()])


#: The DAgger student's final round, and the actions its threshold and size
#: are chosen from on training sessions.
DAGGER_ROUND = int(os.environ.get("C9_DAGGER_ROUND", "2"))
DAGGER_ACTIONS = [{"dag_thr": thr, "dag_widen_bp": w}
                  for thr in (0.4, 0.5, 0.6, 0.7) for w in (0.0, 2.0, 5.0)]
#: The second teacher (the oracle large-move policy): its events are rare, so
#: its probabilities are small and so are the thresholds; its last round.
PULL_TEACHER = "pull:30:0.99"
PULL_ROUND = int(os.environ.get("C9_PULL_ROUND", "1"))
PULL_ACTIONS = [{"dag_thr": thr, "dag_widen_bp": w}
                for thr in (0.05, 0.1, 0.2, 0.3) for w in (0.0, 5.0)]


def dagger_students() -> list[tuple[str, int, list[dict]]]:
    """(tag, final round, actions) for each teacher."""
    return [("", DAGGER_ROUND, DAGGER_ACTIONS),
            ("-" + PULL_TEACHER.replace(":", ""), PULL_ROUND, PULL_ACTIONS)]


def dagger_configs(strategy: str, symbol: str, tag: str = "") -> list[dict] | None:
    import dagger

    rnd, actions = {t: (r, a) for t, r, a in dagger_students()}[tag]
    base = base_choice(strategy, symbol)
    name = strategy + tag
    if base is None or not (dagger.ROOT / f"{name}_{symbol}_r{rnd}.json").exists():
        return None
    return [c8.on_top(base, dagger.student_sets(name, symbol, rnd, a)) for a in actions]


def fair_families(strategy: str) -> dict[str, list[dict]]:
    """Fair-price candidates: name -> overrides. Chosen on training sessions.

    Mode 1 moves the quotes' centre by the series on top of the strategy's
    own signal; mode 2 replaces the signal with it (only meaningful where
    there is a signal: OBI's imbalance, hfgrid's micro-price).
    """
    modes = (1.0, 2.0) if strategy in ("obi", "hfgrid") else (1.0,)
    srcs = {
        "micro-price": "feat:micro_bp",
        "micro-price, fitted": f"pred:micro:{PRED_K}",
        "linear forecast": f"pred:lin:{PRED_K}",
        "boosted-tree forecast": f"pred:gbm:{PRED_K}",
    }
    return {name: [{"fair_src": src, "fair_mode": m, "fair_coef": c}
                   for m in modes for c in (1.0, 2.0)]
            for name, src in srcs.items()}


def fair_configs(strategy: str, symbol: str) -> dict[str, list[dict]] | None:
    base = base_choice(strategy, symbol)
    if base is None:
        return None
    return {name: [c8.on_top(base, x) for x in variants]
            for name, variants in fair_families(strategy).items()}


def fair_best(strategy: str, symbol: str) -> dict[str, dict] | None:
    fams = fair_configs(strategy, symbol)
    if fams is None:
        return None
    best = {}
    for name, cfgs in fams.items():
        pick = c8.choose(cfgs)
        if pick is None:
            return None
        best[name] = pick
    return best


def base_choice(strategy: str, symbol: str) -> dict | None:
    """C8's training-session choice: the stage-A grid, or the breadth grid."""
    if (strategy, symbol) in c8.grid_search.PAIRS:
        return c8.choose(c8.stage_a(strategy, symbol))
    return c8.choose(c8.breadth_grid(strategy, symbol))


def denoisers(strategy: str) -> dict[str, list[dict]]:
    """Denoising families: name -> parameter overrides.

    GLFT has no fair-price signal to smooth (its quotes are centred on the
    mid), so only the markout average applies to it.
    """
    fams = {"markout half-life": [{"markout_halflife": h} for h in (5.0, 10.0, 50.0)]}
    if strategy == "obi":
        fams |= {
            # An exponential average of the imbalance-driven fair-price shift.
            "signal smoothing": [{"signal_hl": h} for h in (10.0, 50.0, 200.0)],
            # Clip the standardised imbalance at +/- this many deviations.
            "alpha clipping": [{"alpha_clip": c} for c in (1.0, 1.5, 2.5)],
            # The window the imbalance is standardised over (default ten
            # minutes): longer is a steadier mean and scale.
            "standardisation window": [{"obi_window": w} for w in (3000.0, 18000.0, 36000.0)],
            # How deep into the book the imbalance sums (default 10 bp): a
            # deeper sum averages over more resting orders.
            "imbalance depth": [{"obi_depth": d} for d in (0.0005, 0.002, 0.004)],
        }
    return fams


def denoise_configs(strategy: str, symbol: str) -> dict[str, list[dict]] | None:
    base = base_choice(strategy, symbol)
    if base is None:
        return None
    fams = denoisers(strategy)
    if base["sets"].get("markout_kappa", 0.0) == 0.0:
        del fams["markout half-life"]  # no markout widening to average
    return {name: [c8.on_top(base, x) for x in variants] for name, variants in fams.items()}


def denoise_best(strategy: str, symbol: str) -> dict[str, dict] | None:
    """Each family's best variant on training sessions (None until all are in)."""
    fams = denoise_configs(strategy, symbol)
    if fams is None:
        return None
    best = {}
    for name, cfgs in fams.items():
        pick = c8.choose(cfgs)
        if pick is None:
            return None
        best[name] = pick
    return best


def jobs_for(stage: str) -> list[tuple[dict, list[str]]]:
    jobs = []
    if stage == "oracle":
        for strategy, symbol in ORACLE_TARGETS:
            base = base_choice(strategy, symbol)
            if base is not None:
                jobs += [(c8.on_top(base, v), DAYS) for v in oracle_variants(strategy).values()]
        return jobs
    if stage in ("dagger", "dagger-E"):
        import dagger

        for (strategy, symbol), (tag, _, _) in itertools.product(dagger.TARGETS, dagger_students()):
            cfgs = dagger_configs(strategy, symbol, tag)
            if cfgs is None:
                continue
            if stage == "dagger":
                jobs += [(c, TRAIN) for c in cfgs]
                continue
            pick = c8.choose(cfgs)
            if pick is not None:
                jobs += [(pick, TEST)] + [(c, TEST) for c in c8.stress_configs(pick).values()]
        return jobs
    if stage in ("largemove", "largemove-E"):
        for strategy, symbol in LARGE_TARGETS:
            base = base_choice(strategy, symbol)
            if base is None:
                continue
            if stage == "largemove":
                jobs += [(c8.on_top(base, v), DAYS) for v in large_oracle_variants().values()]
                jobs += [(c8.on_top(base, v), TRAIN) for v in large_forecast_variants()]
                continue
            pick = large_choice(strategy, symbol)
            if pick is not None:
                jobs += [(pick, TEST)] + [(c, TEST) for c in c8.stress_configs(pick).values()]
        return jobs
    if stage in ("predict", "predict-E"):
        for strategy, symbol in ORACLE_TARGETS:
            if stage == "predict":
                fams = fair_configs(strategy, symbol)
                if fams is not None:
                    jobs += [(c, TRAIN) for cfgs in fams.values() for c in cfgs]
                continue
            best = fair_best(strategy, symbol)
            if best is None:
                continue
            # Every family's training-best, held out and stress-tested.
            for cfg in best.values():
                jobs += [(cfg, TEST)] + [(c, TEST) for c in c8.stress_configs(cfg).values()]
        return jobs
    for strategy, symbol in TARGETS:
        if stage == "denoise":
            fams = denoise_configs(strategy, symbol)
            if fams is not None:
                jobs += [(c, DAYS) for cfgs in fams.values() for c in cfgs]
        elif stage == "denoise-E":
            # Stress tests of the base choice and of every family's
            # training-best variant, on the held-out sessions a stress row
            # reports -- whether or not the variant looked better, so no
            # denoiser is dismissed on the plain backtest alone.
            base = base_choice(strategy, symbol)
            best = denoise_best(strategy, symbol)
            if base is None or best is None:
                continue
            for cfg in [base, *best.values()]:
                jobs += [(c, TEST) for c in c8.stress_configs(cfg).values()]
    return jobs


def pending_units(stage: str) -> list[tuple[dict, str]]:
    seen, units = set(), []
    for cfg, days in jobs_for(stage):
        for date in days:
            unit = (c8.key(cfg), date)
            if unit in seen or c8.result(cfg, date) is not None:
                continue
            seen.add(unit)
            units.append((cfg, date))
    units.sort(key=lambda u: (u[0]["symbol"], u[1], c8.key(u[0])))
    return units


def run_one(cfg: dict, date: str, keep_out: bool = False):
    """`c8.run_one`, plus the fair-price series a `fair_src` names.

    With `keep_out` the quoter's per-step log is returned (DAgger's rollouts).
    """
    src = cfg["sets"].get("fair_src")
    if src is None and not keep_out:
        return c8.run_one(cfg, date)
    import gc

    import fair
    import run_baseline
    import strategies

    overrides = {"beta": 1.0, "queue_model": "power2", **cfg["sets"]}
    series = None if src is None else fair.series(src, cfg["symbol"], date)
    stats, _, out = run_baseline.run_day(cfg["strategy"], cfg["symbol"], date, overrides,
                                         fair=series)
    stats.update(run_baseline.quoter_stats(out, strategies.INTERVAL_NS,
                                           strategies.MAX_POSITION_DOLLAR))
    stats["date"] = date
    folder = c8.CACHE / c8.key(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(cfg, indent=1))
    tmp = folder / f"{date}.json.tmp"
    tmp.write_text(json.dumps(stats))
    tmp.rename(folder / f"{date}.json")
    del series
    if keep_out:
        return out
    del out
    gc.collect()


def cmd_run(stage: str, shard: str, units_file: str | None) -> None:
    """Replay pending sessions, one shard of them; see `c8.cmd_run`."""
    index, count = (int(x) for x in shard.split("/"))
    units = ([(c, d) for c, d in json.loads(Path(units_file).read_text())]
             if units_file else pending_units(stage))
    units = units[index::count]
    if os.environ.get("C8_REVERSE") == "1":
        units = units[::-1]
    print(f"### C9 stage {stage} shard {shard}: {len(units)} session replays", flush=True)
    deadline = float(os.environ.get("C8_DEADLINE", "inf"))
    for i, (cfg, date) in enumerate(units):
        if time.time() > deadline:
            print("### deadline reached", flush=True)
            break
        if c8.result(cfg, date) is not None:
            continue
        start = time.time()
        try:
            run_one(cfg, date)
            status = "ok"
        except Exception as exc:  # one broken unit must not stop a shard
            status = f"FAILED {exc!r}"
        print(f"{i + 1}/{len(units)} {cfg['strategy']} {cfg['symbol']} {date} {c8.key(cfg)} "
              f"{status} {time.time() - start:.0f}s", flush=True)
    print(f"### SHARD DONE {stage} {shard}", flush=True)


STAGES = ("denoise", "denoise-E", "oracle", "predict", "predict-E", "largemove", "largemove-E",
          "dagger", "dagger-E")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--stage", required=True, choices=STAGES)
    run.add_argument("--shard", default="0/1")
    run.add_argument("--units", default=None)
    export = sub.add_parser("export")
    export.add_argument("--stage", nargs="+", required=True)
    export.add_argument("--out", required=True)
    sub.add_parser("status")
    args = parser.parse_args()
    if args.cmd == "run":
        cmd_run(args.stage, args.shard, args.units)
    elif args.cmd == "export":
        units, seen = [], set()
        for stage in args.stage:
            for cfg, date in pending_units(stage):
                if (c8.key(cfg), date) not in seen:
                    seen.add((c8.key(cfg), date))
                    units.append([cfg, date])
        Path(args.out).write_text(json.dumps(units))
        print(f"{len(units)} units -> {args.out}")
    else:
        for stage in STAGES:
            print(f"{stage:10s} pending {len(pending_units(stage))}")


if __name__ == "__main__":
    main()
