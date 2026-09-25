"""C7's baseline variants, as an idempotent queue.

Two families, both fixed-parameter replays through C6's quoter, written under
`runs/c7/` so that C6's tables never pick them up:

    vol       sigma replaced, on the held-out sessions, for the two strategies
              that consume it (glft, hfgrid), published and widened:
                volscale  C6's sigma times sqrt(the variance bias measured on
                          the training days by `vol_bakeoff.py`)
                har       the HAR forecast, coefficients fitted on the
                          training days by `vol_bakeoff.py`
              Nothing is tuned on held-out data: both come from the training
              days, and the reference is C6's own baseline on the same days.

    markout   the markout rule's kappa swept over {1, 2, 4} on four training
              sessions (the same four the spread floor was swept on, and for
              the same reason: a sweep is a search, so it gets the least data
              it can run on); `--kind markout-test` then runs each
              (strategy, asset)'s best kappa on the held-out sessions, if it
              beat kappa = 0 on the sweep.

    python experiments/c7_signals/run_variants.py --kind vol --shard 0/2
    python experiments/c7_signals/run_variants.py --kind markout-sweep
    python experiments/c7_signals/run_variants.py --kind markout-test
    python experiments/c7_signals/run_variants.py --kind markout-breadth

`markout-breadth` is the untuned check: kappa = 2 on the seven assets the
sweep never saw, on their held-out sessions, with nothing searched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "c6_strategy_zoo"))
sys.path.insert(0, str(REPO / "experiments" / "c1_glft_baseline"))

import assets  # noqa: E402
import config  # noqa: E402

RUNS = REPO / "runs"
SYMBOLS = ("ethusdc", "xrpusdc")
VOL_STRATEGIES = ("glft", "glft-wide", "hfgrid", "hfgrid-wide")
MARKOUT_STRATEGIES = ("glft-wide", "hfgrid-wide", "obi-wide")
KAPPAS = (1.0, 2.0, 4.0)
SWEEP_DAYS = config.PAIR_TRAIN_DAYS


def finished(prefix: str, days: list[str]) -> Path | None:
    root = RUNS / "c7"
    if not root.exists():
        return None
    for path in sorted(root.iterdir(), reverse=True):
        if path.name.startswith(prefix + "__") and (path / "summary.json").exists():
            if json.loads((path / "config.json").read_text()).get("days") == list(days):
                return path
    return None


def vol_sets(symbol: str) -> dict[str, dict[str, float]]:
    fit = json.loads((RUNS / "c7-vol" / f"{symbol}.json").read_text())["results"]["y1"]
    b = fit["har_beta"]
    return {
        "volscale": {"vol_scale": fit["rows"]["current"]["scale"]},
        "har": {"har_b0": fit["har_b0"], "har_b1": b[1], "har_b2": b[2], "har_b3": b[3]},
    }


def vol_jobs() -> list[dict]:
    jobs = []
    for symbol in SYMBOLS:
        days = assets.test_days(symbol)
        for variant, sets in vol_sets(symbol).items():
            for strategy in VOL_STRATEGIES:
                jobs.append({"tag": f"{strategy}-{symbol}-{variant}", "strategy": strategy,
                             "symbol": symbol, "days": days, "sets": sets})
    return jobs


def markout_sweep_jobs() -> list[dict]:
    return [
        {"tag": f"{strategy}-{symbol}-mk{kappa:g}-sweep", "strategy": strategy,
         "symbol": symbol, "days": SWEEP_DAYS, "sets": {"markout_kappa": kappa}}
        for kappa in KAPPAS for symbol in SYMBOLS for strategy in MARKOUT_STRATEGIES
    ]


def reference_return(strategy: str, symbol: str, days: list[str]) -> float | None:
    """C6's own baseline (kappa = 0) chained over `days`, from its newest run."""
    root = RUNS / "c6"
    for path in sorted(root.glob(f"base-{strategy}-{symbol}__*"), reverse=True):
        summary = path / "summary.json"
        if not summary.exists():
            continue
        per_day = {d["date"]: d["Return"] for d in json.loads(summary.read_text())["per_day"]}
        if all(d in per_day for d in days):
            total = 1.0
            for d in days:
                total *= 1.0 + per_day[d]
            return total - 1.0
    return None


def chained(path: Path) -> float:
    total = 1.0
    for d in json.loads((path / "summary.json").read_text())["per_day"]:
        total *= 1.0 + d["Return"]
    return total - 1.0


def best_kappa(strategy: str, symbol: str) -> tuple[float, dict[float, float]] | None:
    scores = {0.0: reference_return(strategy, symbol, SWEEP_DAYS)}
    for kappa in KAPPAS:
        path = finished(f"{strategy}-{symbol}-mk{kappa:g}-sweep", SWEEP_DAYS)
        if path is None:
            return None
        scores[kappa] = chained(path)
    if scores[0.0] is None:
        raise SystemExit(f"no C6 baseline for {strategy} {symbol} on the sweep days")
    return max(scores, key=scores.get), scores


def markout_test_jobs() -> list[dict]:
    jobs = []
    for symbol in SYMBOLS:
        for strategy in MARKOUT_STRATEGIES:
            pick = best_kappa(strategy, symbol)
            if pick is None or pick[0] == 0.0:
                continue
            jobs.append({"tag": f"{strategy}-{symbol}-mk{pick[0]:g}", "strategy": strategy,
                         "symbol": symbol, "days": assets.test_days(symbol),
                         "sets": {"markout_kappa": pick[0]}})
    return jobs


#: The untuned test: one kappa, picked in advance as the middle of the swept
#: grid, on every asset the sweep never saw, on their held-out sessions.
BREADTH_KAPPA = 2.0


def markout_breadth_jobs() -> list[dict]:
    return [
        {"tag": f"{strategy}-{symbol}-mk{BREADTH_KAPPA:g}", "strategy": strategy,
         "symbol": symbol, "days": assets.test_days(symbol),
         "sets": {"markout_kappa": BREADTH_KAPPA}}
        for symbol in assets.SINGLE_LEG if symbol not in SYMBOLS
        for strategy in MARKOUT_STRATEGIES
    ]


KINDS = {"vol": vol_jobs, "markout-sweep": markout_sweep_jobs, "markout-test": markout_test_jobs,
         "markout-breadth": markout_breadth_jobs}


def run(job: dict) -> bool:
    cmd = [sys.executable, str(REPO / "experiments" / "c6_strategy_zoo" / "run_baseline.py"),
           "--strategy", job["strategy"], "--symbol", job["symbol"], "--phase", "c7",
           "--tag", job["tag"], "--no-records", "--dates", *job["days"]]
    for name, value in job["sets"].items():
        cmd += ["--set", f"{name}={value!r}"]
    start = time.time()
    print(f"### {job['tag']}  {time.strftime('%H:%M:%S')}", flush=True)
    rc = subprocess.call(cmd)
    print(f"### {job['tag']} {'ok' if rc == 0 else f'FAILED rc={rc}'} in "
          f"{time.time() - start:.0f}s", flush=True)
    return rc == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", required=True, choices=sorted(KINDS))
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    index, count = (int(x) for x in args.shard.split("/"))
    jobs = KINDS[args.kind]()[index::count]
    pending = [j for j in jobs if finished(j["tag"], j["days"]) is None]
    if args.list:
        for job in pending:
            print(job["tag"], job["sets"])
        print(f"{len(pending)} pending of {len(jobs)} (shard {args.shard})")
        return
    for job in pending:
        run(job)


if __name__ == "__main__":
    main()
