"""Run C6's remaining work, skipping whatever is already finished.

The machine this runs on kills long background jobs without warning -- a
per-invocation memory cap, and anything still running when a session turn
ends.  Several hours of replays were lost that way before this existed.

(The name matters: every C6 script puts its own directory first on
`sys.path`, so a module called `queue.py` here shadows the standard library's
and breaks any dependency that imports it -- polars does, and every job in
the first run of this file died on it.)

So the queue is idempotent rather than long-lived: it enumerates every job
C6 needs, asks `runs/` which ones already have a `summary.json`, and runs the
rest in order.  Re-running it after a kill resumes; running it when
everything is done prints "nothing to do".  That makes an interrupted job
cost one job rather than a stream.

    python experiments/c6_strategy_zoo/run_queue.py --list
    python experiments/c6_strategy_zoo/run_queue.py --kind baseline --shard 0/2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import assets  # noqa: E402
import build_traces  # noqa: E402

REPO = HERE.parents[1]
RUNS = REPO / "runs"
PYTHON = str(REPO / ".venv" / "bin" / "python")

#: Single-leg strategies run on every asset; the pair strategies only on the
#: assets that have a reference leg.
SINGLE = ("glft", "hfgrid", "obi")
PAIR = ("apt", "basis")

#: Seeds on the two deep assets.  Five: C3 measured a 3.74-point spread
#: between the best and worst seed of one configuration, and C6's first round,
#: at two seeds, found single-seed rows systematically kinder to RL than
#: two-seed rows -- the signature of reading a draw as a measurement.
DEEP_SEEDS = (0, 1, 2, 3, 4)

#: v2 with one of its changes removed, each run on one pair only.
ABLATIONS = ("v2-no-cf", "v2-no-residual")
ABLATION_PAIR = ("glft", "xrpusdc")


def strategies_for(symbol: str) -> tuple[str, ...]:
    return SINGLE + (PAIR if symbol in assets.PAIRS else ())


def jobs() -> list[dict]:
    """Every job C6 needs, in dependency order."""
    out: list[dict] = []
    deep = [s for s in assets.SINGLE_LEG if s in assets.DEEP]
    breadth = [s for s in assets.SINGLE_LEG if s not in assets.DEEP]

    # 1. A spread floor per asset, from a sweep on training days only.
    for symbol in assets.SINGLE_LEG:
        out.append({
            "kind": "sweep", "id": f"sweep-half_spread_bp-glft-{symbol}",
            "check": ("c6-sweep", f"half_spread_bp-glft-{symbol}"),
            "cmd": [PYTHON, str(HERE / "sweep_spread.py"), "--strategy", "glft",
                    "--param", "half_spread_bp", "--symbol", symbol,
                    "--days", "4" if symbol in assets.DEEP else "3", "--write"],
        })

    # 2. Baselines: published and widened, every strategy on every asset.
    for symbol in deep + breadth:
        for strategy in strategies_for(symbol):
            for variant in ("", "-wide"):
                name = f"{strategy}{variant}"
                out.append({
                    "kind": "baseline", "id": f"base-{name}-{symbol}",
                    "check": ("c6", f"base-{name}-{symbol}", assets.day_sets(symbol)["all"]),
                    "needs_floor": bool(variant),
                    "cmd": [PYTHON, str(HERE / "run_baseline.py"), "--strategy", name,
                            "--symbol", symbol, "--days", "all",
                            "--tag", f"base-{name}-{symbol}"],
                })

    # 2b. Baseline reward traces for the counterfactual recipes, before any
    #     of the five seeds that need them start at once.
    for symbol in deep:
        for strategy in strategies_for(symbol):
            name = f"{strategy}-wide"
            out.append({
                "kind": "trace", "id": f"trace-{name}-{symbol}",
                "done": lambda n=name, s=symbol: all(
                    build_traces.trace_path(n, s, d).exists() for d in assets.train_days(s)),
                "cmd": [PYTHON, str(HERE / "build_traces.py"), "--strategy", name,
                        "--symbol", symbol],
            })

    # 3. Learned policies, on the widened variant: the point of C6 is whether
    #    RL adds anything once the spread is already sensible.
    #
    #    On the two deep assets: five seeds of both recipes, because C3
    #    measured a 3.74-point spread between seeds and two seeds cannot see
    #    past it.  v2 first, since it is the new thing; then v1's extra
    #    seeds; then the two ablations, on one pair only, which say which of
    #    v2's changes did the work.
    for recipe in ("v2", "v1"):
        for symbol in deep:
            for strategy in strategies_for(symbol):
                for seed in DEEP_SEEDS:
                    out.append(train_job(f"{strategy}-wide", symbol, seed, recipe))
    for recipe in ABLATIONS:
        for seed in DEEP_SEEDS:
            out.append(train_job(f"{ABLATION_PAIR[0]}-wide", ABLATION_PAIR[1], seed, recipe))
    for symbol in breadth:
        for strategy in strategies_for(symbol):
            out.append(train_job(f"{strategy}-wide", symbol, 0, "v1"))
    return out


def train_job(name: str, symbol: str, seed: int, recipe: str) -> dict:
    tag = "forced-all" if recipe == "v1" else recipe
    passes = 1 if symbol == "ethusdc" else 2
    return {
        "kind": "train", "id": f"train-{name}-{symbol}-{tag}-s{seed}",
        "check": ("c6", f"{name}-{symbol}-td3-{tag}-s{seed}"),
        "needs_floor": True,
        "cmd": [PYTHON, str(HERE / "train.py"), "--strategy", name, "--symbol", symbol,
                "--seed", str(seed), "--passes", str(passes), "--algo", "td3",
                "--recipe", recipe],
    }


def finished(phase: str, prefix: str, days: list[str] | None = None) -> Path | None:
    """The newest finished run of this job, if it covered the right sessions.

    A run over an older, shorter day set is not "done": when XRPUSDC's
    held-out set was widened from four sessions to eleven, every XRP baseline
    and evaluation on the old four would otherwise have been skipped.
    """
    root = RUNS / phase
    if not root.exists():
        return None
    for path in sorted(root.iterdir(), reverse=True):
        if path.name.startswith(prefix + "__") and (path / "summary.json").exists():
            if days is not None:
                ran = json.loads((path / "config.json").read_text()).get("days")
                if ran != list(days):
                    return None
            return path
    return None


def policy_jobs() -> list[dict]:
    """Evaluation jobs, discovered from the checkpoints that exist right now."""
    out = []
    for path in sorted((RUNS / "c6").iterdir()) if (RUNS / "c6").exists() else []:
        checkpoint = path / "policy.pt"
        if not checkpoint.exists() or not (path / "summary.json").exists():
            continue
        # Smoke-test runs cap the episode at a few hundred steps, so their
        # policies saw minutes of trading rather than sessions.  One of them
        # reached the results table before this check existed.
        config_path = path / "config.json"
        run_config = json.loads(config_path.read_text()) if config_path.exists() else {}
        if run_config.get("max_steps"):
            continue
        out.append({
            "kind": "eval", "id": f"eval-{path.name}",
            "check": ("c6-eval", path.name,
                      assets.test_days(run_config["symbol"]) if "symbol" in run_config else None),
            "cmd": [PYTHON, str(HERE / "evaluate.py"), "--policy", str(checkpoint),
                    "--days", "test"],
        })
    return out


def have_floor(symbol: str) -> bool:
    path = HERE / "spread_floor.json"
    return path.exists() and symbol in json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", default=None,
                        choices=["sweep", "baseline", "trace", "train", "eval"])
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--shard", default="0/1", help="`i/n`: run every n-th job")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=0)
    args = parser.parse_args()

    index, total = (int(x) for x in args.shard.split("/"))
    pending = [j for j in jobs() + policy_jobs()
               if not (j["done"]() if "done" in j else finished(*j["check"]))]
    if args.kind:
        pending = [j for j in pending if j["kind"] == args.kind]
    if args.symbol:
        pending = [j for j in pending if args.symbol in j["id"]]
    # A widened variant is meaningless before its asset has a floor; skip
    # rather than fail, so the sweep shard can fill it in and a later pass
    # picks it up.
    pending = [j for j in pending
               if not j.get("needs_floor") or have_floor(_symbol_of(j))]
    mine = [j for i, j in enumerate(pending) if i % total == index]
    if args.max_jobs:
        mine = mine[: args.max_jobs]

    if args.list:
        for job in mine:
            print(f"{job['kind']:9s} {job['id']}")
        print(f"\n{len(mine)} pending of {len(pending)} (shard {args.shard})")
        return

    if not mine:
        print("nothing to do")
        return
    for job in mine:
        started = time.time()
        print(f"\n### {job['id']}  {time.strftime('%T')}", flush=True)
        result = subprocess.run(job["cmd"], cwd=REPO)
        status = "ok" if result.returncode == 0 else f"FAILED rc={result.returncode}"
        print(f"### {job['id']} {status} in {time.time() - started:.0f}s", flush=True)
    print(f"\n### SHARD DONE {time.strftime('%T')}")


def _symbol_of(job: dict) -> str:
    for symbol in sorted(assets.SINGLE_LEG, key=len, reverse=True):
        if symbol in job["id"]:
            return symbol
    return ""


if __name__ == "__main__":
    main()
