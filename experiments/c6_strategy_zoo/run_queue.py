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

REPO = HERE.parents[1]
RUNS = REPO / "runs"
PYTHON = str(REPO / ".venv" / "bin" / "python")

#: Single-leg strategies run on every asset; the pair strategies only on the
#: assets that have a reference leg.
SINGLE = ("glft", "hfgrid", "obi")
PAIR = ("apt", "basis")

#: Seeds for the learned policies.  Two, for the reason `train.py` records.
SEEDS = (0, 1)


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
                    "check": ("c6", f"base-{name}-{symbol}"),
                    "needs_floor": bool(variant),
                    "cmd": [PYTHON, str(HERE / "run_baseline.py"), "--strategy", name,
                            "--symbol", symbol, "--days", "all",
                            "--tag", f"base-{name}-{symbol}"],
                })

    # 3. Learned policies, on the widened variant: the point of C6 is whether
    #    RL adds anything once the spread is already sensible.
    for symbol in deep + breadth:
        for strategy in strategies_for(symbol):
            for seed in SEEDS if symbol in assets.DEEP else (0,):
                name = f"{strategy}-wide"
                passes = 1 if symbol == "ethusdc" else 2
                out.append({
                    "kind": "train", "id": f"train-{name}-{symbol}-s{seed}",
                    "check": ("c6", f"{name}-{symbol}-td3-forced-all-s{seed}"),
                    "needs_floor": True,
                    "cmd": [PYTHON, str(HERE / "train.py"), "--strategy", name,
                            "--symbol", symbol, "--seed", str(seed),
                            "--passes", str(passes), "--algo", "td3",
                            "--participation", "forced"],
                })
    return out


def finished(phase: str, prefix: str) -> Path | None:
    root = RUNS / phase
    if not root.exists():
        return None
    for path in sorted(root.iterdir(), reverse=True):
        if path.name.startswith(prefix + "__") and (path / "summary.json").exists():
            return path
    return None


def policy_jobs() -> list[dict]:
    """Evaluation jobs, discovered from the checkpoints that exist right now."""
    out = []
    for path in sorted((RUNS / "c6").iterdir()) if (RUNS / "c6").exists() else []:
        checkpoint = path / "policy.pt"
        if not checkpoint.exists() or not (path / "summary.json").exists():
            continue
        out.append({
            "kind": "eval", "id": f"eval-{path.name}",
            "check": ("c6-eval", path.name),
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
                        choices=["sweep", "baseline", "train", "eval"])
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--shard", default="0/1", help="`i/n`: run every n-th job")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=0)
    args = parser.parse_args()

    index, total = (int(x) for x in args.shard.split("/"))
    pending = [j for j in jobs() + policy_jobs() if not finished(*j["check"])]
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
