"""C4, step 1: the search baseline -- a joint grid over the fixed parameters.

Part C's two positive findings are both fixed choices: a wider, per-asset
spread floor (C6) and a spread that widens after toxic fills (C7). Neither was
chosen jointly with the other, nor with the inventory skew. Before any
learning method is asked to search this space, the plain search has to be run:
it is the number GRPO- and PPO-style search must beat at equal budget, and it
is the most direct route to a configuration that makes money held out.

The grid, relative to each strategy's widened preset:

    floor_mult   {1, 1.5, 2}   times the asset's calibrated half-spread floor
    kappa        {0, 2, 4}     C7's markout widening
    skew_mult    {1, 2}        times the strategy's inventory skew

with C7's volatility bias correction on for the two strategies that use
sigma. Every point runs on **all** of the asset's training sessions, and one
point per (strategy, asset) is chosen there -- by chained return, among the
points that take at least `MIN_FILLS` fills on a median session, so that a
configuration cannot win by not trading. `--kind test` then runs each chosen
point once on the held-out sessions. Nothing is ever chosen on held-out data.

    python experiments/c4_ppo_grpo/grid_search.py --kind grid --shard 0/3
    python experiments/c4_ppo_grpo/grid_search.py --kind select
    python experiments/c4_ppo_grpo/grid_search.py --kind test
    python experiments/c4_ppo_grpo/grid_search.py --kind search-test   # GRPO/PPO finals

`--train-set train20` repeats the grid and the choice on all twenty pre-split
days (after `fetch_train20.py`), instead of C6's ten (ETH) and four (XRP).
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "c6_strategy_zoo"))
sys.path.insert(0, str(REPO / "experiments" / "c1_glft_baseline"))

import assets  # noqa: E402
import strategies  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_train20 import TRAIN20  # noqa: E402

RUNS = REPO / "runs"
PAIRS = (
    ("glft", "ethusdc"), ("hfgrid", "ethusdc"), ("obi", "ethusdc"),
    ("glft", "xrpusdc"), ("hfgrid", "xrpusdc"), ("obi", "xrpusdc"),
    ("apt", "xrpusdc"), ("basis", "xrpusdc"),
)
FLOOR_MULTS = (1.0, 1.5, 2.0)
KAPPAS = (0.0, 2.0, 4.0)
SKEW_MULTS = (1.0, 2.0)
#: A configuration that barely trades is not a market maker, however well it
#: scores; C6's floor sweep used 50 a day, which the widened `glft` on XRPUSDC
#: already fails (10 a day), so the bar here is lower and reported.
MIN_FILLS = 20

#: Which training sessions the grid is scored on. `c6` is C6's split (ten
#: ETHUSDC days, four XRPUSDC); `train20` is every pre-split day for both
#: (`fetch_train20.py`). Run tags carry the set so the two never mix.
TRAIN_SETS = {"c6": ("train", assets.train_days), "train20": ("train20", lambda _: TRAIN20)}


def sigma_users(strategy: str) -> bool:
    return strategy in ("glft", "hfgrid")


def point_sets(strategy: str, symbol: str, floor_mult: float, kappa: float,
               skew_mult: float) -> dict[str, float]:
    """Quoter overrides for one grid point, relative to the widened preset."""
    kwargs = {"beta": 1.0} if strategy == "apt" else {}
    base = strategies.describe(strategies.build(f"{strategy}-wide", symbol, **kwargs))
    sets = {"half_spread_bp": base["half_spread_bp"] * floor_mult, "markout_kappa": kappa}
    skew_name = "adj2" if strategy == "glft" else "skew_bp"
    sets[skew_name] = base[skew_name] * skew_mult
    if sigma_users(strategy):
        fit = json.loads((RUNS / "c7-vol" / f"{symbol}.json").read_text())["results"]["y1"]
        sets["vol_scale"] = fit["rows"]["current"]["scale"]
    return sets


def tag(strategy: str, symbol: str, floor_mult: float, kappa: float, skew_mult: float) -> str:
    return f"{strategy}-{symbol}-f{floor_mult:g}-k{kappa:g}-s{skew_mult:g}"


def finished(prefix: str, days: list[str]) -> Path | None:
    root = RUNS / "c4"
    if not root.exists():
        return None
    for path in sorted(root.iterdir(), reverse=True):
        if path.name.startswith(prefix + "__") and (path / "summary.json").exists():
            if json.loads((path / "config.json").read_text()).get("days") == list(days):
                return path
    return None


def grid_jobs(train_set: str = "c6") -> list[dict]:
    suffix, days_for = TRAIN_SETS[train_set]
    jobs = []
    for strategy, symbol in PAIRS:
        days = days_for(symbol)
        for f, k, s in itertools.product(FLOOR_MULTS, KAPPAS, SKEW_MULTS):
            jobs.append({"tag": tag(strategy, symbol, f, k, s) + f"-{suffix}",
                         "strategy": strategy, "symbol": symbol, "days": days,
                         "sets": point_sets(strategy, symbol, f, k, s)})
    return jobs


def summarise(path: Path) -> dict:
    per_day = json.loads((path / "summary.json").read_text())["per_day"]
    returns = np.array([d["Return"] for d in per_day])
    return {"chained": float(np.prod(1.0 + returns) - 1.0) * 100,
            "median_fills": float(np.median([d["DailyNumberOfTrades"] for d in per_day])),
            "days_up": int((returns > 0).sum()), "n": len(returns)}


def select(train_set: str = "c6") -> dict[tuple[str, str], dict]:
    """The chosen point per (strategy, asset), from training sessions only."""
    suffix, days_for = TRAIN_SETS[train_set]
    chosen = {}
    for strategy, symbol in PAIRS:
        days = days_for(symbol)
        scored = []
        for f, k, s in itertools.product(FLOOR_MULTS, KAPPAS, SKEW_MULTS):
            path = finished(tag(strategy, symbol, f, k, s) + f"-{suffix}", days)
            if path is not None:
                scored.append({"point": (f, k, s), **summarise(path)})
        eligible = [r for r in scored if r["median_fills"] >= MIN_FILLS]
        pool = eligible or scored
        if not pool:
            continue
        best = max(pool, key=lambda r: r["chained"])
        chosen[(strategy, symbol)] = {**best, "eligible": bool(eligible),
                                      "scored": len(scored), "grid": scored}
    return chosen


def test_jobs(train_set: str = "c6") -> list[dict]:
    # A point chosen under either training set runs under the same `-test`
    # tag: the same parameters on the same held-out days is the same run.
    jobs = []
    for (strategy, symbol), pick in select(train_set).items():
        if pick["scored"] < len(FLOOR_MULTS) * len(KAPPAS) * len(SKEW_MULTS):
            continue  # choose only once the whole grid is in
        f, k, s = pick["point"]
        jobs.append({"tag": tag(strategy, symbol, f, k, s) + "-test", "strategy": strategy,
                     "symbol": symbol, "days": assets.test_days(symbol),
                     "sets": point_sets(strategy, symbol, f, k, s)})
    return jobs


def search_test_jobs() -> list[dict]:
    """Held-out runs of every finished GRPO/PPO search's final point, one per seed."""
    jobs = []
    root = RUNS / "c4-search"
    for path in sorted(root.glob("*__*")) if root.exists() else []:
        summary = path / "summary.json"
        if not summary.exists():
            continue
        config = json.loads((path / "config.json").read_text())
        if config["iterations"] < 20:
            continue  # smoke tests
        final = json.loads(summary.read_text())
        point = (final["floor_mult"], final["kappa"], final["skew_mult"])
        strategy, symbol = config["strategy"], config["symbol"]
        target = f"-F{config['fill_target']:g}" if config.get("fill_target") else ""
        name = f"{config['method']}{target}-{strategy}-{symbol}-s{config['seed']}-test"
        jobs.append({"tag": name, "strategy": strategy, "symbol": symbol,
                     "days": assets.test_days(symbol),
                     "sets": point_sets(strategy, symbol, *point)})
    return jobs


def run(job: dict) -> None:
    cmd = [sys.executable, str(REPO / "experiments" / "c6_strategy_zoo" / "run_baseline.py"),
           "--strategy", f"{job['strategy']}-wide", "--symbol", job["symbol"],
           "--phase", "c4", "--tag", job["tag"], "--no-records", "--dates", *job["days"]]
    for name, value in job["sets"].items():
        cmd += ["--set", f"{name}={value!r}"]
    start = time.time()
    print(f"### {job['tag']}  {time.strftime('%H:%M:%S')}", flush=True)
    rc = subprocess.call(cmd, stdout=subprocess.DEVNULL)
    print(f"### {job['tag']} {'ok' if rc == 0 else f'FAILED rc={rc}'} in "
          f"{time.time() - start:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", required=True,
                        choices=["grid", "select", "test", "search-test"])
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--train-set", default="c6", choices=sorted(TRAIN_SETS))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.kind == "select":
        for (strategy, symbol), pick in select(args.train_set).items():
            f, k, s = pick["point"]
            flag = "" if pick["eligible"] else "  (no point took enough fills)"
            print(f"{strategy:<7}{symbol:<9} floor x{f:g} kappa {k:g} skew x{s:g}  "
                  f"train {pick['chained']:+6.2f}%  up {pick['days_up']}/{pick['n']}  "
                  f"fills/day {pick['median_fills']:.0f}  [{pick['scored']} scored]{flag}")
        return

    index, count = (int(x) for x in args.shard.split("/"))
    make = {"grid": lambda: grid_jobs(args.train_set),
            "test": lambda: test_jobs(args.train_set),
            "search-test": search_test_jobs}[args.kind]
    jobs = make()[index::count]
    pending = [j for j in jobs if finished(j["tag"], j["days"]) is None]
    if args.list:
        for job in pending:
            print(job["tag"])
        print(f"{len(pending)} pending of {len(jobs)} (shard {args.shard})")
        return
    for job in pending:
        run(job)
    print("### SHARD DONE", flush=True)


if __name__ == "__main__":
    main()
