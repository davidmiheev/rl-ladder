"""C6's master table: every strategy, baseline and learned, in one place.

    python experiments/c6_strategy_zoo/master_table.py
    python experiments/c6_strategy_zoo/master_table.py --markdown --out docs/writeups/c6-table.md

Reads every finished run under `runs/c6`, `runs/c6-eval` and `runs/c6-sweep`
and prints one row per (strategy, variant, day set).  Unfinished runs -- the
ones without a `summary.json`, including anything the OOM killer took -- are
skipped rather than half-reported.

Two blocks of columns, and the split matters:

**Track record** is the sessions chained as one period, which is what an
operator running the strategy every day would actually have got.  Cumulative
return, the annualised Sharpe of the daily returns, the worst single session,
and how many sessions made money.

**Per session** is the median across sessions.  C3 found these two rank
strategies almost inversely on this data: a strategy with ten flat sessions
and one catastrophe has the best median session in the table and the second
worst track record.  Both are printed so the gap is visible; the track record
is the headline.

**Trade intensity** is three separate things that are easy to conflate:

- `A` and `k` fit `lambda(d) = A exp(-k d)` to the market's own aggressive
  order arrivals.  They describe the *market*, not the strategy, and are
  measured identically for all five, so they are the yardstick the rest of
  the row is read against.
- `1/k` ticks is the depth at which arrivals fall by a factor of e.  C1's
  proof that `inf_gamma H(gamma) = 1/k` means no amount of risk aversion lets
  a GLFT quoter sit inside it.
- `fills/h` is what the strategy actually got. A strategy quoting at 30 bp on
  an asset whose arrivals decay over 1 bp is not a market maker with a wide
  spread; it is a strategy that does not trade, and the two are only
  distinguishable by this column.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RUNS = HERE.parents[1] / "runs"

COLUMNS = [
    ("strategy", "{:<8}", "strategy"),
    ("policy", "{:<12}", "policy"),
    ("spread", "{:<10}", "spread"),
    ("symbol", "{:<9}", "symbol"),
    ("days", "{:<6}", "days"),
    ("n", "{:>3}", "n"),
    ("cumulative", "{:>9}", "cum %"),
    ("sharpe", "{:>7}", "SR"),
    ("worst", "{:>8}", "worst %"),
    ("profitable", "{:>6}", "win"),
    ("median_SR", "{:>8}", "med SR"),
    ("median_return", "{:>9}", "med %"),
    ("max_dd", "{:>7}", "maxDD%"),
    ("half_spread_bp", "{:>8}", "half bp"),
    ("A", "{:>7}", "A"),
    ("k", "{:>8}", "k"),
    ("inv_k", "{:>7}", "1/k t"),
    ("fills_h", "{:>8}", "fills/h"),
    ("abs_q", "{:>7}", "|q|"),
    ("signal_bp", "{:>8}", "sig bp"),
]


def finished(phase: str):
    """Every run directory under `phase` that has a summary."""
    root = RUNS / phase
    if not root.exists():
        return []
    out = []
    for path in sorted(root.iterdir()):
        summary = path / "summary.json"
        config = path / "config.json"
        if summary.exists() and config.exists():
            out.append((path, json.loads(config.read_text()), json.loads(summary.read_text())))
    return out


def _period(returns: np.ndarray) -> dict[str, float]:
    return {
        "cumulative": float(np.prod(1.0 + returns) - 1.0) * 100,
        "sharpe": float(np.mean(returns) / np.std(returns) * np.sqrt(365))
        if np.std(returns) > 0 else float("nan"),
        "worst": float(returns.min()) * 100,
        "profitable": f"{int((returns > 0).sum())}/{len(returns)}",
    }


#: The split, repeated here rather than imported, because the table has to
#: stay readable from a runs directory alone.
SPLIT_DATE = "2025-09-15"

#: A run over fewer sessions than this is a smoke test, not a result.
MIN_SESSIONS = 4


def _summarise(per_day: list[dict], fields: dict[str, str]) -> dict:
    """Period and per-session statistics over one block of sessions.

    `fields` maps the table's column names onto the per-session keys, because
    the baseline runner and the evaluator name a few of them differently and
    a row from each has to land in the same column.
    """
    returns = np.array([d["Return"] for d in per_day])
    row = {"n": len(per_day), **_period(returns),
           "median_SR": float(np.median([d["SR"] for d in per_day])),
           "median_return": float(np.median(returns)) * 100,
           "max_dd": float(np.max([d["MaxDrawdown"] for d in per_day])) * 100}
    for column, key in fields.items():
        values = [d[key] for d in per_day if key in d]
        row[column] = float(np.median(values)) if values else float("nan")
    return row


def _split(per_day: list[dict]) -> list[tuple[str, list[dict]]]:
    """Separate a run's sessions into their train and held-out halves.

    A run over the `all` day set covers both sides of the split, and pooling
    them would put a strategy's training days into the number the held-out
    comparison is made on.  Splitting here means one run yields up to two
    rows and every row is about one side.
    """
    train = [d for d in per_day if d["date"] < SPLIT_DATE]
    test = [d for d in per_day if d["date"] >= SPLIT_DATE]
    return [(name, block) for name, block in (("train", train), ("test", test))
            if len(block) >= MIN_SESSIONS]


BASELINE_FIELDS = {
    "half_spread_bp": "median_half_spread_bp",
    "A": "median_A",
    "k": "median_k",
    "inv_k": "spread_floor_tick",
    "fills_h": "fills_per_hour",
    "abs_q": "median_abs_position",
    "signal_bp": "median_abs_signal_bp",
}


def baseline_rows() -> list[dict]:
    """Rows from `run_baseline.py`, which replays fixed parameters."""
    rows = []
    for path, config, summary in finished("c6"):
        if "per_day" not in summary or "strategy" not in summary:
            continue  # a training run, not a baseline replay
        del config
        name = summary["strategy"]
        base, _, suffix = name.partition("-")
        for day_set, block in _split(summary["per_day"]):
            rows.append({
                "strategy": base,
                "policy": "baseline",
                "spread": "wide" if suffix == "wide" else "published",
                "symbol": summary["symbol"],
                "days": day_set,
                **_summarise(block, BASELINE_FIELDS),
                "_dir": path.name,
            })
    return rows


def eval_rows() -> list[dict]:
    """Rows from `evaluate.py`, which replays a learned policy or its baseline."""
    rows = []
    for path, config, summary in finished("c6-eval"):
        if "period" not in summary:
            continue
        per_day = summary["per_day"]
        if len(per_day) < MIN_SESSIONS:
            continue
        for d in per_day:
            d["inv_k"] = 1.0 / d["median_k"] if d["median_k"] else float("nan")
        name = summary["strategy"]
        base, _, suffix = name.partition("-")
        learned = config["policy"] != "baseline"
        seed = path.name.split("-s")[-1].split("__")[0] if learned else ""
        rows.append({
            "strategy": base,
            "policy": (f"RL TD3 s{seed}" if seed.isdigit() else "RL TD3") if learned else "baseline",
            "spread": "wide" if suffix == "wide" else "published",
            "symbol": summary["symbol"],
            "days": config["day_set"],
            **_summarise(per_day, EVAL_FIELDS),
            "_dir": path.name,
        })
    return rows


EVAL_FIELDS = {
    "half_spread_bp": "median_half_spread_bp",
    "A": "median_A",
    "k": "median_k",
    "inv_k": "inv_k",
    "fills_h": "fills_per_hour",
    "abs_q": "mean_abs_inventory",
    "signal_bp": "median_abs_signal_bp",
}


def fmt(row: dict, key: str, spec: str) -> str:
    value = row.get(key, "")
    if isinstance(value, float):
        if not np.isfinite(value):
            text = "-"
        elif key in ("k",):
            text = f"{value:.4f}"
        elif key in ("A", "inv_k", "fills_h", "abs_q", "half_spread_bp", "signal_bp"):
            text = f"{value:.2f}"
        else:
            text = f"{value:+.2f}" if key in ("cumulative", "worst", "median_return") else f"{value:.2f}"
    else:
        text = str(value)
    return spec.format(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--days", default=None, choices=["train", "test"])
    args = parser.parse_args()

    rows = baseline_rows() + eval_rows()
    if args.days:
        rows = [r for r in rows if r["days"] == args.days]
    if not rows:
        raise SystemExit("no finished C6 runs found under runs/c6 or runs/c6-eval")
    # Grouped so each strategy's three rows -- published baseline, widened
    # baseline, and the policy learned on top of the widened one -- sit
    # together, in the order they were built.
    order = {"glft": 0, "hfgrid": 1, "apt": 2, "basis": 3, "obi": 4}
    spread_order = {"published": 0, "wide": 1}
    rows.sort(key=lambda r: (r["days"] != "test", r["symbol"],
                             order.get(r["strategy"], 9),
                             spread_order.get(r["spread"], 2),
                             r["policy"] != "baseline", r["policy"]))

    if args.markdown:
        head = "| " + " | ".join(label for _, _, label in COLUMNS) + " |"
        rule = "| " + " | ".join("---" for _ in COLUMNS) + " |"
        lines = [head, rule]
        for row in rows:
            lines.append("| " + " | ".join(fmt(row, key, "{}").strip()
                                           for key, _, _ in COLUMNS) + " |")
        text = "\n".join(lines)
    else:
        header = " ".join(spec.format(label) for _, spec, label in COLUMNS)
        lines = [header, "-" * len(header)]
        previous = None
        for row in rows:
            group = (row["days"], row["symbol"], row["strategy"])
            if previous is not None and group != previous:
                lines.append("")
            previous = group
            lines.append(" ".join(fmt(row, key, spec) for key, spec, _ in COLUMNS))
        text = "\n".join(lines)

    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
