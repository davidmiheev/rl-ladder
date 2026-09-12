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
    ("strategy", "{:<14}", "strategy"),
    ("variant", "{:<10}", "variant"),
    ("symbol", "{:<8}", "symbol"),
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


def _day_set(days: list[str]) -> str:
    return "test" if days and days[0] >= "2025-09-15" else "train"


def baseline_rows() -> list[dict]:
    """Rows from `run_baseline.py`, which replays fixed parameters."""
    rows = []
    for path, config, summary in finished("c6"):
        if "per_day" not in summary or "strategy" not in summary:
            continue  # a training run, not a baseline replay
        per_day = summary["per_day"]
        returns = np.array([d["Return"] for d in per_day])
        name = summary["strategy"]
        base, _, suffix = name.partition("-")
        rows.append({
            "strategy": base,
            "variant": suffix or "published",
            "symbol": summary["symbol"],
            "days": _day_set(config["days"]),
            "n": len(per_day),
            **_period(returns),
            "median_SR": float(np.median([d["SR"] for d in per_day])),
            "median_return": float(np.median(returns)) * 100,
            "max_dd": float(np.max([d["MaxDrawdown"] for d in per_day])) * 100,
            "half_spread_bp": float(np.median([d["median_half_spread_bp"] for d in per_day])),
            "A": float(np.median([d["median_A"] for d in per_day])),
            "k": float(np.median([d["median_k"] for d in per_day])),
            "inv_k": float(np.median([d["spread_floor_tick"] for d in per_day])),
            "fills_h": float(np.median([d["fills_per_hour"] for d in per_day])),
            "abs_q": float(np.median([d["median_abs_position"] for d in per_day])),
            "signal_bp": float(np.median([d["median_abs_signal_bp"] for d in per_day])),
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
        returns = np.array([d["Return"] for d in per_day])
        name = summary["strategy"]
        base, _, suffix = name.partition("-")
        learned = config["policy"] != "baseline"
        variant = "TD3" if learned else "baseline"
        if suffix:
            variant = f"{variant}/{suffix}"
        rows.append({
            "strategy": base,
            "variant": variant,
            "symbol": summary["symbol"],
            "days": config["day_set"],
            "n": len(per_day),
            **_period(returns),
            "median_SR": float(np.median([d["SR"] for d in per_day])),
            "median_return": float(np.median(returns)) * 100,
            "max_dd": float(np.max([d["MaxDrawdown"] for d in per_day])) * 100,
            "half_spread_bp": float(np.median([d["median_half_spread_bp"] for d in per_day])),
            "A": float(np.median([d["median_A"] for d in per_day])),
            "k": float(np.median([d["median_k"] for d in per_day])),
            "inv_k": float(np.median([1.0 / d["median_k"] if d["median_k"] else np.nan
                                      for d in per_day])),
            "fills_h": float(np.median([d["fills_per_hour"] for d in per_day])),
            "abs_q": float(np.median([d["mean_abs_inventory"] for d in per_day])),
            "signal_bp": float(np.median([d["median_abs_signal_bp"] for d in per_day])),
            "_dir": path.name,
        })
    return rows


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
    order = {"glft": 0, "hfgrid": 1, "apt": 2, "basis": 3, "obi": 4}
    rows.sort(key=lambda r: (r["days"] != "test", order.get(r["strategy"], 9),
                             r["variant"], r["symbol"]))

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
        for row in rows:
            lines.append(" ".join(fmt(row, key, spec) for key, spec, _ in COLUMNS))
        text = "\n".join(lines)

    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
