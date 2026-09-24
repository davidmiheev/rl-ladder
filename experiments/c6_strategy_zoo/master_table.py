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
- `1/k` is the depth at which arrivals fall by a factor of e.  C1's proof
  that `inf_gamma H(gamma) = 1/k` means no amount of risk aversion lets a GLFT
  quoter sit inside it.  **Units:** the notebook's intensity histogram buckets
  depth in *half*-ticks (a factor-of-two slip it documents and keeps), so the
  fitted `k` is per half-tick.  The table converts once, here: `k` is shown
  per tick and `1/k` in ticks and in bp.  An earlier version printed the raw
  half-tick value under a "1/k t" heading, which overstated the depth by 2x.
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

import assets  # noqa: E402

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
    ("k", "{:>8}", "k/tick"),
    ("inv_k", "{:>8}", "1/k tick"),
    ("inv_k_bp", "{:>7}", "1/k bp"),
    ("fills_h", "{:>8}", "fills/h"),
    ("abs_q", "{:>7}", "|q|"),
    ("signal_bp", "{:>8}", "sig bp"),
]


def finished(phase: str):
    """Every run directory under `phase` that has a summary, newest per name.

    Run directories are `<name>__<timestamp>`, so a job that was killed and
    re-run leaves more than one. Only the newest of each name is reported:
    otherwise a re-run silently doubles its row, and two rows that differ
    only by when they were run look like two measurements.
    """
    root = RUNS / phase
    if not root.exists():
        return []
    newest: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if (path / "summary.json").exists() and (path / "config.json").exists():
            newest[path.name.split("__")[0]] = path  # sorted, so the last wins
    return [(path, json.loads((path / "config.json").read_text()),
             json.loads((path / "summary.json").read_text()))
            for path in newest.values()]


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

#: A run over fewer sessions than this is a smoke test, not a result.  Three,
#: because the breadth tier is three training and three held-out sessions by
#: design; the session count is a column, so a three-session row cannot be
#: mistaken for an eleven-session one.
MIN_SESSIONS = 3


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


def _intensity_in_ticks(row: dict, symbol: str) -> dict:
    """Convert the fitted intensity from half-tick buckets to ticks and bp.

    `fit_intensity` regresses on bucket indices that are half-ticks apart, so
    its `k` is per half-tick: per tick it is twice that, and `1/k` in ticks is
    half the raw value.  Doing it here rather than in the quoter keeps the
    quoter bit-identical to the notebook it transcribes.
    """
    tick_bp = None
    entry = assets.ensure_registered(symbol)
    if entry.get("price"):
        tick_bp = entry["tick_size"] / entry["price"] * 1e4
    row["k"] = row["k"] * 2.0
    row["inv_k"] = row["inv_k"] / 2.0
    row["inv_k_bp"] = row["inv_k"] * tick_bp if tick_bp else float("nan")
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
    "abs_q": "median_abs_norm_pos",
    "signal_bp": "median_abs_signal_bp",
}


def _backfill_inventory(per_day: list[dict], symbol: str) -> None:
    """Give older runs the normalised inventory they were written without.

    The baseline runner used to report inventory in units of the asset while
    the evaluator reported it normalised by the notional cap, so the two
    could not go in one column. Runs made since record the normalised figure
    directly; these few are converted with the asset's median price, which is
    exact to within the session's own price range.
    """
    if not per_day or "median_abs_norm_pos" in per_day[0]:
        return
    price = assets.ensure_registered(symbol)["price"]
    for day in per_day:
        day["median_abs_norm_pos"] = day["median_abs_position"] * price / MAX_POSITION_DOLLAR


#: Matches `strategies.MAX_POSITION_DOLLAR`.
MAX_POSITION_DOLLAR = 100_000.0


def baseline_rows() -> list[dict]:
    """Rows from `run_baseline.py`, which replays fixed parameters."""
    rows = []
    for path, config, summary in finished("c6"):
        if "per_day" not in summary or "strategy" not in summary:
            continue  # a training run, not a baseline replay
        del config
        name = summary["strategy"]
        base, _, suffix = name.partition("-")
        _backfill_inventory(summary["per_day"], summary["symbol"])
        for day_set, block in _split(summary["per_day"]):
            rows.append(_intensity_in_ticks({
                "strategy": base,
                "recipe": "baseline",
                "policy": "baseline",
                "spread": "wide" if suffix == "wide" else "published",
                "symbol": summary["symbol"],
                "days": day_set,
                **_summarise(block, BASELINE_FIELDS),
                "_dir": path.name,
            }, summary["symbol"]))
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
        # Evaluations written before training recipes existed are v1 by
        # construction: that was the only recipe.
        recipe = config.get("recipe", "v1") if learned else "baseline"
        rows.append(_intensity_in_ticks({
            "strategy": base,
            "recipe": recipe,
            "policy": (f"RL {recipe} s{seed}" if seed.isdigit() else f"RL {recipe}") if learned
            else "baseline",
            "spread": "wide" if suffix == "wide" else "published",
            "symbol": summary["symbol"],
            "days": config["day_set"],
            **_summarise(per_day, EVAL_FIELDS),
            "_dir": path.name,
        }, summary["symbol"]))
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
        elif key in ("A", "inv_k", "inv_k_bp", "fills_h", "abs_q", "half_spread_bp", "signal_bp",
                     "half_bp", "fills_pub", "fills_wide", "fills_rl", "q_wide", "q_rl"):
            text = f"{value:.2f}"
        elif key in ("cumulative", "worst", "median_return", "published", "wide", "rl",
                     "spread_gain", "rl_gain"):
            text = f"{value:+.2f}"
        else:
            text = f"{value:.2f}"
    else:
        text = str(value)
    return spec.format(text)


#: The comparison view's columns: one row per (asset, strategy), with the
#: three configurations side by side and the thing the phase is actually
#: asking about -- what RL added on top of an already-widened spread -- as
#: its own column rather than a subtraction the reader has to do.
COMPARE = [
    ("symbol", "{:<9}", "symbol"),
    ("strategy", "{:<8}", "strategy"),
    ("n", "{:>3}", "n"),
    ("published", "{:>10}", "published"),
    ("wide", "{:>9}", "wide"),
    ("rl", "{:>9}", "RL"),
    ("seeds", "{:>6}", "seeds"),
    ("spread_gain", "{:>9}", "d wide"),
    ("rl_gain", "{:>9}", "d RL"),
    ("half_bp", "{:>8}", "half bp"),
    ("fills_pub", "{:>9}", "f/h pub"),
    ("fills_wide", "{:>9}", "f/h wide"),
    ("fills_rl", "{:>9}", "f/h RL"),
    ("q_wide", "{:>8}", "|q| wide"),
    ("q_rl", "{:>8}", "|q| RL"),
]


def compare_rows(rows: list[dict]) -> list[dict]:
    """Collapse the long table to one row per (asset, strategy).

    Learned policies are summarised by the **median over seeds**, and the
    seed count is printed next to it: C3 measured a 3.74-point spread between
    the best and worst seed on this problem, so a single-seed number is not a
    number, and a reader has to be able to see how many went into it.
    """
    groups: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["symbol"], row["strategy"])
        entry = groups.setdefault(key, {"symbol": row["symbol"], "strategy": row["strategy"],
                                        "n": row["n"], "rl_values": [], "rl_fills": [],
                                        "rl_q": []})
        if row["policy"] == "baseline":
            slot = "published" if row["spread"] == "published" else "wide"
            entry[slot] = row["cumulative"]
            entry[f"fills_{'pub' if slot == 'published' else 'wide'}"] = row["fills_h"]
            if slot == "wide":
                entry["half_bp"] = row["half_spread_bp"]
                entry["q_wide"] = row["abs_q"]
            entry.setdefault("half_bp", row["half_spread_bp"])
        elif row["spread"] == "wide":
            # Only policies trained on the widened baseline belong in this
            # column: the whole question is what RL adds *on top of* a
            # sensible spread, so a policy trained on the published variant
            # is answering a different one and would be averaged in silently.
            entry["rl_values"].append(row["cumulative"])
            entry["rl_fills"].append(row["fills_h"])
            entry["rl_q"].append(row["abs_q"])

    out = []
    for entry in groups.values():
        rl = entry.pop("rl_values")
        entry["seeds"] = len(rl)
        entry["rl"] = float(np.median(rl)) if rl else float("nan")
        entry["fills_rl"] = float(np.median(entry.pop("rl_fills"))) if rl else float("nan")
        entry["q_rl"] = float(np.median(entry.pop("rl_q"))) if rl else float("nan")
        published, wide = entry.get("published"), entry.get("wide")
        entry["spread_gain"] = (wide - published) if published is not None and wide is not None \
            else float("nan")
        entry["rl_gain"] = (entry["rl"] - wide) if wide is not None and rl else float("nan")
        out.append(entry)
    order = {"glft": 0, "hfgrid": 1, "apt": 2, "basis": 3, "obi": 4}
    out.sort(key=lambda r: (r["symbol"], order.get(r["strategy"], 9)))
    return out


def render(rows: list[dict], columns, markdown: bool, group_by=None) -> str:
    if markdown:
        # A pipe inside a cell ends the cell, and two of these columns are
        # named |q|.  Escaping is the header's problem, not the reader's.
        head = "| " + " | ".join(label.replace("|", "\\|") for _, _, label in columns) + " |"
        rule = "| " + " | ".join("---" for _ in columns) + " |"
        lines = [head, rule]
        for row in rows:
            lines.append("| " + " | ".join(fmt(row, key, "{}").strip()
                                           for key, _, _ in columns) + " |")
        return "\n".join(lines)
    header = " ".join(spec.format(label) for _, spec, label in columns)
    lines = [header, "-" * len(header)]
    previous = None
    for row in rows:
        group = group_by(row) if group_by else None
        if group_by and previous is not None and group != previous:
            lines.append("")
        previous = group
        lines.append(" ".join(fmt(row, key, spec) for key, spec, _ in columns))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--compare", action="store_true",
                        help="one row per asset and strategy, baseline against RL")
    parser.add_argument("--out", default=None)
    parser.add_argument("--days", default="test", choices=["train", "test", "both"])
    args = parser.parse_args()
    if args.days == "both":
        args.days = None

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

    if args.compare:
        text = render(compare_rows(rows), COMPARE, args.markdown,
                      group_by=lambda r: r["symbol"])
    else:
        text = render(rows, COLUMNS, args.markdown,
                      group_by=lambda r: (r["days"], r["symbol"], r["strategy"]))

    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
