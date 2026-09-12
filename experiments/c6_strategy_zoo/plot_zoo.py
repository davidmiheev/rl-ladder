"""C6's figures: equity and inventory, per strategy, over the held-out sessions.

    python experiments/c6_strategy_zoo/plot_zoo.py --days test --out docs/assets/c6

Two figures, because they answer different questions and overlaying them
would let each hide the other's failure:

`pnl.png`        cumulative equity, as a percentage of book, session by
                 session and then chained into one track record.  Baselines
                 solid, learned policies dashed, one colour per strategy.
`inventory.png`  the position path over the same sessions, normalised by the
                 cap.  A market maker that ends every session flat and one
                 that quietly accumulates a directional position can produce
                 the same equity curve on a trending day; only this plot
                 separates them, and only this plot shows whether RL bought
                 its return by carrying more risk.

The inventory panel is the one C3 should have had.  Its learned policies were
compared on return and fills alone, so "the policy earned more" and "the
policy held three times the inventory" were not distinguishable in anything
that got published.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RUNS = HERE.parents[1] / "runs"

#: One colour per strategy, so a reader can follow a strategy across both
#: figures and across the baseline/learned pair within each.
COLOURS = {
    "glft": "tab:blue",
    "hfgrid": "tab:orange",
    "apt": "tab:green",
    "basis": "tab:red",
    "obi": "tab:purple",
}


def collect(day_set: str) -> list[dict]:
    """Every finished evaluation run on `day_set`, with its stored paths."""
    root = RUNS / "c6-eval"
    if not root.exists():
        return []
    out = []
    for path in sorted(root.iterdir()):
        summary, config, paths = path / "summary.json", path / "config.json", path / "paths.npz"
        if not (summary.exists() and config.exists() and paths.exists()):
            continue
        config_data = json.loads(config.read_text())
        if config_data["day_set"] != day_set:
            continue
        summary_data = json.loads(summary.read_text())
        out.append({
            "dir": path,
            "strategy": summary_data["strategy"],
            "symbol": summary_data["symbol"],
            "learned": config_data["policy"] != "baseline",
            "days": config_data["days"],
            "per_day": summary_data["per_day"],
            "period": summary_data["period"],
            "paths": np.load(paths),
        })
    return out


def label_for(entry: dict) -> str:
    kind = "TD3" if entry["learned"] else "baseline"
    total = entry["period"]["cumulative_return"] * 100
    return f"{entry['strategy']} {kind} ({total:+.2f}%)"


def plot(entries: list[dict], key: str, ylabel: str, title: str, out: Path,
         cumulative_panel: bool) -> None:
    days = sorted({d for e in entries for d in e["days"]})
    n_panels = len(days) + (1 if cumulative_panel else 0)
    cols = min(4, n_panels)
    rows = int(np.ceil(n_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    flat = axes.ravel()

    for index, date in enumerate(days):
        ax = flat[index]
        for entry in entries:
            array = entry["paths"].get(f"{key}_{date}")
            if array is None:
                continue
            hours = np.linspace(0, 24, len(array))
            ax.plot(hours, array, lw=1.0, color=COLOURS.get(entry["strategy"], "grey"),
                    ls="--" if entry["learned"] else "-", alpha=0.9)
        ax.set_title(date, fontsize=10)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel("hours")
        ax.set_ylabel(ylabel)

    if cumulative_panel:
        ax = flat[len(days)]
        for entry in entries:
            returns = np.array([d["Return"] for d in entry["per_day"]])
            chained = (np.cumprod(1.0 + returns) - 1.0) * 100
            ax.plot(range(1, len(chained) + 1), chained, marker="o", ms=3, lw=1.2,
                    color=COLOURS.get(entry["strategy"], "grey"),
                    ls="--" if entry["learned"] else "-", label=label_for(entry))
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title("all sessions, chained")
        ax.set_xlabel("sessions elapsed")
        ax.set_ylabel("cumulative, % of book")
        ax.legend(fontsize=6, loc="best")

    for ax in flat[n_panels:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", default="test", choices=["test", "train"])
    parser.add_argument("--symbol", default=None, help="restrict to one asset")
    parser.add_argument("--out", default="docs/assets/c6")
    args = parser.parse_args()

    entries = collect(args.days)
    if args.symbol:
        entries = [e for e in entries if e["symbol"] == args.symbol]
    if not entries:
        raise SystemExit(f"no finished C6 evaluations on the {args.days} days")

    # One asset per figure: ETHUSDC has eleven held-out sessions and XRPUSDC
    # four, and putting them on one grid would imply they are the same period.
    out = Path(args.out)
    for symbol in sorted({e["symbol"] for e in entries}):
        subset = [e for e in entries if e["symbol"] == symbol]
        suffix = "" if len({e["symbol"] for e in entries}) == 1 else f"-{symbol}"
        plot(subset, "equity", "equity, % of book",
             f"C6 {symbol}: equity on {args.days} sessions "
             f"(solid: published parameters, dashed: TD3 on the multipliers)",
             out / f"pnl{suffix}.png", cumulative_panel=True)
        plot(subset, "position", "position, units",
             f"C6 {symbol}: inventory on {args.days} sessions "
             f"(solid: published parameters, dashed: TD3 on the multipliers)",
             out / f"inventory{suffix}.png", cumulative_panel=False)


if __name__ == "__main__":
    main()
