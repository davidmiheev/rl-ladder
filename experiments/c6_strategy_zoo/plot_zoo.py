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
import re
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

#: One line style per variant, so colour carries the strategy and style
#: carries what was done to it.  Dotted is the published parameters, solid the
#: widened spread, dashed and dash-dotted the two training recipes learned on
#: top of the widened spread -- which is also the order they were built in.
STYLES = {"published": ":", "wide": "-", "RL v1": "--", "RL v2": "-."}

#: Capital the equity paths are expressed as a percentage of, and the
#: notional cap the inventory paths are normalised by.  Both match
#: `strategies.py`, repeated so the plots read from `runs/` alone.
BOOK_SIZE = 100_000.0
MAX_POSITION_DOLLAR = 100_000.0


#: The split, so a baseline run over the `all` day set can be cut down to the
#: side being plotted.
SPLIT_DATE = "2025-09-15"
#: Points per session in the plotted paths.  The stored records are ~864,000
#: samples a day and a figure cannot show that; 4,000 is more than a 4-inch
#: panel has pixels.
PLOT_POINTS = 4_000
#: Fewer sessions than this and the run is a smoke test.  Three: the breadth
#: tier runs three held-out sessions.
MIN_SESSIONS = 3


def _thin(values: np.ndarray) -> np.ndarray:
    return values[:: max(1, len(values) // PLOT_POINTS)]


def collect_learned(day_set: str, symbol: str | None = None) -> list[dict]:
    """The median-seed evaluation of each (strategy, recipe) on `day_set`.

    Five seeds per recipe drawn together are thirty dashed lines on one
    panel, which reads as noise.  The seed whose cumulative return is the
    median stands in for the recipe, and the master table carries the rest.
    A policy evaluated twice (the XRP held-out set grew from four sessions to
    eleven) keeps its evaluation over the most sessions.
    """
    root = RUNS / "c6-eval"
    if not root.exists():
        return []
    best: dict[tuple[str, str, str], dict] = {}
    for path in sorted(root.iterdir()):
        summary, config, paths = path / "summary.json", path / "config.json", path / "paths.npz"
        if not (summary.exists() and config.exists() and paths.exists()):
            continue
        config_data = json.loads(config.read_text())
        if config_data["day_set"] != day_set:
            continue
        recipe = config_data.get("recipe", "v1")
        if f"RL {recipe}" not in STYLES:
            continue  # the ablations run on one pair and are reported in the writeup
        summary_data = json.loads(summary.read_text())
        if len(summary_data["per_day"]) < MIN_SESSIONS:
            continue
        if symbol is not None and summary_data["symbol"] != symbol:
            continue
        seed = re.search(r"-s(\d+)__", config_data["policy"]).group(1)
        key = (summary_data["strategy"], recipe, seed)
        if key in best and len(best[key]["per_day"]) > len(summary_data["per_day"]):
            continue
        # `summary["strategy"]` is the full preset name, e.g. "glft-wide".
        # The colour map is keyed on the base strategy, so storing the preset
        # name here made every learned curve miss the lookup and come out
        # grey -- three indistinguishable grey lines under a caption
        # promising one colour per strategy.
        best[key] = {
            "strategy": summary_data["strategy"].partition("-")[0],
            "variant": f"RL {recipe}",
            "symbol": summary_data["symbol"],
            "learned": True,
            "days": config_data["days"],
            "per_day": summary_data["per_day"],
            "paths": paths,
        }

    groups: dict[tuple[str, str], list[dict]] = {}
    for entry in best.values():
        groups.setdefault((entry["strategy"], entry["variant"]), []).append(entry)
    out = []
    for group in groups.values():
        group.sort(key=lambda e: cumulative(e)[-1])
        entry = group[(len(group) - 1) // 2]
        blob = np.load(entry.pop("paths"))
        entry["seeds"] = len(group)
        entry["equity"] = {d: blob[f"equity_{d}"] for d in entry["days"] if f"equity_{d}" in blob}
        entry["position"] = {d: blob[f"position_{d}"] for d in entry["days"]
                             if f"position_{d}" in blob}
        out.append(entry)
    return out


def collect_baselines(day_set: str, symbol: str | None = None) -> list[dict]:
    """Fixed-parameter replays from `run_baseline.py`, on the same footing.

    The runner stores each session's raw `hftbacktest` record rather than a
    pre-computed equity path, so equity is marked to market here the same way
    the environment does it -- balance plus position at the mid, less fees --
    and both sources end up on the same axis.
    """
    root = RUNS / "c6"
    if not root.exists():
        return []
    best: dict[str, tuple[Path, dict, list[dict]]] = {}
    for path in sorted(root.iterdir()):
        summary_path = path / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        if "per_day" not in summary or "strategy" not in summary:
            continue  # a training run, not a replay
        if symbol is not None and summary["symbol"] != symbol:
            continue
        keep = [d for d in summary["per_day"]
                if (d["date"] >= SPLIT_DATE) == (day_set == "test")]
        if len(keep) < MIN_SESSIONS:
            continue
        # A preset replayed twice (the XRP held-out set grew from four
        # sessions to eleven) keeps its replay over the most sessions.
        previous = best.get(summary["strategy"])
        if previous is None or len(keep) >= len(previous[2]):
            best[summary["strategy"]] = (path, summary, keep)

    out = []
    for path, summary, keep in best.values():
        equity, position = {}, {}
        for day in keep:
            record_path = path / f"record_{day['date']}.npz"
            if not record_path.exists():
                continue
            blob = np.load(record_path)
            record = blob["record"]
            marked = record["balance"] + record["position"] * record["price"] - record["fee"]
            equity[day["date"]] = _thin(marked)
            position[day["date"]] = blob["position"] * blob["mid"] / MAX_POSITION_DOLLAR
        if not equity:
            continue
        name = summary["strategy"]
        base, _, suffix = name.partition("-")
        out.append({
            "strategy": base,
            "variant": suffix or "published",
            "symbol": summary["symbol"],
            "learned": False,
            "days": [d["date"] for d in keep],
            "per_day": keep,
            "equity": equity,
            "position": position,
        })
    return out


def collect(day_set: str, symbol: str | None = None) -> list[dict]:
    """Entries for one asset, or for all of them.

    The symbol filter is applied *before* any record is opened. A session's
    record is ~55 MB and the study has nine assets, so collecting everything
    and filtering afterwards reads tens of gigabytes to draw one figure.
    """
    return collect_baselines(day_set, symbol) + collect_learned(day_set, symbol)


def symbols_with_runs() -> list[str]:
    """Which assets have anything to plot, without opening a record."""
    found = set()
    for phase in ("c6", "c6-eval"):
        root = RUNS / phase
        if not root.exists():
            continue
        for path in sorted(root.iterdir()):
            summary_path = path / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            if "symbol" in summary and "per_day" in summary:
                found.add(summary["symbol"])
    return sorted(found)


#: How each line style reads in a caption.  Derived from `STYLES` rather than
#: written beside it, because the two drifted apart the moment a third
#: variant appeared and a figure with a wrong key is worse than no key.
STYLE_WORDS = {":": "dotted", "-": "solid", "--": "dashed", "-.": "dash-dotted"}
VARIANT_WORDS = {
    "published": "the published parameters",
    "wide": "the widened spread",
    "RL v1": "TD3 recipe v1, median seed",
    "RL v2": "TD3 recipe v2 (residual), median seed",
}


def legend(entries: list[dict]) -> str:
    """A caption naming only the variants actually drawn."""
    present = [v for v in STYLES if any(e["variant"] == v for e in entries)]
    return ", ".join(
        f"{STYLE_WORDS[STYLES[v]]}: {VARIANT_WORDS.get(v, v)}" for v in present
    )


def cumulative(entry: dict) -> np.ndarray:
    """The sessions chained into one track record, in percent of book."""
    returns = np.array([d["Return"] for d in entry["per_day"]])
    return (np.cumprod(1.0 + returns) - 1.0) * 100


def label_for(entry: dict) -> str:
    total = cumulative(entry)[-1]
    count = entry.get("seeds")
    seeds = f", {count} seed{'s' if count > 1 else ''}" if count else ""
    return f"{entry['strategy']} {entry['variant']} ({total:+.2f}%{seeds})"


def plot(entries: list[dict], key: str, ylabel: str, title: str, out: Path,
         cumulative_panel: bool) -> None:
    days = sorted({d for e in entries for d in e["days"]})
    n_panels = len(days) + (1 if cumulative_panel else 0)
    cols = min(4, n_panels) if n_panels > 6 else min(3, n_panels)
    rows = int(np.ceil(n_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    flat = axes.ravel()

    for index, date in enumerate(days):
        ax = flat[index]
        for entry in entries:
            array = entry[key].get(date)
            if array is None or not len(array):
                continue
            if key == "equity":
                array = np.asarray(array) / BOOK_SIZE * 100
            hours = np.linspace(0, 24, len(array))
            ax.plot(hours, array, lw=1.0, color=COLOURS.get(entry["strategy"], "grey"),
                    ls=STYLES.get(entry["variant"], ":"), alpha=0.9)
        ax.set_title(date, fontsize=10)
        ax.axhline(0, color="grey", lw=0.5)
        if key == "position":
            # The notional cap gates *new* quotes; it does not truncate a
            # fill, so one grid's worth of overshoot past +/-1 is expected and
            # the lines mark how much of it there is.
            for level in (-1.0, 1.0):
                ax.axhline(level, color="grey", lw=0.5, ls=":")
        ax.set_xlabel("hours")
        ax.set_ylabel(ylabel)

    if cumulative_panel:
        ax = flat[len(days)]
        for entry in entries:
            chained = cumulative(entry)
            ax.plot(range(1, len(chained) + 1), chained, marker="o", ms=3, lw=1.2,
                    color=COLOURS.get(entry["strategy"], "grey"),
                    ls=STYLES.get(entry["variant"], ":"), label=label_for(entry))
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title("all sessions, chained")
        ax.set_xlabel("sessions elapsed")
        ax.set_ylabel("cumulative, % of book")

    for ax in flat[n_panels:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    if cumulative_panel:
        # Below the grid rather than inside the chained panel: with two
        # recipes and five strategies there are up to eighteen entries, and
        # inside the panel they covered the lines they label.
        handles, labels = flat[len(days)].get_legend_handles_labels()
        rows_needed = int(np.ceil(len(labels) / 4))
        fig.subplots_adjust(bottom=(0.17 * rows_needed + 0.65) / fig.get_figheight())
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, frameon=False)
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

    # One asset per figure, collected one asset at a time: the held-out
    # periods differ in length between the deep and breadth tiers, and
    # putting them on one grid would imply they are the same period.
    out = Path(args.out)
    wanted = [args.symbol] if args.symbol else symbols_with_runs()
    drawn = 0
    for symbol in wanted:
        subset = collect(args.days, symbol)
        if not subset:
            continue
        key = legend(subset)
        plot(subset, "equity", "equity, % of book",
             f"C6 {symbol}: equity on {args.days} sessions ({key})",
             out / f"pnl-{symbol}.png", cumulative_panel=True)
        plot(subset, "position", "inventory / cap",
             f"C6 {symbol}: inventory on {args.days} sessions ({key})",
             out / f"inventory-{symbol}.png", cumulative_panel=False)
        drawn += 1
    if not drawn:
        raise SystemExit(f"no finished C6 runs on the {args.days} days")


if __name__ == "__main__":
    main()
