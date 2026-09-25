"""C7's figures, drawn from `runs/c7*/` and `runs/c6/` alone.

    vol-forecast.png     QLIKE of each volatility estimator, held out
    vol-in-strategy.png  what swapping sigma did to held-out returns
    markout.png          C6's widened baseline against the markout rule
    book-info.png        held-out R^2 each feature group adds

    python experiments/c7_signals/plot_c7.py [--only NAME]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
OUT = REPO / "docs" / "assets" / "c7"
SYMBOLS = ("ethusdc", "xrpusdc")

# Categorical slots in fixed order (validated: lightness, chroma, CVD
# separation); ink and grid stay neutral so colour only ever means a series.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e4e3df"

plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
})


def c6_reference(strategy: str, symbol: str, days: list[str]) -> list[dict] | None:
    for path in sorted((RUNS / "c6").glob(f"base-{strategy}-{symbol}__*"), reverse=True):
        summary = path / "summary.json"
        if summary.exists():
            per_day = {d["date"]: d for d in json.loads(summary.read_text())["per_day"]}
            if all(d in per_day for d in days):
                return [per_day[d] for d in days]
    return None


def c7_run(tag: str) -> list[dict] | None:
    paths = sorted((RUNS / "c7").glob(f"{tag}__*/summary.json"))
    return json.loads(paths[-1].read_text())["per_day"] if paths else None


def chained(rows: list[dict]) -> float:
    return (np.prod([1.0 + r["Return"] for r in rows]) - 1.0) * 100


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT / name}")


def vol_forecast() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharex=False)
    for ax, symbol in zip(axes, SYMBOLS, strict=True):
        rows = json.loads((RUNS / "c7-vol" / f"{symbol}.json").read_text())
        rows = rows["results"]["y1"]["rows"]
        names = [n for n in rows if n != "rv_1s_1m"]  # off the scale; in the table
        names.sort(key=lambda n: rows[n]["qlike_cal"], reverse=True)
        values = [rows[n]["qlike_cal"] for n in names]
        colours = [ORANGE if n == "current" else BLUE for n in names]
        ax.barh(names, values, color=colours, height=0.6)
        for y, v in enumerate(values):
            ax.text(v, y, f" {v:.3f}", va="center", fontsize=7, color=INK)
        ax.set_title(symbol.upper())
        ax.set_xlabel("QLIKE, held out, bias-corrected (lower is better)")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(values) * 1.25)
    fig.suptitle("Forecasting the next five minutes' variance: C6's estimator (orange) "
                 "against ten others", fontsize=10)
    save(fig, "vol-forecast.png")


def vol_in_strategy() -> None:
    labels, series = [], {"C6 sigma": [], "bias-corrected": [], "HAR": []}
    for symbol in SYMBOLS:
        for strategy in ("glft", "glft-wide", "hfgrid", "hfgrid-wide"):
            scale, har = c7_run(f"{strategy}-{symbol}-volscale"), c7_run(f"{strategy}-{symbol}-har")
            if scale is None or har is None:
                continue
            ref = c6_reference(strategy, symbol, [d["date"] for d in har])
            labels.append(f"{strategy}\n{symbol[:3].upper()}")
            series["C6 sigma"].append(chained(ref))
            series["bias-corrected"].append(chained(scale))
            series["HAR"].append(chained(har))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    for i, (name, colour) in enumerate(zip(series, (BLUE, ORANGE, AQUA), strict=True)):
        ax.bar(x + (i - 1) * 0.27, series[name], width=0.25, color=colour, label=name)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("held-out cumulative return, % of book")
    ax.legend(ncol=3, loc="lower right")
    ax.set_title("Swapping sigma inside the strategies (11 held-out sessions per asset)")
    ax.grid(axis="x", visible=False)
    save(fig, "vol-in-strategy.png")


def markout() -> None:
    rows = []
    for path in sorted((RUNS / "c7").glob("*-mk[0-9]__*/summary.json")):
        tag = path.parent.name.split("__")[0]
        strategy, symbol, kappa = tag.rsplit("-", 2)
        test = json.loads(path.read_text())["per_day"]
        ref = c6_reference(strategy, symbol, [d["date"] for d in test])
        rows.append((f"{strategy} {symbol[:3].upper()}  (kappa {kappa[2:]})",
                     chained(ref), chained(test)))
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    for y, (_, before, after) in enumerate(rows):
        ax.plot([before, after], [y, y], color=GRID, lw=3, zorder=1)
        ax.scatter([before], [y], s=46, color=BLUE, zorder=2,
                   label="C6 widened baseline" if y == 0 else None)
        ax.scatter([after], [y], s=46, color=ORANGE, zorder=2,
                   label="+ markout widening" if y == 0 else None)
        ax.text(after, y + 0.22, f"{after - before:+.2f}", ha="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("held-out cumulative return, % of book")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.legend(loc="lower right")
    ax.set_title("The markout rule on held-out sessions, kappa chosen on four training days")
    ax.grid(axis="y", visible=False)
    save(fig, "markout.png")


def book_info() -> None:
    groups = ("c2", "levels", "bands", "ofi", "returns", "btc")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, symbol in zip(axes, SYMBOLS, strict=True):
        nested = json.loads((RUNS / "c7-book" / f"{symbol}.json").read_text())["nested"]
        x = np.arange(len(groups))
        for i, (h, colour) in enumerate(zip(("1", "10"), (BLUE, ORANGE), strict=True)):
            prev, added = 0.0, []
            for g in groups:
                added.append(nested[g][h] - prev)
                prev = nested[g][h]
            ax.bar(x + (i - 0.5) * 0.36, added, width=0.34, color=colour, label=f"{h} s ahead")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x, ["C2 (in the\nRL state)", "deeper\nlevels", "bp\nbands", "OFI",
                          "own\nreturns", "BTCUSDT"], fontsize=7.5)
        ax.set_title(symbol.upper())
        ax.grid(axis="x", visible=False)
        ax.set_yscale("symlog", linthresh=0.2)
    axes[0].set_ylabel("held-out R$^2$ added, % (symlog)")
    axes[0].legend()
    fig.suptitle("What each feature group adds, in order, to a linear forecast of the "
                 "mid (held out)", fontsize=10)
    save(fig, "book-info.png")


FIGURES = {"vol-forecast": vol_forecast, "vol-in-strategy": vol_in_strategy,
           "markout": markout, "book-info": book_info}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=sorted(FIGURES))
    args = parser.parse_args()
    for name, draw in FIGURES.items():
        if args.only in (None, name):
            draw()


if __name__ == "__main__":
    main()
