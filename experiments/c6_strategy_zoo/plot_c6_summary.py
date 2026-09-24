"""C6's summary figures: the ones the writeup's claims rest on.

    python experiments/c6_strategy_zoo/plot_c6_summary.py --out docs/assets/c6

Each figure is drawn only if its data exists, and says so if not:

  spread-sweep.png        cumulative return against the half-spread floor, per
                          asset, on the training sessions the floor was chosen
                          on -- the curve behind every number in section 8.1
  floor-vs-intensity.png  the chosen floor against the tick size and against
                          the intensity decay 1/k, both in bp
  adverse-selection.png   what a fill costs on the tape, against what a quote
                          at that depth earns
  grid-depth.png          return against the number of grid orders, at the
                          published and the widened spread
  widening-and-rl.png     every (asset, strategy) pair: published baseline,
                          widened baseline, and each RL recipe, one seed a dot
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

import assets  # noqa: E402
import master_table  # noqa: E402

RUNS = HERE.parents[1] / "runs"
COLOURS = {"glft": "tab:blue", "hfgrid": "tab:orange", "apt": "tab:green",
           "basis": "tab:red", "obi": "tab:purple"}


def newest(phase: str, prefix: str) -> dict | None:
    hits = sorted(p for p in (RUNS / phase).glob(f"{prefix}__*") if (p / "summary.json").exists())
    return json.loads((hits[-1] / "summary.json").read_text()) if hits else None


def spread_sweep(out: Path) -> None:
    floors = json.loads((HERE / "spread_floor.json").read_text())
    symbols = [s for s in assets.SINGLE_LEG if newest("c6-sweep", f"half_spread_bp-glft-{s}")]
    cols = 3
    rows = int(np.ceil(len(symbols) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 3.2 * rows), squeeze=False)
    for ax, symbol in zip(axes.ravel(), symbols, strict=False):
        summary = newest("c6-sweep", f"half_spread_bp-glft-{symbol}")
        values = [r["half_spread_bp"] for r in summary["rows"]]
        returns = [r["cumulative_return"] * 100 for r in summary["rows"]]
        fills = [r["total_fills"] for r in summary["rows"]]
        days = len(summary["rows"][0]["per_day_return"])
        x = [max(v, 0.5) for v in values]  # 0 means "model's own", drawn at 0.5 on a log axis
        trading = [f >= 50 * days for f in fills]
        ax.plot(x, returns, color="0.6", lw=1)
        x, returns, trading = np.array(x), np.array(returns), np.array(trading)
        ax.scatter(x[trading], returns[trading],
                   color="tab:blue", zorder=3, label="takes >= 50 fills/day")
        ax.scatter(x[~trading], returns[~trading],
                   facecolor="white", edgecolor="tab:blue", zorder=3, label="barely trades (excluded)")
        chosen = floors.get(symbol)
        if chosen is not None:
            ax.axvline(chosen, color="tab:red", ls="--", lw=1)
            ax.text(chosen, ax.get_ylim()[1], f" floor {chosen:g} bp", color="tab:red",
                    va="top", fontsize=8)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.5, 1, 2, 4, 8, 16, 32])
        ax.set_xticklabels(["model", "1", "2", "4", "8", "16", "32"])
        ax.set_title(f"{symbol} ({days} training sessions)", fontsize=10)
        ax.set_xlabel("half-spread floor, bp")
        ax.set_ylabel("cumulative return, %")
    for ax in axes.ravel()[len(symbols):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("GLFT with a half-spread floor, on training sessions: the sweep that set each floor")
    fig.tight_layout()
    fig.savefig(out / "spread-sweep.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out / 'spread-sweep.png'}")


def floor_vs_intensity(out: Path) -> None:
    floors = json.loads((HERE / "spread_floor.json").read_text())
    rows = [r for r in master_table.baseline_rows()
            if r["strategy"] == "glft" and r["spread"] == "published" and r["days"] == "train"]
    inv_k = {r["symbol"]: r["inv_k_bp"] for r in rows}
    cached = json.loads((HERE / "assets.json").read_text())
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    points = []
    for symbol, floor in floors.items():
        entry = cached[symbol]
        tick_bp = entry["tick_size"] / entry["price"] * 1e4
        points.append((tick_bp, floor, symbol))
        ax.scatter(tick_bp, floor, color="tab:blue", zorder=3)
        if symbol in inv_k and np.isfinite(inv_k[symbol]):
            ax.scatter(tick_bp, inv_k[symbol], color="tab:orange", marker="s", zorder=3)
            ax.plot([tick_bp, tick_bp], [inv_k[symbol], floor], color="0.8", lw=1, zorder=1)
    # Assets that land on the same point (the same underlying in two
    # stablecoins, often) share one label; near neighbours alternate sides.
    points.sort()
    labels: list[tuple[float, float, str]] = []
    for tick_bp, floor, symbol in points:
        if labels and labels[-1][1] == floor and abs(np.log(tick_bp / labels[-1][0])) < 0.05:
            labels[-1] = (labels[-1][0], floor, labels[-1][2] + " / " + symbol)
        else:
            labels.append((tick_bp, floor, symbol))
    for i, (tick_bp, floor, text) in enumerate(labels):
        crowded = i > 0 and labels[i - 1][1] == floor and abs(np.log(tick_bp / labels[i - 1][0])) < 0.4
        ax.annotate(text, (tick_bp, floor), textcoords="offset points",
                    xytext=(5, -12 if crowded else 4), fontsize=8)
    ax.scatter([], [], color="tab:blue", label="chosen half-spread floor")
    ax.scatter([], [], color="tab:orange", marker="s", label="1/k, where arrivals fall by e")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("tick size, bp of price")
    ax.set_ylabel("bp of price")
    ax.set_title("Each asset's spread floor sits far outside its intensity decay 1/k")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "floor-vs-intensity.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out / 'floor-vs-intensity.png'}")


def adverse_selection(out: Path) -> None:
    path = RUNS / "c6-adverse" / "ethusdc.json"
    if not path.exists():
        print("skip adverse-selection.png: run measure_adverse_selection.py first")
        return
    data = json.loads(path.read_text())
    depth = np.array(data["depth_bp"])
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(depth, data["cost_bp"], marker="o", color="tab:red", label="cost of a fill, volume weighted")
    ax.plot(depth, data["cost_unweighted_bp"], marker="o", color="tab:red", alpha=0.4, ls=":",
            label="cost of a fill, unweighted")
    ax.plot(depth, depth + data["rebate_bp"], color="tab:blue", label="what the quote earns: depth + rebate")
    ax.set_xscale("log")
    ax.set_xlabel("quote depth from the mid, bp")
    ax.set_ylabel("bp per fill")
    ax.set_title(f"ETHUSDC on the tape, {data['horizon_s']:g} s holding horizon: "
                 f"where earning outruns cost")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "adverse-selection.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out / 'adverse-selection.png'}")


def grid_depth(out: Path) -> None:
    panels = [(p, newest("c6-sweep", p)) for p in
              ("grid_num-glft-xrpusdc", "grid_num-glft-wide-xrpusdc", "grid_num-glft-wide-ethusdc")]
    panels = [(p, s) for p, s in panels if s]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.4), squeeze=False)
    for ax, (name, summary) in zip(axes.ravel(), panels, strict=True):
        n = [r["grid_num"] for r in summary["rows"]]
        ret = [r["cumulative_return"] * 100 for r in summary["rows"]]
        fills = [r["total_fills"] for r in summary["rows"]]
        ax.plot(n, ret, marker="o")
        for x, y, f in zip(n, ret, fills, strict=True):
            ax.annotate(f"{f:,}", (x, y), textcoords="offset points", xytext=(0, 6), fontsize=7,
                        ha="center")
        ax.set_title(name.replace("grid_num-", "").replace("-", " "), fontsize=10)
        ax.set_xlabel("orders per side")
        ax.set_ylabel("cumulative return, %")
    fig.suptitle("Orders per side, on training sessions (labels: total fills). "
                 "Past about 20 the extra orders never fill.")
    fig.tight_layout()
    fig.savefig(out / "grid-depth.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out / 'grid-depth.png'}")


def widening_and_rl(out: Path) -> None:
    rows = [r for r in master_table.baseline_rows() + master_table.eval_rows() if r["days"] == "test"]
    pairs = sorted({(r["symbol"], r["strategy"]) for r in rows},
                   key=lambda p: (p[0] not in assets.DEEP, p[0], p[1]))
    recipes = sorted({r["recipe"] for r in rows if r["policy"] != "baseline"})
    fig, ax = plt.subplots(figsize=(11, 0.34 * len(pairs) + 1.6))
    offsets = {"published": -0.3, "wide": -0.1}
    offsets |= {rec: 0.1 + 0.2 * i for i, rec in enumerate(recipes)}
    markers = {"published": ("x", "0.45"), "wide": ("D", "black")}
    rec_colours = ["tab:orange", "tab:green", "tab:purple", "tab:brown"]
    for i, (symbol, strategy) in enumerate(pairs):
        for r in rows:
            if (r["symbol"], r["strategy"]) != (symbol, strategy):
                continue
            if r["policy"] == "baseline":
                marker, colour = markers[r["spread"]]
                ax.scatter(r["cumulative"], i + offsets[r["spread"]], marker=marker, color=colour,
                           s=26, zorder=3)
            elif r["spread"] == "wide":
                j = recipes.index(r["recipe"])
                ax.scatter(r["cumulative"], i + offsets[r["recipe"]], color=rec_colours[j % 4],
                           s=14, alpha=0.8, zorder=2)
    ax.set_yticks(range(len(pairs)))
    sessions = {(r["symbol"], r["strategy"]): r["n"] for r in rows}
    ax.set_yticklabels([f"{s} {st}  (n={sessions[(s, st)]})" for s, st in pairs], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="0.7", lw=0.8)
    ax.scatter([], [], marker="x", color="0.45", label="published baseline")
    ax.scatter([], [], marker="D", color="black", label="widened baseline")
    for j, rec in enumerate(recipes):
        ax.scatter([], [], color=rec_colours[j % 4], s=14, label=f"TD3 {rec}, one dot per seed")
    ax.set_xlabel("cumulative return over the held-out sessions, % of book")
    ax.set_title("Held out: widening the spread against learning on top of it (n = held-out sessions)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "widening-and-rl.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out / 'widening-and-rl.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/assets/c6")
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figures = {"spread-sweep": spread_sweep, "floor-vs-intensity": floor_vs_intensity,
               "adverse-selection": adverse_selection, "grid-depth": grid_depth,
               "widening-and-rl": widening_and_rl}
    for name, draw in figures.items():
        if args.only is None or args.only == name:
            draw(out)


if __name__ == "__main__":
    main()
