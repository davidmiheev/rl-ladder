"""C4's figures and its held-out comparison table, from `runs/` alone.

    grid-heatmaps.png   training-session return over floor x kappa, per pair
    heldout.png         every method on every pair, held out
    train-vs-test.png   the overfitting lens: training return against held out
    search-curves.png   GRPO-style against PPO-style search, per iteration
    frontier.png        PnL against fills over the grid: what volume costs

    python experiments/c4_ppo_grpo/plot_c4.py [--only NAME] [--table]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments" / "c6_strategy_zoo"))

import assets  # noqa: E402
import grid_search  # noqa: E402

RUNS = REPO / "runs"
OUT = REPO / "docs" / "assets" / "c4"

# Categorical slots in fixed order (validated for lightness, chroma and CVD
# separation); ink and grid neutral. Diverging: blue (loss) to orange (gain)
# through a neutral grey, never a hue at zero.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e4e3df"
plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 9.5, "legend.frameon": False,
})


def chained(per_day: list[dict]) -> float:
    return (np.prod([1.0 + d["Return"] for d in per_day]) - 1.0) * 100


def newest(root: Path, prefix: str, days: list[str] | None = None) -> list[dict] | None:
    for path in sorted(root.glob(f"{prefix}__*"), reverse=True):
        summary = path / "summary.json"
        if not summary.exists():
            continue
        per_day = json.loads(summary.read_text())["per_day"]
        if days is None:
            return per_day
        by_date = {d["date"]: d for d in per_day}
        if all(d in by_date for d in days):
            return [by_date[d] for d in days]
    return None


def c6_wide(strategy: str, symbol: str, days: list[str]) -> list[dict] | None:
    return newest(RUNS / "c6", f"base-{strategy}-wide-{symbol}", days)


def c7_markout(strategy: str, symbol: str) -> list[dict] | None:
    """C7's markout rule at the kappa C7 chose (four training sessions)."""
    sys.path.insert(0, str(REPO / "experiments" / "c7_signals"))
    import run_variants

    if strategy not in ("glft", "hfgrid", "obi"):
        return None
    pick = run_variants.best_kappa(f"{strategy}-wide", symbol)
    if pick is None or pick[0] == 0.0:
        return None
    return newest(RUNS / "c7", f"{strategy}-wide-{symbol}-mk{pick[0]:g}")


def c6_rl_median(strategy: str, symbol: str, recipe: str) -> float | None:
    sys.path.insert(0, str(REPO / "experiments" / "c6_strategy_zoo"))
    import master_table

    values = [r["cumulative"] for r in master_table.eval_rows()
              if r["days"] == "test" and r["symbol"] == symbol and r["strategy"] == strategy
              and r["spread"] == "wide" and r["recipe"] == recipe]
    return float(np.median(values)) if len(values) >= 3 else None


def search_choice(method: str, strategy: str, symbol: str) -> float | None:
    """Median over seeds of each search's final point, held out."""
    values = []
    for seed in range(10):
        test = newest(RUNS / "c4", f"{method}-{strategy}-{symbol}-s{seed}-test")
        if test:
            values.append(chained(test))
    return float(np.median(values)) if values else None


def ex_crash(per_day: list[dict]) -> float:
    """Chained return without 2025-09-22, the one session a crash dominates."""
    return chained([d for d in per_day if d["date"] != "2025-09-22"])


def grid_row(row: dict, pick: dict, strategy: str, symbol: str, days: list[str], key: str):
    f, k, s = pick["point"]
    row[f"{key}_point"] = f"×{f:g} / {k:g} / ×{s:g}"
    row[f"{key}_train"] = pick["chained"]
    row[f"{key}_eligible"] = pick["eligible"]
    test = newest(RUNS / "c4", grid_search.tag(strategy, symbol, f, k, s) + "-test", days)
    if test:
        row[key] = chained(test)
        row[f"{key}_ex"] = ex_crash(test)
        row[f"{key}_up"] = sum(d["Return"] > 0 for d in test)
        row[f"{key}_fills"] = float(np.median([d["DailyNumberOfTrades"] for d in test]))


def comparison() -> list[dict]:
    rows = []
    chosen, chosen20 = grid_search.select("c6"), grid_search.select("train20")
    for strategy, symbol in grid_search.PAIRS:
        days = assets.test_days(symbol)
        row = {"strategy": strategy, "symbol": symbol, "n": len(days)}
        wide = c6_wide(strategy, symbol, days)
        row["wide"] = chained(wide) if wide else None
        markout = c7_markout(strategy, symbol)
        row["markout"] = chained(markout) if markout else None
        row["markout_ex"] = ex_crash(markout) if markout else None
        row["rl_v2"] = c6_rl_median(strategy, symbol, "v2")
        row["rl_edge"] = c6_rl_median(strategy, symbol, "v2-edge")
        if (strategy, symbol) in chosen:
            grid_row(row, chosen[(strategy, symbol)], strategy, symbol, days, "grid")
        if (strategy, symbol) in chosen20:
            grid_row(row, chosen20[(strategy, symbol)], strategy, symbol, days, "grid20")
        for method in ("grpo", "ppo"):
            row[method] = search_choice(method, strategy, symbol)
        rows.append(row)
    return rows


def fmt(value, spec="{:+.2f}") -> str:
    return "–" if value is None else spec.format(value)


def table() -> str:
    """Held-out cumulative return (%) of every method on every pair."""
    head = ("| asset | strategy | C6 widened | C6 RL v2 | C7 markout | C4 grid (C6 days) | "
            "**C4 grid (20 days)** | C4 GRPO | C4 PPO | RL v2-edge |")
    lines = [head, "| " + " | ".join(["---"] * 10) + " |"]
    for r in comparison():
        lines.append("| " + " | ".join([
            r["symbol"][:3].upper(), r["strategy"], fmt(r["wide"]), fmt(r["rl_v2"]),
            fmt(r["markout"]), fmt(r.get("grid")), f"**{fmt(r.get('grid20'))}**",
            fmt(r["grpo"]), fmt(r["ppo"]), fmt(r["rl_edge"]),
        ]) + " |")
    return "\n".join(lines)


def grid20_table() -> str:
    """The 20-day grid choices in detail: point, training, held out, and the crash day."""
    head = ("| asset | strategy | point (floor / κ / skew) | training (20 d) | held out (11 d) | "
            "held out without 09-22 | days up | fills/day |")
    lines = [head, "| " + " | ".join(["---"] * 8) + " |"]
    for r in comparison():
        flag = "" if r.get("grid20_eligible", True) else " †"
        lines.append("| " + " | ".join([
            r["symbol"][:3].upper(), r["strategy"] + flag, r.get("grid20_point", "–"),
            fmt(r.get("grid20_train")), f"**{fmt(r.get('grid20'))}**", fmt(r.get("grid20_ex")),
            f"{r['grid20_up']}/{r['n']}" if r.get("grid20_up") is not None else "–",
            fmt(r.get("grid20_fills"), "{:.0f}"),
        ]) + " |")
    return "\n".join(lines)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT / name}")


def grid_heatmaps(train_set: str = "train20") -> None:
    chosen = grid_search.select(train_set)
    pairs = [p for p in grid_search.PAIRS if p in chosen]
    cols = 4
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.5 * rows), squeeze=False)
    for ax, (strategy, symbol) in zip(axes.ravel(), pairs, strict=False):
        pick = chosen[(strategy, symbol)]
        skew = pick["point"][2]
        grid = np.full((len(grid_search.FLOOR_MULTS), len(grid_search.KAPPAS)), np.nan)
        thin = np.zeros_like(grid, dtype=bool)
        for r in pick["grid"]:
            f, k, s = r["point"]
            if s != skew:
                continue
            i, j = grid_search.FLOOR_MULTS.index(f), grid_search.KAPPAS.index(k)
            grid[i, j] = r["chained"]
            thin[i, j] = r["median_fills"] < grid_search.MIN_FILLS
        limit = max(np.nanmax(np.abs(grid)), 0.5)
        ax.imshow(grid, cmap="PuOr_r", vmin=-limit, vmax=limit, aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isfinite(grid[i, j]):
                    label = f"{grid[i, j]:+.1f}" + ("\nthin" if thin[i, j] else "")
                    dark = abs(grid[i, j]) > 0.6 * limit
                    ax.text(j, i, label, ha="center", va="center", fontsize=7,
                            color="white" if dark else INK)
        f, k, _ = pick["point"]
        ax.add_patch(plt.Rectangle((grid_search.KAPPAS.index(k) - 0.5,
                                    grid_search.FLOOR_MULTS.index(f) - 0.5), 1, 1,
                                   fill=False, lw=2, ec=INK))
        ax.set_xticks(range(len(grid_search.KAPPAS)), [f"κ {k:g}" for k in grid_search.KAPPAS])
        ax.set_yticks(range(len(grid_search.FLOOR_MULTS)),
                      [f"floor ×{f:g}" for f in grid_search.FLOOR_MULTS])
        ax.set_title(f"{strategy} {symbol[:3].upper()}  (skew ×{skew:g})")
        ax.grid(False)
    for ax in axes.ravel()[len(pairs):]:
        ax.axis("off")
    fig.suptitle("All 20 pre-split sessions: chained return (%) over the grid; box = chosen "
                 "point; 'thin' = under 20 fills a day", fontsize=9.5)
    save(fig, "grid-heatmaps.png")


def heldout() -> None:
    rows = comparison()
    series = [("wide", "C6 widened", BLUE), ("rl_v2", "C6 RL v2 (median of 5)", MAGENTA),
              ("markout", "C7 markout", AQUA), ("grid20", "C4 grid, 20-day choice", ORANGE),
              ("grpo", "C4 GRPO search (median of 2)", YELLOW)]
    series = [s for s in series if any(r[s[0]] is not None for r in rows if s[0] in r)]
    x = np.arange(len(rows))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, (key, label, colour) in enumerate(series):
        values = [r.get(key) if r.get(key) is not None else np.nan for r in rows]
        ax.bar(x + (i - (len(series) - 1) / 2) * width, values, width=width * 0.92,
               color=colour, label=label)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, [f"{r['strategy']}\n{r['symbol'][:3].upper()}" for r in rows], fontsize=8)
    ax.set_ylabel("held-out cumulative return, % of book")
    ax.set_title("Held out (11 sessions per asset): every method on every pair", pad=42)
    ax.legend(ncol=3, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.08))
    ax.grid(axis="x", visible=False)
    save(fig, "heldout.png")


def train_vs_test() -> None:
    rows = [dict(r, grid_test=r["grid20"], grid_train=r["grid20_train"])
            for r in comparison() if r.get("grid20") is not None]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for r in rows:
        ax.scatter(r["grid_train"], r["grid_test"], s=40, color=ORANGE, zorder=3)
        ax.annotate(f"{r['strategy']} {r['symbol'][:3].upper()}",
                    (r["grid_train"], r["grid_test"]), fontsize=7, xytext=(4, 3),
                    textcoords="offset points", color=INK)
    lo = min([r["grid_test"] for r in rows] + [r["grid_train"] for r in rows] + [0]) - 0.5
    hi = max([r["grid_test"] for r in rows] + [r["grid_train"] for r in rows] + [0]) + 0.5
    ax.plot([lo, hi], [lo, hi], color=MUTED, ls="--", lw=0.8, label="held out = training")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("training sessions, chained %")
    ax.set_ylabel("held-out sessions, chained %")
    ax.set_title("20-day grid choices: what training promised, what held out gave")
    ax.legend(fontsize=7.5, loc="upper left")
    save(fig, "train-vs-test.png")


def search_curves() -> None:
    root = RUNS / "c4-search"
    runs = sorted(p for p in root.glob("*__*") if (p / "summary.json").exists()) \
        if root.exists() else []
    if not runs:
        print("no search runs yet")
        return
    groups: dict[str, list[tuple[str, np.ndarray]]] = {}
    for path in runs:
        config = json.loads((path / "config.json").read_text())
        if config.get("fill_target"):
            continue  # a different objective; drawn in fill-target.png
        metrics = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
        key = f"{config['strategy']} {config['symbol'][:3].upper()}"
        groups.setdefault(key, []).append(
            (config["method"], np.array([m["mean_reward"] for m in metrics])))
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.2), squeeze=False)
    for ax, (key, curves) in zip(axes.ravel(), groups.items(), strict=True):
        for method, curve in curves:
            ax.plot(np.arange(1, len(curve) + 1), np.cumsum(curve) / np.arange(1, len(curve) + 1),
                    color=ORANGE if method == "grpo" else BLUE, lw=1.4, alpha=0.85,
                    label=method.upper())
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles, strict=True))
        ax.legend(unique.values(), unique.keys(), fontsize=7.5)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(key)
        ax.set_xlabel("iteration (6 session replays each)")
    axes[0, 0].set_ylabel("running mean sampled reward, bp")
    save(fig, "search-curves.png")


def frontier() -> None:
    """PnL against fills for every grid point: what extra volume costs.

    The upper-left envelope of each panel is the best training return
    available at each level of trading; a point below it is dominated. The
    slope of the envelope is what a fill costs (or pays) at the margin.
    """
    chosen = grid_search.select("train20")
    pairs = [p for p in grid_search.PAIRS if p in chosen]
    cols = 4
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.7 * rows), squeeze=False)
    for ax, (strategy, symbol) in zip(axes.ravel(), pairs, strict=False):
        pick = chosen[(strategy, symbol)]
        points = sorted(pick["grid"], key=lambda r: r["median_fills"])
        fills = np.array([max(r["median_fills"], 0.5) for r in points])
        pnl = np.array([r["chained"] for r in points])
        colours = [ORANGE if r["point"][1] > 0 else BLUE for r in points]
        ax.scatter(fills, pnl, s=22, c=colours, zorder=3)
        # The envelope: best return at or above each fill level.
        best = np.maximum.accumulate(pnl[::-1])[::-1]
        ax.step(fills, best, where="post", color=MUTED, lw=1, zorder=2)
        ax.scatter([max(pick["median_fills"], 0.5)], [pick["chained"]], s=90,
                   facecolors="none", edgecolors=INK, lw=1.3, zorder=4)
        ax.axvline(grid_search.MIN_FILLS, color=MUTED, ls=":", lw=0.8)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"{strategy} {symbol[:3].upper()}")
        ax.set_xlabel("fills, median session (log)")
    for ax in axes.ravel()[len(pairs):]:
        ax.axis("off")
    axes[0, 0].set_ylabel("training return (20 sessions), %")
    fig.suptitle("PnL against trading, every grid point (orange: markout rule on, blue: off; "
                 "ring: chosen; dotted: 20-fill bar; grey: best at or above each fill level)",
                 fontsize=9)
    save(fig, "frontier.png")


def fill_target() -> None:
    """GRPO on obi ETHUSDC with a fill target: what forcing volume does."""
    root = RUNS / "c4-search"
    points = []
    for path in sorted(root.glob("grpo*-obi-ethusdc-s*__*")):
        if not (path / "summary.json").exists():
            continue
        config = json.loads((path / "config.json").read_text())
        final = json.loads((path / "summary.json").read_text())
        name = path.name.split("__")[0]
        test = newest(RUNS / "c4", f"{name}-test")
        if test is None:
            continue
        points.append((config.get("fill_target", 0.0), final.get("fill_lambda", 0.0),
                       float(np.median([d["DailyNumberOfTrades"] for d in test])), chained(test)))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    colours = {0.0: BLUE, 1500.0: ORANGE, 3000.0: AQUA}
    for target, lam, fills, held in points:
        ax.scatter(fills, held, s=46, color=colours.get(target, MUTED), zorder=3)
        if target:
            ax.annotate(f"λ={lam:.3f}", (fills, held), fontsize=7, xytext=(5, 3),
                        textcoords="offset points", color=INK)
    for target, colour in colours.items():
        label = "no target" if not target else f"target {target:.0f} fills/day"
        ax.scatter([], [], color=colour, label=label)
    ax.set_xscale("log")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("held-out fills, median session (log)")
    ax.set_ylabel("held-out cumulative return, %")
    ax.set_title("obi-wide ETHUSDC, GRPO with a fill target: volume costs money\n"
                 "(λ = learned bp paid per fill to hit the target)")
    ax.legend(fontsize=7.5, loc="lower left")
    save(fig, "fill-target.png")


FIGURES = {"grid-heatmaps": grid_heatmaps, "frontier": frontier, "fill-target": fill_target,
           "heldout": heldout, "train-vs-test": train_vs_test, "search-curves": search_curves}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=sorted(FIGURES))
    parser.add_argument("--table", action="store_true", help="print the comparison tables")
    args = parser.parse_args()
    if args.table:
        print(table() + "\n\n" + grid20_table())
        return
    for name, draw in FIGURES.items():
        if args.only in (None, name):
            draw()


if __name__ == "__main__":
    main()
