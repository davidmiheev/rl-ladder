"""C8's tables and figures, from the (configuration, session) cache alone.

    python experiments/c8_risk_controls/report.py            # every figure and table
    python experiments/c8_risk_controls/report.py --tables   # tables only, as markdown

Figures (docs/assets/c8/):

    heldout.png        held-out track records: C6 widened, C7 markout, C8 grid (A), C8 final (D)
    families.png       what each family of ideas adds over the stage-A choice, train and held out
    stress.png         the chosen configurations under each stress
    walk-forward.png   expanding-window selection over the whole month
    breadth.png        the breadth tier: chosen point per asset and strategy, held out

Everything is chosen on the 24 training sessions; the 7 held-out sessions
(2025-09-19 to 09-25) are only ever reported.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments" / "c4_ppo_grpo"))

import c8  # noqa: E402
import plot_c4  # noqa: E402

OUT = REPO / "docs" / "assets" / "c8"
BLUE, ORANGE, AQUA = plot_c4.BLUE, plot_c4.ORANGE, plot_c4.AQUA
YELLOW, MAGENTA = plot_c4.YELLOW, plot_c4.MAGENTA
INK, MUTED = plot_c4.INK, plot_c4.MUTED
ASSET = {"ethusdc": "ETH", "xrpusdc": "XRP"}


def chained(per_day: list[dict] | None) -> float | None:
    return None if per_day is None else plot_c4.chained(per_day)


def ex_crash(per_day: list[dict] | None) -> float | None:
    return None if per_day is None else plot_c4.ex_crash(per_day)


def fmt(v, spec="{:+.2f}") -> str:
    return "–" if v is None else spec.format(v)


def reference(kind: str, strategy: str, symbol: str) -> list[dict] | None:
    """C6's widened preset and C7's markout rule on C8's held-out sessions (earlier runs)."""
    if kind == "c6":
        return plot_c4.c6_wide(strategy, symbol, c8.TEST)
    per_day = plot_c4.c7_markout(strategy, symbol)
    if per_day is None:
        return None
    by_date = {d["date"]: d for d in per_day}
    return [by_date[d] for d in c8.TEST] if all(d in by_date for d in c8.TEST) else None


def describe(cfg: dict, base: dict | None = None) -> str:
    """The parameters that differ from `base` (or the headline ones), compactly."""
    sets = cfg["sets"]
    if base is None:
        return (f"floor {sets['half_spread_bp']:.1f} bp, κ {sets.get('markout_kappa', 0):g}")
    diff = {k: v for k, v in sets.items() if base["sets"].get(k) != v}
    return ", ".join(f"{k} {v:g}" if isinstance(v, float) else f"{k} {v}" for k, v in diff.items()) or "—"


def pair_rows() -> list[dict]:
    rows = []
    for strategy, symbol in c8.PAIRS:
        row = {"strategy": strategy, "symbol": symbol}
        for kind in ("c6", "c7"):
            row[kind] = chained(reference(kind, strategy, symbol))
        a = c8.choice_a(strategy, symbol)
        if a is not None:
            train, test = c8.results(a, c8.TRAIN), c8.results(a, c8.TEST)
            row |= {"a": a, "a_train": chained(train), "a_test": chained(test),
                    "a_ex": ex_crash(test), "a_desc": describe(a),
                    "a_up": sum(d["Return"] > 0 for d in test) if test else None,
                    "a_fills": float(np.median([d["DailyNumberOfTrades"] for d in test])) if test else None}
        final = c8.final_choices(strategy, symbol).get("D")
        if final is not None:
            test = c8.results(final, c8.TEST)
            _, _, added = c8.greedy(strategy, symbol)
            row |= {"d": final, "d_train": chained(c8.results(final, c8.TRAIN)),
                    "d_test": chained(test), "d_ex": ex_crash(test), "d_added": added,
                    "d_up": sum(d["Return"] > 0 for d in test) if test else None,
                    "d_fills": float(np.median([d["DailyNumberOfTrades"] for d in test])) if test else None}
        rows.append(row)
    return rows


def table_main(rows: list[dict]) -> str:
    out = ["| asset | strategy | C6 widened | C7 markout | **C8 grid (A)** | A: train | A: ex-09-22 | "
           "A: days up | A: fills/day | **C8 final (D)** | D: ex-09-22 | D adds |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        out.append(
            f"| {ASSET[r['symbol']]} | {r['strategy']} | {fmt(r['c6'])} | {fmt(r['c7'])} | "
            f"**{fmt(r.get('a_test'))}** | {fmt(r.get('a_train'))} | {fmt(r.get('a_ex'))} | "
            f"{r.get('a_up', '–')}/7 | {fmt(r.get('a_fills'), '{:.0f}')} | "
            f"**{fmt(r.get('d_test'))}** | {fmt(r.get('d_ex'))} | "
            f"{', '.join(r.get('d_added', [])) or '—'} |")
    return "\n".join(out)


def family_gains() -> dict[tuple[str, str], dict[str, tuple[float, float]]]:
    """(train gain, held-out gain) of each family's train-best variant over the stage-A choice."""
    out = {}
    for strategy, symbol in c8.PAIRS:
        a = c8.choice_a(strategy, symbol)
        fams = c8.family_configs(strategy, symbol)
        if a is None or fams is None:
            continue
        a_train, a_test = chained(c8.results(a, c8.TRAIN)), chained(c8.results(a, c8.TEST))
        gains = {}
        for name, cfgs in fams.items():
            pick = c8.choose(cfgs)
            if pick is None or c8.results(pick, c8.TEST) is None:
                continue
            gains[name] = (chained(c8.results(pick, c8.TRAIN)) - a_train,
                           chained(c8.results(pick, c8.TEST)) - a_test)
        out[(strategy, symbol)] = gains
    return out


def table_families(gains) -> str:
    names = sorted({n for g in gains.values() for n in g})
    head = "| family | " + " | ".join(f"{ASSET[s]} {st}" for st, s in gains) + " |"
    out = [head, "| --- |" + " --- |" * len(gains)]
    for n in names:
        cells = []
        for g in gains.values():
            cells.append("–" if n not in g else f"{g[n][0]:+.2f} / {g[n][1]:+.2f}")
        out.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def stress_rows() -> list[dict]:
    rows = []
    for strategy, symbol in c8.PAIRS:
        for label, cfg in c8.final_choices(strategy, symbol).items():
            per_day = c8.results(cfg, c8.TEST)
            base = chained(per_day)
            row = {"strategy": strategy, "symbol": symbol, "choice": label, "base": base}
            if per_day is not None:
                # Session-removal stresses: without the crash, and without
                # whichever held-out session was best for this configuration.
                best = max(per_day, key=lambda d: d["Return"])["date"]
                row["ex 09-22"] = ex_crash(per_day)
                row["ex best day"] = chained([d for d in per_day if d["date"] != best])
                row["fills/day"] = float(np.median([d["DailyNumberOfTrades"] for d in per_day]))
            for name, variant in c8.stress_configs(cfg).items():
                test = c8.results(variant, c8.TEST)
                row[name] = chained(test)
            rows.append(row)
    return rows


def table_stress(rows: list[dict]) -> str:
    names = (["ex 09-22", "ex best day"] + list(c8.STRESS)
             + ["floor x0.8", "floor x1.25", "kappa x0.5", "kappa x2"])
    out = ["| asset | strategy | choice | as chosen | fills/day | " + " | ".join(names) + " |",
           "| --- | --- | --- | --- | --- |" + " --- |" * len(names)]
    for r in rows:
        out.append(f"| {ASSET[r['symbol']]} | {r['strategy']} | {r['choice']} | {fmt(r['base'])} | "
                   f"{fmt(r.get('fills/day'), '{:.0f}')} | "
                   + " | ".join(fmt(r.get(n)) for n in names) + " |")
    return "\n".join(out)


def walk_forward(strategy: str, symbol: str, first: int = 10) -> dict | None:
    """Each session from the `first`-th on, traded with the stage-A grid point that had the
    best chained return over every earlier session (and took MIN_FILLS a day)."""
    cfgs = c8.stage_a(strategy, symbol)
    per = [c8.results(c, c8.DAYS) for c in cfgs]
    if any(p is None for p in per):
        return None
    returns = np.array([[d["Return"] for d in p] for p in per])  # configs x days
    fills = np.array([[d["DailyNumberOfTrades"] for d in p] for p in per])
    picked = []
    for t in range(first, len(c8.DAYS)):
        past = np.prod(1 + returns[:, :t], axis=1)
        ok = np.median(fills[:, :t], axis=1) >= c8.MIN_FILLS
        score = np.where(ok, past, -np.inf) if ok.any() else past
        picked.append(returns[int(np.argmax(score)), t])
    return {"dates": c8.DAYS[first:], "returns": np.array(picked)}


# ------------------------------------------------------------------ figures

def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}")


def fig_heldout(rows: list[dict]) -> None:
    rows = [r for r in rows if "a" in r]
    if not rows:
        return
    n = len(rows)
    cols = 4
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(3.2 * cols, 2.6 * nrows), squeeze=False)
    for ax, r in zip(axes.flat, rows, strict=False):
        series = [("C6 widened", reference("c6", r["strategy"], r["symbol"]), MUTED, ":"),
                  ("C7 markout", reference("c7", r["strategy"], r["symbol"]), BLUE, "-"),
                  ("C8 grid (A)", c8.results(r["a"], c8.TEST), ORANGE, "-"),
                  ("C8 final (D)", c8.results(r["d"], c8.TEST) if "d" in r else None, AQUA, "-")]
        for label, per_day, color, ls in series:
            if per_day is None:
                continue
            curve = (np.cumprod([1 + d["Return"] for d in per_day]) - 1) * 100
            ax.plot(range(1, len(curve) + 1), curve, color=color, ls=ls, lw=1.6, label=label)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.axvline(c8.TEST.index(c8.CRASH) + 1, color=MUTED, lw=0.6, ls="--")
        ax.set_title(f"{r['strategy']} {ASSET[r['symbol']]}")
        ax.set_xticks(range(1, len(c8.TEST) + 1))
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Held-out cumulative return, % of book (dashed: 2025-09-22)", fontsize=10)
    fig.tight_layout()
    save(fig, "heldout.png")


def fig_families(gains) -> None:
    if not gains:
        return
    names = sorted({n for g in gains.values() for n in g})
    pairs = list(gains)
    fig, axes = plt.subplots(1, 2, figsize=(2.2 + 0.9 * len(pairs) * 2, 0.4 * len(names) + 1.6))
    for ax, idx, title in ((axes[0], 0, "training sessions"), (axes[1], 1, "held-out sessions")):
        m = np.full((len(names), len(pairs)), np.nan)
        for j, p in enumerate(pairs):
            for i, n in enumerate(names):
                if n in gains[p]:
                    m[i, j] = gains[p][n][idx]
        lim = np.nanmax(np.abs(m)) if np.isfinite(m).any() else 1.0
        im = ax.imshow(m, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(len(names)):
            for j in range(len(pairs)):
                if np.isfinite(m[i, j]):
                    ax.text(j, i, f"{m[i, j]:+.1f}", ha="center", va="center", fontsize=7, color=INK)
        ax.set_xticks(range(len(pairs)), [f"{s} {ASSET[a]}" for s, a in pairs], rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names if idx == 0 else [])
        ax.set_title(f"gain over stage A, {title} (points)")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    save(fig, "families.png")


def fig_stress(rows: list[dict]) -> None:
    rows = [r for r in rows if r["base"] is not None]
    if not rows:
        return
    names = ["base", "ex 09-22", "ex best day"] + list(c8.STRESS) + ["floor x0.8", "floor x1.25"]
    fig, ax = plt.subplots(figsize=(11, 0.35 * len(rows) + 1.8))
    m = np.array([[r.get(n) if r.get(n) is not None else np.nan for n in names] for r in rows])
    lim = np.nanmax(np.abs(m))
    im = ax.imshow(m, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if np.isfinite(m[i, j]):
                ax.text(j, i, f"{m[i, j]:+.1f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(names)), ["as chosen"] + names[1:], rotation=40, ha="right")
    ax.set_yticks(range(len(rows)), [f"{r['strategy']} {ASSET[r['symbol']]} ({r['choice']})" for r in rows])
    ax.set_title("Held-out return (%) of each chosen configuration under stress")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    save(fig, "stress.png")


def fig_walk_forward() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    colors = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, INK, MUTED, "#8f6ad6"]
    drew = False
    for color, (strategy, symbol) in zip(colors, c8.PAIRS, strict=False):
        wf = walk_forward(strategy, symbol)
        if wf is None:
            continue
        curve = (np.cumprod(1 + wf["returns"]) - 1) * 100
        ax.plot(range(len(curve)), curve, color=color, lw=1.5, label=f"{strategy} {ASSET[symbol]}")
        drew = True
    if not drew:
        plt.close(fig)
        return
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axvline(len(c8.TRAIN) - 10 - 0.5, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("session (from the 11th of the month; dashed: start of the held-out week)")
    ax.set_ylabel("cumulative return, % of book")
    ax.set_title("Walk-forward: each session traded with the grid point best on all earlier ones")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "walk-forward.png")


def breadth_rows() -> list[dict]:
    rows = []
    for strategy, symbol in c8.BREADTH_PAIRS:
        pick = c8.choose(c8.breadth_grid(strategy, symbol))
        if pick is None:
            continue
        test = c8.results(pick, c8.TEST)
        rows.append({"strategy": strategy, "symbol": symbol, "pick": pick,
                     "train": chained(c8.results(pick, c8.TRAIN)), "test": chained(test),
                     "ex": ex_crash(test), "up": sum(d["Return"] > 0 for d in test),
                     "fills": float(np.median([d["DailyNumberOfTrades"] for d in test])),
                     "kappa": pick["sets"].get("markout_kappa", 0.0)})
    return rows


def table_breadth(rows: list[dict]) -> str:
    out = ["| asset | strategy | chosen point | train | held out | ex-09-22 | days up | fills/day |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        out.append(f"| {r['symbol'].upper()} | {r['strategy']} | {describe(r['pick'])} | "
                   f"{fmt(r['train'])} | {fmt(r['test'])} | {fmt(r['ex'])} | {r['up']}/7 | "
                   f"{r['fills']:.0f} |")
    return "\n".join(out)


def fig_breadth(rows: list[dict]) -> None:
    if not rows:
        return
    symbols = sorted({r["symbol"] for r in rows})
    strategies = ["glft", "hfgrid", "obi", "apt", "basis"]
    m = np.full((len(strategies), len(symbols)), np.nan)
    for r in rows:
        m[strategies.index(r["strategy"]), symbols.index(r["symbol"])] = r["test"]
    fig, ax = plt.subplots(figsize=(1.1 * len(symbols) + 2, 3.2))
    lim = np.nanmax(np.abs(m))
    im = ax.imshow(m, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if np.isfinite(m[i, j]):
                ax.text(j, i, f"{m[i, j]:+.1f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(symbols)), [s.upper() for s in symbols], rotation=40, ha="right")
    ax.set_yticks(range(len(strategies)), strategies)
    ax.set_title("Breadth tier: held-out return (%) of the training-chosen point")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    save(fig, "breadth.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", action="store_true", help="print the tables only")
    args = parser.parse_args()
    rows = pair_rows()
    gains = family_gains()
    stress = stress_rows()
    breadth = breadth_rows()
    print("## Main comparison (held-out chained %, 7 sessions)\n\n" + table_main(rows))
    print("\n## Family gains over stage A (train / held out, points)\n\n" + table_families(gains))
    print("\n## Stress (held-out chained %)\n\n" + table_stress(stress))
    print("\n## Breadth\n\n" + table_breadth(breadth))
    if args.tables:
        return
    fig_heldout(rows)
    fig_families(gains)
    fig_stress(stress)
    fig_walk_forward()
    fig_breadth(breadth)


if __name__ == "__main__":
    main()
