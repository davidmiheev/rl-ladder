"""C9's tables and figures, from the (configuration, session) cache and the fitted models.

    python experiments/c9_fair_price/report.py            # every figure and table
    python experiments/c9_fair_price/report.py --tables   # tables only, as markdown

Figures (docs/assets/c9/):

    predictability.png  held-out R^2 of each mid-price forecast, by horizon and asset
    oracle.png          return when the quotes are centred on the future mid, by horizon
    r2-curve.png        return against the R^2 of a (noisy-oracle) forecast, with the
                        R^2 real forecasts reach marked
    predict.png         the causal fair prices, each chosen on training sessions, held out
    denoise.png         what each denoiser adds over the C8 choice, per asset, held out
    stress.png          the stress tests of every reported C9 configuration

Everything is chosen on the 24 training sessions; the 7 held-out sessions
(2025-09-19 to 09-25) are only ever reported.
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
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "experiments" / "c8_risk_controls"),
                str(REPO / "experiments" / "c4_ppo_grpo")]

import c8  # noqa: E402
import c9  # noqa: E402
import plot_c4  # noqa: E402
import predict  # noqa: E402

OUT = REPO / "docs" / "assets" / "c9"
BLUE, ORANGE, AQUA = plot_c4.BLUE, plot_c4.ORANGE, plot_c4.AQUA
YELLOW, MAGENTA = plot_c4.YELLOW, plot_c4.MAGENTA
INK, MUTED = plot_c4.INK, plot_c4.MUTED
STRESS_NAMES = (["ex 09-22", "ex best day"] + list(c8.STRESS)
                + ["floor x0.8", "floor x1.25", "kappa x0.5", "kappa x2"])


def fmt(v, spec="{:+.2f}") -> str:
    return "–" if v is None else spec.format(v)


def chained(per_day) -> float | None:
    return None if per_day is None else plot_c4.chained(per_day)


def fills(per_day) -> float | None:
    return None if per_day is None else float(np.median([d["DailyNumberOfTrades"] for d in per_day]))


def name(strategy: str, symbol: str) -> str:
    return f"{symbol.upper()} {strategy}"


def summary(cfg: dict) -> dict:
    train, test = c8.results(cfg, c8.TRAIN), c8.results(cfg, c8.TEST)
    return {"train": chained(train), "test": chained(test),
            "ex_crash": None if test is None else plot_c4.ex_crash(test),
            "fills": fills(test), "train_fills": fills(train)}


def stress_row(cfg: dict) -> dict:
    per_day = c8.results(cfg, c8.TEST)
    row = {"base": chained(per_day), "fills/day": fills(per_day)}
    if per_day is not None:
        best = max(per_day, key=lambda d: d["Return"])["date"]
        row["ex 09-22"] = plot_c4.ex_crash(per_day)
        row["ex best day"] = chained([d for d in per_day if d["date"] != best])
    for label, variant in c8.stress_configs(cfg).items():
        row[label] = chained(c8.results(variant, c8.TEST))
    return row


def describe(cfg: dict, base: dict) -> str:
    extra = {k: v for k, v in cfg["sets"].items() if base["sets"].get(k) != v}
    s = cfg["sets"]
    if "dag_mode" in extra:
        act = "pull" if s["dag_widen_bp"] == 0 else f"widen {s['dag_widen_bp']:g} bp"
        return f"DAgger student {s['fair_src'].split(':', 1)[1]}, p > {s['dag_thr']:g}: {act}"
    if "fair_src" in extra or "fair_mode" in extra:
        mode = {1.0: "added to own signal", 2.0: "replaces own signal",
                3.0: "widens the endangered side", 4.0: "pulls the endangered side"}[s["fair_mode"]]
        return f"{s['fair_src']}, ×{s['fair_coef']:g}, {mode}"
    return ", ".join(f"{k} {v:g}" for k, v in extra.items())


# ------------------------------------------------------------ predictability

def predictability() -> dict[str, dict]:
    out = {}
    for path in sorted(predict.MODELS.glob("report_*.json")):
        out[path.stem[len("report_"):]] = json.loads(path.read_text())
    return out


def table_predictability(rep: dict) -> str:
    lines = ["| asset | horizon | move std, bp | moves | micro-price | OBI signal | linear | "
             "boosted trees | trees, training | trees hit rate |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for symbol, rows in rep.items():
        for k, row in rows.items():
            g = row["gbm"]
            lines.append(
                f"| {symbol.upper()} | {int(k) / 10:g} s | {g['std_bp']:.2f} | {g['moved_frac']:.0f}% | "
                f"{row['micro']['r2']:.1f} | {row['obi']['r2']:.1f} | {row['lin']['r2']:.1f} | "
                f"**{g['r2']:.1f}** | {g['train']:.1f} | {g['hit']:.0f}% |")
    return "\n".join(lines)


def fig_predictability(rep: dict) -> None:
    if not rep:
        return
    n = len(rep)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.2), squeeze=False, sharey=True)
    for ax, (symbol, rows) in zip(axes[0], rep.items()):
        ks = sorted(int(k) for k in rows)
        for model, color, label in (("micro", BLUE, "micro-price"), ("obi", MUTED, "OBI signal"),
                                    ("lin", AQUA, "linear"), ("gbm", ORANGE, "boosted trees")):
            ax.plot([k / 10 for k in ks], [rows[str(k)][model]["r2"] for k in ks], "o-",
                    color=color, label=label, lw=1.6, ms=4)
        ax.set_title(symbol.upper(), fontsize=10)
        ax.set_xlabel("horizon, s")
        ax.axhline(0, color=INK, lw=0.6)
    axes[0][0].set_ylabel("held-out R², %")
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle("How well the next 0.5–3 s of the mid can be forecast, held out", fontsize=11)
    save(fig, "predictability.png")


# --------------------------------------------------------------------- oracle

def oracle_rows() -> list[dict]:
    rows = []
    for strategy, symbol in c9.ORACLE_TARGETS:
        base = c9.base_choice(strategy, symbol)
        if base is None:
            continue
        row = {"strategy": strategy, "symbol": symbol, "base": summary(base), "variants": {}}
        for label, extra in c9.oracle_variants(strategy).items():
            row["variants"][label] = summary(c8.on_top(base, extra))
        rows.append(row)
    return rows


ORACLE_COLS = ("oracle +0.5s", "oracle +1s", "oracle +1.5s", "oracle +3s",
               "oracle +1s, own signal off")


def table_oracle(rows: list[dict]) -> str:
    lines = ["| configuration | sessions | as chosen | " + " | ".join(
        c.replace("oracle ", "mid ") for c in ORACLE_COLS) + " |",
             "| --- | --- | --- |" + " --- |" * len(ORACLE_COLS)]
    for r in rows:
        for part, label in (("test", "held out"), ("train", "training")):
            cells = [fmt(r["variants"][c][part]) if c in r["variants"] else "–" for c in ORACLE_COLS]
            lines.append(f"| {name(r['strategy'], r['symbol'])} | {label} | {fmt(r['base'][part])} | "
                         + " | ".join(cells) + " |")
        cells = [fmt(r["variants"][c]["fills"], "{:.0f}") if c in r["variants"] else "–"
                 for c in ORACLE_COLS]
        lines.append(f"| | fills/day held out | {fmt(r['base']['fills'], '{:.0f}')} | "
                     + " | ".join(cells) + " |")
    return "\n".join(lines)


def fig_oracle(rows: list[dict]) -> None:
    rows = [r for r in rows if r["base"]["test"] is not None]
    if not rows:
        return
    n = len(rows)
    cols = min(n, 4)
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(3.4 * cols, 2.9 * nrows), squeeze=False)
    for ax, r in zip(axes.flat, rows):
        hs = [0.0, 0.5, 1.0, 1.5, 3.0]
        labels = [None, "oracle +0.5s", "oracle +1s", "oracle +1.5s", "oracle +3s"]
        for part, color, lab in (("train", MUTED, "training (24)"), ("test", BLUE, "held out (7)")):
            ys = [r["base"][part]] + [r["variants"].get(l, {}).get(part) for l in labels[1:]]
            ax.plot(hs, [np.nan if y is None else y for y in ys], "o-", color=color, label=lab, ms=4)
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_title(name(r["strategy"], r["symbol"]), fontsize=10)
        ax.set_xlabel("quotes centred on the mid this far ahead, s")
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("cumulative return, %")
    axes.flat[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Quoting around the future mid: what perfect foresight is worth", fontsize=11)
    fig.tight_layout()
    save(fig, "oracle.png")


def fig_r2_curve(rows: list[dict], rep: dict) -> None:
    rows = [r for r in rows if "noisy +1s R2=0.1" in r["variants"]
            and r["variants"]["noisy +1s R2=0.1"]["test"] is not None]
    if not rows:
        return
    fig, axes = plt.subplots(1, len(rows), figsize=(3.6 * len(rows), 3.2), squeeze=False)
    for ax, r in zip(axes[0], rows):
        r2s = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]
        labels = [None] + [f"noisy +1s R2={x:g}" for x in r2s[1:-1]] + ["oracle +1s"]
        for part, color, lab in (("train", MUTED, "training"), ("test", BLUE, "held out")):
            ys = [r["base"][part]] + [r["variants"][l][part] for l in labels[1:]]
            ax.plot(r2s, [np.nan if y is None else y for y in ys], "o-", color=color, label=lab, ms=4)
        got = rep.get(r["symbol"], {}).get("10")
        if got:
            for model, color in (("lin", AQUA), ("gbm", ORANGE)):
                ax.axvline(got[model]["r2"] / 100, color=color, ls="--", lw=1,
                           label=f"{'linear' if model == 'lin' else 'trees'}: {got[model]['r2']:.1f}%")
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_xticks([0, 0.05, 0.1, 0.2, 0.4, 1.0], ["0", ".05", ".1", ".2", ".4", "1"])
        ax.minorticks_off()
        ax.set_xlabel("R² of the 1 s forecast the quotes are centred on")
        ax.set_title(name(r["strategy"], r["symbol"]), fontsize=10)
        ax.legend(fontsize=7, frameon=False)
    axes[0][0].set_ylabel("cumulative return, %")
    fig.suptitle("How good a 1 s forecast has to be before it pays", fontsize=11)
    fig.tight_layout()
    save(fig, "r2-curve.png")


# ------------------------------------------------------------ fair price

def predict_rows() -> list[dict]:
    rows = []
    for strategy, symbol in c9.ORACLE_TARGETS:
        base = c9.base_choice(strategy, symbol)
        best = c9.fair_best(strategy, symbol)
        if base is None or best is None:
            continue
        rows.append({"strategy": strategy, "symbol": symbol, "base_cfg": base,
                     "base": summary(base),
                     "fams": {n: {"cfg": c, "what": describe(c, base), **summary(c)}
                              for n, c in best.items()}})
    return rows


def table_predict(rows: list[dict]) -> str:
    lines = ["| configuration | fair price | chosen variant | training | held out | "
             "without 09-22 | fills/day |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        b = r["base"]
        lines.append(f"| {name(r['strategy'], r['symbol'])} | the mid (C8) | – | {fmt(b['train'])} | "
                     f"{fmt(b['test'])} | {fmt(b['ex_crash'])} | {fmt(b['fills'], '{:.0f}')} |")
        for fam, v in r["fams"].items():
            lines.append(f"| | {fam} | {v['what']} | {fmt(v['train'])} | {fmt(v['test'])} | "
                         f"{fmt(v['ex_crash'])} | {fmt(v['fills'], '{:.0f}')} |")
    return "\n".join(lines)


def fig_predict(rows: list[dict]) -> None:
    rows = [r for r in rows if r["base"]["test"] is not None]
    if not rows:
        return
    fams = list(rows[0]["fams"])
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(rows), 3.4))
    width = 0.8 / (len(fams) + 1)
    colors = [INK, BLUE, AQUA, YELLOW, ORANGE]
    for i, lab in enumerate(["the mid (C8)"] + fams):
        ys = [r["base"]["test"] if i == 0 else r["fams"][lab]["test"] for r in rows]
        ax.bar(np.arange(len(rows)) + (i - len(fams) / 2) * width,
               [np.nan if y is None else y for y in ys], width, color=colors[i % 5], label=lab)
    ax.set_xticks(range(len(rows)), [name(r["strategy"], r["symbol"]) for r in rows],
                  rotation=30, ha="right", fontsize=8)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_ylabel("held-out cumulative return, %")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title("Quotes centred on a causal fair price, chosen on training sessions", fontsize=10)
    save(fig, "predict.png")


# -------------------------------------------------------------------- denoise

def denoise_rows() -> list[dict]:
    rows = []
    for strategy, symbol in c9.TARGETS:
        base = c9.base_choice(strategy, symbol)
        best = c9.denoise_best(strategy, symbol)
        if base is None or best is None:
            continue
        rows.append({"strategy": strategy, "symbol": symbol, "base_cfg": base,
                     "base": summary(base),
                     "fams": {n: {"cfg": c, "what": describe(c, base), **summary(c)}
                              for n, c in best.items()}})
    return rows


DENOISE_FAMS = ("signal smoothing", "alpha clipping", "standardisation window",
                "imbalance depth", "markout half-life")


def table_denoise(rows: list[dict]) -> str:
    lines = ["| configuration | C8 choice, training / held out | fills/day | "
             + " | ".join(DENOISE_FAMS) + " |", "| --- | --- | --- |" + " --- |" * len(DENOISE_FAMS)]
    for r in rows:
        b = r["base"]
        cells = []
        for fam in DENOISE_FAMS:
            v = r["fams"].get(fam)
            if v is None or v["train"] is None or b["train"] is None:
                cells.append("–")
                continue
            cells.append(f"{v['train'] - b['train']:+.2f} / {fmt(None if v['test'] is None else v['test'] - b['test'])}")
        lines.append(f"| {name(r['strategy'], r['symbol'])} | {fmt(b['train'])} / {fmt(b['test'])} | "
                     f"{fmt(b['fills'], '{:.0f}')} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_denoise_choices(rows: list[dict]) -> str:
    lines = ["| configuration | " + " | ".join(DENOISE_FAMS) + " |",
             "| --- |" + " --- |" * len(DENOISE_FAMS)]
    for r in rows:
        lines.append(f"| {name(r['strategy'], r['symbol'])} | " + " | ".join(
            r["fams"][f]["what"] if f in r["fams"] else "–" for f in DENOISE_FAMS) + " |")
    return "\n".join(lines)


def fig_denoise(rows: list[dict]) -> None:
    rows = [r for r in rows if r["base"]["test"] is not None]
    if not rows:
        return
    grid = np.full((len(rows), len(DENOISE_FAMS)), np.nan)
    train = np.full_like(grid, np.nan)
    for i, r in enumerate(rows):
        for j, fam in enumerate(DENOISE_FAMS):
            v = r["fams"].get(fam)
            if v and v["test"] is not None:
                grid[i, j] = v["test"] - r["base"]["test"]
                train[i, j] = v["train"] - r["base"]["train"]
    lim = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1.0
    fig, ax = plt.subplots(figsize=(1.5 + 1.25 * len(DENOISE_FAMS), 0.9 + 0.42 * len(rows)))
    im = ax.imshow(grid, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:+.2f}\n({train[i, j]:+.2f})", ha="center",
                        va="center", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(DENOISE_FAMS)), DENOISE_FAMS, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)), [name(r["strategy"], r["symbol"]) for r in rows], fontsize=8)
    fig.colorbar(im, ax=ax, label="held-out gain, points")
    ax.set_title("Each denoiser's training-best variant over the C8 choice:\n"
                 "held out (training in brackets)", fontsize=10)
    save(fig, "denoise.png")


# ---------------------------------------------------------- large moves, DAgger

def large_rows() -> list[dict]:
    rows = []
    for strategy, symbol in c9.LARGE_TARGETS:
        base = c9.base_choice(strategy, symbol)
        pick = c9.large_choice(strategy, symbol)
        if base is None or pick is None:
            continue
        rows.append({"strategy": strategy, "symbol": symbol, "base_cfg": base, "base": summary(base),
                     "oracle": {lab: summary(c8.on_top(base, v))
                                for lab, v in c9.large_oracle_variants().items()},
                     "fams": {"large-move forecast rule": {"cfg": pick, "what": describe(pick, base),
                                                           **summary(pick)}}})
    return rows


LARGE_COLS = ("pull, top 10% of 1 s moves", "pull, top 1% of 3 s moves",
              "widen, top 10% of 1 s moves", "widen endangered side by the 1 s move")


def table_large(rows: list[dict]) -> str:
    lines = ["| configuration | sessions | C8 choice | " + " | ".join(
        f"oracle: {c}" for c in LARGE_COLS) + " | forecast rule (chosen on training) |",
             "| --- | --- | --- |" + " --- |" * len(LARGE_COLS) + " --- |"]
    for r in rows:
        rule = r["fams"]["large-move forecast rule"]
        for part, label in (("test", "held out"), ("train", "training")):
            lines.append(f"| {name(r['strategy'], r['symbol'])} | {label} | {fmt(r['base'][part])} | "
                         + " | ".join(fmt(r["oracle"][c][part]) for c in LARGE_COLS)
                         + f" | {fmt(rule[part])} |")
        lines.append(f"| | fills/day held out | {fmt(r['base']['fills'], '{:.0f}')} | "
                     + " | ".join(fmt(r["oracle"][c]["fills"], "{:.0f}") for c in LARGE_COLS)
                     + f" | {fmt(rule['fills'], '{:.0f}')} |")
    return "\n".join(lines)


def dagger_rows() -> list[dict]:
    import dagger

    rows = []
    for strategy, symbol in dagger.TARGETS:
        base = c9.base_choice(strategy, symbol)
        fams = {}
        for tag, rnd, _ in c9.dagger_students():
            cfgs = c9.dagger_configs(strategy, symbol, tag)
            pick = c8.choose(cfgs) if cfgs else None
            if pick is None:
                continue
            teacher = "5 s markout teacher" if tag == "" else "large-move teacher"
            fit = json.loads((dagger.ROOT / f"{strategy}{tag}_{symbol}_r{rnd}.json").read_text())
            fams[f"DAgger, {teacher}"] = {"cfg": pick, "what": describe(pick, base), "fit": fit,
                                         **summary(pick)}
        if fams:
            rows.append({"strategy": strategy, "symbol": symbol, "base_cfg": base,
                         "base": summary(base), "fams": fams})
    return rows


def table_dagger(rows: list[dict]) -> str:
    lines = ["| configuration | student | rounds, labelled rows | teacher says pull | "
             "held-out AUC | chosen action | training | held out | without 09-22 | fills/day |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        b = r["base"]
        lines.append(f"| {name(r['strategy'], r['symbol'])} | none (C8 choice) | – | – | – | – | "
                     f"{fmt(b['train'])} | {fmt(b['test'])} | {fmt(b['ex_crash'])} | "
                     f"{fmt(b['fills'], '{:.0f}')} |")
        for fam, v in r["fams"].items():
            fit = v["fit"]
            auc = fmt(fit.get("val_auc"), "{:.2f}")
            lines.append(f"| | {fam} | {fit['round'] + 1}, {fit['rows']:,} | "
                         f"{100 * fit['weighted_adverse']:.1f}% | {auc} | "
                         f"{v['what'].split(': ', 1)[-1]} (p > {v['cfg']['sets']['dag_thr']:g}) | "
                         f"{fmt(v['train'])} | {fmt(v['test'])} | {fmt(v['ex_crash'])} | "
                         f"{fmt(v['fills'], '{:.0f}')} |")
    return "\n".join(lines)


def fig_large(large: list[dict], dag: list[dict]) -> None:
    if not large:
        return
    dag_by = {(r["strategy"], r["symbol"]): r for r in dag}
    series = [("C8 choice", INK, lambda r: r["base"]["test"]),
              ("oracle: pull, top 1% of 3 s moves", BLUE,
               lambda r: r["oracle"]["pull, top 1% of 3 s moves"]["test"]),
              ("forecast rule", AQUA, lambda r: r["fams"]["large-move forecast rule"]["test"]),
              ("DAgger, 5 s markout teacher", YELLOW,
               lambda r: dag_by.get((r["strategy"], r["symbol"]), {}).get("fams", {})
               .get("DAgger, 5 s markout teacher", {}).get("test")),
              ("DAgger, large-move teacher", ORANGE,
               lambda r: dag_by.get((r["strategy"], r["symbol"]), {}).get("fams", {})
               .get("DAgger, large-move teacher", {}).get("test"))]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    width = 0.8 / len(series)
    for i, (lab, color, get) in enumerate(series):
        ys = [get(r) for r in large]
        ax.bar(np.arange(len(large)) + (i - (len(series) - 1) / 2) * width,
               [np.nan if y is None else y for y in ys], width, color=color, label=lab)
    ax.set_xticks(range(len(large)), [name(r["strategy"], r["symbol"]) for r in large], fontsize=9)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_ylabel("held-out cumulative return, %")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title("Acting on large moves: the oracle, a forecast rule, and two DAgger students",
                 fontsize=10)
    save(fig, "large-moves.png")


# --------------------------------------------------------------------- stress

def stress_rows(pred: list[dict], den: list[dict], large: list[dict] = (),
                dag: list[dict] = ()) -> list[dict]:
    out = []
    for kind, rows in (("fair price", pred), ("denoise", den), ("large move", large),
                       ("dagger", dag)):
        for r in rows:
            out.append({"config": name(r["strategy"], r["symbol"]), "what": "C8 choice",
                        "kind": kind, "train": r["base"]["train"], **stress_row(r["base_cfg"])})
            for fam, v in r["fams"].items():
                out.append({"config": name(r["strategy"], r["symbol"]), "what": fam, "kind": kind,
                            "train": v["train"], "base_train": r["base"]["train"],
                            **stress_row(v["cfg"])})
    return out


def table_stress(rows: list[dict], kind: str) -> str:
    lines = ["| configuration | variant | held out | fills/day | " + " | ".join(STRESS_NAMES) + " |",
             "| --- | --- | --- | --- |" + " --- |" * len(STRESS_NAMES)]
    seen = set()
    for r in rows:
        if r["kind"] != kind or (r["config"], r["what"]) in seen:
            continue
        seen.add((r["config"], r["what"]))
        lines.append(f"| {r['config']} | {r['what']} | {fmt(r['base'])} | "
                     f"{fmt(r.get('fills/day'), '{:.0f}')} | "
                     + " | ".join(fmt(r.get(n)) for n in STRESS_NAMES) + " |")
    return "\n".join(lines)


def fig_stress(rows: list[dict]) -> None:
    # The C8 choices, and every C9 variant that beat its C8 choice on the
    # training sessions -- the ones a training-session selection would adopt.
    rows = [r for r in rows if r["base"] is not None and r.get("fee 0") is not None
            and r["kind"] != "denoise"
            and ("base_train" not in r or (r["train"] or -1e9) > (r["base_train"] or 1e9))]
    seen, uniq = set(), []
    for r in rows:
        if (r["config"], r["what"]) not in seen:
            seen.add((r["config"], r["what"]))
            uniq.append(r)
    rows = uniq
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(10, 0.9 + 0.3 * len(rows)))
    for i, r in enumerate(rows):
        vals = [r.get(n) for n in STRESS_NAMES if r.get(n) is not None]
        ax.plot([min(vals), max(vals)], [i, i], color=MUTED, lw=1)
        ax.plot(r["base"], i, "o", color=BLUE, ms=6)
        for n, marker in (("fee +1bp", "s"), ("latency x4", "^"), ("ex 09-22", "x"),
                          ("ex best day", "v")):
            if r.get(n) is not None:
                ax.plot(r[n], i, marker, color=ORANGE if "fee" in n else MAGENTA if "latency" in n
                        else INK, ms=5, mfc="none")
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(range(len(rows)), [f"{r['config']} · {r['what']}" for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("held-out cumulative return, % (line: range over all stresses)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="", label="as chosen"),
                       Line2D([], [], marker="s", color=ORANGE, mfc="none", ls="", label="fee +1 bp"),
                       Line2D([], [], marker="^", color=MAGENTA, mfc="none", ls="", label="latency ×4"),
                       Line2D([], [], marker="x", color=INK, ls="", label="without 09-22"),
                       Line2D([], [], marker="v", color=INK, mfc="none", ls="", label="without best day")],
              fontsize=7, frameon=False, loc="lower right")
    ax.set_title("Stress tests: the C8 choices and every C9 variant that beat them on training "
                 "sessions", fontsize=10)
    save(fig, "stress.png")


def save(fig, fname: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / fname}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", action="store_true")
    args = parser.parse_args()
    rep = predictability()
    orc = oracle_rows()
    pred = predict_rows()
    den = denoise_rows()
    large = large_rows()
    dag = dagger_rows()
    stress = stress_rows(pred, den, large, dag)
    print("## Predictability (held-out R², %)\n\n" + table_predictability(rep))
    print("\n## Oracle (cumulative return, %)\n\n" + table_oracle(orc))
    print("\n## Causal fair prices\n\n" + table_predict(pred))
    print("\n## Denoising: gain over the C8 choice, training / held out\n\n" + table_denoise(den))
    print("\n## Denoising: chosen variants\n\n" + table_denoise_choices(den))
    print("\n## Stress, fair prices\n\n" + table_stress(stress, "fair price"))
    print("\n## Stress, denoising\n\n" + table_stress(stress, "denoise"))
    print("\n## Large moves (cumulative return, %)\n\n" + table_large(large))
    print("\n## DAgger\n\n" + table_dagger(dag))
    print("\n## Stress, large moves and DAgger\n\n" + table_stress(stress, "large move")
          + "\n" + "\n".join(table_stress(stress, "dagger").splitlines()[2:]))
    if not args.tables:
        fig_predictability(rep)
        fig_oracle(orc)
        fig_r2_curve(orc, rep)
        fig_predict(pred)
        fig_denoise(den)
        fig_large(large, dag)
        fig_stress(stress)


if __name__ == "__main__":
    main()
