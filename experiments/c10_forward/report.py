"""C10's tables and figures, from the (configuration, session) cache and runs/c10.

    python experiments/c10_forward/report.py            # every figure and table
    python experiments/c10_forward/report.py --tables   # tables only, as markdown

Figures (docs/assets/c10/):

    forward.png   each frozen configuration's cumulative return over the
                  forward year on Bybit, as frozen and with the picked stop
    crash.png     the 2025-10-10 crash session's maximum drawdown under every
                  control and stop, per configuration
    stops.png     the forward year without the crash: as frozen against
                  every stop, per configuration
    leadlag.png   held-out forecast quality with outside data, by arrival delay
    llbt.png      the lead-lag forecasts through the backtest, held out
                  (written once `llbacktest.py` results are in)
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
sys.path[:0] = [str(HERE), str(REPO / "experiments" / "c4_ppo_grpo")]

import forward  # noqa: E402  (puts c8 on the path)
import c8  # noqa: E402
import plot_c4  # noqa: E402

OUT = REPO / "docs" / "assets" / "c10"
BLUE, ORANGE, AQUA = plot_c4.BLUE, plot_c4.ORANGE, plot_c4.AQUA
YELLOW, MAGENTA = plot_c4.YELLOW, plot_c4.MAGENTA
INK, MUTED = plot_c4.INK, plot_c4.MUTED
FROZEN = "as frozen (-0.5 bp)"
FW_EX = [d for d in forward.FORWARD if d != forward.CRASH]
LEADLAG = ("xrpusdc", "suiusdc", "ethusdc")


def fmt(v, spec="{:+.2f}") -> str:
    return "–" if v is None else spec.format(v)


def short(label: str) -> str:
    """'XRPUSDC OBI → XRPPERP' -> 'XRPUSDC → XRPPERP'."""
    return label.replace(" OBI", "")


# -------------------------------------------------------------- forward test

def table_forward() -> str:
    lines = []
    for period, dates in (("venue control (study month's held-out week, on Bybit)", forward.CONTROL),
                          ("forward, all 54 weekly sessions", forward.FORWARD),
                          ("forward without the 2025-10-10 crash", FW_EX)):
        lines += [f"\n**{period}**\n",
                  "| configuration | " + " | ".join(forward.VARIANTS) + " | fills/day | sessions up |",
                  "| --- |" + " --- |" * (len(forward.VARIANTS) + 2)]
        for label in forward.FROZEN:
            cells = [forward.summary(label, v, dates) for v in forward.VARIANTS]
            b = cells[0]
            lines.append(f"| {short(label)} | " + " | ".join(
                fmt(None if c is None else c["chained"]) for c in cells)
                + f" | {fmt(None if b is None else b['fills'], '{:.0f}')} | "
                + ("–" if b is None else f"{b['up']}/{b['n']}") + " |")
    return "\n".join(lines)


def path(label: str, variant: str) -> tuple[list[str], np.ndarray]:
    dates, r = [], []
    for d in forward.FORWARD:
        res = c8.result(forward.cfg(label, variant), d)
        if res is not None:
            dates.append(d)
            r.append(res["Return"])
    return dates, (np.cumprod(1 + np.array(r)) - 1) * 100


def fig_forward() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True)
    ci = forward.FORWARD.index(forward.CRASH)
    for ax, label in zip(axes.flat, forward.FROZEN):
        for variant, color, name in ((FROZEN, BLUE, "as frozen"),
                                     (forward.STOP_PICK, ORANGE, forward.STOP_PICK)):
            dates, cum = path(label, variant)
            ax.plot([forward.FORWARD.index(d) for d in dates], cum, color=color, lw=1.6, label=name)
        ax.axvline(ci, color=MUTED, lw=0.8, ls="--")
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_title(short(label), fontsize=9)
        ax.grid(alpha=0.25, lw=0.6)
    for ax in axes[1]:
        ax.set_xlabel("weekly session since 2025-09-26")
    for ax in axes[:, 0]:
        ax.set_ylabel("cumulative return, %")
    axes[0][0].legend(fontsize=8, frameon=False)
    axes[0][0].text(ci + 1, 0.02, "2025-10-10 crash", transform=axes[0][0].get_xaxis_transform(),
                    fontsize=7, color=MUTED, va="bottom")
    fig.suptitle("The frozen C8 configurations on Bybit, one session a week for a year", fontsize=11)
    fig.tight_layout()
    save(fig, "forward.png")


# ---------------------------------------------------------- crash and stops

def controls_rows(controls: dict) -> list[dict]:
    rows = []
    for label in forward.FROZEN:
        for v in [FROZEN] + list(controls):
            c = c8.result(forward.cfg(label, v), forward.CRASH)
            s = forward.summary(label, v, FW_EX)
            rows.append({"config": label, "variant": v,
                         "crash": None if c is None else c["Return"] * 100,
                         "crash_mdd": None if c is None else c["MaxDrawdown"] * 100,
                         "crash_maxpos": None if c is None else c["MaxPositionValue"] / 1e3,
                         "fwd": None if s is None else s["chained"],
                         "fills": None if s is None else s["fills"]})
    return rows


def table_controls(rows: list[dict]) -> str:
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    lines = ["| configuration | " + " | ".join(variants) + " |",
             "| --- |" + " --- |" * len(variants)]
    for label in forward.FROZEN:
        by = {r["variant"]: r for r in rows if r["config"] == label}
        lines.append(f"| {short(label)} | " + " | ".join(
            "–" if by[v]["crash"] is None else
            f"{by[v]['crash']:+.1f} / {by[v]['crash_mdd']:.1f} ; {fmt(by[v]['fwd'])}"
            for v in variants) + " |")
    return "\n".join(lines)


def table_rule(rows: list[dict]) -> str:
    """The pre-registered rule: worst crash drawdown <= 10%, then the summed forward return."""
    lines = ["| variant | worst crash-day drawdown, % | passes (≤ 10%) | summed forward return "
             "without the crash, % | median fills/day (per configuration) |",
             "| --- | --- | --- | --- | --- |"]
    for v in dict.fromkeys(r["variant"] for r in rows):
        rs = [r for r in rows if r["variant"] == v]
        if any(r["crash_mdd"] is None or r["fwd"] is None for r in rs):
            continue
        worst = max(r["crash_mdd"] for r in rs)
        lines.append(f"| {v} | {worst:.1f} | {'yes' if worst <= 10 else 'no'} | "
                     f"{sum(r['fwd'] for r in rs):+.2f} | "
                     + ", ".join(f"{r['fills']:.0f}" for r in rs) + " |")
    return "\n".join(lines)


def fig_crash(rows: list[dict]) -> None:
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    colors = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, INK]
    fig, ax = plt.subplots(figsize=(9, 0.6 + 0.32 * len(variants)))
    for i, v in enumerate(variants):
        vals = [r["crash_mdd"] for r in rows if r["variant"] == v and r["crash_mdd"] is not None]
        ax.plot([min(vals), max(vals)], [i, i], color=MUTED, lw=1)
        for label, color in zip(forward.FROZEN, colors):
            r = next(r for r in rows if r["variant"] == v and r["config"] == label)
            if r["crash_mdd"] is not None:
                ax.plot(r["crash_mdd"], i, "o", color=color, ms=5,
                        label=short(label) if i == 0 else None)
    ax.axvline(10, color=INK, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    ax.set_yticks(range(len(variants)), variants, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("maximum drawdown on 2025-10-10, % of capital (log scale; dashed: the 10% bound)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title("The crash session under every control: C8's controls bound none of it, "
                 "a stop does", fontsize=10)
    save(fig, "crash.png")


def fig_stops(rows: list[dict]) -> None:
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    colors = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, INK]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    width = 0.8 / len(variants)
    for j, (v, color) in enumerate(zip(variants, colors)):
        ys = [next(r["fwd"] for r in rows if r["variant"] == v and r["config"] == label)
              for label in forward.FROZEN]
        xs = np.arange(len(forward.FROZEN)) - 0.4 + width * (j + 0.5)
        ax.bar(xs, ys, width * 0.85, color=color, label=v)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(range(len(forward.FROZEN)),
                  [short(x).replace(" → ", "\n→ ") for x in forward.FROZEN], fontsize=7)
    ax.set_ylabel("chained return, %")
    ax.legend(fontsize=7, frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_title("What the stops cost on the forward year without the crash (53 sessions)",
                 fontsize=10)
    save(fig, "stops.png")


def table_stop_stress() -> str:
    names = [FROZEN] + list(forward.STOP_STRESS)
    head = ["as picked"] + [n.split(", ", 1)[1] for n in forward.STOP_STRESS]
    lines = [f"| configuration ({forward.STOP_PICK}) | " + " | ".join(head) + " | fills/day |",
             "| --- |" + " --- |" * (len(head) + 1)]
    for label in forward.FROZEN:
        cells = [forward.summary(label, forward.STOP_PICK, FW_EX)]
        cells += [forward.summary(label, n, FW_EX) for n in forward.STOP_STRESS]
        lines.append(f"| {short(label)} | " + " | ".join(
            fmt(None if c is None else c["chained"]) for c in cells)
            + f" | {fmt(None if cells[0] is None else cells[0]['fills'], '{:.0f}')} |")
    # Does the stop still bound the crash when it is slower or pays more?
    worst = []
    for n in [forward.STOP_PICK] + list(forward.STOP_STRESS):
        r = [c8.result(forward.cfg(label, n), forward.CRASH) for label in forward.FROZEN]
        worst.append(None if any(x is None for x in r) else max(x["MaxDrawdown"] for x in r) * 100)
    lines.append("| *worst crash-day drawdown, %* | " + " | ".join(fmt(w, "{:.1f}") for w in worst)
                 + " | |")
    del names
    return "\n".join(lines)


# ------------------------------------------------------------------ lead-lag

def leadlag() -> dict[str, dict]:
    out = {}
    for s in LEADLAG:
        p = REPO / "runs" / "c10" / f"leadlag_{s}.json"
        if p.exists():
            out[s] = json.loads(p.read_text())["tasks"]
    return out


def table_leadlag(ll: dict) -> str:
    lines = ["| asset | inputs | R² 1 s, % | R² 3 s, % | large-move AUC | precision at 0.3% "
             "flagged, % | precision at 1% flagged, % |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for s, t in ll.items():
        rows = [("own book", "book", t["cls_book"])]
        rows += [(f"+ outside, +{d} ms", f"book+ext@{d}", t[f"cls_book+ext@{d}"]) for d in (0, 100, 250)]
        base = t["cls_book"]["base_rate"] * 100
        for name, key, c in rows:
            lines.append(f"| {s.upper()} (base rate {base:.2f}%) | {name} | {t[f'r2_10_{key}']:.2f} | "
                         f"{t[f'r2_30_{key}']:.2f} | {c['auc']:.3f} | {c['precision_0.003'] * 100:.1f} | "
                         f"{c['precision_0.01'] * 100:.1f} |")
    return "\n".join(lines)


def fig_leadlag(ll: dict) -> None:
    if not ll:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    xs = ["own book", "+0 ms", "+100 ms", "+250 ms"]
    for (s, t), color in zip(ll.items(), (BLUE, ORANGE, AQUA)):
        r2 = [t["r2_10_book"]] + [t[f"r2_10_book+ext@{d}"] for d in (0, 100, 250)]
        pr = [t["cls_book"]["precision_0.003"] * 100] + [
            t[f"cls_book+ext@{d}"]["precision_0.003"] * 100 for d in (0, 100, 250)]
        axes[0].plot(range(4), r2, "o-", color=color, lw=1.6, ms=4, label=s.upper())
        axes[1].plot(range(4), pr, "o-", color=color, lw=1.6, ms=4, label=s.upper())
    for ax in axes:
        ax.set_xticks(range(4), xs, fontsize=8)
        ax.grid(alpha=0.25, lw=0.6)
    axes[0].set_ylabel("held-out R² of the 1 s move, %")
    axes[1].set_ylabel("precision when 0.3% of steps are flagged, %")
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_title("1 s forecast", fontsize=10)
    axes[1].set_title("large-move classifier (base rate 0.5–0.7%)", fontsize=10)
    fig.suptitle("Outside data, by how late it arrives (+ ms on top of the assumed delay)",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "leadlag.png")


def llbt() -> dict | None:
    p = REPO / "runs" / "c10" / "llbacktest.json"
    return json.loads(p.read_text()) if p.exists() else None


def table_llbt(t: dict | None) -> str:
    if not t:
        return "(lead-lag backtests not in yet)"
    lines = ["| asset | variant | training, % | held out, % | held-out fills/day |",
             "| --- | --- | --- | --- | --- |"]
    for s, sym in t.items():
        for label, v in sym["variants"].items():
            tr, te = v["train"], v["test"]
            lines.append(f"| {s.upper()} | {label} | {fmt(None if tr is None else tr['chained'])} | "
                         f"{fmt(None if te is None else te['chained'])} | "
                         f"{fmt(None if te is None else te['fills'], '{:.0f}')} |")
    return "\n".join(lines)


def table_llbt_stress(t: dict | None) -> str:
    if not t:
        return ""
    names = sorted({n for sym in t.values() for ch in sym["choices"].values() for n in ch["stress"]},
                   key=lambda n: (n.startswith("outside"), n))
    lines = ["| asset | inputs | training pick | held out | " + " | ".join(names) + " |",
             "| --- | --- | --- | --- |" + " --- |" * len(names)]
    for s, sym in t.items():
        for fam, ch in sym["choices"].items():
            lines.append(f"| {s.upper()} | {fam} | {ch['pick']} | "
                         f"{fmt(None if ch['test'] is None else ch['test']['chained'])} | " + " | ".join(
                             fmt(None if ch["stress"].get(n) is None else ch["stress"][n]["chained"])
                             for n in names) + " |")
    return "\n".join(lines)


def fig_llbt(t: dict | None) -> None:
    if not t:
        return
    fig, axes = plt.subplots(1, len(t), figsize=(4.2 * len(t), 3.8), squeeze=False, sharey=True)
    for ax, (s, sym) in zip(axes[0], t.items()):
        labels = list(sym["variants"])
        for i, label in enumerate(labels):
            te = sym["variants"][label]["test"]
            if te is None:
                continue
            color = INK if i == 0 else BLUE if label.startswith("own book:") else ORANGE
            ax.barh(i, te["chained"], color=color, height=0.7)
        ax.axvline(0, color=INK, lw=0.6)
        ax.set_yticks(range(len(labels)), [x.replace("own book + outside: ", "+ outside: ")
                                          .replace("own book: ", "book: ") for x in labels],
                      fontsize=7)
        ax.invert_yaxis()
        ax.set_title(f"{s.upper()} OBI", fontsize=10)
        ax.set_xlabel("held-out cumulative return, %")
        ax.grid(axis="x", alpha=0.25, lw=0.6)
    fig.suptitle("Quoting with the forecasts: own book (blue) against own book + outside data "
                 "(orange), held out", fontsize=11)
    fig.tight_layout()
    save(fig, "llbt.png")


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
    ctl = controls_rows(forward.CONTROLS)
    stops = controls_rows(forward.STOPS)
    both = ctl + [r for r in stops if r["variant"] != FROZEN]
    ll = leadlag()
    bt = llbt()
    print("## Forward test (chained return, %)\n" + table_forward())
    print("\n## C8's controls: crash return / max drawdown ; forward without the crash\n\n"
          + table_controls(ctl))
    print("\n## Stops: crash return / max drawdown ; forward without the crash\n\n"
          + table_controls(stops))
    print("\n## The rule\n\n" + table_rule(both))
    print(f"\n## {forward.STOP_PICK}, stress-tested (forward without the crash)\n\n"
          + table_stop_stress())
    print("\n## Lead-lag forecasts, held out\n\n" + table_leadlag(ll))
    print("\n## Lead-lag through the backtest\n\n" + table_llbt(bt))
    print("\n## Lead-lag picks, stress-tested (held out)\n\n" + table_llbt_stress(bt))
    if not args.tables:
        fig_forward()
        fig_crash(both)
        fig_stops(stops)
        fig_leadlag(ll)
        fig_llbt(bt)


if __name__ == "__main__":
    main()
