"""C13's tables and figures, from the (configuration, session) cache.

    python experiments/c13_bybit_contest/report.py            # every figure and table
    python experiments/c13_bybit_contest/report.py --tables   # tables only, as markdown

Figures (docs/assets/c13/):

    test.png     each tested contract's cumulative return over the locked test
                 block: the pick under each rule and C8's choice, unchanged
    crash.png    the greedy result under each rule on training, split into the
                 2025-10-10 crash session and the other 47 sessions
    rebate.png   the screen leaders' return as backtested, without the assumed
                 0.5 bp maker rebate, and at Bybit's standard +2 bp fee
    stress.png   each tested pick's test return under C8's stress set
    ena.png      ENAUSDT's pick, per session: return against its spread floor in
                 ticks, by block (the rebate artefact)
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
sys.path[:0] = [str(HERE), str(REPO / "experiments" / "c4_ppo_grpo")]

import contest as k  # noqa: E402  (puts c8 and forward on the path)
import c8  # noqa: E402
import forward  # noqa: E402
import plot_c4  # noqa: E402

OUT = REPO / "docs" / "assets" / "c13"
BLUE, ORANGE, AQUA = plot_c4.BLUE, plot_c4.ORANGE, plot_c4.AQUA
INK, MUTED = plot_c4.INK, plot_c4.MUTED
REBATE = 0.00005  # config.MAKER_FEE is -0.5 bp: a rebate of turnover x 0.5 bp
VIP0 = 0.0002     # Bybit's standard maker fee
FROZEN = {"XRPPERP": "XRPUSDC OBI → XRPPERP", "SUIPERP": "SUIUSDC OBI → SUIPERP",
          "XRPUSDT": "XRPUSDT OBI → XRPUSDT", "SUIUSDT": "SUIUSDT OBI → SUIUSDT",
          "AVAXUSDT": "AVAXUSDC OBI → AVAXUSDT"}
RULE_LABEL = {"all sessions": "C8's rule (all sessions)", "crash-guarded": "crash-guarded"}


def fmt(v, spec="{:+.2f}") -> str:
    return "–" if v is None else spec.format(v)


def summary(cfg: dict, days: list[str], fee: float = 0.0) -> dict | None:
    """Chained return (%) with `fee` x turnover taken off each session, and the rest."""
    rows = c8.results(cfg, days)
    if rows is None:
        return None
    r = np.array([x["Return"] for x in rows])
    t = np.array([x.get("DailyTurnover", 0.0) for x in rows])
    net = r - t * fee
    best = int(np.argmax(net))
    return {"ret": float(np.prod(1 + net) - 1) * 100,
            "ex_best": float(np.prod(1 + np.delete(net, best)) - 1) * 100,
            "up": int((net > 0).sum()), "n": len(net),
            "fills": float(np.median([x["DailyNumberOfTrades"] for x in rows])),
            "mdd": float(max(x["MaxDrawdown"] for x in rows)) * 100,
            "path": np.cumprod(1 + net) * 100 - 100}


def frozen(contract: str) -> dict | None:
    label = FROZEN.get(contract)
    return None if label is None else forward.cfg(label, "as frozen (-0.5 bp)")


def tested() -> dict[str, dict[str, tuple[str, dict]]]:
    """contract -> rule -> (candidate name, pick) for contracts whose picks have test results."""
    out = {}
    for contract in k.CONTRACTS:
        picks = {}
        for rule in k.RULES:
            fp = k.final_pick(contract, rule)
            if fp is not None and c8.results(fp[1], k.TEST) is not None:
                picks[rule] = fp
        if len(picks) == len(k.RULES):
            out[contract] = picks
    return out


# ------------------------------------------------------------------ tables

ORIGINAL = ("XRPPERP", "SUIPERP", "XRPUSDT", "SUIUSDT", "AVAXUSDT")  # C10's five
# Backtest artefacts, reported but left out of the sums: ENAUSDT's pick farms the
# rebate where its spread floor is under one tick and stops at its loss limit elsewhere.
ARTEFACT = ("ENAUSDT",)


def fee_stress(cfg: dict, name: str) -> float:
    """The backtest's own fee stress on the test block (a fee can change how a quoter trades)."""
    return summary(c8.stress_configs(cfg)[name], k.TEST)["ret"]


def table_scoreboard(t: dict) -> str:
    lines = ["| contract | pick, C8's rule | pick, crash-guarded | C8's choice, unchanged | "
             "crash-guarded at maker fee +1 bp | crash-guarded at +2 bp |",
             "| --- | --- | --- | --- | --- | --- |"]
    groups = {"five original": ORIGINAL, "added": tuple(c for c in t if c not in ORIGINAL and c not in ARTEFACT)}
    sums = {g: {"a": 0.0, "b": 0.0, "c": 0.0} for g in groups}
    for contract, picks in t.items():
        a = summary(picks["all sessions"][1], k.TEST)
        b = summary(picks["crash-guarded"][1], k.TEST)
        f = frozen(contract)
        c = summary(f, k.TEST) if f is not None else None
        b1, b2 = (fee_stress(picks["crash-guarded"][1], n) for n in ("fee +1bp", "fee +2bp"))
        for g, members in groups.items():
            if contract in members:
                sums[g]["a"] += a["ret"]
                sums[g]["b"] += b["ret"]
                sums[g]["c"] += c["ret"] if c else 0.0
        note = " (artefact)" if contract in ARTEFACT else ""
        lines.append(f"| {contract}{note} | {fmt(a['ret'])}% | {fmt(b['ret'])}% | "
                     f"{fmt(c['ret']) + '%' if c else '–'} | {fmt(b1)}% | {fmt(b2)}% |")
    for g, members in groups.items():
        if any(m in t for m in members):
            c_sum = fmt(sums[g]["c"]) if any(frozen(m) for m in members if m in t) else "–"
            lines.append(f"| sum, {g} | {fmt(sums[g]['a'])} | {fmt(sums[g]['b'])} | {c_sum} | | |")
    return "\n".join(lines)


def table_contract(contract: str, picks: dict) -> str:
    lines = ["| entry | training | validation | test | test without its best session | "
             "test without the rebate | test sessions up | fills/session (test) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    entries = [(f"pick, {RULE_LABEL[r]}: {picks[r][0]}", picks[r][1]) for r in k.RULES]
    f = frozen(contract)
    if f is not None:
        entries.append(("C8's choice, unchanged", f))
    for name, cfg in entries:
        tr, va, te = (summary(cfg, d) for d in (k.TRAIN, k.VALID, k.TEST))
        nr = summary(cfg, k.TEST, REBATE)
        lines.append(f"| {name} | {fmt(tr['ret'])}% | {fmt(va['ret'])}% | **{fmt(te['ret'])}%** | "
                     f"{fmt(te['ex_best'])}% | {fmt(nr['ret'])}% | {te['up']}/{te['n']} | {te['fills']:.0f} |")
    return "\n".join(lines)


def stress_rows(cfg: dict) -> list[tuple[str, float | None]]:
    rows = [("as chosen", summary(cfg, k.TEST)["ret"])]
    for name, sc in c8.stress_configs(cfg).items():
        s = summary(sc, k.TEST)
        rows.append((name, None if s is None else s["ret"]))
    return rows


def table_stress(picks: dict) -> str:
    a, b = (stress_rows(picks[r][1]) for r in k.RULES)
    lines = ["| stress (test block) | pick, C8's rule | pick, crash-guarded |", "| --- | --- | --- |"]
    names = dict.fromkeys([n for n, _ in a] + [n for n, _ in b])
    da, db = dict(a), dict(b)
    for n in names:
        lines.append(f"| {n} | {fmt(da.get(n))}% | {fmt(db.get(n))}% |")
    return "\n".join(lines)


def table_rebate() -> str:
    lines = ["| contract | screen leader | screen | without the rebate | at +2 bp | turnover/session |",
             "| --- | --- | --- | --- | --- | --- |"]
    for contract in k.CONTRACTS:
        top = k.screen_pick(contract, 1)
        if top is None:
            continue
        s, nr, vip = (summary(top[0], k.SCREEN, fee) for fee in (0.0, REBATE, REBATE + VIP0))
        turnover = np.median([x.get("DailyTurnover", 0.0) for x in c8.results(top[0], k.SCREEN)])
        lines.append(f"| {contract} | {k.describe(top[0])} | {fmt(s['ret'])}% | {fmt(nr['ret'])}% | "
                     f"{fmt(vip['ret'])}% | {turnover:.1f}x |")
    return "\n".join(lines)


def crash_rows() -> list[dict]:
    rows = []
    for contract in k.CONTRACTS:
        for rule in k.RULES:
            cur, pending, added = k.greedy(contract, rule)
            if cur is None or pending:
                continue
            res = c8.results(cur, k.TRAIN)
            if res is None:
                continue
            r = np.array([x["Return"] for x in res])
            crash = np.array([d == forward.CRASH for d in k.TRAIN])
            rows.append({"contract": contract, "rule": rule, "added": added,
                         "crash": float(r[crash][0]) * 100,
                         "other": float(np.prod(1 + r[~crash]) - 1) * 100,
                         "crash_mdd": float([x["MaxDrawdown"] for x, c in zip(res, crash) if c][0]) * 100})
    return rows


def table_crash(rows: list[dict]) -> str:
    lines = ["| contract | rule | added | crash session | other 47 sessions | crash-session drawdown |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['contract']} | {RULE_LABEL[r['rule']]} | {', '.join(r['added']) or 'nothing'} | "
                     f"{fmt(r['crash'])}% | {fmt(r['other'])}% | {r['crash_mdd']:.1f}% |")
    return "\n".join(lines)


# ----------------------------------------------------------------- figures

def save(fig, fname: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_test(t: dict) -> None:
    n = len(t)
    if n == 0:
        return
    cols = min(3, n)
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(4.2 * cols, 3.0 * ((n + cols - 1) // cols)),
                             squeeze=False)
    for ax, (contract, picks) in zip(axes.flat, t.items()):
        x = np.arange(1, len(k.TEST) + 1)
        for rule, color in zip(k.RULES, (ORANGE, BLUE)):
            ax.plot(x, summary(picks[rule][1], k.TEST)["path"], color=color, lw=1.8, label=RULE_LABEL[rule])
        f = frozen(contract)
        if f is not None:
            ax.plot(x, summary(f, k.TEST)["path"], color=MUTED, lw=1.4, ls="--", label="C8's choice, unchanged")
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_title(contract, fontsize=10)
        ax.set_xlabel("test session")
        ax.set_ylabel("cumulative return, %")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Locked test (27 sessions, Jun-Oct 2026)", fontsize=11)
    fig.tight_layout()
    save(fig, "test.png")


def fig_crash(rows: list[dict]) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
    for ax, rule in zip(axes, k.RULES):
        sub = {r["contract"]: r for r in rows if r["rule"] == rule}
        contracts = list(sub)
        x = np.arange(len(contracts))
        ax.bar(x - 0.2, [sub[c]["crash"] for c in contracts], 0.4, color=ORANGE,
               label="2025-10-10 crash session")
        ax.bar(x + 0.2, [sub[c]["other"] for c in contracts], 0.4, color=BLUE,
               label="other 47 training sessions")
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_xticks(x, contracts, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"greedy result, {RULE_LABEL[rule]}", fontsize=10)
        ax.set_ylabel("return, %")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, "crash.png")


def fig_rebate() -> None:
    names, vals = [], []
    for contract in k.CONTRACTS:
        top = k.screen_pick(contract, 1)
        if top is None:
            continue
        names.append(contract)
        vals.append([summary(top[0], k.SCREEN, fee)["ret"] for fee in (0.0, REBATE, REBATE + VIP0)])
    if not names:
        return
    vals = np.array(vals)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for i, (label, color) in enumerate((("as backtested (0.5 bp rebate)", BLUE),
                                        ("no rebate", AQUA), ("Bybit standard +2 bp", ORANGE))):
        ax.bar(x + (i - 1) * 0.27, np.clip(vals[:, i], -60, None), 0.27, color=color, label=label)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(x, names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("screen return, % (clipped at -60)")
    ax.set_title("Stage 1 leaders: how much is the assumed maker rebate", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, "rebate.png")


def fig_stress(t: dict) -> None:
    t = {c: p for c, p in t.items() if c not in ARTEFACT}
    if not t:
        return
    fig, ax = plt.subplots(figsize=(max(10, 1.3 * len(t)), 3.8))
    labels = []
    for i, (contract, picks) in enumerate(t.items()):
        for j, (rule, color) in enumerate(zip(k.RULES, (ORANGE, BLUE))):
            vals = [v for _, v in stress_rows(picks[rule][1]) if v is not None]
            xpos = i * 3 + j
            ax.scatter([xpos] * len(vals), vals, color=color, s=14, alpha=0.8,
                       label=RULE_LABEL[rule] if i == 0 else None)
            ax.scatter([xpos], [vals[0]], color=INK, marker="_", s=200)
        labels.append((i * 3 + 0.5, contract))
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks([p for p, _ in labels], [c for _, c in labels], fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("test return, %")
    ax.set_title("Each pick's test return under C8's stress set (bar = as chosen)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, "stress.png")


def ena_rows() -> list[tuple[str, float, float, float]]:
    """(block, floor in ticks, return %, fills) per session for ENAUSDT's pick."""
    fp = k.final_pick("ENAUSDT", "crash-guarded")
    if fp is None:
        return []
    out = []
    for block, days in (("training", k.TRAIN), ("validation", k.VALID), ("test", k.TEST)):
        for r in c8.results(fp[1], days) or []:
            out.append((block, r["spread_floor_tick"], r["Return"] * 100, r["DailyNumberOfTrades"]))
    return out


def table_ena(rows: list) -> str:
    lines = ["| block | sessions | floor under 1 tick | median return, floor < 1 tick | "
             "median return, floor ≥ 1 tick | median fills, floor < 1 tick | median fills, floor ≥ 1 tick |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for block in ("training", "validation", "test"):
        sub = [r for r in rows if r[0] == block]
        lo = [r for r in sub if r[1] < 1]
        hi = [r for r in sub if r[1] >= 1]
        med = lambda xs, i: fmt(float(np.median([x[i] for x in xs]))) + "%" if xs else "–"
        fills = lambda xs: f"{np.median([x[3] for x in xs]):,.0f}" if xs else "–"
        lines.append(f"| {block} | {len(sub)} | {len(lo)} | {med(lo, 2)} | {med(hi, 2)} | {fills(lo)} | {fills(hi)} |")
    return "\n".join(lines)


def fig_ena(rows: list) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for block, color in (("training", MUTED), ("validation", BLUE), ("test", ORANGE)):
        sub = [r for r in rows if r[0] == block]
        ax.scatter([r[1] for r in sub], [r[2] for r in sub], color=color, s=16, alpha=0.85, label=block)
    ax.axvline(1, color=INK, lw=0.6, ls="--")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("8 bp spread floor, in price ticks (log)")
    ax.set_ylabel("session return, %")
    ax.set_title("ENAUSDT pick: the return is the rebate when the floor is under one tick", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, "ena.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", action="store_true", help="tables only")
    args = parser.parse_args()
    t = tested()
    rows = crash_rows()
    print("## Scoreboard\n\n" + table_scoreboard(t))
    for contract, picks in t.items():
        print(f"\n## {contract}\n\n" + table_contract(contract, picks) + "\n\n" + table_stress(picks))
    print("\n## Rebate\n\n" + table_rebate())
    print("\n## Crash\n\n" + table_crash(rows))
    ena = ena_rows()
    print("\n## ENAUSDT\n\n" + table_ena(ena))
    if not args.tables:
        fig_test(t)
        fig_crash(rows)
        fig_rebate()
        fig_stress(t)
        fig_ena(ena)


if __name__ == "__main__":
    main()
