"""B1's tables and figures, from the run directories under runs/b1/.

    python experiments/b1_ppo/report.py            # figures into docs/assets/b1/ and tables
    python experiments/b1_ppo/report.py --tables   # tables only

Figures (docs/assets/b1/):

    reinforce.png   REINFORCE without and with a KL penalty, mean ± std over seeds:
                    proxy reward, gold P(positive), KL to the reference, distinct-2,
                    reference NLL (held-out evaluation prompts)
    ppo_kl.png      PPO, KL-coefficient sweep: reward model score against KL (training), gold
                    P(positive) against KL (held-out), and the share of samples in Chinese by update
    ppo_clip.png    PPO, clip-range sweep at β = 0.2 at two learning rates, one dot per seed:
                    clipped fraction, peak training KL, final KL and final gold
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src")]
from rl_ladder.run import load_metrics  # noqa: E402

RUNS = REPO / "runs" / "b1"
OUT = REPO / "docs" / "assets" / "b1"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7a5fd6", "#898781"]


def runs(prefix: str) -> dict[str, list[list[dict]]]:
    """config name (seed stripped) -> one metrics list per seed; the latest run of each name wins."""
    latest: dict[str, Path] = {}
    for d in sorted(RUNS.glob(f"{prefix}*__*")):
        if (d / "summary.json").exists():
            latest[d.name.split("__")[0]] = d
    out = defaultdict(list)
    for name, d in sorted(latest.items()):
        out[re.sub(r"-s\d+$", "", name)].append(load_metrics(d))
    return dict(out)


def series(metrics: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    pts = [(m["step"], m[key]) for m in metrics if key in m]
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts], dtype=float)


def band(ax, seeds: list[list[dict]], key: str, label: str, color: str) -> None:
    curves = [series(m, key) for m in seeds]
    curves = [c for c in curves if len(c[0])]
    if not curves:
        return
    n = min(len(c[0]) for c in curves)
    x = curves[0][0][:n]
    y = np.stack([c[1][:n] for c in curves])
    ax.plot(x, y.mean(0), color=color, lw=1.8, label=f"{label} (n={len(curves)})")
    if len(curves) > 1:
        ax.fill_between(x, y.mean(0) - y.std(0), y.mean(0) + y.std(0), color=color, alpha=0.18, lw=0)


def last(metrics: list[dict], key: str) -> float | None:
    vals = [m[key] for m in metrics if key in m]
    return vals[-1] if vals else None


def first(metrics: list[dict], key: str) -> float | None:
    vals = [m[key] for m in metrics if key in m]
    return vals[0] if vals else None


def mean_sd(seeds: list[list[dict]], key: str, pick=None) -> str:
    v = [x for x in ((pick or last)(m, key) for m in seeds) if x is not None]
    if not v:
        return "–"
    return f"{np.mean(v):.2f} ± {np.std(v):.2f}" if len(v) > 1 else f"{v[0]:.2f}"


# ------------------------------------------------------------------ REINFORCE

REINFORCE_KEYS = [("eval_r_proxy", "proxy reward (log-odds)"), ("eval_p_gold", "gold P(positive)"),
                  ("eval_kl", "KL to reference (nats/seq)"), ("eval_distinct2", "distinct-2"),
                  ("eval_ref_nll", "reference NLL (nats/token)")]


def label_reinforce(name: str) -> str:
    kl = name.split("-kl")[1]
    return "no KL penalty" if float(kl) == 0 else f"KL penalty β = {kl}"


def table_reinforce(r: dict) -> str:
    lines = ["| run | seeds | proxy reward | gold P(positive) | KL to reference | distinct-2 | reference NLL |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    every = [m for seeds in r.values() for m in seeds]
    if every:
        lines.append(f"| before training (update 0) | {len(every)} | " +
                     " | ".join(mean_sd(every, k, first) for k, _ in REINFORCE_KEYS) + " |")
    for name, seeds in r.items():
        lines.append(f"| {label_reinforce(name)}, after 200 updates | {len(seeds)} | " +
                     " | ".join(mean_sd(seeds, k) for k, _ in REINFORCE_KEYS) + " |")
    return "\n".join(lines)


def fig_reinforce(r: dict) -> None:
    if not r:
        return
    fig, axes = plt.subplots(1, len(REINFORCE_KEYS), figsize=(4 * len(REINFORCE_KEYS), 3.2))
    for ax, (key, title) in zip(axes, REINFORCE_KEYS):
        for (name, seeds), c in zip(r.items(), COLORS):
            band(ax, seeds, key, label_reinforce(name), c)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("update")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("REINFORCE on Qwen2.5-0.5B (LoRA), IMDB continuations, held-out prompts; mean ± std over seeds",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "reinforce.png")


# ------------------------------------------------------------------ PPO

PPO_KEYS = [("eval_score", "reward model score"), ("eval_p_gold", "gold P(positive)"), ("eval_kl", "KL to reference"),
            ("eval_distinct2", "distinct-2"), ("eval_ref_nll", "reference NLL")]


CJK = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")


def chinese_share(name: str) -> str:
    """Share of each seed's final saved samples that contain Chinese characters (the PPO hack)."""
    shares = []
    for d in sorted(RUNS.glob(f"{name}-s*__*")):
        f = d / "samples.jsonl"
        if not f.exists() or not (d / "summary.json").exists():
            continue
        rows = [json.loads(line) for line in f.read_text().splitlines()]
        final = [r for r in rows if r["step"] == rows[-1]["step"]]
        shares.append(np.mean([bool(CJK.search(r["completion"])) for r in final]))
    return f"{np.mean(shares):.2f}" if shares else "–"


def parse(name: str) -> tuple[str, float, float]:
    """(learning rate, β, clip range) from a run name like ppo-lr1e-4-kl0.2-clip0.05."""
    m = re.search(r"-lr(.+?)-kl", name)
    lr = m.group(1) if m else "1e-5"
    return lr, float(name.split("-kl")[1].split("-")[0]), float(name.split("-clip")[1])


def ordered(p: dict) -> list:
    return sorted(p.items(), key=lambda kv: (float(parse(kv[0])[0]), parse(kv[0])[1], parse(kv[0])[2]))


def mean_last(seeds: list[list[dict]], key: str) -> float:
    v = [x for x in (last(m, key) for m in seeds) if x is not None]
    return float(np.mean(v)) if v else float("nan")


def table_ppo(p: dict) -> str:
    lines = ["| lr | KL coefficient β | clip range | seeds | RM score | gold P(positive) | KL to reference | distinct-2 | "
             "reference NLL | final samples in Chinese | clipped fraction |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, seeds in ordered(p):
        lr, kl, clip = parse(name)
        clipped = np.mean([np.mean(series(m, "policy_clipfrac_avg")[1]) for m in seeds])
        lines.append(f"| {lr} | {kl:g} | {clip:g} | {len(seeds)} | " +
                     " | ".join(mean_sd(seeds, k) for k, _ in PPO_KEYS) +
                     f" | {chinese_share(name)} | {clipped:.4f} |")
    return "\n".join(lines)


def chinese_by_step(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Mean share of saved samples containing Chinese at each evaluation step, over seeds."""
    per = defaultdict(list)
    for d in sorted(RUNS.glob(f"{name}-s*__*")):
        f = d / "samples.jsonl"
        if f.exists() and (d / "summary.json").exists():
            rows = [json.loads(line) for line in f.read_text().splitlines()]
            for step in sorted({r["step"] for r in rows}):
                per[step].append(np.mean([bool(CJK.search(r["completion"])) for r in rows if r["step"] == step]))
    steps = np.array(sorted(per))
    return steps, np.array([np.mean(per[s]) for s in steps])


def fig_ppo(p: dict) -> None:
    """The β sweep at lr 1e-5, clip 0.2: what the reward buys, what gold sees, and when the hack sets in."""
    beta = [(n, s) for n, s in ordered(p) if parse(n)[0] == "1e-5" and parse(n)[2] == 0.2]
    if not beta:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for (name, seeds), c in zip(beta, COLORS):
        label = f"β = {parse(name)[1]:g} (n={len(seeds)})"
        xs, ys = [series(m, "objective_kl")[1] for m in seeds], [series(m, "objective_scores")[1] for m in seeds]
        n = min(len(x) for x in xs)
        if n:
            w = max(1, n // 30)
            smooth = lambda v: np.convolve(v, np.ones(w) / w, mode="valid")
            axes[0].plot(smooth(np.stack([x[:n] for x in xs]).mean(0)), smooth(np.stack([y[:n] for y in ys]).mean(0)),
                         color=c, lw=1.6, label=label)
        ex, ey = [series(m, "eval_kl")[1] for m in seeds], [series(m, "eval_p_gold")[1] for m in seeds]
        k = min(len(x) for x in ex)
        if k:
            axes[1].plot(np.stack([x[:k] for x in ex]).mean(0), np.stack([y[:k] for y in ey]).mean(0),
                         "o-", color=c, lw=1.4, ms=3, label=label)
        steps, share = chinese_by_step(name)
        axes[2].plot(steps, share, "o-", color=c, lw=1.4, ms=3, label=label)
    axes[0].set(xlabel="KL to reference (nats/sequence, training batches)", ylabel="reward model score",
                title="what the reward model pays for KL")
    axes[1].set(xlabel="KL to reference (nats/sequence, held-out)", ylabel="gold P(positive)",
                title="what the gold judge sees at the same KL")
    axes[2].set(xlabel="update", ylabel="share of saved samples with Chinese",
                title="when the language switch sets in")
    for ax in axes:
        ax.title.set_fontsize(10)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("PPO, KL-coefficient sweep (lr 1e-5, clip 0.2); held-out points every 25 updates, mean over seeds",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "ppo_kl.png")


def fig_clip(p: dict) -> None:
    """The clip-range sweep at β = 0.2 at two learning rates; one dot per seed, the line through the means."""
    rows = [(parse(n), s) for n, s in ordered(p) if parse(n)[1] == 0.2]
    lrs = sorted({lr for (lr, _, _), _ in rows}, key=float)
    if not rows:
        return
    peak = lambda m: float(np.max(series(m, "objective_kl")[1]))
    keys = [("clipped fraction (mean over updates)", lambda m: float(np.mean(series(m, "policy_clipfrac_avg")[1]))),
            ("peak KL to reference during training", peak),
            ("final KL to reference (held-out)", lambda m: last(m, "eval_kl")),
            ("final gold P(positive)", lambda m: last(m, "eval_p_gold"))]
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.4))
    for k, (lr, c) in enumerate(zip(lrs, COLORS)):
        pts = sorted((clip, seeds) for (l, _, clip), seeds in rows if l == lr)
        clips = [x for x, _ in pts]
        for ax, (title, f) in zip(axes, keys):
            vals = [[f(m) for m in seeds] for _, seeds in pts]
            for x, v in zip(clips, vals):
                ax.scatter([x * (1 + 0.06 * (k - 0.5))] * len(v), v, color=c, s=18, alpha=0.8)
            ax.plot(clips, [np.mean(v) for v in vals], color=c, lw=1.4,
                    label=f"lr {lr} (seeds per point: {'/'.join(str(len(v)) for v in vals)})")
            ax.set_xscale("log")
            ax.set_xticks(clips, [f"{x:g}" for x in clips])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlabel("clip range ε")
            ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("PPO, clip-range sweep at β = 0.2: one dot per seed, line through the means", fontsize=10)
    fig.tight_layout()
    save(fig, "ppo_clip.png")


def save(fig, fname: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", action="store_true")
    args = ap.parse_args()
    r, p = runs("reinforce-"), runs("ppo-")
    print("## REINFORCE (final held-out evaluation)\n\n" + table_reinforce(r))
    if p:
        print("\n## PPO (final held-out evaluation)\n\n" + table_ppo(p))
    if not args.tables:
        fig_reinforce(r)
        fig_ppo(p)
        fig_clip(p)


if __name__ == "__main__":
    main()
