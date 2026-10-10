"""B1's tables and figures, from the run directories under runs/b1/.

    python experiments/b1_ppo/report.py            # figures into docs/assets/b1/ and tables
    python experiments/b1_ppo/report.py --tables   # tables only

Figures (docs/assets/b1/):

    reinforce.png   REINFORCE without and with a KL penalty, mean ± std over seeds:
                    proxy reward, gold P(positive), KL to the reference, distinct-2,
                    reference NLL (held-out evaluation prompts)
    ppo_kl.png      PPO: reward against KL to the reference, one curve per
                    (KL coefficient, clip range), mean over seeds
    ppo_sweep.png   PPO: final held-out score, gold, KL and distinct-2 per configuration
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


def table_ppo(p: dict) -> str:
    lines = ["| KL coefficient β | clip range | seeds | RM score | gold P(positive) | KL to reference | distinct-2 | reference NLL |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, seeds in sorted(p.items(), key=lambda kv: (float(kv[0].split("-kl")[1].split("-")[0]),
                                                         float(kv[0].split("-clip")[1]))):
        kl = name.split("-kl")[1].split("-")[0]
        clip = name.split("-clip")[1]
        lines.append(f"| {kl} | {clip} | {len(seeds)} | " + " | ".join(mean_sd(seeds, k) for k, _ in PPO_KEYS) + " |")
    return "\n".join(lines)


def fig_ppo(p: dict) -> None:
    if not p:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for (name, seeds), c in zip(sorted(p.items()), COLORS):
        xs = [series(m, "objective_kl") for m in seeds]
        ys = [series(m, "objective_scores") for m in seeds]
        n = min(len(x[1]) for x in xs)
        if n == 0:
            continue
        kl = np.stack([x[1][:n] for x in xs]).mean(0)
        sc = np.stack([y[1][:n] for y in ys]).mean(0)
        w = max(1, n // 30)
        smooth = lambda v: np.convolve(v, np.ones(w) / w, mode="valid")
        ax.plot(smooth(kl), smooth(sc), color=c, lw=1.6,
                label=f"β {name.split('-kl')[1].split('-')[0]}, clip {name.split('-clip')[1]} (n={len(seeds)})")
    ax.set_xlabel("KL to reference (nats per sequence, training batches)")
    ax.set_ylabel("reward model score")
    ax.set_title("PPO: reward bought per unit of KL", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "ppo_kl.png")


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


if __name__ == "__main__":
    main()
