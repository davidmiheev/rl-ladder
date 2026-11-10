"""B2's tables and figures, from the run directories under runs/b2/ (and B1's PPO runs under runs/b1/).

    python experiments/b2_dpo/report.py            # figures into docs/assets/b2/ and tables
    python experiments/b2_dpo/report.py --tables   # tables only

Runs are grouped by their configuration (loss, β/τ, learning rate, pairs, epochs, pair rule), not by
name; seeds are the replicates. "Final" is the last held-out evaluation of a run.

Figures (docs/assets/b2/):

    frontier.png   gold P(positive) and reward-model score against KL to the reference, final held-out
                   evaluation: every full-set DPO/IPO configuration, B1's PPO configurations, and the
                   best-of-4 reference samples the pairs were built from
    trajectory.png the same plane as training proceeds: DPO's best configuration against B1's best PPO
    overfit.png    items 2-3: DPO and IPO on 64 pairs for many epochs, mean ± std over seeds: loss,
                   held-out pair accuracy, chosen and rejected log-ratios, chosen log-probability per
                   token, sampled KL to the reference, gold P(positive), distinct-2
    ipo_tau.png    IPO's τ against the margin it settles at, and the KL it costs
"""

from __future__ import annotations

import argparse
import json
import math
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

RUNS = REPO / "runs" / "b2"
B1 = REPO / "runs" / "b1"
OUT = REPO / "docs" / "assets" / "b2"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7a5fd6", "#898781", "#3b3b3b"]
#: B1's two best PPO configurations (README table): (lr, β, clip range).
PPO_BEST = [(1e-4, 0.2, 0.05), (1e-5, 0.2, 0.2)]


def latest_runs(root: Path, prefix: str) -> list[Path]:
    """Finished runs, the latest of each name."""
    latest: dict[str, Path] = {}
    for d in sorted(root.glob(f"{prefix}*__*")):
        if (d / "summary.json").exists():
            latest[d.name.split("__")[0]] = d
    return list(latest.values())


def b2_groups() -> dict[tuple, list[list[dict]]]:
    """(loss, beta, lr, pairs, epochs, pair_rule) -> one metrics list per seed."""
    out = defaultdict(list)
    for d in latest_runs(RUNS, ""):
        if d.name.startswith("prefs"):
            continue
        c = json.loads((d / "config.json").read_text())
        c = c.get("config", c)
        out[(c["loss"], c["beta"], c["lr"], c["pairs"], c["epochs"], c["pair_rule"])].append(load_metrics(d))
    return dict(sorted(out.items()))


def ppo_groups() -> dict[tuple, list[list[dict]]]:
    out = defaultdict(list)
    for d in latest_runs(B1, "ppo-"):
        c = json.loads((d / "config.json").read_text())
        c = c.get("config", c)
        out[(c["lr"], c["kl_coef"], c["cliprange"])].append(load_metrics(d))
    return dict(out)


def final(metrics: list[dict], key: str) -> float:
    vals = [m[key] for m in metrics if key in m]
    return float(vals[-1]) if vals else float("nan")


def series(metrics: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    pts = [(m["step"], m[key]) for m in metrics if key in m]
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts], dtype=float)


def minutes(metrics: list[dict]) -> float:
    return max((m.get("seconds", 0.0) for m in metrics), default=0.0) / 60


def ms(vals: list[float], fmt: str = "{:.2f}") -> str:
    v = np.asarray(vals, dtype=float)
    return fmt.format(v.mean()) + ("" if len(v) == 1 else " ± " + fmt.format(v.std()))


def lr_str(lr: float) -> str:
    m, e = f"{lr:.0e}".split("e")
    return f"{m}e{int(e)}"


def label(key: tuple) -> str:
    loss, beta, lr, pairs, epochs, rule = key
    s = f"{loss.upper()} {'β' if loss == 'dpo' else 'τ'}={beta:g}, lr {lr_str(lr)}"
    if pairs:
        s += f", {pairs} pairs"
    if epochs != 1:
        s += f", {epochs:g} ep"
    if rule != "best-worst":
        s += f", {rule} pairs"
    return s


# ------------------------------------------------------------------ tables

def prefs_table() -> None:
    runs = latest_runs(RUNS, "prefs")
    if not runs:
        return
    s = json.loads((runs[-1] / "summary.json").read_text())
    s = s.get("summary", s)
    print("\n## The preference set (reference samples, K = 4 per prompt)\n")
    print("| split | prompts | RM score, all samples | gold, all samples | RM, best of 4 | gold, best of 4 "
          "| RM, worst of 4 | gold, worst of 4 | median RM margin best-worst | gold agrees best ≥ worst |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for sp in ("train", "test"):
        g = lambda k: s[f"{sp}_{k}"]  # noqa: E731
        print(f"| {sp} | {g('prompts')} | {g('rm_mean'):.2f} | {g('gold_mean'):.3f} | {g('best_of_k_rm'):.2f} | "
              f"{g('best_of_k_gold'):.3f} | {g('worst_of_k_rm'):.2f} | {g('worst_of_k_gold'):.3f} | "
              f"{g('margin_median'):.2f} | {g('gold_agrees_best_vs_worst'):.3f} |")
    print(f"\nWall clock {s['seconds'] / 60:.1f} min on one T4; "
          f"{s['test_prompts_in_b1_eval_set']} test prompts coincide with B1's 64 evaluation prompts.")


COLS = ("eval_score", "eval_p_gold", "eval_kl", "eval_distinct2", "eval_ref_nll", "eval_chinese")


def main_table(groups) -> None:
    print("\n## Full preference set and PPO (final held-out evaluation)\n")
    print("| method | seeds | RM score | gold P(positive) | KL to reference | distinct-2 | reference NLL "
          "| samples in Chinese | held-out pair accuracy | chosen log-ratio (held out) "
          "| rejected log-ratio (held out) | minutes per run (T4) |")
    print("|" + " --- |" * 12)
    for key, seeds in groups.items():
        if key[3]:
            continue
        cells = [ms([final(m, c) for m in seeds], "{:.3f}" if c == "eval_p_gold" else "{:.2f}") for c in COLS]
        held = [ms([final(m, c) for m in seeds]) for c in ("held_acc", "held_chosen_logratio",
                                                          "held_rejected_logratio")]
        print(f"| {label(key)} | {len(seeds)} | " + " | ".join(cells + held)
              + f" | {np.mean([minutes(m) for m in seeds]):.0f} |")
    ppo = ppo_groups()
    for lr, beta, clip in PPO_BEST:
        seeds = ppo.get((lr, beta, clip))
        if not seeds:
            continue
        cells = [ms([final(m, c) for m in seeds], "{:.3f}" if c == "eval_p_gold" else "{:.2f}")
                 for c in COLS[:5]]
        print(f"| B1 PPO β={beta:g}, clip {clip:g}, lr {lr_str(lr)} | {len(seeds)} | " + " | ".join(cells)
              + f" | – | – | – | – | {np.mean([minutes(m) for m in seeds]):.0f} |")


def overfit_table(groups) -> None:
    print("\n## Small preference sets, many epochs (items 2-3; final and extreme values)\n")
    print("| method | seeds | train loss | train accuracy | train margin h | chosen log-ratio (train) "
          "| lowest chosen log-ratio (train) | rejected log-ratio (train) | held-out accuracy | KL to reference "
          "| peak KL | gold P(positive) | distinct-2 | samples in Chinese |")
    print("|" + " --- |" * 14)
    for key, seeds in groups.items():
        if not key[3]:
            continue
        f = lambda c, fmt="{:.2f}": ms([final(m, c) for m in seeds], fmt)  # noqa: E731
        low = ms([series(m, "train_chosen_logratio")[1].min() for m in seeds])
        peak = ms([series(m, "eval_kl")[1].max() for m in seeds])
        print(f"| {label(key)} | {len(seeds)} | {f('train_loss', '{:.3g}')} | {f('train_acc')} | "
              f"{f('train_margin', '{:.1f}')} | {f('train_chosen_logratio', '{:.1f}')} | {low} | "
              f"{f('train_rejected_logratio', '{:.1f}')} | {f('held_acc')} | {f('eval_kl', '{:.1f}')} | {peak} | "
              f"{f('eval_p_gold', '{:.3f}')} | {f('eval_distinct2')} | {f('eval_chinese')} |")


# ------------------------------------------------------------------ figures

def smooth_steps(seeds, key):
    """Mean and std over seeds on the steps all seeds logged."""
    curves = [series(m, key) for m in seeds]
    common = sorted(set.intersection(*(set(x.tolist()) for x, _ in curves)))
    ys = np.array([[dict(zip(x.tolist(), y))[s] for s in common] for x, y in curves])
    return np.array(common), ys.mean(0), ys.std(0)


def fig_frontier(groups) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ppo = ppo_groups()
    for (lr, beta, clip), seeds in sorted(ppo.items()):
        kl = [final(m, "eval_kl") for m in seeds]
        for ax, key in zip(axes, ("eval_p_gold", "eval_score")):
            ax.errorbar(np.mean(kl), np.mean([final(m, key) for m in seeds]), xerr=np.std(kl),
                        yerr=np.std([final(m, key) for m in seeds]), fmt="s", color="#898781", ms=5,
                        alpha=0.8, label="B1 PPO (each β, clip, lr)" if (lr, beta, clip) == (1e-5, 0.0, 0.2) else None)
            if beta == 0.2 and (lr, clip) in [(a, c) for a, _, c in PPO_BEST]:
                ax.annotate(f"PPO β={beta:g}, clip {clip:g}, lr {lr_str(lr)}", (np.mean(kl), np.mean([final(m, key) for m in seeds])),
                            textcoords="offset points", xytext=(6, -10), fontsize=7, color="#555")
    prefs = latest_runs(RUNS, "prefs")
    if prefs:
        s = json.loads((prefs[-1] / "summary.json").read_text())
        s = s.get("summary", s)
        bon_kl = math.log(4) - 3 / 4  # KL(best-of-n || reference) for n = 4, distinct samples
        for ax, k in zip(axes, ("test_best_of_k_gold", "test_best_of_k_rm")):
            ax.plot(bon_kl, s[k], "*", color="#eda100", ms=13, label="best of 4 reference samples")
            ax.plot(0, s["test_gold_mean" if "gold" in k else "test_rm_mean"],
                    "o", color="black", ms=6, label="reference policy")
    full = [(k, v) for k, v in groups.items() if not k[3]]
    for i, (key, seeds) in enumerate(full):
        kl = [final(m, "eval_kl") for m in seeds]
        for ax, k in zip(axes, ("eval_p_gold", "eval_score")):
            y = [final(m, k) for m in seeds]
            ax.errorbar(np.mean(kl), np.mean(y), xerr=np.std(kl), yerr=np.std(y), fmt="o",
                        color=COLORS[i % len(COLORS)], ms=7, label=label(key))
    axes[0].set_ylabel("gold P(positive), held-out prompts")
    axes[1].set_ylabel("reward-model score (log-odds)")
    for ax in axes:
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("KL to reference (nats per completion, sampled)")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="upper left")
    fig.suptitle("DPO/IPO on the full preference set against B1's PPO: final held-out evaluation, mean ± std over seeds")
    fig.tight_layout()
    fig.savefig(OUT / "frontier.png", dpi=150)
    plt.close(fig)


def fig_trajectory(groups, best_key) -> None:
    ppo = ppo_groups()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    lines = [(f"DPO ({label(best_key)})", groups[best_key], COLORS[0])]
    for (lr, beta, clip), color in zip(PPO_BEST, (COLORS[1], COLORS[3])):
        if (lr, beta, clip) in ppo:
            lines.append((f"PPO β={beta:g}, clip {clip:g}, lr {lr_str(lr)}", ppo[(lr, beta, clip)], color))
    for name, seeds, color in lines:
        for j, m in enumerate(seeds):
            _, kl = series(m, "eval_kl")
            for ax, k in zip(axes, ("eval_p_gold", "eval_score")):
                _, y = series(m, k)
                ax.plot(kl, y, "-o", color=color, ms=3, alpha=0.8, label=name if j == 0 else None)
    for ax, yl in zip(axes, ("gold P(positive)", "reward-model score")):
        ax.set_xlabel("KL to reference (nats per completion)")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Held-out evaluations during training (one line per seed; each dot is one evaluation)")
    fig.tight_layout()
    fig.savefig(OUT / "trajectory.png", dpi=150)
    plt.close(fig)


def fig_overfit(groups, keys) -> None:
    panels = [("train_loss", "training loss", True), ("held_acc", "held-out pair accuracy", False),
              ("train_chosen_logratio", "chosen log-ratio (train pairs)", False),
              ("train_rejected_logratio", "rejected log-ratio (train pairs)", False),
              ("train_chosen_logp_tok", "chosen log π per token (train pairs)", False),
              ("eval_kl", "KL to reference (sampled)", False),
              ("eval_p_gold", "gold P(positive)", False), ("eval_distinct2", "distinct-2", False)]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for (key, title, logy), ax in zip(panels, axes.flat):
        for i, k in enumerate(keys):
            x, mu, sd = smooth_steps(groups[k], key)
            ax.plot(x, mu, color=COLORS[i], label=label(k) + f" ({len(groups[k])} seeds)")
            ax.fill_between(x, mu - sd, mu + sd, color=COLORS[i], alpha=0.2)
        if key == "train_chosen_logp_tok":
            x, mu, _ = smooth_steps(groups[keys[0]], "train_ref_chosen_logp_tok")
            ax.axhline(mu[0], color="black", lw=0.8, ls=":", label="reference")
        if key.endswith("logratio"):
            ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("update")
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Overfitting a small preference set: DPO against IPO (mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "overfit.png", dpi=150)
    plt.close(fig)


def fig_ipo_tau(groups, pairs: int) -> None:
    ipo = sorted((k, v) for k, v in groups.items() if k[0] == "ipo" and k[3] == pairs)
    dpo = [(k, v) for k, v in groups.items() if k[0] == "dpo" and k[3] == pairs]
    if not ipo:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for (key, seeds) in ipo:
        tau = key[1]
        h = [final(m, "train_margin") for m in seeds]
        kl = [final(m, "eval_kl") for m in seeds]
        axes[0].plot([1 / (2 * tau)] * len(h), h, "o", color=COLORS[1])
        axes[1].plot([1 / (2 * tau)] * len(kl), kl, "o", color=COLORS[1])
    for key, seeds in dpo:
        h = [final(m, "train_margin") for m in seeds]
        kl = [final(m, "eval_kl") for m in seeds]
        axes[0].axhline(np.mean(h), color=COLORS[0], ls="--", label=f"{label(key)}: final margin")
        axes[1].axhline(np.mean(kl), color=COLORS[0], ls="--", label=label(key))
    lim = max(1 / (2 * k[1]) for k, _ in ipo) * 1.3
    axes[0].plot([0.5, lim], [0.5, lim], color="black", lw=0.8, ls=":", label="h = 1/(2τ), IPO's optimum")
    for ax, yl in zip(axes, ("final margin h on the training pairs (nats)", "KL to reference (nats)")):
        ax.set_xscale("log")
        ax.set_xlabel("IPO target margin 1/(2τ) (nats)")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"IPO's τ sets the margin it stops at ({pairs} pairs; one dot per seed)")
    fig.tight_layout()
    fig.savefig(OUT / "ipo_tau.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tables", action="store_true")
    p.add_argument("--overfit-pairs", type=int, default=64)
    a = p.parse_args()
    groups = b2_groups()
    prefs_table()
    main_table(groups)
    overfit_table(groups)
    if a.tables:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    fig_frontier(groups)
    full = {k: v for k, v in groups.items() if not k[3]}
    if full:
        # The full-set configuration with the best mean gold score on the held-out prompts.
        best = max(full, key=lambda k: np.mean([final(m, "eval_p_gold") for m in full[k]]))
        fig_trajectory(groups, best)
    small = [k for k in groups if k[3] == a.overfit_pairs and k[1] == 0.1]
    if small:
        fig_overfit(groups, sorted(small))
    fig_ipo_tau(groups, a.overfit_pairs)
    print(f"\nfigures in {OUT}")


if __name__ == "__main__":
    main()
