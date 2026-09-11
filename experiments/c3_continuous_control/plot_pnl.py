"""C3 -- equity curves for learned policies against the fixed baseline.

    python experiments/c3_continuous_control/plot_pnl.py --days test

Replays a handful of policies over every session in a day set and draws the
equity path, plus the cumulative result across sessions. The summary tables
say a policy lost; these say how. A steady bleed, one catastrophic hour and a
flat line are three different failures, and only the path distinguishes them.

The policies are chosen to span the phase's outcomes rather than to flatter
it: the best held-out seed, the seed that most clearly overfitted, the most
stable and worst algorithm, and one of the seeds that stopped quoting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "a5_continuous_control"))
sys.path.insert(0, str(HERE.parent / "c1_glft_baseline"))
sys.path.insert(0, str(HERE.parent / "c2_env"))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import config as market_config  # noqa: E402
from env import MarketMakingEnv  # noqa: E402
from evaluate import act, load_policy  # noqa: E402

from rl_ladder.run import Run  # noqa: E402

#: (directory glob, label). Chosen to span the outcomes, not to flatter them.
CHOSEN = [
    ("td3-forced-all-s2", "TD3 forced, seed 2 (best held out)"),
    ("ddpg-forced-all-s2", "DDPG forced, seed 2 (best on training days)"),
    ("sac-forced-all-s0", "SAC forced, seed 0 (most stable, worst)"),
    ("td3-forced-no-gamma-s0", "TD3, gamma pinned, seed 0 (stopped quoting)"),
]


def session(date: str, agent, baseline, mask, is_sac, participation) -> dict:
    ranges = MarketMakingEnv.FORCED_PARTICIPATION if participation == "forced" else None
    env = MarketMakingEnv(date=date, allow_held_out=True, action_ranges=ranges)
    obs, _ = env.reset(seed=0)
    truncated = False
    while not truncated:
        action = baseline.copy()
        if agent is not None:
            action[mask] = np.clip(act(agent, obs, is_sac), -1.0, 1.0)
        obs, _, _, truncated, info = env.step(action)
    record = env.equity_record()
    equity = record["balance"] + record["position"] * record["price"] - record["fee"]
    keep = np.isfinite(equity)
    equity, stamps = equity[keep], record["timestamp"][keep]
    stride = max(1, len(equity) // 1200)
    out = {
        "hours": ((stamps[::stride] - stamps[0]) / 3.6e12).tolist(),
        "equity": (equity[::stride] / env.book_size * 100).tolist(),
        "final": float(equity[-1] / env.book_size * 100),
        "fills": int(info["num_trades"]),
    }
    env.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", default="test", choices=["train", "test"])
    parser.add_argument("--out", default="docs/assets/c3/pnl.png")
    args = parser.parse_args()

    days = market_config.TEST_DAYS if args.days == "test" else market_config.TRAIN_DAYS
    env = MarketMakingEnv(date=market_config.TRAIN_DAYS[0])
    baseline_action = env.baseline_action()
    env.close()

    policies = [("C1 baseline (fixed)", None, baseline_action, [], False, "free")]
    for stem, label in CHOSEN:
        matches = sorted(Path("runs/c3").glob(f"{stem}__*"))
        if not matches:
            print(f"skipping {stem}: no run found")
            continue
        agent, baseline, mask, config, participation = load_policy(matches[-1] / "policy.pt")
        policies.append((label, agent, baseline, mask, config.algo == "sac", participation))

    run_dir = Run(phase="c3", name=f"pnl-{args.days}",
                  config={"days": days, "policies": [p[0] for p in policies]})
    results = {}
    for label, agent, baseline, mask, is_sac, participation in policies:
        results[label] = [session(d, agent, baseline, mask, is_sac, participation) for d in days]
        total = sum(s["final"] for s in results[label])
        fills = sum(s["fills"] for s in results[label])
        print(f"{label:44s} total {total:8.2f}% of book over {len(days)} sessions, "
              f"{fills:7d} fills", flush=True)

    plot(results, days, Path(args.out), args.days)
    run_dir.finish({label: [{k: v for k, v in s.items() if k not in ("hours", "equity")}
                            for s in sessions] for label, sessions in results.items()})


def plot(results: dict, days: list[str], out: Path, which: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    out.parent.mkdir(parents=True, exist_ok=True)
    labels = list(results)
    colours = ["black", *plt.get_cmap("tab10")(np.linspace(0, 1, 10))[: len(labels) - 1]]

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.5))
    for i, date in enumerate(days):
        ax = axes.flat[i]
        for label, colour in zip(labels, colours, strict=True):
            s = results[label][i]
            ax.plot(s["hours"], s["equity"], color=colour, lw=1.2,
                    ls="--" if label.startswith("C1") else "-")
        ax.axhline(0, color="0.6", lw=0.7)
        ax.set_title(date, fontsize=10)
        ax.set_xlabel("hours")
        if i % 3 == 0:
            ax.set_ylabel("equity, % of book")

    # Last panel: the cumulative result across sessions.
    ax = axes.flat[len(days)]
    for label, colour in zip(labels, colours, strict=True):
        totals = np.cumsum([s["final"] for s in results[label]])
        ax.plot(range(1, len(days) + 1), totals, marker="o", ms=4, color=colour,
                ls="--" if label.startswith("C1") else "-", label=label)
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_xticks(range(1, len(days) + 1))
    ax.set_xlabel("sessions elapsed")
    ax.set_ylabel("cumulative, % of book")
    ax.set_title("all sessions, cumulative", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")

    fig.suptitle(f"C3: equity paths on {which} sessions, learned policies against the "
                 f"fixed formula", y=0.99)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
