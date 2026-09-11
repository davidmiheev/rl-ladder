"""C2 -- equity curves for the fixed corners of the action space.

    python experiments/c2_env/plot_pnl.py --date 2025-08-26

The probe in `run_random.py` reports each policy's session as one number.
This draws the path, which says something the number cannot: whether a
session's result came from a steady bleed, one bad hour, or a position that
never recovered.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "c1_glft_baseline"))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import config as market_config  # noqa: E402
from env import MarketMakingEnv  # noqa: E402

from rl_ladder.run import Run  # noqa: E402


def equity_curve(env: MarketMakingEnv) -> tuple[np.ndarray, np.ndarray]:
    """Hours into the session, and equity as a percentage of the book."""
    record = env.equity_record()
    equity = record["balance"] + record["position"] * record["price"] - record["fee"]
    keep = np.isfinite(equity)
    equity, stamps = equity[keep], record["timestamp"][keep]
    stride = max(1, len(equity) // 2000)
    return ((stamps[::stride] - stamps[0]) / 3.6e12, equity[::stride] / env.book_size * 100)


def run(env: MarketMakingEnv, policy, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    truncated = False
    while not truncated:
        obs, _, _, truncated, info = env.step(policy(obs, rng))
    hours, equity = equity_curve(env)
    return {"hours": hours, "equity": equity, "fills": int(info["num_trades"]),
            "final": float(equity[-1])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=market_config.TRAIN_DAYS[0])
    parser.add_argument("--out", default="docs/assets/c2/pnl.png")
    args = parser.parse_args()

    env = MarketMakingEnv(date=args.date)
    baseline = env.baseline_action()

    def fixed(index: int | None, value: float = 0.0):
        def policy(obs, rng):
            action = baseline.copy()
            if index is not None:
                action[index] = value
            return action
        return policy

    policies = [
        ("baseline (C1 glft-grid)", fixed(None), 0),
        ("widest: half-spread scale at max", fixed(2, 1.0), 0),
        ("tightest: half-spread scale at min", fixed(2, -1.0), 0),
        ("uniform random", lambda obs, rng: rng.uniform(-1, 1, 4).astype(np.float32), 0),
    ]

    run_dir = Run(phase="c2", name=f"pnl-{args.date}", config={"date": args.date})
    curves = []
    for label, policy, seed in policies:
        env = MarketMakingEnv(date=args.date)
        result = run(env, policy, seed)
        env.close()
        result["label"] = label
        curves.append(result)
        print(f"{label:36s} final {result['final']:8.3f}% of book, {result['fills']:7d} fills",
              flush=True)

    plot(curves, args.date, Path(args.out))
    run_dir.finish({"curves": [{k: v for k, v in c.items() if k not in ("hours", "equity")}
                               for c in curves]})


def plot(curves: list[dict], date: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    out.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8),
                                   gridspec_kw={"width_ratios": [3, 2]})
    colours = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[: len(curves)]
    for curve, colour in zip(curves, colours, strict=True):
        ax1.plot(curve["hours"], curve["equity"], color=colour, lw=1.3,
                 label=f"{curve['label']} ({curve['fills']:,} fills)")
    ax1.axhline(0, color="0.4", lw=0.8)
    ax1.set_xlabel(f"hours into {date}")
    ax1.set_ylabel("equity, % of a 100k book")
    ax1.set_title("(a) the action space's corners, over one session")
    ax1.legend(fontsize=7.5, loc="lower left")

    # The tight corner dominates the axis, so the interesting rows get their own.
    for curve, colour in zip(curves, colours, strict=True):
        if curve["final"] < -20:
            continue
        ax2.plot(curve["hours"], curve["equity"], color=colour, lw=1.3)
    ax2.axhline(0, color="0.4", lw=0.8)
    ax2.set_xlabel(f"hours into {date}")
    ax2.set_title("(b) the same, without the tightest corner")

    fig.suptitle("C2: what the coefficient action space can do to a session", y=0.99)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
