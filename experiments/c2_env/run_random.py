"""C2 -- does the action space do anything, and what scale is the reward on?

    python experiments/c2_env/run_random.py --date 2025-08-26

Two questions an environment has to answer before anything is trained in it.

**Does the action matter?** An action space the policy cannot move the outcome
with is worse than no policy at all, because it produces a flat learning curve
that looks like a hard problem. This drives the env with a handful of fixed
coefficient settings and with a uniformly random policy, and reports how far
apart they land.

**What is the reward's scale?** The inventory penalty `eta * q^2` has to be
set against the PnL term, not guessed. Setting `eta` too small makes it
decorative; too large and the best policy is to hold no inventory, which means
not quoting. This reports the per-step distribution of both terms so `eta` can
be chosen against a measurement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "c1_glft_baseline"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import config  # noqa: E402
from env import FEATURES, MarketMakingEnv  # noqa: E402

from rl_ladder.run import Run  # noqa: E402


def episode(env: MarketMakingEnv, policy, seed: int) -> dict:
    """Run one session and summarise it."""
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    rewards, pnls, inventories, observations = [], [], [], [obs]
    half_spreads = []
    truncated = False
    while not truncated:
        action = policy(obs, rng)
        obs, reward, _, truncated, info = env.step(action)
        rewards.append(reward)
        pnls.append(info["pnl_bp"])
        inventories.append(info["inventory"])
        half_spreads.append(info["half_spread_tick"])
        observations.append(obs)
    trades = info["num_trades"]
    equity = info["equity"]
    return {
        "steps": len(rewards),
        "return": float(np.sum(rewards)),
        "pnl_bp": float(np.sum(pnls)),
        "pnl_pct_of_book": float(equity / env.book_size * 100),
        "trades": int(trades),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "pnl_std": float(np.std(pnls)),
        "mean_q2": float(np.mean(np.square(inventories))),
        "max_abs_inventory": float(np.max(np.abs(inventories))),
        "median_half_spread_tick": float(np.median(half_spreads)),
        "obs": np.array(observations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=config.TRAIN_DAYS[0])
    parser.add_argument("--symbol", default="ethusdc")
    parser.add_argument("--seeds", type=int, default=3, help="seeds for the random policy")
    args = parser.parse_args()

    if args.date in config.TEST_DAYS:
        raise SystemExit(f"{args.date} is a held-out day")

    env = MarketMakingEnv(date=args.date, symbol=args.symbol)
    baseline = env.baseline_action()

    def constant(action):
        return lambda obs, rng: action

    def widest(obs, rng):
        # Half spread at its maximum: the "stop quoting" corner of the space.
        action = baseline.copy()
        action[2] = 1.0
        return action

    def tightest(obs, rng):
        action = baseline.copy()
        action[2] = -1.0
        return action

    def uniform(obs, rng):
        return rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)

    policies = [("baseline", constant(baseline), 0), ("widest", widest, 0),
                ("tightest", tightest, 0)]
    policies += [(f"uniform-random s{s}", uniform, s) for s in range(args.seeds)]

    run = Run(phase="c2", name=f"probe-{args.date}",
              config={"date": args.date, "symbol": args.symbol, "eta": env.eta,
                      "features": list(FEATURES)})

    print(f"{args.symbol} {args.date}, one session per row\n")
    print(f"{'policy':>20} {'return':>10} {'PnL bp':>10} {'% of book':>10} {'trades':>9} "
          f"{'half-spr':>9} {'max |q|':>8} {'mean q^2':>9} {'reward sd':>10}")
    rows = []
    for i, (name, policy, seed) in enumerate(policies):
        result = episode(env, policy, seed)
        observations = result.pop("obs")
        result["policy"] = name
        rows.append(result)
        run.log(step=i, **result)
        print(f"{name:>20} {result['return']:10.1f} {result['pnl_bp']:10.1f} "
              f"{result['pnl_pct_of_book']:10.3f} {result['trades']:9d} "
              f"{result['median_half_spread_tick']:9.1f} {result['max_abs_inventory']:8.2f} "
              f"{result['mean_q2']:9.3f} {result['reward_std']:10.4f}", flush=True)
        if name == "baseline":
            describe_observations(observations)
    env.close()

    # Calibrating eta against the PnL term's per-step *standard deviation* is
    # the wrong comparison: that term is close to zero-mean noise while the
    # penalty is one-signed and accumulates over the whole session.  What the
    # two have to be commensurate in is their episode totals.
    baseline_row = next(r for r in rows if r["policy"] == "baseline")
    steps = baseline_row["steps"]
    pnl_total = abs(baseline_row["pnl_bp"])
    q2_total = baseline_row["mean_q2"] * steps
    pnl_std = float(np.median([r["pnl_std"] for r in rows]))
    print("\nreward scale, over a whole session under the baseline policy:")
    print(f"  PnL term          {baseline_row['pnl_bp']:10.1f} bp   "
          f"(per-step sd {pnl_std:.4f} bp, essentially zero-mean)")
    print(f"  sum of q^2        {q2_total:10.1f}      (mean q^2 {baseline_row['mean_q2']:.4f}"
          f" over {steps} steps)")
    for fraction in (0.1, 0.25, 0.5):
        print(f"  eta = {fraction * pnl_total / max(q2_total, 1e-9):.4f} makes the penalty "
              f"{fraction:.0%} of the PnL term over a session")
    suggested = 0.25 * pnl_total / max(q2_total, 1e-9)

    spread = max(r["pnl_pct_of_book"] for r in rows) - min(r["pnl_pct_of_book"] for r in rows)
    print(f"\nthe action space moves the day's PnL over a range of {spread:.2f}% of book,")
    print("so the policy has something to choose between.")

    run.finish({"rows": rows, "pnl_std": pnl_std, "q2_total": q2_total,
                "suggested_eta": suggested, "pnl_range_pct": spread})
    print(f"\nrun dir: {run.dir}")


def describe_observations(observations: np.ndarray) -> None:
    """Every feature's realised range, so nothing is silently saturating."""
    print("\n    observation ranges under the baseline policy:")
    print(f"    {'feature':>18} {'min':>8} {'p05':>8} {'median':>8} {'p95':>8} {'max':>8}")
    for i, name in enumerate(FEATURES):
        column = observations[:, i]
        q = np.percentile(column, [5, 50, 95])
        print(f"    {name:>18} {column.min():8.3f} {q[0]:8.3f} {q[1]:8.3f} {q[2]:8.3f} "
              f"{column.max():8.3f}")

    # Two features that always agree are one feature and a maintenance cost,
    # so the pairs are printed rather than assumed distinct.
    print("\n    most correlated feature pairs:")
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(observations.T)
    pairs = [
        (abs(corr[i, j]), FEATURES[i], FEATURES[j])
        for i in range(len(FEATURES))
        for j in range(i + 1, len(FEATURES))
        if np.isfinite(corr[i, j])
    ]
    for value, a, b in sorted(pairs, reverse=True)[:5]:
        print(f"    {value:6.3f}  {a} / {b}")
    constant = [FEATURES[i] for i in range(len(FEATURES)) if observations[:, i].std() < 1e-6]
    if constant:
        print(f"    constant on this session: {', '.join(constant)}")
    print()


if __name__ == "__main__":
    main()
