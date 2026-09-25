"""C4, step 2: GRPO-style against PPO-style search over three fixed parameters.

The action is a three-dimensional vector chosen once per session,

    theta = (log floor_mult, log skew_mult, kappa)

mapped to the widened preset exactly as `grid_search.py` maps its grid, and
the policy is a diagonal Gaussian over it, started on C6's widened baseline
(floor x1, skew x1, kappa 0). One "episode" is one training session replayed
with the sampled parameters; its return is the reward.

Both methods take the same PPO-clipped policy-gradient step on the same K
samples per iteration. They differ only in where the samples are evaluated and
what the baseline is -- which is the question C4 asks:

    grpo   all K samples replay the **same** training session, and the
           advantage is the sample's return standardised within its group.
           The group mean is the baseline; no value function exists.
    ppo    the K samples replay K **independently drawn** sessions, and the
           advantage is the return minus a learned per-session value (an
           exponential average of what that session has paid so far), then
           standardised over the batch. The critic is tabular because the
           context is: ten training sessions, nothing else.

Returns are in bp of book. A sample taking fewer than MIN_FILLS fills is
charged up to NO_TRADE_PENALTY bp, linearly, so that "stop trading" is not a
free answer; C3 found that corner the first time the action space allowed it.

Budget is counted in session replays, the unit the grid search pays in.

    python experiments/c4_ppo_grpo/param_search.py --method grpo --strategy obi \\
        --symbol ethusdc --seed 0
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "c6_strategy_zoo"))
sys.path.insert(0, str(REPO / "experiments" / "c1_glft_baseline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import assets  # noqa: E402
import grid_search  # noqa: E402

from rl_ladder.run import Run  # noqa: E402

K = 6
ITERATIONS = 20
EPOCHS = 4
CLIP = 0.2
LR = 0.05
#: Initial spread of the search, per dimension: a factor of e^0.3 on the floor,
#: e^0.5 on the skew, and 1.5 of kappa.
SIGMA0 = np.array([0.3, 0.5, 1.5])
MIN_FILLS = grid_search.MIN_FILLS
NO_TRADE_PENALTY = 10.0
VALUE_DECAY = 0.3


def to_point(theta: np.ndarray) -> tuple[float, float, float]:
    """theta -> (floor_mult, kappa, skew_mult), clipped to a sane box."""
    floor_mult = float(np.exp(np.clip(theta[0], np.log(0.5), np.log(4.0))))
    skew_mult = float(np.exp(np.clip(theta[1], np.log(0.25), np.log(4.0))))
    kappa = float(np.clip(theta[2], 0.0, 8.0))
    return floor_mult, kappa, skew_mult


def evaluate(args: tuple) -> tuple[float, float]:
    """Replay one session with one parameter point; (return in bp, fills)."""
    import run_baseline

    strategy, symbol, date, point = args
    sets = grid_search.point_sets(strategy, symbol, *point)
    stats, record, out = run_baseline.run_day(f"{strategy}-wide", symbol, date,
                                              {"beta": 1.0, **sets})
    result = float(stats["Return"]) * 1e4, float(stats["DailyNumberOfTrades"])
    del record, out
    gc.collect()
    return result


def reward(return_bp: float, fills: float) -> float:
    shortfall = max(0.0, 1.0 - fills / MIN_FILLS)
    return return_bp - NO_TRADE_PENALTY * shortfall


def log_prob(theta: np.ndarray, mu: np.ndarray, log_sigma: np.ndarray) -> np.ndarray:
    z = (theta - mu) / np.exp(log_sigma)
    return (-0.5 * z**2 - log_sigma).sum(axis=-1)


def ppo_step(mu, log_sigma, thetas, advantages, adam):
    """EPOCHS of PPO-clipped ascent on the Gaussian's parameters.

    With one action per episode the ratio is pi_new(theta)/pi_old(theta), and
    the gradient of log pi for a diagonal Gaussian is closed form, so no
    autograd is needed: d/dmu = z/sigma, d/dlog_sigma = z^2 - 1.
    """
    old = log_prob(thetas, mu, log_sigma)
    for _ in range(EPOCHS):
        ratio = np.exp(log_prob(thetas, mu, log_sigma) - old)
        active = np.where(advantages >= 0, ratio < 1 + CLIP, ratio > 1 - CLIP)
        weight = (ratio * advantages * active)[:, None]
        sigma = np.exp(log_sigma)
        z = (thetas - mu) / sigma
        grad_mu = (weight * z / sigma).mean(axis=0)
        grad_ls = (weight * (z**2 - 1.0)).mean(axis=0)
        mu, log_sigma = adam.step(mu, log_sigma, grad_mu, grad_ls)
    return mu, log_sigma


class Adam:
    def __init__(self, lr: float) -> None:
        self.lr, self.t = lr, 0
        self.m = [np.zeros(3), np.zeros(3)]
        self.v = [np.zeros(3), np.zeros(3)]

    def step(self, mu, log_sigma, g_mu, g_ls):
        self.t += 1
        out = []
        for i, (p, g) in enumerate(((mu, g_mu), (log_sigma, g_ls))):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g**2
            m_hat = self.m[i] / (1 - 0.9**self.t)
            v_hat = self.v[i] / (1 - 0.999**self.t)
            out.append(p + self.lr * m_hat / (np.sqrt(v_hat) + 1e-8))  # ascent
        return out[0], out[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=["grpo", "ppo"])
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=ITERATIONS,
                        help="smoke tests only; the comparison uses the default")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    days = assets.train_days(args.symbol)
    mu, log_sigma = np.zeros(3), np.log(SIGMA0)
    adam = Adam(LR)
    value = {d: 0.0 for d in days}
    seen = {d: False for d in days}
    run = Run("c4-search", f"{args.method}-{args.strategy}-{args.symbol}-s{args.seed}", config={
        "method": args.method, "strategy": args.strategy, "symbol": args.symbol,
        "seed": args.seed, "K": K, "iterations": args.iterations, "epochs": EPOCHS, "clip": CLIP,
        "lr": LR, "sigma0": SIGMA0.tolist(), "train_days": days, "min_fills": MIN_FILLS,
        "no_trade_penalty": NO_TRADE_PENALTY,
    })
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        for it in range(args.iterations):
            thetas = mu + np.exp(log_sigma) * rng.standard_normal((K, 3))
            if args.method == "grpo":
                sessions = [days[rng.integers(len(days))]] * K
            else:
                sessions = [days[i] for i in rng.integers(len(days), size=K)]
            started = time.time()
            results = pool.map(evaluate, [(args.strategy, args.symbol, d, to_point(t))
                                          for d, t in zip(sessions, thetas, strict=True)])
            rewards = np.array([reward(r, f) for r, f in results])
            if args.method == "grpo":
                advantages = rewards - rewards.mean()
            else:
                baseline = np.array([value[d] for d in sessions])
                advantages = rewards - baseline
                for d, r in zip(sessions, rewards, strict=True):
                    value[d] = r if not seen[d] else (1 - VALUE_DECAY) * value[d] + VALUE_DECAY * r
                    seen[d] = True
            advantages = advantages / (advantages.std() + 1e-8)
            mu, log_sigma = ppo_step(mu, log_sigma, thetas, advantages, adam)
            f, k, s = to_point(mu)
            run.log(step=it, mean_reward=float(rewards.mean()), best_reward=float(rewards.max()),
                    floor_mult=f, kappa=k, skew_mult=s, sigma=np.exp(log_sigma).tolist(),
                    sessions=sessions, rewards=rewards.tolist(),
                    fills=[fl for _, fl in results])
            print(f"it {it:2d}  {sessions[0] if args.method == 'grpo' else 'mixed':>10}  "
                  f"mean {rewards.mean():+7.1f} bp  mu: floor x{f:.2f} kappa {k:.2f} "
                  f"skew x{s:.2f}  sigma {np.round(np.exp(log_sigma), 2)}  "
                  f"{time.time() - started:.0f}s", flush=True)

    f, k, s = to_point(mu)
    run.finish({"floor_mult": f, "kappa": k, "skew_mult": s, "mu": mu.tolist(),
                "sigma": np.exp(log_sigma).tolist(), "replays": K * args.iterations})
    print(json.dumps({"floor_mult": f, "kappa": k, "skew_mult": s}))


if __name__ == "__main__":
    main()
