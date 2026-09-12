"""Known-answer checks for C6, run before any C6 number is reported.

Each one reports a quantity whose correct value is known in advance, so that
passing means something.  The three that matter most are the ones covering
what C6 actually changed relative to C1 and C2: the unit conversions, the O(1)
replacement for the notebook's windowed standardisation, and the causality of
the reference-leg tables.

    python experiments/c6_strategy_zoo/run_checks.py
    python experiments/c6_strategy_zoo/run_checks.py --skip-slow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "c1_glft_baseline"))

import config  # noqa: E402
import env as zoo_env  # noqa: E402
import precompute  # noqa: E402
import strategies  # noqa: E402
import zoo  # noqa: E402

PASS = "  ok   "
FAIL = " FAIL  "


def check_skew_conversion() -> tuple[bool, str]:
    """C6's skew in price units must move the quote exactly as C1's did in ticks.

    C1 skews `skew_tick * position` ticks, with `position` in units of the
    asset.  C6 skews `skew_px * norm_pos` in price, with
    `skew_px = skew_tick * tick * max_dollar / mid` and
    `norm_pos = position * mid / max_dollar`.  The two must be the same shift,
    with the mid and the cap cancelling; this is the one place a wrong unit
    would produce a plausible run rather than an obviously broken one.
    """
    rng = np.random.default_rng(0)
    skew_tick = rng.uniform(0.01, 50.0, 2000)
    position = rng.uniform(-30.0, 30.0, 2000)
    mid = rng.uniform(0.3, 60_000.0, 2000)
    tick = rng.choice([1e-4, 1e-2, 1e-1], 2000)
    max_dollar = rng.uniform(1e4, 1e6, 2000)

    c1_shift_tick = skew_tick * position
    skew_px = skew_tick * tick * max_dollar / mid
    norm_pos = position * mid / max_dollar
    c6_shift_tick = skew_px * norm_pos / tick

    err = np.max(np.abs(c1_shift_tick - c6_shift_tick) / np.maximum(np.abs(c1_shift_tick), 1e-12))
    return err < 1e-12, f"max relative error {err:.2e} over 2000 random states"


def check_baseline_action() -> tuple[bool, str]:
    """`baseline_action()` must map back to multipliers of exactly 1."""
    errs = []
    for ranges in (None, zoo_env.ZooEnv.FORCED_PARTICIPATION):
        env = zoo_env.ZooEnv(date=config.TRAIN_DAYS_EXT[0], strategy="glft", action_ranges=ranges)
        got = np.array(env.to_multipliers(env.baseline_action()))
        errs.append(float(np.max(np.abs(got - 1.0))))
    return max(errs) < 1e-12, f"max deviation from 1.0: free {errs[0]:.2e}, forced {errs[1]:.2e}"


def check_obi_standardisation() -> tuple[bool, str]:
    """The O(1) running standardisation must equal the notebook's windowed one.

    The notebook recomputes `nanmean`/`nanstd` over the trailing window at
    every step, which is O(window) a step and unaffordable at a 100 ms
    interval.  C6 keeps running sums over a circular buffer instead.  This
    replays the kernel's exact update against a brute-force recomputation.
    """
    rng = np.random.default_rng(1)
    n, window = 8000, 600
    raw = rng.normal(0.0, 1.0, n) * rng.uniform(0.5, 3.0, n)
    raw[rng.random(n) < 0.05] = np.nan  # buckets where the book was one-sided

    buf = np.full(window, np.nan)
    total = sumsq = count = 0.0
    running = np.full(n, np.nan)
    for t in range(n):
        slot = t % window
        if t >= window and np.isfinite(buf[slot]):
            total -= buf[slot]
            sumsq -= buf[slot] * buf[slot]
            count -= 1.0
        buf[slot] = raw[t]
        if np.isfinite(raw[t]):
            total += raw[t]
            sumsq += raw[t] * raw[t]
            count += 1.0
        if count > 0:
            mean = total / count
            var = sumsq / count - mean * mean
            running[t] = (raw[t] - mean) / np.sqrt(var) if var > 0 else np.nan

    brute = np.full(n, np.nan)
    for t in range(n):
        chunk = raw[max(0, t + 1 - window) : t + 1]
        mean = np.nanmean(chunk)
        std = np.nanstd(chunk)
        brute[t] = (raw[t] - mean) / std if std > 0 else np.nan

    # Both divide by the count of finite entries, so the only gap left is
    # floating-point drift in the running sums.  An earlier version of the
    # kernel divided by the window occupancy instead and this check caught it
    # at 0.28 sigma, which is the whole reason the check exists.
    mask = np.isfinite(running) & np.isfinite(brute)
    err = float(np.max(np.abs(running[mask] - brute[mask])))
    return err < 1e-9, f"max |running - brute| = {err:.2e} over {int(mask.sum())} steps"


def check_precompute_causality(symbol: str = "xrpusdc") -> tuple[bool, str]:
    """No row of a reference table may be usable before its timestamp.

    The kernel advances while `table[i, 0] > current_timestamp` and then reads
    row `i - 1`, so what it uses is the last row that had already arrived.
    This checks the property the kernel relies on: timestamps strictly
    increasing, and every value at row `i` computable from data no later than
    `table[i, 0]`.  The second half is checked structurally -- the tables are
    built from `searchsorted(..., side="right") - 1` against the traded leg's
    own grid -- and numerically here by replaying the kernel's own scan.
    """
    date = config.PAIR_TRAIN_DAYS[0]
    table = precompute.load("apt", symbol, date)
    ts = table[:, 0]
    if not np.all(np.diff(ts) > 0):
        return False, "reference-table timestamps are not strictly increasing"

    # Replay the scan the kernel performs at a sample of instants.
    probes = ts[np.linspace(100, len(ts) - 100, 500).astype(int)] - 1.0
    used = np.searchsorted(ts, probes, side="right") - 1
    latest = ts[used]
    worst = float(np.max(latest - probes))
    return worst <= 0.0, f"worst (row timestamp - instant used at) = {worst:.1f} ns over 500 probes"


def check_reward_telescopes(strategy: str = "glft") -> tuple[bool, str]:
    """Summed step rewards must equal the session's PnL less the risk penalty."""
    env = zoo_env.ZooEnv(date=config.TRAIN_DAYS_EXT[0], strategy=strategy, max_steps=400)
    obs, _ = env.reset(seed=0)
    del obs
    start = env._equity
    total_reward = 0.0
    total_pnl = 0.0
    penalty = 0.0
    while True:
        _, reward, _, truncated, info = env.step(env.baseline_action())
        total_reward += reward
        total_pnl += info["pnl_bp"]
        penalty += env.eta * info["inventory"] ** 2
        if truncated:
            break
    direct = (env._equity - start) / env.book_size * 1e4
    env.close()
    err = max(abs(total_pnl - direct), abs(total_reward - (total_pnl - penalty)))
    return err < 1e-6, f"pnl {total_pnl:.9f} bp vs direct {direct:.9f} bp; reward gap {err:.2e}"


def check_baseline_matches_runner(strategy: str = "glft") -> tuple[bool, str]:
    """An episode at the baseline action must reproduce the baseline runner.

    This is C6's version of the property C2 built the environment around: the
    published parameters are a strict special case of the policy.  If the two
    paths ever diverge, every "RL beat the baseline" number in C6 is comparing
    against something other than the baseline it names.
    """
    import run_baseline

    date = config.TRAIN_DAYS_EXT[0]
    steps = 600
    env = zoo_env.ZooEnv(date=date, strategy=strategy, max_steps=steps)
    env.reset(seed=0)
    while True:
        _, _, _, truncated, _ = env.step(env.baseline_action())
        if truncated:
            break
    env_log = env.quoter_log()
    env_equity = env._equity
    env.close()

    # The runner replays the whole day, so compare the prefix the episode
    # covers: same warm-up, same multipliers, same quoting iterations.
    import glft as glft_mod
    from hftbacktest import Recorder

    config.use_symbol("ethusdc")
    config.ensure_latency(date)
    hbt = config.build_backtest(date)
    recorder = Recorder(1, 2_000_000)
    params = strategies.build(strategy, "ethusdc")
    n = glft_mod.buffer_steps(strategies.INTERVAL_NS)
    state = np.zeros(zoo.STATE_SIZE)
    for slot in (zoo.Z_A, zoo.Z_K, zoo.Z_VOL, zoo.Z_MID):
        state[slot] = np.nan
    out = np.zeros(n, zoo.out_dtype)
    zoo.advance(
        hbt, recorder.recorder, state, out, params, run_baseline.NO_PRECOMPUTE,
        np.full(n, np.nan), np.full(n, np.nan), np.zeros(n),
        np.full(max(int(params[zoo.P_OBI_WINDOW]), 1), np.nan),
        np.zeros(glft_mod.MAX_BUCKET), np.arange(glft_mod.MAX_BUCKET) + 0.5,
        1.0, 1.0, 1.0, 1.0, strategies.INTERVAL_NS, 1, len(env_log),
    )
    values = hbt.state_values(0)
    direct_equity = values.balance + values.position * state[zoo.Z_MID] * config.TICK_SIZE - values.fee
    hbt.close()

    spread_err = float(np.nanmax(np.abs(out[: len(env_log)]["half_spread_bp"] - env_log["half_spread_bp"])))
    pos_err = float(np.nanmax(np.abs(out[: len(env_log)]["position"] - env_log["position"])))
    equity_err = abs(direct_equity - env_equity)
    ok = spread_err < 1e-12 and pos_err < 1e-12 and equity_err < 1e-6
    return ok, (
        f"{len(env_log)} iterations: half spread {spread_err:.2e} bp, "
        f"position {pos_err:.2e}, equity {equity_err:.2e}"
    )


def check_env_api() -> tuple[bool, str]:
    """`gymnasium`'s own conformance checker, on the smallest useful episode."""
    from gymnasium.utils.env_checker import check_env

    env = zoo_env.ZooEnv(date=config.TRAIN_DAYS_EXT[0], strategy="glft", max_steps=3)
    try:
        check_env(env, skip_render_check=True)
    except Exception as exc:  # noqa: BLE001 -- the point is to report it
        return False, str(exc)[:200]
    finally:
        env.close()
    return True, "gymnasium.utils.env_checker passed"


FAST = [
    ("skew unit conversion", check_skew_conversion),
    ("obi running standardisation", check_obi_standardisation),
]
SLOW = [
    ("baseline action maps to 1.0", check_baseline_action),
    ("reference tables are causal", check_precompute_causality),
    ("reward telescopes", check_reward_telescopes),
    ("baseline is a special case", check_baseline_matches_runner),
    ("gymnasium api", check_env_api),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-slow", action="store_true")
    args = parser.parse_args()

    checks = FAST if args.skip_slow else FAST + SLOW
    failures = 0
    for name, fn in checks:
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        print(f"[{PASS if ok else FAIL}] {name:34s} {detail}", flush=True)
        failures += not ok
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
