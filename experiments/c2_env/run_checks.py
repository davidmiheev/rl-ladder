"""C2 -- four checks on the environment, each against an answer known in advance.

    python experiments/c2_env/run_checks.py

An environment is the one piece of Part C that every later result passes
through, and its failures are silent: a leaked future, a reward that does not
add up, an action space that cannot express the baseline. None of those show
up as an exception. They show up as an unusually good learning curve.

1. **The baseline is a strict special case.** Driving the env with the constant
   action whose coefficients are C1's `glft-grid` must reproduce that backtest
   exactly -- same balance, same position, same fees, same trade count. This is
   the claim the whole "did RL help" comparison rests on, so it is tested
   rather than asserted.
2. **No lookahead.** Replaying a truncated copy of the same session must
   produce the same observations, step for step, until the truncation. If any
   feature saw the future, the two runs would disagree from the first step.
3. **The reward adds up.** The PnL part of the reward must telescope to the
   change in mark-to-market equity over the episode.
4. **Gymnasium conformance**, via `gymnasium.utils.env_checker`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from hftbacktest import Recorder

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "c1_glft_baseline"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import config  # noqa: E402
import glft  # noqa: E402
from env import MarketMakingEnv  # noqa: E402

DAY = "2025-08-26"

#: Tolerance on the action -> coefficient round trip.  Gymnasium action spaces
#: are float32 by convention and a policy emits float32, so ~1e-7 relative is
#: the best any action can do; demanding more would be testing the dtype, not
#: the mapping.  The check that actually binds is the backtest comparison
#: below, which is exact in trade count and at float64 noise in balance.
MAPPING_TOLERANCE = 1e-6

#: The lookahead check needs a truncated copy of a session, and truncating a
#: 60-million-row ETHUSDC day costs minutes.  ENAUSDC has the same structure
#: with a tenth of the rows, and the property under test is the environment's,
#: not the asset's.
SMALL_SYMBOL = "enausdc"
TRUNCATE_HOURS = 1.5


def check_baseline_is_special_case() -> dict[str, float]:
    """The env under a constant baseline action == C1's `glft-grid` backtest."""
    config.use_symbol("ethusdc")
    config.ensure_latency(DAY)

    hbt = config.build_backtest(DAY)
    recorder = Recorder(1, 1_000_000)
    glft.gridtrading_glft_mm(
        hbt, recorder.recorder, 0.05, 1.0, 1.0, 0.05, 1.0, 20.0, 20, glft.INTERVAL_NS, 1
    )
    reference = hbt.state_values(0)
    want = (reference.balance, reference.position, reference.fee, reference.num_trades)
    hbt.close()

    env = MarketMakingEnv(date=DAY, symbol="ethusdc")
    action = env.baseline_action()
    coefficients = env.to_coefficients(action)
    mapping_error = max(abs(a - b) for a, b in zip(coefficients, env.BASELINE, strict=True))

    env.reset(seed=0)
    truncated = False
    while not truncated:
        _, _, _, truncated, _ = env.step(action)
    state = env._hbt.state_values(0)
    got = (state.balance, state.position, state.fee, state.num_trades)
    env.close()

    print(f"    action -> coefficients round trip: {mapping_error:.2e} (float32 action space)")
    for name, a, b in zip(("balance", "position", "fee", "trades"), want, got, strict=True):
        print(f"    {name:9s} reference {a:>18.6f}   env {b:>18.6f}   diff {abs(a - b):.3e}")
    return {
        "mapping_error": mapping_error,
        "balance_diff": abs(want[0] - got[0]),
        "position_diff": abs(want[1] - got[1]),
        "fee_diff": abs(want[2] - got[2]),
        "trades_diff": abs(want[3] - got[3]),
    }


def _truncated_day(symbol: str, hours: float) -> str:
    """Write a copy of the session cut off after `hours`, once."""
    config.use_symbol(symbol)
    date = f"{DAY}-first{hours:g}h"
    out = config.DATA / f"{symbol}_{date}.npz"
    if not out.exists():
        data = np.load(config.DATA / f"{symbol}_{DAY}.npz")["data"]
        cutoff = data["local_ts"][0] + int(hours * 3600 * 1e9)
        np.savez_compressed(out, data=data[data["local_ts"] <= cutoff])
        del data
    config.ensure_latency(date)
    return date


def check_no_lookahead(steps: int = 900) -> dict[str, float]:
    """Observations must not change when the future is removed."""
    config.register_symbol(SMALL_SYMBOL, 0.0001, 1.0, roi_ub=4.0)
    short_date = _truncated_day(SMALL_SYMBOL, TRUNCATE_HOURS)

    runs = []
    for date in (DAY, short_date):
        env = MarketMakingEnv(
            date=date, symbol=SMALL_SYMBOL, order_qty=1000.0, max_position=20000.0
        )
        obs, _ = env.reset(seed=0)
        collected = [obs]
        action = env.baseline_action()
        for _ in range(steps):
            obs, _, _, truncated, _ = env.step(action)
            collected.append(obs)
            if truncated:
                break
        env.close()
        runs.append(np.array(collected))

    overlap = min(len(runs[0]), len(runs[1]))
    worst = float(np.max(np.abs(runs[0][:overlap] - runs[1][:overlap]))) if overlap else float("inf")
    print(f"    compared {overlap} observations of {len(FEATURES_NAMES)} features")
    if overlap and worst > 0:
        column = int(np.argmax(np.max(np.abs(runs[0][:overlap] - runs[1][:overlap]), axis=0)))
        print(f"    worst feature: {FEATURES_NAMES[column]}")
    return {"compared": overlap, "max_abs_diff": worst}


def check_reward_telescopes(steps: int = 600) -> dict[str, float]:
    """Summed PnL rewards must equal the change in equity."""
    env = MarketMakingEnv(date=DAY, symbol="ethusdc")
    env.reset(seed=0)
    start = env._equity
    action = env.baseline_action()
    total = 0.0
    for _ in range(steps):
        _, _, _, truncated, info = env.step(action)
        total += info["pnl_bp"]
        if truncated:
            break
    end = env._equity
    env.close()
    direct = (end - start) / 100_000.0 * 1e4
    print(f"    summed rewards {total:.9f} bp   direct {direct:.9f} bp")
    return {"telescoping_error": abs(total - direct)}


def check_gym_api() -> dict[str, float]:
    """Gymnasium's own conformance checker."""
    from gymnasium.utils.env_checker import check_env

    env = MarketMakingEnv(date=DAY, symbol=SMALL_SYMBOL, order_qty=1000.0, max_position=20000.0)
    check_env(env, skip_render_check=True)
    env.close()
    print("    gymnasium.utils.env_checker: passed")
    return {"gym_api": 0.0}


FEATURES_NAMES: tuple[str, ...] = ()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-baseline", action="store_true",
                        help="skip check 1, which replays a whole session twice")
    args = parser.parse_args()

    global FEATURES_NAMES
    from env import FEATURES

    FEATURES_NAMES = FEATURES

    results: dict[str, float] = {}
    print("2. no lookahead")
    results |= check_no_lookahead()
    print("3. the reward telescopes")
    results |= check_reward_telescopes()
    print("4. gymnasium conformance")
    results |= check_gym_api()
    if not args.skip_baseline:
        print("1. the baseline is a strict special case (replays a full session twice)")
        results |= check_baseline_is_special_case()

    print("\nverdict")
    failures = []
    if results["max_abs_diff"] > 0:
        failures.append("observations depend on data after the step")
    if results["telescoping_error"] > 1e-6:
        failures.append("reward does not telescope to the equity change")
    if not args.skip_baseline:
        if results["mapping_error"] > MAPPING_TOLERANCE:
            failures.append("baseline_action does not map to BASELINE")
        for name in ("balance", "position", "fee", "trades"):
            if results[f"{name}_diff"] > 1e-6:
                failures.append(f"env does not reproduce glft-grid's {name}")
    if failures:
        for line in failures:
            print(f"  FAILED: {line}")
        raise SystemExit(1)
    print("  all checks passed")


if __name__ == "__main__":
    main()
