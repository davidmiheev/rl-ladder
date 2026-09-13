"""Does a learned policy's alpha weight track the alpha signal it actually sees?

C5's second item, asked the way it has to be asked. `probe_policy.py` tried
to answer it by sweeping the observation, and its own negative control killed
that: `glft` policies, whose signal feature is identically zero throughout
training, respond to a swept signal exactly as strongly as policies whose
signal is real. Sweeping an input a network never saw vary measures
extrapolation, not learning.

This asks in distribution instead. Replay one held-out session, record what
the policy saw and what it emitted at every step, and correlate them. The
numbers reported are:

    r(signal, alpha)   how the alpha weight moves with the signal
    r(signal, skew)    the interaction C5 actually names -- skew responding
                       to the alpha signal, which is the thing a fixed-form
                       model would need a hand-written term for
    r(inv, skew)       the control. Every one of these policies is trained
                       against an inventory penalty, so a policy that shows
                       nothing here is not responding to anything and its
                       other correlations are noise.
    sd(signal)         the signal's own spread over the session, in the
                       observation's units. A correlation against a feature
                       that barely moved is not a finding.

    python experiments/c6_strategy_zoo/probe_in_session.py --max-policies 8
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
sys.path.insert(0, str(HERE.parents[1] / "src"))

import env as zoo_env  # noqa: E402
import evaluate  # noqa: E402
from train import ACTION_NAMES  # noqa: E402

RUNS = HERE.parents[1] / "runs"
HAS_SIGNAL = {"hfgrid", "apt", "basis", "obi"}


def correlate(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r, or `nan` when either side is constant."""
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def session(policy: dict, date: str, max_steps: int) -> dict:
    ranges = (zoo_env.ZooEnv.FORCED_PARTICIPATION
              if policy["participation"] == "forced" else None)
    env = zoo_env.ZooEnv(date=date, strategy=policy["strategy"], symbol=policy["symbol"],
                         allow_held_out=True, action_ranges=ranges,
                         overrides=policy["overrides"], beta=policy["beta"],
                         max_steps=max_steps)
    obs, _ = env.reset(seed=0)
    signal_index = zoo_env.FEATURES.index("signal")
    inventory_index = zoo_env.FEATURES.index("inventory")
    seen_signal, seen_inventory, emitted = [], [], []
    truncated = False
    while not truncated:
        learned = np.clip(evaluate.act(policy["agent"], obs, policy["is_sac"]), -1.0, 1.0)
        action = policy["baseline"].copy()
        action[policy["mask"]] = learned
        seen_signal.append(float(obs[signal_index]))
        seen_inventory.append(float(obs[inventory_index]))
        obs, _, _, truncated, info = env.step(action)
        emitted.append([info[name] for name in ACTION_NAMES])
    env.close()

    signal = np.array(seen_signal)
    inventory = np.array(seen_inventory)
    acts = np.array(emitted)
    alpha = acts[:, ACTION_NAMES.index("alpha_mult")]
    skew = acts[:, ACTION_NAMES.index("skew_mult")]
    return {
        "steps": len(signal),
        "sd_signal": float(np.std(signal)),
        "r_signal_alpha": correlate(signal, alpha),
        "r_signal_skew": correlate(signal, skew),
        "r_inventory_skew": correlate(inventory, skew),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-policies", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=20_000,
                        help="policy steps per session; 0 for the whole session")
    args = parser.parse_args()

    checkpoints = sorted((RUNS / "c6").glob("*/policy.pt"))
    if not checkpoints:
        raise SystemExit("no trained policies under runs/c6")

    # One policy per (strategy, asset), and the signal-carrying ones first, so
    # a truncated run still covers the question rather than one strategy.
    seen: set[tuple[str, str]] = set()
    chosen = []
    for with_signal in (True, False):
        for checkpoint in checkpoints:
            name = checkpoint.parent.name
            base = name.split("-wide")[0].split("-td3")[0]
            if (base in HAS_SIGNAL) != with_signal:
                continue
            key = (base, name.split("-td3")[0].rsplit("-", 1)[-1])
            if key in seen:
                continue
            seen.add(key)
            chosen.append(checkpoint)
    chosen = chosen[: args.max_policies]

    header = (f"{'policy':<36} {'sig?':>5} {'steps':>7} {'sd sig':>7} "
              f"{'r(sig,alpha)':>13} {'r(sig,skew)':>12} {'r(inv,skew)':>12}")
    print(header)
    print("-" * len(header))
    rows = []
    for checkpoint in chosen:
        policy = evaluate.load_policy(checkpoint)
        base = policy["strategy"].split("-")[0]
        date = evaluate.held_out_days(policy["symbol"])[0]
        result = session(policy, date, args.max_steps)
        result |= {"name": checkpoint.parent.name.split("__")[0], "signal": base in HAS_SIGNAL}
        rows.append(result)
        print(f"{result['name'][:36]:<36} {'yes' if result['signal'] else 'no':>5} "
              f"{result['steps']:7d} {result['sd_signal']:7.3f} "
              f"{result['r_signal_alpha']:13.3f} {result['r_signal_skew']:12.3f} "
              f"{result['r_inventory_skew']:12.3f}", flush=True)

    for label, group in (("with a signal", [r for r in rows if r["signal"]]),
                         ("without one", [r for r in rows if not r["signal"]])):
        if not group:
            continue
        print(f"\n{label}: median |r(signal, alpha)| = "
              f"{np.nanmedian([abs(r['r_signal_alpha']) for r in group]):.3f}, "
              f"|r(signal, skew)| = {np.nanmedian([abs(r['r_signal_skew']) for r in group]):.3f}, "
              f"control |r(inventory, skew)| = "
              f"{np.nanmedian([abs(r['r_inventory_skew']) for r in group]):.3f}")


if __name__ == "__main__":
    main()
