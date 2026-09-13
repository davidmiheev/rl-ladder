"""Does a learned policy respond to the alpha signal, or ignore it?

C5's second item asks the question Part C is really about: does RL discover
the *interaction* between an alpha signal and the quoting parameters on its
own, or does it just reproduce what a person would have written by hand? If
the latter, the analytic model with a manual extension is the better
engineering choice, and knowing that is worth the experiment.

The tempting shortcut is that no backtest is needed: the actor is a
deterministic function from the eleven-feature observation to the four
multipliers, so hold every feature neutral, sweep one, and read off the
response.

**That shortcut does not work, and this script exists to show why.** `glft`
quotes off the mid, so its `signal` feature is identically zero in every step
of training -- its policy cannot have learned anything about that input. Yet
swept the same way, `glft` policies move `alpha_mult` across 181% of its
range, against 181% for the strategies whose signal is real. The response is
the network extrapolating on an input it never saw vary, not evidence of
anything learned.

So a large sensitivity is not evidence, and this measurement cannot answer
C5's item on its own. It is kept because the failing control is the finding:
any reading of a learned policy by sweeping its inputs has to show that the
same sweep gives a different answer where the input is meaningless, and here
it does not. `probe_in_session.py` asks the question the in-distribution way
instead, by correlating what the policy emitted with what it actually saw.

    python experiments/c6_strategy_zoo/probe_policy.py
    python experiments/c6_strategy_zoo/probe_policy.py --feature inventory
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

#: How far to sweep the probed feature, in the observation's own units. The
#: features are scaled so that 1.0 is roughly one typical unit, and the
#: observation is clipped at 10, so +/-3 spans the range a session actually
#: visits without asking the network to extrapolate.
SWEEP = np.linspace(-3.0, 3.0, 25)

#: Strategies whose fair price carries a signal at all. `glft` quotes off the
#: mid, so its policy has nothing to respond to and it is the second control.
HAS_SIGNAL = {"hfgrid", "apt", "basis", "obi"}


def probe(policy: dict, feature: str) -> np.ndarray:
    """Multipliers as one observation feature is swept, others held neutral."""
    index = zoo_env.FEATURES.index(feature)
    obs = np.zeros((len(SWEEP), len(zoo_env.FEATURES)), dtype=np.float32)
    obs[:, index] = SWEEP
    out = []
    ranges = (zoo_env.ZooEnv.FORCED_PARTICIPATION
              if policy["participation"] == "forced" else None)
    env = zoo_env.ZooEnv(date=evaluate.train_days(policy["symbol"])[0],
                         strategy=policy["strategy"], symbol=policy["symbol"],
                         action_ranges=ranges, overrides=policy["overrides"],
                         beta=policy["beta"])
    for row in obs:
        learned = np.clip(evaluate.act(policy["agent"], row, policy["is_sac"]), -1.0, 1.0)
        action = policy["baseline"].copy()
        action[policy["mask"]] = learned
        out.append(env.to_multipliers(action))
    env.close()
    return np.array(out)


def span(values: np.ndarray) -> float:
    """Relative range of a multiplier over the sweep, as a percentage."""
    lo, hi = float(values.min()), float(values.max())
    return (hi - lo) / max(abs(lo + hi) / 2, 1e-9) * 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", default="signal", choices=list(zoo_env.FEATURES))
    parser.add_argument("--control", default="inventory", choices=list(zoo_env.FEATURES))
    args = parser.parse_args()

    checkpoints = sorted((RUNS / "c6").glob("*/policy.pt"))
    if not checkpoints:
        raise SystemExit("no trained policies under runs/c6")

    print(f"response to `{args.feature}` (control: `{args.control}`), as the percentage")
    print("range each multiplier covers while that one feature sweeps +/-3\n")
    header = (f"{'policy':<34} {'sig?':>5} " +
              " ".join(f"{n.replace('_mult', ''):>8}" for n in ACTION_NAMES) +
              f" {'|':>2} " + " ".join(f"{n.replace('_mult', ''):>8}" for n in ACTION_NAMES))
    print(header)
    print("-" * len(header))

    rows = []
    for checkpoint in checkpoints:
        try:
            policy = evaluate.load_policy(checkpoint)
        except Exception as exc:  # noqa: BLE001 -- report and continue
            print(f"{checkpoint.parent.name[:34]:<34} load failed: {type(exc).__name__}")
            continue
        base = policy["strategy"].split("-")[0]
        probed = probe(policy, args.feature)
        control = probe(policy, args.control)
        name = checkpoint.parent.name.split("__")[0]
        rows.append({"name": name, "signal": base in HAS_SIGNAL,
                     "probed": [span(probed[:, i]) for i in range(len(ACTION_NAMES))],
                     "control": [span(control[:, i]) for i in range(len(ACTION_NAMES))]})
        print(f"{name[:34]:<34} {'yes' if base in HAS_SIGNAL else 'no':>5} " +
              " ".join(f"{v:8.1f}" for v in rows[-1]["probed"]) + f" {'|':>2} " +
              " ".join(f"{v:8.1f}" for v in rows[-1]["control"]), flush=True)

    if not rows:
        return
    with_signal = [r for r in rows if r["signal"]]
    without = [r for r in rows if not r["signal"]]
    print(f"\n{'':34} median response of alpha_mult to `{args.feature}`:")
    alpha = ACTION_NAMES.index("alpha_mult")
    for label, group in (("strategies with a signal", with_signal),
                         ("strategies without one", without)):
        if group:
            print(f"  {label:<30} {np.median([r['probed'][alpha] for r in group]):8.1f}%"
                  f"   (control {np.median([r['control'][alpha] for r in group]):.1f}%)")


if __name__ == "__main__":
    main()
