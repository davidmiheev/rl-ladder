"""Build the baseline reward traces the counterfactual recipes subtract.

    python experiments/c6_strategy_zoo/build_traces.py --strategy glft-wide --symbol xrpusdc

One replay of each training session at the baseline action, cached under
`data/precompute/`. Training builds a missing trace on first use anyway; this
exists so that five seeds starting at once do not all build the same one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "c1_glft_baseline"))

import assets  # noqa: E402
import env as zoo_env  # noqa: E402

STEP_NS = 5_000_000_000
ETA = 0.015


def trace_path(strategy: str, symbol: str, date: str) -> Path:
    return (zoo_env.precompute.PRECOMP /
            f"cf_{strategy}_{symbol}_{date}_{STEP_NS / 1e9:g}s_eta{ETA:g}_b1_none.npy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--symbol", required=True)
    args = parser.parse_args()
    assets.ensure_registered(args.symbol)
    for date in assets.train_days(args.symbol):
        trace = zoo_env.baseline_trace(args.strategy, args.symbol, date, STEP_NS, ETA, {}, 1.0)
        print(f"{args.strategy} {args.symbol} {date}: {len(trace)} steps, "
              f"session baseline reward {trace.sum():+.2f} bp", flush=True)


if __name__ == "__main__":
    main()
