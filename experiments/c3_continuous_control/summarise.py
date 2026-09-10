"""C3 -- collect the evaluation runs into one table.

    python experiments/c3_continuous_control/summarise.py --days test

Reads every `eval-*` run under `runs/c3/` and prints them together with C1's
fixed baseline, so item 2's question -- do learned coefficients beat a
well-calibrated fixed formula on days neither saw -- is answered by one table
rather than by comparing two documents.

Seeds are reported as separate rows and then as a median, never as a mean:
five held-out days is few enough that one session decides a mean, which is the
mistake Part A kept making.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "runs" / "c3"


def load(days: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(ROOT.glob(f"eval-*-{days}__*/summary.json")):
        blob = json.loads(path.read_text())
        label = blob.get("label", path.parent.name)
        # Strip the seed so the seeds of one configuration group together.
        family = label.rsplit("-s", 1)[0] if "-s" in label else label
        grouped[family].append(blob)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", default="test", choices=["train", "test"])
    args = parser.parse_args()

    grouped = load(args.days)
    if not grouped:
        raise SystemExit(f"no eval runs for {args.days} days under {ROOT}")

    print(f"{args.days} days, median over sessions; one row per seed\n")
    print(f"{'configuration':>26} {'seed':>5} {'SR':>9} {'Sortino':>9} {'Return':>9} "
          f"{'RoMDD':>8} {'MaxPos':>10} {'fills/day':>10} {'days +':>7}")
    for family in sorted(grouped):
        rows = grouped[family]
        for blob in rows:
            label = blob.get("label", "")
            seed = label.rsplit("-s", 1)[1] if "-s" in label else "-"
            print(f"{family:>26} {seed:>5} {blob['SR_median']:9.2f} "
                  f"{blob['Sortino_median']:9.2f} {blob['Return_median']:9.4f} "
                  f"{blob['ReturnOverMDD_median']:8.3f} "
                  f"{blob['MaxPositionValue_median']:10.0f} "
                  f"{blob['fills_median']:10.0f} "
                  f"{blob['days_profitable']}/{blob['n_days']}")
        if len(rows) > 1:
            for key, fmt in (("SR", "9.2f"), ("Return", "9.4f"), ("fills", "10.0f")):
                values = [b[f"{key}_median"] for b in rows]
                print(f"{'':>26} {'med':>5} {key}: {np.median(values):{fmt}} "
                      f"(seeds: {', '.join(f'{v:.2f}' for v in values)})")
        print()

    print("A configuration whose fills collapse towards zero has learned to stop")
    print("quoting, which scores well on every risk-adjusted measure and is not a")
    print("market maker. Read the fills column before the Sharpe column.")


if __name__ == "__main__":
    main()
