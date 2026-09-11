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


def plot(test: dict[str, list[dict]], train: dict[str, list[dict]], out: Path) -> None:
    """Two panels: what a policy did, and whether it did it out of sample."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt  # noqa: PLC0415

    families = [f for f in sorted(test) if f != "baseline"]
    colours = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[: len(families)]
    base = test.get("baseline", [{}])[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))

    # (a) held out: Sharpe against how much the policy actually traded.
    for family, colour in zip(families, colours, strict=True):
        for blob in test[family]:
            fills = blob["fills_median"]
            sharpe = blob["SR_median"]
            if fills == 0 or not np.isfinite(sharpe):
                ax1.scatter(0.5, 0, marker="x", s=70, color=colour)
                continue
            ax1.scatter(fills, sharpe, s=55, color=colour, alpha=0.85)
        ax1.scatter([], [], color=colour, label=family)
    if base:
        ax1.scatter(base["fills_median"], base["SR_median"], marker="*", s=320,
                    color="black", zorder=5, label="C1 baseline")
        ax1.axhline(base["SR_median"], color="0.6", lw=0.9, ls="--")
    ax1.set_xscale("symlog", linthresh=100)
    ax1.set_xlabel("fills per held-out session (log; x at zero = stopped quoting)")
    ax1.set_ylabel("Sharpe, median over held-out sessions")
    ax1.set_title("(a) held out: nothing learned clears the fixed formula")
    ax1.legend(fontsize=7, loc="lower right")

    # (b) training days against held-out days, per seed.
    for family, colour in zip(families, colours, strict=True):
        for blob in test[family]:
            label = blob.get("label", "")
            match = next((b for b in train.get(family, []) if b.get("label") == label), None)
            if match is None:
                continue
            x, y = match["Return_median"] * 100, blob["Return_median"] * 100
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            ax2.scatter(x, y, s=55, color=colour, alpha=0.85)
        ax2.scatter([], [], color=colour, label=family)
    if base and train.get("baseline"):
        ax2.scatter(train["baseline"][0]["Return_median"] * 100,
                    base["Return_median"] * 100, marker="*", s=320, color="black",
                    zorder=5, label="C1 baseline")
    lo = min(ax2.get_xlim()[0], ax2.get_ylim()[0])
    hi = max(ax2.get_xlim()[1], ax2.get_ylim()[1])
    ax2.plot([lo, hi], [lo, hi], "k:", lw=1, label="no generalisation gap")
    ax2.axhline(0, color="0.7", lw=0.8)
    ax2.axvline(0, color="0.7", lw=0.8)
    ax2.set_xlabel("return per training session (% of book)")
    ax2.set_ylabel("return per held-out session (% of book)")
    ax2.set_title("(b) the training days were not the hard part")
    ax2.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nwrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", default="test", choices=["train", "test"])
    parser.add_argument("--plot", metavar="PATH", help="also write the two-panel figure here")
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
            fills = blob["fills_median"]
            if fills == 0:
                # Sharpe over an empty equity curve is nan, and printing that
                # invites reading it as "missing" rather than as "this policy
                # stopped quoting", which is the actual result.
                print(f"{family:>26} {seed:>5} {'no trades':>9} {'':>9} "
                      f"{0.0:9.4f} {'':>8} {0:10.0f} {0:10.0f} "
                      f"{blob['days_profitable']}/{blob['n_days']}")
                continue
            print(f"{family:>26} {seed:>5} {blob['SR_median']:9.2f} "
                  f"{blob['Sortino_median']:9.2f} {blob['Return_median']:9.4f} "
                  f"{blob['ReturnOverMDD_median']:8.3f} "
                  f"{blob['MaxPositionValue_median']:10.0f} "
                  f"{fills:10.0f} "
                  f"{blob['days_profitable']}/{blob['n_days']}")
        if len(rows) > 1:
            silent = sum(1 for b in rows if b["fills_median"] == 0)
            for key, fmt in (("SR", "9.2f"), ("Return", "9.4f"), ("fills", "10.0f")):
                values = [b[f"{key}_median"] for b in rows]
                # nanmedian, so a seed that stopped quoting does not erase the
                # others; the count of such seeds is printed separately.
                print(f"{'':>26} {'med':>5} {key}: {np.nanmedian(values):{fmt}} "
                      f"(seeds: {', '.join('n/a' if np.isnan(v) else f'{v:.2f}' for v in values)})")
            if silent:
                print(f"{'':>26} {'':>5} {silent} of {len(rows)} seeds took no trades at all")
        print()

    if args.plot:
        plot(load("test"), load("train"), Path(args.plot))

    print("A configuration whose fills collapse towards zero has learned to stop")
    print("quoting, which scores well on every risk-adjusted measure and is not a")
    print("market maker. Read the fills column before the Sharpe column.")


if __name__ == "__main__":
    main()
