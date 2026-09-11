"""C3 -- collect the evaluation runs into one table.

    python experiments/c3_continuous_control/summarise.py --days test

Reads every `eval-*` run under `runs/c3/` and prints them together with C1's
fixed baseline, so item 2's question -- do learned coefficients beat a
well-calibrated fixed formula on days neither saw -- is answered by one table
rather than by comparing two documents.

Two summaries, because they disagree and the disagreement is the result.

`--per-session` medians each statistic over the five sessions: the typical
day. That is what this script reported first, and it is the wrong summary for
a market maker. A median over five days is blind to the tail, and in market
making the tail is the strategy: the fixed baseline earns hundredths of a
percent on quiet days and loses whole percents on volatile ones, so its median
day looks fine and its period is a disaster.

The default is the **period**: the five sessions treated as one track record,
which is what an allocator would see. Cumulative return, the annualised Sharpe
of the daily returns, the worst single session, and total fills. The two
rankings are close to inverses of each other and both are printed.
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
            fills = float(np.sum(blob["fills_per_day"]))
            total = float(np.sum(np.asarray(blob["Return_per_day"], dtype=float)) * 100)
            ax1.scatter(max(fills, 0.5), total, s=55, color=colour, alpha=0.85)
        ax1.scatter([], [], color=colour, label=family)
    if base:
        base_fills = float(np.sum(base["fills_per_day"]))
        base_total = float(np.sum(np.asarray(base["Return_per_day"], dtype=float)) * 100)
        ax1.scatter(base_fills, base_total, marker="*", s=320,
                    color="black", zorder=5, label="C1 baseline")
        ax1.axhline(base_total, color="0.6", lw=0.9, ls="--")
    ax1.axhline(0, color="0.85", lw=0.8)
    ax1.annotate("a handful of fills is not\na market maker", (0.03, 0.55),
                 xycoords="axes fraction", fontsize=8, color="0.35")
    ax1.set_xscale("log")
    ax1.set_xlabel("total fills over the five held-out sessions (log)")
    ax1.set_ylabel("cumulative return over held-out sessions (% of book)")
    ax1.set_title("(a) held out: the fixed formula is near the bottom")
    ax1.legend(fontsize=7, loc="lower left", framealpha=0.9)

    # (b) Cumulative return over each day set, relative to the baseline on
    # that same set.  Relative, because the held-out sessions are not a
    # like-for-like sample -- two of the five are the most volatile in the
    # whole month -- so an absolute train-to-held-out comparison would measure
    # the regime rather than the policy.  Cumulative rather than a median of
    # per-session numbers, because the median is what made the first version
    # of this figure say the opposite of the truth.
    def cumulative(blob):
        return float(np.sum(np.asarray(blob["Return_per_day"], dtype=float)) * 100)

    base_train = cumulative(train["baseline"][0]) if train.get("baseline") else np.nan
    base_test = cumulative(base) if base else np.nan
    for family, colour in zip(families, colours, strict=True):
        for blob in test[family]:
            label = blob.get("label", "")
            match = next((b for b in train.get(family, []) if b.get("label") == label), None)
            if match is None:
                continue
            ax2.scatter(cumulative(match) - base_train, cumulative(blob) - base_test,
                        s=55, color=colour, alpha=0.85)
        ax2.scatter([], [], color=colour, label=family)
    ax2.axhline(0, color="black", lw=1.1)
    ax2.axvline(0, color="black", lw=1.1)
    lo = min(ax2.get_xlim()[0], ax2.get_ylim()[0])
    hi = max(ax2.get_xlim()[1], ax2.get_ylim()[1])
    ax2.plot([lo, hi], [lo, hi], "k:", lw=1, label="same edge on both")
    ax2.annotate("above the dotted line:\nthe edge is larger\nout of sample",
                 (0.04, 0.74), xycoords="axes fraction", fontsize=8, color="0.3")
    ax2.set_xlabel("cumulative return minus baseline, training sessions (% of book)")
    ax2.set_ylabel("cumulative return minus baseline, held-out sessions")
    ax2.set_title("(b) the edge is not confined to the fitted days")
    ax2.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nwrote {out}")


#: Trading days in a year, for annualising a Sharpe built from daily returns.
TRADING_DAYS = 252


def period_stats(blob: dict) -> dict:
    """The five sessions as one track record rather than five experiments."""
    daily = np.array(blob["Return_per_day"], dtype=float) * 100
    fills = np.array(blob["fills_per_day"], dtype=float)
    spread = daily.std(ddof=1)
    return {
        "cumulative_pct": float(daily.sum()),
        "sharpe_annualised": float(daily.mean() / spread * np.sqrt(TRADING_DAYS))
        if spread > 0 else float("nan"),
        "worst_session_pct": float(daily.min()),
        "sessions_positive": int((daily > 0).sum()),
        "fills": int(fills.sum()),
    }


def report_period(grouped: dict[str, list[dict]], days: str) -> None:
    print(f"\n=== {days} sessions as one track record ===\n")
    print(f"{'configuration':>26} {'seed':>5} {'cumulative %':>13} {'Sharpe (ann.)':>14} "
          f"{'worst session %':>16} {'fills':>9} {'up':>5}")
    ranked = []
    for family, rows in grouped.items():
        for blob in rows:
            label = blob.get("label", "")
            seed = label.rsplit("-s", 1)[1] if "-s" in label else "-"
            ranked.append((family, seed, period_stats(blob)))
    ranked.sort(key=lambda row: -row[2]["cumulative_pct"])
    for family, seed, stats in ranked:
        sharpe = ("n/a" if not np.isfinite(stats["sharpe_annualised"])
                  else f"{stats['sharpe_annualised']:14.2f}")
        marker = "  <- fixed baseline" if family == "baseline" else ""
        print(f"{family:>26} {seed:>5} {stats['cumulative_pct']:13.2f} {sharpe:>14} "
              f"{stats['worst_session_pct']:16.2f} {stats['fills']:9d} "
              f"{stats['sessions_positive']:5d}{marker}")
    print("\nA row with a handful of fills did not earn its place at the top; it")
    print("declined to play. Read the fills column before the cumulative column.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", default="test", choices=["train", "test"])
    parser.add_argument("--per-session", action="store_true",
                        help="also print the median-over-sessions table")
    parser.add_argument("--plot", metavar="PATH", help="also write the two-panel figure here")
    args = parser.parse_args()

    grouped = load(args.days)
    if not grouped:
        raise SystemExit(f"no eval runs for {args.days} days under {ROOT}")

    report_period(grouped, args.days)
    if not args.per_session:
        if args.plot:
            plot(load("test"), load("train"), Path(args.plot))
        return

    print(f"\n=== {args.days} days, median over sessions; one row per seed ===\n")
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
