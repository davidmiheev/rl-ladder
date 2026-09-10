"""C1 -- why the published baseline does not transfer: the reaction interval.

The faithful reproduction of hftbacktest's GLFT notebook loses money on 2025
ETHUSDC (see the phase README).  This is the diagnosis.

A market maker's gross edge per fill is at most half the touch spread, which on
this venue is half a tick.  Its risk per fill is the price move it is exposed to
between deciding on a quote and being able to withdraw it -- one reaction
interval plus one order round trip.  With mid-price volatility `sigma` in ticks
per sqrt(second) that exposure is about

    sigma * sqrt(interval + latency)

so the strategy is viable only while `sigma * sqrt(interval + latency)` stays
comparable to the half spread plus the rebate.  The notebook's 100 ms interval
was chosen against ETHUSDT in 2022, where `sigma` was about 10.7 ticks per
sqrt(second).  On our days `sigma` is about 47: the same tick size on a price
3.3x higher, in a more volatile market.  The exposure grew 4.4x; the half spread
did not grow at all, because it cannot go below one tick.

This runs the simplest possible market maker -- one lot at the touch, no model
-- across reaction intervals and latency multipliers, and reports what a fill
was worth the instant it happened.

    python experiments/c1_glft_baseline/run_interval_study.py --hours 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from hftbacktest import Recorder

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import config  # noqa: E402
import glft  # noqa: E402

from rl_ladder.run import Run  # noqa: E402

#: (reaction interval in ms, latency multiplier, queue model).  The first block
#: varies how fast the quoter reacts under the recorded feed's own latency; the
#: second assumes colocation; the third holds the fastest setting and varies
#: only the simulator's queue-position model.
GRID = [
    (100.0, 1.0, "power2"),
    (25.0, 1.0, "power2"),
    (10.0, 1.0, "power2"),
    (2.0, 1.0, "power2"),
    (100.0, 0.1, "power2"),
    (10.0, 0.1, "power2"),
    (2.0, 0.1, "power2"),
    (2.0, 0.1, "risk-adverse"),
    (2.0, 0.1, "power2-2"),
    (2.0, 0.1, "power3-3"),
    (2.0, 0.1, "log"),
    (2.0, 0.1, "log2"),
    (100.0, 1.0, "log2"),
]

#: Horizons, in seconds after a fill, at which the position is marked to the
#: mid.  0 is execution quality; the rest is adverse selection.
HORIZONS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 30.0]


def one(
    date: str, interval_ms: float, latency_mul: float, queue_model: str, hours: float
) -> dict[str, float]:
    config.ensure_latency(date, config.MUL_ENTRY * latency_mul, config.MUL_RESP * latency_mul)
    hbt = config.build_backtest(date, latency_mul=latency_mul, queue_model=queue_model)
    interval_ns = int(interval_ms * 1e6)
    record_every = max(1, round(100e6 / interval_ns))
    recorder = Recorder(1, 2_000_000)

    # Every setting replays the same stretch of the same day, so the rows differ
    # only in how fast the quoter reacts.
    n_steps = int(hours * 3600 * 1e9 / interval_ns)
    fills = np.zeros((4_000_000, 3))
    mids = np.full(n_steps + 1, np.nan)
    n = glft.touch_quoter(
        hbt, recorder.recorder, 1.0, 1.0, interval_ns, record_every, fills, mids, n_steps
    )
    hbt.close()

    fills = fills[:n]
    side = fills[:, 0]
    price = fills[:, 1]
    step = fills[:, 2].astype(np.int64)

    row = {
        "interval_ms": interval_ms,
        "latency_mul": latency_mul,
        "queue_model": queue_model,
        "fills": int(n),
    }
    for horizon in HORIZONS:
        ahead = step + int(horizon * 1e9 / interval_ns)
        keep = ahead < len(mids)
        marked = side[keep] * (mids[ahead[keep]] - price[keep])
        row[f"mark_{horizon:g}s"] = float(np.nanmean(marked))
    return row


def market_reference(date: str, interval_ms: float, hours: float) -> dict[str, float]:
    """The same adverse selection, measured without placing a single order."""
    config.ensure_latency(date)
    hbt = config.build_backtest(date)
    interval_ns = int(interval_ms * 1e6)
    n_steps = int(hours * 3600 * 1e9 / interval_ns)
    mids = np.full(n_steps + 1, np.nan)
    flow = np.zeros(n_steps + 1)
    qty = np.zeros(n_steps + 1)
    t = glft.measure_flow_continuation(hbt, interval_ns, n_steps, mids, flow, qty)
    hbt.close()

    row: dict[str, float] = {"steps": t, "steps_with_aggressive_trade": float((qty[:t] > 0).mean())}
    for horizon in HORIZONS[1:]:
        ahead = int(horizon * 1e9 / interval_ns)
        future = np.full(t, np.nan)
        future[: t - ahead] = mids[ahead:t] - mids[: t - ahead]
        # A maker is filled by the opposite side: sell sweeps hit its bid.
        felt = np.concatenate([future[flow[:t] < 0], -future[flow[:t] > 0]])
        row[f"felt_{horizon:g}s"] = float(np.nanmean(felt))
    row["unconditional_std_0.1s"] = float(
        np.nanstd(mids[int(0.1e9 / interval_ns) : t] - mids[: t - int(0.1e9 / interval_ns)])
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=config.CALIBRATION_DAY)
    parser.add_argument("--hours", type=float, default=4.0, help="hours of the day to replay")
    args = parser.parse_args()

    if args.date in config.TEST_DAYS:
        raise SystemExit(f"{args.date} is a held-out day")

    run = Run(phase="c1", name=f"interval-study-{args.date}",
              config={"symbol": config.SYMBOL, "date": args.date, "hours": args.hours, "grid": GRID})

    print(f"one-lot touch quoter, {config.SYMBOL} {args.date}, first {args.hours:g} hours")
    print("mean mark-to-mid of the position a fill opened, in ticks, by horizon")
    header = "".join(f"{h:>9g}s" for h in HORIZONS)
    print(f"{'interval':>10} {'latency':>9} {'queue':>13} {'fills':>9}{header}")
    rows = []
    for i, (interval_ms, latency_mul, queue_model) in enumerate(GRID):
        row = one(args.date, interval_ms, latency_mul, queue_model, args.hours)
        rows.append(row)
        run.log(step=i, **row)
        marks = "".join(f"{row[f'mark_{h:g}s']:10.2f}" for h in HORIZONS)
        print(
            f"{interval_ms:8.0f}ms {latency_mul:9g} {queue_model:>13} {row['fills']:9d}{marks}",
            flush=True,
        )

    print("\ncontrol: the same thing measured from market data alone, no orders placed")
    reference = market_reference(args.date, 2.0, args.hours)
    print(f"  steps with an aggressive trade: {reference['steps_with_aggressive_trade'] * 100:.2f}%")
    print(f"  unconditional 100 ms mid move, std: {reference['unconditional_std_0.1s']:.2f} ticks")
    for horizon in HORIZONS[1:]:
        print(f"  mid move felt by a maker filled by that flow, {horizon:g}s: "
              f"{reference[f'felt_{horizon:g}s']:7.2f} ticks")

    run.finish({"rows": rows, "market_reference": reference})
    print(f"\nrun dir: {run.dir}")


if __name__ == "__main__":
    main()
