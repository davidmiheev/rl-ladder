"""What does a fill cost, as a function of how far out it was quoted?

C1's central negative result was that the calibrated GLFT half spread on
ETHUSDC -- about 1.7 bp -- does not cover the adverse selection the fills
bring.  This script measures that cost directly from the tape, placing no
orders, and reports it as a *curve* over quote depth rather than as C1's
single regression.

It is a diagnostic, not the thing that sets C6's spread floor.  Its estimate
and the simulator disagree by about 0.9 bp per fill, and the disagreement is
the point: this measurement sees every aggressive trade, while a resting quote
is filled by a *selected* subset of them -- the ones that consumed the queue
ahead of it, which are the large informed sweeps.  Volume weighting narrows
the gap but cannot close it, because queue position and latency are not on the
tape.  `sweep_spread.py` sets the floor with the simulator, which does see
them; this script is what says whether the simulator's answer is the right
shape.

1. Replay a training day on a 100 ms grid.  In each bucket record the mid
   price and, separately for aggressive buys and aggressive sells, the
   deepest tick the aggressor reached relative to the mid.
2. A resting quote `d` ticks from the mid is filled in that bucket only if the
   aggressor reached at least `d`.  Conditioning on that is the whole point:
   averaging over all aggressive flow instead measures the cost of the fills
   a touch quoter gets, which are mostly one-lot prints, and understates the
   cost of the fills a quote further out gets, which are the sweeps.
3. An aggressive buy lifts our ask and leaves us **short**, so we lose when
   the mid then rises: `cost = +(mid[t+h] - mid[t])`.  An aggressive sell
   leaves us long and the sign flips.  Averaging the two sides gives
   `cost(d)`, in bp, the expected adverse move on a fill at depth `d`.
4. A quote at depth `d` earns `d` (in bp) plus the maker rebate when it is
   filled, so it breaks even at the smallest `d` with

       depth_bp(d) + rebate_bp >= cost(d)

   and that `d` is the floor reported.  It is a crossing point rather than a
   single expectation because both sides move with `d`: quoting further out
   earns more per fill *and* meets worse flow, and whether the crossing exists
   at all is the question C1's negative result turns on.

The holding horizon is the one real judgement call, so the whole curve over
`HORIZONS` is printed and the floor is taken at `FLOOR_HORIZON`.  Ten seconds
is the order of the median time a grid quoter holds inventory in C1's runs.

Training days only.  Reading the held-out set to decide how wide to quote
would be the exact leak C4's third item exists to catch.

    python experiments/c6_strategy_zoo/calibrate_floor.py --symbol ethusdc --write
    python experiments/c6_strategy_zoo/calibrate_floor.py --symbol xrpusdc --days pair-train --write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from hftbacktest import BUY_EVENT
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "c1_glft_baseline"))

import config  # noqa: E402
import glft  # noqa: E402

INTERVAL_NS = 100_000_000
#: Holding horizons to report, in seconds.
HORIZONS = (0.5, 1.0, 5.0, 10.0, 30.0, 60.0)
#: The one the floor is taken from.
FLOOR_HORIZON = 10.0
#: Quote depths to evaluate, in ticks from the mid.
DEPTH_TICKS = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0, 80.0, 120.0, 200.0)
#: Depths with fewer fills than this over the whole training set are reported
#: but never chosen as the floor: a crossing supported by a handful of fills is
#: a statement about those fills.
MIN_FILLS = 200

OUT = Path(__file__).resolve().parent / "spread_floor.json"
CACHE = Path(__file__).resolve().parents[2] / "data" / "measured"


@njit
def measure_depth_and_mid(hbt, interval_ns, out_mid, out_buy, out_sell, out_buy_qty, out_sell_qty):
    """Mid price and the deepest tick each side's aggressors reached. No orders.

    `nan` in a depth column means that side saw no aggressive trade in that
    bucket, which is the overwhelming majority of them.
    """
    t = 0
    prev_mid_tick = np.nan
    while hbt.elapse(interval_ns) == 0:
        depth = hbt.depth(0)
        tick_size = depth.tick_size
        best_bid_tick = depth.best_bid_tick
        best_ask_tick = depth.best_ask_tick
        two_sided = best_bid_tick != glft.INVALID_MIN and best_ask_tick != glft.INVALID_MAX

        buy_depth = np.nan
        sell_depth = np.nan
        buy_qty = 0.0
        sell_qty = 0.0
        # Measured against the mid as it stood *before* these trades, which is
        # where our quote was actually resting.
        if not np.isnan(prev_mid_tick):
            for last_trade in hbt.last_trades(0):
                trade_tick = last_trade.px / tick_size
                if last_trade.ev & BUY_EVENT == BUY_EVENT:
                    reached = trade_tick - prev_mid_tick
                    if np.isnan(buy_depth) or reached > buy_depth:
                        buy_depth = reached
                    buy_qty += last_trade.qty
                else:
                    reached = prev_mid_tick - trade_tick
                    if np.isnan(sell_depth) or reached > sell_depth:
                        sell_depth = reached
                    sell_qty += last_trade.qty
        hbt.clear_last_trades(0)

        mid_tick = (best_bid_tick + best_ask_tick) / 2.0 if two_sided else np.nan
        out_mid[t] = mid_tick * tick_size
        out_buy[t] = buy_depth
        out_sell[t] = sell_depth
        out_buy_qty[t] = buy_qty
        out_sell_qty[t] = sell_qty
        prev_mid_tick = mid_tick
        t += 1
        if t >= len(out_mid):
            break
    return t


def day_series(symbol: str, date: str):
    """Cached (mid, buy depth, sell depth, buy volume, sell volume) for one day."""
    path = CACHE / f"depth_{symbol}_{date}.npz"
    if path.exists():
        blob = np.load(path)
        return blob["mid"], blob["buy"], blob["sell"], blob["buy_qty"], blob["sell_qty"]
    config.use_symbol(symbol)
    config.ensure_latency(date)
    hbt = config.build_backtest(date)
    n = glft.buffer_steps(INTERVAL_NS)
    mid = np.full(n, np.nan, np.float64)
    buy = np.full(n, np.nan, np.float64)
    sell = np.full(n, np.nan, np.float64)
    buy_qty = np.zeros(n, np.float64)
    sell_qty = np.zeros(n, np.float64)
    used = measure_depth_and_mid(hbt, INTERVAL_NS, mid, buy, sell, buy_qty, sell_qty)
    hbt.close()
    mid, buy, sell = mid[:used], buy[:used], sell[:used]
    buy_qty, sell_qty = buy_qty[:used], sell_qty[:used]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mid=mid, buy=buy, sell=sell, buy_qty=buy_qty, sell_qty=sell_qty)
    return mid, buy, sell, buy_qty, sell_qty


def cost_curve(
    mid: np.ndarray, buy: np.ndarray, sell: np.ndarray,
    buy_qty: np.ndarray, sell_qty: np.ndarray, horizon_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """`(cost_bp, fills)` at each depth in `DEPTH_TICKS`, for one day.

    Volume weighted: a maker's exposure to a bucket is proportional to how
    much traded in it, so a one-lot print must not count as much as a sweep.
    Unweighted, the curve sits about 0.2 bp lower at every depth, which is
    enough to flip the sign of the break-even -- the weighting is not a
    refinement.
    """
    h = int(round(horizon_s * 1e9 / INTERVAL_NS))
    move_bp = np.full_like(mid, np.nan)
    move_bp[:-h] = (mid[h:] - mid[:-h]) / mid[:-h] * 1e4

    costs = np.full(len(DEPTH_TICKS), np.nan)
    fills = np.zeros(len(DEPTH_TICKS), np.int64)
    for i, d in enumerate(DEPTH_TICKS):
        # Filled on the ask by a buy that reached >= d: we are short, and a
        # rising mid costs us.  Filled on the bid by a sell: the sign flips.
        short = np.isfinite(move_bp) & (buy >= d)
        long_ = np.isfinite(move_bp) & (sell >= d)
        fills[i] = int(short.sum() + long_.sum())
        weight = buy_qty[short].sum() + sell_qty[long_].sum()
        if weight > 0:
            costs[i] = float(
                (np.dot(move_bp[short], buy_qty[short]) - np.dot(move_bp[long_], sell_qty[long_]))
                / weight
            )
    return costs, fills


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--symbol", default="ethusdc")
    parser.add_argument("--days", default="train")
    args = parser.parse_args()

    days = {
        "train": config.TRAIN_DAYS_EXT,
        "train-c1": config.TRAIN_DAYS,
        "pair-train": config.PAIR_TRAIN_DAYS,
    }[args.days]

    rebate_bp = -config.MAKER_FEE * 1e4
    tick_size = config.SYMBOLS[args.symbol]["tick_size"]

    curves = {h: [] for h in HORIZONS}
    counts = {h: [] for h in HORIZONS}
    mids = []
    for date in days:
        mid, buy, sell, buy_qty, sell_qty = day_series(args.symbol, date)
        mids.append(np.nanmedian(mid))
        for h in HORIZONS:
            costs, fills = cost_curve(mid, buy, sell, buy_qty, sell_qty, h)
            curves[h].append(costs)
            counts[h].append(fills)
        print(f"{date}  done", flush=True)

    median_mid = float(np.median(mids))
    depth_bp = np.array(DEPTH_TICKS) * tick_size / median_mid * 1e4

    print(f"\n{args.symbol}: median mid {median_mid:.4f}, tick {tick_size:g} "
          f"({tick_size / median_mid * 1e4:.4f} bp), maker rebate {rebate_bp:.3f} bp")
    print(f"adverse selection on a fill at depth d, bp, mean over {len(days)} training days\n")
    header = "  depth       bp  " + "".join(f"{h:>8.1f}s" for h in HORIZONS) + "     fills"
    print(header)
    for i, d in enumerate(DEPTH_TICKS):
        row = "".join(f"{np.nanmean([c[i] for c in curves[h]]):9.3f}" for h in HORIZONS)
        n = int(np.sum([c[i] for c in counts[FLOOR_HORIZON]]))
        print(f"  {d:6.1f}t {depth_bp[i]:8.3f}{row}  {n:9d}")

    cost = np.array([np.nanmean([c[i] for c in curves[FLOOR_HORIZON]]) for i in range(len(DEPTH_TICKS))])
    fills = np.array([int(np.sum([c[i] for c in counts[FLOOR_HORIZON]])) for i in range(len(DEPTH_TICKS))])
    earns = depth_bp + rebate_bp
    viable = (earns >= cost) & (fills >= MIN_FILLS) & np.isfinite(cost)

    print(f"\nat {FLOOR_HORIZON:g}s: a quote at depth d earns d + {rebate_bp:.3f} bp and pays cost(d)")
    if viable.any():
        i = int(np.argmax(viable))
        print(f"  tape break-even from {DEPTH_TICKS[i]:g} ticks = {depth_bp[i]:.3f} bp "
              f"(earns {earns[i]:.3f}, pays {cost[i]:.3f}, {fills[i]} fills)")
    else:
        print("  no depth on the tape breaks even: cost outruns the spread everywhere")
    print("\nThis is the cost of the average aggressive trade at that depth, not of the\n"
          "fills a resting quote actually wins.  `sweep_spread.py` is what sets the floor.")


if __name__ == "__main__":
    main()
