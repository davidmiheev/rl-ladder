"""Reference-leg series for C6's two pair strategies.

APT and basis both need a second instrument: the notebooks use **spot** against
the perpetual, and read it from a precomputed `(timestamp, x, y)` table rather
than from a second backtest asset, so that the strategy loop stays a single
asset and the alignment is done once, offline, where it can be inspected.

This dataset has no spot feed.  What it does have is the same underlying quoted
in two stablecoins on the same venue: `xrpusdc` and `xrpusdt`.  C6 trades the
USDC leg and uses the USDT leg as the reference.  The pair is cointegrated --
both track XRP, and the residual is the USDC/USDT basis -- so the shape of both
strategies survives.  What changes is the economics:

- the notebooks' basis is the perpetual's premium over spot, which is a funding
  and cash-and-carry object with a slow, mean-reverting, sign-stable term
  structure;
- ours is a stablecoin spread plus a venue lead-lag, which is much smaller in
  bp, much faster, and has no funding anchor pulling it back.

So a weaker result here is the expected result, and a *better* one would be the
suspicious one.  This is recorded up front because it is exactly the kind of
substitution that is easy to report as if it were the published experiment.

The two tables, both on a 100 ms grid of local timestamps:

    apt    (ts, reference_return, own_past_price)
           `reference_return` is the reference leg's return over the last five
           minutes and `own_past_price` our own mid five minutes ago, so that
           `fair = (1 + beta * reference_return) * own_past_price` is a
           lead-lag forecast and not a restatement of the current price.
    basis  (ts, reference_price, rolling_mean_basis)
           with the basis `own - reference` averaged over five minutes.

Both use the notebook's trick of averaging the price over a window *centred* on
the point five minutes ago rather than taking the single tick there: a lone
sample at the far end of a return is as noisy as the return itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "c1_glft_baseline"))

import config  # noqa: E402
import glft  # noqa: E402

PRECOMP = Path(__file__).resolve().parents[2] / "data" / "precompute"

#: Sampling grid, matching the quoter's reaction interval.
INTERVAL_NS = 100_000_000
#: Five minutes, in samples.
ROLLING = 3_000
#: Half of it: the centring shift, as in the notebook.
SHIFT = 1_500


@njit
def measure_mid(hbt, interval_ns, out_ts, out_mid):
    """Walk a day and record (local timestamp, mid price) on a fixed grid.

    A bare elapse loop -- no orders, no depth sums -- so this costs a fraction
    of a strategy run even though it replays the whole feed.
    """
    t = 0
    while hbt.elapse(interval_ns) == 0:
        depth = hbt.depth(0)
        best_bid_tick = depth.best_bid_tick
        best_ask_tick = depth.best_ask_tick
        two_sided = best_bid_tick != glft.INVALID_MIN and best_ask_tick != glft.INVALID_MAX
        out_ts[t] = hbt.current_timestamp
        out_mid[t] = (best_bid_tick + best_ask_tick) / 2.0 * depth.tick_size if two_sided else np.nan
        t += 1
        if t >= len(out_ts):
            break
    return t


def _mid_series(symbol: str, date: str) -> tuple[np.ndarray, np.ndarray]:
    config.use_symbol(symbol)
    config.ensure_latency(date)
    hbt = config.build_backtest(date)
    n = glft.buffer_steps(INTERVAL_NS)
    ts = np.zeros(n, np.int64)
    mid = np.full(n, np.nan, np.float64)
    used = measure_mid(hbt, INTERVAL_NS, ts, mid)
    hbt.close()
    return ts[:used], mid[:used]


def _forward_fill(x: np.ndarray) -> np.ndarray:
    idx = np.where(np.isfinite(x), np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    out[: np.argmax(np.isfinite(x))] = np.nan
    return out


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over `window` samples, `nan` until the window is full.

    `np.convolve` would be shorter but allocates twice the array and silently
    turns a single `nan` into `window` of them; the cumulative form keeps the
    `nan` handling explicit.
    """
    finite = np.isfinite(x)
    filled = np.where(finite, x, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(filled)))
    ccnt = np.concatenate(([0], np.cumsum(finite)))
    total = csum[window:] - csum[:-window]
    count = ccnt[window:] - ccnt[:-window]
    out = np.full(len(x), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[window - 1 :] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return out


def build(symbol: str, date: str, overwrite: bool = False) -> tuple[Path, Path]:
    """Write this day's APT and basis tables for `symbol` against its reference."""
    reference = config.REFERENCE_LEG[symbol]
    apt_path = PRECOMP / f"apt_{symbol}_{date}.npy"
    basis_path = PRECOMP / f"basis_{symbol}_{date}.npy"
    if apt_path.exists() and basis_path.exists() and not overwrite:
        return apt_path, basis_path

    own_ts, own_mid = _mid_series(symbol, date)
    ref_ts, ref_mid = _mid_series(reference, date)

    # Put the reference on our own timestamp grid.  `searchsorted` with
    # `side="right"` minus one gives, for each of our samples, the last
    # reference sample that had already arrived -- never a later one, which is
    # the whole point of aligning on the *local* timestamp.
    pos = np.searchsorted(ref_ts, own_ts, side="right") - 1
    ref_on_grid = np.where(pos >= 0, ref_mid[np.clip(pos, 0, len(ref_mid) - 1)], np.nan)

    own = _forward_fill(own_mid)
    ref = _forward_fill(ref_on_grid)

    ref_past = np.roll(_rolling_mean(ref, ROLLING), SHIFT)
    own_past = np.roll(_rolling_mean(own, ROLLING), SHIFT)
    ref_past[:SHIFT] = np.nan
    own_past[:SHIFT] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        ref_return = ref / ref_past - 1.0

    apt = np.column_stack([own_ts.astype(np.float64), ref_return, own_past])
    basis = np.column_stack([own_ts.astype(np.float64), ref, _rolling_mean(own - ref, ROLLING)])

    PRECOMP.mkdir(parents=True, exist_ok=True)
    np.save(apt_path, np.ascontiguousarray(apt))
    np.save(basis_path, np.ascontiguousarray(basis))
    return apt_path, basis_path


def load(kind: str, symbol: str, date: str) -> np.ndarray:
    """Read a table, building it first if it is not there yet."""
    build(symbol, date)
    return np.load(PRECOMP / f"{kind}_{symbol}_{date}.npy")


def own_mid(symbol: str, date: str) -> np.ndarray:
    """The traded leg's own mid on the grid, cached next to the tables."""
    path = PRECOMP / f"mid_{symbol}_{date}.npy"
    if not path.exists():
        _, mid = _mid_series(symbol, date)
        PRECOMP.mkdir(parents=True, exist_ok=True)
        np.save(path, _forward_fill(mid))
    return np.load(path)


def fit_beta(symbol: str, dates: list[str]) -> tuple[float, float]:
    """OLS of our five-minute return on the reference leg's, over whole days.

    The notebook offers `beta = 1` and a fitted beta and reports both; C6 fits
    it, because the question the APT strategy asks is how much of the reference
    leg's move carries over, and on a stablecoin pair there is no reason for
    the answer to be one.  No intercept: an intercept here is a claim that the
    pair drifts apart at a constant rate, which over four days is a way of
    fitting the sample.  Returns `(beta, r_squared)`.
    """
    xs, ys = [], []
    for date in dates:
        apt = load("apt", symbol, date)
        reference_return, own_past = apt[:, 1], apt[:, 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            return_ = own_mid(symbol, date) / own_past - 1.0
        mask = np.isfinite(reference_return) & np.isfinite(return_)
        xs.append(reference_return[mask])
        ys.append(return_[mask])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    beta = float(np.dot(x, y) / np.dot(x, x))
    r2 = float(1.0 - np.var(y - beta * x) / np.var(y))
    return beta, r2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="xrpusdc")
    parser.add_argument("--days", default="pair-all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    days = {
        "pair-train": config.PAIR_TRAIN_DAYS,
        "pair-test": config.PAIR_TEST_DAYS,
        "pair-all": config.PAIR_TRAIN_DAYS + config.PAIR_TEST_DAYS,
    }[args.days]

    for date in days:
        apt_path, basis_path = build(args.symbol, date, overwrite=args.overwrite)
        own_mid(args.symbol, date)
        apt = np.load(apt_path)
        basis = np.load(basis_path)
        ok = np.isfinite(apt[:, 1]).mean()
        spread_bp = np.nanmedian(basis[:, 2] / basis[:, 1]) * 1e4
        print(
            f"{date}  {len(apt):8d} rows  reference return finite {ok:6.1%}  "
            f"median basis {spread_bp:+7.3f} bp"
        )

    beta, r2 = fit_beta(args.symbol, config.PAIR_TRAIN_DAYS)
    print(f"\nbeta fitted on training days: {beta:.4f}   R^2 {r2:.4f}")


if __name__ == "__main__":
    main()
