"""C1 item 1 -- calibrate A, k, sigma; derive c1, c2; reproduce the GLFT quotes.

Replays one day of ETHUSDC through hftbacktest without trading, records how deep
market orders reach and how the mid-price moves, and turns those two series into
the three numbers the GLFT closed form needs:

    lambda(delta) = A exp(-k delta)      order-arrival intensity
    sigma                                mid-price volatility, ticks / sqrt(s)

then reports the half spread `c1 + (Delta/2) sigma c2` and the inventory skew
`sigma c2` those imply.

    python experiments/c1_glft_baseline/run_calibrate.py

Before reporting anything it runs three checks whose right answers are known in
advance (`--check-only` runs just those).  Calibration code is easy to get
subtly wrong and impossible to eyeball, so the checks come first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import config  # noqa: E402
import glft  # noqa: E402

from rl_ladder.run import Run  # noqa: E402
from rl_ladder.seeding import seed_everything  # noqa: E402

CACHE = Path(__file__).resolve().parents[2] / "data" / "measured"

#: Steps in one 10-minute calibration window at the 100 ms measurement interval.
WINDOW = glft.window_steps(glft.INTERVAL_NS)


# --------------------------------------------------------------------------
# Checks with known answers
# --------------------------------------------------------------------------


def check_decomposition(rng: np.random.Generator, n: int = 2000) -> float:
    """c1/c2 must reproduce equations (4.6)/(4.7) exactly, not approximately.

    The paper gives the optimal depths as one expression; the notebook rewrites
    them as `half spread +/- skew * q`.  That rewrite is an identity, so if the
    two disagree anywhere the transcription is wrong.  Returns the largest
    absolute disagreement over random parameter draws.
    """
    worst = 0.0
    for _ in range(n):
        gamma = rng.uniform(1e-3, 1.0)
        delta = rng.uniform(0.1, 5.0)
        A = rng.uniform(0.1, 100.0)
        k = rng.uniform(0.05, 20.0)
        sigma = rng.uniform(0.1, 50.0)
        q = rng.uniform(-30, 30)
        xi = gamma

        # Equation (4.6)/(4.7) written out directly.
        base = 1 / (xi * delta) * np.log(1 + xi * delta / k)
        root = np.sqrt(
            gamma * sigma**2 / (2 * A * delta * k) * (1 + xi * delta / k) ** (k / (xi * delta) + 1)
        )
        direct_b = base + (2 * q + delta) / 2 * root
        direct_a = base - (2 * q - delta) / 2 * root

        c1, c2 = glft.compute_coeff(xi, gamma, delta, A, k)
        half_spread = c1 + delta / 2 * sigma * c2
        skew = sigma * c2
        worst = max(worst, abs(direct_b - (half_spread + skew * q)))
        worst = max(worst, abs(direct_a - (half_spread - skew * q)))
    return worst


def check_intensity_histogram(rng: np.random.Generator, n: int = 5000) -> int:
    """`measure_trading_intensity` against a plain-Python reference.

    The njit version writes into a shared buffer with a slice assignment, which
    is exactly the kind of thing that silently goes off by one.  This counts
    disagreements with a loop that is obviously correct (and equally obviously
    too slow to use).
    """
    depths = rng.normal(20, 15, n)
    depths[rng.random(n) < 0.05] = np.nan

    size = glft.MAX_BUCKET
    reference = np.zeros(size)
    for d in depths:
        if not np.isfinite(d):
            continue
        tick = round(d / 0.5) - 1
        if tick < 0 or tick >= size:
            continue
        for i in range(tick):
            reference[i] += 1

    out = np.zeros(size, np.float64)
    got = glft.measure_trading_intensity(depths, out)
    return int(np.sum(reference[: len(got)] != got))


def check_regression(rng: np.random.Generator, n: int = 500) -> float:
    """The hand-rolled OLS against `numpy.polyfit` on data with known slope."""
    worst = 0.0
    for _ in range(n):
        slope = rng.uniform(-3, 3)
        intercept = rng.uniform(-5, 5)
        x = np.sort(rng.uniform(0, 50, 40))
        y = slope * x + intercept + rng.normal(0, 0.05, 40)
        got_slope, got_intercept = glft.linear_regression(x, y)
        want_slope, want_intercept = np.polyfit(x, y, 1)
        worst = max(worst, abs(got_slope - want_slope), abs(got_intercept - want_intercept))
    return worst


def check_latency(date: str) -> bool:
    """Our order-latency generator against the one hftbacktest ships.

    We reimplemented it (`config._resample_feed_latency`) only because the
    packaged version copies a whole day through polars and does not fit in
    memory here, so the two must agree bit for bit or the reimplementation is a
    silent change to the backtest's fill timing.
    """
    from hftbacktest.data.utils.feed_order_latency import generate_order_latency

    reference = generate_order_latency(
        str(config.day_path(date)), mul_entry=config.MUL_ENTRY, mul_resp=config.MUL_RESP
    )
    ours = np.load(config.ensure_latency(date))["data"]
    if len(reference) != len(ours):
        return False
    return all(bool((reference[f] == ours[f]).all()) for f in ("req_ts", "exch_ts", "resp_ts"))


def run_checks() -> dict[str, float]:
    rng = np.random.default_rng(0)
    results = {
        "decomposition_max_abs_err": check_decomposition(rng),
        "intensity_histogram_mismatches": check_intensity_histogram(rng),
        "regression_max_abs_err": check_regression(rng),
    }
    for name, value in results.items():
        print(f"  {name:38s} {value:.3e}")
    assert results["decomposition_max_abs_err"] < 1e-9, "c1/c2 do not reproduce (4.6)/(4.7)"
    assert results["intensity_histogram_mismatches"] == 0, "intensity histogram is wrong"
    assert results["regression_max_abs_err"] < 1e-9, "linear_regression disagrees with polyfit"
    return results


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def measure(date: str) -> dict[str, np.ndarray]:
    """Replay `date` and cache the series calibration needs."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{config.SYMBOL}_{date}.npz"
    keys = ("arrival_depth", "mid_price_chg", "mid_price", "spread")
    if path.exists():
        blob = np.load(path)
        if all(key in blob for key in keys):
            return {key: blob[key] for key in keys}

    config.ensure_latency(date)
    hbt = config.build_backtest(date)
    measured = dict(zip(keys, glft.measure_market(hbt, glft.INTERVAL_NS), strict=True))
    hbt.close()
    np.savez_compressed(path, **measured)
    return measured


def fit_window(arrival_depth: np.ndarray, upto: int) -> tuple[np.ndarray, np.ndarray]:
    """Empirical `lambda` per second, and the matching depth axis in half-ticks."""
    tmp = np.zeros(glft.MAX_BUCKET, np.float64)
    lambda_ = glft.measure_trading_intensity(arrival_depth[:upto], tmp).copy()
    lambda_ = lambda_ / (upto * glft.INTERVAL_NS / 1e9)
    ticks = np.arange(len(lambda_)) + 0.5
    return ticks, lambda_


def true_depth_axis(n: int) -> np.ndarray:
    """The depths the intensity buckets actually correspond to, in ticks.

    Bucket `i` collects trades whose arrival depth rounds to `0.5 * (i + 1)`
    ticks, so its depth is `0.5 * (i + 1)` and not the `i + 0.5` the source
    notebook plots it at.  The notebook flags this ("the trading intensity is
    incorrectly in half-tick units") and keeps it to preserve published results;
    we keep it too for the reproduction, and report both fits here so the size
    of the resulting distortion is on the record rather than folded into a
    fudge factor.
    """
    return 0.5 * (np.arange(n) + 1)


def fit(ticks: np.ndarray, lambda_: np.ndarray) -> tuple[float, float, float]:
    """Least-squares `(A, k)` plus the R^2 of the log-linear fit."""
    y = np.log(lambda_)
    k_, logA = glft.linear_regression(ticks, y)
    A = float(np.exp(logA))
    k = float(-k_)
    pred = logA + k_ * ticks
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return A, k, 1.0 - ss_res / ss_tot


def rolling_calibration(
    arrival_depth: np.ndarray, mid_price_chg: np.ndarray, gamma: float, delta: float
) -> dict[str, np.ndarray]:
    """Refit every 5 s on a trailing 10-minute window, as the live quoter does."""
    tmp = np.zeros(glft.MAX_BUCKET, np.float64)
    ticks = np.arange(glft.MAX_BUCKET) + 0.5
    steps, As, ks, sigmas, halves, skews = [], [], [], [], [], []
    for t in range(WINDOW - 1, len(arrival_depth), 50):
        A, k = glft.fit_intensity(arrival_depth[t + 1 - WINDOW : t + 1], tmp, ticks)
        if not np.isfinite(A) or k <= 0:
            continue
        sigma = glft.realised_volatility(mid_price_chg[t + 1 - WINDOW : t + 1], glft.INTERVAL_NS)
        c1, c2 = glft.compute_coeff(gamma, gamma, delta, A, k)
        steps.append(t)
        As.append(A)
        ks.append(k)
        sigmas.append(sigma)
        halves.append(c1 + delta / 2 * c2 * sigma)
        skews.append(c2 * sigma)
    return {
        "step": np.array(steps),
        "A": np.array(As),
        "k": np.array(ks),
        "sigma": np.array(sigmas),
        "half_spread_tick": np.array(halves),
        "skew": np.array(skews),
    }


def quantiles(name: str, values: np.ndarray) -> dict[str, float]:
    q = np.nanpercentile(values, [5, 25, 50, 75, 95])
    return {
        f"{name}_p05": float(q[0]),
        f"{name}_p25": float(q[1]),
        f"{name}_median": float(q[2]),
        f"{name}_p75": float(q[3]),
        f"{name}_p95": float(q[4]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=config.CALIBRATION_DAY)
    parser.add_argument("--gamma", type=float, default=0.05, help="risk aversion; also xi")
    parser.add_argument("--delta", type=float, default=1.0, help="order size in the model")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--verify-latency",
        metavar="DATE",
        help="check our order-latency generator against hftbacktest's own helper on DATE",
    )
    args = parser.parse_args()

    seed_everything(0)

    print("checks (all must pass before any number below is worth reading):")
    checks = run_checks()
    if args.verify_latency:
        checks["latency_matches_hftbacktest"] = float(check_latency(args.verify_latency))
        print(f"  {'latency_matches_hftbacktest':38s} {checks['latency_matches_hftbacktest']:.0f}")
    if args.check_only:
        return

    if args.date in config.TEST_DAYS:
        raise SystemExit(f"{args.date} is a held-out day; calibrate on a training day")

    print(f"\nreplaying {config.SYMBOL} {args.date} ...")
    measured = measure(args.date)
    arrival_depth = measured["arrival_depth"]
    mid_price_chg = measured["mid_price_chg"]
    spread = measured["spread"]
    print(f"  {len(arrival_depth):,} steps of 100 ms")
    print(f"  {np.isfinite(arrival_depth).sum():,} steps with at least one trade")
    finite_spread = spread[np.isfinite(spread)]
    at_touch = float((finite_spread == 1).mean() * 100)
    print(
        f"  touch spread: median {np.median(finite_spread):.1f} ticks, "
        f"{at_touch:.1f}% of steps at 1 tick, p95 {np.percentile(finite_spread, 95):.1f}"
    )

    run = Run(
        phase="c1",
        name=f"calibrate-{args.date}",
        config={
            "symbol": config.SYMBOL,
            "date": args.date,
            "gamma": args.gamma,
            "delta": args.delta,
            "window_steps": WINDOW,
            "shallow_buckets": glft.SHALLOW,
        },
    )

    # --- the notebook's calibration: first 10-minute window ----------------
    ticks, lambda_ = fit_window(arrival_depth, WINDOW)
    A_full, k_full, r2_full = fit(ticks, lambda_)
    A, k, r2 = fit(ticks[: glft.SHALLOW], lambda_[: glft.SHALLOW])
    true_ticks = true_depth_axis(len(lambda_))
    A_true, k_true, r2_true = fit(true_ticks[: glft.SHALLOW], lambda_[: glft.SHALLOW])
    print(f"\nintensity fit on the first 10-minute window ({len(lambda_)} buckets populated):")
    print(f"  full range      A={A_full:9.4f}  k={k_full:7.4f}  R2={r2_full:.4f}")
    print(f"  first {glft.SHALLOW:3d}       A={A:9.4f}  k={k:7.4f}  R2={r2:.4f}")
    print(
        f"  first {glft.SHALLOW:3d}, true A={A_true:9.4f}  k={k_true:7.4f}  R2={r2_true:.4f}"
        "  (buckets at their real depth, in ticks)"
    )

    sigma_window = glft.realised_volatility(mid_price_chg[:WINDOW], glft.INTERVAL_NS)
    sigma_day = glft.realised_volatility(mid_price_chg, glft.INTERVAL_NS)
    print(f"\nvolatility (ticks / sqrt(s)):  first window {sigma_window:.3f}   whole day {sigma_day:.3f}")

    c1, c2 = glft.compute_coeff(args.gamma, args.gamma, args.delta, A, k)
    half_spread = c1 + args.delta / 2 * c2 * sigma_window
    skew = c2 * sigma_window
    print(f"\ngamma = xi = {args.gamma}, Delta = {args.delta}")
    print(f"  c1 = {c1:.4f}   c2 = {c2:.6f}")
    print(f"  half spread = {half_spread:.3f} ticks     skew = {skew:.3f} ticks per unit position")
    print(f"  a position of {half_spread / skew:.2f} units cancels the whole half spread")

    finite = arrival_depth[np.isfinite(arrival_depth)]
    reach = float((finite >= half_spread).mean() * 100)
    print(f"  {reach:.2f}% of trading steps see a trade reaching {half_spread:.1f} ticks deep")

    # --- how stable is any of this over a day? -----------------------------
    roll = rolling_calibration(arrival_depth, mid_price_chg, args.gamma, args.delta)
    print(f"\nrolling recalibration, {len(roll['step'])} windows over the day (median [p05, p95]):")
    for key, unit in [
        ("A", ""),
        ("k", ""),
        ("sigma", " ticks/sqrt(s)"),
        ("half_spread_tick", " ticks"),
        ("skew", " ticks/unit"),
    ]:
        v = roll[key]
        lo, hi = np.percentile(v, 5), np.percentile(v, 95)
        print(f"  {key:17s} {np.median(v):9.4f}  [{lo:8.4f}, {hi:9.4f}]{unit}")
        for step, value in zip(roll["step"], v, strict=True):
            run.log(step=int(step), **{key: float(value)})

    plot(run, ticks, lambda_, A_full, k_full, A, k, roll)

    summary = {
        "checks": checks,
        "date": args.date,
        "steps": int(len(arrival_depth)),
        "steps_with_trade": int(np.isfinite(arrival_depth).sum()),
        "A_full_range": A_full,
        "k_full_range": k_full,
        "r2_full_range": r2_full,
        "A": A,
        "k": k,
        "r2_shallow": r2,
        "A_true_ticks": A_true,
        "k_true_ticks": k_true,
        "median_spread_tick": float(np.median(finite_spread)),
        "pct_steps_spread_1_tick": at_touch,
        "sigma_first_window": float(sigma_window),
        "sigma_day": float(sigma_day),
        "c1": float(c1),
        "c2": float(c2),
        "half_spread_tick": float(half_spread),
        "skew_tick_per_unit": float(skew),
        "position_that_cancels_half_spread": float(half_spread / skew),
        "pct_steps_reaching_half_spread": reach,
    }
    for key in ("A", "k", "sigma", "half_spread_tick", "skew"):
        summary.update(quantiles(f"rolling_{key}", roll[key]))
    run.finish(summary)
    print(f"\nrun dir: {run.dir}")


def plot(run, ticks, lambda_, A_full, k_full, A, k, roll) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(ticks, lambda_, label="measured")
    ax1.plot(ticks, A_full * np.exp(-k_full * ticks), "--", label=f"fit, full range (k={k_full:.3f})")
    ax1.plot(
        ticks[: glft.SHALLOW],
        A * np.exp(-k * ticks[: glft.SHALLOW]),
        label=f"fit, first {glft.SHALLOW} (k={k:.3f})",
    )
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$\delta$ (half-ticks from the mid-price)")
    ax1.set_ylabel(r"$\lambda(\delta)$ (per second)")
    ax1.set_title("order-arrival intensity, first 10 min")
    ax1.legend(fontsize=8)

    minutes = roll["step"] * glft.INTERVAL_NS / 1e9 / 60
    ax2.plot(minutes, roll["half_spread_tick"], label="half spread (ticks)")
    ax2.plot(minutes, roll["skew"], label="skew (ticks/unit)")
    ax2.set_xlabel("minutes into the day")
    ax2.set_title("rolling recalibration, 10-min window every 5 s")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(run.path("calibration.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
