"""Which volatility estimate forecasts the next five minutes best?

Every strategy in C6 that has a model uses the same sigma: the standard
deviation of 100 ms mid changes over the trailing ten minutes, refit every five
seconds (`glft.realised_volatility`). GLFT puts it in the half spread and the
skew, hfgrid sets its half spread to five of it. Nobody checked whether it is a
good forecast of the volatility the quotes are about to face.

This scores a set of standard estimators as *forecasts*: at every evaluation
point (each 5 s, from one hour into the session), each estimator produces a
variance per second from trailing data only, and it is compared with the
realised variance per second over the next five minutes.

The loss is QLIKE, `y/f - log(y/f) - 1`, because it stays consistent when the
target is itself a noisy realised measure (Patton 2011, "Volatility forecast
comparison using imperfect volatility proxies"); MSE on levels does not, and
ranks estimators by how well they chase outliers. Lower is better and a
perfect forecast scores 0.

The target is computed two ways, from 1 s and from 5 s returns, because on a
coarse-tick asset the 1 s mid path is mostly the bid-ask bounce and a target
built on it rewards estimators that share the same noise.

Each estimator is scored raw and *calibrated*: multiplied by its median ratio
of target to forecast on the training days. GLFT consumes sigma in a
formula, so a biased-but-responsive estimator is usable once its bias is known,
and the calibrated score separates "wrong level" from "wrong timing".

    python experiments/c7_signals/vol_bakeoff.py --symbol ethusdc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import snapshots  # noqa: E402

RUNS = Path(__file__).resolve().parents[2] / "runs" / "c7-vol"

EVERY_S = 5
WARMUP_S = 3_600
HORIZON_S = 300
WINDOW_S = 600

#: Estimators, in the order they are reported. `current` is what C6 runs.
NAMES = (
    "current",        # std of 100 ms mid changes over 10 min, as in glft.py
    "rv_1s_10m",      # realised variance of 1 s returns over 10 min
    "rv_5s_10m",      # the same, sparsely sampled at 5 s
    "rv_1s_1m",       # 1 s returns over the last minute only
    "tsrv_10m",       # two-scale RV (Zhang, Mykland & Ait-Sahalia 2005)
    "parkinson_10m",  # high-low range of each 1 s bar
    "bipower_10m",    # bipower variation of 1 s returns, jump robust
    "ewma_30s",       # exponentially weighted 1 s squared returns
    "ewma_2m",
    "ewma_10m",
    "har",            # HAR on 1 min / 10 min / 1 h RV, fitted on training days
)


def load_mid(symbol: str, date: str) -> tuple[np.ndarray, float]:
    """The 100 ms mid in price units, forward-filled over one-sided moments."""
    blob = np.load(snapshots.feature_path(symbol, date))
    half_ticks = blob["mid100"].astype(np.float64)
    half_ticks[half_ticks < 0] = np.nan
    idx = np.where(np.isfinite(half_ticks), np.arange(len(half_ticks)), 0)
    np.maximum.accumulate(idx, out=idx)
    half_ticks = half_ticks[idx]
    # Before the first two-sided book there is nothing to carry forward;
    # back-fill from the first mid rather than let a NaN run through every
    # cumulative sum downstream.
    half_ticks[: np.argmax(np.isfinite(half_ticks))] = half_ticks[np.isfinite(half_ticks)][0]
    return half_ticks * float(blob["tick"]) / 2.0, float(blob["tick"])


def _trailing_sum(x: np.ndarray, window: int) -> np.ndarray:
    """Sum of `x[t - window + 1 : t + 1]` at each t (NaN before it is full)."""
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    out = np.full(len(x), np.nan)
    out[window - 1:] = c[window:] - c[:-window]
    return out


def _ewma(x: np.ndarray, halflife_s: float) -> np.ndarray:
    alpha = 1.0 - 0.5 ** (1.0 / halflife_s)
    out = np.empty(len(x))
    level = np.nanmean(x[:600]) if len(x) >= 600 else np.nanmean(x)
    for i, v in enumerate(x):
        if np.isfinite(v):
            level += alpha * (v - level)
        out[i] = level
    return out


def day_series(symbol: str, date: str) -> dict[str, np.ndarray]:
    """Every estimator and both targets, at every evaluation second, in bp^2/s."""
    mid100, tick = load_mid(symbol, date)
    log100 = np.log(mid100) * 1e4  # in bp
    n_sec = len(log100) // 10
    log1 = log100[9: n_sec * 10: 10]  # the last mid of each second
    r1 = np.diff(log1, prepend=log1[0])
    r1[0] = 0.0
    sq1 = r1**2

    # 100 ms changes, grouped by the second they end in.
    d100 = np.diff(log100[: n_sec * 10], prepend=log100[0])
    per_sec = d100.reshape(n_sec, 10)
    sum_d = per_sec.sum(axis=1)
    sum_d2 = (per_sec**2).sum(axis=1)
    hi = log100[: n_sec * 10].reshape(n_sec, 10).max(axis=1)
    lo = log100[: n_sec * 10].reshape(n_sec, 10).min(axis=1)

    w = WINDOW_S
    out: dict[str, np.ndarray] = {}
    # current: np.nanstd of 6,000 100 ms changes, scaled to per second.
    s1 = _trailing_sum(sum_d, w)
    s2 = _trailing_sum(sum_d2, w)
    n = w * 10
    out["current"] = (s2 / n - (s1 / n) ** 2) * 10.0
    out["rv_1s_10m"] = _trailing_sum(sq1, w) / w
    log5 = log1[4::5]
    r5 = np.diff(log5, prepend=log5[0])
    sq5_on_1s = np.zeros(n_sec)
    sq5_on_1s[4::5] = r5**2
    out["rv_5s_10m"] = _trailing_sum(sq5_on_1s, w) / w
    out["rv_1s_1m"] = _trailing_sum(sq1, 60) / 60
    # TSRV: average of K=10 sparse 1 s grids (offset by 100 ms each) minus the
    # noise bias estimated from the full 100 ms grid.
    rv_all = s2  # sum of squared 100 ms changes over the window
    sub = np.zeros(n_sec)
    for offset in range(10):
        shifted = log100[offset: offset + n_sec * 10: 10]
        r = np.diff(shifted, prepend=shifted[0])
        r[0] = 0.0
        sub[: len(r)] += r**2
    rv_avg = _trailing_sum(sub, w) / 10.0
    nbar = (n - 10 + 1) / 10.0
    out["tsrv_10m"] = np.maximum(rv_avg - (nbar / n) * rv_all, 0.0) / w
    out["parkinson_10m"] = _trailing_sum((hi - lo) ** 2 / (4.0 * np.log(2.0)), w) / w
    bp = np.abs(r1) * np.abs(np.roll(r1, 1))
    bp[0] = 0.0
    out["bipower_10m"] = (np.pi / 2.0) * _trailing_sum(bp, w) / w
    out["ewma_30s"] = _ewma(sq1, 30)
    out["ewma_2m"] = _ewma(sq1, 120)
    out["ewma_10m"] = _ewma(sq1, 600)
    out["_rv_1m"] = out["rv_1s_1m"]
    out["_rv_1h"] = _trailing_sum(sq1, 3600) / 3600

    # Targets: realised variance per second over the next HORIZON_S.
    fwd1 = np.full(n_sec, np.nan)
    c = np.concatenate([[0.0], np.cumsum(sq1)])
    fwd1[: n_sec - HORIZON_S] = (c[1 + HORIZON_S: n_sec + 1] - c[1: n_sec - HORIZON_S + 1]) \
        / HORIZON_S
    c5 = np.concatenate([[0.0], np.cumsum(sq5_on_1s)])
    fwd5 = np.full(n_sec, np.nan)
    fwd5[: n_sec - HORIZON_S] = (c5[1 + HORIZON_S: n_sec + 1] - c5[1: n_sec - HORIZON_S + 1]) \
        / HORIZON_S
    out["_y1"] = fwd1
    out["_y5"] = fwd5

    # The resolution floor: one half-tick move of the mid over the horizon.
    # A quiet book can hold the mid still for twenty minutes (XRPUSDC on
    # 2025-08-29), and a short EWMA then forecasts a variance of ~0, which
    # QLIKE punishes without bound. No estimator can resolve less than one
    # mid move, so every forecast is floored at it, the same for all.
    floor = (tick / 2.0 / np.nanmedian(mid100) * 1e4) ** 2 / HORIZON_S
    for name in NAMES:
        if name in out:
            out[name] = np.maximum(out[name], floor)
    keep = np.arange(WARMUP_S, n_sec - HORIZON_S, EVERY_S)
    return {k: v[keep] for k, v in out.items()}


def qlike(y: np.ndarray, f: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(f) & (y > 0) & (f > 0)
    ratio = y[ok] / f[ok]
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def fit_har(days: list[dict[str, np.ndarray]], target: str) -> np.ndarray:
    """log y = b0 + b1 log RV_1m + b2 log RV_10m + b3 log RV_1h, by OLS."""
    rows, ys = [], []
    for d in days:
        x = np.column_stack([np.ones(len(d[target])), np.log(d["_rv_1m"]),
                             np.log(d["rv_1s_10m"]), np.log(d["_rv_1h"])])
        y = np.log(d[target])
        ok = np.all(np.isfinite(x), axis=1) & np.isfinite(y)
        rows.append(x[ok])
        ys.append(y[ok])
    beta, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(ys), rcond=None)
    return beta


def har_residual_var(train: list[dict[str, np.ndarray]], beta: np.ndarray, target: str) -> float:
    """Variance of the training residuals of log y, for the log-normal correction."""
    resid = []
    for t in train:
        xt = np.column_stack([np.ones(len(t["_rv_1m"])), np.log(t["_rv_1m"]),
                              np.log(t["rv_1s_10m"]), np.log(t["_rv_1h"])])
        r = np.log(t[target]) - xt @ beta
        resid.append(r[np.isfinite(r)])
    return float(np.var(np.concatenate(resid)))


def har_forecast(d: dict[str, np.ndarray], beta: np.ndarray, s2: float) -> np.ndarray:
    """exp(E[log y]) underestimates E[y]; `s2 / 2` is the log-normal correction."""
    x = np.column_stack([np.ones(len(d["_rv_1m"])), np.log(d["_rv_1m"]),
                         np.log(d["rv_1s_10m"]), np.log(d["_rv_1h"])])
    return np.exp(x @ beta + 0.5 * s2)


def evaluate(symbol: str, train_days: list[str], test_days: list[str]) -> dict:
    train = {d: day_series(symbol, d) for d in train_days}
    test = {d: day_series(symbol, d) for d in test_days}
    results = {}
    for target in ("_y1", "_y5"):
        beta = fit_har(list(train.values()), target)
        s2 = har_residual_var(list(train.values()), beta, target)
        for split in (train, test):
            for d in split.values():
                d["har"] = np.maximum(har_forecast(d, beta, s2), np.nanmin(d["rv_1s_10m"]))
        rows = {}
        for name in NAMES:
            # Calibration constant, from the training days. The scale that
            # minimises QLIKE in sample is mean(y/f), but that mean belongs to
            # the handful of seconds when a frozen mid starts moving again
            # (y/f ~ 1e4) and calibrates everything else badly; the median
            # ratio is the typical bias, and is what is used.
            ratios = np.concatenate([t[target] / t[name] for t in train.values()])
            ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
            scale = float(np.median(ratios))
            raw = [qlike(d[target], d[name]) for d in test.values()]
            cal = [qlike(d[target], scale * d[name]) for d in test.values()]
            rows[name] = {"scale": scale, "qlike_raw": float(np.mean(raw)),
                          "qlike_cal": float(np.mean(cal)),
                          "per_day_cal": dict(zip(test, cal, strict=True))}
        # `har_b0` is the intercept with the log-normal correction folded in:
        # what the quoter's `har_b0` parameter takes.
        results[target.strip("_")] = {"har_beta": beta.tolist(), "har_s2": s2,
                                      "har_b0": float(beta[0] + 0.5 * s2), "rows": rows}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", required=True)
    args = parser.parse_args()
    days = snapshots.days_for(args.symbol)
    split = "2025-09-15"
    train_days = [d for d in days if d < split]
    test_days = [d for d in days if d >= split]
    results = evaluate(args.symbol, train_days, test_days)

    for target, block in results.items():
        label = "1 s" if target == "y1" else "5 s"
        print(f"\n{args.symbol}: target = next {HORIZON_S} s realised variance from {label} "
              f"returns; QLIKE on {len(test_days)} held-out days (lower is better)")
        print(f"{'estimator':<15} {'raw':>8} {'calibrated':>11} {'scale':>7}")
        order = sorted(block["rows"], key=lambda k: block["rows"][k]["qlike_cal"])
        for name in order:
            r = block["rows"][name]
            print(f"{name:<15} {r['qlike_raw']:8.4f} {r['qlike_cal']:11.4f} {r['scale']:7.3f}")
        print("HAR coefficients (const, 1m, 10m, 1h):",
              ", ".join(f"{b:+.3f}" for b in block["har_beta"]))

    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"{args.symbol}.json"
    out.write_text(json.dumps({"symbol": args.symbol, "train_days": train_days,
                               "test_days": test_days, "horizon_s": HORIZON_S,
                               "results": results}, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
