"""Is there information deeper in the book, or in BTCUSDT's book?

C2 gave the policy two imbalance features: the queue at the touch and the sum
over five ticks a side. Five ticks is 0.11 bp on ETHUSDC, so on the asset C6
trains most on the "deep" feature is the touch again, and nothing the policy
sees reaches further than that. BTC leads most of the crypto complex, and none
of its book is in the state at all.

Before any of it goes into a policy, this measures whether it predicts
anything. Nested linear models, fitted on the training days and scored on the
held-out days, forecast the traded asset's mid return over the next 1-60 s.
Each group of features is added on top of the previous ones, and the number
reported is the **out-of-sample R^2** and what the group *added* to it. A
group that adds nothing out of sample is not information a policy can use,
however large its in-sample fit.

Features, all known at the end of second t:

    c2       touch imbalance, 5-tick imbalance, signed flow over 1 s and 10 s:
             what the RL state already has
    levels   imbalance out to 20, 50, 200 ticks
    bands    imbalance within 1, 2, 5, 10, 25, 50, 100 bp of the mid
    ofi      order-flow imbalance at the touch over 1, 10, 60 s, in units of
             the trailing 10-minute mean touch size
    returns  own mid return over the last 1, 10, 60 s
    btc      BTCUSDT: return over the last 1, 5, 10, 60 s, touch and band
             imbalances, OFI and signed flow

Linear on purpose: a linear model is the floor on what a feature carries, and
the RL actor is small enough that a signal a linear model cannot find at all
is unlikely to be one it finds either.

    python experiments/c7_signals/book_info.py --symbol ethusdc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import snapshots  # noqa: E402

RUNS = Path(__file__).resolve().parents[2] / "runs" / "c7-book"
REFERENCE = "btcusdt"
HORIZONS_S = (1, 5, 10, 30, 60)
#: Evaluation seconds are thinned to every EVERY_S-th, from WARMUP_S on.
#: Overlapping targets do not bias an out-of-sample R^2, only thin its
#: effective sample; thinning keeps the design matrix small.
EVERY_S = 2
WARMUP_S = 900
GROUPS = ("c2", "levels", "bands", "ofi", "returns", "btc")


def _imb(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    total = b + a
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (b - a) / total
    out[~np.isfinite(out)] = 0.0
    return out


def _trailing_sum(x: np.ndarray, window: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    out = np.zeros(len(x))
    idx = np.arange(len(x))
    start = np.maximum(idx + 1 - window, 0)
    out[:] = c[idx + 1] - c[start]
    return out


def _trailing_mean(x: np.ndarray, window: int) -> np.ndarray:
    n = _trailing_sum(np.isfinite(x).astype(float), window)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _trailing_sum(x, window) / n


def _log_mid(blob) -> np.ndarray:
    """log mid in bp at the end of each second, forward-filled."""
    m = blob["mid100"].astype(np.float64)
    m[m < 0] = np.nan
    n_sec = len(blob["ts"])
    m = m[9: n_sec * 10: 10]
    idx = np.where(np.isfinite(m), np.arange(len(m)), 0)
    np.maximum.accumulate(idx, out=idx)
    m = m[idx]
    m[: np.argmax(np.isfinite(m))] = m[np.isfinite(m)][0]
    return np.log(m * float(blob["tick"]) / 2.0) * 1e4


def book_features(blob, prefix: str) -> dict[str, np.ndarray]:
    """Every per-second feature of one symbol-day, keyed `<prefix>_<name>`."""
    lm = _log_mid(blob)
    bl, al = blob["bid_lvl"].astype(np.float64), blob["ask_lvl"].astype(np.float64)
    bb, ab = blob["bid_bp"].astype(np.float64), blob["ask_bp"].astype(np.float64)
    levels, bands = blob["levels"], blob["bands_bp"]
    f: dict[str, np.ndarray] = {}
    for i, lvl in enumerate(levels):
        f[f"{prefix}_imb_l{lvl}"] = _imb(bl[:, i], al[:, i])
    for i, band in enumerate(bands):
        f[f"{prefix}_imb_{band:g}bp"] = _imb(bb[:, i], ab[:, i])
    touch = _trailing_mean((bl[:, 0] + al[:, 0]) / 2.0, 600)
    volume = _trailing_mean(blob["buy_qty"] + blob["sell_qty"], 600)
    ofi = blob["ofi"].astype(np.float64)
    flow = (blob["buy_qty"] - blob["sell_qty"]).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        for w in (1, 10, 60):
            f[f"{prefix}_ofi_{w}s"] = np.nan_to_num(np.tanh(_trailing_sum(ofi, w) / (touch * w)))
            f[f"{prefix}_flow_{w}s"] = np.nan_to_num(np.tanh(_trailing_sum(flow, w)
                                                             / (volume * w)))
    for w in (1, 5, 10, 60):
        r = np.zeros(len(lm))
        r[w:] = lm[w:] - lm[:-w]
        f[f"{prefix}_ret_{w}s"] = r
    f["_log_mid"] = lm
    f["_ts"] = blob["ts"]
    return f


def group_columns(own: str, levels, bands) -> dict[str, list[str]]:
    return {
        "c2": [f"{own}_imb_l1", f"{own}_imb_l5", f"{own}_flow_1s", f"{own}_flow_10s"],
        "levels": [f"{own}_imb_l{lvl}" for lvl in levels if lvl > 5],
        "bands": [f"{own}_imb_{b:g}bp" for b in bands],
        "ofi": [f"{own}_ofi_{w}s" for w in (1, 10, 60)] + [f"{own}_flow_60s"],
        "returns": [f"{own}_ret_{w}s" for w in (1, 10, 60)],
        "btc": [f"ref_ret_{w}s" for w in (1, 5, 10, 60)]
        + ["ref_imb_l1", "ref_imb_l5", "ref_imb_l50"]
        + [f"ref_imb_{b:g}bp" for b in bands]
        + [f"ref_ofi_{w}s" for w in (1, 10)] + [f"ref_flow_{w}s" for w in (1, 10)],
    }


def day_table(symbol: str, date: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Features and forward returns for one day, on the thinned evaluation grid."""
    own_blob = np.load(snapshots.feature_path(symbol, date))
    ref_blob = np.load(snapshots.feature_path(REFERENCE, date))
    own = book_features(own_blob, "own")
    ref = book_features(ref_blob, "ref")
    # The reference sample in force at each of our seconds: the last one
    # stamped at or before it, so nothing from the future leaks in.
    pos = np.searchsorted(ref["_ts"], own["_ts"], side="right") - 1
    ok_ref = pos >= 0
    pos = np.clip(pos, 0, len(ref["_ts"]) - 1)
    x = {k: v for k, v in own.items() if not k.startswith("_")}
    for k, v in ref.items():
        if not k.startswith("_"):
            x[k] = np.where(ok_ref, v[pos], 0.0)
    lm = own["_log_mid"]
    n = len(lm)
    y = {}
    for h in HORIZONS_S:
        r = np.full(n, np.nan)
        r[: n - h] = lm[h:] - lm[: n - h]
        y[f"fwd_{h}s"] = r
    keep = np.arange(WARMUP_S, n - max(HORIZONS_S), EVERY_S)
    keep = keep[np.isfinite(lm[keep])]
    return {k: v[keep] for k, v in x.items()}, {k: v[keep] for k, v in y.items()}


def _stack(tables, cols):
    return np.vstack([np.column_stack([t[c] for c in cols]) for t in tables])


def oos_r2(xtr, ytr, xte, yte) -> float:
    """R^2 on held-out data of an OLS fit on training data (with intercept).

    Out-of-sample R^2 can be negative: a model worse than predicting the
    training mean is worse than nothing, and reported as such.
    """
    a = np.column_stack([np.ones(len(xtr)), xtr])
    ok = np.all(np.isfinite(a), axis=1) & np.isfinite(ytr)
    beta, *_ = np.linalg.lstsq(a[ok], ytr[ok], rcond=None)
    b = np.column_stack([np.ones(len(xte)), xte])
    ok = np.all(np.isfinite(b), axis=1) & np.isfinite(yte)
    pred = b[ok] @ beta
    resid = yte[ok] - pred
    base = yte[ok] - np.mean(ytr[np.isfinite(ytr)])
    return float(1.0 - np.sum(resid**2) / np.sum(base**2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", required=True)
    args = parser.parse_args()
    days = snapshots.days_for(args.symbol)
    split = "2025-09-15"
    train_days = [d for d in days if d < split]
    test_days = [d for d in days if d >= split]
    tables = {d: day_table(args.symbol, d) for d in days}
    groups = group_columns("own", snapshots.LEVELS, snapshots.BANDS_BP)

    xtr = [tables[d][0] for d in train_days]
    xte = [tables[d][0] for d in test_days]
    results: dict = {"nested": {}, "alone": {}, "single": {}}
    print(f"{args.symbol}: out-of-sample R^2 in %, {len(train_days)} training days -> "
          f"{len(test_days)} held-out days")
    print(f"{'added group':<12}" + "".join(f"{f'{h}s':>9}" for h in HORIZONS_S))
    cols: list[str] = []
    prev = {h: 0.0 for h in HORIZONS_S}
    for g in GROUPS:
        cols = cols + groups[g]
        row = {}
        for h in HORIZONS_S:
            ytr = np.concatenate([tables[d][1][f"fwd_{h}s"] for d in train_days])
            yte = np.concatenate([tables[d][1][f"fwd_{h}s"] for d in test_days])
            row[h] = 100 * oos_r2(_stack(xtr, cols), ytr, _stack(xte, cols), yte)
        results["nested"][g] = row
        print(f"+{g:<11}" + "".join(f"{row[h]:9.3f}" for h in HORIZONS_S))
        print(f"{'  (added)':<12}" + "".join(f"{row[h] - prev[h]:+9.3f}" for h in HORIZONS_S))
        prev = row

    print("\neach group on its own:")
    for g in GROUPS:
        row = {}
        for h in HORIZONS_S:
            ytr = np.concatenate([tables[d][1][f"fwd_{h}s"] for d in train_days])
            yte = np.concatenate([tables[d][1][f"fwd_{h}s"] for d in test_days])
            row[h] = 100 * oos_r2(_stack(xtr, groups[g]), ytr, _stack(xte, groups[g]), yte)
        results["alone"][g] = row
        print(f" {g:<11}" + "".join(f"{row[h]:9.3f}" for h in HORIZONS_S))

    # Single-feature out-of-sample correlation with the 10 s forward return:
    # the sign has to hold across the split for it to count.
    print("\nsingle features, correlation with the 10 s forward return (train / held out):")
    ytr = np.concatenate([tables[d][1]["fwd_10s"] for d in train_days])
    yte = np.concatenate([tables[d][1]["fwd_10s"] for d in test_days])
    for c in [c for g in GROUPS for c in groups[g]]:
        a = np.concatenate([t[c] for t in xtr])
        b = np.concatenate([t[c] for t in xte])
        oka, okb = np.isfinite(a) & np.isfinite(ytr), np.isfinite(b) & np.isfinite(yte)
        ctr = float(np.corrcoef(a[oka], ytr[oka])[0, 1]) if a[oka].std() > 0 else 0.0
        cte = float(np.corrcoef(b[okb], yte[okb])[0, 1]) if b[okb].std() > 0 else 0.0
        results["single"][c] = {"train": ctr, "test": cte}
        print(f"  {c:<18} {ctr:+.4f} {cte:+.4f}")

    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"{args.symbol}.json"
    out.write_text(json.dumps({"symbol": args.symbol, "train_days": train_days,
                               "test_days": test_days, "horizons_s": list(HORIZONS_S),
                               **results}, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
