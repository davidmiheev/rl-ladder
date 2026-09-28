"""Can the mid 0.5-3 s ahead be predicted from what the quoter sees?

Features are computed per 100 ms quoting step from `features.py`'s files and
are strictly causal: step t uses the book at step t and flow up to step t.
The target is the mid's move over the next k steps, in bp. Models are fitted
on the 24 training sessions and scored on the 7 held-out ones:

    micro   the micro-price's offset from the mid, one coefficient
    obi     OBI's own signal (the 10 bp band imbalance, standardised over ten
            minutes, as the kernel computes it), one coefficient
    lin     every feature below, least squares
    gbm     every feature below, LightGBM (small, regularised trees)

Scores: R^2 on every held-out step, correlation, and the direction hit rate
on steps where the mid does move and the forecast is not zero.

    python experiments/c9_fair_price/predict.py fit --symbol ethusdc
    python experiments/c9_fair_price/predict.py report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "experiments" / "c8_risk_controls")]

import c8  # noqa: E402
import features  # noqa: E402

MODELS = REPO / "runs" / "c9" / "models"
HORIZONS = (5, 10, 15, 30)  # steps of 100 ms
WINDOW = 6000  # OBI's standardisation window, ten minutes
EPS = 1e-12


def _roll_sum(x: np.ndarray, n: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = c[n:] - c[:-n]
    return np.concatenate([c[1:n], out])


def _ewma(x: np.ndarray, hl: float) -> np.ndarray:
    from scipy.signal import lfilter

    a = 1.0 - 0.5 ** (1.0 / hl)
    y = lfilter([a], [1.0, a - 1.0], x)
    return y / (1.0 - (1.0 - a) ** np.arange(1, len(x) + 1))  # unbiased start


FEATURES = (
    "qi1", "micro_bp", "qi_2", "qi_10", "obi_z", "spread",
    "ofi_1", "ofi_5", "ofi_20", "ofi_100",
    "tfi_5", "tfi_20", "tfi_100",
    "ret_1", "ret_5", "ret_20", "ret_100",
)


def build(f: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(features, mid): features float32 (steps x len(FEATURES)), causal."""
    tick = float(f["tick"])
    ok = f["mid2"] >= 0
    mid = np.where(ok, f["mid2"] * tick / 2.0, np.nan)
    # Carry the last two-sided mid through one-sided moments.
    idx = np.where(ok, np.arange(len(mid)), 0)
    np.maximum.accumulate(idx, out=idx)
    mid_ff = mid[idx]
    spread = f["spread"].astype(np.float64)
    bq, aq = f["bq1"].astype(np.float64), f["aq1"].astype(np.float64)
    qi1 = (bq - aq) / (bq + aq + EPS)
    micro_bp = qi1 * spread * tick / 2.0 / mid_ff * 1e4
    qi = {}
    for band in ("2", "10"):
        b, a = f[f"bb_{band}"].astype(np.float64), f[f"ab_{band}"].astype(np.float64)
        qi[band] = (b - a) / (b + a + EPS)
    # OBI's signal as the kernel computes it: bid-minus-ask quantity in the
    # 10 bp band, standardised over the last ten minutes.
    raw = np.where(ok, f["bb_10"].astype(np.float64) - f["ab_10"], 0.0)
    n = _roll_sum(ok.astype(np.float64), WINDOW)
    s1, s2 = _roll_sum(raw, WINDOW), _roll_sum(raw * raw, WINDOW)
    mean = s1 / np.maximum(n, 1)
    var = s2 / np.maximum(n, 1) - mean * mean
    obi_z = np.where(var > 0, (raw - mean) / np.sqrt(np.maximum(var, EPS)), 0.0)
    depth = _ewma((bq + aq) / 2.0, 3000.0) + EPS
    ofi = f["ofi"].astype(np.float64)
    buy, sell = f["buy"].astype(np.float64), f["sell"].astype(np.float64)
    cols = {"qi1": qi1, "micro_bp": micro_bp, "qi_2": qi["2"], "qi_10": qi["10"],
            "obi_z": obi_z, "spread": spread}
    for m in (1, 5, 20, 100):
        cols[f"ofi_{m}"] = _roll_sum(ofi, m) / depth
    for m in (5, 20, 100):
        b, s = _roll_sum(buy, m), _roll_sum(sell, m)
        cols[f"tfi_{m}"] = (b - s) / (b + s + EPS)
    for m in (1, 5, 20, 100):
        r = np.zeros(len(mid))
        r[m:] = (mid_ff[m:] / mid_ff[:-m] - 1.0) * 1e4
        cols[f"ret_{m}"] = r
    x = np.column_stack([cols[c] for c in FEATURES]).astype(np.float32)
    x[~np.isfinite(x)] = 0.0
    return x, mid


def target(mid: np.ndarray, k: int) -> np.ndarray:
    y = np.full(len(mid), np.nan)
    y[:-k] = (mid[k:] / mid[:-k] - 1.0) * 1e4
    return y


def _stack(symbol: str, days: list[str], k: int, stride: int):
    xs, ys = [], []
    for date in days:
        x, mid = build(features.load(symbol, date))
        y = target(mid, k)
        sel = np.zeros(len(y), bool)
        sel[::stride] = True
        sel &= np.isfinite(y) & np.isfinite(mid)
        xs.append(x[sel])
        ys.append(y[sel].astype(np.float32))
    return np.concatenate(xs), np.concatenate(ys)


def _scores(y: np.ndarray, p: np.ndarray) -> dict:
    r2 = 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    moved = (y != 0) & (p != 0)
    return {"r2": float(r2) * 100, "corr": float(np.corrcoef(y, p)[0, 1]),
            "hit": float(np.mean(np.sign(y[moved]) == np.sign(p[moved]))) * 100,
            "moved_frac": float(np.mean(y != 0)) * 100, "std_bp": float(np.std(y)),
            "pred_std_bp": float(np.std(p))}


def fit(symbol: str, stride: int = 3) -> dict:
    import lightgbm as lgb

    MODELS.mkdir(parents=True, exist_ok=True)
    report = {}
    for k in HORIZONS:
        x, y = _stack(symbol, c8.TRAIN, k, stride)
        xt, yt = _stack(symbol, c8.TEST, k, 1)
        row = {}
        for name, cols in (("micro", ["micro_bp"]), ("obi", ["obi_z"]), ("lin", list(FEATURES))):
            j = [FEATURES.index(c) for c in cols]
            a = np.column_stack([x[:, j], np.ones(len(x))]).astype(np.float64)
            coef, *_ = np.linalg.lstsq(a, y, rcond=None)
            at = np.column_stack([xt[:, j], np.ones(len(xt))]).astype(np.float64)
            row[name] = _scores(yt, at @ coef)
            row[name]["train"] = _scores(y, a @ coef)["r2"]
            (MODELS / f"{name}_{symbol}_k{k}.json").write_text(
                json.dumps({"features": cols, "coef": coef.tolist()}))
        # LightGBM on every 3rd training step; the last training session is
        # the early-stopping set, never a held-out one.
        last = c8.TRAIN[-1]
        xv, yv = _stack(symbol, [last], k, stride)
        xf, yf = _stack(symbol, c8.TRAIN[:-1], k, stride)
        model = lgb.train(
            {"objective": "regression", "learning_rate": 0.05, "num_leaves": 31,
             "min_data_in_leaf": 2000, "feature_fraction": 0.8, "bagging_fraction": 0.5,
             "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "num_threads": 2},
            lgb.Dataset(xf, yf, feature_name=list(FEATURES)), num_boost_round=600,
            valid_sets=[lgb.Dataset(xv, yv)], callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        model.save_model(str(MODELS / f"gbm_{symbol}_k{k}.txt"))
        row["gbm"] = _scores(yt, model.predict(xt))
        row["gbm"]["train"] = _scores(y, model.predict(x))["r2"]
        row["gbm"]["rounds"] = model.best_iteration
        gain = model.feature_importance("gain")
        row["gbm"]["importance"] = dict(zip(FEATURES, (gain / gain.sum()).round(4).tolist()))
        report[k] = row
        print(symbol, k, {m: round(v["r2"], 2) for m, v in row.items()}, flush=True)
        del x, y, xt, yt, xf, yf, xv, yv
    (MODELS / f"report_{symbol}.json").write_text(json.dumps(report, indent=1))
    return report


_CACHE: dict = {}


def series(model: str, k: int, symbol: str, date: str, f: dict) -> np.ndarray:
    """A fitted model's forecast of the k-step move, bp, for every step of a session."""
    x, _ = build(f)
    if model == "gbm":
        import lightgbm as lgb

        key = (model, symbol, k)
        if key not in _CACHE:
            _CACHE[key] = lgb.Booster(model_file=str(MODELS / f"gbm_{symbol}_k{k}.txt"))
        return _CACHE[key].predict(x, num_threads=1)
    spec = json.loads((MODELS / f"{model}_{symbol}_k{k}.json").read_text())
    j = [FEATURES.index(c) for c in spec["features"]]
    coef = np.array(spec["coef"])
    return x[:, j].astype(np.float64) @ coef[:-1] + coef[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--symbol", required=True)
    p.add_argument("--stride", type=int, default=3)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.cmd == "fit":
        fit(args.symbol, args.stride)
    else:
        for path in sorted(MODELS.glob("report_*.json")):
            rep = json.loads(path.read_text())
            for k, row in rep.items():
                print(path.stem[7:], k, {m: (round(v["r2"], 2), round(v["hit"], 1))
                                         for m, v in row.items()})


if __name__ == "__main__":
    main()
