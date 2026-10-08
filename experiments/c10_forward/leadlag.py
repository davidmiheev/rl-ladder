"""C10 lead-lag: do markets outside the traded book anticipate its next 1-3 s?

C9's ceiling: quoting around the mid 1-3 s ahead is worth up to +40 points,
but the traded contract's own book forecasts only 2-10% of the 1 s move, and
a classifier of the top 1% of 3 s moves built on it ranks them with AUC 0.71
at 3-11% precision. This asks whether adding the same asset's other markets
(`external.py`) raises either number on held-out sessions -- and whether the
gain survives a realistic and a pessimistic arrival delay.

Per 100 ms step of the traded book (C9's `features.py` grid, now with each
step's local timestamp), from each outside source, using only events that
would have arrived by then (`external.LATENCY_NS` plus `extra` delay):

    ret_<m>     its price change over the last m (0.1, 0.5, 1, 5 s), bp
    flow_<m>    signed traded notional over the last m, over its 10-min mean
    basis       its price over our mid, bp
    basis_dev   that, minus its 60 s exponential average
    burst       trades in the last second, over the 10-min mean

Compared, held out, with C9's 17 own-book features alone:

* R^2 of the 1 s and 3 s mid move (boosted trees, the same settings as C9);
* AUC and precision of a large-move classifier: will the mid move against a
  side by the training sessions' top-1% 3 s move? One model scores both
  sides (signed features mirrored), as C9's DAgger student did.

At three delays: as assumed, +100 ms, +250 ms.

    python experiments/c10_forward/leadlag.py prepare --symbol ethusdc --shard 0/4
    python experiments/c10_forward/leadlag.py fit --symbol ethusdc
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "experiments" / "c9_fair_price"),
                str(REPO / "experiments" / "c8_risk_controls")]

import c8  # noqa: E402
import external  # noqa: E402
import features  # noqa: E402
import predict  # noqa: E402

OUT = REPO / "runs" / "c10"
WINDOWS_S = (0.1, 0.5, 1.0, 5.0)
DELAYS_MS = (0, 100, 250)
STRIDE = int(os.environ.get("C10_STRIDE", "5"))  # every 5th 100 ms step
K_REG = (10, 30)
K_LARGE, Q_LARGE = 30, 0.99


def ext_names(source: str) -> list[str]:
    s = source.replace("-", "_")
    names = [f"{s}_ret_{m:g}" for m in WINDOWS_S] + [f"{s}_basis", f"{s}_basis_dev"]
    if source != "bybit-book":
        names += [f"{s}_flow_{m:g}" for m in WINDOWS_S] + [f"{s}_burst"]
    return names


def _ewma(x: np.ndarray, hl_steps: float) -> np.ndarray:
    from scipy.signal import lfilter

    a = 1.0 - 0.5 ** (1.0 / hl_steps)
    return lfilter([a], [1.0, a - 1.0], x)


def ext_features(source: str, ext: dict, ts: np.ndarray, mid: np.ndarray,
                 extra_ns: int) -> np.ndarray:
    """One source's features on the step grid `ts`, causal at `ts`."""
    u = ext["t"] + external.LATENCY_NS[source] + extra_ns
    px = ext["px"]
    n = len(ts)
    if len(u) == 0:
        return np.zeros((n, len(ext_names(source))), np.float32)
    idx = np.searchsorted(u, ts, side="right") - 1
    last = np.where(idx >= 0, px[np.maximum(idx, 0)], np.nan)
    cols = []
    for m in WINDOWS_S:
        j = np.searchsorted(u, ts - int(m * 1e9), side="right") - 1
        then = np.where(j >= 0, px[np.maximum(j, 0)], np.nan)
        cols.append(np.log(last / then) * 1e4)
    basis = (last / mid - 1.0) * 1e4
    basis_f = np.nan_to_num(basis, nan=0.0)
    cols += [basis, basis_f - _ewma(basis_f, 600.0)]
    if source != "bybit-book":
        flow = np.concatenate([[0.0], np.cumsum(ext["flow"])])
        absflow = np.concatenate([[0.0], np.cumsum(np.abs(ext["flow"]))])
        count = np.arange(len(u) + 1, dtype=np.float64)
        k_now = idx + 1
        k_10m = np.searchsorted(u, ts - 600_000_000_000, side="right")
        scale = (absflow[k_now] - absflow[k_10m]) / 6000.0 + 1e-9  # mean |notional| per 100 ms
        for m in WINDOWS_S:
            k = np.searchsorted(u, ts - int(m * 1e9), side="right")
            cols.append((flow[k_now] - flow[k]) / (scale * m * 10.0))
        k_1s = np.searchsorted(u, ts - 1_000_000_000, side="right")
        rate = (count[k_now] - count[k_10m]) / 6000.0 + 1e-9
        cols.append((count[k_now] - count[k_1s]) / (rate * 10.0))
    x = np.column_stack(cols).astype(np.float32)
    x[~np.isfinite(x)] = 0.0
    return x


def day_matrix(symbol: str, date: str, extra_ns: int, sources=external.SOURCES):
    """(book features, external features, mid) for one session, every STRIDE-th step."""
    f = features.load(symbol, date)
    xb, mid = predict.build(f)
    ts = f["ts"].astype(np.int64)
    asset = external.asset_of(symbol)
    mid_ff = mid.copy()
    bad = ~np.isfinite(mid_ff)
    if bad.any():  # carry the last two-sided mid, as predict.build does
        i = np.where(~bad, np.arange(len(mid_ff)), 0)
        np.maximum.accumulate(i, out=i)
        mid_ff = mid_ff[i]
    xe = np.hstack([ext_features(s, external.load(s, asset, date), ts, mid_ff, extra_ns)
                    for s in sources])
    keep = slice(None, None, STRIDE)
    return xb[keep], xe[keep], mid, ts


def _targets(mid: np.ndarray) -> dict[int, np.ndarray]:
    return {k: predict.target(mid, k)[::STRIDE] for k in set(K_REG) | {K_LARGE}}


def _stack(symbol, days, extra_ns, sources):
    xb, xe, ys = [], [], {k: [] for k in set(K_REG) | {K_LARGE}}
    for date in days:
        b, e, mid, _ = day_matrix(symbol, date, extra_ns, sources)
        t = _targets(mid)
        xb.append(b)
        xe.append(e)
        for k in ys:
            ys[k].append(t[k].astype(np.float32))
    return np.concatenate(xb), np.concatenate(xe), {k: np.concatenate(v) for k, v in ys.items()}


def _mirror(n_ext_cols: list[str]) -> np.ndarray:
    """Sign flips that turn a bid-side row into an ask-side one."""
    book = [-1.0 if c in predict.FEATURES and c not in ("spread",) else 1.0
            for c in predict.FEATURES]
    ext = [1.0 if c.endswith("_burst") else -1.0 for c in n_ext_cols]
    return np.array(book + ext, np.float32)


def _gbm(x, y, xv, yv, objective, weight=None):
    import lightgbm as lgb

    params = {"objective": objective, "learning_rate": 0.05, "num_leaves": 31,
              "min_data_in_leaf": 2000, "feature_fraction": 0.8, "bagging_fraction": 0.5,
              "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1,
              "num_threads": int(os.environ.get("C10_THREADS", "4"))}
    return lgb.train(params, lgb.Dataset(x, y, weight=weight), num_boost_round=800,
                     valid_sets=[lgb.Dataset(xv, yv)],
                     callbacks=[lgb.early_stopping(40, verbose=False)])


def _auc(y, s):
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    pos = y > 0
    npos, nneg = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")


def _at_rate(y, s, rate):
    """Precision and recall when the top `rate` share of scores raises an alarm."""
    thr = np.quantile(s, 1.0 - rate)
    hit = s >= thr
    tp = (y[hit] > 0).sum()
    return float(tp / max(hit.sum(), 1)), float(tp / max((y > 0).sum(), 1))


def fit(symbol: str, train: list[str], test: list[str], sources=external.SOURCES) -> dict:
    names = [n for s in sources for n in ext_names(s)]
    mirror = _mirror(names)
    out_path = OUT / f"leadlag_{symbol}.json"
    report = {"symbol": symbol, "sources": list(sources), "train": train, "test": test,
              "n_ext_features": len(names), "stride": STRIDE, "delays": {}}
    if out_path.exists():  # resume: delays already fitted are kept
        old = json.loads(out_path.read_text())
        if old.get("train") == train and old.get("test") == test and old.get("stride") == STRIDE:
            report["delays"] = old["delays"]
    for d_ms in DELAYS_MS:
        if str(d_ms) in report["delays"]:
            continue
        extra = d_ms * 1_000_000
        xb, xe, y = _stack(symbol, train, extra, sources)
        tb, te, yt = _stack(symbol, test, extra, sources)
        n_val = len(xb) // len(train)  # the last training session: early stopping only
        row = {}
        for k in K_REG:
            ok, okt = np.isfinite(y[k]), np.isfinite(yt[k])
            for label, x, xt in (("book", xb, tb), ("book+ext", np.hstack([xb, xe]),
                                                      np.hstack([tb, te]))):
                xo, yo = x[ok], y[k][ok]
                m = _gbm(xo[:-n_val], yo[:-n_val], xo[-n_val:], yo[-n_val:], "regression")
                p = m.predict(xt[okt])
                yy = yt[k][okt]
                row[f"r2_{k}_{label}"] = float(1 - np.sum((yy - p) ** 2) / np.sum((yy - yy.mean()) ** 2)) * 100
        # Large moves: one row per side, the bid side mirrored.
        mv, mvt = y[K_LARGE], yt[K_LARGE]
        ok, okt = np.isfinite(mv), np.isfinite(mvt)
        thr = float(np.quantile(np.abs(mv[ok]), Q_LARGE))
        row["large_threshold_bp"] = thr
        for label, x, xt, mir in (("book", xb, tb, mirror[: xb.shape[1]]),
                                  ("book+ext", np.hstack([xb, xe]), np.hstack([tb, te]), mirror)):
            xa = np.vstack([x[ok], x[ok] * mir])
            ya = np.concatenate([(mv[ok] >= thr), (-mv[ok] >= thr)]).astype(np.float32)
            xta = np.vstack([xt[okt], xt[okt] * mir])
            yta = np.concatenate([(mvt[okt] >= thr), (-mvt[okt] >= thr)]).astype(np.float32)
            # Keep every positive and every 10th negative, reweighted.
            rng = np.random.default_rng(0)
            keep = (ya > 0) | (rng.random(len(ya)) < 0.1)
            w = np.where(ya[keep] > 0, 1.0, 10.0)
            val = np.zeros(len(ya), bool)
            half = len(ya) // 2
            val[half - n_val:half] = True
            val[-n_val:] = True
            tr = keep & ~val
            m = _gbm(xa[tr], ya[tr], xa[val], ya[val], "binary", weight=np.where(ya[tr] > 0, 1.0, 10.0))
            s = m.predict(xta)
            row[f"auc_{label}"] = _auc(yta, s)
            row[f"base_rate"] = float(yta.mean())
            for rate in (0.003, 0.01):
                prec, rec = _at_rate(yta, s, rate)
                row[f"precision_{rate:g}_{label}"] = prec
                row[f"recall_{rate:g}_{label}"] = rec
            if label == "book+ext":
                gain = m.feature_importance("gain")
                cols = list(predict.FEATURES) + names
                top = np.argsort(gain)[::-1][:8]
                row["top_features"] = {cols[i]: round(float(gain[i] / gain.sum()), 3) for i in top}
            del xa, ya, xta, yta
        report["delays"][str(d_ms)] = row
        print(symbol, f"+{d_ms} ms", json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                 for k, v in row.items()}), flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=1))
    return report


def prepare(symbol: str, days: list[str], shard: str) -> None:
    """Own-book features (needs the session's L2 file) and every outside source."""
    import config
    import snapshots

    index, count = (int(x) for x in shard.split("/"))
    asset = external.asset_of(symbol)
    for date in days[index::count]:
        if not features.path_for(symbol, date).exists():
            l2 = config.DATA / f"{symbol}_{date}.npz"
            fetched = False
            if not l2.exists() and not symbol.startswith("bybit-"):
                snapshots.download(symbol, date)
                fetched = True
            features.extract(symbol, date)
            if fetched:
                l2.unlink(missing_ok=True)
        for source in external.SOURCES:
            external.ensure(source, asset, date)
        print("prepared", symbol, date, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "fit"):
        p = sub.add_parser(name)
        p.add_argument("--symbol", required=True)
        p.add_argument("--train", nargs="*", default=None)
        p.add_argument("--test", nargs="*", default=None)
        if name == "prepare":
            p.add_argument("--shard", default="0/1")
    args = parser.parse_args()
    train = args.train or c8.TRAIN
    test = args.test or c8.TEST
    if args.cmd == "prepare":
        prepare(args.symbol, train + test, args.shard)
    else:
        fit(args.symbol, train, test)


if __name__ == "__main__":
    main()
