"""The nonlinear check on `book_info.py`: does a boosted tree find what OLS did not?

`book_info.py` is linear on purpose, as a floor. A floor can hide information
that only shows up through interactions -- BTC's move mattering only when our
own book is thin, say. So the same features and the same split go through
LightGBM, and the question is asked once more: does adding a group raise the
**held-out** R^2 above what the C2 features alone reach?

Early stopping uses the last training day as validation, never a held-out
one. The trees are kept small and heavily regularised: a flexible model on a
signal this weak finds noise first, and this is a test of whether the signal
exists, not an attempt to maximise it.

Needs LightGBM, which the project does not depend on:

    uv pip install lightgbm
    python experiments/c7_signals/book_info_gbm.py --symbol ethusdc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import book_info  # noqa: E402
import snapshots  # noqa: E402

RUNS = Path(__file__).resolve().parents[2] / "runs" / "c7-book"
HORIZONS_S = (1, 10, 60)
SETS = {
    "c2": ["c2"],
    "c2 + btc fresh returns": ["c2", "btc_fresh"],
    "own book, everything": ["c2", "levels", "bands", "ofi", "returns"],
    "everything incl. btc": ["c2", "levels", "bands", "ofi", "returns", "btc"],
}
PARAMS = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 15,
          "min_data_in_leaf": 2000, "feature_fraction": 0.8, "bagging_fraction": 0.8,
          "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "num_threads": 1}


def r2(y: np.ndarray, pred: np.ndarray, train_mean: float) -> float:
    return float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - train_mean) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", required=True)
    args = parser.parse_args()
    days = snapshots.days_for(args.symbol)
    train_days = [d for d in days if d < "2025-09-15"]
    test_days = [d for d in days if d >= "2025-09-15"]
    tables = {d: book_info.day_table(args.symbol, d) for d in days}
    groups = book_info.group_columns("own", snapshots.LEVELS, snapshots.BANDS_BP)
    groups["btc_fresh"] = [c for c in groups["btc"] if c.startswith("ref_fresh")]

    fit_days, val_days = train_days[:-1], train_days[-1:]
    results: dict = {}
    print(f"{args.symbol}: LightGBM, held-out R^2 in % ({len(fit_days)} fit + 1 validation "
          f"day -> {len(test_days)} held-out days)")
    print(f"{'features':<26}" + "".join(f"{f'{h}s':>9}" for h in HORIZONS_S))
    for label, names in SETS.items():
        cols = [c for g in names for c in groups[g]]
        row = {}
        for h in HORIZONS_S:
            def xy(ds, cols=cols, h=h):
                x = np.vstack([np.column_stack([tables[d][0][c] for c in cols]) for d in ds])
                y = np.concatenate([tables[d][1][f"fwd_{h}s"] for d in ds])
                ok = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
                return x[ok], y[ok]
            xf, yf = xy(fit_days)
            xv, yv = xy(val_days)
            xt, yt = xy(test_days)
            model = lgb.train(PARAMS, lgb.Dataset(xf, yf), num_boost_round=500,
                              valid_sets=[lgb.Dataset(xv, yv)],
                              callbacks=[lgb.early_stopping(30, verbose=False)])
            row[h] = 100 * r2(yt, model.predict(xt), float(np.mean(yf)))
        results[label] = row
        print(f"{label:<26}" + "".join(f"{row[h]:9.3f}" for h in HORIZONS_S), flush=True)

    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"{args.symbol}-gbm.json"
    out.write_text(json.dumps({"symbol": args.symbol, "horizons_s": list(HORIZONS_S),
                               "r2_pct": results}, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
