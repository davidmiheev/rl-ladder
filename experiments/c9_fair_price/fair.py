"""Fair-price series for C9: the oracle, noisy oracles and causal predictions.

Every series is one value per quoting step, in bp of the current mid: how far
the centre of the quotes is moved from the mid (on top of the strategy's own
signal, or instead of it -- `fair_mode` in the kernel). A configuration names
its series with `fair_src`:

    oracle:<k>            the mid k steps (k x 100 ms) ahead: mid[t+k]/mid[t]-1
    noisy:<k>:<r2>        a prediction of that move with R^2 = r2 exactly:
                          r2 * (move + noise), the noise a k-step moving sum of
                          white noise (so it is as autocorrelated as the move
                          itself), scaled to make the R^2 r2; seeded by date
    pred:<model>:<k>      a causal model's prediction (see `predict.py`)
    gate:<q>:<src>        <src> where its size is in the top (1-q) of the training
                          sessions' values, zero elsewhere: act on large
                          moves only (the threshold is fitted on training
                          sessions and cached in runs/c9/gates.json)
    dag:<strategy>:<round> the DAgger student's per-side market scores (`dagger.py`),
                          columns 1 and 2 of the kernel's series
    feat:<name>           one raw feature from `predict.py`, e.g. `feat:micro_bp`,
                          the micro-price's offset: quoting around the
                          micro-price instead of the mid

The oracle is not a strategy: it is the ceiling on what knowing the next
second is worth, and the noisy oracles map a predictor's R^2 to return.
"""

from __future__ import annotations

import os
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import features  # noqa: E402


def mid_of(f: dict[str, np.ndarray]) -> np.ndarray:
    mid = f["mid2"].astype(np.float64) * float(f["tick"]) / 2.0
    mid[f["mid2"] < 0] = np.nan
    return mid


def future_move_bp(mid: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(mid), np.nan)
    out[:-k] = (mid[k:] / mid[:-k] - 1.0) * 1e4
    return out


def noisy(move: np.ndarray, k: int, r2: float, seed: int) -> np.ndarray:
    ok = np.isfinite(move)
    var = np.nanvar(move)
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(len(move) + k)
    noise = np.convolve(white, np.ones(k), "valid")[: len(move)]
    noise *= np.sqrt(var * (1.0 / r2 - 1.0) / np.var(noise))
    out = np.full(len(move), np.nan)
    out[ok] = r2 * (move[ok] + noise[ok])
    return out


GATES = HERE.parents[1] / "runs" / "c9" / "gates.json"


QUANTILES = (0.9, 0.95, 0.99, 0.999)


def gate_threshold(q: float, src: str, symbol: str) -> float:
    """The q-quantile of |src| over the training sessions, cached (every
    quantile in `QUANTILES` from one pass, since a forecast's pass is slow)."""
    import json

    import c8

    key = f"{symbol}|{q:g}|{src}"
    table = json.loads(GATES.read_text()) if GATES.exists() else {}
    if key not in table:
        vals = []
        for date in c8.TRAIN:
            x = np.abs(series(src, symbol, date))
            vals.append(x[np.isfinite(x)][::7])
        vals = np.concatenate(vals)
        table = json.loads(GATES.read_text()) if GATES.exists() else {}
        for qq in sorted(set(QUANTILES) | {q}):
            table[f"{symbol}|{qq:g}|{src}"] = float(np.quantile(vals, qq))
        GATES.parent.mkdir(parents=True, exist_ok=True)
        tmp = GATES.with_name(f"gates.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(table, indent=1, sort_keys=True))
        os.replace(tmp, GATES)
    return table[key]


def series(src: str, symbol: str, date: str) -> np.ndarray:
    kind, *args = src.split(":")
    if kind == "gate":
        q, inner = float(args[0]), ":".join(args[1:])
        x = series(inner, symbol, date)
        return np.where(np.abs(x) >= gate_threshold(q, inner, symbol), x, 0.0)
    f = features.load(symbol, date)
    if kind == "oracle":
        return future_move_bp(mid_of(f), int(args[0]))
    if kind == "noisy":
        k, r2 = int(args[0]), float(args[1])
        seed = zlib.crc32(f"{symbol}{date}{k}{r2}".encode())
        return noisy(future_move_bp(mid_of(f), k), k, r2, seed)
    if kind == "feat":
        import predict

        x, _ = predict.build(f)
        return x[:, predict.FEATURES.index(args[0])].astype(np.float64)
    if kind == "dag":
        import dagger

        return dagger.series(args[0], int(args[1]), symbol, date, f)
    if kind == "pred":
        import predict

        return predict.series(args[0], int(args[1]), symbol, date, f)
    raise ValueError(f"unknown fair_src {src!r}")
