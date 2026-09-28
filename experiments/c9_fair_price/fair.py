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
    feat:<name>           one raw feature from `predict.py`, e.g. `feat:micro_bp`,
                          the micro-price's offset: quoting around the
                          micro-price instead of the mid

The oracle is not a strategy: it is the ceiling on what knowing the next
second is worth, and the noisy oracles map a predictor's R^2 to return.
"""

from __future__ import annotations

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


def series(src: str, symbol: str, date: str) -> np.ndarray:
    kind, *args = src.split(":")
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
    if kind == "pred":
        import predict

        return predict.series(args[0], int(args[1]), symbol, date, f)
    raise ValueError(f"unknown fair_src {src!r}")
