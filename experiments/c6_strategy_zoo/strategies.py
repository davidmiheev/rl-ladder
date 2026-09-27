"""The five strategies' published parameters, converted into C6's units.

Every number here is traceable to a notebook cell.  Where a notebook states a
value in ticks or in price units of the asset it was written for, the
conversion to basis points is written out in the comment, because that
conversion is the only place a transcription error can hide silently -- a wrong
`gamma` produces an obviously broken run, a wrong unit produces a plausible one.

Two families of preset:

  `<name>`        the published parameters, converted, nothing else changed.
  `<name>-wide`   the same with the asset-dependent spread floor raised.

The wide family exists because C1's central negative result was that on this
venue the calibrated GLFT half spread -- about 2 bp -- sits well inside the
break-even spread implied by measured adverse selection, and because three of
the five published strategies independently chose a half spread near 30 bp on
the assets they were tuned for.  A spread floor is the cheapest possible fix,
and if it is enough then the RL modification has a much harder baseline to beat
than C3 had.  It is set per asset from the asset's own measured adverse
selection, not chosen by hand -- see `calibrate_floor.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "c1_glft_baseline"))

import config  # noqa: E402
import zoo  # noqa: E402

#: Sizing, shared by all five so the master table compares strategies and not
#: position limits.  Twenty orders of $5,000 exactly fill a $100,000 cap, and
#: the cap equals the book the statistics are computed against, so the numbers
#: are returns on the capital actually at risk and leverage is 1x.  C1 ran
#: 1 ETH a side to 20 ETH against a $100,000 book, which at ETH ~ 4,400 is
#: $4,400 and $88,000 -- close enough that C1's Sharpe stays comparable, and
#: the round numbers are asset-independent, which C1's were not.
ORDER_QTY_DOLLAR = 5_000.0
MAX_POSITION_DOLLAR = 100_000.0
BOOK_SIZE = 100_000.0

#: Grid depth.  The notebooks use 20 (glft, hfgrid) or 1-5 (the alpha three);
#: C6 gives all five the same grid so the comparison is about the fair price
#: and the spread, and sweeps the depth separately.
GRID_NUM = 20
GRID_SWEEP = (1, 5, 20, 40, 60)

#: Reaction interval.  The notebooks' 100 ms, and C1's interval study found
#: nothing that changes if it shrinks.
INTERVAL_NS = 100_000_000
#: Equity is recorded on a 100 ms grid whatever the reaction interval, so
#: Sharpe is computed on the same sampling grid across every run.
RECORD_EVERY_NS = 100_000_000

#: Spread floors, in bp of price, for the `-wide` family.  Set per asset by
#: `calibrate_floor.py` from that asset's own measured adverse selection on
#: *training* days only.  The entries are filled in by that script; a symbol
#: missing from here has no `-wide` preset rather than a guessed one.
SPREAD_FLOOR_BP: dict[str, float] = {}

_FLOOR_PATH = Path(__file__).resolve().parent / "spread_floor.json"
if _FLOOR_PATH.exists():
    import json

    SPREAD_FLOOR_BP.update({k: float(v) for k, v in json.loads(_FLOOR_PATH.read_text()).items()})


def _base() -> np.ndarray:
    params = np.zeros(zoo.PARAM_SIZE, np.float64)
    params[zoo.P_ORDER_QTY_DOLLAR] = ORDER_QTY_DOLLAR
    params[zoo.P_MAX_POSITION_DOLLAR] = MAX_POSITION_DOLLAR
    params[zoo.P_GRID_NUM] = GRID_NUM
    params[zoo.P_OBI_WINDOW] = 1  # unused unless the strategy is `obi`
    params[zoo.P_BETA] = 1.0
    # C7's additions, at values that leave C6's quoter unchanged: sigma at its
    # measured scale, no HAR (`har_b1 = 0`), no markout widening (`kappa = 0`).
    params[zoo.P_VOL_SCALE] = 1.0
    params[zoo.P_MARKOUT_STEPS] = 50  # 5 s at the 100 ms quoting interval
    params[zoo.P_MARKOUT_HALFLIFE] = 20.0  # fills
    # C8's loss limit measures PnL against the book the statistics use.
    params[zoo.P_BOOK_DOLLAR] = BOOK_SIZE
    return params


def glft(symbol: str) -> np.ndarray:
    """`GLFT Market Making Model and Grid Trading.ipynb`, second half.

    `gamma = 0.05`, `adj1 = 1`, `adj2 = 0.05`: the notebook's own values, and
    C1's `glft-grid`.  The half spread and skew come from the live fit of
    `A`, `k` and `sigma`, so there is nothing to convert -- the bp floor is 0
    and the model decides.
    """
    params = _base()
    params[zoo.P_STRAT] = zoo.GLFT
    params[zoo.P_GAMMA] = 0.05
    params[zoo.P_ADJ1] = 1.0
    params[zoo.P_ADJ2] = 0.05
    params[zoo.P_HALF_SPREAD_BP] = 0.0
    params[zoo.P_GRID_INTERVAL_BP] = 0.0  # grid step follows the half spread
    return params


def hfgrid(symbol: str) -> np.ndarray:
    """`High-Frequency Grid Trading.ipynb`.

    `vol_to_half_spread = 5`, `skew = 1`, `min_grid_step = tick_size`,
    `grid_num = 20`.  The skew there is multiplicative on the half spread and
    against notional position over the notional cap, which is exactly C6's
    `norm_pos`, so `skew_bp` carries the dimensionless 1.0 unchanged and the
    kernel applies it multiplicatively for this strategy alone.
    """
    params = _base()
    params[zoo.P_STRAT] = zoo.HFGRID
    params[zoo.P_VOL_TO_HALF_SPREAD] = 5.0
    params[zoo.P_SKEW_BP] = 1.0
    params[zoo.P_HALF_SPREAD_BP] = 0.0
    params[zoo.P_GRID_INTERVAL_BP] = 0.0
    return params


def apt(symbol: str, beta: float = 1.0) -> np.ndarray:
    """`Market Making with Alpha - APT.ipynb`.

    `half_spread = 0.0003` as a ratio, i.e. 3 bp.  `skew = half_spread / 20`
    per unit of `position / order_qty`, and the notebook caps position at
    `20 * order_qty`, so the shift at a full position is
    `0.0003/20 * 20 = 0.0003` -- 3 bp in C6's convention, the same as the half
    spread.  `grid_interval = 0.0003` is likewise 3 bp.
    """
    params = _base()
    params[zoo.P_STRAT] = zoo.APT
    params[zoo.P_HALF_SPREAD_BP] = 3.0
    params[zoo.P_SKEW_BP] = 3.0
    params[zoo.P_GRID_INTERVAL_BP] = 3.0
    params[zoo.P_BETA] = beta
    return params


def basis(symbol: str) -> np.ndarray:
    """`Market Making with Alpha - Basis.ipynb`.

    Same spread, skew and cap as APT (`half_spread = 0.0003`,
    `skew = half_spread / 20`, `max = 20 * order_qty`); the notebook's grid is
    one order at `tick_size`, which C6 replaces with the shared 20-deep grid so
    the five are comparable, and `grid_interval_bp = 0` lets the step follow
    the half spread as it does for glft and hfgrid.
    """
    params = _base()
    params[zoo.P_STRAT] = zoo.BASIS
    params[zoo.P_HALF_SPREAD_BP] = 3.0
    params[zoo.P_SKEW_BP] = 3.0
    params[zoo.P_GRID_INTERVAL_BP] = 0.0
    return params


def obi(symbol: str) -> np.ndarray:
    """`Market Making with Alpha - Order Book Imbalance.ipynb`, volume variant.

    The notebook gives two tunings per asset and they agree once converted,
    which is the useful check that the conversion is right:

        wide    BTC ~27,000: half 80, skew 3.5/lot, c1 160, depth 2.5%
                -> 29.6 bp, 3.5*20/27000 = 25.9 bp, 59.3 bp
                ETH ~1,800:  half 5,  skew 0.2/lot, c1 10,  depth 2.5%
                -> 27.8 bp, 0.2*20/1800  = 22.2 bp, 55.6 bp
        volume  BTC ~27,000: half 10, skew 2/lot,   c1 20,  depth 0.1%
                -> 3.7 bp,  2*20/27000   = 14.8 bp, 7.4 bp

    C6 takes the volume variant as the baseline: it is the one that reports the
    higher Sharpe (14.01 against 10.83), and a tuning that barely trades is not
    a market maker to compare market makers against.  The wide tuning is
    reachable as `obi-wide`.

    `looking_depth = 0.001` and a ten-minute standardisation window, as there.
    """
    params = _base()
    params[zoo.P_STRAT] = zoo.OBI
    params[zoo.P_HALF_SPREAD_BP] = 3.7
    params[zoo.P_SKEW_BP] = 14.8
    params[zoo.P_ALPHA_BP] = 7.4
    params[zoo.P_OBI_DEPTH] = 0.001
    params[zoo.P_OBI_WINDOW] = int(600e9 / INTERVAL_NS)  # ten minutes
    params[zoo.P_GRID_INTERVAL_BP] = 0.0
    tick = config.SYMBOLS[symbol]["tick_size"]
    params[zoo.P_ROI_LB_TICK] = 0.0
    params[zoo.P_ROI_UB_TICK] = round(config.SYMBOLS[symbol]["roi_ub"] / tick)
    return params


BUILDERS = {"glft": glft, "hfgrid": hfgrid, "apt": apt, "basis": basis, "obi": obi}

#: Which strategies need a reference leg, and therefore only run on a symbol
#: that has one.
PAIR_STRATEGIES = ("apt", "basis")

NAMES = tuple(BUILDERS) + tuple(f"{name}-wide" for name in BUILDERS)


def build(name: str, symbol: str, **kwargs) -> np.ndarray:
    """Parameters for `name` on `symbol`.

    A `-wide` suffix raises the spread floor to the asset's calibrated value;
    everything else is unchanged, so the pair differs in exactly one number.
    """
    wide = name.endswith("-wide")
    base = name[: -len("-wide")] if wide else name
    if base not in BUILDERS:
        raise SystemExit(f"unknown strategy {name!r}; known: {sorted(NAMES)}")
    params = BUILDERS[base](symbol, **kwargs)
    if wide:
        if symbol not in SPREAD_FLOOR_BP:
            raise SystemExit(
                f"no calibrated spread floor for {symbol}; run calibrate_floor.py --symbol {symbol}"
            )
        params[zoo.P_HALF_SPREAD_BP] = max(params[zoo.P_HALF_SPREAD_BP], SPREAD_FLOOR_BP[symbol])
    return params


def describe(params: np.ndarray) -> dict[str, float]:
    return {name: float(params[i]) for i, name in enumerate(zoo.PARAM_NAMES)}
