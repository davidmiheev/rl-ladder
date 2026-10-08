"""C6 -- five published market-making strategies behind one quoting core.

C1 to C3 asked whether RL can improve *one* strategy (GLFT). The obvious
objection is that GLFT was the wrong strategy: it loses to adverse selection on
this venue, and "RL beats a losing baseline" is a weak claim. C6 answers it by
running the same experiment against the four other strategies hftbacktest
publishes, and by giving all five the same treatment: same days, same latency,
same queue model, same fee model, same sizing, same scoring.

The five, with the notebook each is transcribed from:

  glft    `GLFT Market Making Model and Grid Trading.ipynb` -- half spread and
          skew from the closed-form solution of Guéant, Lehalle &
          Fernandez-Tapia (https://huggingface.co/papers/1605.01862), driven by
          a live fit of the order-arrival intensity `lambda(d) = A exp(-k d)`
          and the mid-price volatility.  This is C1's baseline.
  hfgrid  `High-Frequency Grid Trading.ipynb` -- no model of arrival intensity
          at all: the half spread is a multiple of realised volatility, and the
          fair price is the micro-price rather than the mid.
  apt     `Market Making with Alpha - APT.ipynb` -- a lead-lag fair price,
          `fair = (1 + beta * ref_return) * own_past_price`, where the
          reference leg's recent return predicts the traded leg.
  basis   `Market Making with Alpha - Basis.ipynb` -- `fair = ref + basis`,
          with the basis a rolling mean of (own - reference).
  obi     `Market Making with Alpha - Order Book Imbalance.ipynb` -- the fair
          price is the mid shifted by a standardised order-book imbalance.

Two deliberate departures from the notebooks, both so that the five are
comparable rather than each in its own units:

1. **Everything is a ratio of price.**  Order size, position cap, half spread,
   skew and grid step are specified in dollars or basis points, never in ticks
   or lots.  The notebooks hard-code tick counts that only mean something for
   the asset they were written for (`half_spread=80` is 30 bp on 2023 BTC and
   nonsense on XRP).  A bp knob is the same knob on every asset, which is what
   makes "a wider spread, appropriate to the asset" a single number here.
2. **One inventory convention.**  Every strategy skews against
   `norm_pos = position * mid / max_position_dollar`, which is -1 at a full
   short and +1 at a full long, and `skew_bp` is the fair-price shift at a full
   position.  The notebooks variously skew per lot, per notional unit, or
   multiplicatively on the half spread; those are the same quantity in
   different units, and converting them once here is what lets the master table
   put a skew column next to a skew column.

Where a notebook value is reproduced it is converted into these units and the
conversion is shown in `strategies.py`.

The four multipliers `spread_mult`, `skew_mult`, `alpha_mult` and `grid_mult`
are the handle C6's RL modification turns.  All four are 1.0 in every baseline,
so each baseline is a strict special case of its learned policy -- the same
property C2 built into the GLFT environment, now holding for all five.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from hftbacktest import BUY, BUY_EVENT, GTX, LIMIT, SELL
from hftbacktest.order import IOC
from numba import njit
from numba.typed import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "c1_glft_baseline"))

import glft  # noqa: E402

#: Strategy ids.  Integers rather than strings because the quoting core is a
#: single njit kernel and numba branches on an int far more happily than on a
#: unicode comparison.
GLFT, HFGRID, APT, BASIS, OBI = range(5)
STRATEGY_IDS = {"glft": GLFT, "hfgrid": HFGRID, "apt": APT, "basis": BASIS, "obi": OBI}
STRATEGY_NAMES = {v: k for k, v in STRATEGY_IDS.items()}

#: Indices into the persistent state vector carried between kernel calls, so
#: that the RL environment can stop the loop after every policy step and resume
#: it without the quoter losing its calibration windows.
(
    Z_T, Z_A, Z_K, Z_VOL, Z_MID, Z_PREV_MID, Z_SPREAD, Z_BIDQ, Z_ASKQ,
    Z_DEEPBIDQ, Z_DEEPASKQ, Z_HALF_PX, Z_SKEW_PX, Z_SIGNAL_PX, Z_FAIR_PX,
    Z_DATA_I, Z_REF_A, Z_REF_B, Z_OBI_SUM, Z_OBI_SUMSQ, Z_OBI_N, Z_OBI_RAW,
    Z_ALPHA, Z_ORDER_QTY, Z_NORM_POS, Z_TOX,
    # C8's risk controls, denoising and quoting ideas; all idle at defaults.
    Z_TOX_BUY, Z_TOX_SELL, Z_VAR_FAST, Z_VAR_SLOW, Z_FLOW_S, Z_FLOW_A,
    Z_FLOW_A_SLOW, Z_HALT_UNTIL, Z_HALT_SIDE, Z_PEAK_PNL, Z_LOSS_HALT_UNTIL,
    Z_SIG_S, Z_MOM, Z_BID_ANCHOR, Z_ASK_ANCHOR, Z_LAST_FLATTEN,
    Z_CAP_PX,
) = range(43)
STATE_SIZE = 43

#: Slots in the `params` vector, which carries the union of every strategy's
#: knobs.  A vector rather than a dozen keyword arguments because numba
#: specialises a kernel on its signature: five differently-shaped calls would
#: compile the quoter five times, and the quoter is most of the compile budget.
(
    P_STRAT, P_HALF_SPREAD_BP, P_SKEW_BP, P_ALPHA_BP, P_GRID_NUM,
    P_GRID_INTERVAL_BP, P_ORDER_QTY_DOLLAR, P_MAX_POSITION_DOLLAR,
    P_GAMMA, P_ADJ1, P_ADJ2, P_VOL_TO_HALF_SPREAD, P_BETA,
    P_OBI_DEPTH, P_OBI_WINDOW, P_ROI_LB_TICK, P_ROI_UB_TICK,
    P_VOL_SCALE, P_HAR_B0, P_HAR_B1, P_HAR_B2, P_HAR_B3,
    P_MARKOUT_KAPPA, P_MARKOUT_STEPS, P_MARKOUT_HALFLIFE,
    P_CB_MODE, P_CB_VOL, P_CB_FLOW, P_CB_FAST_HL, P_CB_COOLDOWN,
    P_LOSS_LIMIT_BP, P_LOSS_MODE, P_LOSS_COOLDOWN, P_BOOK_DOLLAR,
    P_SIGNAL_HL, P_ALPHA_CLIP, P_MOM_COEF, P_MOM_HL,
    P_MARKOUT_SIDED, P_REQUOTE_STEPS, P_TOX_SIZE, P_INV_FLATTEN,
    P_TOD0, P_TOD1, P_TOD2, P_TOD3, P_TOD4, P_TOD5,
    P_FAIR_MODE, P_FAIR_COEF,
    P_DAG_MODE, P_DAG_B, P_DAG_W_MKT, P_DAG_W_DIST, P_DAG_W_POS, P_DAG_W_TOX, P_DAG_THR,
    P_DAG_WIDEN_BP,
    P_CAP_REF, P_CAP_RESTING, P_LOSS_FLATTEN,
) = range(61)
PARAM_SIZE = 61

PARAM_NAMES = (
    "strat", "half_spread_bp", "skew_bp", "alpha_bp", "grid_num",
    "grid_interval_bp", "order_qty_dollar", "max_position_dollar",
    "gamma", "adj1", "adj2", "vol_to_half_spread", "beta",
    "obi_depth", "obi_window", "roi_lb_tick", "roi_ub_tick",
    "vol_scale", "har_b0", "har_b1", "har_b2", "har_b3",
    "markout_kappa", "markout_steps", "markout_halflife",
    "cb_mode", "cb_vol", "cb_flow", "cb_fast_hl", "cb_cooldown",
    "loss_limit_bp", "loss_mode", "loss_cooldown", "book_dollar",
    "signal_hl", "alpha_clip", "mom_coef", "mom_hl",
    "markout_sided", "requote_steps", "tox_size", "inv_flatten",
    "tod0", "tod1", "tod2", "tod3", "tod4", "tod5",
    "fair_mode", "fair_coef",
    "dag_mode", "dag_b", "dag_w_mkt", "dag_w_dist", "dag_w_pos", "dag_w_tox", "dag_thr",
    "dag_widen_bp",
    "cap_ref", "cap_resting", "loss_flatten",
)

#: C9's external per-step series, one row per quoting step: column 0 is the
#: fair-price shift (bp of the mid) `fair_mode` uses; columns 1 and 2 are the
#: market part of the DAgger student's danger score for the bid and the ask.
#: "No series" is one row of NaN: numba needs a fixed type, not `None`.
FAIR_COLS = 3
NO_FAIR = np.full((1, FAIR_COLS), np.nan, np.float64)

#: Half-life, in quoting steps, of the slow averages the circuit breaker
#: compares its fast ones against: thirty minutes at 100 ms.
CB_SLOW_HL = 18_000.0
#: No breaker decision before this many steps (ten minutes): the slow
#: averages need data before a ratio against them means anything.
CB_WARMUP = 6_000

#: Price levels each side that the deep-imbalance feature sums over, as in C2.
DEPTH_LEVELS = 5

#: What the quoter logs each iteration, for the master table's spread and
#: intensity columns.  Recorded in bp of price so the five are comparable.
out_dtype = np.dtype(
    [
        ("half_spread_bp", "f8"),
        ("skew_bp", "f8"),
        ("signal_bp", "f8"),
        ("volatility", "f8"),
        ("A", "f8"),
        ("k", "f8"),
        ("position", "f8"),
        ("mid", "f8"),
        # 0 quoting normally; 1 breaker, both sides; 2 breaker, one side;
        # 3 loss limit (reduce-only).
        ("paused", "f8"),
        # C9: the innermost bid and ask the rule would quote, before a
        # forecast or the DAgger student pulls or widens them (NaN when the
        # position cap stops a side), for the DAgger teacher to label.
        ("bid_px", "f8"),
        ("ask_px", "f8"),
        # C9: the per-side markout toxicity the DAgger student reads.
        ("tox_buy", "f8"),
        ("tox_sell", "f8"),
    ]
)


@njit
def _obi_alpha(depth, mid_price, tick_size, looking_depth, roi_lb_tick, roi_ub_tick):
    """Signed book imbalance over a band `looking_depth` wide either side.

    The notebook's own sum, unchanged: quantity resting between the touch and
    `mid * (1 +/- looking_depth)`, bid minus ask.  Raw, not yet standardised --
    the standardisation is running and lives in the caller, because recomputing
    `nanmean`/`nanstd` over an hour-long window at every 100 ms iteration is
    quadratic and the notebook only gets away with it at a 1 s interval.

    Read through `*_qty_at_tick` (C11), which every depth type has: inside
    the ROI it is the same array element, so results are unchanged.
    """
    sum_ask_qty = 0.0
    from_tick = max(depth.best_ask_tick, roi_lb_tick)
    upto_tick = min(int(np.floor(mid_price * (1 + looking_depth) / tick_size)), roi_ub_tick)
    for price_tick in range(from_tick, upto_tick):
        sum_ask_qty += depth.ask_qty_at_tick(price_tick)

    sum_bid_qty = 0.0
    from_tick = min(depth.best_bid_tick, roi_ub_tick)
    upto_tick = max(int(np.ceil(mid_price * (1 - looking_depth) / tick_size)), roi_lb_tick)
    for price_tick in range(from_tick, upto_tick, -1):
        sum_bid_qty += depth.bid_qty_at_tick(price_tick)

    return sum_bid_qty - sum_ask_qty


@njit
def quote_once(
    hbt, recorder, state, out, params, precomp, fair_ext, buf_arrival, buf_midchg, buf_flow,
    buf_obi, tmp, ticks, spread_mult, skew_mult, alpha_mult, grid_mult, interval_ns,
):
    """One quoting iteration of whichever strategy `params` selects.

    Returns 0 while the day still has data and 1 when it is exhausted.

    `params` carries every strategy's knobs in the slots named by
    `PARAM_NAMES`; `strategies.py` is what fills it in.
    """
    strat = int(params[P_STRAT])
    t = int(state[Z_T])
    tick_size = hbt.depth(0).tick_size
    lot_size = hbt.depth(0).lot_size
    window = glft.window_steps(interval_ns)
    recalibrate = glft.recalibrate_steps(interval_ns)

    if hbt.elapse(interval_ns) != 0:
        return 1

    mid_price_tick = state[Z_MID]

    # Arrival depth of aggressive trades -- what the intensity fit consumes --
    # and the signed volume C1 found to be the only predictor of its loss.
    signed_flow = 0.0
    if not np.isnan(mid_price_tick):
        depth_reached = -np.inf
        for last_trade in hbt.last_trades(0):
            trade_price_tick = last_trade.px / tick_size
            if last_trade.ev & BUY_EVENT == BUY_EVENT:
                depth_reached = np.nanmax([trade_price_tick - mid_price_tick, depth_reached])
                signed_flow += last_trade.qty
            else:
                depth_reached = np.nanmax([mid_price_tick - trade_price_tick, depth_reached])
                signed_flow -= last_trade.qty
        buf_arrival[t] = depth_reached
    buf_flow[t] = signed_flow

    hbt.clear_last_trades(0)
    hbt.clear_inactive_orders(0)

    depth = hbt.depth(0)
    position = hbt.position(0)
    orders = hbt.orders(0)

    best_bid_tick = depth.best_bid_tick
    best_ask_tick = depth.best_ask_tick
    two_sided = best_bid_tick != glft.INVALID_MIN and best_ask_tick != glft.INVALID_MAX

    state[Z_PREV_MID] = mid_price_tick
    mid_price_tick = (best_bid_tick + best_ask_tick) / 2.0 if two_sided else np.nan
    state[Z_MID] = mid_price_tick
    buf_midchg[t] = mid_price_tick - state[Z_PREV_MID]

    # C8: fast and slow exponential averages of squared mid changes and of
    # signed and absolute aggressive flow, for the circuit breaker. Averages,
    # not raw per-step values, so that one large print does not trip it.
    if params[P_CB_MODE] > 0.0:
        a_fast = 1.0 - 0.5 ** (1.0 / max(params[P_CB_FAST_HL], 1.0))
        a_slow = 1.0 - 0.5 ** (1.0 / CB_SLOW_HL)
        move = buf_midchg[t]
        if np.isfinite(move):
            state[Z_VAR_FAST] += a_fast * (move * move - state[Z_VAR_FAST])
            state[Z_VAR_SLOW] += a_slow * (move * move - state[Z_VAR_SLOW])
        state[Z_FLOW_S] += a_fast * (signed_flow - state[Z_FLOW_S])
        state[Z_FLOW_A] += a_fast * (abs(signed_flow) - state[Z_FLOW_A])
        state[Z_FLOW_A_SLOW] += a_slow * (abs(signed_flow) - state[Z_FLOW_A_SLOW])
        if t >= CB_WARMUP:
            vol_trip = (params[P_CB_VOL] > 0.0 and state[Z_VAR_SLOW] > 0.0
                        and state[Z_VAR_FAST] > params[P_CB_VOL] ** 2 * state[Z_VAR_SLOW])
            imbalance = state[Z_FLOW_S] / state[Z_FLOW_A] if state[Z_FLOW_A] > 0.0 else 0.0
            flow_trip = (params[P_CB_FLOW] > 0.0 and abs(imbalance) > params[P_CB_FLOW]
                         and state[Z_FLOW_A] > state[Z_FLOW_A_SLOW])
            if vol_trip or flow_trip:
                state[Z_HALT_UNTIL] = t + params[P_CB_COOLDOWN]
                # One-sided mode pulls the side the flow is hitting; a
                # volatility trip has no side and pulls both.
                state[Z_HALT_SIDE] = 0.0 if vol_trip else np.sign(imbalance)
    if params[P_MOM_COEF] != 0.0 and np.isfinite(buf_midchg[t]):
        a_mom = 1.0 - 0.5 ** (1.0 / max(params[P_MOM_HL], 1.0))
        state[Z_MOM] += a_mom * (buf_midchg[t] * tick_size - state[Z_MOM])

    bid_qty = np.nan
    ask_qty = np.nan
    if two_sided:
        # hftbacktest 2.4.4 wires `best_bid_qty` to the price for this depth
        # type; see C1's writeup.  Read the quantity by tick instead.
        bid_qty = depth.bid_qty_at_tick(best_bid_tick)
        ask_qty = depth.ask_qty_at_tick(best_ask_tick)
        state[Z_BIDQ] = bid_qty
        state[Z_ASKQ] = ask_qty
        state[Z_SPREAD] = best_ask_tick - best_bid_tick
        deep_bid = 0.0
        deep_ask = 0.0
        for level in range(DEPTH_LEVELS):
            deep_bid += depth.bid_qty_at_tick(best_bid_tick - level)
            deep_ask += depth.ask_qty_at_tick(best_ask_tick + level)
        state[Z_DEEPBIDQ] = deep_bid
        state[Z_DEEPASKQ] = deep_ask

    # A, k and sigma are refit on the same schedule for every strategy, even
    # the three that do not quote off them, because the master table reports
    # trade intensity for all five and it has to be the same measurement.
    if t % recalibrate == 0 and t >= window - 1:
        A_, k_ = glft.fit_intensity(buf_arrival[t + 1 - window : t + 1], tmp, ticks)
        if np.isfinite(A_):
            state[Z_A] = A_
            state[Z_K] = k_
        state[Z_VOL] = glft.realised_volatility(buf_midchg[t + 1 - window : t + 1], interval_ns)
        # C7: `vol_scale` multiplies the variance (C7 measured the estimator
        # above reading 1.5-1.7x low); a non-zero `har_b1` replaces it with a
        # HAR forecast from 1 s returns over the last minute, ten minutes and
        # hour, once an hour of data exists. Both default to C6's sigma.
        state[Z_VOL] *= np.sqrt(params[P_VOL_SCALE])
        per_second = int(1e9 / interval_ns)
        hour = 3600 * per_second
        if params[P_HAR_B1] != 0.0 and t >= hour and np.isfinite(mid_price_tick):
            tick_bp = 1e4 / mid_price_tick  # one tick in bp at this mid
            rv = np.zeros(3)
            for j, seconds in enumerate((60, 600, 3600)):
                total = 0.0
                for s in range(seconds):
                    end = t + 1 - s * per_second
                    move = np.nansum(buf_midchg[end - per_second : end])
                    total += move * move
                rv[j] = total / seconds * tick_bp * tick_bp
            if rv[0] > 0.0 and rv[1] > 0.0 and rv[2] > 0.0:
                var_bp = np.exp(params[P_HAR_B0] + params[P_HAR_B1] * np.log(rv[0])
                                + params[P_HAR_B2] * np.log(rv[1])
                                + params[P_HAR_B3] * np.log(rv[2]))
                state[Z_VOL] = np.sqrt(var_bp) / tick_bp

    mid_price = mid_price_tick * tick_size
    order_qty = max(round((params[P_ORDER_QTY_DOLLAR] / mid_price) / lot_size) * lot_size, lot_size)
    state[Z_ORDER_QTY] = order_qty
    norm_pos = position * mid_price / params[P_MAX_POSITION_DOLLAR]
    state[Z_NORM_POS] = norm_pos

    # ---------------------------------------------------------------- signal
    # Each strategy's own view of where the fair price sits relative to the
    # mid, in price units.  `alpha_mult` scales it; 1.0 is the published value
    # and 0.0 turns the strategy into a pure mid-price quoter, which is how C6
    # measures what each signal is actually worth.
    signal_px = 0.0
    if strat == HFGRID:
        if two_sided and bid_qty + ask_qty > 0.0:
            micro = (best_bid_tick * tick_size * ask_qty + best_ask_tick * tick_size * bid_qty) / (
                bid_qty + ask_qty
            )
            signal_px = micro - mid_price
    elif strat == OBI:
        raw = np.nan
        if two_sided:
            raw = _obi_alpha(
                depth, mid_price, tick_size, params[P_OBI_DEPTH],
                int(params[P_ROI_LB_TICK]), int(params[P_ROI_UB_TICK]),
            )
        # Running standardisation over a circular window: O(1) a step, where
        # the notebook's nanmean/nanstd over the window is O(window).  At the
        # notebook's 1 s interval that is 3,600 elements; at C6's 100 ms it
        # would be 36,000 at every one of 864,000 steps.
        obi_window = int(params[P_OBI_WINDOW])
        slot = t % obi_window
        if t >= obi_window and np.isfinite(buf_obi[slot]):
            state[Z_OBI_SUM] -= buf_obi[slot]
            state[Z_OBI_SUMSQ] -= buf_obi[slot] * buf_obi[slot]
            state[Z_OBI_N] -= 1.0
        buf_obi[slot] = raw
        if np.isfinite(raw):
            state[Z_OBI_SUM] += raw
            state[Z_OBI_SUMSQ] += raw * raw
            state[Z_OBI_N] += 1.0
        state[Z_OBI_RAW] = raw
        # Divide by the count of *finite* entries, not by the window
        # occupancy: the notebook uses `nanmean`/`nanstd`, and on a one-sided
        # book the imbalance is undefined rather than zero.  Counting the
        # holes as zeros biases both moments and moves the standardised alpha
        # by up to 0.28 sigma at a 5% hole rate -- `run_checks.py` measures it.
        n = state[Z_OBI_N]
        alpha = np.nan
        if n > 0.0:
            mean = state[Z_OBI_SUM] / n
            var = state[Z_OBI_SUMSQ] / n - mean * mean
            alpha = (raw - mean) / np.sqrt(var) if var > 0.0 else np.nan
        if params[P_ALPHA_CLIP] > 0.0 and np.isfinite(alpha):
            alpha = min(max(alpha, -params[P_ALPHA_CLIP]), params[P_ALPHA_CLIP])
        state[Z_ALPHA] = alpha
        if np.isfinite(alpha):
            signal_px = params[P_ALPHA_BP] * 1e-4 * mid_price * alpha
    elif strat == APT or strat == BASIS:
        # Both read a precomputed reference series, as their notebooks do:
        # column 0 is a local timestamp in ns, columns 1 and 2 are the two
        # numbers the strategy needs.  Walking it forward keeps the read
        # strictly causal -- the row used is the last one that had already
        # arrived, never the one straddling the current instant.
        data_i = int(state[Z_DATA_I])
        while data_i < len(precomp):
            if precomp[data_i, 0] > hbt.current_timestamp:
                if data_i > 0:
                    state[Z_REF_A] = precomp[data_i - 1, 1]
                    state[Z_REF_B] = precomp[data_i - 1, 2]
                break
            data_i += 1
        state[Z_DATA_I] = data_i
        if strat == APT:
            # fair = (1 + beta * reference return) * our own price 5 min ago.
            fair = (1.0 + params[P_BETA] * state[Z_REF_A]) * state[Z_REF_B]
        else:
            # fair = latest reference price + the rolling mean of the basis.
            fair = state[Z_REF_A] + state[Z_REF_B]
        if np.isfinite(fair):
            signal_px = fair - mid_price
    # C8 denoising: an exponential average of the strategy's signal, in
    # price units, `signal_hl` steps long; zero keeps the raw signal.
    if params[P_SIGNAL_HL] > 0.0:
        a_sig = 1.0 - 0.5 ** (1.0 / params[P_SIGNAL_HL])
        state[Z_SIG_S] += a_sig * (signal_px - state[Z_SIG_S])
        signal_px = state[Z_SIG_S]
    # C8: short-horizon momentum. C7 measured a positive lag-1
    # autocorrelation of 100 ms mid changes; this leans the fair price along
    # the recent drift, `mom_coef` times the average change over `mom_hl`.
    if params[P_MOM_COEF] != 0.0:
        signal_px += params[P_MOM_COEF] * state[Z_MOM] * params[P_MOM_HL]
    # C9: an external fair-price shift, one value per quoting step in bp of
    # the mid (a prediction of the mid's next move, or the future mid itself
    # for the oracle test), scaled by `fair_coef`. Mode 1 adds it to the
    # strategy's own signal; mode 2 replaces the signal with it. Modes 3 and 4
    # never move the centre: a positive value (the mid is expected to rise,
    # so the ask is the side about to be run over) widens the ask by it
    # (mode 3) or pulls the ask (mode 4); a negative one does so to the bid.
    shift_px = 0.0
    if (params[P_FAIR_MODE] != 0.0 and t < fair_ext.shape[0]
            and np.isfinite(fair_ext[t, 0])):
        shift_px = params[P_FAIR_COEF] * fair_ext[t, 0] * 1e-4 * mid_price
        if params[P_FAIR_MODE] == 2.0:
            signal_px = shift_px
        elif params[P_FAIR_MODE] == 1.0:
            signal_px += shift_px
    state[Z_SIGNAL_PX] = signal_px
    fair_px = mid_price + alpha_mult * signal_px
    state[Z_FAIR_PX] = fair_px

    # ----------------------------------------------------- spread and skew
    # `model_half_px` is whatever the strategy derives from market state; the
    # bp floor is the asset-independent widening knob.  GLFT and hfgrid have a
    # model, the three alpha strategies quote a fixed relative spread, so for
    # them the floor *is* the spread.
    model_half_px = 0.0
    if strat == GLFT:
        c1, c2 = glft.compute_coeff(params[P_GAMMA], params[P_GAMMA], 1.0, state[Z_A], state[Z_K])
        model_half_px = (c1 + 0.5 * c2 * state[Z_VOL]) * params[P_ADJ1] * tick_size
        # GLFT's skew is per unit of asset held; converting once here puts it
        # in the same unit as the other four (price shift at a full position).
        model_skew_px = c2 * state[Z_VOL] * params[P_ADJ2] * tick_size
        skew_px = model_skew_px * params[P_MAX_POSITION_DOLLAR] / mid_price
    else:
        skew_px = params[P_SKEW_BP] * 1e-4 * mid_price
    if strat == HFGRID:
        model_half_px = state[Z_VOL] * params[P_VOL_TO_HALF_SPREAD] * tick_size

    floor_half_px = params[P_HALF_SPREAD_BP] * 1e-4 * mid_price
    # C8: a time-of-day floor schedule, one multiplier per four-hour UTC
    # block; zero means 1.
    hour_block = int((hbt.current_timestamp // 3_600_000_000_000) % 24) // 4
    tod = params[P_TOD0 + hour_block]
    if tod > 0.0:
        floor_half_px *= tod
    half_px = spread_mult * max(model_half_px, floor_half_px)

    # C7's markout rule: each fill is marked `markout_steps` later against the
    # mid it was filled at, signed so that a positive number is a loss (the
    # price moved against the position the fill opened). An exponential
    # average over fills, `markout_halflife` fills long, is added to the half
    # spread times `markout_kappa`: quote wider for as long as recent fills
    # have been adversely selected. Zero kappa is C6's quoter exactly.
    # C8: `markout_sided` keeps one average per side (fills on our bids,
    # fills on our asks) and widens each side by its own; `tox_size` shrinks
    # the order size as toxicity rises instead of, or as well as, widening.
    half_bid_px = half_px
    half_ask_px = half_px
    if params[P_MARKOUT_KAPPA] > 0.0 or params[P_TOX_SIZE] > 0.0:
        lag = int(params[P_MARKOUT_STEPS])
        if t > lag:
            filled = out[t - lag].position - out[t - lag - 1].position
            then = out[t - lag].mid
            if filled != 0.0 and then > 0.0:
                loss_bp = -np.sign(filled) * (mid_price - then) / then * 1e4
                weight = abs(filled) / order_qty
                step = 1.0 - 0.5 ** (weight / params[P_MARKOUT_HALFLIFE])
                state[Z_TOX] += step * (loss_bp - state[Z_TOX])
                side = Z_TOX_BUY if filled > 0.0 else Z_TOX_SELL
                state[side] += step * (loss_bp - state[side])
        if params[P_MARKOUT_SIDED] > 0.0:
            half_bid_px += params[P_MARKOUT_KAPPA] * max(state[Z_TOX_BUY], 0.0) * 1e-4 * mid_price
            half_ask_px += params[P_MARKOUT_KAPPA] * max(state[Z_TOX_SELL], 0.0) * 1e-4 * mid_price
        else:
            half_px += params[P_MARKOUT_KAPPA] * max(state[Z_TOX], 0.0) * 1e-4 * mid_price
            half_bid_px = half_px
            half_ask_px = half_px
        if params[P_TOX_SIZE] > 0.0:
            shrink = 1.0 / (1.0 + params[P_TOX_SIZE] * max(state[Z_TOX], 0.0))
            order_qty = max(round(order_qty * shrink / lot_size) * lot_size, lot_size)
    if params[P_FAIR_MODE] == 3.0:
        if shift_px > 0.0:
            half_ask_px += shift_px
        else:
            half_bid_px -= shift_px
    if strat == HFGRID:
        # The notebook skews hfgrid multiplicatively on the half spread, so its
        # skew scales with volatility too; keep that, but express the result in
        # the same price units as everyone else's.
        skew_px = half_px * params[P_SKEW_BP]
    skew_px = skew_mult * skew_px
    state[Z_HALF_PX] = half_px
    state[Z_SKEW_PX] = skew_px

    reservation_px = fair_px - skew_px * norm_pos

    if two_sided and np.isfinite(reservation_px) and np.isfinite(half_px):
        bid_price = min(reservation_px - half_bid_px, best_bid_tick * tick_size)
        ask_price = max(reservation_px + half_ask_px, best_ask_tick * tick_size)
    else:
        bid_price = np.nan
        ask_price = np.nan

    grid_step = params[P_GRID_INTERVAL_BP] * 1e-4 * mid_price
    if grid_step <= 0.0:
        grid_step = half_px
    grid_interval = max(grid_mult * grid_step, tick_size)
    grid_interval = np.round(grid_interval / tick_size) * tick_size
    bid_price = np.floor(bid_price / grid_interval) * grid_interval
    ask_price = np.ceil(ask_price / grid_interval) * grid_interval

    # C8: requote hysteresis. Keep the previous quote while the new one is
    # within `requote_steps` grid steps of it, so the order keeps its queue
    # position instead of being cancelled and re-queued for a one-step move.
    # In grid steps, not ticks: quotes already move only in whole steps.
    if params[P_REQUOTE_STEPS] > 0.0:
        band = (params[P_REQUOTE_STEPS] + 0.5) * grid_interval
        if (np.isfinite(bid_price) and np.isfinite(state[Z_BID_ANCHOR])
                and abs(bid_price - state[Z_BID_ANCHOR]) < band
                and state[Z_BID_ANCHOR] <= best_bid_tick * tick_size):
            bid_price = state[Z_BID_ANCHOR]
        if (np.isfinite(ask_price) and np.isfinite(state[Z_ASK_ANCHOR])
                and abs(ask_price - state[Z_ASK_ANCHOR]) < band
                and state[Z_ASK_ANCHOR] >= best_ask_tick * tick_size):
            ask_price = state[Z_ASK_ANCHOR]
        state[Z_BID_ANCHOR] = bid_price
        state[Z_ASK_ANCHOR] = ask_price

    # C9: the quotes as the rule would post them, before the forecast (mode
    # 4) or the DAgger student act on them -- the states the teacher labels.
    out[t].bid_px = bid_price if position * mid_price < params[P_MAX_POSITION_DOLLAR] else np.nan
    out[t].ask_px = ask_price if position * mid_price > -params[P_MAX_POSITION_DOLLAR] else np.nan
    out[t].tox_buy = state[Z_TOX_BUY]
    out[t].tox_sell = state[Z_TOX_SELL]

    # C9: pull the side a forecast says is about to be run over (mode 4).
    if params[P_FAIR_MODE] == 4.0 and shift_px != 0.0:
        if shift_px > 0.0:
            ask_price = np.nan
        else:
            bid_price = np.nan

    # C9: the DAgger student. Per side, a danger score: the market part
    # (precomputed from market features, `fair_ext` columns 1 and 2) plus
    # the student's own state -- how far its quote sits from the mid, the
    # inventory a fill would add to, and its recent markout toxicity -- with
    # weights fitted by `dagger.py`. Above `dag_thr` (a probability) the side
    # is pulled (`dag_widen_bp` 0) or widened by `dag_widen_bp`.
    if params[P_DAG_MODE] > 0.0 and t < fair_ext.shape[0] and two_sided:
        for side in range(2):
            px = bid_price if side == 0 else ask_price
            mkt = fair_ext[t, 1 + side]
            if not (np.isfinite(px) and np.isfinite(mkt)):
                continue
            dist_bp = abs(px - mid_price) / mid_price * 1e4
            # Inventory a fill on this side adds to: long for a bid fill.
            pos = norm_pos if side == 0 else -norm_pos
            tox = state[Z_TOX_BUY] if side == 0 else state[Z_TOX_SELL]
            z = (params[P_DAG_B] + params[P_DAG_W_MKT] * mkt + params[P_DAG_W_DIST] * dist_bp
                 + params[P_DAG_W_POS] * pos + params[P_DAG_W_TOX] * tox)
            if 1.0 / (1.0 + np.exp(-z)) > params[P_DAG_THR]:
                if params[P_DAG_WIDEN_BP] <= 0.0:
                    px = np.nan
                elif side == 0:
                    px = np.floor((px - params[P_DAG_WIDEN_BP] * 1e-4 * mid_price) / grid_interval) * grid_interval
                else:
                    px = np.ceil((px + params[P_DAG_WIDEN_BP] * 1e-4 * mid_price) / grid_interval) * grid_interval
            if side == 0:
                bid_price = px
            else:
                ask_price = px

    # C8: circuit breaker and loss limit. Both only ever remove quotes.
    paused = 0.0
    if params[P_CB_MODE] > 0.0 and t < state[Z_HALT_UNTIL]:
        side = state[Z_HALT_SIDE]
        if params[P_CB_MODE] == 1.0 or side == 0.0:
            bid_price = np.nan
            ask_price = np.nan
            paused = 1.0
        elif side > 0.0:
            ask_price = np.nan  # buyers are lifting offers: stop selling to them
            paused = 2.0
        else:
            bid_price = np.nan
            paused = 2.0
    if params[P_LOSS_LIMIT_BP] > 0.0:
        values = hbt.state_values(0)
        pnl_bp = (values.balance + position * mid_price - values.fee) / params[P_BOOK_DOLLAR] * 1e4
        if params[P_LOSS_MODE] == 1.0:
            # A hard daily stop: below the limit, reduce-only until the end.
            if pnl_bp < -params[P_LOSS_LIMIT_BP]:
                state[Z_LOSS_HALT_UNTIL] = np.inf
        else:
            # A trailing stop from the session's peak, for `loss_cooldown`
            # steps, after which the peak restarts from where PnL stands.
            if t >= state[Z_LOSS_HALT_UNTIL] and state[Z_LOSS_HALT_UNTIL] > 0.0:
                state[Z_PEAK_PNL] = pnl_bp
                state[Z_LOSS_HALT_UNTIL] = 0.0
            state[Z_PEAK_PNL] = max(state[Z_PEAK_PNL], pnl_bp)
            if pnl_bp < state[Z_PEAK_PNL] - params[P_LOSS_LIMIT_BP] and state[Z_LOSS_HALT_UNTIL] == 0.0:
                state[Z_LOSS_HALT_UNTIL] = t + params[P_LOSS_COOLDOWN]
        if t < state[Z_LOSS_HALT_UNTIL]:
            if position >= 0.0:
                bid_price = np.nan
            if position <= 0.0:
                ask_price = np.nan
            paused = 3.0

    # ------------------------------------------------------------- quoting
    grid_num = int(params[P_GRID_NUM])
    # C10: the position cap. By default it is notional at the current mid,
    # checked against the position only, which in a crash lets the quoter
    # keep buying as the price falls and lets a whole 20-level grid fill past
    # the cap at once. `cap_ref` 1 values the cap at the session's first mid
    # (a cap in units); `cap_resting` 1 counts resting orders against it, so
    # a side posts only the levels the remaining room allows.
    if np.isnan(state[Z_CAP_PX]) or state[Z_CAP_PX] == 0.0:
        state[Z_CAP_PX] = mid_price
    cap_px = state[Z_CAP_PX] if params[P_CAP_REF] > 0.0 else mid_price
    notional = position * cap_px
    bid_levels = grid_num
    ask_levels = grid_num
    if params[P_CAP_RESTING] > 0.0:
        level_notional = order_qty * cap_px
        bid_levels = min(grid_num, int((params[P_MAX_POSITION_DOLLAR] - notional) // level_notional))
        ask_levels = min(grid_num, int((params[P_MAX_POSITION_DOLLAR] + notional) // level_notional))
    new_bid_orders = Dict.empty(np.uint64, np.float64)
    if notional < params[P_MAX_POSITION_DOLLAR] and np.isfinite(bid_price):
        for _ in range(bid_levels):
            new_bid_orders[np.uint64(round(bid_price / tick_size))] = bid_price
            bid_price -= grid_interval

    new_ask_orders = Dict.empty(np.uint64, np.float64)
    if notional > -params[P_MAX_POSITION_DOLLAR] and np.isfinite(ask_price):
        for _ in range(ask_levels):
            new_ask_orders[np.uint64(round(ask_price / tick_size))] = ask_price
            ask_price += grid_interval

    order_values = orders.values()
    while order_values.has_next():
        order = order_values.get()
        if order.cancellable:
            if (order.side == BUY and order.order_id not in new_bid_orders) or (
                order.side == SELL and order.order_id not in new_ask_orders
            ):
                hbt.cancel(0, order.order_id, False)

    for order_id, order_price in new_bid_orders.items():
        if order_id not in orders:
            hbt.submit_buy_order(0, order_id, order_price, order_qty, GTX, LIMIT, False)

    for order_id, order_price in new_ask_orders.items():
        if order_id not in orders:
            hbt.submit_sell_order(0, order_id, order_price, order_qty, GTX, LIMIT, False)

    # C8: cut inventory past `inv_flatten` of the cap back to half of it
    # with an immediate-or-cancel order at the opposite touch, paying the
    # taker fee, at most once a second. Submitted after the quote refresh so
    # the cancel pass above cannot catch it in flight.
    # C10: `loss_flatten` 1 makes the loss limit a stop: while it halts
    # quoting, the whole position goes out the same way, instead of waiting
    # for the passive reduce-only quotes to fill in a market running away.
    loss_exit = params[P_LOSS_FLATTEN] > 0.0 and paused == 3.0 and position != 0.0
    if (two_sided and t - state[Z_LAST_FLATTEN] >= 10
            and (loss_exit or (params[P_INV_FLATTEN] > 0.0 and abs(norm_pos) > params[P_INV_FLATTEN]))):
        if loss_exit:
            excess = abs(position)
        else:
            excess = (abs(norm_pos) - 0.5 * params[P_INV_FLATTEN]) * params[P_MAX_POSITION_DOLLAR] / mid_price
        qty = max(round(excess / lot_size) * lot_size, lot_size)
        flatten_id = np.uint64(1 << 40) + np.uint64(t)
        if position > 0.0:
            hbt.submit_sell_order(0, flatten_id, best_bid_tick * tick_size, qty, IOC, LIMIT, False)
        else:
            hbt.submit_buy_order(0, flatten_id, best_ask_tick * tick_size, qty, IOC, LIMIT, False)
        state[Z_LAST_FLATTEN] = t

    out[t].half_spread_bp = half_px / mid_price * 1e4
    out[t].skew_bp = skew_px / mid_price * 1e4
    out[t].signal_bp = alpha_mult * signal_px / mid_price * 1e4
    out[t].volatility = state[Z_VOL]
    out[t].A = state[Z_A]
    out[t].k = state[Z_K]
    out[t].position = position
    out[t].mid = mid_price
    out[t].paused = paused

    state[Z_T] = t + 1
    return 0


@njit
def advance(
    hbt, recorder, state, out, params, precomp, fair_ext, buf_arrival, buf_midchg, buf_flow,
    buf_obi, tmp, ticks, spread_mult, skew_mult, alpha_mult, grid_mult, interval_ns,
    record_every, n_inner,
):
    """`n_inner` quoting iterations under one set of multipliers."""
    for _ in range(n_inner):
        # numba does not bounds-check, so running off the end of `out` would
        # corrupt the heap rather than raise.  `buffer_steps` sizes it for a
        # 25-hour day against a 24-hour session, so this should never fire;
        # it exists because the failure mode if it ever did is silent.
        if int(state[Z_T]) >= len(out):
            return 1
        if quote_once(
            hbt, recorder, state, out, params, precomp, fair_ext, buf_arrival, buf_midchg,
            buf_flow, buf_obi, tmp, ticks, spread_mult, skew_mult, alpha_mult,
            grid_mult, interval_ns,
        ) != 0:
            return 1
        if int(state[Z_T]) % record_every == 0:
            recorder.record(hbt)
    return 0


@njit
def run_day(
    hbt, recorder, params, precomp, fair_ext, interval_ns, record_every,
    spread_mult, skew_mult, alpha_mult, grid_mult,
):
    """Replay a whole session at fixed multipliers: C6's baseline runner."""
    n = glft.buffer_steps(interval_ns)
    state = np.zeros(STATE_SIZE, np.float64)
    state[Z_A] = np.nan
    state[Z_K] = np.nan
    state[Z_VOL] = np.nan
    state[Z_MID] = np.nan
    buf_arrival = np.full(n, np.nan, np.float64)
    buf_midchg = np.full(n, np.nan, np.float64)
    buf_flow = np.zeros(n, np.float64)
    buf_obi = np.full(max(int(params[P_OBI_WINDOW]), 1), np.nan, np.float64)
    out = np.zeros(n, out_dtype)
    tmp = np.zeros(glft.MAX_BUCKET, np.float64)
    ticks = np.arange(len(tmp)) + 0.5

    while True:
        if advance(
            hbt, recorder, state, out, params, precomp, fair_ext, buf_arrival, buf_midchg,
            buf_flow, buf_obi, tmp, ticks, spread_mult, skew_mult, alpha_mult,
            grid_mult, interval_ns, record_every, 1000,
        ) != 0:
            break
        if int(state[Z_T]) + 1000 >= n:
            break
    return out[: int(state[Z_T])]

