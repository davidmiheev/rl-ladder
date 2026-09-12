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
    Z_ALPHA, Z_ORDER_QTY, Z_NORM_POS,
) = range(25)
STATE_SIZE = 25

#: Slots in the `params` vector, which carries the union of every strategy's
#: knobs.  A vector rather than a dozen keyword arguments because numba
#: specialises a kernel on its signature: five differently-shaped calls would
#: compile the quoter five times, and the quoter is most of the compile budget.
(
    P_STRAT, P_HALF_SPREAD_BP, P_SKEW_BP, P_ALPHA_BP, P_GRID_NUM,
    P_GRID_INTERVAL_BP, P_ORDER_QTY_DOLLAR, P_MAX_POSITION_DOLLAR,
    P_GAMMA, P_ADJ1, P_ADJ2, P_VOL_TO_HALF_SPREAD, P_BETA,
    P_OBI_DEPTH, P_OBI_WINDOW, P_ROI_LB_TICK, P_ROI_UB_TICK,
) = range(17)
PARAM_SIZE = 17

PARAM_NAMES = (
    "strat", "half_spread_bp", "skew_bp", "alpha_bp", "grid_num",
    "grid_interval_bp", "order_qty_dollar", "max_position_dollar",
    "gamma", "adj1", "adj2", "vol_to_half_spread", "beta",
    "obi_depth", "obi_window", "roi_lb_tick", "roi_ub_tick",
)

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
    """
    sum_ask_qty = 0.0
    from_tick = max(depth.best_ask_tick, roi_lb_tick)
    upto_tick = min(int(np.floor(mid_price * (1 + looking_depth) / tick_size)), roi_ub_tick)
    for price_tick in range(from_tick, upto_tick):
        sum_ask_qty += depth.ask_depth[price_tick - roi_lb_tick]

    sum_bid_qty = 0.0
    from_tick = min(depth.best_bid_tick, roi_ub_tick)
    upto_tick = max(int(np.ceil(mid_price * (1 - looking_depth) / tick_size)), roi_lb_tick)
    for price_tick in range(from_tick, upto_tick, -1):
        sum_bid_qty += depth.bid_depth[price_tick - roi_lb_tick]

    return sum_bid_qty - sum_ask_qty


@njit
def quote_once(
    hbt, recorder, state, out, params, precomp, buf_arrival, buf_midchg, buf_flow,
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
    half_px = spread_mult * max(model_half_px, floor_half_px)
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
        bid_price = min(reservation_px - half_px, best_bid_tick * tick_size)
        ask_price = max(reservation_px + half_px, best_ask_tick * tick_size)
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

    # ------------------------------------------------------------- quoting
    grid_num = int(params[P_GRID_NUM])
    notional = position * mid_price
    new_bid_orders = Dict.empty(np.uint64, np.float64)
    if notional < params[P_MAX_POSITION_DOLLAR] and np.isfinite(bid_price):
        for _ in range(grid_num):
            new_bid_orders[np.uint64(round(bid_price / tick_size))] = bid_price
            bid_price -= grid_interval

    new_ask_orders = Dict.empty(np.uint64, np.float64)
    if notional > -params[P_MAX_POSITION_DOLLAR] and np.isfinite(ask_price):
        for _ in range(grid_num):
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

    out[t].half_spread_bp = half_px / mid_price * 1e4
    out[t].skew_bp = skew_px / mid_price * 1e4
    out[t].signal_bp = alpha_mult * signal_px / mid_price * 1e4
    out[t].volatility = state[Z_VOL]
    out[t].A = state[Z_A]
    out[t].k = state[Z_K]
    out[t].position = position
    out[t].mid = mid_price

    state[Z_T] = t + 1
    return 0


@njit
def advance(
    hbt, recorder, state, out, params, precomp, buf_arrival, buf_midchg, buf_flow,
    buf_obi, tmp, ticks, spread_mult, skew_mult, alpha_mult, grid_mult, interval_ns,
    record_every, n_inner,
):
    """`n_inner` quoting iterations under one set of multipliers."""
    for _ in range(n_inner):
        if quote_once(
            hbt, recorder, state, out, params, precomp, buf_arrival, buf_midchg,
            buf_flow, buf_obi, tmp, ticks, spread_mult, skew_mult, alpha_mult,
            grid_mult, interval_ns,
        ) != 0:
            return 1
        if int(state[Z_T]) % record_every == 0:
            recorder.record(hbt)
    return 0


@njit
def run_day(
    hbt, recorder, params, precomp, interval_ns, record_every,
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
            hbt, recorder, state, out, params, precomp, buf_arrival, buf_midchg,
            buf_flow, buf_obi, tmp, ticks, spread_mult, skew_mult, alpha_mult,
            grid_mult, interval_ns, record_every, 1000,
        ) != 0:
            break
        if int(state[Z_T]) + 1000 >= n:
            break
    return out[: int(state[Z_T])]

