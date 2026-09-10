"""Guéant-Lehalle-Fernandez-Tapia market making: calibration and quoting.

The functions here are deliberately faithful transcriptions of hftbacktest's
official notebook

    examples/GLFT Market Making Model and Grid Trading.ipynb

so that C1 reproduces a published baseline before Part C changes anything.
Where the notebook has a known quirk it is reproduced and flagged in a comment
rather than silently fixed -- a corrected baseline is not the published
baseline, and the whole point of C1 is to have a benchmark someone else can
check.

Model (equations 4.6/4.7 of Guéant, Lehalle & Fernandez-Tapia,
https://huggingface.co/papers/1605.01862), with Delta = 1 and xi = gamma:

    delta_b*(q) = c1 + (Delta/2) sigma c2 + q sigma c2
    delta_a*(q) = c1 + (Delta/2) sigma c2 - q sigma c2

    c1 = 1/(xi Delta) log(1 + xi Delta / k)
    c2 = sqrt( gamma / (2 A Delta k) * (1 + xi Delta / k)^(k/(xi Delta) + 1) )

so the half spread is c1 + (Delta/2) sigma c2 and the inventory skew per unit
of position is sigma c2.  A and k come from fitting the order-arrival intensity
lambda(delta) = A exp(-k delta); sigma is the mid-price volatility in ticks per
sqrt(second).
"""

import numpy as np
from hftbacktest import BUY, BUY_EVENT, FILLED, GTX, LIMIT, SELL
from numba import njit
from numba.typed import Dict

#: The reference notebook's reaction interval: the quoter wakes, re-prices and
#: re-posts every 100 ms.  Every quoter below takes it as an argument, because
#: C1 finds that on 2025 ETH this single number decides whether the strategy is
#: viable at all -- see `run_interval_study.py`.
INTERVAL_NS = 100_000_000
#: Calibration window, in seconds.  Ten minutes, as in the notebook.
WINDOW_SECONDS = 600.0
#: How often A, k and sigma are refit, in seconds.
RECALIBRATE_SECONDS = 5.0
#: Deepest bucket the intensity histogram tracks, in half-ticks from the mid.
MAX_BUCKET = 500
#: Depth range (in buckets) the shallow refit uses.  The parametric form fits
#: badly far from the mid, and quotes never go there anyway.
SHALLOW = 70

#: What `best_bid_tick` / `best_ask_tick` return when that side of the book is
#: empty.  Our day files begin mid-stream and the book is only rebuilt when the
#: feed's first snapshot arrives ~450 ms in, so the first few steps have no
#: two-sided book at all.  The reference notebook never sees this because it is
#: handed an external end-of-previous-day snapshot; we have to skip those steps
#: explicitly, and skipping them matters -- an unguarded mid-price jumps from
#: (INVALID_MIN + INVALID_MAX)/2 to the real price, which by itself puts the
#: first window's volatility three orders of magnitude too high.
INVALID_MIN = -9223372036854775808
INVALID_MAX = 9223372036854775807

out_dtype = np.dtype(
    [
        ("half_spread_tick", "f8"),
        ("skew", "f8"),
        ("volatility", "f8"),
        ("A", "f8"),
        ("k", "f8"),
    ]
)


@njit
def measure_trading_intensity(order_arrival_depth, out):
    """Histogram of how many market trades would have reached each quote depth.

    `out[i]` counts time-steps in which a market order swept at least as deep as
    bucket `i`, i.e. an empirical (unnormalised) `lambda(delta_i)`.

    Note: the bucket index is `round(depth / .5) - 1`, so buckets are half-ticks,
    not ticks -- an off-by-a-factor-of-two in the source notebook, kept here on
    purpose so that our A and k are comparable with the published ones.  It
    rescales k by 2 and leaves the *shape* of the fit unchanged.
    """
    max_tick = 0
    for depth in order_arrival_depth:
        if not np.isfinite(depth):
            continue

        # Sets the tick index to 0 for the nearest possible best price
        # as the order arrival depth in ticks is measured from the mid-price.
        tick = round(depth / 0.5) - 1

        # In a fast-moving market, buy trades can occur below the mid-price (and
        # vice versa for sell trades) since the mid-price is measured in a
        # previous time-step; however, to simplify the problem, we exclude those.
        if tick < 0 or tick >= len(out):
            continue

        # All of our possible quotes within the order arrival depth,
        # excluding those at the same price, are considered executed.
        out[:tick] += 1

        max_tick = max(max_tick, tick)
    return out[:max_tick]


@njit
def linear_regression(x, y):
    """Ordinary least squares slope and intercept, closed form."""
    sx = np.sum(x)
    sy = np.sum(y)
    sx2 = np.sum(x**2)
    sxy = np.sum(x * y)
    w = len(x)
    slope = (w * sxy - sx * sy) / (w * sx2 - sx**2)
    intercept = (sy - slope * sx) / w
    return slope, intercept


@njit
def compute_coeff(xi, gamma, delta, A, k):
    """The GLFT coefficients c1 (spread floor) and c2 (volatility loading)."""
    inv_k = np.divide(1, k)
    c1 = 1 / (xi * delta) * np.log(1 + xi * delta * inv_k)
    c2 = np.sqrt(
        np.divide(gamma, 2 * A * delta * k) * ((1 + xi * delta * inv_k) ** (k / (xi * delta) + 1))
    )
    return c1, c2


@njit
def measure_market(hbt, interval_ns):
    """Replay a day and record what calibration needs, without trading.

    Returns four series, all sampled every 100 ms:

    - `arrival_depth[t]`: the deepest a market order reached during step `t`,
      in ticks from the mid-price as of the *previous* step.  Buy trades are
      measured above the mid, sell trades below.
    - `mid_price_chg[t]`: the change in the mid-price in ticks over step `t`.
    - `mid_price[t]`: the mid-price in ticks.
    - `spread[t]`: the touch spread in ticks, which says whether the asset is in
      the small-tick regime GLFT's continuous derivation assumes.

    Steps without a two-sided book are `nan` throughout.
    """
    tick_size = hbt.depth(0).tick_size
    arrival_depth = np.full(buffer_steps(interval_ns), np.nan, np.float64)
    mid_price_chg = np.full(buffer_steps(interval_ns), np.nan, np.float64)
    mid_price = np.full(buffer_steps(interval_ns), np.nan, np.float64)
    spread = np.full(buffer_steps(interval_ns), np.nan, np.float64)

    t = 0
    prev_mid_price_tick = np.nan
    mid_price_tick = np.nan

    while hbt.elapse(interval_ns) == 0:
        # Records market order's arrival depth from the mid-price.
        if not np.isnan(mid_price_tick):
            depth = -np.inf
            for last_trade in hbt.last_trades(0):
                trade_price_tick = last_trade.px / tick_size

                if last_trade.ev & BUY_EVENT == BUY_EVENT:
                    depth = np.nanmax([trade_price_tick - mid_price_tick, depth])
                else:
                    depth = np.nanmax([mid_price_tick - trade_price_tick, depth])
            arrival_depth[t] = depth

        hbt.clear_last_trades(0)

        depth = hbt.depth(0)
        best_bid_tick = depth.best_bid_tick
        best_ask_tick = depth.best_ask_tick

        prev_mid_price_tick = mid_price_tick
        if best_bid_tick == INVALID_MIN or best_ask_tick == INVALID_MAX:
            mid_price_tick = np.nan
        else:
            mid_price_tick = (best_bid_tick + best_ask_tick) / 2.0
            spread[t] = best_ask_tick - best_bid_tick

        # Records the mid-price change for volatility calculation.
        mid_price_chg[t] = mid_price_tick - prev_mid_price_tick
        mid_price[t] = mid_price_tick

        t += 1
        if t >= len(arrival_depth):
            raise Exception
    return arrival_depth[:t], mid_price_chg[:t], mid_price[:t], spread[:t]


@njit
def window_steps(interval_ns):
    """Steps in one calibration window at this reaction interval."""
    return int(WINDOW_SECONDS * 1e9 / interval_ns)


@njit
def recalibrate_steps(interval_ns):
    return max(1, int(RECALIBRATE_SECONDS * 1e9 / interval_ns))


@njit
def buffer_steps(interval_ns):
    """Enough room for a 25-hour day at this interval."""
    return int(90_000 * 1e9 / interval_ns)


@njit
def fit_intensity(arrival_depth, tmp, ticks):
    """Fit `lambda(delta) = A exp(-k delta)` on one window of arrival depths.

    Taking logs turns it into a straight line, `log lambda = log A - k delta`,
    so the fit is one least-squares regression.  Returns `(A, k)`, or
    `(nan, nan)` when the window has too few distinct depths to regress on.
    """
    tmp[:] = 0
    lambda_ = measure_trading_intensity(arrival_depth, tmp)
    if len(lambda_) <= 2:
        return np.nan, np.nan
    # The window is WINDOW_SECONDS long, so divide the counts to get per second.
    lambda_ = lambda_[:SHALLOW] / WINDOW_SECONDS
    x = ticks[: len(lambda_)]
    y = np.log(lambda_)
    k_, logA = linear_regression(x, y)
    return np.exp(logA), -k_


@njit
def realised_volatility(mid_price_chg, interval_ns):
    """Mid-price volatility in ticks per sqrt(second).

    Samples are `interval_ns` apart, so the per-step standard deviation scales
    up by sqrt(steps per second) to become a per-second one.
    """
    return np.nanstd(mid_price_chg) * np.sqrt(1e9 / interval_ns)


@njit
def glft_market_maker(
    hbt, recorder, gamma, delta, adj1, adj2, order_qty, max_position, interval_ns, record_every
):
    """The calibrated GLFT quoter: one bid and one ask, recalibrated as it goes.

    Every 100 ms it re-derives its quotes; every 5 s it refits A, k and sigma on
    the trailing 10-minute window.  `adj1` and `adj2` scale the half spread and
    the skew respectively; with both at 1 this is the model verbatim.
    """
    tick_size = hbt.depth(0).tick_size
    arrival_depth = np.full(buffer_steps(interval_ns), np.nan, np.float64)
    mid_price_chg = np.full(buffer_steps(interval_ns), np.nan, np.float64)
    out = np.zeros(buffer_steps(interval_ns), out_dtype)

    t = 0
    prev_mid_price_tick = np.nan
    mid_price_tick = np.nan

    tmp = np.zeros(MAX_BUCKET, np.float64)
    ticks = np.arange(len(tmp)) + 0.5
    window = window_steps(interval_ns)
    recalibrate = recalibrate_steps(interval_ns)

    A = np.nan
    k = np.nan
    volatility = np.nan

    while hbt.elapse(interval_ns) == 0:
        # Records market order's arrival depth from the mid-price.
        if not np.isnan(mid_price_tick):
            depth = -np.inf
            for last_trade in hbt.last_trades(0):
                trade_price_tick = last_trade.px / tick_size

                if last_trade.ev & BUY_EVENT == BUY_EVENT:
                    depth = np.nanmax([trade_price_tick - mid_price_tick, depth])
                else:
                    depth = np.nanmax([mid_price_tick - trade_price_tick, depth])
            arrival_depth[t] = depth

        hbt.clear_last_trades(0)
        hbt.clear_inactive_orders(0)

        depth = hbt.depth(0)
        position = hbt.position(0)
        orders = hbt.orders(0)

        best_bid_tick = depth.best_bid_tick
        best_ask_tick = depth.best_ask_tick

        prev_mid_price_tick = mid_price_tick
        two_sided = best_bid_tick != INVALID_MIN and best_ask_tick != INVALID_MAX
        mid_price_tick = (best_bid_tick + best_ask_tick) / 2.0 if two_sided else np.nan

        # Records the mid-price change for volatility calculation.
        mid_price_chg[t] = mid_price_tick - prev_mid_price_tick

        # Refits A, k and the volatility every 5 s on a 10-minute window.
        if t % recalibrate == 0 and t >= window - 1:
            A_, k_ = fit_intensity(arrival_depth[t + 1 - window : t + 1], tmp, ticks)
            if np.isfinite(A_):
                A = A_
                k = k_
            volatility = realised_volatility(mid_price_chg[t + 1 - window : t + 1], interval_ns)

        c1, c2 = compute_coeff(gamma, gamma, delta, A, k)

        half_spread_tick = (c1 + delta / 2 * c2 * volatility) * adj1
        skew = c2 * volatility * adj2

        reservation_price_tick = mid_price_tick - skew * position

        if two_sided:
            bid_price_tick = np.minimum(np.round(reservation_price_tick - half_spread_tick), best_bid_tick)
            ask_price_tick = np.maximum(np.round(reservation_price_tick + half_spread_tick), best_ask_tick)
        else:
            bid_price_tick = np.nan
            ask_price_tick = np.nan

        bid_price = bid_price_tick * tick_size
        ask_price = ask_price_tick * tick_size

        # Cancels any working order that is not at the new bid/ask.
        order_values = orders.values()
        while order_values.has_next():
            order = order_values.get()
            if order.cancellable:
                if (order.side == BUY and order.price_tick != bid_price_tick) or (
                    order.side == SELL and order.price_tick != ask_price_tick
                ):
                    hbt.cancel(0, order.order_id, False)

        # Quotes only while inside the position limit, and only if nothing is
        # already working at that price.
        if position < max_position and np.isfinite(bid_price):
            bid_price_as_order_id = round(bid_price / tick_size)
            if bid_price_as_order_id not in orders:
                hbt.submit_buy_order(0, bid_price_as_order_id, bid_price, order_qty, GTX, LIMIT, False)
        if position > -max_position and np.isfinite(ask_price):
            ask_price_as_order_id = round(ask_price / tick_size)
            if ask_price_as_order_id not in orders:
                hbt.submit_sell_order(0, ask_price_as_order_id, ask_price, order_qty, GTX, LIMIT, False)

        out[t].half_spread_tick = half_spread_tick
        out[t].skew = skew
        out[t].volatility = volatility
        out[t].A = A
        out[t].k = k

        t += 1
        if t >= len(arrival_depth) or t >= len(mid_price_chg) or t >= len(out):
            raise Exception

        if t % record_every == 0:
            recorder.record(hbt)
    return out[:t]


@njit
def gridtrading_glft_mm(
    hbt, recorder, gamma, delta, adj1, adj2, order_qty, max_position, grid_num,
    interval_ns, record_every,
):
    """GLFT quotes turned into a grid, as in the second half of the notebook.

    Identical calibration and identical bid/ask; the difference is that instead
    of one order a side it lays `grid_num` orders spaced by `half_spread_tick`
    and aligned to that grid, so an order that is already correctly placed
    survives a requote and keeps its queue position.  On a venue whose touch
    spread is one tick almost all the time, fills are won in the queue rather
    than by the price, so this is not a cosmetic change.
    """
    tick_size = hbt.depth(0).tick_size
    arrival_depth = np.full(buffer_steps(interval_ns), np.nan, np.float64)
    mid_price_chg = np.full(buffer_steps(interval_ns), np.nan, np.float64)
    out = np.zeros(buffer_steps(interval_ns), out_dtype)

    t = 0
    prev_mid_price_tick = np.nan
    mid_price_tick = np.nan

    tmp = np.zeros(MAX_BUCKET, np.float64)
    ticks = np.arange(len(tmp)) + 0.5
    window = window_steps(interval_ns)
    recalibrate = recalibrate_steps(interval_ns)

    A = np.nan
    k = np.nan
    volatility = np.nan

    while hbt.elapse(interval_ns) == 0:
        if not np.isnan(mid_price_tick):
            depth = -np.inf
            for last_trade in hbt.last_trades(0):
                trade_price_tick = last_trade.px / tick_size

                if last_trade.ev & BUY_EVENT == BUY_EVENT:
                    depth = np.nanmax([trade_price_tick - mid_price_tick, depth])
                else:
                    depth = np.nanmax([mid_price_tick - trade_price_tick, depth])
            arrival_depth[t] = depth

        hbt.clear_last_trades(0)
        hbt.clear_inactive_orders(0)

        depth = hbt.depth(0)
        position = hbt.position(0)
        orders = hbt.orders(0)

        best_bid_tick = depth.best_bid_tick
        best_ask_tick = depth.best_ask_tick

        prev_mid_price_tick = mid_price_tick
        two_sided = best_bid_tick != INVALID_MIN and best_ask_tick != INVALID_MAX
        mid_price_tick = (best_bid_tick + best_ask_tick) / 2.0 if two_sided else np.nan

        mid_price_chg[t] = mid_price_tick - prev_mid_price_tick

        if t % recalibrate == 0 and t >= window - 1:
            A_, k_ = fit_intensity(arrival_depth[t + 1 - window : t + 1], tmp, ticks)
            if np.isfinite(A_):
                A = A_
                k = k_
            volatility = realised_volatility(mid_price_chg[t + 1 - window : t + 1], interval_ns)

        c1, c2 = compute_coeff(gamma, gamma, delta, A, k)

        half_spread_tick = (c1 + delta / 2 * c2 * volatility) * adj1
        skew = c2 * volatility * adj2

        reservation_price_tick = mid_price_tick - skew * position

        if two_sided:
            bid_price_tick = np.minimum(np.round(reservation_price_tick - half_spread_tick), best_bid_tick)
            ask_price_tick = np.maximum(np.round(reservation_price_tick + half_spread_tick), best_ask_tick)
        else:
            bid_price_tick = np.nan
            ask_price_tick = np.nan

        bid_price = bid_price_tick * tick_size
        ask_price = ask_price_tick * tick_size

        grid_interval = max(np.round(half_spread_tick) * tick_size, tick_size)
        bid_price = np.floor(bid_price / grid_interval) * grid_interval
        ask_price = np.ceil(ask_price / grid_interval) * grid_interval

        new_bid_orders = Dict.empty(np.uint64, np.float64)
        if position < max_position and np.isfinite(bid_price):
            for _ in range(grid_num):
                new_bid_orders[np.uint64(round(bid_price / tick_size))] = bid_price
                bid_price -= grid_interval

        new_ask_orders = Dict.empty(np.uint64, np.float64)
        if position > -max_position and np.isfinite(ask_price):
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

        out[t].half_spread_tick = half_spread_tick
        out[t].skew = skew
        out[t].volatility = volatility
        out[t].A = A
        out[t].k = k

        t += 1
        if t >= len(arrival_depth):
            raise Exception

        if t % record_every == 0:
            recorder.record(hbt)
    return out[:t]


@njit
def nanpercentile(values, q):
    """Percentile of the finite entries of `values`; `nan` if there are none."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.nan
    return np.percentile(finite, q)


@njit
def simplified_grid_mm(
    hbt, recorder, percentile, skew_factor, order_qty, max_position, grid_num, use_micro_price,
    interval_ns, record_every,
):
    """The non-parametric variant: empirical quote depth and a micro-price fair value.

    Two substitutions relative to the GLFT quoter above:

    - the half spread is the `percentile`-th percentile of the arrival depths
      seen in the trailing 10 minutes, instead of the closed form
      `c1 + (Delta/2) sigma c2`.  It reads directly: quoting there means about
      `100 - percentile` percent of trading steps produce a market order deep
      enough to reach us, with no assumption that `lambda` is exponential in
      depth.
    - the fair price is the size-weighted micro-price
      `(bid * ask_qty + ask * bid_qty) / (bid_qty + ask_qty)` instead of the
      mid.  Where the spread is one tick almost always, the mid takes only two
      values and the queue imbalance carries the information.

    Inventory skews the depth multiplicatively (`1 +/- skew_factor * q / q_max`)
    as in hftbacktest's simplified example, which keeps skew on the same scale
    as the spread by construction rather than by an `adj2` fudge factor.
    """
    tick_size = hbt.depth(0).tick_size
    arrival_depth = np.full(buffer_steps(interval_ns), np.nan, np.float64)
    mid_price_chg = np.full(buffer_steps(interval_ns), np.nan, np.float64)
    out = np.zeros(buffer_steps(interval_ns), out_dtype)

    t = 0
    prev_mid_price_tick = np.nan
    mid_price_tick = np.nan
    half_spread_tick = np.nan
    volatility = np.nan
    window = window_steps(interval_ns)
    recalibrate = recalibrate_steps(interval_ns)

    while hbt.elapse(interval_ns) == 0:
        if not np.isnan(mid_price_tick):
            depth = -np.inf
            for last_trade in hbt.last_trades(0):
                trade_price_tick = last_trade.px / tick_size

                if last_trade.ev & BUY_EVENT == BUY_EVENT:
                    depth = np.nanmax([trade_price_tick - mid_price_tick, depth])
                else:
                    depth = np.nanmax([mid_price_tick - trade_price_tick, depth])
            arrival_depth[t] = depth

        hbt.clear_last_trades(0)
        hbt.clear_inactive_orders(0)

        depth = hbt.depth(0)
        position = hbt.position(0)
        orders = hbt.orders(0)

        best_bid_tick = depth.best_bid_tick
        best_ask_tick = depth.best_ask_tick

        prev_mid_price_tick = mid_price_tick
        two_sided = best_bid_tick != INVALID_MIN and best_ask_tick != INVALID_MAX
        mid_price_tick = (best_bid_tick + best_ask_tick) / 2.0 if two_sided else np.nan

        mid_price_chg[t] = mid_price_tick - prev_mid_price_tick

        if two_sided and use_micro_price:
            # NOT depth.best_bid_qty: in hftbacktest 2.4.4 that property is
            # wired to the wrong FFI symbol for ROIVectorMarketDepth and
            # returns the best *price* (binding.py line 284 calls
            # roivecdepth_best_bid, though roivecdepth_best_bid_qty is
            # declared right above it at line 204; the HashMapMarketDepth
            # version is correct).  Using it makes the micro-price collapse to
            # exactly the mid, silently.
            bid_qty = depth.bid_qty_at_tick(best_bid_tick)
            ask_qty = depth.ask_qty_at_tick(best_ask_tick)
            fair_price_tick = (best_bid_tick * ask_qty + best_ask_tick * bid_qty) / (bid_qty + ask_qty)
        else:
            fair_price_tick = mid_price_tick

        if t % recalibrate == 0 and t >= window - 1:
            half_spread_tick = nanpercentile(arrival_depth[t + 1 - window : t + 1], percentile)
            volatility = realised_volatility(mid_price_chg[t + 1 - window : t + 1], interval_ns)

        normalised_position = position / max_position
        bid_depth_tick = half_spread_tick * (1 + skew_factor * normalised_position)
        ask_depth_tick = half_spread_tick * (1 - skew_factor * normalised_position)

        if two_sided:
            bid_price_tick = np.minimum(np.round(fair_price_tick - bid_depth_tick), best_bid_tick)
            ask_price_tick = np.maximum(np.round(fair_price_tick + ask_depth_tick), best_ask_tick)
        else:
            bid_price_tick = np.nan
            ask_price_tick = np.nan

        bid_price = bid_price_tick * tick_size
        ask_price = ask_price_tick * tick_size

        grid_interval = max(np.round(half_spread_tick) * tick_size, tick_size)
        bid_price = np.floor(bid_price / grid_interval) * grid_interval
        ask_price = np.ceil(ask_price / grid_interval) * grid_interval

        new_bid_orders = Dict.empty(np.uint64, np.float64)
        if position < max_position and np.isfinite(bid_price):
            for _ in range(grid_num):
                new_bid_orders[np.uint64(round(bid_price / tick_size))] = bid_price
                bid_price -= grid_interval

        new_ask_orders = Dict.empty(np.uint64, np.float64)
        if position > -max_position and np.isfinite(ask_price):
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

        out[t].half_spread_tick = half_spread_tick
        out[t].skew = skew_factor * half_spread_tick / max_position
        out[t].volatility = volatility
        out[t].A = np.nan
        out[t].k = np.nan

        t += 1
        if t >= len(arrival_depth):
            raise Exception

        if t % record_every == 0:
            recorder.record(hbt)
    return out[:t]


@njit
def touch_quoter(
    hbt, recorder, order_qty, max_position, interval_ns, record_every, fills, mids, n_steps
):
    """The simplest market maker there is: one lot at the touch, both sides.

    No model, no calibration, no grid -- it joins the best bid and the best ask
    and stops quoting a side once its inventory hits `max_position`.  It exists
    as a floor: whatever GLFT's closed form buys, it has to buy it over this.

    It also isolates something nothing else here does.  Every fill is logged as
    `(side, execution price in ticks, step)` and every step's mid-price goes
    into `mids`, so for any horizon `h`

        side * (mids[step + h] - execution price)

    is what the position taken by that fill was worth `h` steps later.  At
    `h = 0` that is execution quality: how far through the market the fill
    happened.  As `h` grows it becomes adverse selection: how far the price
    moved against the position afterwards.  C1's interval study separates the
    two, because they have different causes and only one of them is fixed by
    reacting faster.

    `n_steps` stops the replay early (0 replays the whole day), so a study can
    hold the simulated time fixed while varying the interval.  Pass length-1
    dummies for `fills` and `mids` when the log is not wanted.
    """
    tick_size = hbt.depth(0).tick_size
    t = 0
    n_fills = 0

    while hbt.elapse(interval_ns) == 0:
        hbt.clear_last_trades(0)

        depth = hbt.depth(0)
        position = hbt.position(0)
        orders = hbt.orders(0)

        best_bid_tick = depth.best_bid_tick
        best_ask_tick = depth.best_ask_tick
        if best_bid_tick == INVALID_MIN or best_ask_tick == INVALID_MAX:
            t += 1
            if n_steps > 0 and t >= n_steps:
                break
            continue
        mid_price_tick = (best_bid_tick + best_ask_tick) / 2.0
        if t < len(mids):
            mids[t] = mid_price_tick

        # Log fills before clearing them away.
        order_values = orders.values()
        while order_values.has_next():
            order = order_values.get()
            if order.status == FILLED and n_fills < len(fills):
                fills[n_fills, 0] = order.side
                fills[n_fills, 1] = order.exec_price_tick
                fills[n_fills, 2] = t
                n_fills += 1

        hbt.clear_inactive_orders(0)

        order_values = orders.values()
        while order_values.has_next():
            order = order_values.get()
            if order.cancellable and (
                (order.side == BUY and order.price_tick != best_bid_tick)
                or (order.side == SELL and order.price_tick != best_ask_tick)
            ):
                hbt.cancel(0, order.order_id, False)

        if position < max_position:
            order_id = np.uint64(best_bid_tick)
            if order_id not in orders:
                hbt.submit_buy_order(
                    0, order_id, best_bid_tick * tick_size, order_qty, GTX, LIMIT, False
                )
        if position > -max_position:
            order_id = np.uint64(best_ask_tick)
            if order_id not in orders:
                hbt.submit_sell_order(
                    0, order_id, best_ask_tick * tick_size, order_qty, GTX, LIMIT, False
                )

        t += 1
        if t % record_every == 0:
            recorder.record(hbt)
        if n_steps > 0 and t >= n_steps:
            break
    return n_fills


@njit
def measure_flow_continuation(hbt, interval_ns, n_steps, mids, flow, qty):
    """Order-flow continuation, from market data alone -- no orders, no fill model.

    For each step, records the mid-price, the signed aggressive volume (buys at
    or above the best ask minus sells at or below the best bid) and its
    magnitude.  Comparing the mid a fixed horizon later against the sign of the
    flow gives what a market maker filled by that flow would have felt, without
    the simulator's queue model being involved at all.

    This exists as a control.  C1's touch quoter shows fills marked tens of
    ticks against it within 100 ms, which is large enough to be a simulation
    artifact rather than a property of the market; this measures the same thing
    with the simulator's order handling taken out of the loop.
    """
    t = 0
    while hbt.elapse(interval_ns) == 0:
        depth = hbt.depth(0)
        best_bid_tick = depth.best_bid_tick
        best_ask_tick = depth.best_ask_tick
        buy_aggressive = 0.0
        sell_aggressive = 0.0
        if best_bid_tick != INVALID_MIN and best_ask_tick != INVALID_MAX:
            mids[t] = (best_bid_tick + best_ask_tick) / 2.0
            for last_trade in hbt.last_trades(0):
                price_tick = last_trade.px / depth.tick_size
                if last_trade.ev & BUY_EVENT == BUY_EVENT:
                    if price_tick >= best_ask_tick:
                        buy_aggressive += last_trade.qty
                elif price_tick <= best_bid_tick:
                    sell_aggressive += last_trade.qty
        hbt.clear_last_trades(0)
        flow[t] = buy_aggressive - sell_aggressive
        qty[t] = buy_aggressive + sell_aggressive
        t += 1
        if t >= n_steps:
            break
    return t
