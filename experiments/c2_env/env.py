"""C2 -- hftbacktest wrapped as a Gymnasium environment.

The design decision that defines Part C: the policy does not emit quote prices.
It emits the *coefficients* of the GLFT formula C1 calibrated, and the formula
turns those into quotes. Three consequences, all of which C3 and C4 depend on:

- the fixed-coefficient baseline is a strict special case of the policy, so
  "did RL help" is a well-posed question with a checkable answer
  (`run_checks.py` verifies the env reproduces `glft-grid` to the cent);
- every action is interpretable in the language the model already uses --
  risk aversion, spread floor, spread scale, skew scale -- rather than as an
  opaque price;
- the policy cannot emit a nonsense quote, because the formula bounds what it
  can express.

What C1 forced into this design, rather than taste:

- **The action space includes not quoting.** Every parameter sweep in C1 was
  monotone towards trading less, so a policy that cannot decline to quote is
  being asked to choose the least bad way to lose. The half-spread scale
  reaches 8x, which on these assets puts both quotes far outside anything that
  trades.
- **The state carries order flow.** C1's whole loss is adverse selection, and
  the only thing found to predict it is aggressive trade flow: one at-touch
  trade calls the next 100 ms with a 1.9 sigma mean. A state of (inventory,
  volatility, spread) cannot see that; this one can.
- **The reward is marked to market every step.** Almost none of C1's loss
  appears in the execution price; nearly all of it accrues while the position
  is held. A reward that only credits closed round trips would give no
  gradient on the thing that actually costs money.
- **The policy acts every second, the quoter re-posts every 100 ms.** C1's
  interval study showed reacting faster fixes execution quality and leaves
  post-fill drift -- the dominant term -- untouched. So the expensive fast loop
  buys the small term, and the agent can act at the interval the coefficients
  are actually recalibrated at.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from hftbacktest import BUY, BUY_EVENT, GTX, LIMIT, SELL, Recorder
from numba import njit
from numba.typed import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "c1_glft_baseline"))

import config  # noqa: E402
import glft  # noqa: E402

#: How often the quoter re-prices and re-posts.  The notebook's interval, and
#: C1 found nothing that changes if it shrinks.
QUOTE_INTERVAL_NS = 100_000_000
#: How often the policy chooses coefficients.  The analytic model refits A, k
#: and sigma every 5 s, so acting faster than that mostly means repeating
#: yourself; 1 s is a compromise that keeps an episode at 86,400 steps.
STEP_INTERVAL_NS = 1_000_000_000

#: Indices into the persistent scalar state carried between steps.
S_T, S_A, S_K, S_VOL, S_MID, S_PREV_MID, S_EQUITY, S_SPREAD = range(8)
S_BIDQ, S_ASKQ, S_DEEPBIDQ, S_DEEPASKQ, S_HALF, S_SKEW = range(8, 14)
STATE_SIZE = 14

#: Price levels each side that the deep-imbalance feature sums over.
DEPTH_LEVELS = 5

#: Names of the observation's components, in order.  Keeping them as data
#: rather than as comments lets C5 extend the vector and lets the checks print
#: something readable when one of them goes out of range.
FEATURES = (
    "inventory",          # position / max_position, in [-1, 1] but not clipped there
    "volatility",         # sigma in bp per sqrt(second), scaled
    "spread",             # touch spread in ticks, scaled
    "imbalance",          # (bid_qty - ask_qty) / (bid_qty + ask_qty) at the touch
    "imbalance_depth",    # the same over the best DEPTH_LEVELS levels a side
    "flow_1s",            # signed aggressive volume over the last second, squashed
    "flow_10s",           # the same over ten seconds, squashed
    "log_k",              # the fitted intensity decay, standardised
    "model_half_spread",  # where the unmodified formula wants to quote, in bp
    "time_of_day",        # fraction of the session elapsed
)

#: Fixed scalings, not running statistics: running normalisation leaks state
#: across episodes and makes a run impossible to reproduce from its seed alone.
#: Each divisor is the rough scale of that feature measured across the ten
#: assets in C1's cross-asset study.
SCALE_VOLATILITY_BP = 1.0
SCALE_SPREAD_TICKS = 4.0
SCALE_HALF_SPREAD_BP = 2.0
SCALE_LOG_K = 2.0


@njit
def _squash(x: float, scale: float) -> float:
    return np.tanh(x / scale)


@njit
def quote_once(
    hbt, recorder, buf_arrival, buf_midchg, buf_flow, tmp, ticks, state,
    gamma, c1_scale, half_scale, skew_scale,
    order_qty, max_position, grid_num, interval_ns,
):
    """One 100 ms quoting iteration: the C1 grid quoter with free coefficients.

    Identical to `glft.gridtrading_glft_mm`'s loop body except that `gamma`,
    the weight on `c1`, and the scales on the half spread and the skew arrive
    as arguments instead of being constants.  Setting them to
    `(0.05, 1.0, 1.0, 0.05)` recovers that function exactly, which is what
    makes the baseline a strict special case; `run_checks.py` tests it rather
    than asserting it.

    Returns 0 while the day still has data and 1 when it is exhausted.
    """
    t = int(state[S_T])
    tick_size = hbt.depth(0).tick_size
    window = glft.window_steps(interval_ns)
    recalibrate = glft.recalibrate_steps(interval_ns)

    if hbt.elapse(interval_ns) != 0:
        return 1

    mid_price_tick = state[S_MID]

    # Arrival depth of market orders, and the signed aggressive volume that
    # C1 identified as the only predictor of the loss.
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

    state[S_PREV_MID] = mid_price_tick
    mid_price_tick = (best_bid_tick + best_ask_tick) / 2.0 if two_sided else np.nan
    state[S_MID] = mid_price_tick
    buf_midchg[t] = mid_price_tick - state[S_PREV_MID]

    if two_sided:
        # hftbacktest 2.4.4's `best_bid_qty` is wired to the wrong FFI symbol
        # for this depth type and returns the price; see C1's writeup.
        state[S_BIDQ] = depth.bid_qty_at_tick(best_bid_tick)
        state[S_ASKQ] = depth.ask_qty_at_tick(best_ask_tick)
        state[S_SPREAD] = best_ask_tick - best_bid_tick
        deep_bid = 0.0
        deep_ask = 0.0
        for level in range(DEPTH_LEVELS):
            deep_bid += depth.bid_qty_at_tick(best_bid_tick - level)
            deep_ask += depth.ask_qty_at_tick(best_ask_tick + level)
        state[S_DEEPBIDQ] = deep_bid
        state[S_DEEPASKQ] = deep_ask

    if t % recalibrate == 0 and t >= window - 1:
        A_, k_ = glft.fit_intensity(buf_arrival[t + 1 - window : t + 1], tmp, ticks)
        if np.isfinite(A_):
            state[S_A] = A_
            state[S_K] = k_
        state[S_VOL] = glft.realised_volatility(buf_midchg[t + 1 - window : t + 1], interval_ns)

    c1, c2 = glft.compute_coeff(gamma, gamma, 1.0, state[S_A], state[S_K])
    volatility = state[S_VOL]
    half_spread_tick = (c1_scale * c1 + 0.5 * c2 * volatility) * half_scale
    skew = c2 * volatility * skew_scale
    state[S_HALF] = half_spread_tick
    state[S_SKEW] = skew

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

    recorder.record(hbt)
    state[S_T] = t + 1
    return 0


@njit
def advance(
    hbt, recorder, buf_arrival, buf_midchg, buf_flow, tmp, ticks, state,
    gamma, c1_scale, half_scale, skew_scale,
    order_qty, max_position, grid_num, interval_ns, n_inner,
):
    """`n_inner` quoting iterations under one set of coefficients."""
    for _ in range(n_inner):
        if quote_once(
            hbt, recorder, buf_arrival, buf_midchg, buf_flow, tmp, ticks, state,
            gamma, c1_scale, half_scale, skew_scale,
            order_qty, max_position, grid_num, interval_ns,
        ) != 0:
            return 1
    return 0


@njit
def flow_over(buf_flow, t, steps):
    """Signed aggressive volume over the last `steps` quoting iterations."""
    start = max(0, t - steps)
    return np.sum(buf_flow[start:t])


class MarketMakingEnv(gym.Env):
    """One trading session per episode, one coefficient decision per second.

    Args:
        date (`str`):
            The session to replay. Must be a training day unless
            `allow_held_out` is set.
        symbol (`str`, *optional*, defaults to `"ethusdc"`):
            Which asset. Anything in `config.SYMBOLS`.
        order_qty (`float`, *optional*, defaults to `1.0`):
            Size of each grid order, in units of the asset.
        max_position (`float`, *optional*, defaults to `20.0`):
            Inventory cap, in units of the asset. The reward's risk penalty is
            measured against it.
        grid_num (`int`, *optional*, defaults to `20`):
            Orders per side.
        book_size (`float`, *optional*, defaults to `100000.0`):
            Capital the reward is expressed as basis points of.
        eta (`float`, *optional*, defaults to `0.015`):
            Weight on the inventory penalty, in basis points per step at a full
            position. Not a guess: `run_random.py` measures that a baseline
            session accumulates `sum q^2 = 1545` against a PnL term of −93 bp,
            so 0.015 makes the penalty a quarter of the PnL term over a
            session. Calibrating it against the PnL term's per-step standard
            deviation instead would be wrong by two orders of magnitude, since
            that term is near zero-mean noise while the penalty accumulates.
        allow_held_out (`bool`, *optional*, defaults to `False`):
            Permit a held-out day. Off by default so a training loop cannot
            silently read the test set.

    Examples:

    ```python
    >>> env = MarketMakingEnv(date="2025-08-26")
    >>> obs, info = env.reset(seed=0)
    >>> obs, reward, terminated, truncated, info = env.step(env.baseline_action())
    ```
    """

    metadata = {"render_modes": []}

    #: (low, high) for each action, in the units the formula uses.  The action
    #: itself is in [-1, 1] and is mapped through these logarithmically, so a
    #: policy's uniform prior is uniform over orders of magnitude.
    ACTION_RANGES = (
        ("gamma", 0.005, 1.0),
        ("c1_scale", 0.1, 4.0),
        ("half_spread_scale", 0.125, 8.0),
        ("skew_scale", 0.01, 2.0),
    )

    #: The coefficients that reproduce C1's `glft-grid` baseline exactly.
    BASELINE = (0.05, 1.0, 1.0, 0.05)

    def __init__(
        self,
        date: str,
        symbol: str = "ethusdc",
        order_qty: float = 1.0,
        max_position: float = 20.0,
        grid_num: int = 20,
        book_size: float = 100_000.0,
        eta: float = 0.015,
        step_interval_ns: int = STEP_INTERVAL_NS,
        quote_interval_ns: int = QUOTE_INTERVAL_NS,
        allow_held_out: bool = False,
    ) -> None:
        super().__init__()
        if date in config.TEST_DAYS and not allow_held_out:
            raise ValueError(
                f"{date} is a held-out day; pass allow_held_out=True and mean it"
            )
        config.use_symbol(symbol)
        self.date = date
        self.symbol = symbol
        self.order_qty = order_qty
        self.max_position = max_position
        self.grid_num = grid_num
        self.book_size = book_size
        self.eta = eta
        self.quote_interval_ns = quote_interval_ns
        self.n_inner = max(1, step_interval_ns // quote_interval_ns)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(self.ACTION_RANGES),), dtype=np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(len(FEATURES),), dtype=np.float32)

        self._hbt = None
        self._recorder = None

    # -- action mapping ----------------------------------------------------

    def to_coefficients(self, action: np.ndarray) -> tuple[float, float, float, float]:
        """Map an action in [-1, 1]^4 to GLFT coefficients, log-spaced."""
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        out = []
        for value, (_, low, high) in zip(action, self.ACTION_RANGES, strict=True):
            fraction = (value + 1.0) / 2.0
            out.append(float(np.exp(np.log(low) + fraction * (np.log(high) - np.log(low)))))
        return tuple(out)  # type: ignore[return-value]

    def baseline_action(self) -> np.ndarray:
        """The action whose coefficients are C1's fixed baseline."""
        out = []
        for target, (_, low, high) in zip(self.BASELINE, self.ACTION_RANGES, strict=True):
            fraction = (np.log(target) - np.log(low)) / (np.log(high) - np.log(low))
            out.append(2.0 * fraction - 1.0)
        return np.array(out, dtype=np.float32)

    # -- gym api -----------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.close()
        config.use_symbol(self.symbol)
        config.ensure_latency(self.date)
        self._hbt = config.build_backtest(self.date)
        # Recorded every quoting iteration, so an episode can be scored with
        # exactly the machinery C1 used and the two tables are comparable.
        self._recorder = Recorder(1, glft.buffer_steps(self.quote_interval_ns) + 1024)

        size = glft.buffer_steps(self.quote_interval_ns)
        self._arrival = np.full(size, np.nan)
        self._midchg = np.full(size, np.nan)
        self._flow = np.zeros(size)
        self._tmp = np.zeros(glft.MAX_BUCKET)
        self._ticks = np.arange(glft.MAX_BUCKET) + 0.5
        self._state = np.zeros(STATE_SIZE)
        self._state[S_A] = np.nan
        self._state[S_K] = np.nan
        self._state[S_VOL] = np.nan
        self._state[S_MID] = np.nan
        self._done = False
        self._steps = 0
        self._equity = 0.0
        self._session_steps = int(24 * 3600 * 1e9 / (self.n_inner * self.quote_interval_ns))

        # The model needs one calibration window before it can quote at all, so
        # the episode starts after it.  The policy is not asked for decisions
        # it cannot inform, and no reward is attributed to the warm-up.
        warmup = glft.window_steps(self.quote_interval_ns)
        gamma, c1_scale, half_scale, skew_scale = self.BASELINE
        status = advance(
            self._hbt, self._recorder.recorder, self._arrival, self._midchg, self._flow,
            self._tmp, self._ticks, self._state, gamma, c1_scale, half_scale, skew_scale,
            self.order_qty, self.max_position, self.grid_num, self.quote_interval_ns, warmup,
        )
        self._done = status != 0
        self._equity = self._mark_to_market()
        return self._observe(), {"warmup_steps": warmup}

    def step(self, action):
        if self._hbt is None:
            raise RuntimeError("call reset() before step()")
        gamma, c1_scale, half_scale, skew_scale = self.to_coefficients(action)
        status = advance(
            self._hbt, self._recorder.recorder, self._arrival, self._midchg, self._flow,
            self._tmp, self._ticks, self._state, gamma, c1_scale, half_scale, skew_scale,
            self.order_qty, self.max_position, self.grid_num, self.quote_interval_ns,
            self.n_inner,
        )
        self._steps += 1

        equity = self._mark_to_market()
        pnl_bp = (equity - self._equity) / self.book_size * 1e4
        self._equity = equity
        inventory = self._hbt.position(0) / self.max_position
        reward = pnl_bp - self.eta * inventory**2

        truncated = status != 0 or self._steps >= self._session_steps
        self._done = truncated
        info = {
            "pnl_bp": pnl_bp,
            "inventory": inventory,
            "equity": equity,
            "gamma": gamma,
            "c1_scale": c1_scale,
            "half_spread_scale": half_scale,
            "skew_scale": skew_scale,
            "half_spread_tick": self._state[S_HALF],
            "num_trades": self._num_trades(),
        }
        return self._observe(), float(reward), False, bool(truncated), info

    def equity_record(self) -> np.ndarray:
        """The episode's equity path, in the format `hftbacktest.stats` expects.

        Sampled every quoting iteration, so C3 can score a learned policy with
        the same `LinearAssetRecord` call C1 used for the fixed baseline and
        the two result tables mean the same thing.
        """
        return self._recorder.get(0)

    def close(self) -> None:
        if self._hbt is not None:
            self._hbt.close()
            self._hbt = None

    # -- internals ---------------------------------------------------------

    def _mark_to_market(self) -> float:
        state = self._hbt.state_values(0)
        mid = self._state[S_MID] * config.TICK_SIZE
        if not np.isfinite(mid):
            mid = 0.0
        return state.balance + state.position * mid - state.fee

    def _num_trades(self) -> int:
        return int(self._hbt.state_values(0).num_trades)

    def _observe(self) -> np.ndarray:
        state = self._state
        t = int(state[S_T])
        mid = state[S_MID]
        price = mid * config.TICK_SIZE if np.isfinite(mid) else np.nan
        tick_bp = config.TICK_SIZE / price * 1e4 if price and np.isfinite(price) else 0.0

        imbalance = _imbalance(state[S_BIDQ], state[S_ASKQ])
        imbalance_depth = _imbalance(state[S_DEEPBIDQ], state[S_DEEPASKQ])
        per_second = int(1e9 / self.quote_interval_ns)
        flow_1s = flow_over(self._flow, t, per_second)
        flow_10s = flow_over(self._flow, t, 10 * per_second)
        scale = max(self.order_qty, 1e-9) * 10.0

        vol_bp = state[S_VOL] * tick_bp
        half_bp = state[S_HALF] * tick_bp
        k = state[S_K]

        obs = np.array(
            [
                self._hbt.position(0) / self.max_position,
                _nan_to(vol_bp / SCALE_VOLATILITY_BP, 0.0),
                _nan_to(state[S_SPREAD] / SCALE_SPREAD_TICKS, 0.0),
                imbalance,
                imbalance_depth,
                _squash(flow_1s, scale),
                _squash(flow_10s, 10 * scale),
                _nan_to(np.log(k) / SCALE_LOG_K if k > 0 else 0.0, 0.0),
                _nan_to(half_bp / SCALE_HALF_SPREAD_BP, 0.0),
                min(self._steps / max(self._session_steps, 1), 1.0),
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -10.0, 10.0)


def _imbalance(bid_qty: float, ask_qty: float) -> float:
    """Queue imbalance, in [-1, 1]; zero when there is nothing on either side.

    Note that the touch micro-price carries no information this does not: for
    `mid = (P_b + P_a)/2` and `micro = (P_b q_a + P_a q_b)/(q_a + q_b)`,

        micro - mid = (P_a - P_b)/2 * (q_b - q_a)/(q_a + q_b)
                    = (spread/2) * imbalance

    exactly and always.  A micro-price feature alongside a touch-imbalance one
    would be the same number scaled by the spread, which is already its own
    feature.  The ladder's item lists both; measurement says they are one, so
    the second slot carries depth beyond the touch instead.
    """
    total = bid_qty + ask_qty
    return (bid_qty - ask_qty) / total if total > 0 else 0.0


def _nan_to(value: Any, fallback: float) -> float:
    value = float(value)
    return fallback if not np.isfinite(value) else value
