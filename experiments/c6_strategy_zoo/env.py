"""C6 -- one Gymnasium environment over all five strategies.

C2 wrapped GLFT and gave the policy GLFT's own coefficients as its action.
That made the baseline a strict special case of the policy, which is what
turns "did RL help" into a question with a checkable answer.  C6 keeps the
property and drops the strategy-specific part: the action is four
*multipliers* that every one of the five strategies already has a slot for.

    spread_mult   on the half spread, whatever the strategy derives it from
    skew_mult     on the inventory skew
    alpha_mult    on the strategy's own fair-price signal
    grid_mult     on the grid step

All four are 1.0 at the published parameters, so each baseline sits in the
interior of its own policy's action space and the comparison is like for like
five times over rather than once.  It also means one policy architecture, one
observation vector and one reward serve all five, so a difference in the
results is a difference between the strategies and not between five
differently-tuned experiments.

What the multipliers give up relative to C2: the policy can no longer reach
`gamma` or `c1` separately for GLFT, so C3's γ-pinning ablation has no
counterpart here.  C3 already answered that question (pinning γ shrinks the
seed spread eightfold and costs 4.6 points of return), and repeating it for
one of five strategies would not generalise.

`alpha_mult` is the interesting one, and it is C5's item in disguise: setting
it to 0 turns each strategy into a pure mid-price quoter with the same spread
and skew, so the policy's learned value for it measures what that strategy's
signal is actually worth on these days -- and whether the policy leans on the
signal more when the signal is informative.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from hftbacktest import Recorder
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "c1_glft_baseline"))

import assets  # noqa: E402
import config  # noqa: E402
import glft  # noqa: E402
import precompute  # noqa: E402
import strategies  # noqa: E402
import zoo  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "c7_signals"))

import book_info  # noqa: E402

#: How often the policy chooses multipliers.  One second, as in C2: the
#: analytic model refits A, k and sigma every 5 s, so acting much faster is
#: mostly repeating yourself, and 1 s keeps an episode at 86,400 steps.
STEP_INTERVAL_NS = 1_000_000_000

FEATURES = (
    "inventory",          # position notional / cap, in [-1, 1]
    "volatility",         # sigma in bp per sqrt(second)
    "spread",             # touch spread in ticks, scaled
    "imbalance",          # queue imbalance at the touch
    "imbalance_depth",    # the same over the best five levels a side
    "flow_1s",            # signed aggressive volume over the last second
    "flow_10s",           # the same over ten seconds
    "log_k",              # the fitted intensity decay, standardised
    "model_half_spread",  # where the strategy wants to quote, in bp
    "signal",             # the strategy's own fair-price offset, in bp
    "time_of_day",        # fraction of the session elapsed
)

#: C7's extra observations (recipe `v2-deep`): deeper book imbalance in ticks
#: and in bp, order-flow imbalance, our own recent returns, and BTCUSDT's
#: returns (on its 100 ms grid), touch imbalance and OFI -- each with the
#: fixed scale it is divided by. Read per second from `data/features/`, the
#: second in force at each step, so nothing from the future is visible.
EXTRA_FEATURES = (
    ("own_imb_l20", 1.0), ("own_imb_l50", 1.0), ("own_imb_l200", 1.0),
    ("own_imb_1bp", 1.0), ("own_imb_5bp", 1.0), ("own_imb_25bp", 1.0),
    ("own_ofi_1s", 1.0), ("own_ofi_10s", 1.0),
    ("own_ret_10s", 5.0), ("own_ret_60s", 10.0),
    ("ref_fresh_500ms", 2.0), ("ref_fresh_5s", 5.0),
    ("ref_imb_l1", 1.0), ("ref_ofi_10s", 1.0),
)

#: Fixed scalings, not running statistics: running normalisation leaks state
#: across episodes and makes a run impossible to reproduce from its seed.
SCALE_VOLATILITY_BP = 1.0
SCALE_SPREAD_TICKS = 4.0
SCALE_HALF_SPREAD_BP = 4.0
SCALE_SIGNAL_BP = 2.0
SCALE_LOG_K = 2.0


@njit
def flow_over(buf_flow, t, steps):
    """Signed aggressive volume over the last `steps` quoting iterations."""
    start = max(0, t - steps)
    return np.sum(buf_flow[start:t])


class ZooEnv(gym.Env):
    """One session per episode, one multiplier decision per second.

    Args:
        date (`str`):
            The session to replay. Must be a training day unless
            `allow_held_out` is set.
        strategy (`str`, *optional*, defaults to `"glft"`):
            Any name in `strategies.NAMES`. The `-wide` variants differ only
            in their spread floor, so training against one and evaluating
            against the other is a meaningful ablation.
        symbol (`str`, *optional*, defaults to `"ethusdc"`):
            Which asset. `apt` and `basis` need one with a reference leg.
        book_size (`float`, *optional*, defaults to `100000.0`):
            Capital the reward is expressed as basis points of.
        eta (`float`, *optional*, defaults to `0.015`):
            Weight on the inventory penalty, in bp per step at a full
            position. C2 measured this against a baseline session rather than
            guessing it; C6 keeps the value so its rewards are on C3's scale.
        max_steps (`int`, *optional*, defaults to `0`):
            Cap the episode at this many policy steps. `0` runs the whole
            session, which is what every reported result uses.
        action_ranges (`dict[str, tuple[float, float]]`, *optional*):
            Override named action ranges, e.g. `ZooEnv.FORCED_PARTICIPATION`.
        overrides (`dict[str, float]`, *optional*):
            Direct overrides of named entries of the strategy's parameter
            vector, e.g. `{"grid_num": 40}`.
        allow_held_out (`bool`, *optional*, defaults to `False`):
            Permit a held-out day. Off by default so a training loop cannot
            silently read the test set.

    Examples:

    ```python
    >>> env = ZooEnv(date="2025-08-26", strategy="hfgrid")
    >>> obs, info = env.reset(seed=0)
    >>> obs, reward, terminated, truncated, info = env.step(env.baseline_action())
    ```
    """

    metadata = {"render_modes": []}

    #: (low, high) for each multiplier.  The action is in [-1, 1] and maps
    #: through these logarithmically, so a uniform prior over actions is
    #: uniform over orders of magnitude -- which is the right prior for a
    #: multiplier, and the wrong one for an additive offset.
    ACTION_RANGES = (
        ("spread_mult", 0.25, 8.0),
        ("skew_mult", 0.05, 10.0),
        ("alpha_mult", 0.05, 4.0),
        ("grid_mult", 0.25, 4.0),
    )

    #: The multipliers that reproduce the published baseline exactly.
    BASELINE = (1.0, 1.0, 1.0, 1.0)

    #: C3 found that with a free spread range the best policy available on a
    #: training day is the corner that quotes so far out it takes no fills and
    #: returns exactly zero -- a true answer to the wrong question.  Capping
    #: the spread multiplier at 2x forces the policy to make a market, so the
    #: comparison is between two market makers rather than between a market
    #: maker and an empty order book.  The baseline stays interior.
    FORCED_PARTICIPATION = {"spread_mult": (0.5, 2.0)}

    def __init__(
        self,
        date: str,
        strategy: str = "glft",
        symbol: str = "ethusdc",
        book_size: float = strategies.BOOK_SIZE,
        eta: float = 0.015,
        step_interval_ns: int = STEP_INTERVAL_NS,
        quote_interval_ns: int = strategies.INTERVAL_NS,
        max_steps: int = 0,
        action_ranges: dict[str, tuple[float, float]] | None = None,
        overrides: dict[str, float] | None = None,
        beta: float = 1.0,
        allow_held_out: bool = False,
        residual: float = 0.0,
        reward_mode: str = "absolute",
        deviation_penalty: float = 0.0,
        cf_coef: float = 1.0,
        extra_features: bool = False,
        inventory_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if reward_mode not in ("absolute", "counterfactual"):
            raise ValueError(f"unknown reward_mode {reward_mode!r}")
        assets.ensure_registered(symbol)
        if date in assets.test_days(symbol) and not allow_held_out:
            raise ValueError(f"{date} is a held-out day; pass allow_held_out=True and mean it")
        config.use_symbol(symbol)
        self.date = date
        self.strategy = strategy
        self.symbol = symbol
        self.book_size = book_size
        self.eta = eta
        self.quote_interval_ns = quote_interval_ns
        self.max_steps = max_steps
        self.n_inner = max(1, step_interval_ns // quote_interval_ns)

        kwargs = {"beta": beta} if strategy.split("-")[0] == "apt" else {}
        self.params = strategies.build(strategy, symbol, **kwargs)
        for name, value in (overrides or {}).items():
            self.params[zoo.PARAM_NAMES.index(name)] = value
        self.max_position_dollar = float(self.params[zoo.P_MAX_POSITION_DOLLAR])

        base = strategy.split("-")[0]
        self.table = (
            np.ascontiguousarray(precompute.load(base, symbol, date), dtype=np.float64)
            if base in strategies.PAIR_STRATEGIES
            else np.zeros((1, 3), np.float64)
        )
        config.use_symbol(symbol)  # `precompute` repoints the symbol while building

        ranges = action_ranges or {}
        if residual:
            # A residual policy: every multiplier ranges over [1/R, R], which
            # is symmetric in log space about 1, so the action 0 *is* the
            # baseline.  A zero-initialised actor therefore starts exactly on
            # the published-plus-floor parameters and can only learn to move
            # away from them, which is the question being asked.
            if residual <= 1.0:
                raise ValueError("residual range must exceed 1")
            ranges = {name: (1.0 / residual, residual) for name, _, _ in self.ACTION_RANGES}
        self.residual = float(residual)
        self.action_ranges = tuple(
            (name, *ranges.get(name, (low, high))) for name, low, high in self.ACTION_RANGES
        )
        self.reward_mode = reward_mode
        self.deviation_penalty = float(deviation_penalty)
        # How much of the baseline's reward to subtract.  1 is the plain
        # counterfactual; the variance-optimal constant is Cov(r, b) / Var(b),
        # which is 1 only when the policy trades like the baseline.  Any
        # constant leaves the optimal policy unchanged, because the baseline's
        # reward never depends on the action.
        self.cf_coef = float(cf_coef)
        # How much of the step's PnL that comes from *holding* inventory
        # (last step's position times the mid's move) counts, against the PnL
        # that comes from *trading* (this step's fills marked at the step's
        # closing mid, plus rebates and fees). The two sum to the step's PnL
        # exactly, so 1 is the plain reward. Below 1, the reward credits the
        # market making and discounts the direction: a windfall from holding
        # the right inventory through a crash is not something to learn.
        self.inventory_weight = float(inventory_weight)
        self.step_interval_ns = int(step_interval_ns)
        self._overrides = dict(overrides or {})
        self._beta = beta
        self._baseline_trace = None
        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(self.action_ranges),), dtype=np.float32)
        self.extra_features = bool(extra_features)
        n_obs = len(FEATURES) + (len(EXTRA_FEATURES) if extra_features else 0)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(n_obs,), dtype=np.float32)
        self._extra_ts = None
        self._extra = None

        self._hbt = None
        self._recorder = None

    # -- action mapping ----------------------------------------------------

    def to_multipliers(self, action: np.ndarray) -> tuple[float, float, float, float]:
        """Map an action in [-1, 1]^4 to the four multipliers, log-spaced."""
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        out = []
        for value, (_, low, high) in zip(action, self.action_ranges, strict=True):
            fraction = (value + 1.0) / 2.0
            out.append(float(np.exp(np.log(low) + fraction * (np.log(high) - np.log(low)))))
        return tuple(out)  # type: ignore[return-value]

    def baseline_action(self) -> np.ndarray:
        """The action whose multipliers are the published baseline."""
        out = []
        for target, (_, low, high) in zip(self.BASELINE, self.action_ranges, strict=True):
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
        self._recorder = Recorder(1, glft.buffer_steps(self.quote_interval_ns) + 1024)

        size = glft.buffer_steps(self.quote_interval_ns)
        self._arrival = np.full(size, np.nan)
        self._midchg = np.full(size, np.nan)
        self._flow = np.zeros(size)
        self._obi = np.full(max(int(self.params[zoo.P_OBI_WINDOW]), 1), np.nan)
        self._out = np.zeros(size, zoo.out_dtype)
        self._tmp = np.zeros(glft.MAX_BUCKET)
        self._ticks = np.arange(glft.MAX_BUCKET) + 0.5
        self._state = np.zeros(zoo.STATE_SIZE)
        for slot in (zoo.Z_A, zoo.Z_K, zoo.Z_VOL, zoo.Z_MID):
            self._state[slot] = np.nan
        self._done = False
        self._steps = 0
        self._equity = 0.0
        self._session_steps = int(24 * 3600 * 1e9 / (self.n_inner * self.quote_interval_ns))
        if self.max_steps:
            self._session_steps = min(self._session_steps, self.max_steps)

        # The strategies that fit A, k and sigma need one calibration window
        # before they can quote at all, so every episode starts after one and
        # no reward is attributed to the warm-up.  Running it for all five
        # keeps the episodes the same length.
        warmup = glft.window_steps(self.quote_interval_ns)
        self._done = self._advance(self.BASELINE, warmup) != 0
        self._equity = self._mark_to_market()
        if self.reward_mode == "counterfactual" and self._baseline_trace is None:
            self._baseline_trace = baseline_trace(
                self.strategy, self.symbol, self.date, self.step_interval_ns, self.eta,
                self._overrides, self._beta, self.inventory_weight,
            )
        self._reference = self.baseline_action()
        self._last_position = self._hbt.position(0)
        self._last_mid = self._mid()
        if self.extra_features and self._extra is None:
            x, _, ts = book_info.day_features(self.symbol, self.date)
            self._extra_ts = ts
            self._extra = np.column_stack(
                [np.nan_to_num(x[name]) / scale for name, scale in EXTRA_FEATURES]
            ).astype(np.float32)
        return self._observe(), {"warmup_steps": warmup}

    def step(self, action):
        if self._hbt is None:
            raise RuntimeError("call reset() before step()")
        multipliers = self.to_multipliers(action)
        status = self._advance(multipliers, self.n_inner)
        self._steps += 1

        equity = self._mark_to_market()
        pnl_bp = (equity - self._equity) / self.book_size * 1e4
        self._equity = equity
        position, mid = self._hbt.position(0), self._mid()
        inventory = position * mid / self.max_position_dollar
        holding_bp = self._last_position * (mid - self._last_mid) / self.book_size * 1e4
        trading_bp = pnl_bp - holding_bp
        self._last_position, self._last_mid = position, mid
        absolute = trading_bp + self.inventory_weight * holding_bp - self.eta * inventory**2
        reward = absolute
        baseline_reward = 0.0
        if self._baseline_trace is not None and self._steps - 1 < len(self._baseline_trace):
            # The same session replayed at the baseline action produced this
            # reward at this step.  Subtracting it leaves only what the policy
            # changed: the market's own move, which dominates both, cancels.
            # It depends on the day and the step but never on the action, so
            # it cannot change which policy is optimal -- only how noisy the
            # signal pointing to it is.
            baseline_reward = float(self._baseline_trace[self._steps - 1])
            reward = absolute - self.cf_coef * baseline_reward
        if self.deviation_penalty:
            deviation = np.asarray(action, dtype=np.float64) - self._reference
            reward -= self.deviation_penalty * float(np.mean(deviation**2))

        truncated = status != 0 or self._steps >= self._session_steps
        self._done = truncated
        info = {
            "pnl_bp": pnl_bp,
            "trading_bp": trading_bp,
            "holding_bp": holding_bp,
            "absolute_reward": absolute,
            "baseline_reward": baseline_reward,
            "inventory": inventory,
            "equity": equity,
            "spread_mult": multipliers[0],
            "skew_mult": multipliers[1],
            "alpha_mult": multipliers[2],
            "grid_mult": multipliers[3],
            "half_spread_bp": self._half_spread_bp(),
            "num_trades": int(self._hbt.state_values(0).num_trades),
        }
        return self._observe(), float(reward), False, bool(truncated), info

    def equity_record(self) -> np.ndarray:
        """The episode's equity path, in the format `hftbacktest.stats` expects."""
        return self._recorder.get(0)

    def quoter_log(self) -> np.ndarray:
        """What the quoter did each iteration: spread, skew, signal, position."""
        return self._out[: int(self._state[zoo.Z_T])]

    def close(self) -> None:
        if self._hbt is not None:
            self._hbt.close()
            self._hbt = None

    # -- internals ---------------------------------------------------------

    def _advance(self, multipliers, n_inner: int) -> int:
        return zoo.advance(
            self._hbt, self._recorder.recorder, self._state, self._out, self.params,
            self.table, self._arrival, self._midchg, self._flow, self._obi,
            self._tmp, self._ticks, *multipliers, self.quote_interval_ns, 1, n_inner,
        )

    def _mid(self) -> float:
        mid = self._state[zoo.Z_MID] * config.TICK_SIZE
        return mid if np.isfinite(mid) else 0.0

    def _mark_to_market(self) -> float:
        state = self._hbt.state_values(0)
        return state.balance + state.position * self._mid() - state.fee

    def _half_spread_bp(self) -> float:
        mid = self._mid()
        return self._state[zoo.Z_HALF_PX] / mid * 1e4 if mid else float("nan")

    def _observe(self) -> np.ndarray:
        state = self._state
        t = int(state[zoo.Z_T])
        price = self._mid()
        tick_bp = config.TICK_SIZE / price * 1e4 if price else 0.0

        per_second = int(1e9 / self.quote_interval_ns)
        scale = max(self.params[zoo.P_ORDER_QTY_DOLLAR] / price if price else 1.0, 1e-9) * 10.0
        k = state[zoo.Z_K]

        obs = np.array(
            [
                state[zoo.Z_NORM_POS],
                _nan_to(state[zoo.Z_VOL] * tick_bp / SCALE_VOLATILITY_BP, 0.0),
                _nan_to(state[zoo.Z_SPREAD] / SCALE_SPREAD_TICKS, 0.0),
                _imbalance(state[zoo.Z_BIDQ], state[zoo.Z_ASKQ]),
                _imbalance(state[zoo.Z_DEEPBIDQ], state[zoo.Z_DEEPASKQ]),
                np.tanh(flow_over(self._flow, t, per_second) / scale),
                np.tanh(flow_over(self._flow, t, 10 * per_second) / (10 * scale)),
                _nan_to(np.log(k) / SCALE_LOG_K if k > 0 else 0.0, 0.0),
                _nan_to(self._half_spread_bp() / SCALE_HALF_SPREAD_BP, 0.0),
                _nan_to(state[zoo.Z_SIGNAL_PX] / price * 1e4 / SCALE_SIGNAL_BP if price else 0.0, 0.0),
                min(self._steps / max(self._session_steps, 1), 1.0),
            ],
            dtype=np.float32,
        )
        if self.extra_features:
            row = np.searchsorted(self._extra_ts, self._hbt.current_timestamp, side="right") - 1
            extra = self._extra[row] if row >= 0 else np.zeros(len(EXTRA_FEATURES), np.float32)
            obs = np.concatenate([obs, extra])
        return np.clip(obs, -10.0, 10.0)


def baseline_trace(strategy: str, symbol: str, date: str, step_interval_ns: int,
                   eta: float, overrides: dict | None = None, beta: float = 1.0,
                   inventory_weight: float = 1.0) -> np.ndarray:
    """Per-step reward of the baseline on one session, computed once and cached.

    The counterfactual reward is the policy's step reward minus this. Because
    a replay is deterministic, a policy that always plays the baseline action
    earns exactly zero at every step -- `run_checks.py` tests that rather than
    assuming it, since an off-by-one in the alignment would silently hand the
    agent a reward that leaks the next step's market move.
    """
    overrides = dict(overrides or {})
    tag = "-".join(f"{k}{v:g}" for k, v in sorted(overrides.items())) or "none"
    if inventory_weight != 1.0:
        tag += f"_w{inventory_weight:g}"  # weight 1 keeps the existing cache names
    step_s = step_interval_ns / 1e9
    path = (precompute.PRECOMP /
            f"cf_{strategy}_{symbol}_{date}_{step_s:g}s_eta{eta:g}_b{beta:g}_{tag}.npy")
    if path.exists():
        return np.load(path)
    env = ZooEnv(date=date, strategy=strategy, symbol=symbol, eta=eta,
                 step_interval_ns=step_interval_ns, overrides=overrides, beta=beta,
                 allow_held_out=True, inventory_weight=inventory_weight)
    env.reset(seed=0)
    action = env.baseline_action()
    rewards = []
    truncated = False
    while not truncated:
        _, reward, _, truncated, _ = env.step(action)
        rewards.append(reward)
    env.close()
    trace = np.asarray(rewards, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written to a temporary name and renamed, so two training runs that
    # build the same trace at once cannot leave a half-written file behind.
    tmp = path.with_suffix(f".tmp{os.getpid()}.npy")
    np.save(tmp, trace)
    os.replace(tmp, path)
    return trace


def _imbalance(bid_qty: float, ask_qty: float) -> float:
    """Queue imbalance, in [-1, 1]; zero when there is nothing on either side.

    The touch micro-price carries no information this does not: at the touch
    `micro - mid = (spread/2) * imbalance` exactly, and the spread is already
    its own feature.  C2 measured that identity rather than assuming it.
    """
    total = bid_qty + ask_qty
    return (bid_qty - ask_qty) / total if total > 0 else 0.0


def _nan_to(value: Any, fallback: float) -> float:
    value = float(value)
    return fallback if not np.isfinite(value) else value
