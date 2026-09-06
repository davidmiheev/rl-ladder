"""k-armed bandit testbeds and the three A1 agents.

The stationary testbed follows Sutton & Barto ch. 2: q*(a) ~ N(0, 1) and
reward ~ N(q*(a), 1). The non-stationary variant starts every arm equal and
lets each q*(a) take an independent random walk (S&B exercise 2.5).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def argmax_random_tiebreak(values: np.ndarray, rng: np.random.Generator) -> int:
    """Argmax that breaks ties uniformly.

    Matters more than it looks: with zero-initialized estimates every arm ties
    on the first step, and `np.argmax` would always open on arm 0.
    """
    candidates = np.flatnonzero(values == values.max())
    return int(rng.choice(candidates))


class GaussianBandit:
    """Stationary k-armed Gaussian bandit."""

    def __init__(self, k: int, rng: np.random.Generator):
        self.k = k
        self.rng = rng
        self.q_star = rng.normal(size=k)

    def pull(self, arm: int) -> float:
        return float(self.rng.normal(self.q_star[arm], 1.0))

    def step(self) -> None:
        """Advance the environment between pulls. Stationary: nothing moves."""

    @property
    def optimal_arm(self) -> int:
        return int(self.q_star.argmax())

    def regret(self, arm: int) -> float:
        """Expected regret of pulling `arm`, against the current best arm."""
        return float(self.q_star.max() - self.q_star[arm])


class RandomWalkBandit(GaussianBandit):
    """Non-stationary: every q*(a) takes an independent random walk."""

    def __init__(self, k: int, rng: np.random.Generator, walk_std: float = 0.01, start: float = 0.0):
        super().__init__(k, rng)
        self.q_star = np.full(k, float(start))
        self.walk_std = walk_std

    def step(self) -> None:
        self.q_star += self.rng.normal(0.0, self.walk_std, size=self.k)


class EpsilonGreedy:
    """epsilon-greedy. `step_size=None` uses sample averages, else a constant alpha."""

    def __init__(self, k: int, rng: np.random.Generator, eps: float = 0.1, step_size: float | None = None):
        self.k = k
        self.rng = rng
        self.eps = eps
        self.step_size = step_size
        self.q = np.zeros(k)
        self.n = np.zeros(k, dtype=int)

    def select(self) -> int:
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.k))
        return argmax_random_tiebreak(self.q, self.rng)

    def update(self, arm: int, reward: float) -> None:
        self.n[arm] += 1
        alpha = self.step_size if self.step_size is not None else 1.0 / self.n[arm]
        self.q[arm] += alpha * (reward - self.q[arm])


class UCB1:
    """UCB1: pick argmax Q(a) + c * sqrt(ln t / N(a)), opening with one pull per arm."""

    def __init__(self, k: int, rng: np.random.Generator, c: float = 2.0):
        self.k = k
        self.rng = rng
        self.c = c
        self.q = np.zeros(k)
        self.n = np.zeros(k, dtype=int)
        self.t = 0

    def select(self) -> int:
        untried = np.flatnonzero(self.n == 0)
        if untried.size:
            return int(untried[0])
        bonus = self.c * np.sqrt(np.log(self.t) / self.n)
        return argmax_random_tiebreak(self.q + bonus, self.rng)

    def update(self, arm: int, reward: float) -> None:
        self.t += 1
        self.n[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]


class ThompsonGaussian:
    """Thompson sampling with a Gaussian prior and known observation noise.

    Prior N(0, prior_var) and likelihood N(theta, noise_var) are conjugate, so
    after n pulls summing to s the posterior is Gaussian with
    precision = 1/prior_var + n/noise_var and mean = (s/noise_var) / precision.
    """

    def __init__(self, k: int, rng: np.random.Generator, prior_var: float = 1.0, noise_var: float = 1.0):
        self.k = k
        self.rng = rng
        self.prior_var = prior_var
        self.noise_var = noise_var
        self.sums = np.zeros(k)
        self.n = np.zeros(k, dtype=int)

    def select(self) -> int:
        precision = 1.0 / self.prior_var + self.n / self.noise_var
        var = 1.0 / precision
        mean = (self.sums / self.noise_var) * var
        return argmax_random_tiebreak(self.rng.normal(mean, np.sqrt(var)), self.rng)

    def update(self, arm: int, reward: float) -> None:
        self.n[arm] += 1
        self.sums[arm] += reward


Agent = EpsilonGreedy | UCB1 | ThompsonGaussian


def run_episode(
    make_agent: Callable[[np.random.Generator], Agent],
    make_bandit: Callable[[np.random.Generator], GaussianBandit],
    steps: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Run one agent on one bandit for `steps` pulls.

    Returns:
        `dict` with keys:
            - `cumulative_regret` (`np.ndarray`):
                Expected cumulative regret after each pull.
            - `optimal_action` (`np.ndarray`):
                1.0 where the pulled arm was the best arm at that moment.
            - `reward` (`np.ndarray`):
                Realized reward per pull.
    """
    rng = np.random.default_rng(seed)
    bandit = make_bandit(rng)
    agent = make_agent(rng)

    regret = 0.0
    cumulative, optimal, rewards = [], [], []
    for _ in range(steps):
        arm = agent.select()
        best = bandit.optimal_arm
        regret += bandit.regret(arm)
        reward = bandit.pull(arm)
        agent.update(arm, reward)

        cumulative.append(regret)
        optimal.append(float(arm == best))
        rewards.append(reward)
        bandit.step()

    return {
        "cumulative_regret": np.asarray(cumulative),
        "optimal_action": np.asarray(optimal),
        "reward": np.asarray(rewards),
    }
