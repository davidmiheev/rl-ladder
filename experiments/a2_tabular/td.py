"""Model-free control: Q-learning, SARSA, and Dyna-Q.

These never touch `GridWorld.to_mdp()` — they see only the transitions they
actually experience. That is the whole distinction from `mdp.py`.
"""

from __future__ import annotations

import numpy as np


def argmax_random_tiebreak(values: np.ndarray, rng: np.random.Generator) -> int:
    """Argmax with uniform tie-breaking (copied from A1 — phases stay self-contained)."""
    candidates = np.flatnonzero(values == values.max())
    return int(rng.choice(candidates))


class TabularTD:
    """Shared machinery: an epsilon-greedy policy over a Q-table.

    Subclasses differ only in `td_target`, which is the entire on-policy vs
    off-policy distinction.
    """

    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.5,
                 gamma: float = 1.0, eps: float = 0.1):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.q = np.zeros((n_states, n_actions))

    def act(self, state: int, rng: np.random.Generator) -> int:
        if rng.random() < self.eps:
            return int(rng.integers(self.n_actions))
        return argmax_random_tiebreak(self.q[state], rng)

    def td_target(self, reward: float, next_state: int, next_action: int, done: bool) -> float:
        raise NotImplementedError

    def update(self, state: int, action: int, reward: float, next_state: int,
               next_action: int, done: bool) -> None:
        target = self.td_target(reward, next_state, next_action, done)
        self.q[state, action] += self.alpha * (target - self.q[state, action])

    def greedy_policy(self) -> np.ndarray:
        return self.q.argmax(axis=1)


class QLearning(TabularTD):
    """Off-policy: bootstrap from the best next action, whatever the policy did."""

    def td_target(self, reward, next_state, next_action, done):
        return reward if done else reward + self.gamma * self.q[next_state].max()


class Sarsa(TabularTD):
    """On-policy: bootstrap from the action the epsilon-greedy policy actually took."""

    def td_target(self, reward, next_state, next_action, done):
        return reward if done else reward + self.gamma * self.q[next_state, next_action]


class DynaQ(QLearning):
    """Q-learning plus `planning_steps` replayed updates from a learned model.

    The model is a dict `(s, a) -> (reward, next_state)`, which assumes
    deterministic dynamics — true for the maze this is run on, and the
    assumption Sutton & Barto's Dyna-Q makes.
    """

    def __init__(self, *args, planning_steps: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.planning_steps = planning_steps
        self.model: dict[tuple[int, int], tuple[float, int, bool]] = {}

    def update(self, state, action, reward, next_state, next_action, done):
        super().update(state, action, reward, next_state, next_action, done)
        self.model[(state, action)] = (reward, next_state, done)

    def plan(self, rng: np.random.Generator) -> None:
        """Replay `planning_steps` remembered transitions, chosen uniformly."""
        if not self.model:
            return
        keys = list(self.model)
        for index in rng.integers(len(keys), size=self.planning_steps):
            state, action = keys[index]
            reward, next_state, done = self.model[(state, action)]
            target = reward if done else reward + self.gamma * self.q[next_state].max()
            self.q[state, action] += self.alpha * (target - self.q[state, action])


def train(env, agent, episodes: int, rng: np.random.Generator, max_steps: int = 1_000
          ) -> tuple[np.ndarray, np.ndarray]:
    """Run `episodes` episodes, returning per-episode (undiscounted return, length)."""
    returns, lengths = [], []
    for _ in range(episodes):
        state = env.reset()
        action = agent.act(state, rng)
        total, steps = 0.0, 0

        for steps in range(1, max_steps + 1):  # noqa: B007 — recorded as the episode length
            next_state, reward, done = env.step(action, rng)
            next_action = agent.act(next_state, rng)
            agent.update(state, action, reward, next_state, next_action, done)
            if isinstance(agent, DynaQ):
                agent.plan(rng)

            total += reward
            state, action = next_state, next_action
            if done:
                break

        returns.append(total)
        lengths.append(steps)
    return np.asarray(returns), np.asarray(lengths)
