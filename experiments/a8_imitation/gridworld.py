"""A small gridworld MDP with exact solvers, for imitation and inverse RL.

Copied and adapted from A2 rather than imported — phases stay self-contained
(docs/repo-layout.md). The additions here are the soft (entropy-regularised)
Bellman solvers that MaxEnt IRL needs, and an expert-demonstration sampler.

Why a gridworld and not CartPole (which the ladder suggests): MaxEnt IRL's
gradient needs expected state-visitation frequencies, which are a dynamic
program over a discrete state space. A tabular env also makes item 3's "states
the demonstrations never visited" an exactly enumerable set rather than a fuzzy
region of a continuous space.
"""

from __future__ import annotations

import numpy as np

MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_CHARS = "^>v<"

LAYOUT = [
    "S..#....",
    ".#.#.##.",
    ".#....#.",
    ".###.#..",
    "...#.#.#",
    ".#...#..",
    ".#.###..",
    "...#...G",
]


class GridWorld:
    def __init__(self, layout: list[str] = LAYOUT, slip: float = 0.1, step_reward: float = -1.0,
                 goal_reward: float = 10.0):
        self.layout = layout
        self.rows, self.cols = len(layout), len(layout[0])
        self.slip = slip
        self.step_reward = step_reward
        self.goal_reward = goal_reward

        self.cells = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        self.index = {cell: i for i, cell in enumerate(self.cells)}
        self.n_states, self.n_actions = len(self.cells), 4
        self.walls = np.array([self._char(c) == "#" for c in self.cells])
        self.terminal = np.array([self._char(c) == "G" for c in self.cells])
        self.start = self.index[self._find("S")]
        self.transitions, self.rewards = self._build()

    def _char(self, cell):
        return self.layout[cell[0]][cell[1]]

    def _find(self, ch):
        return next(c for c in self.cells if self._char(c) == ch)

    def _move(self, cell, delta):
        r, c = cell[0] + delta[0], cell[1] + delta[1]
        if not (0 <= r < self.rows and 0 <= c < self.cols) or self.layout[r][c] == "#":
            return cell
        return (r, c)

    def _build(self):
        transitions = np.zeros((self.n_states, self.n_actions, self.n_states))
        rewards = np.zeros(self.n_states)
        for s, cell in enumerate(self.cells):
            if self.walls[s]:
                transitions[s, :, s] = 1.0
                continue
            rewards[s] = self.goal_reward if self.terminal[s] else self.step_reward
            if self.terminal[s]:
                transitions[s, :, s] = 1.0
                continue
            for a in range(self.n_actions):
                outcomes = [(1 - self.slip, a), (self.slip / 2, (a - 1) % 4), (self.slip / 2, (a + 1) % 4)]
                for prob, move in outcomes:
                    transitions[s, a, self.index[self._move(cell, MOVES[move])]] += prob
        return transitions, rewards

    # ---------------- exact solvers ----------------

    def value_iteration(self, rewards=None, gamma: float = 0.95, tol: float = 1e-10):
        """Hard Bellman backup. Returns `(values, greedy policy)`."""
        rewards = self.rewards if rewards is None else rewards
        v = np.zeros(self.n_states)
        for _ in range(10_000):
            q = rewards[:, None] + gamma * self.transitions @ v
            nxt = q.max(axis=1)
            nxt[self.terminal] = rewards[self.terminal]
            if np.abs(nxt - v).max() < tol:
                v = nxt
                break
            v = nxt
        q = rewards[:, None] + gamma * self.transitions @ v
        return v, q.argmax(axis=1)

    def soft_value_iteration(self, rewards, gamma: float = 0.95, tol: float = 1e-10):
        """Entropy-regularised backup: `V(s) = log sum_a exp Q(s,a)`.

        This is the planner implied by the MaxEnt model — trajectories are
        assumed exponentially preferred by return rather than strictly optimal,
        which is what makes the likelihood differentiable and lets suboptimal
        demonstrations carry information instead of being contradictions.
        """
        v = np.zeros(self.n_states)
        for _ in range(10_000):
            q = rewards[:, None] + gamma * self.transitions @ v
            nxt = np.log(np.exp(q - q.max(axis=1, keepdims=True)).sum(axis=1)) + q.max(axis=1)
            nxt[self.terminal] = rewards[self.terminal]
            if np.abs(nxt - v).max() < tol:
                v = nxt
                break
            v = nxt
        q = rewards[:, None] + gamma * self.transitions @ v
        policy = np.exp(q - v[:, None])
        policy[self.terminal] = 1.0 / self.n_actions
        return v, policy / policy.sum(axis=1, keepdims=True)

    def rollout(self, policy, start: int, rng: np.random.Generator, max_steps: int = 200):
        """Follow `policy` (deterministic array or stochastic matrix) from `start`."""
        state, states, actions, total = start, [], [], 0.0
        for _ in range(max_steps):
            if self.terminal[state]:
                break
            if policy.ndim == 1:
                action = int(policy[state])
            else:
                action = int(rng.choice(self.n_actions, p=policy[state]))
            states.append(state)
            actions.append(action)
            state = int(rng.choice(self.n_states, p=self.transitions[state, action]))
            total += self.rewards[state]
        return states, actions, total, bool(self.terminal[state])

    def free_states(self) -> np.ndarray:
        return np.flatnonzero(~self.walls & ~self.terminal)
