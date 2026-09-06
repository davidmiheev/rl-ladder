"""Tabular MDPs and the two dynamic-programming solvers.

DP is the "cheating" baseline for this phase: it needs the transition model,
which A2's TD methods never see. Having the exact optimum available is what
lets item 2 talk about how close Q-learning and SARSA get, and why they differ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TabularMDP:
    """A finite MDP given by its model.

    Args:
        transitions (`np.ndarray`):
            `P[s, a, s']`, the probability of landing in `s'`.
        rewards (`np.ndarray`):
            `R[s, a]`, the expected immediate reward.
        terminal (`np.ndarray`):
            Boolean mask of absorbing states.
    """

    transitions: np.ndarray
    rewards: np.ndarray
    terminal: np.ndarray

    @property
    def n_states(self) -> int:
        return self.rewards.shape[0]

    @property
    def n_actions(self) -> int:
        return self.rewards.shape[1]

    def q_from_v(self, v: np.ndarray, gamma: float) -> np.ndarray:
        """One Bellman backup: `Q[s, a] = R[s, a] + gamma * sum_s' P[s, a, s'] V[s']`."""
        return self.rewards + gamma * self.transitions @ v

    def bellman_residual(self, v: np.ndarray, gamma: float) -> float:
        """`max_s |V(s) - max_a Q(s, a)|`. Zero exactly at the optimal value function."""
        return float(np.abs(v - self.q_from_v(v, gamma).max(axis=1)).max())


def value_iteration(
    mdp: TabularMDP, gamma: float, tol: float = 1e-12, max_iters: int = 100_000
) -> tuple[np.ndarray, np.ndarray, int]:
    """Sweep `V <- max_a Q(s, a)` to a fixed point.

    Returns:
        `(v, policy, iterations)`, the optimal values, a greedy policy, and the
        number of sweeps taken.
    """
    v = np.zeros(mdp.n_states)
    for iteration in range(1, max_iters + 1):  # noqa: B007 — returned as the sweep count
        v_next = mdp.q_from_v(v, gamma).max(axis=1)
        v_next[mdp.terminal] = 0.0
        delta = np.abs(v_next - v).max()
        v = v_next
        if delta < tol:
            break
    return v, mdp.q_from_v(v, gamma).argmax(axis=1), iteration


def evaluate_policy(mdp: TabularMDP, policy: np.ndarray, gamma: float) -> np.ndarray:
    """Exact policy evaluation: solve `(I - gamma * P_pi) V = R_pi`.

    Requires `gamma < 1`, or a policy under which every state reaches a terminal
    state — otherwise `I - P_pi` is singular. This is why `policy_iteration`
    below is only used with a discount.
    """
    states = np.arange(mdp.n_states)
    p_pi = mdp.transitions[states, policy]
    r_pi = mdp.rewards[states, policy]
    v = np.linalg.solve(np.eye(mdp.n_states) - gamma * p_pi, r_pi)
    v[mdp.terminal] = 0.0
    return v


def policy_iteration(
    mdp: TabularMDP, gamma: float, max_iters: int = 1_000
) -> tuple[np.ndarray, np.ndarray, int]:
    """Alternate exact evaluation and greedy improvement until the policy stops changing."""
    policy = np.zeros(mdp.n_states, dtype=int)
    for iteration in range(1, max_iters + 1):  # noqa: B007 — returned as the step count
        v = evaluate_policy(mdp, policy, gamma)
        greedy = mdp.q_from_v(v, gamma).argmax(axis=1)
        if np.array_equal(greedy, policy):
            break
        policy = greedy
    return evaluate_policy(mdp, policy, gamma), policy, iteration
