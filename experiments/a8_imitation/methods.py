"""Behavioural cloning and MaxEnt inverse RL.

The contrast the phase is built on: BC fits a *conditional distribution over
actions* and therefore says nothing where it has no data, while IRL fits a
*reward* and then plans, which produces an action everywhere — including states
the expert never visited.
"""

from __future__ import annotations

import numpy as np


def collect_demonstrations(env, expert_policy, starts, episodes_per_start: int,
                           rng: np.random.Generator):
    """Roll the expert out from each start. Returns `(state, action)` pairs."""
    pairs = []
    for start in starts:
        for _ in range(episodes_per_start):
            states, actions, _, _ = env.rollout(expert_policy, start, rng)
            pairs.extend(zip(states, actions, strict=True))
    return pairs


def behavioural_cloning_tabular(env, pairs, rng: np.random.Generator):
    """Per-state action counts; uniform where the expert was never seen.

    The uniform fallback is not a weak baseline choice — it is what tabular BC
    *is*. With no data at a state there is nothing to imitate, and the honest
    representation of that is a policy with no information.
    """
    counts = np.zeros((env.n_states, env.n_actions))
    for state, action in pairs:
        counts[state, action] += 1
    policy = np.full((env.n_states, env.n_actions), 1.0 / env.n_actions)
    seen = counts.sum(axis=1) > 0
    policy[seen] = counts[seen] / counts[seen].sum(axis=1, keepdims=True)
    return policy, seen


def state_features(env) -> np.ndarray:
    """Coordinates and local wall structure — enough to generalise between states."""
    features = []
    for r, c in env.cells:
        neighbours = [float(env.layout[nr][nc] == "#")
                      if 0 <= nr < env.rows and 0 <= nc < env.cols else 1.0
                      for nr, nc in ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))]
        features.append([r / env.rows, c / env.cols, *neighbours, 1.0])
    return np.asarray(features, dtype=np.float64)


def behavioural_cloning_features(env, pairs, rng: np.random.Generator,
                                 steps: int = 3000, lr: float = 0.2):
    """Multinomial logistic regression on state features.

    Included so that BC is not represented only by its tabular form: this one
    *does* produce an action at unvisited states. Whether that action is any
    good is the question item 3 asks — generalising is not the same as
    generalising correctly.
    """
    features = state_features(env)
    x = np.asarray([features[s] for s, _ in pairs])
    y = np.asarray([a for _, a in pairs])
    weights = np.zeros((features.shape[1], env.n_actions))

    for _ in range(steps):
        logits = x @ weights
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(len(y)), y] -= 1.0
        weights -= lr * (x.T @ probs) / len(y)

    logits = features @ weights
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    return probs / probs.sum(axis=1, keepdims=True)


def maxent_irl(env, pairs, starts, n_trajectories: int, gamma: float = 0.95,
               iterations: int = 200, lr: float = 0.1, horizon: int = 60):
    """Maximum-entropy IRL (Ziebart et al. 2008) with one-hot state features.

    The model says a trajectory is drawn with probability proportional to
    `exp(sum of its rewards)`, which makes the log-likelihood gradient the
    difference between the expert's expected feature counts and the learner's:

        grad = phi_expert - sum_s D(s) phi(s)

    where `D` is the expected state-visitation frequency under the current
    reward's soft-optimal policy. One-hot features mean "one reward per state",
    the most permissive choice available — and the one that makes the
    identifiability problem in the results unmistakable.

    Args:
        n_trajectories: how many demonstrations `pairs` came from. Expert
            visitation counts are per-trajectory, so they must match the scale
            of the forward pass, which starts from a single unit of mass.
    """
    expert_counts = np.zeros(env.n_states)
    for state, _ in pairs:
        expert_counts[state] += 1.0
    expert_counts /= n_trajectories

    start_dist = np.zeros(env.n_states)
    for s in starts:
        start_dist[s] += 1.0
    start_dist /= start_dist.sum()

    theta = np.zeros(env.n_states)
    for _ in range(iterations):
        _, policy = env.soft_value_iteration(theta, gamma=gamma)

        # Forward pass: expected state-visitation frequencies under that policy.
        mu = start_dist.copy()
        visitation = mu.copy()
        for _ in range(horizon):
            mu = np.einsum("s,sa,san->n", mu, policy, env.transitions)
            mu[env.terminal] = 0.0        # absorbing: stop accumulating past the goal
            visitation += mu
        theta += lr * (expert_counts - visitation)
        theta[env.walls] = 0.0

    return theta
