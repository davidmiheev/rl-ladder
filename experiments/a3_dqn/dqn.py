"""DQN and its three variations, with every stabiliser individually switchable.

The point of A3 is which fix addresses which part of the deadly triad, so
`replay` and `target_network` are flags rather than assumptions, and Double and
Dueling are flags on top of the same agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


@dataclass
class Config:
    """Hyperparameters. Defaults are tuned for CartPole-v1 on a CPU."""

    env_id: str = "CartPole-v1"
    total_steps: int = 30_000
    hidden: tuple[int, ...] = (128, 128)
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 10_000
    learning_starts: int = 500
    target_update: int = 200
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.2
    eval_every: int = 2_000
    eval_episodes: int = 5
    # The switches this phase is about:
    replay: bool = True
    target_network: bool = True
    double: bool = False
    dueling: bool = False


class QNetwork(nn.Module):
    """MLP action-value head, optionally split into value and advantage streams.

    Dueling computes `Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)`. Subtracting the
    mean is what makes the decomposition identifiable — without it, any constant
    could shift between the two streams.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, ...], dueling: bool):
        super().__init__()
        self.dueling = dueling

        layers: list[nn.Module] = []
        last = obs_dim
        for width in hidden:
            layers += [nn.Linear(last, width), nn.ReLU()]
            last = width
        self.trunk = nn.Sequential(*layers)

        if dueling:
            self.value = nn.Linear(last, 1)
            self.advantage = nn.Linear(last, n_actions)
        else:
            self.head = nn.Linear(last, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.trunk(obs)
        if not self.dueling:
            return self.head(features)
        advantage = self.advantage(features)
        return self.value(features) + advantage - advantage.mean(dim=1, keepdim=True)


class ReplayBuffer:
    """Fixed-capacity circular buffer of transitions, sampled uniformly."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.position = 0

    def add(self, obs, action, reward, next_obs, terminated) -> None:
        i = self.position
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.terminated[i] = next_obs, float(terminated)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        idx = rng.integers(self.size, size=batch_size)
        return (self.obs[idx], self.actions[idx], self.rewards[idx],
                self.next_obs[idx], self.terminated[idx])

    def latest(self) -> tuple[np.ndarray, ...]:
        """The single most recent transition — the no-replay ablation's 'batch'."""
        i = (self.position - 1) % self.capacity
        return (self.obs[i:i + 1], self.actions[i:i + 1], self.rewards[i:i + 1],
                self.next_obs[i:i + 1], self.terminated[i:i + 1])


@dataclass
class Metrics:
    """Everything a run records, all as plain lists so it serializes straight to JSON."""

    episode_returns: list[float] = field(default_factory=list)
    eval_steps: list[int] = field(default_factory=list)
    eval_returns: list[float] = field(default_factory=list)
    predicted_values: list[float] = field(default_factory=list)
    actual_returns: list[float] = field(default_factory=list)


def evaluate(env: gym.Env, net: QNetwork, config: Config, seed: int) -> tuple[float, float, float]:
    """Greedy rollouts.

    Returns:
        `(mean undiscounted return, mean predicted value at the start state,
        mean actual discounted return)`. The last two are the overestimation
        probe: `max_a Q(s_0, a)` against what the policy really collects.
    """
    undiscounted, predicted, actual = [], [], []
    for episode in range(config.eval_episodes):
        obs, _ = env.reset(seed=seed * 1000 + episode)
        with torch.no_grad():
            q_start = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        predicted.append(float(q_start.max()))

        total, discounted, discount, done = 0.0, 0.0, 1.0, False
        while not done:
            with torch.no_grad():
                action = int(net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).argmax())
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            discounted += discount * reward
            discount *= config.gamma
            done = terminated or truncated
        undiscounted.append(total)
        actual.append(discounted)
    return float(np.mean(undiscounted)), float(np.mean(predicted)), float(np.mean(actual))


def train(config: Config, seed: int) -> Metrics:
    """Train one agent. Every stabiliser is read from `config`, nothing is implicit."""
    torch.set_num_threads(1)  # runs are parallelised across processes, not within one
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    env = gym.make(config.env_id)
    eval_env = gym.make(config.env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)

    online = QNetwork(obs_dim, n_actions, config.hidden, config.dueling)
    target = QNetwork(obs_dim, n_actions, config.hidden, config.dueling)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=config.lr)

    buffer = ReplayBuffer(config.buffer_size if config.replay else 2, obs_dim)
    metrics = Metrics()

    obs, _ = env.reset(seed=seed)
    episode_return = 0.0
    decay_steps = max(1, int(config.eps_fraction * config.total_steps))

    for step in range(1, config.total_steps + 1):
        eps = max(config.eps_end,
                  config.eps_start + (config.eps_end - config.eps_start) * step / decay_steps)
        if rng.random() < eps:
            action = int(rng.integers(n_actions))
        else:
            with torch.no_grad():
                action = int(online(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).argmax())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        # Bootstrap through a time-limit truncation; a truncated episode has not
        # actually ended, and treating it as terminal teaches the agent that the
        # world stops at 500 steps.
        buffer.add(obs, action, reward, next_obs, terminated)
        episode_return += reward
        obs = next_obs

        if terminated or truncated:
            metrics.episode_returns.append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset()

        if step >= config.learning_starts:
            batch = buffer.sample(config.batch_size, rng) if config.replay else buffer.latest()
            batch_obs, batch_actions, batch_rewards, batch_next, batch_terminated = (
                torch.as_tensor(x) for x in batch
            )

            with torch.no_grad():
                bootstrap = online if not config.target_network else target
                if config.double:
                    # Select with the online net, evaluate with the target net.
                    best = online(batch_next).argmax(dim=1, keepdim=True)
                    next_q = bootstrap(batch_next).gather(1, best).squeeze(1)
                else:
                    next_q = bootstrap(batch_next).max(dim=1).values
                td_target = batch_rewards + config.gamma * (1.0 - batch_terminated) * next_q

            predicted = online(batch_obs).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(predicted, td_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if config.target_network and step % config.target_update == 0:
            target.load_state_dict(online.state_dict())

        if step % config.eval_every == 0:
            eval_return, predicted_value, actual_return = evaluate(eval_env, online, config, seed)
            metrics.eval_steps.append(step)
            metrics.eval_returns.append(eval_return)
            metrics.predicted_values.append(predicted_value)
            metrics.actual_returns.append(actual_return)

    env.close()
    eval_env.close()
    return metrics
