"""Policy gradients: REINFORCE, a learned baseline, GAE, and A2C.

The whole family is one estimator with progressively better advantage
estimates, so `Config` selects between them by flags rather than by separate
implementations — `baseline=False` is REINFORCE, `baseline=True` adds a critic,
and `gae_lambda` interpolates between a one-step residual and the full return.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class Config:
    """Hyperparameters. Defaults suit CartPole-v1 on a CPU."""

    env_id: str = "CartPole-v1"
    total_steps: int = 30_000
    hidden: tuple[int, ...] = (128, 128)
    lr: float = 3e-3
    value_lr: float = 1e-3
    value_epochs: int = 40
    gamma: float = 0.99
    batch_steps: int = 1_000
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    eval_every: int = 2_000
    eval_episodes: int = 5
    # Which member of the family this is:
    baseline: bool = False
    gae_lambda: float | None = None
    # A2C only:
    n_envs: int = 8
    n_steps: int = 5


class ActorCritic(nn.Module):
    """Categorical policy with an optional value head sharing no trunk.

    Separate trunks rather than a shared one: with a shared body the value loss
    and the policy loss fight over the same features, and the balance becomes a
    hyperparameter this phase is not about.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, ...], baseline: bool):
        super().__init__()
        self.policy = self._mlp(obs_dim, n_actions, hidden)
        self.value = self._mlp(obs_dim, 1, hidden) if baseline else None

    @staticmethod
    def _mlp(inp: int, out: int, hidden: tuple[int, ...]) -> nn.Sequential:
        layers: list[nn.Module] = []
        last = inp
        for width in hidden:
            layers += [nn.Linear(last, width), nn.Tanh()]
            last = width
        layers.append(nn.Linear(last, out))
        return nn.Sequential(*layers)

    def distribution(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.policy(obs))

    def values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value(obs).squeeze(-1)


def discounted_returns(rewards: list[float], gamma: float, bootstrap: float = 0.0) -> np.ndarray:
    """Returns-to-go, `G_t = r_t + gamma * G_{t+1}`, ending at `bootstrap`."""
    out = np.zeros(len(rewards), dtype=np.float32)
    running = bootstrap
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def gae(rewards: np.ndarray, values: np.ndarray, last_value: float, gamma: float, lam: float
        ) -> np.ndarray:
    """Generalised advantage estimate over one trajectory segment.

    `A_t = sum_l (gamma * lam)^l * delta_{t+l}` with
    `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)`. At `lam = 0` this is the
    one-step TD residual (low variance, biased by whatever error `V` carries);
    at `lam = 1` it telescopes into the Monte Carlo return minus `V(s_t)`
    (unbiased, high variance).
    """
    advantages = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    next_value = last_value
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        advantages[t] = running
        next_value = values[t]
    return advantages


@dataclass
class Metrics:
    eval_steps: list[int] = field(default_factory=list)
    eval_returns: list[float] = field(default_factory=list)
    train_returns: list[float] = field(default_factory=list)
    advantage_std: list[float] = field(default_factory=list)
    explained_variance: list[float] = field(default_factory=list)


def evaluate(env: gym.Env, net: ActorCritic, episodes: int, seed: int) -> float:
    """Mean return of the greedy (argmax) policy — the stochastic policy's mode."""
    totals = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed * 1000 + episode)
        total, done = 0.0, False
        while not done:
            with torch.no_grad():
                action = int(net.policy(torch.as_tensor(obs, dtype=torch.float32)).argmax())
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        totals.append(total)
    return float(np.mean(totals))


def train_reinforce(config: Config, seed: int) -> Metrics:
    """REINFORCE, optionally with a learned baseline and GAE.

    Collects whole episodes until `batch_steps` is reached, then makes one
    update. Episodes are never cut mid-way, so every return is a real return.
    """
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    env = gym.make(config.env_id)
    eval_env = gym.make(config.env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)

    net = ActorCritic(obs_dim, n_actions, config.hidden, config.baseline)
    # Separate optimizers: the policy takes one on-policy step per batch, but the
    # critic is a supervised regression and needs many. With a single shared step
    # the value function never fits, the baseline subtracts noise, and the whole
    # point of the item is lost.
    policy_optimizer = torch.optim.Adam(net.policy.parameters(), lr=config.lr)
    value_optimizer = (
        torch.optim.Adam(net.value.parameters(), lr=config.value_lr) if config.baseline else None
    )

    metrics = Metrics()
    obs, _ = env.reset(seed=seed)
    step, next_eval = 0, config.eval_every

    while step < config.total_steps:
        batch_obs, batch_actions, batch_advantages, batch_returns = [], [], [], []
        episode_returns = []
        collected = 0

        while collected < config.batch_steps:
            ep_obs, ep_actions, ep_rewards = [], [], []
            done = False
            while not done:
                with torch.no_grad():
                    action = int(net.distribution(torch.as_tensor(obs, dtype=torch.float32)).sample())
                next_obs, reward, terminated, truncated, _ = env.step(action)
                ep_obs.append(obs)
                ep_actions.append(action)
                ep_rewards.append(float(reward))
                obs = next_obs
                done = terminated or truncated
                step += 1
                collected += 1
            # A time-limit truncation is not a real ending, so bootstrap through
            # it when a critic is available; a true termination has value 0.
            bootstrap = 0.0
            if truncated and config.baseline:
                with torch.no_grad():
                    bootstrap = float(net.values(torch.as_tensor(obs, dtype=torch.float32)))
            returns = discounted_returns(ep_rewards, config.gamma, bootstrap)

            ep_obs_t = torch.as_tensor(np.asarray(ep_obs), dtype=torch.float32)
            if config.baseline:
                with torch.no_grad():
                    values = net.values(ep_obs_t).numpy()
                if config.gae_lambda is None:
                    advantages = returns - values
                else:
                    advantages = gae(np.asarray(ep_rewards, dtype=np.float32), values,
                                     bootstrap, config.gamma, config.gae_lambda)
            else:
                advantages = returns

            batch_obs.append(ep_obs_t)
            batch_actions.append(torch.as_tensor(ep_actions))
            batch_advantages.append(torch.as_tensor(advantages))
            batch_returns.append(torch.as_tensor(returns))
            episode_returns.append(sum(ep_rewards))
            obs, _ = env.reset()

        obs_t = torch.cat(batch_obs)
        actions_t = torch.cat(batch_actions)
        advantages_t = torch.cat(batch_advantages)
        returns_t = torch.cat(batch_returns)

        # Recorded before normalisation: this is the quantity lambda controls,
        # and normalising would erase exactly the effect being measured.
        metrics.advantage_std.append(float(advantages_t.std()))
        if config.baseline:
            with torch.no_grad():
                predicted = net.values(obs_t)
            var = float(returns_t.var())
            metrics.explained_variance.append(
                float(1.0 - (returns_t - predicted).var() / var) if var > 0 else 0.0
            )

        normalised = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        dist = net.distribution(obs_t)
        loss = -(dist.log_prob(actions_t) * normalised).mean() - config.entropy_coef * dist.entropy().mean()

        policy_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.policy.parameters(), 0.5)
        policy_optimizer.step()

        if config.baseline:
            for _ in range(config.value_epochs):
                value_optimizer.zero_grad()
                nn.functional.mse_loss(net.values(obs_t), returns_t).backward()
                value_optimizer.step()

        metrics.train_returns.append(float(np.mean(episode_returns)))
        if step >= next_eval:
            metrics.eval_steps.append(step)
            metrics.eval_returns.append(evaluate(eval_env, net, config.eval_episodes, seed))
            next_eval += config.eval_every

    env.close()
    eval_env.close()
    del rng
    return metrics


def train_a2c(config: Config, seed: int) -> Metrics:
    """Synchronous advantage actor-critic over `n_envs` parallel environments.

    The parallel environments are the on-policy answer to the problem replay
    solved in A3: a batch drawn from `n_envs` independent environments at the
    same timestep is decorrelated without storing anything, so the data stays
    on-policy. Updates every `n_steps`, bootstrapping the truncated tail from
    the critic.
    """
    torch.set_num_threads(1)
    torch.manual_seed(seed)

    envs = gym.make_vec(config.env_id, num_envs=config.n_envs, vectorization_mode="sync")
    eval_env = gym.make(config.env_id)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    n_actions = int(envs.single_action_space.n)

    net = ActorCritic(obs_dim, n_actions, config.hidden, baseline=True)
    policy_optimizer = torch.optim.Adam(net.policy.parameters(), lr=config.lr)
    value_optimizer = torch.optim.Adam(net.value.parameters(), lr=config.value_lr)

    metrics = Metrics()
    obs, _ = envs.reset(seed=seed)
    episode_returns = np.zeros(config.n_envs)
    recent: list[float] = []
    step, next_eval = 0, config.eval_every

    while step < config.total_steps:
        obs_buf, action_buf, reward_buf, done_buf, value_buf = [], [], [], [], []

        for _ in range(config.n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                dist = net.distribution(obs_t)
                actions = dist.sample()
                values = net.values(obs_t)
            next_obs, rewards, terminated, truncated, _ = envs.step(actions.numpy())

            obs_buf.append(obs_t)
            action_buf.append(actions)
            reward_buf.append(np.asarray(rewards, dtype=np.float32))
            # Only a true termination has value zero; a truncation is bootstrapped.
            done_buf.append(np.asarray(terminated, dtype=np.float32))
            value_buf.append(values.numpy())

            episode_returns += np.asarray(rewards)
            for i, finished in enumerate(np.logical_or(terminated, truncated)):
                if finished:
                    recent.append(float(episode_returns[i]))
                    episode_returns[i] = 0.0
            obs = next_obs
            step += config.n_envs

        with torch.no_grad():
            last_values = net.values(torch.as_tensor(obs, dtype=torch.float32)).numpy()

        # GAE per environment column, then flatten.
        rewards_arr = np.asarray(reward_buf)
        values_arr = np.asarray(value_buf)
        dones_arr = np.asarray(done_buf)
        advantages = np.zeros_like(rewards_arr)
        running = np.zeros(config.n_envs, dtype=np.float32)
        next_value = last_values
        lam = 0.95 if config.gae_lambda is None else config.gae_lambda
        for t in reversed(range(config.n_steps)):
            mask = 1.0 - dones_arr[t]
            delta = rewards_arr[t] + config.gamma * next_value * mask - values_arr[t]
            running = delta + config.gamma * lam * mask * running
            advantages[t] = running
            next_value = values_arr[t]
        returns_arr = advantages + values_arr

        obs_t = torch.cat(obs_buf)
        actions_t = torch.cat(action_buf)
        advantages_t = torch.as_tensor(advantages.reshape(-1))
        returns_t = torch.as_tensor(returns_arr.reshape(-1))

        metrics.advantage_std.append(float(advantages_t.std()))
        var = float(returns_t.var())
        predicted = torch.as_tensor(values_arr.reshape(-1))
        metrics.explained_variance.append(
            float(1.0 - (returns_t - predicted).var() / var) if var > 0 else 0.0
        )

        normalised = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        dist = net.distribution(obs_t)
        loss = -(dist.log_prob(actions_t) * normalised).mean() - config.entropy_coef * dist.entropy().mean()

        policy_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.policy.parameters(), 0.5)
        policy_optimizer.step()

        value_optimizer.zero_grad()
        nn.functional.mse_loss(net.values(obs_t), returns_t).backward()
        nn.utils.clip_grad_norm_(net.value.parameters(), 0.5)
        value_optimizer.step()

        if recent:
            metrics.train_returns.append(float(np.mean(recent[-20:])))
        if step >= next_eval:
            metrics.eval_steps.append(step)
            metrics.eval_returns.append(evaluate(eval_env, net, config.eval_episodes, seed))
            next_eval += config.eval_every

    envs.close()
    eval_env.close()
    return metrics
