"""Continuous control: DDPG, TD3 and SAC as one implementation with switches.

TD3 is DDPG plus three separable fixes — twin critics, delayed actor updates,
and target policy smoothing — so each is a flag rather than a separate file,
and item 2 can turn them off one at a time. SAC changes the actor from
deterministic to a squashed Gaussian and adds an entropy term to both losses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


@dataclass
class Config:
    env_id: str = "Pendulum-v1"
    algo: str = "ddpg"                  # ddpg | td3 | sac
    total_steps: int = 30_000
    hidden: tuple[int, ...] = (256, 256)
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005                  # polyak averaging for target nets
    batch_size: int = 256
    buffer_size: int = 100_000
    learning_starts: int = 1_000
    exploration_noise: float = 0.1      # DDPG/TD3 behaviour noise, in action units
    # TD3's three fixes, individually switchable:
    twin_critics: bool = False
    policy_delay: int = 1               # 2 in TD3, 1 disables the delay
    target_noise: float = 0.0           # 0.2 in TD3, 0 disables smoothing
    target_noise_clip: float = 0.5
    # SAC:
    alpha: float = 0.2                  # entropy temperature
    eval_every: int = 2_000
    eval_episodes: int = 5


def mlp(sizes: tuple[int, ...], out_dim: int, out_activation=None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
    layers.append(nn.Linear(sizes[-1], out_dim))
    if out_activation is not None:
        layers.append(out_activation)
    return nn.Sequential(*layers)


class DeterministicActor(nn.Module):
    """DDPG/TD3 actor: a single action, tanh-squashed to the action range."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...], act_limit: float):
        super().__init__()
        self.net = mlp((obs_dim,) + hidden, act_dim, nn.Tanh())
        self.act_limit = act_limit

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.act_limit * self.net(obs)


class GaussianActor(nn.Module):
    """SAC actor: a squashed Gaussian, reparameterised so the sample is differentiable."""

    LOG_STD_RANGE = (-20.0, 2.0)

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...], act_limit: float):
        super().__init__()
        self.trunk = mlp((obs_dim,) + hidden[:-1], hidden[-1], nn.ReLU())
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        """Returns `(action, log_prob)`; `log_prob` is `None` when deterministic."""
        features = self.trunk(obs)
        mu = self.mu(features)
        if deterministic:
            return self.act_limit * torch.tanh(mu), None

        log_std = self.log_std(features).clamp(*self.LOG_STD_RANGE)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        pre_tanh = normal.rsample()
        action = torch.tanh(pre_tanh)

        # Change-of-variables correction for the tanh squash: the density of the
        # squashed variable picks up -sum(log(1 - tanh^2)). Without it the
        # entropy term is wrong and alpha means nothing.
        log_prob = normal.log_prob(pre_tanh).sum(-1)
        log_prob -= (2 * (np.log(2) - pre_tanh - nn.functional.softplus(-2 * pre_tanh))).sum(-1)
        return self.act_limit * action, log_prob


class Critic(nn.Module):
    """One or two Q(s, a) heads. Two is TD3's and SAC's overestimation fix."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...], twin: bool):
        super().__init__()
        self.q1 = mlp((obs_dim + act_dim,) + hidden, 1)
        self.q2 = mlp((obs_dim + act_dim,) + hidden, 1) if twin else None

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        q1 = self.q1(x).squeeze(-1)
        return q1, (self.q2(x).squeeze(-1) if self.q2 is not None else None)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.size, self.position = capacity, 0, 0

    def add(self, obs, action, reward, next_obs, terminated) -> None:
        i = self.position
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.terminated[i] = next_obs, float(terminated)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(self.size, size=batch_size)
        return tuple(torch.as_tensor(x[idx]) for x in
                     (self.obs, self.actions, self.rewards, self.next_obs, self.terminated))


@dataclass
class Agent:
    """The networks, their targets and their optimisers, in one place.

    Split out of `train` so that a different outer loop can drive the same
    update rather than a copy of it: C3 runs these algorithms against a
    market-making environment where one episode costs ninety seconds and the
    loop has to cycle over trading days, but the algorithm must stay the one
    A5 measured.
    """

    actor: nn.Module
    critic: nn.Module
    actor_target: nn.Module
    critic_target: nn.Module
    actor_opt: torch.optim.Optimizer
    critic_opt: torch.optim.Optimizer
    act_limit: float
    is_sac: bool


def build_agent(config: Config, obs_dim: int, act_dim: int, act_limit: float) -> Agent:
    is_sac = config.algo == "sac"
    actor_cls = GaussianActor if is_sac else DeterministicActor
    actor = actor_cls(obs_dim, act_dim, config.hidden, act_limit)
    critic = Critic(obs_dim, act_dim, config.hidden, config.twin_critics)
    actor_target = actor_cls(obs_dim, act_dim, config.hidden, act_limit)
    critic_target = Critic(obs_dim, act_dim, config.hidden, config.twin_critics)
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())
    return Agent(
        actor=actor,
        critic=critic,
        actor_target=actor_target,
        critic_target=critic_target,
        actor_opt=torch.optim.Adam(actor.parameters(), lr=config.lr),
        critic_opt=torch.optim.Adam(critic.parameters(), lr=config.lr),
        act_limit=act_limit,
        is_sac=is_sac,
    )


def select_action(agent: Agent, obs, config: Config, rng: np.random.Generator, act_dim: int):
    """The behaviour policy: SAC samples, DDPG and TD3 add Gaussian noise."""
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        if agent.is_sac:
            action, _ = agent.actor(obs_t)
            return action.squeeze(0).numpy()
        action = agent.actor(obs_t).squeeze(0).numpy()
    noisy = action + config.exploration_noise * agent.act_limit * rng.normal(size=act_dim)
    return np.clip(noisy, -agent.act_limit, agent.act_limit)


def update_step(config: Config, agent: Agent, batch, step: int) -> float:
    """One critic update, plus actor and target updates every `policy_delay`."""
    b_obs, b_act, b_rew, b_next, b_term = batch
    act_limit = agent.act_limit

    with torch.no_grad():
        if agent.is_sac:
            next_action, next_logp = agent.actor_target(b_next)
        else:
            next_action = agent.actor_target(b_next)
            if config.target_noise:
                # Target policy smoothing: a sharp critic error at one
                # action cannot be exploited if the target is averaged
                # over a small neighbourhood of it.
                noise = (config.target_noise * act_limit * torch.randn_like(next_action)
                         ).clamp(-config.target_noise_clip * act_limit,
                                 config.target_noise_clip * act_limit)
                next_action = (next_action + noise).clamp(-act_limit, act_limit)

        q1_t, q2_t = agent.critic_target(b_next, next_action)
        next_q = q1_t if q2_t is None else torch.min(q1_t, q2_t)
        if agent.is_sac:
            next_q = next_q - config.alpha * next_logp
        target_q = b_rew + config.gamma * (1.0 - b_term) * next_q

    q1, q2 = agent.critic(b_obs, b_act)
    critic_loss = nn.functional.mse_loss(q1, target_q)
    if q2 is not None:
        critic_loss = critic_loss + nn.functional.mse_loss(q2, target_q)
    agent.critic_opt.zero_grad()
    critic_loss.backward()
    agent.critic_opt.step()

    if step % config.policy_delay == 0:
        if agent.is_sac:
            new_action, logp = agent.actor(b_obs)
            q1_pi, q2_pi = agent.critic(b_obs, new_action)
            q_pi = q1_pi if q2_pi is None else torch.min(q1_pi, q2_pi)
            actor_loss = (config.alpha * logp - q_pi).mean()
        else:
            q1_pi, _ = agent.critic(b_obs, agent.actor(b_obs))
            actor_loss = -q1_pi.mean()
        agent.actor_opt.zero_grad()
        actor_loss.backward()
        agent.actor_opt.step()

        with torch.no_grad():
            for net, target_net in (
                (agent.actor, agent.actor_target),
                (agent.critic, agent.critic_target),
            ):
                for p, p_t in zip(net.parameters(), target_net.parameters(), strict=True):
                    p_t.mul_(1 - config.tau).add_(config.tau * p)

    return float(critic_loss.detach())


@dataclass
class Metrics:
    eval_steps: list[int] = field(default_factory=list)
    eval_returns: list[float] = field(default_factory=list)
    predicted_q: list[float] = field(default_factory=list)
    actual_return: list[float] = field(default_factory=list)
    critic_loss: list[float] = field(default_factory=list)


@torch.no_grad()
def evaluate(env: gym.Env, actor, critic, config: Config, seed: int,
             action_noise: float = 0.0, obs_noise: float = 0.0):
    """Deterministic-policy rollouts, plus the value-bias probe from A3.

    `predicted` is Q(s0, a0) under the learned critic; `actual` is the
    discounted return the policy really collects from s0. Their difference is
    the same overestimation measure used in A3, which is what TD3's twin
    critics are meant to fix.
    """
    returns, predicted, actual = [], [], []
    for episode in range(config.eval_episodes):
        obs, _ = env.reset(seed=seed * 1000 + episode)
        first = True
        total, discounted, discount, done = 0.0, 0.0, 1.0, False
        while not done:
            noisy = obs + obs_noise * np.random.randn(*np.shape(obs)) if obs_noise else obs
            obs_t = torch.as_tensor(np.asarray(noisy), dtype=torch.float32).unsqueeze(0)
            if config.algo == "sac":
                action, _ = actor(obs_t, deterministic=True)
            else:
                action = actor(obs_t)
            if first:
                q1, _ = critic(obs_t, action)
                predicted.append(float(q1))
                first = False
            act = action.squeeze(0).numpy()
            if action_noise:
                act = act + action_noise * np.random.randn(*act.shape)
            act = np.clip(act, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, _ = env.step(act)
            total += reward
            discounted += discount * reward
            discount *= config.gamma
            done = terminated or truncated
        returns.append(total)
        actual.append(discounted)
    return float(np.mean(returns)), float(np.mean(predicted)), float(np.mean(actual))


def train(config: Config, seed: int):
    """Train one agent. Returns `(Metrics, actor, critic, eval_env)`."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    env = gym.make(config.env_id)
    eval_env = gym.make(config.env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    act_limit = float(env.action_space.high[0])

    agent = build_agent(config, obs_dim, act_dim, act_limit)
    actor, critic = agent.actor, agent.critic
    buffer = ReplayBuffer(config.buffer_size, obs_dim, act_dim)
    metrics = Metrics()

    obs, _ = env.reset(seed=seed)
    # `reset(seed=...)` seeds the environment, not the action space, so the
    # `learning_starts` exploration phase below was drawing from an unseeded
    # RNG and two runs of the same seed did not agree.  Seeding it explicitly
    # is what makes a run reproducible from its seed alone.
    env.action_space.seed(seed)
    for step in range(1, config.total_steps + 1):
        if step < config.learning_starts:
            action = env.action_space.sample()
        else:
            action = select_action(agent, obs, config, rng, act_dim)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

        if step >= config.learning_starts:
            batch = buffer.sample(config.batch_size, rng)
            loss = update_step(config, agent, batch, step)
            if step % 100 == 0:
                metrics.critic_loss.append(loss)

        if step % config.eval_every == 0:
            ret, pred, act = evaluate(eval_env, actor, critic, config, seed)
            metrics.eval_steps.append(step)
            metrics.eval_returns.append(ret)
            metrics.predicted_q.append(pred)
            metrics.actual_return.append(act)

    env.close()
    return metrics, actor, critic, eval_env
