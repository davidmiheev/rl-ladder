"""A3 item 4: DQN on Atari Pong, run on a GPU.

The item is listed as optional in the ladder, but A3's own results promoted it:
CartPole has two actions and short horizons, so neither Double DQN's
overestimation fix nor dueling's advantage stream had room to show an effect
there. Atari supplies both the value noise and the action space they need.

Nature-architecture CNN, standard Atari preprocessing, and the same four
switches as `dqn.py` so the arms differ only where intended.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import ale_py  # noqa: F401  — registers the ALE environments
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation


@dataclass
class Config:
    env_id: str = "ALE/Pong-v5"
    total_steps: int = 1_000_000
    buffer_size: int = 100_000
    batch_size: int = 32
    lr: float = 1e-4
    gamma: float = 0.99
    train_freq: int = 4
    target_update: int = 1_000       # in gradient steps, not env steps
    learning_starts: int = 20_000
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 250_000
    eval_every: int = 50_000
    eval_episodes: int = 5
    double: bool = False
    dueling: bool = False
    seed: int = 0


def make_env(config: Config, seed: int, render: bool = False) -> gym.Env:
    """Standard Atari pipeline: frameskip 4 with max-pooling, grayscale, 84x84, stack 4."""
    env = gym.make(config.env_id, frameskip=1)
    env = AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84,
                             terminal_on_life_loss=False, grayscale_obs=True, scale_obs=False)
    env = FrameStackObservation(env, stack_size=4)
    env.reset(seed=seed)
    return env


class NatureCNN(nn.Module):
    """Mnih et al. 2015 architecture, with an optional dueling head."""

    def __init__(self, n_actions: int, dueling: bool):
        super().__init__()
        self.dueling = dueling
        self.trunk = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(3136, 512), nn.ReLU(),
        )
        if dueling:
            self.value = nn.Linear(512, 1)
            self.advantage = nn.Linear(512, n_actions)
        else:
            self.head = nn.Linear(512, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.trunk(obs.float() / 255.0)
        if not self.dueling:
            return self.head(features)
        advantage = self.advantage(features)
        return self.value(features) + advantage - advantage.mean(dim=1, keepdim=True)


class ReplayBuffer:
    """Circular buffer storing only `obs`; the next state is the following slot.

    Halves memory against storing both: at 100k transitions of uint8 (4,84,84)
    that is 2.8 GB rather than 5.6 GB, which is what makes this fit alongside
    everything else on a 12 GB Colab VM.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, 4, 84, 84), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.position = 0

    def add(self, obs, action, reward, terminated) -> None:
        i = self.position
        self.obs[i], self.actions[i] = obs, action
        self.rewards[i], self.terminated[i] = reward, float(terminated)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        # Exclude the write head: slot position-1 has no valid successor yet.
        idx = rng.integers(self.size - 1, size=batch_size)
        idx = np.where(idx >= self.position - 1, idx + 1, idx) % self.size
        nxt = (idx + 1) % self.size
        return self.obs[idx], self.actions[idx], self.rewards[idx], self.obs[nxt], self.terminated[idx]


@torch.no_grad()
def evaluate(env: gym.Env, net: NatureCNN, device, episodes: int, eps: float = 0.05):
    """Greedy-with-slack evaluation, and the start-state value probe from item 2."""
    returns, predicted, actual = [], [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        q0 = net(torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0))
        predicted.append(float(q0.max()))
        total, discounted, discount, done = 0.0, 0.0, 1.0, False
        while not done:
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                q = net(torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0))
                action = int(q.argmax())
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            discounted += discount * np.sign(reward)   # matches the clipped training reward
            discount *= 0.99
            done = terminated or truncated
        returns.append(total)
        actual.append(discounted)
    return float(np.mean(returns)), float(np.mean(predicted)), float(np.mean(actual))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    args = parser.parse_args()

    config = Config(total_steps=args.steps, double=args.double, dueling=args.dueling, seed=args.seed)
    for name in ("learning_starts", "eval_every", "buffer_size"):
        if getattr(args, name) is not None:
            setattr(config, name, getattr(args, name))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(config), indent=2))
    metrics_file = out / "metrics.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    env = make_env(config, config.seed)
    eval_env = make_env(config, config.seed + 10_000)
    n_actions = int(env.action_space.n)

    online = NatureCNN(n_actions, config.dueling).to(device)
    target = NatureCNN(n_actions, config.dueling).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=config.lr)
    buffer = ReplayBuffer(config.buffer_size)

    # Resume support. The free Colab runtime this is built for gets reclaimed
    # without warning; the first attempt lost 800k steps that way. The replay
    # buffer is deliberately NOT checkpointed (2.8 GB), so a resume refills it
    # over `learning_starts` steps — cheap next to re-running from zero.
    checkpoint_path = out / "ckpt.pt"
    start_step = 0
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        online.load_state_dict(ckpt["online"])
        target.load_state_dict(ckpt["target"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"resumed from {checkpoint_path} at step {start_step}", flush=True)

    obs, _ = env.reset(seed=config.seed)
    episode_return, episode_returns, gradient_steps = 0.0, [], 0
    start = time.time()
    resume_floor = start_step + config.learning_starts

    for step in range(start_step + 1, config.total_steps + 1):
        eps = max(config.eps_end, config.eps_start +
                  (config.eps_end - config.eps_start) * step / config.eps_decay_steps)
        if step < max(config.learning_starts, resume_floor) or rng.random() < eps:
            action = int(rng.integers(n_actions))
        else:
            with torch.no_grad():
                q = online(torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0))
            action = int(q.argmax())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        episode_return += reward
        # Rewards are clipped to their sign for training; the raw game score is
        # what gets reported. Standard for Atari and the reason values stay in
        # a comparable range across games.
        buffer.add(np.asarray(obs), action, float(np.sign(reward)), terminated)
        obs = next_obs

        if terminated or truncated:
            episode_returns.append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset()

        if step >= max(config.learning_starts, resume_floor) and step % config.train_freq == 0:
            b_obs, b_act, b_rew, b_next, b_term = buffer.sample(config.batch_size, rng)
            b_obs = torch.as_tensor(b_obs, device=device)
            b_next = torch.as_tensor(b_next, device=device)
            b_act = torch.as_tensor(b_act, device=device)
            b_rew = torch.as_tensor(b_rew, device=device)
            b_term = torch.as_tensor(b_term, device=device)

            with torch.no_grad():
                if config.double:
                    best = online(b_next).argmax(dim=1, keepdim=True)
                    next_q = target(b_next).gather(1, best).squeeze(1)
                else:
                    next_q = target(b_next).max(dim=1).values
                y = b_rew + config.gamma * (1.0 - b_term) * next_q

            predicted = online(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(predicted, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            optimizer.step()

            gradient_steps += 1
            if gradient_steps % config.target_update == 0:
                target.load_state_dict(online.state_dict())

        if step % config.eval_every == 0:
            eval_return, pred, act = evaluate(eval_env, online, device, config.eval_episodes)
            record = {
                "step": step,
                "eval_return": eval_return,
                "train_return_last20": float(np.mean(episode_returns[-20:])) if episode_returns else None,
                "predicted_value": pred,
                "actual_discounted": act,
                "overestimation": pred - act,
                "epsilon": eps,
                "fps": step / (time.time() - start),
                "elapsed_min": (time.time() - start) / 60,
            }
            with metrics_file.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            torch.save({"online": online.state_dict(), "target": target.state_dict(),
                        "optimizer": optimizer.state_dict(), "step": step}, checkpoint_path)

    torch.save(online.state_dict(), out / "final.pt")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
