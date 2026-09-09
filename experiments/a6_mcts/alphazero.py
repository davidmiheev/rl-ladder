"""AlphaZero-style self-play training, and head-to-head match play.

The loop is the whole idea in three lines: search produces better moves than
the raw network, those searched moves become the network's training targets,
and the improved network makes the next search better. Search is the policy
improvement operator; the network is what remembers the improvement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from mcts import search
from net import PolicyValueNet, net_evaluator


@dataclass
class AZConfig:
    simulations: int = 50
    c_puct: float = 1.5
    root_noise: float = 0.25
    temperature_moves: int = 8      # sample proportionally for this many plies, then argmax
    games_per_iter: int = 60
    iterations: int = 15
    batch_size: int = 128
    lr: float = 2e-3
    train_steps: int = 60
    buffer_games: int = 400


def self_play_game(game, state_dict, config: AZConfig, seed: int):
    """One self-play game. Returns `(boards, policy targets, outcomes)`."""
    torch.set_num_threads(1)
    net = PolicyValueNet(game.rows, game.cols, game.n_actions)
    net.load_state_dict(state_dict)
    net.eval()
    rng = np.random.default_rng(seed)
    evaluate = net_evaluator(game, net, use_value=True, use_prior=True, rng=rng)

    board = game.initial()
    boards, policies = [], []
    ply = 0
    while game.outcome(board) is None:
        counts = search(game, board, evaluate, config.simulations, rng,
                        c=config.c_puct, selection="puct", root_noise=config.root_noise)
        policy = counts / counts.sum()
        boards.append(board.copy())
        policies.append(policy)
        if ply < config.temperature_moves:
            move = int(rng.choice(len(policy), p=policy))
        else:
            move = int(policy.argmax())
        board = game.play(board, move)
        ply += 1

    # The final `outcome` is for the player to move at the terminal position.
    # Walking backwards and flipping each ply gives every stored position its
    # own mover's result.
    result = game.outcome(board)
    outcomes = np.zeros(len(boards), dtype=np.float32)
    value = result
    for i in reversed(range(len(boards))):
        value = -value
        outcomes[i] = value
    return np.asarray(boards), np.asarray(policies, dtype=np.float32), outcomes


def train_step(net: PolicyValueNet, optimizer, boards, policies, outcomes,
               config: AZConfig, rng: np.random.Generator) -> float:
    """One gradient step on cross-entropy (policy) plus MSE (value)."""
    idx = rng.integers(len(boards), size=min(config.batch_size, len(boards)))
    x = net.encode(boards[idx])
    target_policy = torch.from_numpy(policies[idx])
    target_value = torch.from_numpy(outcomes[idx])

    logits, value = net(x)
    policy_loss = -(target_policy * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    value_loss = nn.functional.mse_loss(value, target_value)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def play_match(game, evaluate_a, evaluate_b, sims_a: int, sims_b: int, seed: int,
               a_first: bool, c_a: float = 1.5, c_b: float = 1.4,
               sel_a: str = "puct", sel_b: str = "uct") -> float:
    """One game between two searchers. Returns +1/0/-1 from A's point of view."""
    rng = np.random.default_rng(seed)
    board = game.initial()
    a_turn = a_first
    while game.outcome(board) is None:
        if a_turn:
            counts = search(game, board, evaluate_a, sims_a, rng, c=c_a, selection=sel_a)
        else:
            counts = search(game, board, evaluate_b, sims_b, rng, c=c_b, selection=sel_b)
        board = game.play(board, int(counts.argmax()))
        a_turn = not a_turn
    result = game.outcome(board)
    return result if a_turn else -result
