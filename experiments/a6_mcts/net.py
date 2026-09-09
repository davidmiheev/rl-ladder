"""A small policy-value network, and the evaluator that plugs it into MCTS."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class PolicyValueNet(nn.Module):
    """Two conv layers into separate policy and value heads.

    Input is two planes — the mover's stones and the opponent's — so the network
    always sees the position from the side to move, matching the board
    convention in `games.py`. That is what lets one network play both sides.
    """

    def __init__(self, rows: int, cols: int, n_actions: int, channels: int = 32):
        super().__init__()
        self.rows, self.cols = rows, cols
        self.trunk = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        hidden = channels * rows * cols
        self.policy_head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, n_actions))
        self.value_head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

    def encode(self, boards: np.ndarray) -> torch.Tensor:
        b = np.asarray(boards, dtype=np.float32).reshape(-1, self.rows, self.cols)
        return torch.from_numpy(np.stack([(b == 1), (b == -1)], axis=1).astype(np.float32))

    def forward(self, x: torch.Tensor):
        features = self.trunk(x)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


def net_evaluator(game, net: PolicyValueNet, use_value: bool, use_prior: bool,
                  rng: np.random.Generator):
    """Adapt the network to MCTS's `board -> (priors, value)` interface.

    `use_prior` and `use_value` are separate so item 3 can cross them: the
    network can supply the search prior, the leaf value, both, or neither.
    """
    from mcts import rollout_value

    def evaluate(board):
        if use_prior or use_value:
            with torch.no_grad():
                logits, value = net(net.encode(board))
            priors = torch.softmax(logits[0], dim=-1).numpy()
            leaf = float(value[0])
        if not use_prior:
            priors = np.zeros(game.n_actions, dtype=np.float32)
            moves = game.legal_moves(board)
            priors[moves] = 1.0 / len(moves)
        if not use_value:
            leaf = rollout_value(game, board, rng)
        return priors, leaf

    return evaluate
