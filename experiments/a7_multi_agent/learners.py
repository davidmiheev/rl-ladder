"""Independent Q-learners and self-play, plus an exact policy-vs-policy oracle.

The only difference between the two settings is whether the two seats share a
Q-table. That is a one-line change here *because* boards are stored from the
mover's perspective — the same convention that made one network serve both
players in A6 makes one table serve both seats.
"""

from __future__ import annotations

import game
import numpy as np


class QTable:
    """Tabular Q over board positions, defaulting to zeros for unseen states."""

    def __init__(self):
        self.q: dict[bytes, np.ndarray] = {}

    def values(self, board: np.ndarray) -> np.ndarray:
        key = board.tobytes()
        if key not in self.q:
            self.q[key] = np.zeros(game.N_ACTIONS, dtype=np.float64)
        return self.q[key]

    def act(self, board: np.ndarray, eps: float, rng: np.random.Generator) -> int:
        moves = game.legal_moves(board)
        if rng.random() < eps:
            return int(rng.choice(moves))
        values = self.values(board)[moves]
        best = moves[np.flatnonzero(values == values.max())]
        return int(rng.choice(best))

    def greedy_policy(self):
        """A deterministic policy function, uniform over ties, for the oracles."""
        def policy(board: np.ndarray) -> np.ndarray:
            moves = game.legal_moves(board)
            values = self.values(board)[moves]
            best = moves[np.flatnonzero(values == values.max())]
            probs = np.zeros(game.N_ACTIONS)
            probs[best] = 1.0 / len(best)
            return probs
        return policy

    def snapshot(self) -> QTable:
        clone = QTable()
        clone.q = {k: v.copy() for k, v in self.q.items()}
        return clone


def reachable_positions() -> list[np.ndarray]:
    """Every non-terminal position reachable by legal play, for random starts."""
    seen, out, frontier = set(), [], [game.initial()]
    while frontier:
        board = frontier.pop()
        key = board.tobytes()
        if key in seen:
            continue
        seen.add(key)
        if game.outcome(board) is not None:
            continue
        out.append(board)
        frontier.extend(game.play(board, int(m)) for m in game.legal_moves(board))
    return out


def train(shared: bool, episodes: int, seed: int, alpha: float = 0.1, gamma: float = 1.0,
          eps: float = 0.2, checkpoint_every: int = 0, random_starts: float = 0.0):
    """Play `episodes` self-play games of Tic-Tac-Toe with tabular Q-learning.

    Args:
        shared: `True` for self-play (one table for both seats), `False` for two
            independent learners.
        random_starts: probability of starting an episode from a random
            reachable position instead of the empty board.

    Returns:
        `(tables, checkpoints)` — the final table(s), and snapshots taken every
        `checkpoint_every` episodes for the non-stationarity analysis.
    """
    rng = np.random.default_rng(seed)
    if shared:
        table = QTable()
        tables = [table, table]
    else:
        tables = [QTable(), QTable()]
    starts = reachable_positions() if random_starts > 0 else []

    checkpoints = []
    for episode in range(1, episodes + 1):
        # Random starts exist to test whether self-play's weakness is a
        # *coverage* problem: seeding episodes from arbitrary reachable
        # positions forces the agents to learn states their own play never
        # produces.
        if starts and rng.random() < random_starts:
            board = starts[int(rng.integers(len(starts)))].copy()
        else:
            board = game.initial()
        turn = 0
        pending: list[tuple[np.ndarray, int] | None] = [None, None]

        while True:
            table = tables[turn]
            action = table.act(board, eps, rng)

            # Bootstrap the agent's *previous* move now that it can see the
            # state its opponent's reply produced. In a two-player game the
            # environment step spans the opponent's turn, which is exactly what
            # makes the transition depend on a policy that is itself changing.
            if pending[turn] is not None:
                prev_board, prev_action = pending[turn]
                target = gamma * table.values(board).max()
                q = table.values(prev_board)
                q[prev_action] += alpha * (target - q[prev_action])
            pending[turn] = (board.copy(), action)

            board = game.play(board, action)
            result = game.outcome(board)
            if result is not None:
                # `result` is for the player to move, i.e. the opponent of the
                # agent that just moved.
                rewards = [0.0, 0.0]
                rewards[1 - turn] = result
                rewards[turn] = -result
                for seat in (0, 1):
                    if pending[seat] is None:
                        continue
                    prev_board, prev_action = pending[seat]
                    q = tables[seat].values(prev_board)
                    q[prev_action] += alpha * (rewards[seat] - q[prev_action])
                break
            turn = 1 - turn

        if checkpoint_every and episode % checkpoint_every == 0:
            if shared:
                snap = tables[0].snapshot()
                checkpoints.append((episode, snap, snap))
            else:
                checkpoints.append((episode, tables[0].snapshot(), tables[1].snapshot()))

    return tables, checkpoints


def match_value(policy_first, policy_second, cache: dict | None = None) -> float:
    """Exact expected game value between two policies, from the first player's view.

    No sampling: the whole tree is small enough to take the expectation
    directly, so a difference of 0.01 is a real difference rather than noise.
    """
    cache = {} if cache is None else cache

    def value(board: np.ndarray, first_to_move: bool) -> float:
        key = (board.tobytes(), first_to_move)
        if key in cache:
            return cache[key]
        result = game.outcome(board)
        if result is not None:
            cache[key] = result if first_to_move else -result
            return cache[key]

        moves = game.legal_moves(board)
        policy = policy_first if first_to_move else policy_second
        probs = policy(board)[moves]
        total = probs.sum()
        probs = probs / total if total > 0 else np.full(len(moves), 1.0 / len(moves))
        out = float(sum(p * value(game.play(board, int(m)), not first_to_move)
                        for m, p in zip(moves, probs, strict=True)))
        cache[key] = out
        return out

    return value(game.initial(), True)
