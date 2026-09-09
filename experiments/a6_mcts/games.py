"""Two perfect-information games, plus an exact solver for the small one.

Tic-Tac-Toe is here because it can be solved outright, which turns item 1's
"verify convergence to near-optimal play" into a check against ground truth
rather than an assertion. Connect Four is the one worth learning on.

Convention: `board` holds +1 for the player to move and -1 for the opponent,
so a position is always seen from the mover's perspective and the same network
serves both players. `play` flips the board's sign, which is what keeps that
invariant.
"""

from __future__ import annotations

import numpy as np


class TicTacToe:
    rows, cols, n_actions = 3, 3, 9
    name = "tictactoe"

    @staticmethod
    def initial() -> np.ndarray:
        return np.zeros(9, dtype=np.int8)

    @staticmethod
    def legal_moves(board: np.ndarray) -> np.ndarray:
        return np.flatnonzero(board == 0)

    @staticmethod
    def play(board: np.ndarray, move: int) -> np.ndarray:
        """Place the mover's stone, then flip perspective for the opponent."""
        nxt = board.copy()
        nxt[move] = 1
        return -nxt

    LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
             (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))

    @classmethod
    def outcome(cls, board: np.ndarray) -> float | None:
        """`None` if unfinished, else the result for the player *to move*.

        A win can only ever belong to the player who just moved, so a decided
        position is always `-1` from the mover's point of view.
        """
        for a, b, c in cls.LINES:
            if board[a] != 0 and board[a] == board[b] == board[c]:
                return -1.0 if board[a] == -1 else 1.0
        return 0.0 if not (board == 0).any() else None


class Connect4:
    rows, cols, n_actions = 6, 7, 7
    name = "connect4"

    @staticmethod
    def initial() -> np.ndarray:
        return np.zeros(42, dtype=np.int8)

    @classmethod
    def legal_moves(cls, board: np.ndarray) -> np.ndarray:
        grid = board.reshape(cls.rows, cls.cols)
        return np.flatnonzero(grid[0] == 0)

    @classmethod
    def play(cls, board: np.ndarray, move: int) -> np.ndarray:
        grid = board.reshape(cls.rows, cls.cols).copy()
        row = np.max(np.flatnonzero(grid[:, move] == 0))
        grid[row, move] = 1
        return -grid.reshape(-1)

    @classmethod
    def outcome(cls, board: np.ndarray) -> float | None:
        grid = board.reshape(cls.rows, cls.cols)
        for player in (1, -1):
            mask = grid == player
            # Four in a row, horizontally / vertically / both diagonals.
            if (mask[:, :-3] & mask[:, 1:-2] & mask[:, 2:-1] & mask[:, 3:]).any():
                return -1.0 if player == -1 else 1.0
            if (mask[:-3] & mask[1:-2] & mask[2:-1] & mask[3:]).any():
                return -1.0 if player == -1 else 1.0
            if (mask[:-3, :-3] & mask[1:-2, 1:-2] & mask[2:-1, 2:-1] & mask[3:, 3:]).any():
                return -1.0 if player == -1 else 1.0
            if (mask[3:, :-3] & mask[2:-1, 1:-2] & mask[1:-2, 2:-1] & mask[:-3, 3:]).any():
                return -1.0 if player == -1 else 1.0
        return 0.0 if (grid[0] != 0).all() else None


def solve(game, board: np.ndarray, cache: dict | None = None) -> tuple[float, list[int]]:
    """Exact minimax with memoisation.

    Returns:
        `(value, optimal_moves)` — the game-theoretic value for the player to
        move, and every move achieving it. The move *list* matters: optimal play
        is often not unique, so scoring an agent against a single "best move"
        would understate it.
    """
    cache = {} if cache is None else cache
    key = board.tobytes()
    if key in cache:
        return cache[key]

    result = game.outcome(board)
    if result is not None:
        cache[key] = (result, [])
        return cache[key]

    best, best_moves = -2.0, []
    for move in game.legal_moves(board):
        value = -solve(game, game.play(board, move), cache)[0]
        if value > best:
            best, best_moves = value, [int(move)]
        elif value == best:
            best_moves.append(int(move))
    cache[key] = (best, best_moves)
    return cache[key]
