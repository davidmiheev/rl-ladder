"""Tic-Tac-Toe, an exact solver, and an exact exploitability oracle.

The game and solver are copied from A6 rather than imported — phases stay
self-contained by design (see docs/repo-layout.md), and copying makes the
additions here visible as a diff. What is new is `exploitability`, which is
what turns "the agents are chasing each other" from a story into a number.

Board convention, as in A6: +1 is the player to move, -1 the opponent, and
`play` flips the sign so every position is seen from the mover's side.
"""

from __future__ import annotations

import numpy as np

LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
         (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
N_ACTIONS = 9


def initial() -> np.ndarray:
    return np.zeros(9, dtype=np.int8)


def legal_moves(board: np.ndarray) -> np.ndarray:
    return np.flatnonzero(board == 0)


def play(board: np.ndarray, move: int) -> np.ndarray:
    nxt = board.copy()
    nxt[move] = 1
    return -nxt


def outcome(board: np.ndarray) -> float | None:
    """`None` if unfinished, else the result for the player *to move*."""
    for a, b, c in LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return -1.0 if board[a] == -1 else 1.0
    return 0.0 if not (board == 0).any() else None


def solve(board: np.ndarray, cache: dict) -> tuple[float, list[int]]:
    """Exact minimax. Returns `(value, every optimal move)` for the mover."""
    key = board.tobytes()
    if key in cache:
        return cache[key]
    result = outcome(board)
    if result is not None:
        cache[key] = (result, [])
        return cache[key]
    best, best_moves = -2.0, []
    for move in legal_moves(board):
        value = -solve(play(board, move), cache)[0]
        if value > best:
            best, best_moves = value, [int(move)]
        elif value == best:
            best_moves.append(int(move))
    cache[key] = (best, best_moves)
    return cache[key]


def exploitability(policy, pi_moves_first: bool, cache: dict | None = None) -> float:
    """Value a *perfect* opponent can extract against `policy`.

    Args:
        policy: `board -> probability vector over the 9 cells`, from the
            perspective of the player to move.
        pi_moves_first: whether the policy plays first.

    Returns:
        The game value **to the exploiter**: `0.0` if the policy is
        unexploitable (Tic-Tac-Toe's perfect play draws), up to `+1.0` if a
        perfect opponent can force a win.

    This is the honest measure of a policy in a two-player game. A win rate
    against a *particular* opponent says only that the two are mismatched;
    exploitability says how far the policy is from the equilibrium, and it is
    computable exactly here.
    """
    cache = {} if cache is None else cache

    def value(board: np.ndarray, pi_turn: bool) -> float:
        key = (board.tobytes(), pi_turn)
        if key in cache:
            return cache[key]

        result = outcome(board)
        if result is not None:
            # `result` is for the player to move; flip it when that is the
            # policy, since the return is always from the exploiter's side.
            cache[key] = -result if pi_turn else result
            return cache[key]

        moves = legal_moves(board)
        if pi_turn:
            probs = policy(board)[moves]
            total = probs.sum()
            probs = probs / total if total > 0 else np.full(len(moves), 1.0 / len(moves))
            out = float(sum(p * value(play(board, int(m)), False)
                            for m, p in zip(moves, probs, strict=True)))
        else:
            out = max(value(play(board, int(m)), True) for m in moves)
        cache[key] = out
        return out

    return value(initial(), pi_moves_first)
