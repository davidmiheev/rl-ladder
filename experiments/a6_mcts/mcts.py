"""Monte Carlo tree search, with the prior and the leaf value as inputs.

One implementation covers plain UCT and AlphaZero-style PUCT: they differ only
in the selection formula and in where the leaf value comes from. Making both an
argument is what lets item 3 run a clean 2x2 — {uniform, learned} priors
crossed with {rollout, learned} values — with the search itself held fixed.
"""

from __future__ import annotations

import math

import numpy as np


class Node:
    __slots__ = ("prior", "visits", "value_sum", "children")

    def __init__(self, prior: float = 0.0):
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children: dict[int, Node] = {}

    @property
    def value(self) -> float:
        """Mean value from the perspective of the player to move at this node."""
        return self.value_sum / self.visits if self.visits else 0.0


def rollout_value(game, board: np.ndarray, rng: np.random.Generator) -> float:
    """Play uniformly at random to the end; return the result for the mover."""
    result = game.outcome(board)
    sign = 1.0
    while result is None:
        moves = game.legal_moves(board)
        board = game.play(board, int(rng.choice(moves)))
        sign = -sign
        result = game.outcome(board)
    return sign * result


def uniform_evaluator(game, rng: np.random.Generator):
    """Plain MCTS: no prior knowledge, leaf value from a random playout."""
    def evaluate(board):
        priors = np.zeros(game.n_actions, dtype=np.float32)
        moves = game.legal_moves(board)
        priors[moves] = 1.0 / len(moves)
        return priors, rollout_value(game, board, rng)
    return evaluate


def search(game, board: np.ndarray, evaluate, n_sims: int, rng: np.random.Generator,
           c: float = 1.4, selection: str = "puct", root_noise: float = 0.0) -> np.ndarray:
    """Run `n_sims` simulations from `board`; return the root visit counts.

    Args:
        evaluate: `board -> (priors, value)`, value from the mover's perspective.
        selection: `"uct"` for `Q + c*sqrt(ln N_parent / N_child)`, `"puct"` for
            AlphaZero's `Q + c*P*sqrt(N_parent)/(1+N_child)`.
        root_noise: Dirichlet noise weight at the root, for self-play
            exploration. Zero at evaluation time, where noise only adds error.
    """
    root = Node()
    priors, _ = evaluate(board)
    _expand(game, root, board, priors)
    if root_noise > 0 and root.children:
        moves = list(root.children)
        noise = rng.dirichlet([0.3] * len(moves))
        for move, n in zip(moves, noise, strict=True):
            root.children[move].prior = (1 - root_noise) * root.children[move].prior + root_noise * n

    for _ in range(n_sims):
        node, state, path = root, board, [root]
        # 1. Selection: descend by the tree policy while the node is expanded.
        while node.children:
            move = _select(node, c, selection)
            state = game.play(state, move)
            node = node.children[move]
            path.append(node)

        # 2. Expansion and evaluation, unless the game is already over.
        result = game.outcome(state)
        if result is None:
            priors, value = evaluate(state)
            _expand(game, node, state, priors)
        else:
            value = result

        # 3. Backup, flipping sign at every ply because the players alternate.
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value

    counts = np.zeros(game.n_actions, dtype=np.float64)
    for move, child in root.children.items():
        counts[move] = child.visits
    return counts


def _expand(game, node: Node, board: np.ndarray, priors: np.ndarray) -> None:
    moves = game.legal_moves(board)
    if len(moves) == 0:
        return
    legal = priors[moves]
    legal = legal / legal.sum() if legal.sum() > 0 else np.full(len(moves), 1.0 / len(moves))
    for move, p in zip(moves, legal, strict=True):
        node.children[int(move)] = Node(float(p))


def _select(node: Node, c: float, selection: str) -> int:
    best_score, best_move = -math.inf, -1
    for move, child in node.children.items():
        if selection == "uct":
            if child.visits == 0:
                score = math.inf
            else:
                # Q is the child's value for the child's mover, i.e. the
                # opponent, so negate it to score from this node's view.
                score = -child.value + c * math.sqrt(math.log(node.visits + 1) / child.visits)
        else:
            score = -child.value + c * child.prior * math.sqrt(node.visits) / (1 + child.visits)
        if score > best_score:
            best_score, best_move = score, move
    return best_move
