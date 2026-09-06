"""GridWorld: one layout language for every A2 environment.

A layout is a list of equal-length strings:

    '.'  free cell          'S'  start
    '#'  wall               'G'  goal, terminal
    'C'  cliff — large penalty and teleport back to start, NOT terminal

Each grid exposes both a model (`to_mdp()`, for the DP solvers) and a
`reset`/`step` interface (for the TD agents), so item 1 and item 2 run on the
same dynamics rather than on two implementations that might disagree.
"""

from __future__ import annotations

import numpy as np
from mdp import TabularMDP

# (row delta, col delta): up, right, down, left
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_CHARS = "^>v<"

SIMPLE_GRID = [
    "S....",
    ".###.",
    ".....",
    ".###.",
    "....G",
]

# Sutton & Barto example 6.6.
CLIFF = [
    "............",
    "............",
    "............",
    "SCCCCCCCCCCG",
]

# Sutton & Barto figure 8.2, the Dyna maze. Optimal path is 14 steps.
DYNA_MAZE = [
    ".......#G",
    "..#....#.",
    "S.#....#.",
    "..#......",
    ".....#...",
    ".........",
]


class GridWorld:
    """A grid MDP with optional slip.

    Args:
        layout (`list[str]`):
            Rows of the layout language above.
        slip (`float`, *optional*, defaults to `0.0`):
            Probability of moving perpendicular to the chosen direction, split
            evenly between the two perpendiculars.
        step_reward (`float`, *optional*, defaults to `-1.0`):
            Reward for any transition out of a non-terminal state.
        goal_reward (`float`, *optional*, defaults to `0.0`):
            Added on entering the goal.
        cliff_reward (`float`, *optional*, defaults to `-100.0`):
            Replaces the step reward when entering a cliff cell.
    """

    def __init__(
        self,
        layout: list[str],
        slip: float = 0.0,
        step_reward: float = -1.0,
        goal_reward: float = 0.0,
        cliff_reward: float = -100.0,
    ):
        self.layout = [row for row in layout]
        self.rows, self.cols = len(layout), len(layout[0])
        self.slip = slip
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.cliff_reward = cliff_reward

        self.cells = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        self.index = {cell: i for i, cell in enumerate(self.cells)}
        self.n_states = len(self.cells)
        self.n_actions = len(MOVES)

        self.start = self.index[self._find("S")]
        self.terminal = np.array([self._char(cell) == "G" for cell in self.cells])
        self.state = self.start

    def _find(self, char: str) -> tuple[int, int]:
        for cell in self.cells:
            if self._char(cell) == char:
                return cell
        raise ValueError(f"no {char!r} in layout")

    def _char(self, cell: tuple[int, int]) -> str:
        return self.layout[cell[0]][cell[1]]

    def _move(self, cell: tuple[int, int], move: tuple[int, int]) -> tuple[int, int]:
        """Apply one move, staying put at walls and grid edges."""
        row, col = cell[0] + move[0], cell[1] + move[1]
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return cell
        return cell if self.layout[row][col] == "#" else (row, col)

    def _outcomes(self, state: int, action: int) -> list[tuple[float, int, float]]:
        """`(probability, next state, reward)` triples for one state-action pair."""
        if self.terminal[state]:
            return [(1.0, state, 0.0)]

        perpendicular = [(action - 1) % 4, (action + 1) % 4]
        weighted = [(1.0 - self.slip, action)] + [(self.slip / 2, a) for a in perpendicular]

        outcomes = []
        for probability, move in weighted:
            if probability == 0.0:
                continue
            landing = self._move(self.cells[state], MOVES[move])
            char = self._char(landing)
            if char == "C":
                outcomes.append((probability, self.start, self.cliff_reward))
            elif char == "G":
                outcomes.append((probability, self.index[landing], self.step_reward + self.goal_reward))
            else:
                outcomes.append((probability, self.index[landing], self.step_reward))
        return outcomes

    def to_mdp(self) -> TabularMDP:
        """Build the explicit `P` and `R` arrays the DP solvers need."""
        transitions = np.zeros((self.n_states, self.n_actions, self.n_states))
        rewards = np.zeros((self.n_states, self.n_actions))
        for state in range(self.n_states):
            for action in range(self.n_actions):
                for probability, nxt, reward in self._outcomes(state, action):
                    transitions[state, action, nxt] += probability
                    rewards[state, action] += probability * reward
        return TabularMDP(transitions, rewards, self.terminal)

    def reset(self) -> int:
        self.state = self.start
        return self.state

    def step(self, action: int, rng: np.random.Generator) -> tuple[int, float, bool]:
        outcomes = self._outcomes(self.state, action)
        probabilities = [o[0] for o in outcomes]
        _, nxt, reward = outcomes[rng.choice(len(outcomes), p=probabilities)]
        self.state = nxt
        return nxt, reward, bool(self.terminal[nxt])

    def render_policy(self, policy: np.ndarray) -> str:
        """The policy as an arrow map, for eyeballing what a run learned."""
        lines = []
        for row in range(self.rows):
            line = ""
            for col in range(self.cols):
                char = self.layout[row][col]
                line += char if char in "#GC" else ACTION_CHARS[policy[self.index[(row, col)]]]
            lines.append(line)
        return "\n".join(lines)

    def shortest_paths(self) -> np.ndarray:
        """BFS distance from every cell to the goal, ignoring rewards entirely.

        An independent check on value iteration: with deterministic moves,
        `step_reward = -1` and `gamma = 1`, the optimal value is exactly the
        negated distance.
        """
        goal = self.index[self._find("G")]
        distance = np.full(self.n_states, np.inf)
        distance[goal] = 0.0
        frontier = [goal]
        while frontier:
            current = frontier.pop(0)
            for state in range(self.n_states):
                if distance[state] < np.inf or self._char(self.cells[state]) == "#":
                    continue
                if any(self._move(self.cells[state], m) == self.cells[current] for m in MOVES):
                    distance[state] = distance[current] + 1
                    frontier.append(state)
        return distance
