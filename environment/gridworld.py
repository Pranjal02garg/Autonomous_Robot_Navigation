"""Gridworld simulation environment for autonomous robot navigation."""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np

from config import (
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_NAMES,
    ACTION_DELTAS,
    EnvConfig,
)


class GridWorldEnv:
    """A configurable 2D GridWorld navigation environment.

    The robot starts at a designated start position (default (0, 0)) and must
    navigate to a target goal (default (9, 9)) while avoiding static obstacles
    and staying within the grid boundaries.

    State representation:
        Tuple[int, int]: (x, y) coordinates where x in [0, width-1]
        and y in [0, height-1].

    Actions:
        0: UP    (dx=0, dy=-1)
        1: DOWN  (dx=0, dy=1)
        2: LEFT  (dx=-1, dy=0)
        3: RIGHT (dx=1, dy=0)
    """

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config if config is not None else EnvConfig()

        self.width: int = self.config.grid_width
        self.height: int = self.config.grid_height
        self.start_pos: Tuple[int, int] = tuple(self.config.start_pos)
        self.goal_pos: Tuple[int, int] = tuple(self.config.goal_pos)
        self.obstacles: Set[Tuple[int, int]] = set(map(tuple, self.config.obstacles))
        self.max_steps: int = self.config.max_steps

        # Reward parameters
        self.goal_reward: float = self.config.goal_reward
        self.obstacle_penalty: float = self.config.obstacle_penalty
        self.boundary_penalty: float = self.config.boundary_penalty
        self.step_penalty: float = self.config.step_penalty
        self.use_potential_shaping: bool = self.config.use_potential_shaping
        self.potential_gamma: float = self.config.potential_gamma

        # Runtime state
        self.current_pos: Tuple[int, int] = self.start_pos
        self.step_count: int = 0
        self.trajectory: List[Tuple[int, int]] = [self.start_pos]

        # Validate layout integrity
        self._validate_layout()

    def _validate_layout(self) -> None:
        """Ensure start and goal positions are valid and obstacle-free."""
        if not (0 <= self.start_pos[0] < self.width and 0 <= self.start_pos[1] < self.height):
            raise ValueError(f"Start position {self.start_pos} is outside grid boundaries ({self.width}x{self.height}).")
        if not (0 <= self.goal_pos[0] < self.width and 0 <= self.goal_pos[1] < self.height):
            raise ValueError(f"Goal position {self.goal_pos} is outside grid boundaries ({self.width}x{self.height}).")
        if self.start_pos in self.obstacles:
            raise ValueError(f"Start position {self.start_pos} cannot be an obstacle.")
        if self.goal_pos in self.obstacles:
            raise ValueError(f"Goal position {self.goal_pos} cannot be an obstacle.")

    def reset(self, start_pos: Optional[Tuple[int, int]] = None) -> Tuple[Tuple[int, int], Dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            start_pos: Optional override for the starting position.

        Returns:
            initial_state: (x, y) tuple.
            info: Dictionary containing initialization details.
        """
        if start_pos is not None:
            if start_pos in self.obstacles or not self._is_within_bounds(start_pos):
                raise ValueError(f"Invalid custom start position: {start_pos}")
            self.current_pos = tuple(start_pos)
        else:
            self.current_pos = self.start_pos

        self.step_count = 0
        self.trajectory = [self.current_pos]

        info = {
            "start_pos": self.current_pos,
            "goal_pos": self.goal_pos,
            "grid_size": (self.width, self.height),
            "step_count": self.step_count,
        }
        return self.current_pos, info

    def get_state(self) -> Tuple[int, int]:
        """Return the current robot coordinates (x, y)."""
        return self.current_pos

    def is_goal(self, pos: Optional[Tuple[int, int]] = None) -> bool:
        """Check whether a position matches the goal."""
        target = pos if pos is not None else self.current_pos
        return target == self.goal_pos

    def _is_within_bounds(self, pos: Tuple[int, int]) -> bool:
        """Check if coordinates fall within the grid world bounds."""
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def is_obstacle(self, pos: Tuple[int, int]) -> bool:
        """Check if coordinates coincide with a static obstacle."""
        return pos in self.obstacles

    def get_valid_actions(self, pos: Optional[Tuple[int, int]] = None) -> List[int]:
        """Return list of actions that do not collide with walls or obstacles.

        Useful for baseline pathfinders and action masking experiments.
        """
        state = pos if pos is not None else self.current_pos
        valid_actions = []
        for action, (dx, dy) in ACTION_DELTAS.items():
            nxt = (state[0] + dx, state[1] + dy)
            if self._is_within_bounds(nxt) and not self.is_obstacle(nxt):
                valid_actions.append(action)
        return valid_actions

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two cells."""
        return float(abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))

    def _potential(self, pos: Tuple[int, int]) -> float:
        """Potential-based reward shaping function: Phi(s) = -Manhattan(s, goal)."""
        return -self._manhattan_distance(pos, self.goal_pos)

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        """Execute action, transition the environment, and calculate reward.

        Args:
            action: Discrete action in {0: UP, 1: DOWN, 2: LEFT, 3: RIGHT}.

        Returns:
            next_state: Tuple[int, int]
            reward: float
            done: bool
            info: dict containing collision, success, and step diagnostics.
        """
        if action not in ACTION_DELTAS:
            raise ValueError(f"Invalid action {action}. Must be in {list(ACTION_DELTAS.keys())}.")

        self.step_count += 1
        dx, dy = ACTION_DELTAS[action]
        intended_next = (self.current_pos[0] + dx, self.current_pos[1] + dy)

        collision = False
        obstacle_hit = False
        boundary_hit = False
        success = False
        done = False

        # Check boundary collision
        if not self._is_within_bounds(intended_next):
            collision = True
            boundary_hit = True
            next_state = self.current_pos
            reward = self.boundary_penalty + self.step_penalty
        # Check obstacle collision
        elif self.is_obstacle(intended_next):
            collision = True
            obstacle_hit = True
            next_state = self.current_pos
            reward = self.obstacle_penalty + self.step_penalty
        # Valid movement into open cell
        else:
            next_state = intended_next
            if self.is_goal(next_state):
                success = True
                done = True
                reward = self.goal_reward
            else:
                reward = self.step_penalty

                # Optional potential-based shaping: F = gamma * Phi(s') - Phi(s)
                if self.use_potential_shaping:
                    shaping = (self.potential_gamma * self._potential(next_state)) - self._potential(self.current_pos)
                    reward += shaping

        # Episode step truncation check
        if self.step_count >= self.max_steps and not done:
            done = True

        self.current_pos = next_state
        self.trajectory.append(self.current_pos)

        info = {
            "step": self.step_count,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "collision": collision,
            "obstacle_hit": obstacle_hit,
            "boundary_hit": boundary_hit,
            "success": success,
            "truncated": (self.step_count >= self.max_steps and not success),
            "trajectory_length": len(self.trajectory),
        }

        return next_state, reward, done, info

    def find_shortest_path_bfs(self) -> Optional[List[Tuple[int, int]]]:
        """Compute the deterministic shortest path from start to goal using BFS.

        Returns:
            List of (x, y) coordinates from start to goal, or None if no path exists.
        """
        queue = deque([(self.start_pos, [self.start_pos])])
        visited = {self.start_pos}

        while queue:
            current, path = queue.popleft()
            if current == self.goal_pos:
                return path

            for _, (dx, dy) in ACTION_DELTAS.items():
                nbr = (current[0] + dx, current[1] + dy)
                if self._is_within_bounds(nbr) and not self.is_obstacle(nbr) and nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, path + [nbr]))

        return None

    def render(self, mode: str = "ansi") -> Optional[str]:
        """Render a text-based ASCII representation of the gridworld.

        Legend:
            S: Start
            G: Goal
            R: Robot current position
            #: Obstacle
            .: Free cell
            *: Visited path
        """
        grid = []
        path_set = set(self.trajectory[:-1])

        for y in range(self.height):
            row = []
            for x in range(self.width):
                pos = (x, y)
                if pos == self.current_pos:
                    row.append("R")
                elif pos == self.goal_pos:
                    row.append("G")
                elif pos == self.start_pos:
                    row.append("S")
                elif pos in self.obstacles:
                    row.append("#")
                elif pos in path_set:
                    row.append("*")
                else:
                    row.append(".")
            grid.append(" ".join(row))

        output = "\n".join(grid)
        if mode == "ansi":
            print(output)
            print(f"Step: {self.step_count}/{self.max_steps} | Pos: {self.current_pos} | Goal: {self.goal_pos}\n")
        return output
