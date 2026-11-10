"""Base class for tabular reinforcement learning agents."""

import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class TabularAgent(ABC):
    """Abstract base class for tabular discrete RL algorithms.

    State space: 2D coordinate grid (x, y) with x in [0, width-1], y in [0, height-1].
    Action space: Discrete 4 actions {0: UP, 1: DOWN, 2: LEFT, 3: RIGHT}.
    Q-table shape: (width, height, 4)
    """

    def __init__(
        self,
        grid_width: int = 10,
        grid_height: int = 10,
        num_actions: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        random_seed: Optional[int] = 42,
    ):
        self.width = grid_width
        self.height = grid_height
        self.num_actions = num_actions
        self.alpha = float(learning_rate)
        self.gamma = float(discount_factor)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.random_seed = random_seed

        self.rng = np.random.default_rng(random_seed)

        # Initialize Q-table with zeros: Q(x, y, a)
        self.q_table: np.ndarray = np.zeros((self.width, self.height, self.num_actions), dtype=np.float64)

    def choose_action(self, state: Tuple[int, int], epsilon: Optional[float] = None, greedy: bool = False) -> int:
        """Select action using epsilon-greedy policy with random tie-breaking.

        Args:
            state: (x, y) coordinate tuple.
            epsilon: Optional override for exploration rate.
            greedy: If True, always select optimal action (epsilon = 0).

        Returns:
            Selected discrete action integer in [0..num_actions-1].
        """
        eps = 0.0 if greedy else (self.epsilon if epsilon is None else float(epsilon))

        # Exploration: choose random action uniformly
        if eps > 0.0 and self.rng.random() < eps:
            return int(self.rng.integers(0, self.num_actions))

        # Exploitation: argmax over Q-values for current state
        q_values = self.q_table[state[0], state[1], :]
        max_val = np.max(q_values)
        # Random tie-breaking among all actions achieving the maximum Q-value
        best_actions = np.flatnonzero(np.isclose(q_values, max_val, atol=1e-8))
        return int(self.rng.choice(best_actions))

    def decay_epsilon(self) -> float:
        """Apply multiplicative decay to epsilon down to epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def save(self, filepath: str) -> None:
        """Save Q-table to a .npy file with directory creation and validation."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self.validate_q_table(self.q_table)
        np.save(filepath, self.q_table)

    def load(self, filepath: str) -> None:
        """Load and validate Q-table from a .npy file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at: {filepath}")

        loaded_table = np.load(filepath)
        self.validate_q_table(loaded_table)
        self.q_table = loaded_table.astype(np.float64)

    def validate_q_table(self, table: np.ndarray) -> None:
        """Ensure Q-table has correct shape, valid numeric entries, and no NaNs/Infs."""
        expected_shape = (self.width, self.height, self.num_actions)
        if table.shape != expected_shape:
            raise ValueError(
                f"Invalid Q-table shape: expected {expected_shape}, got {table.shape}."
            )
        if not np.all(np.isfinite(table)):
            raise ValueError("Q-table contains NaN or infinite values.")

    def get_policy(self) -> np.ndarray:
        """Return deterministic greedy policy array of shape (width, height)."""
        return np.argmax(self.q_table, axis=-1)

    def get_value_function(self) -> np.ndarray:
        """Return state-value function V(s) = max_a Q(s, a) of shape (width, height)."""
        return np.max(self.q_table, axis=-1)

    @abstractmethod
    def update(self, *args, **kwargs) -> float:
        """Perform TD update on Q-table. Must be implemented by subclass."""
        pass
