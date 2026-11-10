"""Tabular Q-Learning (Watkins, 1989) algorithm implementation.

Off-policy temporal-difference control algorithm that learns the optimal
action-value function Q*(s, a) independent of the agent's behavior policy.
"""

from typing import Optional, Tuple
import numpy as np

from algorithms.base_agent import TabularAgent


class QLearningAgent(TabularAgent):
    """Q-Learning agent implementing off-policy TD(0) control.

    Mathematical update:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
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
        super().__init__(
            grid_width=grid_width,
            grid_height=grid_height,
            num_actions=num_actions,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            random_seed=random_seed,
        )
        self.algorithm_name = "Q-Learning"

    def update(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
        done: bool,
    ) -> float:
        """Update Q-value using Bellman optimality operator.

        Args:
            state: Current state coordinates (x, y).
            action: Selected action integer.
            reward: Immediate scalar reward received.
            next_state: Resulting state coordinates (x', y').
            done: Flag indicating episode termination.

        Returns:
            td_error: Scalar temporal difference error.
        """
        sx, sy = state
        nsx, nsy = next_state

        current_q = self.q_table[sx, sy, action]

        if done:
            target = reward
        else:
            # Off-policy: max over all possible actions in next_state
            max_next_q = np.max(self.q_table[nsx, nsy, :])
            target = reward + self.gamma * max_next_q

        td_error = target - current_q
        self.q_table[sx, sy, action] += self.alpha * td_error

        return float(td_error)
