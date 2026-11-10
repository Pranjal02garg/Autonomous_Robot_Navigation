"""Tabular SARSA (Rummery & Niranjan, 1994) algorithm implementation.

On-policy temporal-difference control algorithm that estimates Q_pi(s, a)
for the current exploration-inclusive behavior policy pi.
"""

from typing import Optional, Tuple
import numpy as np

from algorithms.base_agent import TabularAgent


class SARSAAgent(TabularAgent):
    """SARSA agent implementing on-policy TD(0) control.

    Mathematical update:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]
    where a' is sampled directly from the epsilon-greedy policy at state s'.
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
        self.algorithm_name = "SARSA"

    def update(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
        next_action: int,
        done: bool,
    ) -> float:
        """Update Q-value using on-policy SARSA transition tuple (s, a, r, s', a').

        Args:
            state: Current state coordinates (x, y).
            action: Current action integer a.
            reward: Immediate scalar reward received.
            next_state: Resulting state coordinates (x', y').
            next_action: Action integer a' chosen by current policy for next_state.
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
            # On-policy: value of the actual chosen next_action
            target = reward + self.gamma * self.q_table[nsx, nsy, next_action]

        td_error = target - current_q
        self.q_table[sx, sy, action] += self.alpha * td_error

        return float(td_error)
