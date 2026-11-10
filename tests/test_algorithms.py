"""Unit tests for Q-Learning and SARSA mathematical updates and model persistence."""

import os
import tempfile
import numpy as np
import pytest

from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SARSAAgent


def test_q_table_dimensions():
    """Verify initialized Q-table dimensions and data types."""
    agent = QLearningAgent(grid_width=10, grid_height=10, num_actions=4)
    assert agent.q_table.shape == (10, 10, 4)
    assert agent.q_table.dtype == np.float64
    assert np.all(agent.q_table == 0.0)


def test_q_learning_update_math():
    """Mathematically verify Watkins' Q-learning Bellman update formula."""
    agent = QLearningAgent(learning_rate=0.1, discount_factor=0.95)

    # Preset next state max Q value to 10.0
    agent.q_table[0, 1, 0] = 2.0
    agent.q_table[0, 1, 1] = 10.0  # max
    agent.q_table[0, 1, 2] = 4.0
    agent.q_table[0, 1, 3] = 1.0

    # Step: state (0,0), action 3, reward -1.0, next_state (0,1), done=False
    # Expected: target = -1.0 + 0.95 * 10.0 = 8.5
    # td_error = 8.5 - 0.0 = 8.5
    # new_q = 0.0 + 0.1 * 8.5 = 0.85
    td_error = agent.update(state=(0, 0), action=3, reward=-1.0, next_state=(0, 1), done=False)

    assert pytest.approx(td_error, abs=1e-6) == 8.5
    assert pytest.approx(agent.q_table[0, 0, 3], abs=1e-6) == 0.85


def test_sarsa_update_math():
    """Mathematically verify on-policy SARSA update formula."""
    agent = SARSAAgent(learning_rate=0.1, discount_factor=0.95)

    # Next state has different Q values: max is 10.0 (action 1), but next_action chosen is 2 with value 5.0
    agent.q_table[0, 1, 0] = 2.0
    agent.q_table[0, 1, 1] = 10.0
    agent.q_table[0, 1, 2] = 5.0
    agent.q_table[0, 1, 3] = 1.0

    # Step: state (0,0), action 3, reward -1.0, next_state (0,1), next_action 2, done=False
    # Expected target = -1.0 + 0.95 * 5.0 = 3.75
    # td_error = 3.75 - 0.0 = 3.75
    # new_q = 0.0 + 0.1 * 3.75 = 0.375
    td_error = agent.update(state=(0, 0), action=3, reward=-1.0, next_state=(0, 1), next_action=2, done=False)

    assert pytest.approx(td_error, abs=1e-6) == 3.75
    assert pytest.approx(agent.q_table[0, 0, 3], abs=1e-6) == 0.375


def test_terminal_state_update():
    """Verify that when done=True, the target does not include future Q-values."""
    agent = QLearningAgent(learning_rate=0.1, discount_factor=0.95)
    agent.q_table[9, 9, :] = 100.0  # Even if next state has large values

    td_error = agent.update(state=(8, 9), action=3, reward=100.0, next_state=(9, 9), done=True)
    # Target should be 100.0 exactly, new_q = 0 + 0.1 * 100 = 10.0
    assert pytest.approx(td_error, abs=1e-6) == 100.0
    assert pytest.approx(agent.q_table[8, 9, 3], abs=1e-6) == 10.0


def test_save_and_load_q_table():
    """Verify Q-table persistence and reload integrity."""
    agent = QLearningAgent(grid_width=10, grid_height=10)
    agent.q_table[2, 3, 1] = 42.123

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        agent.save(tmp_path)
        new_agent = QLearningAgent(grid_width=10, grid_height=10)
        new_agent.load(tmp_path)

        assert new_agent.q_table.shape == (10, 10, 4)
        assert pytest.approx(new_agent.q_table[2, 3, 1], abs=1e-6) == 42.123
        assert np.array_equal(agent.q_table, new_agent.q_table)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_epsilon_decay():
    """Verify epsilon decay behaves monotonically and respects floor."""
    agent = QLearningAgent(epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.5)
    assert agent.decay_epsilon() == 0.5
    assert agent.decay_epsilon() == 0.25
    assert agent.decay_epsilon() == 0.125
    # Decay until floor
    for _ in range(10):
        agent.decay_epsilon()
    assert agent.epsilon == 0.01
