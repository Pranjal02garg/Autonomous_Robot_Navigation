"""Unit tests for GridWorld environment."""

import pytest
from config import EnvConfig, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT
from environment.gridworld import GridWorldEnv


def test_environment_initialization():
    """Verify default initialization parameters."""
    env = GridWorldEnv()
    assert env.width == 10
    assert env.height == 10
    assert env.start_pos == (0, 0)
    assert env.goal_pos == (9, 9)
    assert len(env.obstacles) > 0
    assert env.current_pos == (0, 0)
    assert env.step_count == 0


def test_environment_reset():
    """Test reset returns initial state and cleans state."""
    env = GridWorldEnv()
    env.step(ACTION_RIGHT)
    state, info = env.reset()
    assert state == (0, 0)
    assert env.current_pos == (0, 0)
    assert env.step_count == 0
    assert info["start_pos"] == (0, 0)
    assert info["goal_pos"] == (9, 9)


def test_movement_actions():
    """Test all four discrete movement directions in free space."""
    cfg = EnvConfig(obstacles=[])  # Empty grid for clean motion test
    env = GridWorldEnv(cfg)

    # Start at (0, 0)
    # Move DOWN -> (0, 1)
    nxt, r, done, info = env.step(ACTION_DOWN)
    assert nxt == (0, 1)
    assert env.current_pos == (0, 1)
    assert not info["collision"]

    # Move RIGHT -> (1, 1)
    nxt, r, done, info = env.step(ACTION_RIGHT)
    assert nxt == (1, 1)
    assert env.current_pos == (1, 1)
    assert not info["collision"]

    # Move UP -> (1, 0)
    nxt, r, done, info = env.step(ACTION_UP)
    assert nxt == (1, 0)
    assert env.current_pos == (1, 0)
    assert not info["collision"]

    # Move LEFT -> (0, 0)
    nxt, r, done, info = env.step(ACTION_LEFT)
    assert nxt == (0, 0)
    assert env.current_pos == (0, 0)
    assert not info["collision"]


def test_boundary_collisions():
    """Test boundary collisions keep the robot in place with penalty."""
    env = GridWorldEnv()
    env.reset()
    # At (0, 0), moving UP or LEFT hits boundary
    nxt, r, done, info = env.step(ACTION_UP)
    assert nxt == (0, 0)
    assert info["collision"] is True
    assert info["boundary_hit"] is True
    assert r == (env.boundary_penalty + env.step_penalty)

    nxt, r, done, info = env.step(ACTION_LEFT)
    assert nxt == (0, 0)
    assert info["collision"] is True
    assert info["boundary_hit"] is True


def test_obstacle_collisions():
    """Test obstacle collision keeps robot in place with obstacle penalty."""
    # Place obstacle at (1, 0)
    cfg = EnvConfig(obstacles=[(1, 0)])
    env = GridWorldEnv(cfg)
    env.reset()

    # Move RIGHT into obstacle at (1, 0)
    nxt, r, done, info = env.step(ACTION_RIGHT)
    assert nxt == (0, 0)
    assert info["collision"] is True
    assert info["obstacle_hit"] is True
    assert r == (env.obstacle_penalty + env.step_penalty)


def test_goal_detection_and_termination():
    """Test reaching goal terminates episode and awards goal reward."""
    cfg = EnvConfig(start_pos=(8, 9), goal_pos=(9, 9), obstacles=[])
    env = GridWorldEnv(cfg)
    env.reset()

    nxt, r, done, info = env.step(ACTION_RIGHT)
    assert nxt == (9, 9)
    assert done is True
    assert info["success"] is True
    assert r == env.goal_reward


def test_max_steps_truncation():
    """Test episode terminates when max steps is reached."""
    cfg = EnvConfig(max_steps=5, obstacles=[])
    env = GridWorldEnv(cfg)
    env.reset()

    for i in range(4):
        _, _, done, _ = env.step(ACTION_RIGHT)
        assert not done

    _, _, done, info = env.step(ACTION_RIGHT)
    assert done is True
    assert info["truncated"] is True
    assert info["success"] is False


def test_bfs_pathfinder():
    """Test that BFS successfully computes shortest path."""
    env = GridWorldEnv()
    path = env.find_shortest_path_bfs()
    assert path is not None
    assert len(path) > 0
    assert path[0] == env.start_pos
    assert path[-1] == env.goal_pos
