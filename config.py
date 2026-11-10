"""Configuration module for UCS760 Reinforcement Learning Robot Navigation.

Defines grid dimensions, obstacle maps, reward structures, action spaces,
and algorithm hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Action definitions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

ACTION_NAMES: Dict[int, str] = {
    ACTION_UP: "UP",
    ACTION_DOWN: "DOWN",
    ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT",
}

# (dx, dy) mapping where x is horizontal [0..width-1] and y is vertical [0..height-1]
# (0, 0) is top-left, (9, 9) is bottom-right
ACTION_DELTAS: Dict[int, Tuple[int, int]] = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}


# Predefined obstacle layouts for fair comparison and generalization testing
DEFAULT_OBSTACLES: List[Tuple[int, int]] = [
    # Top-center wall
    (2, 1), (2, 2), (2, 3), (2, 4),
    # Mid-right horizontal wall
    (5, 3), (6, 3), (7, 3),
    # Central barrier
    (4, 5), (4, 6), (4, 7),
    # Lower obstacles creating a winding path toward (9,9)
    (6, 6), (7, 6), (7, 7), (7, 8),
    (1, 7), (2, 7), (3, 7)
]

MAZE_OBSTACLES: List[Tuple[int, int]] = (
    [(x, 2) for x in range(0, 8)]
    + [(x, 5) for x in range(2, 10)]
    + [(x, 7) for x in range(0, 8)]
)

CLUTTER_OBSTACLES: List[Tuple[int, int]] = [
    (1, 2), (2, 5), (3, 8), (4, 2), (5, 6), (6, 1), (7, 4), (8, 7),
    (2, 2), (5, 2), (8, 2), (3, 5), (6, 8), (4, 7), (7, 2), (1, 6)
]

BOTTLENECK_OBSTACLES: List[Tuple[int, int]] = [
    # Full vertical divider across x=5 with a single doorway at y=5
    (5, 0), (5, 1), (5, 2), (5, 3), (5, 4),
    # Gap at (5, 5)
    (5, 6), (5, 7), (5, 8), (5, 9)
]

MAP_LAYOUTS: Dict[str, List[Tuple[int, int]]] = {
    "default": DEFAULT_OBSTACLES,
    "maze": MAZE_OBSTACLES,
    "clutter": CLUTTER_OBSTACLES,
    "bottleneck": BOTTLENECK_OBSTACLES,
}


@dataclass
class EnvConfig:
    """Environment configuration settings."""
    grid_width: int = 10
    grid_height: int = 10
    start_pos: Tuple[int, int] = (0, 0)
    goal_pos: Tuple[int, int] = (9, 9)
    obstacles: List[Tuple[int, int]] = field(default_factory=lambda: list(DEFAULT_OBSTACLES))
    max_steps: int = 200
    layout_name: str = "default"

    # Reward function parameters
    goal_reward: float = 100.0
    obstacle_penalty: float = -50.0
    boundary_penalty: float = -50.0
    step_penalty: float = -1.0
    use_potential_shaping: bool = False
    potential_gamma: float = 0.95


@dataclass
class RLConfig:
    """Hyperparameters for Q-Learning and SARSA."""
    learning_rate: float = 0.1       # Alpha
    discount_factor: float = 0.95    # Gamma
    epsilon_start: float = 1.0       # Initial exploration rate
    epsilon_min: float = 0.01        # Minimum exploration rate
    epsilon_decay: float = 0.995     # Multiplicative decay per episode
    episodes: int = 3000             # Standard training episodes
    eval_episodes: int = 100         # Greedy evaluation episodes
    random_seed: int = 42            # Default reproducibility seed
    multi_seeds: List[int] = field(default_factory=lambda: [42, 101, 2024, 777, 999])


@dataclass
class PathConfig:
    """Directory and file paths for models, datasets, results, and reports."""
    base_dir: str = BASE_DIR
    models_dir: str = os.path.join(BASE_DIR, "models")
    dataset_dir: str = os.path.join(BASE_DIR, "dataset")
    results_dir: str = os.path.join(BASE_DIR, "results")
    report_dir: str = os.path.join(BASE_DIR, "report")
    presentation_dir: str = os.path.join(BASE_DIR, "presentation")

    # Model file paths
    q_learning_model_path: str = os.path.join(BASE_DIR, "models", "q_learning_table.npy")
    sarsa_model_path: str = os.path.join(BASE_DIR, "models", "sarsa_table.npy")

    # Dataset file paths
    q_learning_csv: str = os.path.join(BASE_DIR, "dataset", "q_learning_training.csv")
    sarsa_csv: str = os.path.join(BASE_DIR, "dataset", "sarsa_training.csv")
    training_summary_csv: str = os.path.join(BASE_DIR, "dataset", "training_summary.csv")
    testing_results_csv: str = os.path.join(BASE_DIR, "dataset", "testing_results.csv")
    env_config_csv: str = os.path.join(BASE_DIR, "dataset", "environment_config.csv")

    # Visual result graphs
    reward_comparison_png: str = os.path.join(BASE_DIR, "results", "reward_comparison.png")
    success_rate_png: str = os.path.join(BASE_DIR, "results", "success_rate.png")
    steps_comparison_png: str = os.path.join(BASE_DIR, "results", "steps_comparison.png")
    collision_comparison_png: str = os.path.join(BASE_DIR, "results", "collision_comparison.png")
    algorithm_comparison_csv: str = os.path.join(BASE_DIR, "results", "algorithm_comparison.csv")
    bfs_comparison_png: str = os.path.join(BASE_DIR, "results", "bfs_comparison.png")
    epsilon_decay_png: str = os.path.join(BASE_DIR, "results", "epsilon_decay.png")
    layout_generalization_png: str = os.path.join(BASE_DIR, "results", "layout_generalization.png")
