"""Unit tests for dataset schema, simulation data generation, and evaluation metrics."""

import os
import tempfile
import pandas as pd
import pytest

from config import EnvConfig, RLConfig, PathConfig
from evaluation.metrics import compute_performance_metrics, compute_moving_average
from training.trainer import RLTrainer


def test_dataset_generation_and_schema():
    """Verify that RLTrainer generates valid CSV datasets matching the academic specification."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = PathConfig(
            base_dir=tmp_dir,
            models_dir=os.path.join(tmp_dir, "models"),
            dataset_dir=os.path.join(tmp_dir, "dataset"),
            results_dir=os.path.join(tmp_dir, "results"),
            q_learning_csv=os.path.join(tmp_dir, "dataset", "q_learning_training.csv"),
            sarsa_csv=os.path.join(tmp_dir, "dataset", "sarsa_training.csv"),
            training_summary_csv=os.path.join(tmp_dir, "dataset", "training_summary.csv"),
            testing_results_csv=os.path.join(tmp_dir, "dataset", "testing_results.csv"),
            env_config_csv=os.path.join(tmp_dir, "dataset", "environment_config.csv"),
            q_learning_model_path=os.path.join(tmp_dir, "models", "q_learning_table.npy"),
            sarsa_model_path=os.path.join(tmp_dir, "models", "sarsa_table.npy"),
        )

        trainer = RLTrainer(path_config=paths)
        agent, step_df, summary_df = trainer.train_q_learning(episodes=10, save_model=True, save_dataset=True, verbose=False)

        # Check CSV existence and size
        assert os.path.exists(paths.q_learning_csv)
        assert os.path.getsize(paths.q_learning_csv) > 0

        # Load CSV and verify schema
        df = pd.read_csv(paths.q_learning_csv)
        expected_columns = [
            "episode",
            "step",
            "state_x",
            "state_y",
            "action",
            "action_name",
            "reward",
            "next_state_x",
            "next_state_y",
            "done",
            "algorithm",
            "epsilon",
            "collision",
            "success",
        ]
        for col in expected_columns:
            assert col in df.columns, f"Missing required column: {col}"

        # Check values and types
        assert len(df) > 0
        assert df["algorithm"].iloc[0] == "Q-Learning"
        assert df["episode"].min() == 1
        assert df["episode"].max() == 10
        assert set(df["action_name"].unique()).issubset({"UP", "DOWN", "LEFT", "RIGHT"})


def test_metric_calculations():
    """Verify performance metrics calculation correctness."""
    mock_eval = pd.DataFrame([
        {"test_episode": 1, "algorithm": "Q-Learning", "total_reward": 82.0, "steps": 18, "collisions": 0, "success": True},
        {"test_episode": 2, "algorithm": "Q-Learning", "total_reward": 75.0, "steps": 25, "collisions": 1, "success": True},
        {"test_episode": 3, "algorithm": "Q-Learning", "total_reward": -200.0, "steps": 200, "collisions": 4, "success": False},
    ])

    metrics = compute_performance_metrics(mock_eval, "Q-Learning")
    assert pytest.approx(metrics["success_rate_pct"], abs=0.1) == 66.67
    assert pytest.approx(metrics["mean_reward"], abs=0.1) == -14.33
    assert pytest.approx(metrics["mean_steps_overall"], abs=0.1) == 81.0
    assert pytest.approx(metrics["mean_steps_success"], abs=0.1) == 21.5
    assert pytest.approx(metrics["collision_episode_rate_pct"], abs=0.1) == 66.67
