"""Evaluation metrics and statistical calculation module."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def compute_performance_metrics(
    eval_df: pd.DataFrame,
    algorithm_name: str = "Agent",
) -> Dict[str, Any]:
    """Calculate core performance metrics from evaluation episode records.

    Metrics:
        - Success rate (%)
        - Average steps (overall and successful only)
        - Average reward
        - Std dev of reward
        - Total collisions & average collisions per episode
        - Collision rate (episodes with at least one collision)
    """
    total_episodes = len(eval_df)
    if total_episodes == 0:
        return {}

    success_mask = eval_df["success"] == True
    success_count = int(success_mask.sum())
    success_rate = (success_count / total_episodes) * 100.0

    mean_reward = float(eval_df["total_reward"].mean())
    std_reward = float(eval_df["total_reward"].std(ddof=1)) if total_episodes > 1 else 0.0

    mean_steps_all = float(eval_df["steps"].mean())
    mean_steps_success = float(eval_df.loc[success_mask, "steps"].mean()) if success_count > 0 else float("nan")

    mean_collisions = float(eval_df["collisions"].mean())
    episodes_with_collisions = int((eval_df["collisions"] > 0).sum())
    collision_episode_rate = (episodes_with_collisions / total_episodes) * 100.0

    return {
        "algorithm": algorithm_name,
        "total_test_episodes": total_episodes,
        "success_rate_pct": round(success_rate, 2),
        "mean_reward": round(mean_reward, 2),
        "std_reward": round(std_reward, 2),
        "mean_steps_overall": round(mean_steps_all, 2),
        "mean_steps_success": round(mean_steps_success, 2),
        "mean_collisions": round(mean_collisions, 2),
        "collision_episode_rate_pct": round(collision_episode_rate, 2),
    }


def compute_moving_average(series: pd.Series, window: int = 50) -> pd.Series:
    """Compute rolling mean with min_periods=1 to preserve series length."""
    return series.rolling(window=window, min_periods=1).mean()


def calculate_convergence_episode(
    summary_df: pd.DataFrame,
    threshold_success: float = 0.90,
    window: int = 50,
) -> Optional[int]:
    """Determine the episode number where rolling success rate sustains above threshold."""
    if "success" not in summary_df.columns:
        return None

    rolling_success = summary_df["success"].astype(float).rolling(window=window, min_periods=window).mean()
    converged_idx = np.where(rolling_success >= threshold_success)[0]
    if len(converged_idx) > 0:
        return int(summary_df.iloc[converged_idx[0]]["episode"])
    return None
