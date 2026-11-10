"""Reusable training and data logging engine for Q-Learning and SARSA."""

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SARSAAgent
from config import (
    EnvConfig,
    RLConfig,
    PathConfig,
    ACTION_NAMES,
)
from environment.gridworld import GridWorldEnv


class RLTrainer:
    """Orchestrates reinforcement learning training, step logging, and dataset export."""

    def __init__(
        self,
        env_config: Optional[EnvConfig] = None,
        rl_config: Optional[RLConfig] = None,
        path_config: Optional[PathConfig] = None,
    ):
        self.env_cfg = env_config if env_config is not None else EnvConfig()
        self.rl_cfg = rl_config if rl_config is not None else RLConfig()
        self.path_cfg = path_config if path_config is not None else PathConfig()

        # Ensure output directories exist
        os.makedirs(self.path_cfg.models_dir, exist_ok=True)
        os.makedirs(self.path_cfg.dataset_dir, exist_ok=True)
        os.makedirs(self.path_cfg.results_dir, exist_ok=True)

    def train_q_learning(
        self,
        episodes: Optional[int] = None,
        seed: Optional[int] = None,
        save_model: bool = True,
        save_dataset: bool = True,
        verbose: bool = True,
    ) -> Tuple[QLearningAgent, pd.DataFrame, pd.DataFrame]:
        """Train a tabular Q-Learning agent on the environment.

        Returns:
            agent: Trained QLearningAgent instance.
            step_df: DataFrame containing step-by-step experience dataset.
            summary_df: DataFrame containing per-episode aggregated metrics.
        """
        run_seed = seed if seed is not None else self.rl_cfg.random_seed
        num_episodes = episodes if episodes is not None else self.rl_cfg.episodes

        env = GridWorldEnv(self.env_cfg)
        agent = QLearningAgent(
            grid_width=self.env_cfg.grid_width,
            grid_height=self.env_cfg.grid_height,
            num_actions=4,
            learning_rate=self.rl_cfg.learning_rate,
            discount_factor=self.rl_cfg.discount_factor,
            epsilon=self.rl_cfg.epsilon_start,
            epsilon_min=self.rl_cfg.epsilon_min,
            epsilon_decay=self.rl_cfg.epsilon_decay,
            random_seed=run_seed,
        )

        step_records: List[Dict[str, Any]] = []
        summary_records: List[Dict[str, Any]] = []

        start_time = time.time()
        if verbose:
            print(f"--- Starting Q-Learning Training ({num_episodes} episodes, seed={run_seed}) ---")

        for ep in range(1, num_episodes + 1):
            state, _ = env.reset()
            ep_reward = 0.0
            ep_collisions = 0
            ep_steps = 0
            ep_success = False

            done = False
            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)

                # Q-Learning update: Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
                agent.update(state, action, reward, next_state, done)

                # Record step-level transition data
                step_records.append({
                    "episode": ep,
                    "step": info["step"],
                    "state_x": state[0],
                    "state_y": state[1],
                    "action": action,
                    "action_name": info["action_name"],
                    "reward": reward,
                    "next_state_x": next_state[0],
                    "next_state_y": next_state[1],
                    "done": done,
                    "algorithm": "Q-Learning",
                    "epsilon": round(agent.epsilon, 5),
                    "collision": info["collision"],
                    "success": info["success"],
                })

                ep_reward += reward
                ep_steps += 1
                if info["collision"]:
                    ep_collisions += 1
                if info["success"]:
                    ep_success = True

                state = next_state

            # Decay epsilon at the end of each episode
            current_eps = agent.decay_epsilon()

            summary_records.append({
                "episode": ep,
                "algorithm": "Q-Learning",
                "total_reward": ep_reward,
                "steps": ep_steps,
                "collisions": ep_collisions,
                "success": ep_success,
                "final_epsilon": current_eps,
                "seed": run_seed,
            })

            if verbose and (ep % 500 == 0 or ep == num_episodes):
                recent_reward = np.mean([r["total_reward"] for r in summary_records[-100:]])
                recent_success = np.mean([r["success"] for r in summary_records[-100:]]) * 100
                recent_steps = np.mean([r["steps"] for r in summary_records[-100:]])
                print(
                    f"Episode {ep:4d}/{num_episodes} | "
                    f"Reward (last 100): {recent_reward:6.1f} | "
                    f"Success: {recent_success:5.1f}% | "
                    f"Avg Steps: {recent_steps:4.1f} | "
                    f"Epsilon: {current_eps:.3f}"
                )

        elapsed = time.time() - start_time
        if verbose:
            print(f"Q-Learning training finished in {elapsed:.2f}s.")

        step_df = pd.DataFrame(step_records)
        summary_df = pd.DataFrame(summary_records)

        if save_model:
            agent.save(self.path_cfg.q_learning_model_path)
            if verbose:
                print(f"Saved Q-Learning model to: {self.path_cfg.q_learning_model_path}")

        if save_dataset:
            step_df.to_csv(self.path_cfg.q_learning_csv, index=False)
            if verbose:
                print(f"Saved Q-Learning dataset ({len(step_df)} rows) to: {self.path_cfg.q_learning_csv}")

        return agent, step_df, summary_df

    def train_sarsa(
        self,
        episodes: Optional[int] = None,
        seed: Optional[int] = None,
        save_model: bool = True,
        save_dataset: bool = True,
        verbose: bool = True,
    ) -> Tuple[SARSAAgent, pd.DataFrame, pd.DataFrame]:
        """Train a tabular SARSA agent on the environment.

        Returns:
            agent: Trained SARSAAgent instance.
            step_df: DataFrame containing step-by-step experience dataset.
            summary_df: DataFrame containing per-episode aggregated metrics.
        """
        run_seed = seed if seed is not None else self.rl_cfg.random_seed
        num_episodes = episodes if episodes is not None else self.rl_cfg.episodes

        env = GridWorldEnv(self.env_cfg)
        agent = SARSAAgent(
            grid_width=self.env_cfg.grid_width,
            grid_height=self.env_cfg.grid_height,
            num_actions=4,
            learning_rate=self.rl_cfg.learning_rate,
            discount_factor=self.rl_cfg.discount_factor,
            epsilon=self.rl_cfg.epsilon_start,
            epsilon_min=self.rl_cfg.epsilon_min,
            epsilon_decay=self.rl_cfg.epsilon_decay,
            random_seed=run_seed,
        )

        step_records: List[Dict[str, Any]] = []
        summary_records: List[Dict[str, Any]] = []

        start_time = time.time()
        if verbose:
            print(f"--- Starting SARSA Training ({num_episodes} episodes, seed={run_seed}) ---")

        for ep in range(1, num_episodes + 1):
            state, _ = env.reset()
            # In SARSA, choose initial action a from initial state s using epsilon-greedy
            action = agent.choose_action(state)

            ep_reward = 0.0
            ep_collisions = 0
            ep_steps = 0
            ep_success = False

            done = False
            while not done:
                next_state, reward, done, info = env.step(action)

                # Next action a' chosen by current epsilon-greedy policy from next_state s'
                if not done:
                    next_action = agent.choose_action(next_state)
                else:
                    next_action = 0  # Dummy value; target depends solely on reward when done=True

                # SARSA update: Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]
                agent.update(state, action, reward, next_state, next_action, done)

                step_records.append({
                    "episode": ep,
                    "step": info["step"],
                    "state_x": state[0],
                    "state_y": state[1],
                    "action": action,
                    "action_name": info["action_name"],
                    "reward": reward,
                    "next_state_x": next_state[0],
                    "next_state_y": next_state[1],
                    "done": done,
                    "algorithm": "SARSA",
                    "epsilon": round(agent.epsilon, 5),
                    "collision": info["collision"],
                    "success": info["success"],
                })

                ep_reward += reward
                ep_steps += 1
                if info["collision"]:
                    ep_collisions += 1
                if info["success"]:
                    ep_success = True

                state = next_state
                action = next_action

            # Decay epsilon at the end of each episode
            current_eps = agent.decay_epsilon()

            summary_records.append({
                "episode": ep,
                "algorithm": "SARSA",
                "total_reward": ep_reward,
                "steps": ep_steps,
                "collisions": ep_collisions,
                "success": ep_success,
                "final_epsilon": current_eps,
                "seed": run_seed,
            })

            if verbose and (ep % 500 == 0 or ep == num_episodes):
                recent_reward = np.mean([r["total_reward"] for r in summary_records[-100:]])
                recent_success = np.mean([r["success"] for r in summary_records[-100:]]) * 100
                recent_steps = np.mean([r["steps"] for r in summary_records[-100:]])
                print(
                    f"Episode {ep:4d}/{num_episodes} | "
                    f"Reward (last 100): {recent_reward:6.1f} | "
                    f"Success: {recent_success:5.1f}% | "
                    f"Avg Steps: {recent_steps:4.1f} | "
                    f"Epsilon: {current_eps:.3f}"
                )

        elapsed = time.time() - start_time
        if verbose:
            print(f"SARSA training finished in {elapsed:.2f}s.")

        step_df = pd.DataFrame(step_records)
        summary_df = pd.DataFrame(summary_records)

        if save_model:
            agent.save(self.path_cfg.sarsa_model_path)
            if verbose:
                print(f"Saved SARSA model to: {self.path_cfg.sarsa_model_path}")

        if save_dataset:
            step_df.to_csv(self.path_cfg.sarsa_csv, index=False)
            if verbose:
                print(f"Saved SARSA dataset ({len(step_df)} rows) to: {self.path_cfg.sarsa_csv}")

        return agent, step_df, summary_df

    def train_both(
        self,
        episodes: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[QLearningAgent, SARSAAgent, pd.DataFrame]:
        """Train both Q-Learning and SARSA agents and write unified summary records."""
        q_agent, q_step_df, q_summary_df = self.train_q_learning(episodes=episodes, seed=seed)
        s_agent, s_step_df, s_summary_df = self.train_sarsa(episodes=episodes, seed=seed)

        combined_summary = pd.concat([q_summary_df, s_summary_df], ignore_index=True)
        combined_summary.to_csv(self.path_cfg.training_summary_csv, index=False)
        print(f"Saved joint training summary to: {self.path_cfg.training_summary_csv}")

        # Save environment configuration metadata CSV
        self.save_environment_config_csv()

        return q_agent, s_agent, combined_summary

    def save_environment_config_csv(self) -> pd.DataFrame:
        """Export environment configuration metadata to CSV."""
        config_rows = [
            {"parameter": "grid_width", "value": self.env_cfg.grid_width, "description": "Horizontal dimension of the grid"},
            {"parameter": "grid_height", "value": self.env_cfg.grid_height, "description": "Vertical dimension of the grid"},
            {"parameter": "start_pos", "value": str(self.env_cfg.start_pos), "description": "Initial robot spawn position (x, y)"},
            {"parameter": "goal_pos", "value": str(self.env_cfg.goal_pos), "description": "Target destination position (x, y)"},
            {"parameter": "obstacle_count", "value": len(self.env_cfg.obstacles), "description": "Total number of static obstacles"},
            {"parameter": "obstacles", "value": ";".join([f"{x},{y}" for x, y in self.env_cfg.obstacles]), "description": "Coordinates of blocked cells"},
            {"parameter": "max_steps", "value": self.env_cfg.max_steps, "description": "Maximum steps allowed per episode"},
            {"parameter": "goal_reward", "value": self.env_cfg.goal_reward, "description": "Scalar reward on reaching goal cell"},
            {"parameter": "obstacle_penalty", "value": self.env_cfg.obstacle_penalty, "description": "Penalty upon attempting to enter obstacle"},
            {"parameter": "boundary_penalty", "value": self.env_cfg.boundary_penalty, "description": "Penalty upon attempting to exit grid"},
            {"parameter": "step_penalty", "value": self.env_cfg.step_penalty, "description": "Standard step cost encouraging shorter paths"},
            {"parameter": "learning_rate_alpha", "value": self.rl_cfg.learning_rate, "description": "Temporal-difference step size"},
            {"parameter": "discount_factor_gamma", "value": self.rl_cfg.discount_factor, "description": "Future reward discount factor"},
            {"parameter": "epsilon_start", "value": self.rl_cfg.epsilon_start, "description": "Initial exploratory probability"},
            {"parameter": "epsilon_min", "value": self.rl_cfg.epsilon_min, "description": "Minimum exploratory probability floor"},
            {"parameter": "epsilon_decay", "value": self.rl_cfg.epsilon_decay, "description": "Multiplicative decay factor per episode"},
            {"parameter": "episodes", "value": self.rl_cfg.episodes, "description": "Standard training episodes count"},
        ]
        df = pd.DataFrame(config_rows)
        df.to_csv(self.path_cfg.env_config_csv, index=False)
        print(f"Saved environment config CSV to: {self.path_cfg.env_config_csv}")
        return df

    def evaluate_agent(
        self,
        agent: Union[QLearningAgent, SARSAAgent],
        num_episodes: int = 100,
        env_config: Optional[EnvConfig] = None,
        greedy: bool = True,
        seed: Optional[int] = 12345,
    ) -> pd.DataFrame:
        """Evaluate an agent greedily across test episodes without exploration.

        Args:
            agent: QLearningAgent or SARSAAgent instance.
            num_episodes: Number of test episodes to execute.
            env_config: Optional custom environment layout.
            greedy: Whether to set epsilon = 0 for pure exploitation.
            seed: Seed for test evaluation run.

        Returns:
            test_df: DataFrame of test performance metrics.
        """
        env = GridWorldEnv(env_config if env_config is not None else self.env_cfg)
        rng = np.random.default_rng(seed)

        results = []
        for ep in range(1, num_episodes + 1):
            state, _ = env.reset()
            ep_reward = 0.0
            ep_steps = 0
            ep_collisions = 0
            ep_success = False

            done = False
            while not done:
                action = agent.choose_action(state, greedy=greedy)
                next_state, reward, done, info = env.step(action)

                ep_reward += reward
                ep_steps += 1
                if info["collision"]:
                    ep_collisions += 1
                if info["success"]:
                    ep_success = True

                state = next_state

            results.append({
                "test_episode": ep,
                "algorithm": agent.algorithm_name,
                "total_reward": ep_reward,
                "steps": ep_steps,
                "collisions": ep_collisions,
                "success": ep_success,
                "greedy": greedy,
            })

        return pd.DataFrame(results)
