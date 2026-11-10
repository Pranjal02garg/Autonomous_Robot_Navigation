"""Comprehensive experimental suite for UCS760 RL autonomous robot navigation.

Executes 7 rigorous experiments, performs statistical evaluations,
and generates publication-quality figures and datasets.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Headless rendering for headless/GUI agnostic environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SARSAAgent
from config import (
    EnvConfig,
    RLConfig,
    PathConfig,
    MAP_LAYOUTS,
    DEFAULT_OBSTACLES,
)
from environment.gridworld import GridWorldEnv
from evaluation.metrics import compute_performance_metrics, compute_moving_average
from training.trainer import RLTrainer


class ExperimentSuite:
    """Orchestrates all experimental evaluations and visualizations."""

    def __init__(
        self,
        env_config: Optional[EnvConfig] = None,
        rl_config: Optional[RLConfig] = None,
        path_config: Optional[PathConfig] = None,
    ):
        self.env_cfg = env_config if env_config is not None else EnvConfig()
        self.rl_cfg = rl_config if rl_config is not None else RLConfig()
        self.path_cfg = path_config if path_config is not None else PathConfig()
        self.trainer = RLTrainer(self.env_cfg, self.rl_cfg, self.path_cfg)

        os.makedirs(self.path_cfg.results_dir, exist_ok=True)
        os.makedirs(self.path_cfg.dataset_dir, exist_ok=True)

    def run_primary_benchmarks(
        self,
        episodes: int = 3000,
        eval_episodes: int = 100,
        seeds: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Execute Experiments 1, 2, 3: Full training and greedy test comparison."""
        run_seeds = seeds if seeds is not None else self.rl_cfg.multi_seeds
        print(f"\n==================================================")
        print(f"RUNNING PRIMARY BENCHMARK: Q-LEARNING VS SARSA")
        print(f"Episodes: {episodes} | Seeds: {run_seeds} | Test Episodes: {eval_episodes}")
        print(f"==================================================")

        # Train primary models using the default seed (42) and save datasets & models
        print("\n>>> Training Production Q-Learning Agent (Seed 42)...")
        q_agent, q_step_df, q_summary_df = self.trainer.train_q_learning(
            episodes=episodes,
            seed=42,
            save_model=True,
            save_dataset=True,
            verbose=True,
        )

        print("\n>>> Training Production SARSA Agent (Seed 42)...")
        s_agent, s_step_df, s_summary_df = self.trainer.train_sarsa(
            episodes=episodes,
            seed=42,
            save_model=True,
            save_dataset=True,
            verbose=True,
        )

        # Joint summary CSV
        combined_summary = pd.concat([q_summary_df, s_summary_df], ignore_index=True)
        combined_summary.to_csv(self.path_cfg.training_summary_csv, index=False)
        self.trainer.save_environment_config_csv()

        # Multi-seed robustness experiment for statistical variance reporting
        multi_seed_results = []
        if len(run_seeds) > 1:
            print(f"\n>>> Running Multi-Seed Stability Evaluations ({len(run_seeds)} seeds)...")
            for seed in run_seeds:
                if seed == 42:
                    q_sub_sum = q_summary_df
                    s_sub_sum = s_summary_df
                else:
                    _, _, q_sub_sum = self.trainer.train_q_learning(
                        episodes=min(episodes, 1500), seed=seed, save_model=False, save_dataset=False, verbose=False
                    )
                    _, _, s_sub_sum = self.trainer.train_sarsa(
                        episodes=min(episodes, 1500), seed=seed, save_model=False, save_dataset=False, verbose=False
                    )

                multi_seed_results.append({
                    "seed": seed,
                    "q_final_reward_mean": q_sub_sum["total_reward"].iloc[-100:].mean(),
                    "q_final_success_pct": q_sub_sum["success"].iloc[-100:].mean() * 100,
                    "s_final_reward_mean": s_sub_sum["total_reward"].iloc[-100:].mean(),
                    "s_final_success_pct": s_sub_sum["success"].iloc[-100:].mean() * 100,
                })

        # Experiment 3: Evaluate both agents on identical greedy test episodes
        print("\n>>> Evaluating Both Agents on Identical Greedy Test Conditions (100 episodes)...")
        q_test_df = self.trainer.evaluate_agent(q_agent, num_episodes=eval_episodes, greedy=True, seed=999)
        s_test_df = self.trainer.evaluate_agent(s_agent, num_episodes=eval_episodes, greedy=True, seed=999)

        all_tests = pd.concat([q_test_df, s_test_df], ignore_index=True)
        all_tests.to_csv(self.path_cfg.testing_results_csv, index=False)
        print(f"Saved test results to: {self.path_cfg.testing_results_csv}")

        # Compute summary metrics table
        q_metrics = compute_performance_metrics(q_test_df, "Q-Learning")
        s_metrics = compute_performance_metrics(s_test_df, "SARSA")

        comp_df = pd.DataFrame([q_metrics, s_metrics])
        comp_df.to_csv(self.path_cfg.algorithm_comparison_csv, index=False)
        print(f"\n--- ALGORITHM COMPARISON SUMMARY ---")
        print(comp_df.to_string(index=False))

        # Generate core training & testing graphs
        self._plot_reward_curves(q_summary_df, s_summary_df)
        self._plot_success_rates(q_summary_df, s_summary_df)
        self._plot_steps_progression(q_summary_df, s_summary_df)
        self._plot_collision_comparison(q_summary_df, s_summary_df, q_test_df, s_test_df)

        return {
            "q_agent": q_agent,
            "s_agent": s_agent,
            "q_summary": q_summary_df,
            "s_summary": s_summary_df,
            "q_test": q_test_df,
            "s_test": s_test_df,
            "comparison": comp_df,
            "multi_seed": pd.DataFrame(multi_seed_results) if multi_seed_results else None,
        }

    def run_layout_generalization(self, episodes: int = 1500) -> pd.DataFrame:
        """Experiment 4: Evaluate Q-Learning and SARSA across multiple obstacle layouts."""
        print(f"\n==================================================")
        print(f"EXPERIMENT 4: OBSTACLE LAYOUT GENERALIZATION")
        print(f"Layouts: {list(MAP_LAYOUTS.keys())}")
        print(f"==================================================")

        records = []
        for layout_name, obstacles in MAP_LAYOUTS.items():
            print(f"\nEvaluating Layout: '{layout_name}' (Obstacles: {len(obstacles)} cells)...")
            cfg = EnvConfig(obstacles=obstacles, layout_name=layout_name)
            sub_trainer = RLTrainer(cfg, self.rl_cfg, self.path_cfg)

            # Train Q-Learning
            q_agent, _, _ = sub_trainer.train_q_learning(episodes=episodes, save_model=False, save_dataset=False, verbose=False)
            q_eval = sub_trainer.evaluate_agent(q_agent, num_episodes=50, env_config=cfg, greedy=True)
            q_metrics = compute_performance_metrics(q_eval, "Q-Learning")

            # Train SARSA
            s_agent, _, _ = sub_trainer.train_sarsa(episodes=episodes, save_model=False, save_dataset=False, verbose=False)
            s_eval = sub_trainer.evaluate_agent(s_agent, num_episodes=50, env_config=cfg, greedy=True)
            s_metrics = compute_performance_metrics(s_eval, "SARSA")

            records.append({
                "layout": layout_name,
                "obstacles_count": len(obstacles),
                "q_success_rate": q_metrics["success_rate_pct"],
                "q_mean_steps": q_metrics["mean_steps_overall"],
                "q_mean_reward": q_metrics["mean_reward"],
                "s_success_rate": s_metrics["success_rate_pct"],
                "s_mean_steps": s_metrics["mean_steps_overall"],
                "s_mean_reward": s_metrics["mean_reward"],
            })

        gen_df = pd.DataFrame(records)
        print("\nGeneralization Results:")
        print(gen_df.to_string(index=False))

        # Plot generalization comparison
        self._plot_layout_generalization(gen_df)
        return gen_df

    def run_exploration_investigation(self, episodes: int = 1500) -> pd.DataFrame:
        """Experiment 5: Compare different epsilon decay rates."""
        print(f"\n==================================================")
        print(f"EXPERIMENT 5: EPSILON EXPLORATION INVESTIGATION")
        print(f"==================================================")

        decay_rates = [0.98, 0.995, 0.999]
        decay_records = []

        fig, ax = plt.subplots(figsize=(10, 5))

        for decay in decay_rates:
            print(f"Testing Epsilon Decay = {decay}...")
            # Simulate theoretical decay curve
            eps = 1.0
            eps_curve = []
            for _ in range(episodes):
                eps_curve.append(eps)
                eps = max(0.01, eps * decay)

            ax.plot(eps_curve, label=f"Decay = {decay} (floor 0.01)", lw=2)

            custom_rl = RLConfig(epsilon_decay=decay)
            sub_trainer = RLTrainer(self.env_cfg, custom_rl, self.path_cfg)
            _, _, q_summary = sub_trainer.train_q_learning(episodes=episodes, save_model=False, save_dataset=False, verbose=False)

            decay_records.append({
                "decay_factor": decay,
                "final_100_reward": q_summary["total_reward"].iloc[-100:].mean(),
                "final_100_success_pct": q_summary["success"].iloc[-100:].mean() * 100,
                "final_100_avg_steps": q_summary["steps"].iloc[-100:].mean(),
            })

        ax.set_title("Exploration Schedule (Epsilon Decay Comparison)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Training Episodes", fontsize=11)
        ax.set_ylabel("Epsilon Exploration Probability", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(self.path_cfg.epsilon_decay_png, dpi=300)
        plt.close()
        print(f"Saved exploration figure to: {self.path_cfg.epsilon_decay_png}")

        exp_df = pd.DataFrame(decay_records)
        print("\nExploration Schedule Performance:")
        print(exp_df.to_string(index=False))
        return exp_df

    def run_reward_shaping_investigation(self, episodes: int = 1500) -> pd.DataFrame:
        """Experiment 6: Investigate impact of obstacle penalties and step costs."""
        print(f"\n==================================================")
        print(f"EXPERIMENT 6: REWARD STRUCTURE INVESTIGATION")
        print(f"==================================================")

        penalty_variations = [
            ("Mild (-10)", -10.0),
            ("Standard (-50)", -50.0),
            ("Severe (-100)", -100.0),
        ]

        results = []
        for label, penalty in penalty_variations:
            print(f"Testing Obstacle Penalty: {label}...")
            custom_env = EnvConfig(obstacle_penalty=penalty, boundary_penalty=penalty)
            sub_trainer = RLTrainer(custom_env, self.rl_cfg, self.path_cfg)

            agent, _, summary_df = sub_trainer.train_q_learning(
                episodes=episodes, save_model=False, save_dataset=False, verbose=False
            )
            eval_df = sub_trainer.evaluate_agent(agent, num_episodes=50, env_config=custom_env, greedy=True)
            metrics = compute_performance_metrics(eval_df, f"Penalty {label}")

            results.append({
                "penalty_setting": label,
                "obstacle_penalty_value": penalty,
                "test_success_rate_pct": metrics["success_rate_pct"],
                "test_mean_steps": metrics["mean_steps_overall"],
                "test_mean_collisions": metrics["mean_collisions"],
                "test_mean_reward": metrics["mean_reward"],
            })

        rew_df = pd.DataFrame(results)
        print("\nReward Sensitivity Analysis:")
        print(rew_df.to_string(index=False))
        return rew_df

    def run_bfs_comparison(
        self,
        q_agent: Optional[QLearningAgent] = None,
        s_agent: Optional[SARSAAgent] = None,
    ) -> Dict[str, Any]:
        """Experiment 7: Compare RL learned policies against BFS deterministic shortest path."""
        print(f"\n==================================================")
        print(f"EXPERIMENT 7: BFS SHORTEST PATH BASELINE COMPARISON")
        print(f"==================================================")

        env = GridWorldEnv(self.env_cfg)
        bfs_path = env.find_shortest_path_bfs()
        if bfs_path is None:
            raise RuntimeError("BFS could not find a path on the default map!")

        bfs_steps = len(bfs_path) - 1
        print(f"BFS Shortest Path Length: {bfs_steps} steps | Coordinates count: {len(bfs_path)}")

        # Extract deterministic policy paths for Q-Learning and SARSA
        def extract_agent_path(agent) -> List[Tuple[int, int]]:
            s_env = GridWorldEnv(self.env_cfg)
            state, _ = s_env.reset()
            path = [state]
            visited = set()
            for _ in range(s_env.max_steps):
                if s_env.is_goal(state):
                    break
                action = agent.choose_action(state, greedy=True)
                nxt, _, done, _ = s_env.step(action)
                state = nxt
                path.append(state)
                if done:
                    break
            return path

        if q_agent is None:
            q_agent = QLearningAgent(self.env_cfg.grid_width, self.env_cfg.grid_height)
            q_agent.load(self.path_cfg.q_learning_model_path)

        if s_agent is None:
            s_agent = SARSAAgent(self.env_cfg.grid_width, self.env_cfg.grid_height)
            s_agent.load(self.path_cfg.sarsa_model_path)

        q_path = extract_agent_path(q_agent)
        s_path = extract_agent_path(s_agent)

        q_steps = len(q_path) - 1
        s_steps = len(s_path) - 1

        print(f"BFS Path Steps:        {bfs_steps}")
        print(f"Q-Learning Path Steps: {q_steps} (Optimality Gap: {q_steps - bfs_steps})")
        print(f"SARSA Path Steps:      {s_steps} (Optimality Gap: {s_steps - bfs_steps})")

        # Plot 2D visual trajectory map
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        titles = ["BFS Shortest Path (Baseline)", "Q-Learning Learned Path", "SARSA Learned Path"]
        paths = [bfs_path, q_path, s_path]
        colors = ["#2563eb", "#16a34a", "#dc2626"]

        for ax, title, path, col in zip(axes, titles, paths, colors):
            # Draw grid
            ax.set_xlim(-0.5, self.env_cfg.grid_width - 0.5)
            ax.set_ylim(self.env_cfg.grid_height - 0.5, -0.5)  # Inverted y for top-to-bottom matrix view
            ax.set_xticks(range(self.env_cfg.grid_width))
            ax.set_yticks(range(self.env_cfg.grid_height))
            ax.grid(True, color="#cbd5e1", linewidth=0.8)

            # Draw obstacles
            for ox, oy in self.env_cfg.obstacles:
                ax.add_patch(plt.Rectangle((ox - 0.5, oy - 0.5), 1, 1, color="#334155", zorder=2))

            # Draw Start and Goal
            sx, sy = self.env_cfg.start_pos
            gx, gy = self.env_cfg.goal_pos
            ax.add_patch(plt.Circle((sx, sy), 0.35, color="#059669", zorder=4, label="Start (0,0)"))
            ax.add_patch(plt.Circle((gx, gy), 0.35, color="#d97706", zorder=4, label="Goal (9,9)"))

            # Draw Path
            px, py = zip(*path)
            ax.plot(px, py, color=col, linewidth=2.5, marker="o", markersize=4, zorder=3, label=f"Path ({len(path)-1} steps)")

            ax.set_title(f"{title}\nLength: {len(path)-1} steps", fontsize=11, fontweight="bold")
            ax.set_aspect("equal")
            ax.legend(loc="lower left", fontsize=8)

        plt.suptitle("Navigation Trajectory Comparison Against Optimal BFS Baseline", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.path_cfg.bfs_comparison_png, dpi=300)
        plt.close()
        print(f"Saved BFS trajectory comparison to: {self.path_cfg.bfs_comparison_png}")

        return {
            "bfs_steps": bfs_steps,
            "q_steps": q_steps,
            "s_steps": s_steps,
            "bfs_path": bfs_path,
            "q_path": q_path,
            "s_path": s_path,
        }

    # ================= VISUALIZATION PLOT GENERATORS =================

    def _plot_reward_curves(self, q_df: pd.DataFrame, s_df: pd.DataFrame) -> None:
        """Plot episode rewards and rolling moving averages."""
        fig, ax = plt.subplots(figsize=(10, 5))

        q_roll = compute_moving_average(q_df["total_reward"], window=50)
        s_roll = compute_moving_average(s_df["total_reward"], window=50)

        # Raw curves in subtle transparent lines
        ax.plot(q_df["episode"], q_df["total_reward"], color="#2563eb", alpha=0.15, label="Q-Learning (Raw)")
        ax.plot(s_df["episode"], s_df["total_reward"], color="#dc2626", alpha=0.15, label="SARSA (Raw)")

        # Moving average curves in bold
        ax.plot(q_df["episode"], q_roll, color="#1d4ed8", linewidth=2.2, label="Q-Learning (50-ep MA)")
        ax.plot(s_df["episode"], s_roll, color="#b91c1c", linewidth=2.2, label="SARSA (50-ep MA)")

        ax.set_title("Training Reward Progression: Q-Learning vs SARSA", fontsize=13, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("Total Cumulative Reward", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        plt.savefig(self.path_cfg.reward_comparison_png, dpi=300)
        plt.close()
        print(f"Saved: {self.path_cfg.reward_comparison_png}")

    def _plot_success_rates(self, q_df: pd.DataFrame, s_df: pd.DataFrame) -> None:
        """Plot rolling success rate progression over training."""
        fig, ax = plt.subplots(figsize=(10, 5))

        q_succ = compute_moving_average(q_df["success"].astype(float) * 100, window=50)
        s_succ = compute_moving_average(s_df["success"].astype(float) * 100, window=50)

        ax.plot(q_df["episode"], q_succ, color="#1d4ed8", linewidth=2.2, label="Q-Learning (50-ep Rolling Success %)")
        ax.plot(s_df["episode"], s_succ, color="#b91c1c", linewidth=2.2, label="SARSA (50-ep Rolling Success %)")

        ax.axhline(100.0, color="#16a34a", linestyle=":", alpha=0.7, label="100% Target")
        ax.set_title("Training Success Rate Convergence", fontsize=13, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("Success Rate (%)", fontsize=11)
        ax.set_ylim(-5, 105)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        plt.savefig(self.path_cfg.success_rate_png, dpi=300)
        plt.close()
        print(f"Saved: {self.path_cfg.success_rate_png}")

    def _plot_steps_progression(self, q_df: pd.DataFrame, s_df: pd.DataFrame) -> None:
        """Plot episode step count convergence toward optimal path."""
        fig, ax = plt.subplots(figsize=(10, 5))

        q_steps_ma = compute_moving_average(q_df["steps"], window=50)
        s_steps_ma = compute_moving_average(s_df["steps"], window=50)

        ax.plot(q_df["episode"], q_steps_ma, color="#1d4ed8", linewidth=2.2, label="Q-Learning (50-ep MA Steps)")
        ax.plot(s_df["episode"], s_steps_ma, color="#b91c1c", linewidth=2.2, label="SARSA (50-ep MA Steps)")

        # Mark optimal BFS steps (18)
        ax.axhline(18, color="#059669", linestyle="--", linewidth=1.5, label="BFS Shortest Path (18 steps)")

        ax.set_title("Navigation Efficiency: Steps per Episode Convergence", fontsize=13, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("Steps Taken", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right", fontsize=10)
        plt.tight_layout()
        plt.savefig(self.path_cfg.steps_comparison_png, dpi=300)
        plt.close()
        print(f"Saved: {self.path_cfg.steps_comparison_png}")

    def _plot_collision_comparison(
        self,
        q_train: pd.DataFrame,
        s_train: pd.DataFrame,
        q_test: pd.DataFrame,
        s_test: pd.DataFrame,
    ) -> None:
        """Plot collision progression during training and greedy test collision distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Subplot 1: Rolling collisions in training
        q_col_ma = compute_moving_average(q_train["collisions"], window=50)
        s_col_ma = compute_moving_average(s_train["collisions"], window=50)

        ax1.plot(q_train["episode"], q_col_ma, color="#1d4ed8", label="Q-Learning", lw=2)
        ax1.plot(s_train["episode"], s_col_ma, color="#b91c1c", label="SARSA", lw=2)
        ax1.set_title("Training Collision Rate (50-ep MA)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Episode", fontsize=10)
        ax1.set_ylabel("Average Collisions per Episode", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        # Subplot 2: Greedy test collision counts
        bars = ax2.bar(
            ["Q-Learning (Greedy)", "SARSA (Greedy)"],
            [q_test["collisions"].mean(), s_test["collisions"].mean()],
            color=["#3b82f6", "#ef4444"],
            width=0.45,
            edgecolor="black",
        )
        ax2.set_title("Evaluation Phase Collision Frequency (100 Greedy Tests)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Mean Collisions per Test Episode", fontsize=10)
        ax2.grid(True, axis="y", linestyle="--", alpha=0.6)

        # Annotate bar values
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f"{height:.2f}",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontweight="bold")

        plt.suptitle("Safety & Obstacle Avoidance Performance Analysis", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.path_cfg.collision_comparison_png, dpi=300)
        plt.close()
        print(f"Saved: {self.path_cfg.collision_comparison_png}")

    def _plot_layout_generalization(self, gen_df: pd.DataFrame) -> None:
        """Plot layout generalization comparison bar charts."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        layouts = gen_df["layout"].tolist()
        x = np.arange(len(layouts))
        width = 0.35

        # Chart 1: Success Rate by Layout
        ax1.bar(x - width/2, gen_df["q_success_rate"], width, label="Q-Learning", color="#3b82f6", edgecolor="black")
        ax1.bar(x + width/2, gen_df["s_success_rate"], width, label="SARSA", color="#ef4444", edgecolor="black")
        ax1.set_title("Success Rate Across Diverse Topologies", fontsize=11, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(layouts)
        ax1.set_ylabel("Success Rate (%)", fontsize=10)
        ax1.set_ylim(0, 110)
        ax1.grid(True, axis="y", linestyle="--", alpha=0.6)
        ax1.legend()

        # Chart 2: Mean Steps by Layout
        ax2.bar(x - width/2, gen_df["q_mean_steps"], width, label="Q-Learning", color="#3b82f6", edgecolor="black")
        ax2.bar(x + width/2, gen_df["s_mean_steps"], width, label="SARSA", color="#ef4444", edgecolor="black")
        ax2.set_title("Navigation Steps Across Diverse Topologies", fontsize=11, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(layouts)
        ax2.set_ylabel("Mean Steps", fontsize=10)
        ax2.grid(True, axis="y", linestyle="--", alpha=0.6)
        ax2.legend()

        plt.suptitle("Topological Generalization Study (Default, Maze, Clutter, Bottleneck)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.path_cfg.layout_generalization_png, dpi=300)
        plt.close()
        print(f"Saved: {self.path_cfg.layout_generalization_png}")

    def run_all(self, episodes: int = 3000) -> Dict[str, Any]:
        """Execute the complete suite of 7 experiments and generate all artifacts."""
        primary_out = self.run_primary_benchmarks(episodes=episodes)
        gen_df = self.run_layout_generalization(episodes=1500)
        exp_df = self.run_exploration_investigation(episodes=1500)
        rew_df = self.run_reward_shaping_investigation(episodes=1500)
        bfs_out = self.run_bfs_comparison(primary_out["q_agent"], primary_out["s_agent"])

        return {
            "primary": primary_out,
            "generalization": gen_df,
            "exploration": exp_df,
            "reward_analysis": rew_df,
            "bfs": bfs_out,
        }
