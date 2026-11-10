"""Main command-line entry point for UCS760 Reinforcement Learning Navigation Project.

Provides CLI interfaces to train models, run interactive Pygame simulations,
execute experimental evaluations, run pytest test suites, and generate PDF reports.
"""

import argparse
import os
import sys

from config import EnvConfig, RLConfig, PathConfig
from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SARSAAgent
from training.trainer import RLTrainer
from evaluation.experiments import ExperimentSuite
from visualization.simulator import PygameSimulator


def train_command(args: argparse.Namespace) -> None:
    """Execute model training routine."""
    trainer = RLTrainer()
    algo = args.algorithm.lower()
    episodes = args.episodes
    seed = args.seed

    print(f"\n==================================================")
    print(f"TRAINING EXECUTION: Algorithm={algo} | Episodes={episodes} | Seed={seed}")
    print(f"==================================================")

    if algo in ("q_learning", "qlearning", "q"):
        trainer.train_q_learning(episodes=episodes, seed=seed)
    elif algo in ("sarsa", "s"):
        trainer.train_sarsa(episodes=episodes, seed=seed)
    elif algo in ("both", "all"):
        trainer.train_both(episodes=episodes, seed=seed)
    else:
        raise ValueError(f"Unknown algorithm: {algo}. Choose from: 'q_learning', 'sarsa', 'both'.")


def demo_command(args: argparse.Namespace) -> None:
    """Launch interactive Pygame visual simulation."""
    sim = PygameSimulator(fps=args.speed)

    # Set default mode based on CLI arg
    algo = args.algorithm.lower()
    if algo in ("random", "untrained"):
        sim.mode = 1
    elif algo in ("q_learning", "qlearning", "q"):
        sim.mode = 2
    elif algo in ("sarsa", "s"):
        sim.mode = 3
    elif algo in ("compare", "comparison", "both"):
        sim.mode = 4

    sim.run(max_seconds=args.duration)


def experiments_command(args: argparse.Namespace) -> None:
    """Execute complete 7-experiment research suite."""
    suite = ExperimentSuite()
    suite.run_all(episodes=args.episodes)


def test_command(args: argparse.Namespace) -> None:
    """Execute automated test suite via pytest."""
    import pytest
    tests_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
    print(f"\nRunning automated test suite in: {tests_dir}\n")
    exit_code = pytest.main(["-v", tests_dir])
    if exit_code != 0:
        print(f"Tests failed with exit code: {exit_code}")
        sys.exit(exit_code)
    print("\nAll automated tests passed successfully!")


def report_command(args: argparse.Namespace) -> None:
    """Generate academic PDF project report."""
    from report.generate_pdf import generate_academic_pdf
    generate_academic_pdf()


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="UCS760 Reinforcement Learning Autonomous Robot Navigation (Q-Learning & SARSA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="mode", help="Execution mode")

    # Train subparser
    train_parser = subparsers.add_parser("train", help="Train Q-Learning and/or SARSA agents")
    train_parser.add_argument("--algorithm", type=str, default="both", choices=["q_learning", "sarsa", "both"], help="Algorithm to train")
    train_parser.add_argument("--episodes", type=int, default=3000, help="Number of training episodes (default 3000)")
    train_parser.add_argument("--seed", type=int, default=42, help="Reproducibility random seed (default 42)")

    # Demo subparser
    demo_parser = subparsers.add_parser("demo", help="Launch interactive Pygame visualizer")
    demo_parser.add_argument("--algorithm", type=str, default="q_learning", choices=["random", "q_learning", "sarsa", "compare"], help="Initial agent mode")
    demo_parser.add_argument("--speed", type=int, default=8, help="Simulation speed in steps/sec (default 8)")
    demo_parser.add_argument("--duration", type=int, default=None, help="Auto-close duration in seconds (optional)")

    # Experiments subparser
    exp_parser = subparsers.add_parser("experiments", help="Run full 7-experiment evaluation and plot graphs")
    exp_parser.add_argument("--episodes", type=int, default=3000, help="Training episodes per benchmark (default 3000)")

    # Tests subparser
    subparsers.add_parser("test", help="Run automated test suite")

    # Report subparser
    subparsers.add_parser("report", help="Compile academic PDF report")

    return parser


def main() -> None:
    """Main program entry point."""
    parser = build_parser()
    if len(sys.argv) == 1:
        # Default behavior when run without flags: display help & run demo or interactive guidance
        parser.print_help()
        print("\n[INFO] No mode specified. To launch simulation, run:\n       python3 main.py demo --algorithm q_learning\n")
        return

    args = parser.parse_args()

    if args.mode == "train":
        train_command(args)
    elif args.mode == "demo":
        demo_command(args)
    elif args.mode == "experiments":
        experiments_command(args)
    elif args.mode == "test":
        test_command(args)
    elif args.mode == "report":
        report_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
