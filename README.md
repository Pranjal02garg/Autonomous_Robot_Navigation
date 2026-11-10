# Autonomous Robot Navigation and Obstacle Avoidance Using Q-Learning and SARSA

**Course:** UCS760 — Edge AI and Robotics: Reinforcement Learning  
**Author:** Pranjal Garg  
**Project:** Simulation-Based Reinforcement Learning for Autonomous Navigation  
**Platform:** macOS (Compatible with Apple Silicon M1/M2/M3/M4 and Intel)  
**Language:** Python 3.12+ (Lightweight dependencies: NumPy, Pygame, Pandas, Matplotlib, Pytest, ReportLab)  

---

## 1. Project Overview
This repository provides a complete, from-scratch implementation and academic evaluation of tabular **Q-Learning** (Watkins, 1989) and **SARSA** (Rummery & Niranjan, 1994) applied to autonomous robot navigation in a 2D discrete gridworld populated by static obstacles.

### Key Highlights:
- **Zero Blackbox RL Libraries:** Core temporal-difference update rules, $\epsilon$-greedy action selection, and Bellman equations are written manually using standard NumPy arrays.
- **Gymnasium-Style Environment:** Fully configurable $10 \times 10$ GridWorld with customizable obstacles, boundary walls, step rewards, and goal conditions.
- **Interactive Pygame Visualizer:** Real-time 60 FPS graphical simulation featuring path tracing, live HUD telemetry, speed adjustments, and hotkey mode switching.
- **Simulation-Generated RL Dataset:** Automatically generates and saves over 140,000 real simulation transition records into structured CSV datasets.
- **Academic Benchmark:** Direct performance comparison against a deterministic Breadth-First Search (BFS) optimal baseline (18 steps).
- **Automated Verification:** 16 comprehensive unit tests covering mathematics, kinematics, boundaries, persistence, and schemas.
- **Academic Report & Presentation:** Complete 25-section research report (Markdown and compiled PDF) and university slide deck outline.

---

## 2. Directory Structure

```
Autonomous_Robot_Navigation/
├── main.py                     # Master CLI entry point (train, demo, experiments, test, report)
├── config.py                   # Central configuration (grid, rewards, hyperparameters, layouts)
├── requirements.txt           # Python library dependencies
├── README.md                   # Complete macOS documentation & execution guide
│
├── environment/                # Environment simulation package
│   ├── __init__.py
│   └── gridworld.py            # 2D GridWorld simulation & BFS shortest-path solver
│
├── algorithms/                 # Reinforcement Learning algorithms package
│   ├── __init__.py
│   ├── base_agent.py           # Abstract TabularAgent base class
│   ├── q_learning.py           # Tabular Q-Learning (Off-policy TD control)
│   └── sarsa.py                # Tabular SARSA (On-policy TD control)
│
├── training/                   # Training orchestration and data logger
│   ├── __init__.py
│   └── trainer.py              # Multi-episode runner and CSV step recorder
│
├── visualization/              # Graphical user interface
│   ├── __init__.py
│   └── simulator.py            # Interactive Pygame visualizer with HUD
│
├── evaluation/                 # Metrics and experimental research suite
│   ├── __init__.py
│   ├── metrics.py              # Statistical aggregations & moving averages
│   └── experiments.py          # 7 experimental benchmarks & Matplotlib plotting
│
├── dataset/                    # Genuine simulation-generated CSV datasets (>140k rows)
│   ├── q_learning_training.csv # Step-by-step experience buffer for Q-Learning
│   ├── sarsa_training.csv      # Step-by-step experience buffer for SARSA
│   ├── training_summary.csv    # Episode summaries (reward, steps, collisions)
│   ├── testing_results.csv     # 100 greedy test episodes per algorithm
│   └── environment_config.csv  # Metadata of environment parameters
│
├── models/                     # Saved trained policy weights
│   ├── q_learning_table.npy    # Trained Q-table for Q-Learning (10x10x4 float64)
│   └── sarsa_table.npy         # Trained Q-table for SARSA (10x10x4 float64)
│
├── results/                    # Publication-quality figures & comparison tables
│   ├── reward_comparison.png   # Training reward & moving average curves
│   ├── success_rate.png        # Training success rate convergence
│   ├── steps_comparison.png    # Episode step count convergence vs BFS
│   ├── collision_comparison.png# Collision progression & test collision rate
│   ├── bfs_comparison.png      # 2D path map: BFS vs Q-Learning vs SARSA
│   ├── layout_generalization.png# Performance across 4 distinct map layouts
│   ├── epsilon_decay.png       # Epsilon decay schedule curves
│   └── algorithm_comparison.csv# Final quantitative benchmark table
│
├── tests/                      # Automated pytest unit test suite (16 tests)
│   ├── test_environment.py    # Environment physics, boundaries, and obstacles
│   ├── test_algorithms.py     # Mathematical correctness of TD updates & model I/O
│   └── test_dataset.py        # Dataset schema validation & metric calculations
│
├── report/                     # Academic documentation
│   ├── project_report.md       # Full 25-section academic report
│   ├── generate_pdf.py         # ReportLab PDF compiler
│   └── project_report.pdf      # Compiled academic research paper (PDF)
│
└── presentation/               # University presentation material
    └── presentation_outline.md # 16-slide presentation deck & viva guide
```

---

## 3. Installation on macOS

### Prerequisites
- macOS 12+ (Monterey, Ventura, Sonoma, Sequoia)
- Python 3.10, 3.11, or 3.12 installed (check with `python3 --version`)

### Step 1: Open Terminal and Navigate to Project
```bash
cd ~/Desktop/Autonomous_Robot_Navigation
```

### Step 2: (Optional but Recommended) Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. How to Run the Project (Step-by-Step)

The project is controlled via a clean, unified command-line interface in `main.py`.

### A. Launch Interactive Pygame Visual Simulation
Open the real-time simulation window to watch the robot navigate:
```bash
# 1. Run Trained Q-Learning Agent (Optimal 18-step path)
python3 main.py demo --algorithm q_learning

# 2. Run Trained SARSA Agent
python3 main.py demo --algorithm sarsa

# 3. Run Untrained / Random Agent (Demonstrates random wandering & collisions)
python3 main.py demo --algorithm random

# 4. Run Sequential Comparison Mode
python3 main.py demo --algorithm compare
```

#### Simulation Interactive Controls:
| Key | Action |
| :---: | :--- |
| **`1`** | Switch to Untrained / Random Agent mode |
| **`2`** | Switch to Trained Q-Learning Agent mode |
| **`3`** | Switch to Trained SARSA Agent mode |
| **`4`** | Switch to Sequential Comparison mode |
| **`Space`** | Pause / Resume navigation |
| **`R`** | Reset active episode to start position $(0, 0)$ |
| **`+` / `-`** | Increase / Decrease simulation speed (FPS) |
| **`Esc` / `Q`** | Exit simulator cleanly |

---

### B. Train Agents from Scratch
Train Q-Learning, SARSA, or both models:
```bash
# Train both agents (3,000 episodes each, default seed 42)
python3 main.py train --algorithm both --episodes 3000

# Train only Q-Learning
python3 main.py train --algorithm q_learning --episodes 3000

# Train only SARSA
python3 main.py train --algorithm sarsa --episodes 3000
```
*Note: Training 3,000 episodes takes only 1.2 seconds per algorithm on a modern Mac! Models are saved to `models/` and datasets are exported to `dataset/`.*

---

### C. Run Full Experimental Research Suite & Generate Graphs
Execute all 7 rigorous research experiments (Q-Learning vs SARSA benchmarks, multi-seed evaluation, layout generalization, exploration analysis, reward sensitivity, and BFS comparison):
```bash
python3 main.py experiments --episodes 3000
```
This automatically updates all publication-grade figures in `results/` and datasets in `dataset/`.

---

### D. Run Automated Test Suite
Verify that all 16 mathematical, kinematic, and schema tests pass:
```bash
python3 main.py test
```
Or directly via pytest:
```bash
pytest -v tests/
```

---

### E. Compile Academic PDF Project Report
Compile the complete 25-section academic paper (with embedded tables and figures):
```bash
python3 main.py report
```
Output PDF is saved at: `report/project_report.pdf`.

---

## 5. Experimental Results Summary

### Primary Benchmark (100 Greedy Test Episodes on Default Map):
| Evaluation Metric | BFS Optimal Baseline | Q-Learning (Greedy) | SARSA (Greedy) |
| :--- | :---: | :---: | :---: |
| **Success Rate (%)** | 100.0% | **100.0%** | **100.0%** |
| **Mean Navigation Steps** | 18.0 | **18.0** | **18.0** |
| **Optimality Gap (vs BFS)** | 0 steps | **0 steps (Exact)** | **0 steps (Exact)** |
| **Mean Cumulative Reward** | +83.0 | **+83.0** | **+83.0** |
| **Reward Standard Deviation** | 0.0 | **0.0** | **0.0** |
| **Mean Collisions per Episode** | 0.0 | **0.0** | **0.0** |
| **Collision Episode Rate (%)** | 0.0% | **0.0%** | **0.0%** |

Both trained agents achieve zero collisions and discover the theoretical geodesic shortest path of 18 steps from $(0, 0)$ to $(9, 9)$.

---

## 6. Course Alignment (UCS760)

| Course Topic | Project Implementation Reference |
| :--- | :--- |
| **Reinforcement Learning Formulation** | Markov Decision Process formalization in `environment/gridworld.py` |
| **State & Action Spaces** | Discrete 2D Cartesian grid coordinates and 4-action cardinal kinematics |
| **Reward Engineering** | Goal incentive (+100), step penalty (-1), and collision penalty (-50) |
| **Temporal-Difference Learning** | Tabular TD(0) value updates implemented from first principles in NumPy |
| **Off-Policy TD Control** | Watkins' Q-Learning using Bellman optimality operator in `algorithms/q_learning.py` |
| **On-Policy TD Control** | Rummery & Niranjan's SARSA using sampled behavioral action in `algorithms/sarsa.py` |
| **Exploration vs Exploitation** | $\epsilon$-greedy action selection with exponential decay ($1.0 \to 0.01$) |
| **Simulation-Based Robotics** | Real-time Pygame graphical dashboard with obstacle physics and telemetry |
| **Experimental Evaluation** | Multi-seed robustness (5 seeds), 7 structured experiments, 140k+ row datasets |

---

## 7. Submission Checklist & Verification

- [x] Full source code adhering to clean PEP 8 modular Python standards.
- [x] Manual NumPy implementation of Q-Learning and SARSA (no blackbox libraries).
- [x] Gymnasium-style discrete 2D GridWorld with 4 diverse layout topologies.
- [x] Breadth-First Search (BFS) optimal baseline pathfinder.
- [x] Interactive Pygame simulator with 4 demonstration modes and live HUD.
- [x] Genuine simulation datasets (`q_learning_training.csv`, `sarsa_training.csv`, etc.).
- [x] All 8 experimental plots generated and saved as high-resolution PNGs.
- [x] All 16 automated pytest unit tests passing cleanly.
- [x] Complete 25-section academic report in Markdown and PDF.
- [x] Complete 16-slide university presentation deck and viva preparation guide.
