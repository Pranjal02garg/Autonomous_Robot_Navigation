# Autonomous Robot Navigation and Obstacle Avoidance Using Q-Learning and SARSA
## University Presentation Deck & Viva Guide
**Course:** UCS760 — Edge AI and Robotics: Reinforcement Learning  
**Target Duration:** 15–20 minutes + Viva Q&A  

---

### Slide 1: Title & Academic Context
- **Title:** Reinforcement Learning-Based Autonomous Robot Navigation and Obstacle Avoidance Using Q-Learning and SARSA
- **Subtitle:** Comparative Empirical Analysis of Tabular Temporal-Difference Control on 2D Grid Worlds
- **Course:** UCS760 — Edge AI and Robotics: Reinforcement Learning
- **Presenter:** Pranjal Garg
- **Platform:** macOS, Python 3.12, NumPy, Pygame, Matplotlib, Pandas
- **Recommended Visual:** Split preview showing the 10x10 Pygame grid simulation and the optimal trajectory comparison graph.

---

### Slide 2: Problem Statement
- **The Core Problem:**
  - An autonomous mobile robot must navigate from a fixed start $(0, 0)$ to a target destination $(9, 9)$ on a 10x10 grid.
  - The map contains static impassable obstacles.
  - **Zero Prior Knowledge:** The robot is not provided with an obstacle map, coordinates, or transition rules.
- **Goal:** Learn an optimal, collision-free control policy $\pi(s)$ via environmental interaction and scalar reward feedback.
- **Recommended Visual:** Diagram of the 10x10 grid with Start `(S)`, Goal `(G)`, and blocked obstacles `(#)`.

---

### Slide 3: Motivation & Real-World Relevance
- **Edge AI & Robotics Context:**
  - Classical path planning (A*, Dijkstra) requires complete geometric maps and pre-computation.
  - In dynamic, unstructured, or GPS-denied environments (search & rescue, warehouse AGVs, lunar rovers), agents must learn through empirical interaction.
  - Model-free Temporal-Difference (TD) methods provide lightweight, edge-computable policies without heavy neural inference hardware.
- **Academic Focus:** Understanding the mathematical and behavioral distinction between **Off-Policy (Q-Learning)** and **On-Policy (SARSA)** algorithms.
- **Recommended Visual:** Real-world Autonomous Mobile Robot (AMR) in a warehouse next to the 2D grid world abstraction.

---

### Slide 4: Project Objectives
1. **From-Scratch Implementation:** Manually code Q-Learning and SARSA using NumPy (no black-box libraries like Stable-Baselines).
2. **Modular Environment:** Develop a Gymnasium-compliant 2D GridWorld with customizable layouts, obstacle densities, and boundaries.
3. **Reward Engineering:** Design a Markovian reward function balancing goal achievement, collision penalties, and step costs.
4. **Empirical Rigor:** Generate a 140,000+ row transition dataset and conduct 7 reproducible experiments across 5 seeds.
5. **Optimal Baseline:** Implement Breadth-First Search (BFS) to establish the true theoretical shortest path (18 steps).
6. **Interactive Visualization:** Build a real-time Pygame simulator with HUD telemetry and dynamic mode toggling.

---

### Slide 5: Reinforcement Learning Fundamentals
- **Markov Decision Process (MDP):** Formulation as $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$.
  - **State Space ($\mathcal{S}$):** 100 discrete coordinates $(x, y) \in [0..9] \times [0..9]$.
  - **Action Space ($\mathcal{A}$):** 4 discrete cardinal movements $\{0: \text{UP}, 1: \text{DOWN}, 2: \text{LEFT}, 3: \text{RIGHT}\}$.
  - **Discount Factor ($\gamma = 0.95$):** Balances immediate step penalties against future goal attainment.
- **Reward Function:**
  $$R(s, a, s') = \begin{cases} +100.0 & \text{Goal reached } (9, 9) \\ -51.0 & \text{Obstacle or boundary collision} \\ -1.0 & \text{Standard movement step} \end{cases}$$
- **Recommended Visual:** Standard RL agent-environment feedback diagram.

---

### Slide 6: System Architecture & Workflow
- **Decoupled Architecture:**
  - `environment/`: GridWorldEnv physics, collision checks, BFS solver.
  - `algorithms/`: TabularAgent base class, QLearningAgent, SARSAAgent.
  - `training/`: RLTrainer engine, step logger, and dataset exporter.
  - `visualization/`: PygameSimulator with 60 FPS event loop.
  - `evaluation/`: Metric aggregator and 7-experiment research suite.
- **Recommended Visual:** High-level software architecture diagram showing data flow between CLI `main.py`, trainer, environment, and datasets.

---

### Slide 7: Environment Design & Layout Topologies
- **Four Benchmark Topologies:**
  1. **Default Layout:** 17 obstacles creating local traps and diagonal passages (BFS: 18 steps).
  2. **Serpentine Maze:** 24 obstacles forcing a 3-pass zigzag corridor (BFS: 32 steps).
  3. **Clutter Layout:** 16 dispersed obstacles testing fine local obstacle dodging.
  4. **Bottleneck Layout:** 9-cell dividing wall with a single doorway at $(5, 5)$.
- **Collision Rule:** Robot stays in place upon collision, incurring $-50.0$ penalty + $-1.0$ step cost.
- **Recommended Visual:** 4-panel image showcasing each obstacle layout.

---

### Slide 8: Q-Learning (Watkins, 1989)
- **Classification:** Off-Policy Temporal-Difference (TD) Control.
- **Update Formula:**
  $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
- **Key Characteristics:**
  - Learns the optimal action-value function $Q^*$ directly.
  - The update assumes the agent will take the greedy action $\max_{a'} Q(s', a')$ next, regardless of actual behavior.
  - Aggressive and optimistic in value propagation.
- **Recommended Visual:** Flow diagram of the Q-Learning Bellman optimality operator.

---

### Slide 9: SARSA (Rummery & Niranjan, 1994)
- **Classification:** On-Policy Temporal-Difference (TD) Control.
- **Update Formula:**
  $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]$$
- **Key Characteristics:**
  - Learns the action-value function $Q^\pi$ for the policy currently being executed.
  - $a'$ is the actual next action sampled via the current $\epsilon$-greedy behavior policy.
  - Incorporates exploratory mistakes into value updates, favoring safer paths away from dangerous cliffs/walls.
- **Recommended Visual:** State-action transition diagram comparing $\max_{a'}$ (Q-Learning) vs sampled $a'$ (SARSA).

---

### Slide 10: Training Process & Exploration Schedule
- **Episodes:** 3,000 episodes per benchmark.
- **Exploration Schedule ($\epsilon$-Greedy):**
  - $\epsilon_0 = 1.0$ (100% random exploration at start).
  - Multiplicative decay: $\epsilon_{t+1} = \max(0.01, \epsilon_t \times 0.995)$.
  - Terminal floor: $\epsilon_{min} = 0.01$.
- **Training Progression:**
  - Episodes 1–200: High exploration, frequent obstacle collisions, low returns.
  - Episodes 200–800: Goal discovery, rapid path optimization, collision rate plummets.
  - Episodes 800–3,000: Policy stabilization, convergence to deterministic geodesic.
- **Recommended Visual:** `results/reward_comparison.png` and `results/epsilon_decay.png`.

---

### Slide 11: Dataset Description & Generation
- **Dataset Label:** *Simulation-Generated Reinforcement Learning Experience Dataset*
- **Contents & Scale:**
  - `q_learning_training.csv`: 69,017 transition rows (4.1 MB).
  - `sarsa_training.csv`: 75,290 transition rows (4.1 MB).
  - `training_summary.csv`: 6,000 episode summary records.
  - `testing_results.csv`: 200 greedy evaluation records.
  - `environment_config.csv`: Complete environment metadata.
- **Features Captured:** `episode`, `step`, `state_x`, `state_y`, `action`, `action_name`, `reward`, `next_state_x`, `next_state_y`, `done`, `algorithm`, `epsilon`, `collision`, `success`.
- **Recommended Visual:** Screenshot of CSV dataframe preview and feature schema table.

---

### Slide 12: Visual Simulation Demonstration
- **Pygame Application Overview:**
  - Real-time 60 FPS interactive dashboard.
  - Path tracing showing agent's visited route in cyan/rose lines.
  - Live HUD telemetry: Position, Goal, Steps, Collisions, Rewards, Status.
- **Four Demonstration Modes:**
  - **Mode A [1]:** Random untrained agent (demonstrates wandering & collisions).
  - **Mode B [2]:** Trained Q-Learning agent (demonstrates smooth optimal path).
  - **Mode C [3]:** Trained SARSA agent (demonstrates learned safety path).
  - **Mode D [4]:** Comparative sequential replay.
- **Live Controls:** Space (pause/resume), R (reset), +/- (speed), Esc (quit).
- **Recommended Visual:** Pygame simulator screenshots showing Mode A (scattered random path) vs Mode B (clean 18-step path).

---

### Slide 13: Experimental Results (Primary Benchmark)
- **Performance Summary (100 Greedy Test Episodes):**

| Evaluation Metric | BFS Optimal Baseline | Q-Learning (Greedy) | SARSA (Greedy) |
| :--- | :---: | :---: | :---: |
| **Success Rate (%)** | 100.0% | **100.0%** | **100.0%** |
| **Mean Navigation Steps** | 18.0 | **18.0** | **18.0** |
| **Optimality Gap (vs BFS)** | 0 steps | **0 steps (Exact)** | **0 steps (Exact)** |
| **Mean Cumulative Reward** | +83.0 | **+83.0** | **+83.0** |
| **Collision Rate (%)** | 0.0% | **0.0%** | **0.0%** |

- **Recommended Visual:** `results/steps_comparison.png` and `results/bfs_comparison.png`.

---

### Slide 14: Algorithmic Comparison: Q-Learning vs SARSA
- **Key Empirical Observations:**
  1. **Standard Map:** Both achieve 100% success and identical 18-step paths once exploration decays.
  2. **Serpentine Maze Generalization (Experiment 4):**
     - **Q-Learning:** 100% success, 32 steps (exact BFS shortest path).
     - **SARSA:** 0% success in 1,500 episodes (truncated at 200 steps).
  3. **Why this happens (Theoretical Insight):**
     - In narrow winding corridors, exploratory actions frequently hit walls.
     - SARSA factors these exploratory collisions into its value updates, severely lowering the value of corridor cells.
     - Q-Learning ignores exploratory noise in updates, propagating positive goal signals back through the corridor rapidly.
- **Recommended Visual:** `results/layout_generalization.png`.

---

### Slide 15: Limitations & Future Scope
- **Current Limitations:**
  - Discrete state space (does not handle continuous $(x, y, \theta)$ poses).
  - Static obstacles (does not account for dynamic moving entities).
  - Point-mass kinematics (ignores steering radius, velocity, acceleration).
- **Future Extensions:**
  - Deep Q-Networks (DQN) with CNN function approximation.
  - Multi-Agent Reinforcement Learning (MARL) for collaborative robot fleets.
  - Sim-to-Real Transfer: ROS 2 integration and physical deployment onto TurtleBot 4.

---

### Slide 16: Conclusion & Summary
- **Accomplishments:**
  - Successfully designed, implemented, trained, and tested Q-Learning and SARSA from first principles.
  - Verified 100% collision-free navigation matching the BFS theoretical optimal path.
  - Discovered and validated fundamental theoretical differences between on-policy and off-policy TD learning.
  - Delivered comprehensive code, unit tests (16/16 passing), datasets (>140k rows), publication graphs, and interactive simulator.
- **Questions & Viva Discussion:** Open to faculty questions.

---

## Academic Viva Preparation Q&A

**Q1: Why does Q-Learning converge to the optimal path even with high exploration noise?**  
*Answer:* Q-Learning's update target is $R + \gamma \max_{a'} Q(S', a')$. Because it takes the maximum over all actions regardless of which action the behavior policy actually selects next, it is off-policy and directly estimates the optimal policy $Q^*$ under the assumption of greedy future actions.

**Q2: Why did SARSA fail to solve the Serpentine Maze in 1,500 episodes while Q-Learning succeeded?**  
*Answer:* SARSA is on-policy; its update target is $R + \gamma Q(S', A')$, where $A'$ is sampled with exploration probability $\epsilon$. In a narrow corridor, exploratory actions frequently hit the wall, incurring a $-50$ penalty. SARSA factors this penalty into the expected value of corridor cells, depressing their value and deterring the agent from entering deep corridors until $\epsilon$ becomes near zero.

**Q3: Why is BFS used as a baseline? Is it an identical-information comparison?**  
*Answer:* BFS is a deterministic graph search algorithm with complete access to the static map. It serves as an upper bound / ground truth for the minimal geodesic path (18 steps). It is not an identical-information comparison because RL agents have zero map knowledge and must discover paths through reward feedback.

**Q4: How does your reward function prevent infinite looping?**  
*Answer:* Every step incurs a step penalty of $-1.0$. If the agent loops or wanders, it accumulates negative return ($G_t \ll 0$). Reaching the goal terminates the episode and awards $+100.0$. Thus, policies that reach the goal in fewer steps strictly achieve higher returns.
