"""Pygame-based graphical visualizer and interactive demonstration suite.

Provides real-time interactive simulation of random, Q-Learning, and SARSA agents
with HUD telemetry, path tracing, speed adjustment, and headless safety.
"""

import os
import sys
import time
from typing import List, Optional, Tuple
import numpy as np

# Ensure pygame does not crash in headless environments
try:
    import pygame
except ImportError:
    pygame = None

from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SARSAAgent
from config import (
    EnvConfig,
    RLConfig,
    PathConfig,
    ACTION_NAMES,
    ACTION_DELTAS,
)
from environment.gridworld import GridWorldEnv


class PygameSimulator:
    """Interactive visual simulation of robot navigation in Pygame."""

    # Color Palette (Modern Academic Dark UI)
    COLOR_BG = (15, 23, 42)          # Slate 900
    COLOR_PANEL = (30, 41, 59)       # Slate 800
    COLOR_GRID = (51, 65, 85)        # Slate 700
    COLOR_CELL = (24, 33, 47)        # Slate 850
    COLOR_OBSTACLE = (100, 116, 139) # Slate 500
    COLOR_OBSTACLE_BORDER = (239, 68, 68) # Red 500
    COLOR_START = (16, 185, 129)     # Emerald 500
    COLOR_GOAL = (245, 158, 11)      # Amber 500
    COLOR_ROBOT = (56, 189, 248)     # Sky 400
    COLOR_ROBOT_SARSA = (244, 63, 94) # Rose 500
    COLOR_PATH_Q = (96, 165, 250)    # Blue 400
    COLOR_PATH_S = (251, 113, 133)   # Rose 400
    COLOR_TEXT = (241, 245, 249)     # Slate 100
    COLOR_TEXT_MUTED = (148, 163, 184)# Slate 400
    COLOR_ACCENT = (129, 140, 248)   # Indigo 400

    def __init__(
        self,
        env_config: Optional[EnvConfig] = None,
        path_config: Optional[PathConfig] = None,
        cell_size: int = 56,
        margin: int = 30,
        fps: int = 8,
    ):
        self.env_cfg = env_config if env_config is not None else EnvConfig()
        self.path_cfg = path_config if path_config is not None else PathConfig()
        self.cell_size = cell_size
        self.margin = margin
        self.fps = fps
        self.paused = False

        # Mode definitions:
        # 1: Random Agent
        # 2: Trained Q-Learning Agent
        # 3: Trained SARSA Agent
        # 4: Sequential Path Comparison
        self.mode = 2
        self.mode_names = {
            1: "Mode A: Random / Untrained Agent",
            2: "Mode B: Trained Q-Learning Agent",
            3: "Mode C: Trained SARSA Agent",
            4: "Mode D: Comparative Path Execution",
        }

        # Window dimensions
        self.grid_pixel_w = self.env_cfg.grid_width * self.cell_size
        self.grid_pixel_h = self.env_cfg.grid_height * self.cell_size
        self.panel_width = 340
        self.screen_width = self.grid_pixel_w + self.margin * 2 + self.panel_width
        self.screen_height = self.grid_pixel_h + self.margin * 2

        self.env = GridWorldEnv(self.env_cfg)
        self.q_agent = self._load_agent("q_learning")
        self.s_agent = self._load_agent("sarsa")

        # Runtime telemetry
        self.current_state = self.env.start_pos
        self.current_reward = 0.0
        self.total_reward = 0.0
        self.collisions = 0
        self.status_msg = "Ready to Navigate"
        self.active_path = [self.current_state]

        # Pygame state
        self.screen = None
        self.clock = None
        self.font = None
        self.font_small = None
        self.font_bold = None

    def _load_agent(self, algo: str):
        """Safely load agent weights or initialize fresh agent if not trained yet."""
        if algo == "q_learning":
            agent = QLearningAgent(self.env_cfg.grid_width, self.env_cfg.grid_height)
            if os.path.exists(self.path_cfg.q_learning_model_path):
                try:
                    agent.load(self.path_cfg.q_learning_model_path)
                except Exception as e:
                    print(f"Warning: Could not load Q-Learning model ({e}). Using fresh agent.")
            return agent
        else:
            agent = SARSAAgent(self.env_cfg.grid_width, self.env_cfg.grid_height)
            if os.path.exists(self.path_cfg.sarsa_model_path):
                try:
                    agent.load(self.path_cfg.sarsa_model_path)
                except Exception as e:
                    print(f"Warning: Could not load SARSA model ({e}). Using fresh agent.")
            return agent

    def _init_pygame(self) -> bool:
        """Initialize Pygame display safely."""
        if pygame is None:
            print("Pygame is not installed. Graphical simulator unavailable.")
            return False

        # Attempt standard visual init
        try:
            pygame.init()
            pygame.display.set_caption("UCS760 Autonomous Robot Navigation: Q-Learning & SARSA")
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Helvetica, Arial", 16)
            self.font_small = pygame.font.SysFont("Helvetica, Arial", 13)
            self.font_bold = pygame.font.SysFont("Helvetica, Arial", 18, bold=True)
            return True
        except Exception as e:
            print(f"Graphical display not available ({e}). Falling back to dummy video driver.")
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            try:
                pygame.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                self.clock = pygame.time.Clock()
                self.font = pygame.font.SysFont(None, 16)
                self.font_small = pygame.font.SysFont(None, 13)
                self.font_bold = pygame.font.SysFont(None, 18)
                return True
            except Exception as dummy_err:
                print(f"Failed to initialize dummy Pygame: {dummy_err}")
                return False

    def reset_episode(self) -> None:
        """Reset environment and state for the active mode."""
        self.current_state, _ = self.env.reset()
        self.current_reward = 0.0
        self.total_reward = 0.0
        self.collisions = 0
        self.status_msg = f"Running {self.mode_names[self.mode].split(':')[1].strip()}"
        self.active_path = [self.current_state]

    def _get_next_action(self) -> int:
        """Query action according to current active mode."""
        if self.mode == 1:
            # Mode A: Random agent
            return int(np.random.randint(0, 4))
        elif self.mode == 2:
            # Mode B: Trained Q-Learning
            return self.q_agent.choose_action(self.current_state, greedy=True)
        elif self.mode == 3:
            # Mode C: Trained SARSA
            return self.s_agent.choose_action(self.current_state, greedy=True)
        elif self.mode == 4:
            # Mode D: Sequential comparative (alternates between Q and SARSA on finish)
            return self.q_agent.choose_action(self.current_state, greedy=True)
        return 0

    def step_simulation(self) -> bool:
        """Advance simulation by one discrete action step."""
        if self.env.is_goal(self.current_state) or self.env.step_count >= self.env.max_steps:
            return False

        action = self._get_next_action()
        next_s, reward, done, info = self.env.step(action)

        self.current_state = next_s
        self.current_reward = reward
        self.total_reward += reward
        self.active_path.append(self.current_state)

        if info["collision"]:
            self.collisions += 1
            self.status_msg = f"Collision! {info['action_name']}"
        elif info["success"]:
            self.status_msg = f"SUCCESS! Goal reached in {info['step']} steps"
        elif info["truncated"]:
            self.status_msg = f"Truncated (max {self.env.max_steps} steps)"
        else:
            self.status_msg = f"Step {info['step']}: {info['action_name']}"

        return done

    def draw(self) -> None:
        """Render complete visual scene with grid, agent, obstacles, and HUD."""
        self.screen.fill(self.COLOR_BG)

        # 1. Draw Grid Area
        grid_origin_x = self.margin
        grid_origin_y = self.margin

        for x in range(self.env.width):
            for y in range(self.env.height):
                rect = pygame.Rect(
                    grid_origin_x + x * self.cell_size,
                    grid_origin_y + y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )

                # Base cell
                pygame.draw.rect(self.screen, self.COLOR_CELL, rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

                cell_pos = (x, y)

                # Obstacles
                if cell_pos in self.env.obstacles:
                    obs_inner = rect.inflate(-4, -4)
                    pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs_inner, border_radius=4)
                    pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_BORDER, obs_inner, 2, border_radius=4)
                # Start
                elif cell_pos == self.env.start_pos:
                    start_inner = rect.inflate(-6, -6)
                    pygame.draw.rect(self.screen, self.COLOR_START, start_inner, border_radius=6)
                    txt = self.font_bold.render("S", True, (255, 255, 255))
                    self.screen.blit(txt, txt.get_rect(center=rect.center))
                # Goal
                elif cell_pos == self.env.goal_pos:
                    goal_inner = rect.inflate(-6, -6)
                    pygame.draw.rect(self.screen, self.COLOR_GOAL, goal_inner, border_radius=6)
                    txt = self.font_bold.render("G", True, (0, 0, 0))
                    self.screen.blit(txt, txt.get_rect(center=rect.center))

        # 2. Draw Visited Path Trajectory
        if len(self.active_path) > 1:
            pts = []
            for px, py in self.active_path:
                cx = grid_origin_x + px * self.cell_size + self.cell_size // 2
                cy = grid_origin_y + py * self.cell_size + self.cell_size // 2
                pts.append((cx, cy))

            path_col = self.COLOR_PATH_S if self.mode == 3 else self.COLOR_PATH_Q
            for i in range(len(pts) - 1):
                pygame.draw.line(self.screen, path_col, pts[i], pts[i + 1], 3)
                pygame.draw.circle(self.screen, path_col, pts[i], 4)

        # 3. Draw Robot
        rx, ry = self.current_state
        robot_center = (
            grid_origin_x + rx * self.cell_size + self.cell_size // 2,
            grid_origin_y + ry * self.cell_size + self.cell_size // 2,
        )
        robot_col = self.COLOR_ROBOT_SARSA if self.mode == 3 else self.COLOR_ROBOT
        pygame.draw.circle(self.screen, robot_col, robot_center, self.cell_size // 3)
        pygame.draw.circle(self.screen, (255, 255, 255), robot_center, self.cell_size // 3, 2)
        # Inner pupil
        pygame.draw.circle(self.screen, (15, 23, 42), robot_center, 4)

        # 4. Draw Right Telemetry Panel
        panel_x = grid_origin_x + self.grid_pixel_w + self.margin
        panel_y = self.margin
        panel_h = self.grid_pixel_h
        panel_rect = pygame.Rect(panel_x, panel_y, self.panel_width, panel_h)

        pygame.draw.rect(self.screen, self.COLOR_PANEL, panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, self.COLOR_GRID, panel_rect, 1, border_radius=8)

        # Dashboard Title
        title_surf = self.font_bold.render("UCS760 RL Dashboard", True, self.COLOR_ACCENT)
        self.screen.blit(title_surf, (panel_x + 16, panel_y + 16))

        # Mode line
        mode_surf = self.font.render(self.mode_names[self.mode], True, self.COLOR_TEXT)
        self.screen.blit(mode_surf, (panel_x + 16, panel_y + 46))

        # Divider
        pygame.draw.line(self.screen, self.COLOR_GRID, (panel_x + 16, panel_y + 75), (panel_x + self.panel_width - 16, panel_y + 75), 1)

        # Telemetry metrics
        y_cursor = panel_y + 90
        metrics = [
            ("Current Position:", f"({self.current_state[0]}, {self.current_state[1]})"),
            ("Goal Target:", f"({self.env.goal_pos[0]}, {self.env.goal_pos[1]})"),
            ("Steps Elapsed:", f"{self.env.step_count} / {self.env.max_steps}"),
            ("Collisions:", f"{self.collisions}"),
            ("Last Reward:", f"{self.current_reward:+.1f}"),
            ("Total Cumulative Reward:", f"{self.total_reward:+.1f}"),
            ("Simulation Speed:", f"{self.fps} steps/sec"),
            ("Status:", self.status_msg),
        ]

        for label, val in metrics:
            lbl_surf = self.font_small.render(label, True, self.COLOR_TEXT_MUTED)
            val_surf = self.font.render(val, True, self.COLOR_TEXT)
            self.screen.blit(lbl_surf, (panel_x + 16, y_cursor))
            self.screen.blit(val_surf, (panel_x + 16, y_cursor + 16))
            y_cursor += 40

        # Controls & Hotkeys Legend
        pygame.draw.line(self.screen, self.COLOR_GRID, (panel_x + 16, y_cursor + 5), (panel_x + self.panel_width - 16, y_cursor + 5), 1)
        y_cursor += 15

        controls_title = self.font_bold.render("Interactive Controls:", True, self.COLOR_ACCENT)
        self.screen.blit(controls_title, (panel_x + 16, y_cursor))
        y_cursor += 24

        keys = [
            "[1] Untrained / Random Agent",
            "[2] Trained Q-Learning Agent",
            "[3] Trained SARSA Agent",
            "[4] Comparative Replay",
            "[Space] Pause / Resume Step",
            "[R] Reset Active Episode",
            "[+/-] Adjust Simulation Speed",
            "[Esc / Q] Exit Simulator",
        ]

        for k in keys:
            k_surf = self.font_small.render(k, True, self.COLOR_TEXT_MUTED)
            self.screen.blit(k_surf, (panel_x + 16, y_cursor))
            y_cursor += 20

        pygame.display.flip()

    def run(self, max_seconds: Optional[int] = None) -> None:
        """Main simulator event loop."""
        if not self._init_pygame():
            print("Cannot run interactive Pygame visualizer without display.")
            return

        self.reset_episode()
        running = True
        start_wall_time = time.time()
        last_step_time = time.time()

        print(f"\n==================================================")
        print(f"PYGAME SIMULATION RUNNING")
        print(f"Active Mode: {self.mode_names[self.mode]}")
        print(f"Controls: [1-4] switch mode | [Space] pause | [R] reset | [Esc] quit")
        print(f"==================================================")

        while running:
            # Handle user input events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self.reset_episode()
                    elif event.key == pygame.K_1:
                        self.mode = 1
                        self.reset_episode()
                    elif event.key == pygame.K_2:
                        self.mode = 2
                        self.reset_episode()
                    elif event.key == pygame.K_3:
                        self.mode = 3
                        self.reset_episode()
                    elif event.key == pygame.K_4:
                        self.mode = 4
                        self.reset_episode()
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        self.fps = min(60, self.fps + 2)
                    elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                        self.fps = max(1, self.fps - 2)

            # Advance simulation step based on FPS timing
            now = time.time()
            step_interval = 1.0 / max(1, self.fps)
            if not self.paused and (now - last_step_time >= step_interval):
                done = self.step_simulation()
                last_step_time = now
                if done:
                    # Brief pause at goal or termination, then auto-reset
                    self.draw()
                    time.sleep(0.8)
                    self.reset_episode()

            self.draw()
            self.clock.tick(60)  # Render loop at smooth 60 FPS while stepping at self.fps

            if max_seconds is not None and (now - start_wall_time) >= max_seconds:
                print(f"Auto-terminating simulation after {max_seconds} seconds.")
                break

        pygame.quit()
        print("Pygame simulator closed cleanly.")
