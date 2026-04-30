"""
environment/space_mission_env.py
---------------------------------
SpaceMissionEnv — custom Gymnasium environment for SpaceRL Version 3.

Render modes:
    - None   : no rendering (training)
    - "ansi" : text-based terminal render
    - "human": pygame visual window (demo mode)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import (
    N_LEVELS, N_EVENTS, N_STATES, N_ACTIONS, MAX_STEPS, MISSION_SUCCESS_STEPS,
    EVENT_NORMAL, EVENT_STORM, EVENT_ASTEROID, EVENT_SAFE_ZONE,
    ACTION_LIFE_SUPPORT, ACTION_PROPULSION, ACTION_GATHER,
    ACTION_REPAIR, ACTION_POWER_SAVING, ACTION_SAFE_ZONE,
    ACTION_NAMES,
    REWARD_SURVIVE, REWARD_BALANCED, REWARD_SAFE_ZONE,
    REWARD_CRITICAL_LOW, REWARD_SEVERE_DAMAGE,
    REWARD_MISSION_FAILED, REWARD_MISSION_SUCCESS,
)


class SpaceMissionEnv(gym.Env):
    """
    Custom Gymnasium environment: autonomous resource management
    aboard a spacecraft during a deep-space mission.
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 6}

    # ── Pygame constants ──────────────────────────────────────────────────────
    _W, _H      = 700, 480
    _FPS        = 6
    _BLACK      = (10,  10,  20)
    _WHITE      = (240, 240, 250)
    _DARK_PANEL = (25,  30,  50)
    _ACCENT     = (80,  140, 220)
    _GREEN      = (60,  200, 100)
    _YELLOW     = (240, 200,  40)
    _RED        = (220,  60,  60)
    _ORANGE     = (230, 130,  30)
    _PURPLE     = (150,  80, 220)
    _CYAN       = (40,  200, 200)
    _GREY       = (100, 110, 130)

    def __init__(self, render_mode: str = None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Discrete(N_STATES)
        self.action_space      = spaces.Discrete(N_ACTIONS)

        self._energy = 1; self._oxygen = 1; self._food = 1
        self._fuel   = 1; self._hull   = 1; self._event = EVENT_NORMAL
        self._step_count    = 0
        self._success_steps = 0
        self._last_action   = 0
        self._last_reward   = 0.0
        self._total_reward  = 0.0
        self.episode_info   = {}

        # Pygame
        self._screen = None
        self._clock  = None
        self._font_l = None
        self._font_m = None
        self._font_s = None

    # ── Encode / Decode ───────────────────────────────────────────────────────

    def _encode_state(self) -> int:
        return int(
            self._event +
            N_EVENTS * (self._hull +
            N_LEVELS * (self._fuel +
            N_LEVELS * (self._food +
            N_LEVELS * (self._oxygen +
            N_LEVELS *  self._energy))))
        )

    @staticmethod
    def decode_state(state: int) -> dict:
        event  = state % N_EVENTS;  state //= N_EVENTS
        hull   = state % N_LEVELS;  state //= N_LEVELS
        fuel   = state % N_LEVELS;  state //= N_LEVELS
        food   = state % N_LEVELS;  state //= N_LEVELS
        oxygen = state % N_LEVELS;  state //= N_LEVELS
        energy = state
        ll = {0: "critical/low", 1: "stable/mid", 2: "optimal/high"}
        el = {EVENT_NORMAL: "Normal", EVENT_STORM: "Solar Storm",
              EVENT_ASTEROID: "Asteroid Impact", EVENT_SAFE_ZONE: "Safe Zone"}
        return {"energy": ll[energy], "oxygen": ll[oxygen], "food": ll[food],
                "fuel": ll[fuel], "hull": ll[hull], "event": el[event]}

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._energy = 1; self._oxygen = 1; self._food = 1
        self._fuel   = 1; self._hull   = 1; self._event = EVENT_NORMAL
        self._step_count    = 0
        self._success_steps = 0
        self._last_action   = 0
        self._last_reward   = 0.0
        self._total_reward  = 0.0
        self.episode_info   = {"total_reward": 0.0, "steps": 0,
                               "terminated_by": None, "mission_success": False}
        if self.render_mode == "human":
            self._init_pygame()
        return self._encode_state(), {}

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: int):
        assert self.action_space.contains(action)
        self._step_count  += 1
        self._last_action  = action

        self._apply_action(action)
        self._apply_event()
        self._event = self._sample_event()
        self._clamp_resources()

        reward, terminated, reason = self._compute_reward(action)
        truncated = self._step_count >= MAX_STEPS

        self._last_reward   = reward
        self._total_reward += reward
        self.episode_info["total_reward"] += reward
        self.episode_info["steps"]         = self._step_count
        if terminated:
            self.episode_info["terminated_by"]   = reason
            self.episode_info["mission_success"] = (reason == "success")

        obs  = self._encode_state()
        info = {"step": self._step_count, "action_name": ACTION_NAMES[action],
                "energy": self._energy, "oxygen": self._oxygen,
                "food": self._food, "fuel": self._fuel, "hull": self._hull,
                "event": self._event,
                "mission_success": self.episode_info.get("mission_success", False)}

        if self.render_mode == "human":
            self._render_pygame()
        elif self.render_mode == "ansi":
            self._render_ansi()

        return obs, reward, terminated, truncated, info

    # ── Dynamics ──────────────────────────────────────────────────────────────

    def _apply_action(self, action):
        if action == ACTION_LIFE_SUPPORT:
            self._oxygen = min(2, self._oxygen + 1)
            self._food   = min(2, self._food + 1)
            self._energy = max(0, self._energy - 1)
        elif action == ACTION_PROPULSION:
            self._fuel = max(0, self._fuel - 1)
        elif action == ACTION_GATHER:
            self._food = min(2, self._food + 1)
            self._fuel = min(2, self._fuel + 1)
        elif action == ACTION_REPAIR:
            self._hull   = min(2, self._hull + 1)
            self._energy = max(0, self._energy - 1)
        elif action == ACTION_POWER_SAVING:
            self._energy = min(2, self._energy + 1)
        elif action == ACTION_SAFE_ZONE:
            self._fuel   = max(0, self._fuel - 1)
            self._energy = min(2, self._energy + 1)
            self._oxygen = min(2, self._oxygen + 1)
            self._food   = min(2, self._food + 1)

    def _apply_event(self):
        if self._event == EVENT_STORM:
            self._energy = max(0, self._energy - 1)
            if np.random.random() < 0.15: self._hull = max(0, self._hull - 1)
        elif self._event == EVENT_ASTEROID:
            self._hull = max(0, self._hull - 1)
            if np.random.random() < 0.20: self._oxygen = max(0, self._oxygen - 1)
        elif self._event == EVENT_SAFE_ZONE:
            self._energy = min(2, self._energy + 1)
            self._food   = min(2, self._food + 1)
        if self._event == EVENT_NORMAL:
            if np.random.random() < 0.15: self._energy = max(0, self._energy - 1)
            if np.random.random() < 0.10: self._oxygen = max(0, self._oxygen - 1)
            if np.random.random() < 0.08: self._food   = max(0, self._food - 1)

    def _sample_event(self):
        return int(np.random.choice(
            [EVENT_NORMAL, EVENT_STORM, EVENT_ASTEROID, EVENT_SAFE_ZONE],
            p=[0.70, 0.15, 0.05, 0.10]
        ))

    def _clamp_resources(self):
        self._energy = int(np.clip(self._energy, 0, 2))
        self._oxygen = int(np.clip(self._oxygen, 0, 2))
        self._food   = int(np.clip(self._food,   0, 2))
        self._fuel   = int(np.clip(self._fuel,   0, 2))
        self._hull   = int(np.clip(self._hull,   0, 2))

    def _compute_reward(self, action):
        if self._energy == 0: return REWARD_MISSION_FAILED, True, "energy_depleted"
        if self._oxygen == 0: return REWARD_MISSION_FAILED, True, "oxygen_depleted"
        if self._hull   == 0: return REWARD_MISSION_FAILED, True, "hull_destroyed"

        reward = REWARD_SURVIVE
        resources = [self._energy, self._oxygen, self._food, self._fuel]
        if all(r >= 1 for r in resources) and self._hull >= 1:
            reward += REWARD_BALANCED
            self._success_steps += 1
        if any(r == 0 for r in resources): reward += REWARD_CRITICAL_LOW
        if self._hull == 0:                reward += REWARD_SEVERE_DAMAGE
        if action == ACTION_SAFE_ZONE and self._event == EVENT_SAFE_ZONE:
            reward += REWARD_SAFE_ZONE
        if self._success_steps >= MISSION_SUCCESS_STEPS:
            return reward + REWARD_MISSION_SUCCESS, True, "success"

        return reward, False, None

    # ── Pygame render ─────────────────────────────────────────────────────────

    def _init_pygame(self):
        if self._screen is not None:
            return
        import pygame
        pygame.init()
        self._screen = pygame.display.set_mode((self._W, self._H))
        pygame.display.set_caption("SpaceRL — Mission Control")
        self._clock  = pygame.time.Clock()
        self._font_l = pygame.font.SysFont("monospace", 20, bold=True)
        self._font_m = pygame.font.SysFont("monospace", 15)
        self._font_s = pygame.font.SysFont("monospace", 13)

    def _render_pygame(self):
        import pygame
        if self._screen is None:
            self._init_pygame()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.close(); return

        self._screen.fill(self._BLACK)
        self._draw_stars()
        self._draw_header()
        self._draw_resources()
        self._draw_event_panel()
        self._draw_action_panel()
        self._draw_mission_progress()
        self._draw_ship()
        pygame.display.flip()
        self._clock.tick(self._FPS)

    def _draw_stars(self):
        import pygame
        rng = np.random.RandomState(42)
        for _ in range(80):
            x = rng.randint(0, self._W)
            y = rng.randint(0, self._H)
            pygame.draw.circle(self._screen, (160, 160, 190), (x, y), rng.randint(1, 2))

    def _draw_header(self):
        import pygame
        pygame.draw.rect(self._screen, self._DARK_PANEL, (0, 0, self._W, 48))
        pygame.draw.line(self._screen, self._ACCENT, (0, 48), (self._W, 48), 2)
        self._screen.blit(
            self._font_l.render("SpaceRL — Mission Control", True, self._WHITE), (20, 13))
        self._screen.blit(
            self._font_m.render(
                f"Step:{self._step_count:>4}   Reward:{self._last_reward:>+6.1f}   Total:{self._total_reward:>7.1f}",
                True, self._CYAN), (360, 15))

    def _draw_resources(self):
        import pygame
        px, py = 20, 62
        pygame.draw.rect(self._screen, self._DARK_PANEL, (px-10, py-10, 310, 290), border_radius=8)
        pygame.draw.rect(self._screen, self._ACCENT,     (px-10, py-10, 310, 290), width=1, border_radius=8)
        self._screen.blit(self._font_m.render("SHIP RESOURCES", True, self._ACCENT), (px, py))

        items = [
            ("E  ENERGY",  self._energy, self._YELLOW),
            ("O  OXYGEN",  self._oxygen, self._CYAN),
            ("F  FOOD",    self._food,   self._GREEN),
            ("R  FUEL",    self._fuel,   self._ORANGE),
            ("H  HULL",    self._hull,   self._PURPLE),
        ]
        for i, (label, val, color) in enumerate(items):
            y = py + 28 + i * 48
            self._screen.blit(self._font_s.render(label, True, self._WHITE), (px, y))
            pygame.draw.rect(self._screen, self._GREY, (px, y+18, 260, 18), border_radius=4)
            fw = int(260 * val / 2)
            if fw > 0:
                pygame.draw.rect(self._screen,
                                 color if val >= 1 else self._RED,
                                 (px, y+18, fw, 18), border_radius=4)
            lvl = {0: "CRITICAL", 1: "STABLE", 2: "OPTIMAL"}[val]
            self._screen.blit(
                self._font_s.render(lvl, True, self._RED if val == 0 else self._WHITE),
                (px + 175, y + 19))

    def _draw_event_panel(self):
        import pygame
        px, py = 350, 62
        ev_data = {
            EVENT_NORMAL   : ("NORMAL",       self._GREEN,  "All systems nominal"),
            EVENT_STORM    : ("SOLAR STORM",  self._YELLOW, "Energy draining fast!"),
            EVENT_ASTEROID : ("ASTEROID",     self._RED,    "Hull under impact!"),
            EVENT_SAFE_ZONE: ("SAFE ZONE",    self._CYAN,   "Resources recovering"),
        }
        name, color, desc = ev_data[self._event]
        pygame.draw.rect(self._screen, self._DARK_PANEL, (px-10, py-10, 330, 115), border_radius=8)
        pygame.draw.rect(self._screen, color,            (px-10, py-10, 330, 115), width=2, border_radius=8)
        self._screen.blit(self._font_s.render("CURRENT EVENT", True, self._GREY), (px, py))
        self._screen.blit(self._font_l.render(name, True, color), (px, py+20))
        self._screen.blit(self._font_s.render(desc, True, self._WHITE), (px, py+50))
        self._screen.blit(self._font_s.render(f"Step: {self._step_count}", True, self._GREY), (px, py+75))

    def _draw_action_panel(self):
        import pygame
        px, py = 350, 205
        pygame.draw.rect(self._screen, self._DARK_PANEL, (px-10, py-10, 330, 160), border_radius=8)
        pygame.draw.rect(self._screen, self._ACCENT,     (px-10, py-10, 330, 160), width=1, border_radius=8)
        self._screen.blit(self._font_s.render("LAST ACTION", True, self._GREY), (px, py))

        act_colors = [self._CYAN, self._ORANGE, self._GREEN, self._PURPLE, self._YELLOW, self._ACCENT]
        self._screen.blit(
            self._font_l.render(ACTION_NAMES[self._last_action], True, act_colors[self._last_action]),
            (px, py + 22))

        for i, name in enumerate(ACTION_NAMES):
            col = act_colors[i] if i == self._last_action else self._GREY
            prefix = "> " if i == self._last_action else "  "
            self._screen.blit(self._font_s.render(f"{prefix}{i}: {name}", True, col), (px, py+55+i*16))

    def _draw_mission_progress(self):
        import pygame
        px, py = 20, 368
        pygame.draw.rect(self._screen, self._DARK_PANEL, (px-10, py-10, 660, 90), border_radius=8)
        pygame.draw.rect(self._screen, self._ACCENT,     (px-10, py-10, 660, 90), width=1, border_radius=8)
        self._screen.blit(self._font_s.render("MISSION PROGRESS", True, self._GREY), (px, py))

        progress = min(1.0, self._success_steps / MISSION_SUCCESS_STEPS)
        pygame.draw.rect(self._screen, self._GREY, (px, py+20, 580, 20), border_radius=5)
        fw = int(580 * progress)
        if fw > 0:
            pygame.draw.rect(self._screen,
                             self._GREEN if progress >= 1.0 else self._ACCENT,
                             (px, py+20, fw, 20), border_radius=5)
        self._screen.blit(
            self._font_m.render(
                f"{self._success_steps}/{MISSION_SUCCESS_STEPS} balanced steps  ({progress*100:.0f}%)",
                True, self._WHITE), (px, py+48))

    def _draw_ship(self):
        import pygame
        cx, cy = 630, 220
        pygame.draw.polygon(self._screen, self._ACCENT,
                            [(cx, cy-28), (cx-14, cy+18), (cx+14, cy+18)])
        pygame.draw.polygon(self._screen, self._GREY,
                            [(cx-14, cy+8), (cx-32, cy+28), (cx-14, cy+23)])
        pygame.draw.polygon(self._screen, self._GREY,
                            [(cx+14, cy+8), (cx+32, cy+28), (cx+14, cy+23)])
        glow = self._ORANGE if self._last_action == ACTION_PROPULSION else self._YELLOW
        pygame.draw.ellipse(self._screen, glow, (cx-7, cy+18, 14, 9))
        if self._hull <= 1:
            col = self._RED if self._hull == 0 else self._YELLOW
            lbl = "HULL CRIT" if self._hull == 0 else "HULL DMG"
            self._screen.blit(self._font_s.render(lbl, True, col), (cx-28, cy+38))

    # ── ANSI render ───────────────────────────────────────────────────────────

    def _render_ansi(self):
        ev = {EVENT_NORMAL: "Normal", EVENT_STORM: "Solar Storm",
              EVENT_ASTEROID: "Asteroid", EVENT_SAFE_ZONE: "Safe Zone"}
        def bar(v): return {0:"░░  CRITICAL", 1:"██░ Stable  ", 2:"███ Optimal "}[v]
        print(f"\n╔══════════════════════════════════════╗")
        print(f"║  SpaceRL Mission — Step {self._step_count:>4}          ║")
        print(f"╠══════════════════════════════════════╣")
        print(f"║  Event  : {ev[self._event]:<28}║")
        print(f"╠══════════════════════════════════════╣")
        print(f"║  Energy : {bar(self._energy)}                  ║")
        print(f"║  Oxygen : {bar(self._oxygen)}                  ║")
        print(f"║  Food   : {bar(self._food)}                  ║")
        print(f"║  Fuel   : {bar(self._fuel)}                  ║")
        print(f"║  Hull   : {bar(self._hull)}                  ║")
        print(f"╠══════════════════════════════════════╣")
        print(f"║  Progress: {self._success_steps:>3}/{MISSION_SUCCESS_STEPS}  Action: {ACTION_NAMES[self._last_action]:<12}║")
        print(f"╚══════════════════════════════════════╝")

    def render(self):
        if self.render_mode == "human":
            self._render_pygame()
        elif self.render_mode == "ansi":
            self._render_ansi()

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None