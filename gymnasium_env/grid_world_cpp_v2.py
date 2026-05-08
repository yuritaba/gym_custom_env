from typing import Optional
import numpy as np
import gymnasium as gym
import pygame

#
# Coverage Path Planning (CPP) environment v2 — improved state representation.
#
# Key differences from v1:
#   - Tracks a persistent partial map (seen_map) as the agent explores
#   - Exposes a fixed 7x7 local window of that map (translation-invariant,
#     works across grid sizes)
#   - Adds a 4D frontier vector: direction to nearest known-free-unvisited cell
#     and direction to nearest unknown cell — critical for escaping local loops
#
# Reward function (unchanged from v1):
#   +1.0  visiting a new cell
#   -0.3  revisiting an already-visited cell
#   -0.5  hitting a wall or obstacle (staying in place)
#   -0.1  step penalty (every action)
#   +10.0 bonus for full coverage
#   -5.0  penalty when max steps reached without full coverage
#
# seen_map encoding (internal):
#   0 = unknown (never in agent's 3×3 view)
#   1 = free and unvisited (agent was nearby, cell confirmed empty)
#   2 = obstacle or out-of-bounds wall
#   3 = visited by agent
#
# local_map in observation: 7×7 window of seen_map normalised to [0, 1] (÷3).
# frontier in observation: [dx_free, dy_free, dx_unknown, dy_unknown] in [-1, 1].
#

LOCAL_MAP_SIZE = 7


class GridWorldCPPEnvV2(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size: int = 5, obs_quantity: int = 3,
                 max_steps: int = 200):
        self.size = size
        self.window_size = 512
        self.obs_quantity = obs_quantity
        self.count_steps = 0
        self.max_steps = max_steps

        self.visited: set = set()
        self.obstacles_locations: list = []
        self._agent_location = np.array([-1, -1], dtype=int)
        self._obstacle_set: set = set()

        # partial map — reset each episode
        self.seen_map = np.zeros((size, size), dtype=np.int8)

        lm = LOCAL_MAP_SIZE
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(
                low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            ),
            "local_map": gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(lm, lm),
                dtype=np.float32,
            ),
            "frontier": gym.spaces.Box(
                low=-1.0, high=1.0,
                shape=(4,),
                dtype=np.float32,
            ),
        })

        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),    # right
            1: np.array([0, -1]),   # up
            2: np.array([-1, 0]),   # left
            3: np.array([0, 1]),    # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_free_cells(self):
        return self.size * self.size - len(self.obstacles_locations)

    @property
    def coverage_ratio(self):
        return len(self.visited) / self.total_free_cells if self.total_free_cells > 0 else 1.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_seen_map(self):
        """Update seen_map with everything visible in the 3×3 window around agent."""
        ax, ay = self._agent_location
        for di in range(-1, 2):
            for dj in range(-1, 2):
                nx, ny = ax + dj, ay + di
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    continue
                if (nx, ny) in self._obstacle_set:
                    self.seen_map[ny, nx] = 2
                elif (nx, ny) in self.visited:
                    self.seen_map[ny, nx] = 3
                else:
                    self.seen_map[ny, nx] = 1

    def _get_local_map(self) -> np.ndarray:
        """7×7 window of seen_map centred on agent, normalised to [0,1]."""
        half = LOCAL_MAP_SIZE // 2
        ax, ay = self._agent_location
        local = np.zeros((LOCAL_MAP_SIZE, LOCAL_MAP_SIZE), dtype=np.float32)
        for i in range(LOCAL_MAP_SIZE):
            for j in range(LOCAL_MAP_SIZE):
                nx = ax + (j - half)
                ny = ay + (i - half)
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    local[i, j] = self.seen_map[ny, nx] / 3.0
                else:
                    local[i, j] = 2.0 / 3.0  # treat out-of-bounds as wall
        return local

    def _get_frontier(self) -> np.ndarray:
        """
        Return (dx_free, dy_free, dx_unknown, dy_unknown) as normalised vectors
        pointing toward the nearest known-free-unvisited cell and the nearest
        unknown cell in seen_map.  Each component is in [-1, 1].
        Vectorised with numpy for speed on larger grids.
        """
        ax, ay = int(self._agent_location[0]), int(self._agent_location[1])
        max_dist = max(self.size - 1, 1)

        ny_arr, nx_arr = np.mgrid[0:self.size, 0:self.size]
        dist = np.abs(nx_arr - ax) + np.abs(ny_arr - ay)

        free_mask = (self.seen_map == 1)
        if free_mask.any():
            idx = np.unravel_index(np.where(free_mask, dist, self.size * 2).argmin(), dist.shape)
            fdx, fdy = (idx[1] - ax) / max_dist, (idx[0] - ay) / max_dist
        else:
            fdx, fdy = 0.0, 0.0

        unk_mask = (self.seen_map == 0)
        if unk_mask.any():
            idx = np.unravel_index(np.where(unk_mask, dist, self.size * 2).argmin(), dist.shape)
            udx, udy = (idx[1] - ax) / max_dist, (idx[0] - ay) / max_dist
        else:
            udx, udy = 0.0, 0.0

        return np.array([fdx, fdy, udx, udy], dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def _get_obs(self):
        norm = max(self.size - 1, 1)
        return {
            "agent": np.array([
                self._agent_location[0] / norm,
                self._agent_location[1] / norm,
                self.coverage_ratio,
            ], dtype=np.float32),
            "local_map": self._get_local_map(),
            "frontier": self._get_frontier(),
        }

    def _get_info(self):
        return {
            "coverage": self.coverage_ratio,
            "visited_cells": len(self.visited),
            "total_free_cells": self.total_free_cells,
            "steps": self.count_steps,
            "size": self.size,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.count_steps = 0
        self.obstacles_locations = []
        self._obstacle_set = set()
        self.visited = set()
        self.seen_map = np.zeros((self.size, self.size), dtype=np.int8)

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        for _ in range(self.obs_quantity):
            loc = self._agent_location.copy()
            while (np.array_equal(loc, self._agent_location) or
                   tuple(loc) in self._obstacle_set):
                loc = self.np_random.integers(0, self.size, size=2, dtype=int)
            self.obstacles_locations.append(loc)
            self._obstacle_set.add(tuple(loc))

        self.visited.add(tuple(self._agent_location))
        self._update_seen_map()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        direction = self._action_to_direction[action]
        old_location = self._agent_location.copy()

        new_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        if tuple(new_location) in self._obstacle_set:
            self._agent_location = old_location
        else:
            self._agent_location = new_location

        self.count_steps += 1

        current_pos = tuple(self._agent_location)
        is_new_cell = current_pos not in self.visited
        stayed_in_place = np.array_equal(self._agent_location, old_location)

        reward = -0.1

        if stayed_in_place:
            reward -= 0.5
        elif is_new_cell:
            reward += 1.0
            self.visited.add(current_pos)
        else:
            reward -= 0.3

        self._update_seen_map()

        full_coverage = len(self.visited) >= self.total_free_cells
        terminated = full_coverage

        if full_coverage:
            reward += 10.0

        truncated = (self.count_steps >= self.max_steps) and not terminated
        if truncated:
            reward -= 5.0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix = self.window_size / self.size

        for cell in self.visited:
            pygame.draw.rect(canvas, (144, 238, 144),
                             pygame.Rect(pix * np.array(cell), (pix, pix)))

        for obs in self.obstacles_locations:
            pygame.draw.rect(canvas, (0, 0, 0),
                             pygame.Rect(pix * obs, (pix, pix)))

        pygame.draw.circle(canvas, (0, 0, 255),
                           (self._agent_location + 0.5) * pix, pix / 3)

        if pygame.font.get_init() or (pygame.init() and pygame.font.get_init()):
            font = pygame.font.SysFont(None, 24)
            text = font.render(
                f"Coverage: {self.coverage_ratio:.1%} | Steps: {self.count_steps}",
                True, (0, 0, 0))
            canvas.blit(text, (5, 5))

        for x in range(self.size + 1):
            pygame.draw.line(canvas, 0, (0, pix * x), (self.window_size, pix * x), width=3)
            pygame.draw.line(canvas, 0, (pix * x, 0), (pix * x, self.window_size), width=3)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
