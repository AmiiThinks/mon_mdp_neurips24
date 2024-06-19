import numpy as np
import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from typing import Optional
from collections import defaultdict

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STAY = 4

# state IDs
EMPTY = -1  # one-direction cell have ID corresponding to the allowed action (0, 1, 2, 3)
QCKSND = -2
GOOD_SMALL = 9
GOOD = 10
BAD = 11
BAD_SMALL = 12
WALL = -3

REWARDS = defaultdict(lambda: 0)
REWARDS[GOOD] = 1
REWARDS[BAD] = -10
REWARDS[GOOD_SMALL] = 0.1
REWARDS[BAD_SMALL] = -0.1

# rendering colors
RED = (255, 0, 0)
PALE_RED = (155, 0, 0)
GREEN = (0, 255, 0)
PALE_GREEN = (0, 155, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 175, 0)
PALE_YELLOW = (255, 255, 155)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)

GRIDS = {
    "river_swim_6": [[EMPTY for _ in range(6)]],
    "20_straight": [[EMPTY for _ in range(20)]],
    "2x2_empty": [
        [EMPTY, EMPTY],
        [EMPTY, GOOD],
    ],
    "3x3_empty": [
        [EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, GOOD],
    ],
    "3x3_empty_loop": [
        [EMPTY, LEFT, EMPTY],
        [EMPTY, RIGHT, UP],
        [EMPTY, EMPTY, GOOD],
    ],
    "3x3_penalty": [
        [EMPTY, BAD, GOOD],
        [EMPTY, BAD, EMPTY],
        [EMPTY, EMPTY, EMPTY],
    ],
    "10x10_empty": [[EMPTY for _ in range(10)] for _ in range(10)],
    "6x6_distract": [[EMPTY for _ in range(6)] for _ in range(6)],
    "4x4_quicksand": [
        [EMPTY, EMPTY, BAD, GOOD],
        [EMPTY, EMPTY, BAD, EMPTY],
        [EMPTY, QCKSND, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY],
    ],
    "4x4_quicksand_distract": [
        [EMPTY, GOOD_SMALL, BAD, GOOD],
        [EMPTY, BAD, EMPTY, EMPTY],
        [EMPTY, QCKSND, GOOD_SMALL, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY],
    ],
    "5x5_full": [
        [EMPTY, EMPTY, GOOD_SMALL, BAD, GOOD],
        [EMPTY, EMPTY, BAD, EMPTY, EMPTY],
        [RIGHT, EMPTY, QCKSND, GOOD_SMALL, EMPTY],
        [UP, EMPTY, EMPTY, EMPTY, EMPTY],
    ],
    "3x5_two_room_quicksand": [
        [EMPTY, EMPTY, LEFT, EMPTY, GOOD],
        [EMPTY, EMPTY, QCKSND, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    ],
    "3x4_corridor": [
        [EMPTY, LEFT, LEFT, LEFT],
        [GOOD_SMALL, BAD_SMALL, BAD_SMALL, GOOD],
        [EMPTY, LEFT, LEFT, LEFT],
    ],
    "2x11_two_room_distract": [
        [GOOD_SMALL, EMPTY, EMPTY, EMPTY, RIGHT, DOWN, LEFT, EMPTY, EMPTY, EMPTY, GOOD],
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    ],
    "5x5_barrier": [
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, LEFT, UP, RIGHT, EMPTY],
        [EMPTY, LEFT, GOOD, EMPTY, EMPTY],
        [EMPTY, LEFT, DOWN, RIGHT, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    ],
}

GRIDS["10x10_empty"][-1][-1] = GOOD
GRIDS["20_straight"][-1][-1] = GOOD
GRIDS["6x6_distract"][-1][-1] = GOOD
GRIDS["6x6_distract"][-1][0] = GOOD_SMALL

GRIDS["river_swim_6"][-1][-1] = GOOD
GRIDS["river_swim_6"][0][0] = GOOD_SMALL


def _move(row, col, a, nrow, ncol):
    if a == LEFT:
        col = max(col - 1, 0)
    elif a == DOWN:
        row = min(row + 1, nrow - 1)
    elif a == RIGHT:
        col = min(col + 1, ncol - 1)
    elif a == UP:
        row = max(row - 1, 0)
    elif a == STAY:
        pass
    else:
        raise ValueError("illegal action")
    return (row, col)


class Gridworld(gym.Env):
    """
    Gridworld where the agent has to reach a goal while avoid penalty cells.
    Harder versions include:
    - Quicksand cells, where the agent gets stuck with 90% probability,
    - Distracting rewards,
    - One-direction cells, where the agent can only move in one direction
    (all other actions will fail).

    ## Grid
    The grid is defined by a 2D array of integers. It is possible to define
    custom grids.

    ## Action Space
    The action shape is discrete in the range `{0, 3}`.

    - 0: Move left
    - 1: Move down
    - 2: Move right
    - 3: Move up
    - 4: Stay (do not move)

    If the agent is in a "quicksand" cell, any action will fail with 90% probability.
    The agent cannot move to "wall" cells.

    ## Observation Space
    The action shape is discrete in the range `{0, n_rows * n_cols - 1}`.
    Each integer denotes the current location of the agent. For a 3x3 grid:

     0 1 2
     3 4 5
     6 7 8

    ## Starting State
    The episode starts with the agent at the top-left cell.

    ## Transition
    By default, the transition is deterministic. It can be made stochastic by
    passing 'random_action_prob'. This is the probability that the action will
    be random. For example, if random_action_prob=0.1, there is a 10% chance
    that instead of doing LEFT / RIGHT / ... as passed in self.step(action)
    the agent will do a random action.

    ## Rewards
    - Doing STAY at the goal: +1
    - Doing STAY at a distracting goal: 0.1
    - Any action in penalty cells: -10
    - Any action in small penalty cells: -0.1
    - Otherwise: 0

    White noise can be added to all rewards by passing 'reward_noise_std'.
    White noise can be added to all NONZERO rewards by passing 'nonzero_reward_noise_std'.

    ## Episode End
    The episode ends if the following happens:

    - Termination:
        1. A positive reward is collected.

    - Truncation:
        1. The length of the episode is max_episode_steps.

    ## Rendering
    Human mode renders the environment as a grid with colored cells.

    - Black: empty cells
    - White: walls
    - Black with gray arrow: empty one-direction cell
    - Green: goal
    - Pale green: distracting goal
    - Red: penalty cells
    - Pale red: penalty cells
    - Blue: agent
    - Pale yellow: quicksand
    - Orange arrow: last agent's action (LEFT, RIGHT, UP, DOWN)
    - Orange dot: last agent's action (STAY)

    """

    metadata = {
        "render_modes": ["human", "rgb_array", "binary"],
        "render_fps": 30,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        grid: Optional[str] = "3x3_empty",
        random_action_prob: Optional[float] = 0.0,
        reward_noise_std: Optional[float] = 0.0,
        nonzero_reward_noise_std: Optional[float] = 0.0,
        **kwargs,
    ):
        self.grid_key = grid
        self.grid = np.asarray(GRIDS[self.grid_key])
        self.random_action_prob = random_action_prob
        self.reward_noise_std = reward_noise_std
        self.nonzero_reward_noise_std = nonzero_reward_noise_std

        self.n_rows, self.n_cols = self.grid.shape
        self.observation_space = gym.spaces.Discrete(self.n_cols * self.n_rows)
        self.action_space = gym.spaces.Discrete(5)
        self.agent_pos = None
        self.last_action = None

        self.render_mode = render_mode
        self.window_surface = None
        self.clock = None
        self.window_size = (
            min(64 * self.n_cols, 512),
            min(64 * self.n_rows, 512)
        )  # fmt: skip
        self.cell_size = (
            self.window_size[0] // self.n_cols,
            self.window_size[1] // self.n_rows,
        )  # fmt: skip

    def set_state(self, state):
        self.agent_pos = np.unravel_index(state, (self.n_rows, self.n_cols))

    def get_state(self):
        return np.ravel_multi_index(self.agent_pos, (self.n_rows, self.n_cols))

    def reset(self, seed: int = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self._reset(seed, **kwargs)
        if self.render_mode is not None and self.render_mode == "human":
            self.render()
        return self.get_state(), {}

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._step(action)
        if self.render_mode is not None and self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def _reset(self, seed: int = None, **kwargs):
        self.grid = np.asarray(GRIDS[self.grid_key])
        self.agent_pos = (0, 0)
        self.last_action = None
        self.last_pos = None

    def _step(self, action: int):
        terminated = False
        reward = REWARDS[self.grid[self.agent_pos]]
        if self.grid[self.agent_pos] in [GOOD, GOOD_SMALL]:
            if action == STAY:  # positive rewards are collected only with STAY
                terminated = True
            else:
                reward = 0
        if self.reward_noise_std > 0.0:
            reward += self.np_random.normal() * self.reward_noise_std
        if reward != 0.0 and self.nonzero_reward_noise_std > 0.0:
            reward += self.np_random.normal() * self.nonzero_reward_noise_std

        self.last_pos = self.agent_pos
        if self.np_random.random() < self.random_action_prob:
            action = self.action_space.sample()
        self.last_action = action
        if self.grid[self.agent_pos] == QCKSND and self.np_random.random() > 0.1:
            pass  # fail to move in quicksand
        else:
            if (
                self.grid[self.agent_pos] == LEFT and action != LEFT or
                self.grid[self.agent_pos] == RIGHT and action != RIGHT or
                self.grid[self.agent_pos] == UP and action != UP or
                self.grid[self.agent_pos] == DOWN and action != DOWN
            ):  # fmt: skip
                pass  # fail to move in one-direction cell
            else:
                self.agent_pos = _move(
                    self.agent_pos[0],
                    self.agent_pos[1],
                    action,
                    self.n_rows,
                    self.n_cols
                )  # fmt: skip

        if self.grid[self.agent_pos] == WALL:
            self.agent_pos = self.last_pos

        return self.get_state(), reward, terminated, False, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        elif self.render_mode == "binary":
            return self._render_binary()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def _render_binary(self):
        obs = np.zeros(self.grid.shape, dtype=np.uint8)
        obs[self.agent_pos] = 1
        return obs

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e

        if self.window_surface is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption(self.unwrapped.spec.id)
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."  # fmt: skip

        if self.clock is None:
            self.clock = pygame.time.Clock()

        grid = self.grid.tolist()
        assert (
            isinstance(grid, list)
        ), f"grid should be a list or an array, got {grid}"  # fmt: skip

        def arrow_head(pos, size, dir):
            if dir == LEFT:
                return (
                    (end_pos[0], end_pos[1] - size[1]),
                    (end_pos[0], end_pos[1] + size[1]),
                    (end_pos[0] - size[0], end_pos[1]),
                )
            elif dir == DOWN:
                return (
                    (end_pos[0] - size[0], end_pos[1]),
                    (end_pos[0] + size[0], end_pos[1]),
                    (end_pos[0], end_pos[1] + size[1]),
                )
            elif dir == RIGHT:
                return (
                    (end_pos[0], end_pos[1] - size[1]),
                    (end_pos[0], end_pos[1] + size[1]),
                    (end_pos[0] + size[0], end_pos[1]),
                )
            elif dir == UP:
                return (
                    (end_pos[0] - size[0], end_pos[1]),
                    (end_pos[0] + size[0], end_pos[1]),
                    (end_pos[0], end_pos[1] - size[1]),
                )

        # draw tiles
        for y in range(self.n_rows):
            for x in range(self.n_cols):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                border = pygame.Rect(pos, tuple(cs * 1.01 for cs in self.cell_size))
                rect = pygame.Rect(pos, tuple(cs * 0.99 for cs in self.cell_size))

                # draw background
                pygame.draw.rect(self.window_surface, WHITE, border)
                if grid[y][x] == GOOD:
                    pygame.draw.rect(self.window_surface, GREEN, rect)
                elif grid[y][x] == GOOD_SMALL:
                    pygame.draw.rect(self.window_surface, PALE_GREEN, rect)
                elif grid[y][x] == BAD:
                    pygame.draw.rect(self.window_surface, RED, rect)
                elif grid[y][x] == BAD_SMALL:
                    pygame.draw.rect(self.window_surface, PALE_RED, rect)
                elif grid[y][x] == QCKSND:
                    pygame.draw.rect(self.window_surface, PALE_YELLOW, rect)
                elif grid[y][x] == WALL:
                    pygame.draw.rect(self.window_surface, WHITE, rect)
                elif grid[y][x] in [EMPTY, LEFT, RIGHT, UP, DOWN]:
                    pygame.draw.rect(self.window_surface, BLACK, rect)

                # draw agent
                if (y, x) == self.agent_pos:
                    pygame.draw.ellipse(self.window_surface, BLUE, rect)

                # draw arrow for one-direction cell
                if grid[y][x] in [LEFT, RIGHT, UP, DOWN]:
                    if grid[y][x] == LEFT:
                        start_pos = (pos[0] + self.cell_size[0], pos[1] + self.cell_size[1] / 2)
                        end_pos = (pos[0] + self.cell_size[0] / 2, pos[1] + self.cell_size[1] / 2)
                        arr_width = -(-self.cell_size[1] // 3)
                    elif grid[y][x] == DOWN:
                        start_pos = (pos[0] + self.cell_size[0] / 2, pos[1])
                        end_pos = (pos[0] + self.cell_size[0] / 2, pos[1] + self.cell_size[1] / 2)
                        arr_width = -(-self.cell_size[0] // 3)
                    elif grid[y][x] == RIGHT:
                        start_pos = (pos[0], pos[1] + self.cell_size[1] / 2)
                        end_pos = (pos[0] + self.cell_size[0] / 2, pos[1] + self.cell_size[1] / 2)
                        arr_width = -(-self.cell_size[1] // 3)
                    elif grid[y][x] == UP:
                        start_pos = (pos[0] + self.cell_size[0] / 2, pos[1] + self.cell_size[1])
                        end_pos = (pos[0] + self.cell_size[0] / 2, pos[1] + self.cell_size[1] / 2)
                        arr_width = -(-self.cell_size[0] // 3)
                    else:
                        pass
                    pygame.draw.polygon(self.window_surface, GRAY, (start_pos, end_pos), arr_width)
                    arr_pos = arrow_head(end_pos, [cs / 2 for cs in self.cell_size], grid[y][x])
                    pygame.draw.polygon(self.window_surface, GRAY, arr_pos, 0)

            # draw action arrow
            if self.last_pos is not None:
                x = self.last_pos[1]
                y = self.last_pos[0]

                if self.last_action == STAY:
                    pos = (
                        x * self.cell_size[0] + self.cell_size[0] / 4,
                        y * self.cell_size[1] + self.cell_size[1] / 4,
                    )
                    rect = pygame.Rect(pos, tuple(cs * 0.5 for cs in self.cell_size))
                    pygame.draw.ellipse(self.window_surface, ORANGE, rect)
                else:
                    pos = (
                        x * self.cell_size[0] + self.cell_size[0] / 2,
                        y * self.cell_size[1] + self.cell_size[1] / 2,
                    )
                    if self.last_action == LEFT:
                        end_pos = (pos[0] - self.cell_size[0] / 4, pos[1])
                        arr_width = -(-self.cell_size[1] // 6)
                    elif self.last_action == DOWN:
                        end_pos = (pos[0], pos[1] + self.cell_size[1] / 4)
                        arr_width = -(-self.cell_size[0] // 6)
                    elif self.last_action == RIGHT:
                        end_pos = (pos[0] + self.cell_size[0] / 4, pos[1])
                        arr_width = -(-self.cell_size[1] // 6)
                    elif self.last_action == UP:
                        end_pos = (pos[0], pos[1] - self.cell_size[1] / 4)
                        arr_width = -(-self.cell_size[0] // 6)
                    else:
                        raise ValueError("illegal action")

                    pygame.draw.polygon(self.window_surface, ORANGE, (pos, end_pos), arr_width)
                    arr_pos = arrow_head(
                        end_pos, [cs / 5 for cs in self.cell_size], self.last_action
                    )
                    pygame.draw.polygon(self.window_surface, ORANGE, arr_pos, 0)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

        else:
            raise NotImplementedError

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


class GridworldMiddleStart(Gridworld):
    """
    Like Gridworld, but the agent starts at the center of the grid.
    """

    def _reset(self, seed: int = None, **kwargs):
        Gridworld._reset(self, seed=seed, **kwargs)
        self.agent_pos = (self.n_rows // 2, self.n_cols // 2)
        return self.get_state(), {}


class GridworldRandomStart(Gridworld):
    """
    Like Gridworld, but the agent can start anywhere.
    """

    def _reset(self, seed: int = None, **kwargs):
        Gridworld._reset(self, seed=seed, **kwargs)
        self.agent_pos = (
            self.np_random.integers(0, self.n_rows),
            self.np_random.integers(0, self.n_cols),
        )
        return self.get_state(), {}


class RiverSwim(Gridworld):
    """
    First presented in "An analysis of model-based Interval Estimation for Markov Decision Processes".
    Implementation according to https://rlgammazero.github.io/docs/2020_AAAI_tut_part1.pdf

    One-dimensional grid with positive rewards at its ends, 0.01 in the leftmost
    cell and 1 in the rightmost.
    The agent starts either in the 2nd or 3rd leftmost cell, the only actions are
    LEFT and RIGHT, and the transition is stochastic.

    This is an infinite horizon MDP, there is no terminal state.
    Episodes are truncated after max_episode_steps (default 200).
    """

    def __init__(self, **kwargs):
        Gridworld.__init__(self, **kwargs)
        self.grid[0] = 0.01  # we use self.grid for rendering
        self.grid[-1] = 1.0
        self.action_space = gym.spaces.Discrete(2)  # only LEFT and RIGHT

    def _reset(self, seed: int = None, **kwargs):
        Gridworld._reset(self, seed=seed, **kwargs)
        self.agent_pos = (0, self.np_random.integers(1, 3))  # init either in 2nd or 3rd cell
        return self.get_state(), {}

    def _step(self, action: int):
        state = self.get_state()
        first = 0
        last = self.observation_space.n - 1

        # Map action to match Gridworld notation
        if action == 0:
            action = LEFT
        elif action == 1:
            action = RIGHT
        else:
            raise NotImplementedError("illegal action")

        # Stochastic transition
        original_action = action
        r = self.np_random.random()
        if action == RIGHT:
            if state == first:
                if r < 0.4:
                    action = LEFT  # or stay, is equivalent
            elif state == last:
                if r < 0.4:
                    action = LEFT
            else:
                if r < 0.05:
                    action = LEFT
                elif r < 0.65:
                    action = STAY

        obs, _, _, truncated, info = Gridworld._step(self, action)
        terminated = False  # infinite horizon
        self.last_action = original_action

        reward = 0.0
        if state == last and action == RIGHT and original_action == RIGHT:
            reward = 1.0
        elif state == first and action == LEFT and original_action == LEFT:
            reward = 0.01

        return obs, reward, terminated, truncated, info
