import copy
from typing import Optional, Tuple

import gym
import numpy as np
from gym.envs.classic_control import rendering

EMPTY = BLACK = 0
WALL = GRAY = 1
AGENT = BLUE = 2
BOMB = RED = 3
GOAL = GREEN = 4
SUCCESS = PINK = 5

COLOR_MAP = {
    BLACK: [0.0, 0.0, 0.0],
    GRAY: [0.5, 0.5, 0.5],
    RED: [1.0, 0.0, 0.0],
    GREEN: [0.0, 1.0, 0.0],
    BLUE: [0.0, 0.0, 1.0],
    PINK: [1.0, 0.0, 1.0],
}

NOOP = 0
DOWN = 1
UP = 2
LEFT = 3
RIGHT = 4


class GridworldEnv:
    def __init__(self, max_steps: int = 100) -> None:
        self.initial_grid_state = np.array(
            [
                [1] * 8,
                [1, 2, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 0, 0, 1],
                [1, 0, 1, 0, 1, 0, 0, 1],
                [1, 0, 1, 4, 1, 0, 0, 1],
                [1, 0, 3, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1] * 8,
            ]
        )
        self.grid_state = copy.deepcopy(self.initial_grid_state)
        self.obs_space = gym.spaces.Box(low=0, high=6, shape=self.grid_state.shape)
        self.img_shape = (256, 256, 3)

        self.action_space = gym.spaces.Discrete(5)
        self.actions = [NOOP, UP, DOWN, LEFT, RIGHT]
        self.action_pos_dict = {
            NOOP: [0, 0],
            UP: [-1, 0],
            DOWN: [1, 0],
            LEFT: [0, -1],
            RIGHT: [0, 1],
        }

        self.agent_state, self.agent_goal_state = self.get_state()
        self.step_num = 0
        self.max_steps = max_steps
        self.done = False
        self.viewer = None

    def get_state(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        start_state = np.where(self.grid_state == AGENT)
        goal_state = np.where(self.grid_state == GOAL)
        if not (start_state[0] and goal_state[0]):
            raise ValueError("Invalid grid state")

        return (
            (start_state[0][0], start_state[1][0]),
            (goal_state[0][0], goal_state[1][0]),
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        reward = 0.0
        next_state = (
            self.agent_state[0] + self.action_pos_dict[action][0],
            self.agent_state[1] + self.action_pos_dict[action][1],
        )

        is_invalid_state = (
            next_state[0] < 0
            or next_state[0] >= self.grid_state.shape[0]
            or next_state[1] < 0
            or next_state[1] >= self.grid_state.shape[1]
        )
        if is_invalid_state:
            next_state = self.agent_state

        next_agent_state = self.grid_state[next_state[0], next_state[1]]

        if next_agent_state == EMPTY:
            self.grid_state[next_state[0], next_state[1]] = AGENT
            self.grid_state[self.agent_state[0], self.agent_state[1]] = EMPTY
            self.agent_state = next_state

        elif next_agent_state == WALL:
            reward = -0.1
        elif next_agent_state == GOAL:
            reward = 1
            self.done = True
        elif next_agent_state == BOMB:
            reward = -1
            self.done = True

        self.step_num += 1

        if self.step_num >= self.max_steps:
            self.done = True

        if self.done:
            end_state = copy.deepcopy(self.grid_state)
            self.reset()
            return end_state, reward, True, {}

        return self.grid_state, reward, False, {}

    def reset(self) -> np.ndarray:
        self.grid_state = copy.deepcopy(self.initial_grid_state)
        self.agent_state, self.agent_goal_state = self.get_state()
        self.step_num = 0
        self.done = False
        return self.grid_state

    def grid_to_img(self, img_shape: Optional[Tuple[int, int, int]] = None):
        if img_shape is None:
            img_shape = self.img_shape
        obs = np.zeros(img_shape)
        scale_x = img_shape[0] // self.grid_state.shape[0]
        scale_y = img_shape[1] // self.grid_state.shape[1]

        for i in range(self.grid_state.shape[0]):
            for j in range(self.grid_state.shape[1]):
                for k in range(3):
                    obs[
                        i * scale_x : (i + 1) * scale_x,
                        j * scale_y : (j + 1) * scale_y,
                        k,
                    ] = COLOR_MAP[self.grid_state[i, j]][k]
        return (255 * obs).astype(np.uint8)

    def render(self, human: bool = True) -> None:
        img = self.grid_to_img()
        if not human:
            return img
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        if self.viewer:
            self.viewer.imshow(img)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    env = GridworldEnv()
    obs = env.reset()
    done = False
    step = 1
    while not done:
        act = int(env.action_space.sample())
        obs, _, done, info = env.step(act)
        step += 1
        env.render()
    print(step)
    env.close()
