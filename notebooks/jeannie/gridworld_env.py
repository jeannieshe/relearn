import gymnasium as gym
from typing import Dict, Optional
import numpy as np

class GridWorldEnv(gym.Env):
    def __init__(self, size: int = 5):
        # the size of the world (5x5 by default)
        self.size = size

        # initialize positions - will be set randomly in reset()
        # use -1, -1 as the "uninitialized" state
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # define what the agent can observe
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size-1, shape=(2,), dtype=int), # [x, y] coordinates
                "target": gym.spaces.Box(0, size-1, shape=(2,), dtype=int), # [x, y] coordinates
            }
        )

        # define what actions are possible (all 4 directions)
        self.action_space = gym.spaces.Discrete(4)

        # map action numbers to actual movements on the grid
        # this makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([0, 1]), # right
            1: np.array([-1, 0]), # up
            2: np.array([0, -1]), # left
            3: np.array([1, 0]), # down
        }

    def _get_obs(self):
        """
        Converts internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {
            "agent": self._agent_location,
            "target": self._target_location
        }

    def _get_info(self):
        """
        Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target. Not meant to be used for training.
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1 # first order matrix norm
            )
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration

        Returns:
            tuple: (observation, info) for the initial state
        """
        # first seed the rng
        super().reset(seed=seed)

        # randomly place agent anywhere on the grid.
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # randomly place target, ensuring its different from agent position
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Takes an action, updates the environment state, and returns the results.
        """
        # translate the action into a direction that makes sense to the environment
        direction = self._action_to_direction[action]

        # apply the action. np.clip prevents the agent from walking off the edge
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size-1
        )

        # calculate the reward if the target was achieved
        terminated = np.array_equal(self._agent_location, self._target_location)

        # no truncation in this simple env, but can add a step limit if you want
        truncated = False

        # simple reward structure: +1 for reaching target, 0 otherwise
        reward = 1 if terminated else -0.01

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    