import gym
from gym import spaces
import pygame
import numpy as np


class CovidEnv(gym.Env):

    def __init__(self, size):
        self.size = int(size)  # The size of the second generation
        self.days = 30  # Assume we observe the 2nd generation for 30 days
        self.p_symptomatic = 0.8  # probability that an infected person is eventually symptomatic

        """
        Use a dictionary to represent the observation. Each category is represented by a matrix.
        The row of each matrix represents a 2nd generation case. The column of each matrix represents one day.
        """
        self.observation_space = spaces.Dict({
            "Day of exposure": spaces.Box(low=0, high=1, shape=(self.size, self.days), dtype=np.int16),
            "Any day before exposure": spaces.Box(low=0, high=1, shape=(self.size, self.days), dtype=np.int16),
            "Showing symptoms": spaces.Box(low=0, high=1, shape=(self.size, self.days), dtype=np.int16),
            "Not showing symptoms": spaces.Box(low=0, high=1, shape=(self.size, self.days), dtype=np.int16),
            "Unobserved future": spaces.Box(low=0, high=1, shape=(self.size, self.days), dtype=np.int16)})

        """
        We choose to use a discrete number to represent an action space. There should be 2^size scenarios.
        For example, let assume size = 3,  0 means we don't quarantine any people (think 
        it as a binary representation 000). 1 means we only quarantine the first person (001).
        We can infer from this 'rule' that the maximum number of our action 7 (7 = 111) means we quarantine 
        all three people.
        """
        self.action_space = spaces.Discrete(2 ** self.size)

    # We start to trace when the 1st generation case is tested positive.
    # We assume the 1st generation case is tested positive at day 4.
    def reset(self, seed=None, return_info=False, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        initial_state = {
            "Day of exposure": np.zeros((self.size, self.days)),
            "Any day before exposure": np.zeros((self.size, self.days)),
            "Showing symptoms": np.zeros((self.size, self.days)),
            "Not showing symptoms": np.zeros((self.size, self.days)),
            "Unobserved future": np.zeros((self.size, self.days))}

        # Assume every 2nd generation cases are exposed on day 3
        for i in self.size:
            initial_state["Day of exposure"][i][2] = 1

        for i in self.size:
            for j in range(2):
                initial_state["Any day before exposure"][i][j] = 1

        # Assume we haven't found any 2nd generation case have symptoms
        for i in self.size:
            for j in range(2):
                initial_state["Any day before exposure"][i][j] = 1

        # Assume start observing at day 4
        for i in self.size:
            for j in range(3, self.days):
                initial_state["Any day before exposure"][i][j] = 1

        if not return_info:
            return initial_state
        else:
            return initial_state, {}

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        done = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if done else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info

    def render(self):
        pass

    def close(self):
        pass
