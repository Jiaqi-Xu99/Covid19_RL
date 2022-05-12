import gym
from gym import spaces
import numpy as np


class CovidEnv(gym.Env):

    def __init__(self, size):
        self.size = int(size)  # The size of the second generation
        self.days = 30  # Assume we observe the 2nd generation for 30 days
        # Incubation time (Log-normal: Log mean 1.57 days and log std 0.65 days)
        self.incubation_time = 0
        self.p_symptomatic = 0.8  # probability that an infected person is eventually symptomatic
        self.quarantine_days = np.array(self.size)  # how many days each 2nd generation cases have been quarantined

        """
        Use a dictionary to represent the observation. Each category is represented by a matrix/vector/discrete number.
        The row represents a 2nd generation case. The column of each matrix represents one day.
        """
        self.observation_space = spaces.Dict({
            "Day of exposure": spaces.Box(low=0, high=1, shape=(self.size, self.days), dtype=np.int16),
            "Showing symptoms": spaces.Box(low=0, high=1, shape=(self.size, self.days), dtype=np.int16),
            "Quarantine days": spaces.Box(low=0, high=self.days, shape=(self.size,), dtype=np.int16),
            "Infectious rate": spaces.Box(low=0, high=self.days, shape=(self.size,), dtype=np.float32),
            "Unobserved future": spaces.Discrete(self.days)})

        self.state = {
            "Day of exposure": np.zeros((self.size, self.days)),
            "Showing symptoms": np.zeros((self.size, self.days)),
            "Quarantine days": np.zeros((self.size,)),
            "Infectious rate": np.zeros((self.size,)),
            "Unobserved future": 5}

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

        self.state = {
            "Day of exposure": np.zeros((self.size, self.days)),
            "Showing symptoms": np.zeros((self.size, self.days)),
            "Quarantine days": np.zeros((self.size,)),
            "Infectious rate": np.zeros((self.size,)),
            "Unobserved future": 5}

        # Assume every 2nd generation cases are exposed on day 3
        for i in range(self.size):
            self.state["Day of exposure"][i][2] = 1

        # Assume we haven't found any 2nd generation case have symptoms

        if not return_info:
            return self.state
        else:
            return self.state, {}

    """
    Two things will change the state of our environment. One is the action. The other is the environment itself.
    First, I change the state of the environment by action. Then, the state will change based on the parameters.
    """
    def step(self, action):
        quarantine_days = self._dec_to_binary(action)
        self.state["Quarantine days"] = np.add(self.state["Quarantine days"], quarantine_days)
        observed_day = self.state["Unobserved future"]

        done = bool(observed_day == self.days)
        reward = -1.0
        return self.state, reward, done, {}

    def _dec_to_binary(self, n):
        # array to store binary number
        binary_num = np.zeros((self.size,))

        # counter for binary array
        i = 0
        while n > 0:
            # storing remainder in binary array
            binary_num[i] = n % 2
            n = int(n / 2)
            i += 1

        return binary_num

    def render(self):
        pass

    def close(self):
        pass
