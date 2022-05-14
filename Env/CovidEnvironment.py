import random

import gym
from gym import spaces
import numpy as np


def incubation_time(time):
    probability = (np.exp(-(np.log(time) - 1.57)**2 / (2 * 0.65**2)) / (time * 0.65 * np.sqrt(2 * np.pi)))
    return probability


class CovidEnv(gym.Env):

    def __init__(self, size):
        self.size = int(size)  # The size of the second generation
        self.days = 30  # Assume we observe the 2nd generation for 30 days
        # Incubation time (Log-normal: Log mean 1.57 days and log std 0.65 days)
        self.p_symptomatic = 0.8  # probability that an infected person is eventually symptomatic

        # We run the simulation from day 1 to self.days to get the whole state of our environment
        self.simulated_state = self._simulation()

        """
        Use a dictionary to represent the observation. Each category is represented by a matrix/discrete number.
        The row of the matrix represents a 2nd generation case. The column of the matrix represents one day.
        The value of unobserved future n means we haven't observe from day n+1
        """
        self.observation_space = spaces.Dict({
            "Day of exposure": spaces.Box(low=0, high=1, shape=(self.size, self.days), dtype=np.int16),
            "Showing symptoms": spaces.Box(low=0, high=1, shape=(self.size, self.days), dtype=np.int16),
            "Unobserved future": spaces.Discrete(self.days)})

        self.current_state = {
            "Day of exposure": np.zeros((self.size, self.days)),
            "Showing symptoms": np.zeros((self.size, self.days)),
            "Unobserved future": 4}

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

        # We run the simulation from day 1 to self.days to get the whole state of our environment
        self.simulated_state = self._simulation()

        self.current_state = {
            "Day of exposure": np.zeros((self.size, self.days)),
            "Showing symptoms": np.zeros((self.size, self.days)),
            "Unobserved future": 4}

        # Assume every 2nd generation cases are exposed on day 3
        for i in range(self.size):
            self.current_state["Day of exposure"][i][2] = 1

        # Assume we haven't found any 2nd generation case have symptoms at first
        if not return_info:
            return self.current_state
        else:
            return self.current_state, {}

    def step(self, action):
        # Get which day are we going to observe
        observing_day = self.current_state["Unobserved future"]
        # Update the state from the result of simulation
        for i in range(self.size):
            self.current_state["Day of exposure"][i][observing_day] = self.simulated_state["Day of exposure"][i][observing_day]
            self.current_state["Showing symptoms"][i][observing_day] = self.simulated_state["Showing symptoms"][i][observing_day]
        # calculate the reward, reward = - a * (infectious & not quarantine) - b * (not infectious & quarantine)
        quarantine = self._dec_to_binary(action)
        sum1 = 0
        sum2 = 0
        for i in range(self.size):
            if self.simulated_state["Whether infected"][i][observing_day] == 1 and quarantine[i] == 0:
                sum1 = sum1 + 1
            if self.simulated_state["Whether infected"][i][observing_day] == 0 and quarantine[i] == 1:
                sum2 = sum2 + 1
        reward = -1 * sum1 - 0.5 * sum2
        self.current_state["Unobserved future"] = self.current_state["Unobserved future"] + 1
        done = bool(self.current_state["Unobserved future"] == self.days)
        return self.current_state, reward, done, {}

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

    def _simulation(self):
        self.simulated_state = {
            "Day of exposure": np.zeros((self.size, self.days)),
            "Showing symptoms": np.zeros((self.size, self.days)),
            "Whether infected": np.zeros((self.size, self.days)),
            "Unobserved future": 4}
        # Assume every 2nd generation cases are exposed on day 3
        for i in range(self.size):
            self.current_state["Day of exposure"][i][2] = 1

        for i in range(self.simulated_state["Unobserved future"], self.days):
            for j in range(self.size):
                #  Whether showing symptom
                flag = 0
                p_exposure_to_symptom = incubation_time(i-2)
                if flag == 0 and random.randint(1, 1000) <= 1000*p_exposure_to_symptom:
                    #  Assume the symptom will last six days when it starts
                    for k in range(6):
                        self.simulated_state["Showing symptoms"][j][i+k] = 1
                    #  Whether infected
                    for k in range(-2, 6):
                        self.simulated_state["Whether infected"][j][i + k] = 1
                    flag = 1

        return self.simulated_state

    def render(self):
        pass

    def close(self):
        pass
