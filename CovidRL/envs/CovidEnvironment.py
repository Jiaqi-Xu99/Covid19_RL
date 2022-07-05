import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy.stats import bernoulli


class CovidEnv(gym.Env):

    def __init__(self):
        self.size = 4  # The size of the second generation
        self.days = 30  # Assume we observe the 2nd generation for 40 days
        self.weight_infect_no_quarantine = -10
        self.weight_no_infect_quarantine = -5
        self.p_infected = 0.8  # probability that a person get infected

        # We run the simulation from day 1 to self.days to get the whole state of our environment
        self.simulated_state = self._simulation()

        """
        Use a long vector to represent the observation. Every self.days elements represent an index case.
        1 means showing symptoms. 0 means not showing symptoms. -1 means unobserved future.
        We assume every cases get exposed at day 0.
        """
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.size * self.days,), dtype=np.int32)
        self.current_state = np.full((self.size * self.days,), -1, dtype=np.int32)
        self.observed_day = 0

        self.seed()

        """
        We choose to use a discrete number to represent an action space. There should be 2^size scenarios.
        For example, let assume size = 3,  0 means we don't quarantine any people (think 
        it as a binary representation 000). 1 means we only quarantine the first person (001).
        We can infer from this 'rule' that the maximum number of our action 7 (7 = 111) means we quarantine 
        all three people.
        """
        self.action_space = spaces.Discrete(2 ** self.size)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # We start to trace when the 1st generation case is tested positive.
    def reset(self, seed=None, return_info=False, options=None):
        # We run the simulation from day 1 to self.days to get the whole state of our environment
        self.simulated_state = self._simulation()
        # Initialize the current state
        self.current_state = np.full((self.size * self.days,), -1, dtype=np.int32)
        self.observed_day = 0

        # Assume we haven't found any 2nd generation case have symptoms at first
        if not return_info:
            return self.current_state
        else:
            return self.current_state, {}

    def step(self, action):
        # Update the state from the result of simulation
        for i in range(self.size):
            self.current_state[i * self.days + self.observed_day] = self.simulated_state["Showing symptoms"][i][self.observed_day]
        # calculate the reward, reward = - a * (infectious & not quarantine) - b * (not infectious & quarantine)
        # quarantine = self._dec_to_binary(action)
        sum1 = 0
        sum2 = 0

        for i in range(self.size):
            if self.simulated_state["Whether infected"][i][self.observed_day] == 1 and 0 <= self.observed_day <= 2 and \
             17 <= self.observed_day <= self.days:
                sum1 = sum1 + 1
            if self.simulated_state["Whether infected"][i][self.observed_day] == 0 and 3 <= self.observed_day <= 16:
                sum2 = sum2 + 1

        """
        for i in range(self.size):
            if self.simulated_state["Whether infected"][i][self.observed_day] == 1 and quarantine[i] == 0:
                sum1 = sum1 + 1
            if self.simulated_state["Whether infected"][i][self.observed_day] == 0 and quarantine[i] == 1:
               sum2 = sum2 + 1
        """

        reward = self.weight_infect_no_quarantine * sum1 + self.weight_no_infect_quarantine * sum2
        self.observed_day = self.observed_day + 1
        done = bool(self.observed_day == self.days)
        # if self.observed_day == self.days:
        #    print("Status:" + str(self.current_state))
        #    print("Error1:" + str(self.sum1))
        # print("Action:" + str(quarantine))
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
            "Showing symptoms": np.zeros((self.size, self.days)),
            "Whether infected": np.zeros((self.size, self.days))}

        """
        Use an array that represents which people get infected. 1 represents get infected.
        Use Bernoulli distribution, p = 0.8
        """
        infected_case = np.array(bernoulli.rvs(0.8, size=self.size))
        for i in range(self.size):
            #  Whether infected
            if infected_case[i] == 1:
                # Use log normal distribution, mean = 1.57, std = 0.65
                symptom_day = int(np.random.lognormal(1.57, 0.65, 1))  # day starts to show symptom
                #  Assume the symptom will last six days when it starts
                for j in range(symptom_day, symptom_day+6):
                    if 0 <= j < self.days:
                        self.simulated_state["Showing symptoms"][i][j] = 1
                #  Whether infected
                for j in range(symptom_day-2, symptom_day+6):
                    if 0 <= j < self.days:
                        self.simulated_state["Whether infected"][i][j] = 1

        return self.simulated_state

    def render(self, mode='None'):
        pass

    def close(self):
        pass
