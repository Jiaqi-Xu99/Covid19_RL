import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy.stats import bernoulli


class CovidEnv(gym.Env):

    def __init__(self):
        self.size = 4  # The size of the second generation
        self.days = 30  # Assume we observe the 2nd generation for 30 days
        # We assume weight_infect_no_quarantine = -1, and calculate weight_no_infect_quarantine from the ratio
        self.ratio = 0.01
        self.p_high_transmissive = 0.109  # Probability that the index case is highly transmissive
        self.p_infected = 0.0116  # Probability that a person get infected (given index case is not highly transmissive)
        self.p_symptomatic = 0.8  # Probability that a person is infected and showing symptom

        # We run the simulation from day 1 to self.days to get the whole state of our environment
        self.simulated_state = self._simulation()

        """
        Use a long vector to represent the observation. Only focus on one 2nd generation.
        1 means showing symptoms. 0 means not showing symptoms.
        We assume every cases get exposed at day 0.
        Index 0 represents whether have showed symptom. Index 1 represents number of others' symptoms.
        Index 2 represents number of other 2nd generation. Index 3 represents when show symptom.
        """
        self.observation_space = spaces.Box(low=0, high=self.days, shape=(4,), dtype=int)
        self.current_state = np.zeros((4,), dtype=int)
        self.observed_day = 0
        self.current_state[0] = self.simulated_state["Showing symptoms"][0][self.observed_day]
        self.current_state[1] = self.size - 1
        symptom_num = 0
        for i in range(1, self.size):
            if self.simulated_state["Showing symptoms"][i][self.observed_day] == 1:
                symptom_num = symptom_num + 1
        self.current_state[2] = symptom_num
        if self.simulated_state["Showing symptoms"][0][self.observed_day] == 1:
            self.current_state[3] = self.observed_day + 1
        self.observed_day = self.observed_day + 1

        self.seed()

        """
        0 for not quarantine, 1 for quarantine
        """
        self.action_space = spaces.Discrete(2)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # We start to trace when the 1st generation case is tested positive.
    def reset(self, return_info=False):
        # We run the simulation from day 1 to self.days to get the whole state of our environment
        self.simulated_state = self._simulation()
        # Initialize the current state
        self.current_state = np.zeros((4,), dtype=int)
        self.observed_day = 0
        self.current_state[0] = self.simulated_state["Showing symptoms"][0][self.observed_day]
        self.current_state[1] = self.size - 1
        symptom_num = 0
        for i in range(1, self.size):
            if self.simulated_state["Showing symptoms"][i][self.observed_day] == 1:
                symptom_num = symptom_num + 1
        self.current_state[2] = symptom_num
        if self.simulated_state["Showing symptoms"][0][self.observed_day] == 1:
            self.current_state[3] = self.observed_day + 1
        self.observed_day = self.observed_day + 1

        # Assume we haven't found any 2nd generation case have symptoms at first
        if not return_info:
            return self.current_state
        else:
            return self.current_state, {}

    def step(self, action):
        # Update the state from the result of simulation
        if self.current_state[3] == 0 and self.simulated_state["Showing symptoms"][0][self.observed_day] == 1:
            self.current_state[3] = self.observed_day + 1
        self.current_state[0] = self.simulated_state["Showing symptoms"][0][self.observed_day]
        symptom_num = 0
        for i in range(1, self.size):
            if self.simulated_state["Showing symptoms"][i][self.observed_day] == 1:
                symptom_num = symptom_num + 1
        self.current_state[2] = symptom_num

        sum1 = 0
        sum2 = 0

        """
        # 14 days quarantine
        if self.simulated_state["Whether infected"][0][self.observed_day] == 1 and 0 <= self.observed_day <= 2 and 17 <= self.observed_day < self.days:
            sum1 = sum1 + 1
        if self.simulated_state["Whether infected"][0][self.observed_day] == 0 and 3 <= self.observed_day <= 16:
            sum2 = sum2 + 1
        #"""

        """
        # No quarantine
        if self.simulated_state["Whether infected"][0][self.observed_day] == 1:
            sum1 = sum1 + 1
        #"""

        #"""
        # RL
        for i in range(self.size):
            if self.simulated_state["Whether infected"][i][self.observed_day] == 1 and action == 0:
                sum1 = sum1 + 1
            if self.simulated_state["Whether infected"][i][self.observed_day] == 0 and action == 1:
                sum2 = sum2 + 1
        # """

        # Calculate the reward, reward = -1 * (infectious & not quarantine) - ratio * (not infectious & quarantine)
        reward = (-1 * sum1 - self.ratio * sum2) * 10
        self.observed_day = self.observed_day + 1
        done = bool(self.observed_day == self.days)
        return self.current_state, reward, done, {}

    def _simulation(self):
        # Use an array that represents which people get infected. 1 represents get infected.
        self.simulated_state = {
            "Showing symptoms": np.zeros((self.size, self.days)),
            "Whether infected": np.zeros((self.size, self.days))}

        # We assume that the index case has 0.109 probability to be highly transmissive.
        # Under that circumstance, the infectiousness rate becomes 24.4 times bigger.
        flag = bernoulli.rvs(self.p_high_transmissive, size=1)
        if flag == 1:
            self.p_infected = self.p_infected * 24.4
        infected_case = np.array(bernoulli.rvs(self.p_infected, size=self.size))
        for i in range(self.size):
            #  Whether infected
            if infected_case[i] == 1:
                # infected and show symptoms
                if bernoulli.rvs(self.p_symptomatic, size=1) == 1:
                    # Use log normal distribution, mean = 1.57, std = 0.65
                    symptom_day = int(np.random.lognormal(1.57, 0.65, 1))  # day starts to show symptom
                    duration = int(np.random.lognormal(2.70, 0.15, 1))  # duration of showing symptom
                    for j in range(symptom_day, symptom_day + duration):
                        if 0 <= j < self.days:
                            self.simulated_state["Showing symptoms"][i][j] = 1
                    #  Whether infected
                    for j in range(symptom_day - 2, symptom_day + 6):
                        if 0 <= j < self.days:
                            self.simulated_state["Whether infected"][i][j] = 1
                # infected but not showing symptoms
                else:
                    symptom_day = int(np.random.lognormal(1.57, 0.65, 1))
                    for j in range(symptom_day - 2, symptom_day + 6):
                        if 0 <= j < self.days:
                            self.simulated_state["Whether infected"][i][j] = 1
            # not infected but show some symptoms
        if flag == 1:
            self.p_infected = self.p_infected / 24.4

        return self.simulated_state

    def render(self, mode='None'):
        pass

    def close(self):
        pass
