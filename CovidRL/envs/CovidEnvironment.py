import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy.stats import bernoulli


class CovidEnv(gym.Env):

    def __init__(self):
        self.size = 4  # The size of the second generation
        self.days = 30  # Assume we observe the 2nd generation for 40 days
        # We assume weight_infect_no_quarantine = -1, and calculate weight_no_infect_quarantine from the ratio
        self.ratio_of_weights = 0.5
        self.p_high_transmissive = 0.2 # Probability that the index case is highly transmissive
        self.p_infected = 0.4  # Probability that a person get infected
        self.p_symptomatic = 0.8 # Probability that a person is infected and showing symptom

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
    def reset(self, return_info=False):
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
        quarantine = self._dec_to_binary(action)
        sum1 = 0
        sum2 = 0

        """
        # Baseline
        for i in range(self.size):
            if self.simulated_state["Whether infected"][i][self.observed_day] == 1 and 0 <= self.observed_day <= 2 and \
             17 <= self.observed_day < self.days:
                sum1 = sum1 + 1
            if self.simulated_state["Whether infected"][i][self.observed_day] == 0 and 3 <= self.observed_day <= 16:
                sum2 = sum2 + 1
       """

        # """
        for i in range(self.size):
            if self.simulated_state["Whether infected"][i][self.observed_day] == 1 and quarantine[i] == 0:
                sum1 = sum1 + 1
            if self.simulated_state["Whether infected"][i][self.observed_day] == 0 and quarantine[i] == 1:
               sum2 = sum2 + 1
        # """
        # Calculate the reward, reward = -1 * (infectious & not quarantine) - ratio * (not infectious & quarantine)
        reward = -1 * sum1 - self.ratio_of_weights * sum2
        self.observed_day = self.observed_day + 1
        done = bool(self.observed_day == self.days)

        return self.current_state, reward, done, {}

    def _dec_to_binary(self, n):
        # Array to store binary number
        binary_num = np.zeros((self.size,))
        # Counter for binary array
        i = 0
        while n > 0:
            # Storing remainder in binary array
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

        # We assume that the index case has 0.2 probability to be highly transmissive.
        # Under that circumstance, the infectiousness rate becomes 2.2 times bigger.
        if bernoulli.rvs(self.p_high_transmissive, size=1) == 1:
            self.p_infected = 0.88
        infected_case = np.array(bernoulli.rvs(self.p_infected, size=self.size))
        for i in range(self.size):
            #  Whether infected
            if infected_case[i] == 1:
                # infected and show symptoms
                if bernoulli.rvs(self.p_symptomatic, size=1) == 1:
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
                # infected but not showing symptoms
                else:
                    symptom_day = int(np.random.lognormal(1.57, 0.65, 1))
                    for j in range(symptom_day-2, symptom_day+6):
                        if 0 <= j < self.days:
                            self.simulated_state["Whether infected"][i][j] = 1
            # not infected but show some symptoms
            else:
                symptom_day = int(np.random.lognormal(1.57, 0.65, 1))
                for j in range(symptom_day, symptom_day + 2): # We assume this kind of symptom will last shorter
                    if 0 <= j < self.days:
                        self.simulated_state["Showing symptoms"][i][j] = 1

        return self.simulated_state

    def render(self, mode='None'):
        pass

    def close(self):
        pass
