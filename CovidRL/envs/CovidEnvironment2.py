import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy.stats import bernoulli
from .Supervised_Learning  import NeuralNetwork
import torch


class CovidEnv2(gym.Env):

    def __init__(self):
        self.size = 4  # The size of the second generation
        self.days = 30  # Assume we observe the 2nd generation for 40 days
        # We assume weight_infect_no_quarantine = -1, and calculate weight_no_infect_quarantine from the ratio
        self.weights = 0.8
        self.ratio = (1 - self.weights) / self.weights
        self.p_high_transmissive = 0.2  # Probability that the index case is highly transmissive
        self.p_infected = 0.08  # Probability that a person get infected
        self.p_symptomatic = 0.8  # Probability that a person is infected and showing symptom

        # We run the simulation from day 1 to self.days to get the whole state of our environment
        self.simulated_state = self._simulation()

        """
        Use a long vector to represent the observation. Every self.days elements represent an index case.
        1 means showing symptoms. 0 means not showing symptoms. -1 means unobserved future.
        We assume every cases get exposed at day 0.
        """
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1 * 1,), dtype=np.float32)
        self.current_state = np.zeros((1,1))
        self.observed_day = 0
        self.current_state[0][0] = self.simulated_state["Prediction"][0][self.observed_day]
        self.observed_day = self.observed_day + 1

        self.seed()

        """
        We choose to use a discrete number to represent an action space. There should be 2^size scenarios.
        For example, let assume size = 3,  0 means we don't quarantine any people (think 
        it as a binary representation 000). 1 means we only quarantine the first person (001).
        We can infer from this 'rule' that the maximum number of our action 7 (7 = 111) means we quarantine 
        all three people.
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
        self.observed_day = 0
        self.current_state[0][0] = self.simulated_state["Prediction"][0][self.observed_day]
        self.observed_day = self.observed_day + 1

        # Assume we haven't found any 2nd generation case have symptoms at first
        if not return_info:
            return self.current_state
        else:
            return self.current_state, {}

    def step(self, action):
        # Update the state from the result of simulation
        self.current_state[0][0] = self.simulated_state["Prediction"][0][self.observed_day]
        sum1 = 0
        sum2 = 0

        """
        # Baseline
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
        if self.simulated_state["Whether infected"][0][self.observed_day] == 1 and action == 0:
            sum1 = sum1 + 1
        if self.simulated_state["Whether infected"][0][self.observed_day] == 0 and action == 1:
            sum2 = sum2 + 1
        # """

        """
        # NN
        model = NeuralNetwork().double()
        model.load_state_dict(torch.load('/Users/kevinxu/Desktop/model_weights.pth'))
        input_data = self.simulated_state["Showing symptoms"].reshape(self.size, self.days)
        # Normalization
        norm = np.linalg.norm(input_data, ord=1)
        if norm != 0:
            input_data = input_data / norm
        input_data = torch.from_numpy(input_data)
        input_data = input_data.unsqueeze(0)
        NN_output = model(input_data)
        prediction = NN_output.detach().numpy()
        act = np.zeros((self.size, self.days))
        for i in range(self.size):
            for j in range(self.days):
                if prediction[0][i][j] > self.weights:
                    act[i][j] = 1
                else:
                    act[i][j] = 0

        for i in range(self.size):
            if self.simulated_state["Whether infected"][i][self.observed_day] == 1 and act[i][self.observed_day] == 0:
                sum1 = sum1 + 1
            if self.simulated_state["Whether infected"][i][self.observed_day] == 0 and act[i][self.observed_day] == 1:
                sum2 = sum2 + 1
        # """

        # Calculate the reward, reward = -1 * (infectious & not quarantine) - ratio * (not infectious & quarantine)
        reward = -1 * sum1 - self.ratio * sum2
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
        # Use an array that represents which people get infected. 1 represents get infected.
        self.simulated_state = {
            "Showing symptoms": np.zeros((self.size, self.days)),
            "Whether infected": np.zeros((self.size, self.days)),
            "Prediction": np.zeros((self.size, self.days))}

        # We assume that the index case has 0.2 probability to be highly transmissive.
        # Under that circumstance, the infectiousness rate becomes 5 times bigger.
        flag = bernoulli.rvs(self.p_high_transmissive, size=1)
        if flag == 1:
            self.p_infected = self.p_infected * 5
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
        if flag == 1:
            self.p_infected = self.p_infected / 5

        # """
        # Put the simulation to the NN
        model = NeuralNetwork().double()
        model.load_state_dict(torch.load('/Users/kevinxu/Desktop/model_weights.pth'))
        input_data = self.simulated_state["Showing symptoms"].reshape(self.size, self.days)
        # Normalization
        norm = np.linalg.norm(input_data, ord=1)
        if norm != 0:
            input_data = input_data / norm
        input_data = torch.from_numpy(input_data)
        input_data = input_data.unsqueeze(0)
        NN_output = model(input_data)
        prediction = NN_output.detach().numpy()
        self.simulated_state["Prediction"] = prediction[0]
        #"""

        return self.simulated_state

    def render(self, mode='None'):
        pass

    def close(self):
        pass
