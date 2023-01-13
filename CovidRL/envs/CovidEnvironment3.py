import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy.stats import bernoulli
from .Supervised_Learning  import NeuralNetwork
import torch


class CovidEnv3(gym.Env):

    def __init__(self):
        self.size = 4  # The size of the second generation
        self.days = 30 # Assume we observe the 2nd generation for 30 days
        # We assume weight_infect_no_quarantine = -1, and calculate weight_no_infect_quarantine from the ratio
        self.ratio = 0.1
        self.weights = 1/(1 + self.ratio)
        self.ratio2 = 0.01
        self.p_high_transmissive = 0.109  # Probability that the index case is highly transmissive
        self.p_infected = 0.0116  # Probability that a person get infected (given index case is not highly transmissive)
        self.p_symptomatic = 0.8  # Probability that a person is infected and showing symptom
        self.p_symptom_not_infected = 0.01  # Probability that a person showing symptom but not infected
        # The agent starts to get involved in day 3
        self.observed_day = 3
        # We run the simulation from day 1 to self.days to get the whole state of our environment
        self.simulated_state = self._simulation()
        self.prediction = np.zeros(self.days)
        # find when first show symptom
        self.first_symptom = -1
        for i in range(self.days):
            if self.simulated_state["Showing symptoms"][0][i] == 1:
                self.first_symptom = i + 1
                break

        # Initialize the model
        self.model = NeuralNetwork().double()
        self.model.load_state_dict(torch.load('/Users/kevinxu/Desktop/CovidRL/CovidRL/envs/model.pth'))
        # Build the input data
        self.input_data = np.full((5, self.days+1), 1)
        for index in range(0, 5):
            self.input_data[index][0] = 0
        if self.first_symptom >= self.observed_day:
            for day in range(0, self.observed_day):
                self.input_data[0][day+1] = 0
        else:
            for day in range(0, self.first_symptom):
                self.input_data[0][day+1] = 0
        for day in range (0,self.observed_day):
            self.input_data[1][day+1] = 0
        symptom_num = 0.0
        for day in range(0, self.observed_day):
            for num in range (1, self.size):
                if self.simulated_state["Showing symptoms"][num][day] == 1.0:
                    symptom_num = symptom_num + 1.0
            self.input_data[2][day+1] = symptom_num
        for day in range(0, self.observed_day):
            self.input_data[3][day+1] = self.size - 1
        self.input_data[4] = range(0, 31)
        self.input_data = (self.input_data - self.input_data.mean()) / self.input_data.std()
        # Put the observed state to the NN
        data = torch.from_numpy(self.input_data.astype(float))
        data = data.view(1, 1, 5, 31)
        NN_output = self.model(data)
        self.prediction = NN_output.detach().numpy()

        """
        Use a long vector to represent the observation. Only focus on one 2nd generation.
        1 means showing symptoms. 0 means not showing symptoms. -1 means unobserved.
        We assume that person get exposed at day 0. 
        Index 0 to 2 represent the prediction of whether show symptoms during three days before the observing day. 
        Index 3 to 5 represent the prediction of whether show symptoms during three days after the observing day. 
        Index 6 to 8 represent whether that person shows symptoms in recent three days.
        Index 9 represents whether testing
        """
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        self.current_state = np.zeros(10)
        for i in range(1, 3):
            self.current_state[3 - i] = self.prediction[0][0][0][self.observed_day - i]
            self.current_state[3 + i - 1] = self.prediction[0][0][0][self.observed_day + i - 1]
            self.current_state[6 + 3 - i] = self.simulated_state["Showing symptoms"][0][self.observed_day - i]
        if self.simulated_state["Showing symptoms"][0][self.observed_day - 1] + \
                self.simulated_state["Showing symptoms"][0][self.observed_day - 2] == 1:
            self.current_state[9] = 1

        """
        0 for not quarantine and not testing, 1 for quarantine but not testing
        2 for not quarantine but testing, 3 for quarantine and testing
        """
        self.action_space = spaces.Discrete(4)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # We start to trace when the 1st generation case is tested positive.
    def reset(self, return_info=False):
        self.observed_day = 3
        # find when first show symptom
        self.first_symptom = -1
        for i in range(self.days):
            if self.simulated_state["Showing symptoms"][0][i] == 1:
                self.first_symptom = i + 1
                break
        self.simulated_state = self._simulation()
        self.prediction = np.zeros(self.days)
        # Build the input data
        self.input_data = np.full((5, self.days + 1), 1)
        for index in range(0, 5):
            self.input_data[index][0] = 0
        if self.first_symptom >= self.observed_day:
            for day in range(0, self.observed_day):
                self.input_data[0][day + 1] = 0
        else:
            for day in range(0, self.first_symptom):
                self.input_data[0][day + 1] = 0
        for day in range(0, self.observed_day):
            self.input_data[1][day + 1] = 0
        symptom_num = 0.0
        for day in range(0, self.observed_day):
            for num in range(1, self.size):
                if self.simulated_state["Showing symptoms"][num][day] == 1.0:
                    symptom_num = symptom_num + 1.0
            self.input_data[2][day + 1] = symptom_num
        for day in range(0, self.observed_day):
            self.input_data[3][day + 1] = self.size - 1
        self.input_data[4] = range(0, 31)
        self.input_data = (self.input_data - self.input_data.mean()) / self.input_data.std()
        # Put the observed state to the NN
        data = torch.from_numpy(self.input_data.astype(float))
        data = data.view(1, 1, 5, 31)
        NN_output = self.model(data)
        self.prediction = NN_output.detach().numpy()
        # Initialize the current state
        self.current_state = np.zeros(10)
        for i in range(1, 3):
            self.current_state[3 - i] = self.prediction[0][0][0][self.observed_day - i]
            self.current_state[3 + i - 1] = self.prediction[0][0][0][self.observed_day + i - 1]
            self.current_state[6 + 3 - i] = self.simulated_state["Showing symptoms"][0][self.observed_day - i]
        if self.simulated_state["Showing symptoms"][0][self.observed_day - 1] + \
                self.simulated_state["Showing symptoms"][0][self.observed_day - 2] == 1:
            self.current_state[9] = 1
        # Assume we haven't found any 2nd generation case have symptoms at first
        if not return_info:
            return self.current_state
        else:
            return self.current_state, {}

    def step(self, action):
        # Update the input data
        if self.first_symptom > self.observed_day:
            self.input_data[0][self.observed_day+1] = 0
        self.input_data[1][self.observed_day+1] = 0
        symptom_num = 0.0
        for i in range(1, self.size):
            if self.simulated_state["Showing symptoms"][i][self.observed_day] == 1:
                symptom_num = symptom_num + 1.0
        self.input_data[2][self.observed_day+1] = symptom_num
        # Put the updated observed state to the NN
        data = torch.from_numpy(self.input_data.astype(float))
        data = data.view(1, 1, 5, 31)
        NN_output = self.model(data)
        prediction = NN_output.detach().numpy()
        self.prediction = prediction
        # Update the state
        for i in range(1, 3):
            self.current_state[3 - i] = self.prediction[0][0][0][self.observed_day - i + 1]
            self.current_state[3 + i - 1] = self.prediction[0][0][0][self.observed_day + i]
            self.current_state[6 + 3 - i] = self.simulated_state["Showing symptoms"][0][self.observed_day - i + 1]
        if self.simulated_state["Showing symptoms"][0][self.observed_day] + self.simulated_state["Showing symptoms"][0][self.observed_day - 1] == 1:
            self.current_state[9] = 1

        sum1 = 0
        sum2 = 0
        sum3 = 0

        #"""
        # RL
        if self.simulated_state["Whether infected"][0][self.observed_day] == 1 and (action == 0 or action == 2):
            sum1 = sum1 + 1
        if self.simulated_state["Whether infected"][0][self.observed_day] == 0 and (action == 1 or action == 3):
            sum2 = sum2 + 1
        if  action == 2 or action == 3:
            sum3 = sum3 + 1
        # """

        """
        # 14 days quarantine
        if self.first_symptom >= 0:
            if self.simulated_state["Whether infected"][0][self.observed_day] == 1 and (self.observed_day < self.first_symptom or self.observed_day > self.first_symptom + 13):
                sum1 = sum1 + 1
            if self.simulated_state["Whether infected"][0][self.observed_day] == 0 and self.first_symptom <= self.observed_day <= self.first_symptom + 13:
                sum2 = sum2 + 1
        else:
            if self.simulated_state["Whether infected"][0][self.observed_day] == 1:
                sum1 = sum1 + 1
        if self.observed_day == self.first_symptom + 13:
            sum3 = sum3 + 1
            if self.simulated_state["Whether infected"][0][self.observed_day] == 1:
                self.first_symptom = self.observed_day
        #"""

        """
        # No quarantine
        if self.simulated_state["Whether infected"][0][self.observed_day] == 1:
            sum1 = sum1 + 1
        #"""

        """
        # Threshold
        if self.prediction[0][0][0][self.observed_day] > self.weights:
            act = 1
        else:
            act = 0

        if self.simulated_state["Whether infected"][0][self.observed_day] == 1 and act == 0:
            sum1 = sum1 + 1
        if self.simulated_state["Whether infected"][0][self.observed_day] == 0 and act == 1:
            sum2 = sum2 + 1
        # """

        """
        # Quarantine based on symptoms
        if self.simulated_state["Whether infected"][0][self.observed_day] == 1 and self.simulated_state["Showing symptoms"][0][self.observed_day] == 0:
            sum1 = sum1 + 1
        if self.simulated_state["Whether infected"][0][self.observed_day] == 0 and self.simulated_state["Showing symptoms"][0][self.observed_day] == 1:
            sum2 = sum2 + 1
        # """

        """
        # Quarantine everyday
        if self.simulated_state["Whether infected"][0][self.observed_day] == 0:
            sum2 = sum2 + 1
        # """

        """
        # CDC policy
        if self.first_symptom >= 0:
            if self.simulated_state["Whether infected"][0][self.observed_day] == 1 and (
                    self.observed_day < self.first_symptom or self.observed_day > self.first_symptom + 5):
                sum1 = sum1 + 1
            if self.simulated_state["Whether infected"][0][
                self.observed_day] == 0 and self.first_symptom <= self.observed_day <= self.first_symptom + 5:
                sum2 = sum2 + 1
            if self.observed_day == self.first_symptom + 5:
                sum3 = sum3 + 1
                if self.simulated_state["Whether infected"][0][self.observed_day] == 1:
                    self.first_symptom = self.observed_day
        else:
            if self.simulated_state["Whether infected"][0][self.observed_day] == 1:
                sum1 = sum1 + 1
        # """

        # Calculate the reward, reward = -1 * (infectious & not quarantine) - ratio * (not infectious & quarantine)
        reward = (-1 * sum1 - self.ratio * sum2 - self.ratio2 * sum3) * 100 # Multiply 100 to compare the results more easily
        self.observed_day = self.observed_day + 1
        done = bool(self.observed_day == self.days-3)

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

            #  developing symptoms that is independent of infection status
            symptom_not_infected = bernoulli.rvs(self.p_symptom_not_infected, size=self.days)
            for j in range (self.days):
                if symptom_not_infected[j] == 1:
                    self.simulated_state["Showing symptoms"][i][j] = 1

        if flag == 1:
            self.p_infected = self.p_infected / 24.4


        return self.simulated_state

    def render(self, mode='None'):
        pass

    def close(self):
        pass
