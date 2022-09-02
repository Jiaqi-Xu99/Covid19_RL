import numpy as np
from scipy.stats import bernoulli


size = 4  # The size of the second generation
days = 30  # Assume we observe the 2nd generation for 40 days
# We assume weight_infect_no_quarantine = -1, and calculate weight_no_infect_quarantine from the ratio
p_high_transmissive = 0.2  # Probability that the index case is highly transmissive
p_symptomatic = 0.8  # Probability that a person is infected and showing symptom

simulated_state = {
    "Showing symptoms": np.zeros((size, days)),
    "Whether infected": np.zeros((size, days))}
ratio_of_weights = 0
p_infected = 0  # Probability that a person get infected

while p_infected <= 0.1:
    while ratio_of_weights <= 1:
        print(p_infected)
        print(ratio_of_weights)
        flag = bernoulli.rvs(p_high_transmissive, size=1)
        if flag == 1:
            p_infected = p_infected * 5
            if p_infected > 1:
                p_infected = 1
        infected_case = np.array(bernoulli.rvs(p_infected, size=size))
        for i in range(size):
            #  Whether infected
            if infected_case[i] == 1:
                # infected and show symptoms
                if bernoulli.rvs(p_symptomatic, size=1) == 1:
                    # Use log normal distribution, mean = 1.57, std = 0.65
                    symptom_day = int(np.random.lognormal(1.57, 0.65, 1))  # day starts to show symptom
                    #  Assume the symptom will last six days when it starts
                    for j in range(symptom_day, symptom_day + 6):
                        if 0 <= j < days:
                            simulated_state["Showing symptoms"][i][j] = 1
                    #  Whether infected
                    for j in range(symptom_day - 2, symptom_day + 6):
                        if 0 <= j < days:
                            simulated_state["Whether infected"][i][j] = 1
                # infected but not showing symptoms
                else:
                    symptom_day = int(np.random.lognormal(1.57, 0.65, 1))
                    for j in range(symptom_day - 2, symptom_day + 6):
                        if 0 <= j < days:
                            simulated_state["Whether infected"][i][j] = 1
        sum1 = 0
        # No quarantine
        for i in range(size):
            for observed_day in range(days):
                if simulated_state["Whether infected"][i][observed_day] == 1:
                    sum1 = sum1 + 1
        reward = -1 * sum1

        sum1 = 0
        sum2 = 0
        for i in range(size):
            for observed_day in range(days):
                if simulated_state["Whether infected"][i][observed_day] == 1 and 0 <= observed_day <= 2 and \
                        17 <= observed_day < days:
                    sum1 = sum1 + 1
                if simulated_state["Whether infected"][i][observed_day] == 0 and 3 <= observed_day <= 16:
                    sum2 = sum2 + 1
        reward2 = -1 * sum1 - ratio_of_weights * sum2

        if reward >= reward2:
            print("N")
        else:
            print("B")

        if flag == 1:
            p_infected = p_infected / 5
        ratio_of_weights = ratio_of_weights + 0.01
    p_infected = p_infected + 0.01
    ratio_of_weights = 0
