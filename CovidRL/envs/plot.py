import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt


def simulation(size, days):
    # Use an array that represents which people get infected. 1 represents get infected.
    simulated_state = {
        "Showing symptoms": np.zeros((size, days)),
        "Whether infected": np.zeros((size, days))}
    # We assume that the index case has 0.109 probability to be highly transmissive.
    # Under that circumstance, the infectiousness rate becomes 24.4 times bigger.
    p_high_transmissive = 0.109  # Probability that the index case is highly transmissive
    p_infected = 0.0116  # Probability that a person get infected (given index case is not highly transmissive)
    p_symptomatic = 0.8  # Probability that a person is infected and showing symptom
    p_symptom_not_infected = 0.01  # Probability that a person showing symptom but not infected
    flag = bernoulli.rvs(p_high_transmissive, size=1)
    if flag == 1:
        p_infected = p_infected * 24.4
    infected_case = np.array(bernoulli.rvs(p_infected, size=size))
    for i in range(size):
        #  Whether infected
        if infected_case[i] == 1:
            # infected and show symptoms
            if bernoulli.rvs(p_symptomatic, size=1) == 1:
                # Use log normal distribution, mean = 1.57, std = 0.65
                symptom_day = int(np.random.lognormal(1.57, 0.65, 1))  # day starts to show symptom
                duration = int(np.random.lognormal(2.70, 0.15, 1))  # duration of showing symptom
                for j in range(symptom_day, symptom_day + duration):
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
        # not infected but show some symptoms

        #  developing symptoms that is independent of infection status
        symptom_not_infected = bernoulli.rvs(p_symptom_not_infected, size=days)
        for j in range(days):
            if symptom_not_infected[j] == 1:
                simulated_state["Showing symptoms"][i][j] = 1

    if flag == 1:
        p_infected = p_infected / 24.4
    return simulated_state


def quarantine(first_symptom):
    # 14 days quarantine
    sum1 = 0
    sum2 = 0
    duration = 13
    first_symptom1 = first_symptom
    for observed_day in range(days):
        if first_symptom1 >= 0:
            if observed_day > first_symptom1 + duration:
                if simulated_state["Showing symptoms"][0][observed_day] == 1:
                    first_symptom1 = observed_day
                    duration = 0
                else:
                    first_symptom1 = -1
            if simulated_state["Whether infected"][0][observed_day] == 1 and (observed_day < first_symptom1 or observed_day > first_symptom1 + duration):
                sum1 = sum1 + 1
            if simulated_state["Whether infected"][0][observed_day] == 0 and first_symptom1 <= observed_day <= first_symptom1 + duration:
                sum2 = sum2 + 1
        else:
            if simulated_state["Whether infected"][0][observed_day] == 1:
                sum1 = sum1 + 1
    rewards = (-1 * sum1 - ratio * sum2) * 100
    return rewards


def no_quarantine():
    # No quarantine
    sum1 = 0
    for observed_day in range(days):
        if simulated_state["Whether infected"][0][observed_day] == 1:
            sum1 = sum1 + 1
    rewards = -1 * sum1 * 100
    return rewards


def cdc(first_symptom):
    # CDC policy
    sum1 = 0
    sum2 = 0
    sum3 = 0
    duration = 4
    first_symptom2 = first_symptom
    for observed_day in range(days):
        if first_symptom2 >= 0:
            if observed_day > first_symptom2 + duration:
                if simulated_state["Showing symptoms"][0][observed_day] == 1:
                    first_symptom2 = observed_day
                    duration = 0
                else:
                    first_symptom2 = -1
            if simulated_state["Whether infected"][0][observed_day] == 1 and (
                    observed_day < first_symptom2 or observed_day > first_symptom2 + duration):
                sum1 = sum1 + 1
            if simulated_state["Whether infected"][0][
                observed_day] == 0 and first_symptom2 <= observed_day <= first_symptom2 + duration:
                sum2 = sum2 + 1
            if observed_day == first_symptom2 + duration:
                sum3 = sum3 + 1
        else:
            if simulated_state["Whether infected"][0][observed_day] == 1:
                sum1 = sum1 + 1
    rewards = (-1 * sum1 - ratio * sum2 - ratio2 * sum3) * 100
    return rewards


if __name__ == "__main__":
    size = 4  # The size of the second generation
    days = 30  # Assume we observe the 2nd generation for 30 days
    ratio2 = 0.01
    rewards1 = np.zeros(501)
    rewards2 = np.zeros(501)
    rewards3 = np.zeros(501)
    for r in range(0, 501):
        ratio = r * 0.0001
        reward1 = 0
        reward2 = 0
        reward3 = 0
        for times in range(0, 1000):
            simulated_state = simulation(size, days)
            # find when first show symptom
            first_symptom = -1
            for index in range(days):
                if simulated_state["Showing symptoms"][0][index] == 1:
                    first_symptom = index
                    break
            reward1 = reward1 + quarantine(first_symptom)
            reward2 = reward2 + no_quarantine()
            reward3 = reward3 + cdc(first_symptom)
        reward1 = reward1 / 1000
        reward2 = reward2 / 1000
        reward3 = reward3 / 1000
        rewards1[r] = reward1
        rewards2[r] = reward2
        rewards3[r] = reward3
    # plotting the points
    plt.plot(np.arange(0.0, 0.0501, 0.0001), rewards1, label="14_Q")
    plt.plot(np.arange(0.0, 0.0501, 0.0001), rewards2, label="no_Q")
    plt.plot(np.arange(0.0, 0.0501, 0.0001), rewards3, label="CDC")
    # naming the x_axis
    plt.xlabel('Ratio')
    # naming the y_axis
    plt.ylabel('Reward')
    # giving a title to my graph
    plt.title('Graph')
    # show a legend on the plot
    plt.legend()
    # function to show the plot
    plt.show()
    plt.savefig('ratio.png')
