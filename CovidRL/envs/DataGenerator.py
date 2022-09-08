from CovidEnvironment import CovidEnv
from csv import writer
import numpy as np

file_s = open('data_symptom.csv', 'w')
writer_s = writer(file_s)
file_i = open('data_infection.csv', 'w')
writer_i = writer(file_i)

for i in range(0, 500):
    env = CovidEnv()
    full_s = np.resize(env.simulated_state["Showing symptoms"], (env.size, env.days))
    now_s = np.full((env.size, env.days), -1)
    # Normalization
    norm = np.linalg.norm(now_s, ord=1)
    if norm != 0:
        now_s = now_s / norm

    for day in range(0, env.days):
        # unfocused matrix for symptoms input
        for j in range(0, env.size):
            now_s[j][day] = full_s[j][day]

        writer_s.writerows(now_s)
        writer_s.writerow([])

        # unfocused matrix for infections output
        writer_i.writerows(env.simulated_state["Whether infected"])
        writer_i.writerow([])

file_s.close()
file_i.close()
