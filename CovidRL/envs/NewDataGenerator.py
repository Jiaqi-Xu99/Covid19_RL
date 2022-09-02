from CovidEnvironment import CovidEnv
from csv import writer
import numpy as np
import copy

file_sf = open('data_symptom_focused.csv', 'w')
writer_sf = writer(file_sf)
file_if = open('data_infection_focused.csv', 'w')
writer_if = writer(file_if)

for i in range(0, 10):
    env = CovidEnv()
    full_s = np.resize(env.simulated_state["Showing symptoms"], (env.size, env.days))
    now_s = np.full((env.size, env.days), -1)

    for day in range(0, env.days):
        # unfocused matrix for symptoms input
        for j in range(0, env.size):
            now_s[j][day] = full_s[j][day]
            # focused matrix for symptoms input
        for contact in range(0, env.size):
            focused_a = [now_s[contact]]
            unfocused = copy.deepcopy(now_s)
            unfocused = np.delete(unfocused, contact, axis=0)
            divider = env.size - 1
            normalized_a = np.zeros((1, env.days))
            for row in unfocused:
                normalized_a = normalized_a + row
            normalized_a = normalized_a / divider
            contact_s = np.append(focused_a, normalized_a, axis=0)

            writer_sf.writerows(contact_s)
            writer_sf.writerow([])

            # focused matrix for infections output
            writer_if.writerow(env.simulated_state["Whether infected"][contact])
            writer_if.writerow([])

file_sf.close()
file_if.close()
