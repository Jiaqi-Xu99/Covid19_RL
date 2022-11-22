from CovidEnvironment3 import CovidEnv3
from csv import writer
import numpy as np
import copy

file_sf = open('data_symptom.csv', 'w')
writer_sf = writer(file_sf)
file_if = open('data_infection.csv', 'w')
writer_if = writer(file_if)
for times in range(0, 500):
    env = CovidEnv3()
    full_s = np.resize(env.simulated_state["Showing symptoms"], (env.size, env.days)).astype(float)
    full_i = np.resize(env.simulated_state["Whether infected"], (env.size, env.days)).astype(float)
    now_s = np.full((env.size, env.days), -1.0)
    now_i = np.full((env.size, env.days), -1.0)

    percentage = np.full((env.size, env.days), -1.0)
    for day in range(0, env.days):
        for j in range(0, env.size):
            now_s[j][day] = full_s[j][day]

        for contact in range(0, env.size):
            symptom_num = 0.0
            for num in range(0, env.size):
                if num != contact and full_s[num][day] == 1.0:
                    symptom_num = symptom_num + 1.0
            percentage[contact][day] = symptom_num / 3.0
            contact_s = np.append(now_s[contact], percentage[contact], axis=0).reshape(2, env.days)

            writer_sf.writerows(contact_s.astype(float))
            writer_sf.writerow([])

            # focused matrix for infections output
            writer_if.writerow(full_i[contact])
            writer_if.writerow([])

file_sf.close()
file_if.close()
