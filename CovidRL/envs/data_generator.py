from CovEnv_base import CovidEnv_base
from csv import writer
import numpy as np
import copy

file_sf = open('./data/data_symptom_size16.csv', 'w')
writer_sf = writer(file_sf)
file_if = open('./data/data_infection_size16.csv', 'w')
writer_if = writer(file_if)
for times in range(0, 100):
    env = CovidEnv_base()
    full_s = np.resize(env.simulated_state["Showing symptoms"], (env.size, env.days)).astype(float)
    full_i = np.resize(env.simulated_state["Whether infected"], (env.size, env.days)).astype(float)
    now_s = np.full((5, env.days),1)
    # now_i = np.full((env.size, env.days))
    for day in range(0, env.days):
        for case in range(0, env.size):
            index = np.nonzero(full_s[case] == 1.)[0]
            if len(index) != 0:
                now_s[0][0: index[0]] = 0
            else: 
                now_s[0][day] = full_s[case][day]
            now_s[1][day] = 0

            symptom_num = 0.0
            for num in range(0, env.size):
                if num != case and full_s[num][day] == 1.0:
                    symptom_num = symptom_num + 1.0
            now_s[2][day] = symptom_num
            now_s[3, :] = env.size - 1
            now_s[4] = range(1, 31)
            
            now_s = (now_s - now_s.mean()) / now_s.std()

            writer_sf.writerows(now_s.astype(float))

            # focused matrix for infections output
            writer_if.writerow(full_i[case])

file_sf.close()
file_if.close()