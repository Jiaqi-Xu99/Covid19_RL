from CovidEnvironment3 import CovidEnv3
from csv import writer
import numpy as np


"""
    This code is for generating dataset for Supervise Learning training.
    Use 7*30 matrix to represent the feature of one case
    Index 0: The first day showing the symptom
    Index 1: Whether it is future or (previous and now), 1 represents future and 0 represents (previous and now).
    Index 2: The number of other cases who show symptom
    Index 3: The total number of other cases
    Index 4: Day
    Index 5: whether we should test
    Index 6: Test Results
"""

file_sf = open('data_feature.csv', 'w')
writer_sf = writer(file_sf)
file_if = open('data_infection.csv', 'w')
writer_if = writer(file_if)

for times in range(0, 400):
    env = CovidEnv3()
    full_s = np.resize(env.simulated_state["Showing symptoms"], (env.size, env.days)).astype(float)
    full_i = np.resize(env.simulated_state["Whether infected"], (env.size, env.days)).astype(float)
    for day in range(0, env.days):
        for case in range(0, env.size):
            feature = np.full((5, env.days), 0)
            # get first day showing symptoms
            idx1 = np.where(full_s[case] == 1)[0] 
            if idx1.size == 0:
                feature[0] = 0
            else:
                feature[0][idx1[0] + 1:env.days] = 1
            # Present future days, set to 1
            feature[1][day+1:env.days] = 1
            
            # Keep the first day that other cases show symptom(res)
            other_s = np.delete(full_s, case, 0)
            res = np.zeros(other_s.shape)
            idx = np.arange(res.shape[0])
            args = other_s.astype(bool).argmax(1)
            res[idx, args] = other_s[idx, args]
            res_sum = res.sum(axis=0)
            # Cumulatively get the number of cases showing symptom
            num = 0
            for i in range(1, env.days):
                if res_sum[i-1] == 1:
                    num = num + 1
                feature[2][i-1:env.days] = num
                
            feature[3, :] = env.size - 1
            feature[4] = range(1, 31) 
                 
            # Normalization
            feature = (feature - feature.mean()) / feature.std()

            writer_sf.writerows(feature.astype(float))
            writer_sf.writerow([])

            # infections output
            writer_if.writerow(full_i[case])
            writer_if.writerow([])

file_sf.close()
file_if.close()
