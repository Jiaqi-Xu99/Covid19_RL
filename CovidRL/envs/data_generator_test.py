# -*- coding: utf-8 -*-
# @Author: xueqiao
# @Date:   2022-12-02 13:50:44
# @Last Modified by:   xueqiao
# @Last Modified time: 2023-02-08 14:19:49
# from CovEnv_base import CovidEnv_base
from Covid3 import CovidEnv3
from csv import writer
import numpy as np
import random


"""
    This code is for generating dataset for Supervise Learning training.
    Use 5*30 matrix to represent the feature of one case
    Index 0: The first day showing the symptom
    Index 1: Whether it is future or (previous and now), 1 represents future and 0 represents (previous and now)
    Index 2: The number of other cases who show symtom
    Index 3: The total number of other cases
    Index 4: Day
    Index 5: Whether we test the case, 1 represents we did test(randomly test 3 days)
    Index 6: Test result, we assume test is 100% correct
"""

# file_sf = open('data_feature.csv', 'w')
# writer_sf = writer(file_sf)
# file_if = open('data_infection.csv', 'w')
# writer_if = writer(file_if)

for times in range(0, 100):
    env = CovidEnv3()
    full_s = np.resize(env.simulated_state["Showing symptoms"], (env.size, env.days)).astype(float)
    full_i = np.resize(env.simulated_state["Whether infected"], (env.size, env.days)).astype(float)
    for day in range(0, env.days):
        for case in range(0, env.size):
            feature = np.full((7, env.days),0)
            #get first day showing symptoms
            idx1 = np.where(full_s[case] == 1)[0] 
            if idx1.size == 0:
                feature[0] = 0
            else:
                feature[0][idx1[0]+1:env.days] = 1
            #present future days, set to 1
            feature[1][day+1:env.days] = 1
            
            #Keep the first day that other cases show syptom(res)
            other_s = np.delete(full_s,case,0)
            res = np.zeros(other_s.shape)
            idx =  np.arange(res.shape[0])
            args = other_s.astype(bool).argmax(1)
            res[idx, args] = other_s[idx, args]
            res_sum = res.sum(axis = 0)
            #cumulatively get the number of cases showing symptom 
            num = 0
            for i in range(1,env.days):
                if res_sum[i-1] == 1:
                    num = num + 1
                feature[2][i-1 : env.days]= num
                
            feature[3, :] = env.size - 1
            feature[4] = range(1, 31) 
            #randomly generate 10% test and assum test is 100% correct
            nums = np.zeros(30)
            nums[:3] = 1
            np.random.shuffle(nums)
            feature[5] = nums
            # print(feature[5])
            for k in range(0, env.days-1):
                if feature[5][k] == 1:
                    feature[6][k+1] = full_i[case][k]
            # normolize
            feature = (feature - feature.mean()) / feature.std()

#             writer_sf.writerows(feature.astype(float))
#             writer_sf.writerow([])

#             # infections output
#             writer_if.writerow(full_i[case])
#             writer_if.writerow([])

# file_sf.close()
# file_if.close()