from CovidEnvironment_test import CovidEnv
from csv import writer
import numpy as np

file_s = open('test.csv', 'w')
writer_s = writer(file_s)

for i in range(0, 100):
    env = CovidEnv()
    full_s = (env.simulated_state["Test results"])


    writer_s.writerows(full_s)
    writer_s.writerow("")

file_s.close()
