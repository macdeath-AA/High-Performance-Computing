import numpy as np
import matplotlib.pyplot as plt

tot_t, comm_t, comp_t = np.loadtxt('times.txt', unpack=True)
with open('pvtime.txt',"a") as final_file:
    final_file.write(
        f'8 {np.average(tot_t)} {np.average(comm_t)} {np.average(comp_t)}\n')
