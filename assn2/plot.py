import numpy as np
import matplotlib.pyplot as plt

p, tot_t, comm_t, comp_t = np.loadtxt('pvtime.txt', unpack=True)

plt.figure(figsize=(9,7))
plt.plot(p,tot_t, '-o', label = 'Total time')
plt.plot(p, comm_t, '-o',label= 'Communication time')
plt.plot(p, comp_t, '-o',label = 'Computation time')
plt.xlabel('Processors')
plt.ylabel('Time taken')
plt.title('Time vs P for NX =10240')
plt.legend()
plt.savefig('constnx.png')