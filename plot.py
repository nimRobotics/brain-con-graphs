import numpy as np
import matplotlib.pyplot as plt


# load csv in numpy array
NHAHR = np.loadtxt('output/NHAHR_fc_graphs_global_efficiency_threshold.csv', delimiter=',')
NHALR = np.loadtxt('output/NHALR_fc_graphs_global_efficiency_threshold.csv', delimiter=',')
FHAHR = np.loadtxt('output/FHAHR_fc_graphs_global_efficiency_threshold.csv', delimiter=',')
FHALR = np.loadtxt('output/FHALR_fc_graphs_global_efficiency_threshold.csv', delimiter=',')

plt.errorbar(NHAHR[0], np.mean(NHAHR[1:], axis=0), yerr=np.std(NHAHR[1:], axis=0), label='NHAHR')
plt.errorbar(NHALR[0], np.mean(NHALR[1:], axis=0), yerr=np.std(NHALR[1:], axis=0), label='NHALR')
plt.errorbar(FHAHR[0], np.mean(FHAHR[1:], axis=0), yerr=np.std(FHAHR[1:], axis=0), label='FHAHR')
plt.errorbar(FHALR[0], np.mean(FHALR[1:], axis=0), yerr=np.std(FHALR[1:], axis=0), label='FHALR')

plt.legend()
plt.show()

