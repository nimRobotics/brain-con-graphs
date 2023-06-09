import numpy as np
import bct as bct
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import utils

# set seed
np.random.seed(123)

if __name__ == '__main__':
    nnodes = 10
    ngraphs = 1000

    # plot the variation of global efficiency with sparsity from 0 to 1
    sparsity = np.linspace(0, 1, 100)

    global_efficiency_mean = []
    global_efficiency_std = []
    global_efficiency_fnirs = []

    cluster_coef_mean = []
    cluster_coef_std = []
    cluster_coef_fnirs = []

    fnirs_adj_matrix_stim_0 = np.loadtxt('output/stim_0_fc_mean.csv', delimiter=',')
    # get min and max weights ignoring nan values
    min_weight = np.nanmin(fnirs_adj_matrix_stim_0)
    max_weight = np.nanmax(fnirs_adj_matrix_stim_0)
    print('min weight: ', min_weight)
    print('max weight: ', max_weight)

    for s in sparsity:
        ge_list = []
        cc_list = []
        for i in range(ngraphs):
            adj_matrix = utils.RandomBinGraph(nnodes=nnodes, binarize_type='threshold', binarize_param=s).generate(min_weight=min_weight, max_weight=max_weight)
            ge = bct.efficiency_bin(adj_matrix)
            cc = np.mean(bct.clustering_coef_bu(adj_matrix))
            ge_list.append(ge)
            cc_list.append(cc)

        ge_mean = np.mean(ge_list)
        cc_mean = np.mean(cc_list)
        ge_std = np.std(ge_list)
        cc_std = np.std(cc_list)

        global_efficiency_mean.append(ge_mean)
        cluster_coef_mean.append(cc_mean)
        global_efficiency_std.append(ge_std)
        cluster_coef_std.append(cc_std)
        global_efficiency_fnirs.append(bct.efficiency_bin(utils.BinarizeMatrix(fnirs_adj_matrix_stim_0, 'threshold', s).binarize()))
        cluster_coef_fnirs.append(np.mean(bct.clustering_coef_bu(utils.BinarizeMatrix(fnirs_adj_matrix_stim_0, 'threshold', s).binarize())))

    plt.errorbar(sparsity, global_efficiency_mean, yerr=global_efficiency_std, fmt='-o', capsize=3)
    plt.plot(sparsity, global_efficiency_fnirs, '-o')
    plt.xlabel('Sparsity')
    plt.ylabel('Global Efficiency - Stim 0')
    plt.legend(['fnirs data', 'random graph'])
    plt.savefig('output/global_efficiency_stim_0.png')

    plt.clf()

    plt.errorbar(sparsity, cluster_coef_mean, yerr=cluster_coef_std, fmt='-o', capsize=3)
    plt.plot(sparsity, cluster_coef_fnirs, '-o')
    plt.xlabel('Sparsity')
    plt.ylabel('Clustering Coefficient - Stim 0')
    plt.legend(['fnirs data', 'random graph'])
    plt.savefig('output/clustering_coef_stim_0.png')