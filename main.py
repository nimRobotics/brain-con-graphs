import numpy as np
import bct as bct
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import utils

# set seed
np.random.seed(123)


# if __name__ == '__main__':
#     # Generate a random binary graph with 10 nodes and 20 edges
#     adj_matrix = gen_random_bin_graph(nnodes=10, edge_param=0.1, is_sparsity=True)
#     print(adj_matrix)
#     print('Number of edges: ', np.sum(adj_matrix)/2)

#     # calculate the global efficiency
#     ge=bct.efficiency_bin(adj_matrix)
#     print(ge)

#     # # generate a random symmetric matrix
#     # adj_matrix = np.random.rand(4, 4)
#     # adj_matrix = (adj_matrix + adj_matrix.T) / 2
#     # adj_matrix = adj_matrix - np.diag(np.diag(adj_matrix))
#     # print(adj_matrix)

#     # load fnirs FC matrix csv file
#     adj_matrix_stim_0 = np.loadtxt('output/stim_0_fc_mean.csv', delimiter=',')

#     # binarize the matrix
#     bin_matrix = binarize_matrix(adj_matrix_stim_0, sparsity=0.5)
#     print(bin_matrix)


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

    fnirs_adj_matrix_stim_0 = np.loadtxt('output/stim_1_fc_mean.csv', delimiter=',')

    for s in sparsity:
        ge_list = []
        cc_list = []
        for i in range(ngraphs):
            adj_matrix = utils.gen_random_bin_graph(nnodes=nnodes, edge_param=s, is_sparsity=True)
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
        global_efficiency_fnirs.append(bct.efficiency_bin(utils.binarize_matrix(fnirs_adj_matrix_stim_0, binarize_type='sparsity', binarize_param=s)))
        cluster_coef_fnirs.append(np.mean(bct.clustering_coef_bu(utils.binarize_matrix(fnirs_adj_matrix_stim_0, binarize_type='sparsity', binarize_param=s))))

    plt.errorbar(sparsity, global_efficiency_mean, yerr=global_efficiency_std, fmt='-o', capsize=3)
    plt.plot(sparsity, global_efficiency_fnirs, '-o')
    plt.xlabel('Sparsity')
    plt.ylabel('Global Efficiency - Stim 1')
    plt.legend(['fnirs data', 'random graph'])
    plt.savefig('output/global_efficiency_stim_1.png')

    plt.clf()

    plt.errorbar(sparsity, cluster_coef_mean, yerr=cluster_coef_std, fmt='-o', capsize=3)
    plt.plot(sparsity, cluster_coef_fnirs, '-o')
    plt.xlabel('Sparsity')
    plt.ylabel('Clustering Coefficient - Stim 1')
    plt.legend(['fnirs data', 'random graph'])
    plt.savefig('output/clustering_coef_stim_1.png')