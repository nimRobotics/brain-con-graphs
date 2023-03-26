import numpy as np
import bct as bct
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt

# set seed
np.random.seed(123)

def gen_random_bin_graph(nnodes, edge_param, is_sparsity=True):
    '''
    Generate a random binary graph with given nodes and edges/sparsity
    nnodes: number of nodes
    edge_param: number of edges or sparsity, depending on is_sparsity
    is_sparsity: if True, use edge_param as sparsity; if False, use edge_param as number of edges
    return: adjacency matrix
    '''
    # Calculate the number of edges based on sparsity, if necessary
    if is_sparsity:
        assert 0 <= edge_param <= 1, 'Sparsity must be between 0 and 1'
        nedges = int(edge_param * nnodes * (nnodes - 1) / 2)
    else:
        nedges = edge_param

    # Check if the number of edges is possible for the number of nodes
    max_edges = nnodes * (nnodes - 1) / 2
    if nedges > max_edges:
        raise ValueError('The number of edges is too large')

    # Create an empty adjacency matrix with nnodes x nnodes dimensions
    adj_matrix = np.zeros((nnodes, nnodes))

    # Generate a list of all possible edges
    all_edges = [(i, j) for i in range(nnodes) for j in range(i + 1, nnodes)]

    # Choose nedges random edges from the list of all possible edges
    random_edges = np.random.choice(len(all_edges), nedges, replace=False)

    # Add the chosen edges to the adjacency matrix
    for edge in random_edges:
        i, j = all_edges[edge]
        adj_matrix[i][j] = 1
        adj_matrix[j][i] = 1

    return adj_matrix

def binarize_matrix(matrix, sparsity=0.4):
    '''
    Binarize a matrix using the given sparsity value.
    Sparsity is defined as the number of non-zero edges divided by the total number of edges.
    
    Parameters:
        matrix (numpy.ndarray): the adjacency matrix to be binarized
        sparsity (float): the desired sparsity of the binarized matrix
    
    Returns:
        numpy.ndarray: the binarized adjacency matrix
    '''
    # set diagonal to 0
    np.fill_diagonal(matrix, 0)

    # Calculate the maximum number of edges in the matrix
    max_edges = matrix.shape[0] * (matrix.shape[0] - 1) / 2
    # print('Max edges: ', max_edges)
    
    # Calculate the number of non-zero edges based on the desired sparsity
    n_nonzero_edges = int(sparsity * max_edges)
    # print('Number of non-zero edges: ', n_nonzero_edges)
    
    # Get the indices of the top n_nonzero_edges values in the matrix
    # Note that we use ravel() to flatten the matrix into a 1D array before sorting
    edge_indices = np.argsort(matrix.ravel())[-n_nonzero_edges*2:]
    # print('Edge indices shape: ', edge_indices.shape)
    
    # Create a new binary matrix with the same shape as the input matrix
    bin_matrix = np.zeros_like(matrix)
    
    # Set the chosen edges to 1 in the binary matrix
    bin_matrix.ravel()[edge_indices] = 1
    # print('Binary matrix: ', bin_matrix)
    
    # Set the diagonal to 0
    np.fill_diagonal(bin_matrix, 0)
    
    # Ensure symmetry by copying the upper triangle to the lower triangle
    bin_matrix = np.triu(bin_matrix) + np.triu(bin_matrix, k=1).T
    
    return bin_matrix

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
            adj_matrix = gen_random_bin_graph(nnodes=nnodes, edge_param=s, is_sparsity=True)
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
        global_efficiency_fnirs.append(bct.efficiency_bin(binarize_matrix(fnirs_adj_matrix_stim_0, sparsity=s)))
        cluster_coef_fnirs.append(np.mean(bct.clustering_coef_bu(binarize_matrix(fnirs_adj_matrix_stim_0, sparsity=s))))

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