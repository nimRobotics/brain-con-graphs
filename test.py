import utils
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == '__main__':
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

    # mat = utils.gen_random_weighted_graph(4, 0.1, 0.9)
    # print(mat)


    # G = nx.DiGraph(mat)

    # # plot the graph with labels and weights
    # nx.draw(G, with_labels=True)
    # plt.show()

    # x = utils.BinarizeMatrix(mat, 'sparsity', 0.5).binarize()
    # print(x)

    a, _ = utils.RandomBinGraph(6, 'sparsity', 0.5).generate()
    print(a)

    rndgraph = utils.RandomBinGraph(4, 'threshold', 0.5).generate_weighted_graph()
    print(rndgraph)
    bin_matrix = utils.BinarizeMatrix(rndgraph, 'threshold', 0.5).binarize()
    print(bin_matrix)

    bin, rang = utils.RandomBinGraph(4, 'threshold', 0.5).generate()
    print(bin)
    print(rang)