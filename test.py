import utils
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import bct as bct
import numpy as np
import pandas as pd

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

    a = utils.RandomBinGraph(11, 'sparsity', 0.5).generate()
    print(a)

    x = bct.efficiency_bin(a)
    print(x)

    # plot using networkx
    # G = nx.DiGraph(a)
    # nx.draw(G, with_labels=True)
    # # plt.show()

    df = pd.read_csv('./input/funcCon.csv', header=None)
    data_df = df[df[3] == 'NHAHR']
    # print(data_df)
    data_df = data_df.iloc[:, 4:]    # remove the first 4 columns
    # get data from 1st row
    data = data_df.iloc[2, :].values
    fc_array = utils.matrix_from_upper_triangle(data)
    # print(fc_array)
    # bin_fc_array = utils.BinarizeMatrix(fc_array, 'threshold', 0.3).binarize()
    # print(bin_fc_array)
    # plot the undirected graph
    # G = nx.Graph(bin_fc_array)
    # nx.draw(G, with_labels=True)
    # plt.show()

    # Create the figure and axis
    fig, ax = plt.subplots()
    threshold = 0.1
    bin_fc_array = utils.BinarizeMatrix(fc_array, 'threshold', threshold).binarize()
    G = nx.Graph(bin_fc_array)
    pos = nx.spring_layout(G)  # layout for the nodes
    nx.draw(G, pos=pos, with_labels=True)

    # Define the update function for the animation
    def update(threshold):
        # Binarize the matrix and create a new graph with the current threshold
        bin_fc_array = utils.BinarizeMatrix(fc_array, 'threshold', threshold).binarize()
        G = nx.Graph(bin_fc_array)
        
        # Clear the axis and draw the new graph
        ax.clear()
        nx.draw(G, pos=pos, with_labels=True)
        
        # Set the title to the current threshold
        ax.set_title("Threshold = {}".format(threshold))

    # Create the animation
    animation = FuncAnimation(fig, update, frames=np.arange(0.1, 1, 0.1), interval=500)
    animation.save('animation.mp4', writer='ffmpeg', fps=1)




    # rndgraph = utils.RandomBinGraph(4, 'threshold', 0.5).generate_weighted_graph()
    # print(rndgraph)
    # bin_matrix = utils.BinarizeMatrix(rndgraph, 'threshold', 0.5).binarize()
    # print(bin_matrix)

    # bin, rang = utils.RandomBinGraph(4, 'threshold', 0.5).generate()
    # print(bin)
    # print(rang)