import utils
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import bct as bct
import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('./input/funcCon.csv', header=None)
    data_df = df[df[3] == 'NHAHR']
    data_df = data_df.iloc[:, 4:]    # remove the first 4 columns
    data_df = data_df.apply(pd.to_numeric, errors='coerce')   # convert to numeric
    data_df = data_df.reset_index(drop=True)    # reset index
    print(data_df)

    # average FC data
    data_df_ = data_df.mean(axis=0)   # get the mean of each column
    fc_array = utils.matrix_from_upper_triangle(data_df_)  # convert to matrix
    print(fc_array)
    print(data_df_)

    # # FC data from single partiopnatn
    # data = data_df.iloc[6, :].values
    # fc_array = utils.matrix_from_upper_triangle(data)
    # print(fc_array)

    # # test array FC
    # fc_array = np.array([[-0.2, 0.38], [0.2, .41]])
    # print(fc_array)
    # print(np.arctanh(fc_array))

    # binarize the matrix
    bin_matrix = utils.BinarizeMatrix(fc_array, 'threshold', 0.7).binarize()
    print(bin_matrix)
    print(np.arctanh(fc_array))

    # coordinates of the 11 brain region nodes
    node_coords = np.array([[0, 8], [0, 7], [0.5, 6.5], [-0.5, 6.5], [0, 6], [0.5, 5], [-0.5, 5], [0, 5], [0, 4], [0, 1], [0, 0]])
    node_labels = labels = ['APFC', 'MDPFC', 'RDPFC', 'LDPFC', 'IFC', 'RBA', 'LBA', 'PMC-SMA', 'M1', 'V2-V3', 'V1']

    # add jitter 0.1 to 0.2 to node coordinates to avoid overlapping edges
    node_coords = node_coords + np.random.uniform(-0.2, 0.2, size=node_coords.shape)

    # plot using networkx
    G = nx.DiGraph(bin_matrix)
    nx.draw(G, with_labels=True, pos=dict(zip(range(11), node_coords)), labels=dict(zip(range(11), node_labels)))
    plt.show()







 