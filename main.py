import numpy as np
import bct as bct
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import utils

# set seed
np.random.seed(123)

class RandomGraphAnalyzer:
    def __init__(self, fc_matrix: np.ndarray, n_rnd_graphs: int=1000, analysis_type: str='sparsity', features: list=['global_efficiency', 'local_efficiency', 'clustering_coefficient']):
        '''
        Parameters:
            fc_matrix (numpy.ndarray): the functional connectivity matrix
            n_rnd_graphs (int): the number of random graphs to be generated
            analysis_type (str): the type of analysis to be performed. Options are 'sparsity' or 'threshold'
            features (list): the features to be analyzed. Options are 'global_efficiency', 'local_efficiency', 'clustering_coefficient'
        '''
        self.fc_matrix = fc_matrix
        self.n_rnd_graphs = n_rnd_graphs
        self.analysis_type = analysis_type
        self.nnodes = fc_matrix.shape[0]
        self.nedges = int(self.nnodes * (self.nnodes - 1) / 2)
        
        if analysis_type == 'sparsity':
            self.sparsity = np.linspace(0, 1, 100)
            self.threshold = None
        elif analysis_type == 'threshold':
            self.threshold = np.linspace(0, 1, 100)
            self.sparsity = None
        else:
            raise ValueError('Invalid analysis_type. Must be either "sparsity" or "threshold".')

        self.features = features

    def gen_random_graphs(self):
        '''
        Generate random graphs using the given analysis type and number of graphs.
        '''
        if self.analysis_type == 'sparsity':
            # generate n_rnd_graphs random graphs for each sparsity value
            self.random_graphs = [[utils.RandomBinGraph(self.nnodes, binarize_type=self.analysis_type, binarize_param=sparsity).generate() for _ in range(self.n_rnd_graphs)] for sparsity in self.sparsity]
        elif self.analysis_type == 'threshold':
            self.random_graphs = [[utils.RandomBinGraph(self.nnodes, binarize_type=self.analysis_type, binarize_param=threshold).generate(min_weight=np.nanmin(self.fc_matrix), max_weight=np.nanmax(self.fc_matrix)) for _ in range(self.n_rnd_graphs)] for threshold in self.threshold]
        return self.random_graphs

    def compute_features(self):
        '''
        Compute the features of the random graphs.
        '''
        self.random_graphs_features = {}
        self.fc_graph_features = {}
        # choose threshold or sparsity based on self.analysis_type
        if self.analysis_type == 'sparsity':
            time_points = self.sparsity
        elif self.analysis_type == 'threshold':
            time_points = self.threshold

        for i, sparsity in enumerate(time_points):
            for feature in self.features:
                if feature == 'global_efficiency':
                    self.random_graphs_features['global_efficiency_{}'.format(sparsity)] = [bct.efficiency_bin(graph) for graph in self.random_graphs[i]]
                    self.fc_graph_features['global_efficiency_{}'.format(sparsity)] = bct.efficiency_bin(utils.BinarizeMatrix(self.fc_matrix, binarize_type=self.analysis_type, binarize_param=sparsity).binarize())
                elif feature == 'local_efficiency':
                    self.random_graphs_features['local_efficiency_{}'.format(sparsity)] = [bct.efficiency_bin(graph, local=True) for graph in self.random_graphs[i]]
                    self.fc_graph_features['local_efficiency_{}'.format(sparsity)] = bct.efficiency_bin(utils.BinarizeMatrix(self.fc_matrix, binarize_type=self.analysis_type, binarize_param=sparsity).binarize())
                elif feature == 'clustering_coefficient':
                    self.random_graphs_features['clustering_coefficient_{}'.format(sparsity)] = [np.mean(bct.clustering_coef_bu(graph)) for graph in self.random_graphs[i]]
                    self.fc_graph_features['clustering_coefficient_{}'.format(sparsity)] = np.mean(bct.clustering_coef_bu(utils.BinarizeMatrix(self.fc_matrix, binarize_type=self.analysis_type, binarize_param=sparsity).binarize()))
                else:
                    raise ValueError('Invalid feature. Must be either "global_efficiency", "local_efficiency", or "clustering_coefficient".')
        return self.random_graphs_features, self.fc_graph_features

    def plot_features(self):
        '''
        Plot the features of the random graphs.
        '''
        for i, feature in enumerate(self.features):
            plt.errorbar(self.sparsity, [np.mean(self.random_graphs_features['{}_{}'.format(feature, sparsity)]) for sparsity in self.sparsity], 
                         yerr=[np.std(self.random_graphs_features['{}_{}'.format(feature, sparsity)]) for sparsity in self.sparsity], label='Random Graphs')
            plt.plot(self.sparsity, [self.fc_graph_features['{}_{}'.format(feature, sparsity)] for sparsity in self.sparsity], label='Real Graph', marker='o')
            plt.xlabel('Sparsity')
            plt.ylabel(feature)
            plt.legend()
            plt.savefig('output/{}_sparsity.png'.format(feature))
            plt.clf()


if __name__ == '__main__':
    ngraphs = 1000
    fc_matrix = np.loadtxt('output/stim_0_fc_mean.csv', delimiter=',')
    x = RandomGraphAnalyzer(fc_matrix, n_rnd_graphs=ngraphs, analysis_type='threshold', features=['global_efficiency', 'local_efficiency', 'clustering_coefficient'])
    rgraphs = x.gen_random_graphs()
    print(len(rgraphs))
    print(len(rgraphs[0]))
    x.compute_features()
    x.plot_features()




# if __name__ == '__main__':
#     nnodes = 10
#     ngraphs = 1000

#     # plot the variation of global efficiency with sparsity from 0 to 1
#     sparsity = np.linspace(0, 1, 100)

#     global_efficiency_mean = []
#     global_efficiency_std = []
#     global_efficiency_fnirs = []

#     cluster_coef_mean = []
#     cluster_coef_std = []
#     cluster_coef_fnirs = []

#     fnirs_adj_matrix_stim_0 = np.loadtxt('output/stim_0_fc_mean.csv', delimiter=',')
#     # get min and max weights ignoring nan values
#     min_weight = np.nanmin(fnirs_adj_matrix_stim_0)
#     max_weight = np.nanmax(fnirs_adj_matrix_stim_0)
#     print('min weight: ', min_weight)
#     print('max weight: ', max_weight)

#     for s in sparsity:
#         ge_list = []
#         cc_list = []
#         for i in range(ngraphs):
#             adj_matrix = utils.RandomBinGraph(nnodes=nnodes, binarize_type='threshold', binarize_param=s).generate(min_weight=min_weight, max_weight=max_weight)
#             ge = bct.efficiency_bin(adj_matrix)
#             cc = np.mean(bct.clustering_coef_bu(adj_matrix))
#             ge_list.append(ge)
#             cc_list.append(cc)

#         ge_mean = np.mean(ge_list)
#         cc_mean = np.mean(cc_list)
#         ge_std = np.std(ge_list)
#         cc_std = np.std(cc_list)

#         global_efficiency_mean.append(ge_mean)
#         cluster_coef_mean.append(cc_mean)
#         global_efficiency_std.append(ge_std)
#         cluster_coef_std.append(cc_std)
#         global_efficiency_fnirs.append(bct.efficiency_bin(utils.BinarizeMatrix(fnirs_adj_matrix_stim_0, 'threshold', s).binarize()))
#         cluster_coef_fnirs.append(np.mean(bct.clustering_coef_bu(utils.BinarizeMatrix(fnirs_adj_matrix_stim_0, 'threshold', s).binarize())))

#     plt.errorbar(sparsity, global_efficiency_mean, yerr=global_efficiency_std, fmt='-o', capsize=3)
#     plt.plot(sparsity, global_efficiency_fnirs, '-o')
#     plt.xlabel('Sparsity')
#     plt.ylabel('Global Efficiency - Stim 0')
#     plt.legend(['fnirs data', 'random graph'])
#     plt.savefig('output/global_efficiency_stim_0.png')

#     plt.clf()

#     plt.errorbar(sparsity, cluster_coef_mean, yerr=cluster_coef_std, fmt='-o', capsize=3)
#     plt.plot(sparsity, cluster_coef_fnirs, '-o')
#     plt.xlabel('Sparsity')
#     plt.ylabel('Clustering Coefficient - Stim 0')
#     plt.legend(['fnirs data', 'random graph'])
#     plt.savefig('output/clustering_coef_stim_0.png')