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
        self.search_space = np.linspace(0, 1, 100)
        assert self.analysis_type in ['sparsity', 'threshold'], 'Invalid analysis_type. Must be either "sparsity" or "threshold".'
        self.features = features

    def gen_random_graphs(self):
        '''
        Generate random graphs using the given analysis type and number of graphs.
        '''
        if self.analysis_type == 'sparsity':
            # generate n_rnd_graphs random graphs for each sparsity value
            self.random_graphs = [[utils.RandomBinGraph(self.nnodes, binarize_type=self.analysis_type, binarize_param=sparsity).generate() for _ in range(self.n_rnd_graphs)] for sparsity in self.search_space]
        elif self.analysis_type == 'threshold':
            self.random_graphs = [[utils.RandomBinGraph(self.nnodes, binarize_type=self.analysis_type, binarize_param=threshold).generate(min_weight=np.nanmin(self.fc_matrix), max_weight=np.nanmax(self.fc_matrix)) for _ in range(self.n_rnd_graphs)] for threshold in self.search_space]
        return self.random_graphs

    def compute_features(self):
        '''
        Compute the features of the random graphs.
        '''
        self.random_graphs_features = {}
        self.fc_graph_features = {}
        for i, sparsity in enumerate(self.search_space):
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
            plt.errorbar(self.search_space, 
                        [np.mean(self.random_graphs_features['{}_{}'.format(feature, sparsity)]) for sparsity in self.search_space], 
                        yerr=[np.std(self.random_graphs_features['{}_{}'.format(feature, sparsity)]) for sparsity in self.search_space], 
                        label='Random Graphs', 
                        fmt='-o',
                        capsize=3)
            plt.plot(self.search_space, 
                    [self.fc_graph_features['{}_{}'.format(feature, sparsity)] for sparsity in self.search_space], 
                    label='Real Graph', 
                    marker='o')
            plt.xlabel(self.analysis_type)
            plt.ylabel(feature)
            plt.legend()
            plt.savefig('output/{}_{}.png'.format(feature, self.analysis_type), dpi=600)
            plt.clf()

    def compute_auc(self):
        '''
        Compute the AUC for a given feature.
        '''
        pass


if __name__ == '__main__':
    ngraphs = 1000
    fc_matrix = np.loadtxt('output/stim_0_fc_mean.csv', delimiter=',')
    x = RandomGraphAnalyzer(fc_matrix, n_rnd_graphs=ngraphs, analysis_type='sparsity', features=['global_efficiency', 'local_efficiency', 'clustering_coefficient'])
    rgraphs = x.gen_random_graphs()
    print(len(rgraphs))
    print(len(rgraphs[0]))
    x.compute_features()
    x.plot_features()