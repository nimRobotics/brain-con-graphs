import numpy as np
import bct as bct
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import utils

# set seed
np.random.seed(123)

class RandomGraphAnalyzer:
    def __init__(self, fc_matrices: list, n_rnd_graphs: int = 1000,
                analysis_type: str = 'sparsity',
                features: list = ['global_efficiency', 'local_efficiency', 'clustering_coefficient']):
        '''
        Parameters:
            fc_matrices (list): a list of FC matrices
            n_rnd_graphs (int): the number of random graphs to be generated
            analysis_type (str): the type of analysis to be performed.
                Options are 'sparsity' or 'threshold'
            features (list): the features to be analyzed.
                Options are 'global_efficiency', 'local_efficiency',
                'clustering_coefficient'
        '''
        self.fc_matrices = fc_matrices
        self.n_rnd_graphs = n_rnd_graphs
        self.analysis_type = analysis_type
        self.nnodes = fc_matrices[0].shape[0]
        self.nedges = int(self.nnodes * (self.nnodes - 1) / 2)
        self.search_space = np.linspace(0, 1, 100)
        assert self.analysis_type in ['sparsity', 'threshold'], \
            'Invalid analysis_type. Must be either "sparsity" or "threshold".'
        self.features = features

    def gen_random_graphs(self):
        '''
        Generate random graphs using the given analysis type and number of graphs.
        '''
        if self.analysis_type == 'sparsity':
            # generate n_rnd_graphs random graphs for each sparsity value
            self.random_graphs = [[utils.RandomBinGraph(
                self.nnodes,
                binarize_type=self.analysis_type,
                binarize_param=sparsity).generate()
                for _ in range(self.n_rnd_graphs)]
                for sparsity in self.search_space]
        elif self.analysis_type == 'threshold':
            self.random_graphs = [[utils.RandomBinGraph(
                self.nnodes,
                binarize_type=self.analysis_type,
                binarize_param=threshold).generate(
                    min_weight=np.nanmin(np.array(self.fc_matrices)),
                    max_weight=np.nanmax(np.array(self.fc_matrices)))
                for _ in range(self.n_rnd_graphs)]
                for threshold in self.search_space]
        return self.random_graphs

    def compute_features(self):
        '''
        Compute the features of the random graphs.
        '''
        self.random_graphs_features = {}
        self.fc_graph_features = {}
        for i, binparam in enumerate(self.search_space):
            for feature in self.features:
                if feature == 'global_efficiency':
                    self.random_graphs_features[f'global_efficiency_{binparam}'] = [
                        bct.efficiency_bin(graph) for graph in self.random_graphs[i]]
                    self.fc_graph_features[f'global_efficiency_{binparam}'] = [
                        bct.efficiency_bin(utils.BinarizeMatrix(fc_matrix, binarize_type=self.analysis_type,
                                                                binarize_param=binparam).binarize())
                        for fc_matrix in self.fc_matrices]
                elif feature == 'local_efficiency':
                    self.random_graphs_features[f'local_efficiency_{binparam}'] = [
                        np.mean(bct.efficiency_bin(graph, local=True)) for graph in self.random_graphs[i]]
                    self.fc_graph_features[f'local_efficiency_{binparam}'] = [
                        np.mean(bct.efficiency_bin(utils.BinarizeMatrix(fc_matrix, binarize_type=self.analysis_type,
                                                                binarize_param=binparam).binarize(), local=True))
                        for fc_matrix in self.fc_matrices]
                elif feature == 'clustering_coefficient':
                    self.random_graphs_features[f'clustering_coefficient_{binparam}'] = [
                        np.mean(bct.clustering_coef_bu(graph)) for graph in self.random_graphs[i]]
                    self.fc_graph_features[f'clustering_coefficient_{binparam}'] = [
                        np.mean(bct.clustering_coef_bu(utils.BinarizeMatrix(fc_matrix, binarize_type=self.analysis_type,
                                                                            binarize_param=binparam).binarize()))
                        for fc_matrix in self.fc_matrices]
                else:
                    raise ValueError('Invalid feature. Must be either "global_efficiency", "local_efficiency", or "clustering_coefficient".')
        return self.random_graphs_features, self.fc_graph_features

    def plot_features(self, filename: str=''):
        '''
        Plot the features of the random graphs and real graphs for each sparsity/threshold value.
        '''
        for i, feature in enumerate(self.features):
            plt.errorbar(self.search_space, 
                        [np.mean(self.random_graphs_features['{}_{}'.format(feature, sparsity)]) for sparsity in self.search_space], 
                        yerr=[np.std(self.random_graphs_features['{}_{}'.format(feature, sparsity)]) for sparsity in self.search_space], 
                        label='Random Graphs', 
                        fmt='-o',
                        capsize=3)
            plt.errorbar(self.search_space, 
                    [np.mean(self.fc_graph_features['{}_{}'.format(feature, sparsity)]) for sparsity in self.search_space],
                    yerr=[np.std(self.fc_graph_features['{}_{}'.format(feature, sparsity)]) for sparsity in self.search_space],
                    label='Real Graphs',
                    fmt='-o',
                    capsize=3)
            plt.xlabel(self.analysis_type)
            plt.ylabel(feature)
            plt.legend()
            plt.savefig('output/{}_{}_{}.png'.format(filename, feature, self.analysis_type), dpi=600)
            plt.clf()

    def compute_auc(self, min_binparam: float=0.1, max_binparam: float=0.4):
        '''
        compute the AUC for a given binparam range for fc graphs.
        params:
            min_binparam: minimum binparam value
            max_binparam: maximum binparam value
        returns:
            auc: dictionary of AUC values for each feature
        '''
        auc = {}
        mean = {}
        for feature in self.features:
            auc[feature] = {}
            mean[feature] = {}
            y_values = [self.fc_graph_features['{}_{}'.format(feature, binparam)] for binparam in self.search_space]
            y_values = np.array(y_values).T
            print('Yval shape',y_values.shape)
            x_value = np.array(self.search_space)
            print('xval_shape',x_value.shape)
            for i, y_value in enumerate(y_values):
                # print('\nA: ',(x_value >= min_binparam) & (x_value <= max_binparam))
                # print('\nB: ',y_value[(x_value >= min_binparam) & (x_value <= max_binparam)])
                # print('\nC: ',x_value[(x_value >= min_binparam) & (x_value <= max_binparam)])
                auc[feature][i] = np.trapz(y_value[(x_value >= min_binparam) & (x_value <= max_binparam)],
                                            x_value[(x_value >= min_binparam) & (x_value <= max_binparam)])
                mean[feature][i] = np.mean(y_value[(x_value >= min_binparam) & (x_value <= max_binparam)])
        return auc, mean

    def save_results(self, filename: str=''):
        '''
        save csv for each feature with sparsity as the column and graph as the row
        '''
        for feature in self.features:
            df = pd.DataFrame()
            for binparam in self.search_space:
                df[binparam] = self.fc_graph_features[f'{feature}_{binparam}']
            df.to_csv(f'output/{filename}_fc_graphs_{feature}_{self.analysis_type}.csv', index=False)


if __name__ == '__main__':
    ngraphs = 10
    stims = ['NHAHR', 'FHAHR', 'NHALR', 'FHALR']
    stim = 'FHAHR'

    df = pd.read_csv('./input/funcCon.csv', header=None)

    auc_data = {}
    mean_data = {}
    for stim in stims:
        print(f'Processing {stim} data...')
        data_df = df[df[3] == stim]
        print(data_df)
        data_df = data_df.iloc[:, 4:]    # remove the first 4 columns
        # print(data_df)
        fc_matrices = [utils.matrix_from_upper_triangle(data_df.iloc[i, :].values) for i in range(1, data_df.shape[0])]

        x = RandomGraphAnalyzer(fc_matrices, 
                                n_rnd_graphs=ngraphs, 
                                analysis_type='threshold', 
                                features=['global_efficiency', 'local_efficiency', 'clustering_coefficient'])
        rgraphs = x.gen_random_graphs()
        print(len(rgraphs))
        print(len(rgraphs[0]))
        x.compute_features()
        x.plot_features(filename='amelia_{}'.format(stim))
        x.save_results(filename=stim)
        auc, mean = x.compute_auc(min_binparam=0.1, max_binparam=0.4)
        auc_data[stim] = auc
        mean_data[stim] = mean


    # save auc to csv with stim as the column
    for feature in x.features:
        max_length = max(len(auc_data[stim][feature]) for stim in stims)
        df = pd.DataFrame()
        for stim in stims:
            # pad data with NAN to make all data the same length
            auc_values = auc_data[stim][feature].values()
            padded_values = list(auc_values) + [np.NAN] * (max_length - len(auc_values))
            df[stim] = padded_values
        df.to_csv(f'output/fc_graphs_auc_{feature}.csv', index=False)

    # save mean to csv with stim as the column
    for feature in x.features:
        max_length = max(len(mean_data[stim][feature]) for stim in stims)
        df = pd.DataFrame()
        for stim in stims:
            # pad data with NAN to make all data the same length
            mean_values = mean_data[stim][feature].values()
            padded_values = list(mean_values) + [np.NAN] * (max_length - len(mean_values))
            df[stim] = padded_values
        df.to_csv(f'output/fc_graphs_mean_{feature}.csv', index=False)






    