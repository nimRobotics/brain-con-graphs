import numpy as np
import math
np.random.seed(3)

class BinarizeMatrix:
    '''
    Binarize a matrix using the given sparsity or threshold.
    Sparsity is defined as the number of non-zero edges divided by the total number of edges.
    Threshold is defined as the value above which all values are set to 1 and below which all values are set to 0.

    '''
    def __init__(self, matrix: np.ndarray, binarize_type: str = 'sparsity', binarize_param: float = 0.5) -> None:
        '''
        Parameters:
            matrix (numpy.ndarray): the adjacency matrix to be binarized
            binarize_type (str): the type of binarization to be performed. Options are 'sparsity' or 'threshold'
            binarize_param (float): the parameter to be used for binarization. If binarize_type is 'sparsity', this is the sparsity. If binarize_type is 'threshold', this is the threshold.
        '''
        self.binarize_type = binarize_type
        self.binarize_param = binarize_param
        self.matrix = matrix

    def binarize(self) -> np.ndarray:
        '''
        Binarize the matrix using the given binarize_type and binarize_param.

        Returns:
            numpy.ndarray: the binarized adjacency matrix
        '''

        if self.binarize_type == 'sparsity':
            assert 0 <= self.binarize_param <= 1, 'Sparsity must be between 0 and 1'
            return self.binarize_sparsity(self.matrix)
        elif self.binarize_type == 'threshold':
            assert 0 <= self.binarize_param <= 1, 'Threshold must be between 0 and 1'
            return self.binarize_threshold(self.matrix)
        else:
            raise ValueError('Invalid binarize_type. Must be either "sparsity" or "threshold".')

    def binarize_threshold(self, matrix : np.ndarray) -> np.ndarray:
        '''
        Binarize a matrix using the given threshold.
        Threshold is defined as the value above which all values are set to 1 and below which all values are set to 0.

        Parameters:
            matrix (numpy.ndarray): the adjacency matrix to be binarized

        Returns:
            numpy.ndarray: the binarized adjacency matrix
        '''
        # create a new binary matrix with the same shape as the input matrix
        bin_matrix = np.zeros_like(matrix)

        # zscores of connectivity matrix
        zscores = np.arctanh(matrix) # Fisher's z transform

        # threshold the connectivity matrix based on zscores and the given threshold
        bin_matrix[abs(zscores) > self.binarize_param] = 1

        return bin_matrix

    def binarize_sparsity(self, matrix: np.ndarray) -> np.ndarray:
        '''
        Binarize a matrix using the given sparsity.
        Sparsity is defined as the number of non-zero edges divided by the total number of edges.

        Parameters:
            matrix (numpy.ndarray): the adjacency matrix to be binarized

        Returns:
            numpy.ndarray: the binarized adjacency matrix
        '''
        # Ensure that the sparsity is between 0 and 1
        assert 0 <= self.binarize_param <= 1, 'Sparsity must be between 0 and 1'

        # set diagonal to 0
        np.fill_diagonal(matrix, 0)

        # Calculate the maximum number of edges in the matrix
        max_edges = matrix.shape[0] * (matrix.shape[0] - 1) / 2
        # print('Max edges: ', max_edges)

        # Calculate the number of non-zero edges based on the desired sparsity
        n_nonzero_edges = int(self.binarize_param * max_edges)
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

class RandomBinGraph:
    '''
    Generate a random binary graph with given nodes and edges/sparsity or threshold
    '''
    def __init__(self, nnodes, binarize_type='sparsity', binarize_param=0.5):
        '''
        Generate a random binary graph with given nodes and edges/sparsity

        Parameters:
            nnodes (int): number of nodes
            binarize_type (str): the type of binarization to be performed. Options are 'sparsity' or 'threshold'
            binarize_param (float): the parameter to be used for binarization. If binarize_type is 'sparsity', this is the sparsity. If binarize_type is 'threshold', this is the threshold.
        '''
        self.nnodes = nnodes
        self.binarize_type = binarize_type  
        self.binarize_param = binarize_param

    def generate(self, min_weight: float = None, max_weight: float = None) -> np.ndarray:
        '''
        Generate a random binary graph with given nodes and edges/sparsity

        Parameters:
            min_weight (float): the minimum weight of the edges
            max_weight (float): the maximum weight of the edges

        Returns:
            numpy.ndarray: the adjacency matrix
        '''
        
        if self.binarize_type == 'sparsity':
            assert 0 <= self.binarize_param <= 1, 'Sparsity must be between 0 and 1'
            return self.generate_sparsity()
        elif self.binarize_type == 'threshold':
            assert 0 <= self.binarize_param <= 1, 'Threshold must be between 0 and 1'
            rndadjmatrix = self.generate_weighted_graph(min_weight, max_weight)
            return BinarizeMatrix(rndadjmatrix, self.binarize_type, self.binarize_param).binarize()
        else:
            raise ValueError('Invalid binarize_type. Must be either "sparsity" or "threshold".')

    def generate_sparsity(self):
        '''
        Generate a random binary graph with given nodes and sparsity

        Returns:
            numpy.ndarray: the adjacency matrix
        '''
        # Ensure that the sparsity is between 0 and 1
        assert 0 <= self.binarize_param <= 1, 'Sparsity must be between 0 and 1'
        nedges = int(self.binarize_param * self.nnodes * (self.nnodes - 1) / 2)

        # Create an empty adjacency matrix with nnodes x nnodes dimensions
        adj_matrix = np.zeros((self.nnodes, self.nnodes))

        # Generate a list of all possible edges
        all_edges = [(i, j) for i in range(self.nnodes) for j in range(i + 1, self.nnodes)]

        # Choose nedges random edges from the list of all possible edges
        random_edges = np.random.choice(len(all_edges), nedges, replace=False)

        # Add the chosen edges to the adjacency matrix
        for edge in random_edges:
            i, j = all_edges[edge]
            adj_matrix[i][j] = 1
            adj_matrix[j][i] = 1

        return adj_matrix

    def generate_weighted_graph(self, min=-1, max=1):
        '''
        Generate a random weighted graph with given nodes and edges/sparsity

        Parameters:
            min (float): the minimum value for the weights
            max (float): the maximum value for the weights

        Returns:
            numpy.ndarray: the adjacency matrix
        '''
        # Create a random matrix with the given parameters
        matrix = np.random.uniform(min, max, size=(self.nnodes, self.nnodes))

        # Set the diagonal to 0
        np.fill_diagonal(matrix, 0)

        # Ensure symmetry by copying the upper triangle to the lower triangle
        matrix = np.triu(matrix) + np.triu(matrix, k=1).T

        return matrix

def matrix_from_upper_triangle(data):
    """Convert flattened upper triangle to matrix"""
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    n = int(math.sqrt(2 * data.shape[0] + 0.25) + 0.5)
    # print('n = ', n)
    # Create an empty matrix of zeros
    mat = np.zeros((n, n))
    # Fill the upper triangle of the matrix using the flattened array
    for i in range(n):
        for j in range(i+1, n):
            index = int(i*n + j - (i*(i+1))/2 - i - 1)
            mat[i][j] = data[index]
    # Fill the lower triangle of the matrix using the upper triangle
    mat = mat + mat.T - np.diag(mat.diagonal())
    return mat  
