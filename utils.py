import numpy as np

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

def binarize_matrix(matrix, binarize_type='sparsity', binarize_param=0.5):
    '''
    Binarize a matrix using the given sparsity or threshold.
    Sparsity is defined as the number of non-zero edges divided by the total number of edges.
    Threshold is defined as the value above which all values are set to 1 and below which all values are set to 0.
    
    Parameters:
        matrix (numpy.ndarray): the adjacency matrix to be binarized
        binarize_type (str): the type of binarization to be used. Options are 'sparsity' or 'threshold'
        binarize_param (float): the sparsity or threshold value to be used for binarization
    
    Returns:
        numpy.ndarray: the binarized adjacency matrix
    '''
    if binarize_type == 'sparsity':
        assert 0 <= binarize_param <= 1, 'Sparsity must be between 0 and 1'
        return binarize_matrix_sparsity(matrix, binarize_param)
    elif binarize_type == 'threshold':
        assert 0 <= binarize_param <= 1, 'Threshold must be between 0 and 1'
        return binarize_matrix_threshold(matrix, binarize_param)
    else:
        raise ValueError('Invalid binarize_type. Must be either "sparsity" or "threshold".')

def binarize_matrix_threshold(matrix, threshold):
    '''
    Binarize a matrix using the given threshold.
    Threshold is defined as the value above which all values are set to 1 and below which all values are set to 0.

    Parameters:
        matrix (numpy.ndarray): the adjacency matrix to be binarized
        threshold (float): the threshold value to be used for binarization

    Returns:
        numpy.ndarray: the binarized adjacency matrix
    '''
    # create a new binary matrix with the same shape as the input matrix
    bin_matrix = np.zeros_like(matrix)

    # set all values above the threshold to 1
    bin_matrix[matrix > threshold] = 1

    return bin_matrix

def binarize_matrix_sparsity(matrix, sparsity):
    '''
    Binarize a matrix using the given sparsity.
    Sparsity is defined as the number of non-zero edges divided by the total number of edges.

    Parameters:
        matrix (numpy.ndarray): the adjacency matrix to be binarized
        sparsity (float): the sparsity value to be used for binarization

    Returns:
        numpy.ndarray: the binarized adjacency matrix
    '''
    # Ensure that the sparsity is between 0 and 1
    assert 0 <= sparsity <= 1, 'Sparsity must be between 0 and 1'

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

def gen_random_weighted_graph(nnodes, min_weight, max_weight):
    '''
    Generate a random weighted graph with given nodes and edge weights
    nnodes: number of nodes
    min_weight: minimum weight
    max_weight: maximum weight
    return: adjacency matrix
    '''
    # create an empty adjacency matrix with nnodes x nnodes dimensions
    adj_matrix = np.zeros((nnodes, nnodes))

    # generate a list of all possible edges
    all_edges = [(i, j) for i in range(nnodes) for j in range(i + 1, nnodes)]

    # choose nedges random edges from the list of all possible edges
    for edge in all_edges:
        i, j = edge
        adj_matrix[i][j] = np.random.uniform(min_weight, max_weight)
        adj_matrix[j][i] = adj_matrix[i][j]

    return adj_matrix
