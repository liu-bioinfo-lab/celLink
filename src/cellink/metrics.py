import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
from scipy.spatial import distance
import networkx as nx

def cell_type_matching_accuracy(m1_source_ct, m1_predict_ct, m2_source_ct, m2_predict_ct):
    """
    Calculate the cell-type prediction accuracy of cell-cell alignment.

    Params
    ----------
    m1_source_ct: list, the original cell type list of modality 1.
    m1_predict_ct: list, the predicted cell type list of modality 1.
    m2_source_ct: list, the original cell type list of modality 2.
    m2_predict_ct: list, the predicted cell type list of modality 2.

    Return
    --------
    acc: float, the cell-type prediction accuracy
    """
    n1 = len(m1_source_ct)
    n2 = len(m2_source_ct)
    r1 = sum(m1_source_ct == m1_predict_ct)
    r2 = sum(m2_source_ct == m2_predict_ct)
    acc = (r1 + r2) / (n1 + n2)
    acc = round(acc, 4)
    return acc

def average_sihouette_width(embedding, cell_type_label):
    """
    Calculate the average sihouette score of integration performance.

    Params
    ----------
    embedding: np.array, the 2d array of embedding, shape: (n_1 + n2, 2).
    cell_type_label: np.array, the 1d array of the cell type, shape: (n_x + n_y,).

    Return
    --------
    sihou_score: float, the average sihouette score
    """
    sihouette_avg = silhouette_score(embedding, cell_type_label) 
    sihouette_avg = round(sihouette_avg, 4)
    return sihouette_avg

def feature_imputation_accuracy_corr(m1_feature, m2_aligned_feature1):
    """
    Calculate the feature imputation accuracy of the aligned feature profile.

    Params
    ----------
    m1_feature: np.array, the 2d array of modality 1 feature profile, shape: (n, g).
    m2_aligned_feature1: np.array, the 2d array of the aligned modality-1 feature profile of modality 2, shape: (n, g).

    Return
    --------
    impute_acc: float, the feature imputation accuracy
    """
    assert m1_feature.shape == m2_aligned_feature1.shape
    n_samples = m1_feature.shape[0]
    corr_vec = np.zeros(n_samples)
    for i in range(n_samples):
        corr = np.corrcoef(m1_feature[i, :], m2_aligned_feature1[i, :])[0, 1]
        corr_vec[i] = round(corr, 4)
    impute_acc = np.mean(corr_vec)
    return impute_acc

def feature_imputation_rmse(m1_feature, m2_aligned_feature1):
    """
    Calculate the RMSE of the aligned feature profile.

    Params
    ----------
    m1_feature: np.array, the 2D array of modality 1 feature profile, shape: (n, g).
    m2_aligned_feature1: np.array, the 2D array of the aligned modality-1 feature profile of modality 2, shape: (n, g).

    Return
    --------
    impute_rmse: float, the RMSE of the feature imputation
    """
    assert m1_feature.shape == m2_aligned_feature1.shape
    error = m1_feature - m2_aligned_feature1
    squared_error = np.square(error)
    mean_squared_error = np.mean(squared_error)
    impute_rmse = np.sqrt(mean_squared_error)
    return impute_rmse

def uniFOSCTTM(m1_embedding, m2_embedding, true_matches_for_m2):
    """
    Calculate the proportion of samples closer than the true paired sample

    Params
    ----------
    m1_embedding: np.array, the embedding of modality 1, shape: (n, d).
    m2_embedding: np.array, the embedding of modality 2, shape: (n, d).
    true_matches_for_m1: 1d array, the indices of the matched cells for modality 1 from modality 2 

    Return
    -------
    foscttm: float, the foscttm score
    """
    distance_matrix = distance.cdist(m2_embedding, m1_embedding, metric = 'euclidean')
    n = len(true_matches_for_m2)
    vec = np.zeros(n)
    for idx, true_match in enumerate(true_matches_for_m2):
        true_distance = distance_matrix[idx, true_match]
        # Count how many cells in modality 1 are closer to cell idx in modality 2 than the true match
        closer_samples = np.sum(distance_matrix[idx, :] < true_distance)
        vec[idx] = closer_samples / distance_matrix.shape[1]
        prop = np.mean(vec)

    return round(prop, 4)


def calculate_graph_connectivity(data, labels, k=15):
    """
    Calculate the Graph Connectivity for each cell type in the dataset.

    Args:
    data (np.ndarray): The dataset where rows are samples and columns are features.
    labels (np.array): The cell type labels for each sample in the dataset.
    k (int): Number of nearest neighbors to consider for each cell.

    Returns:
    float: The Graph Connectivity score.
    """
    kng = kneighbors_graph(data, n_neighbors=k, mode='connectivity', include_self=False)
    G = nx.from_scipy_sparse_array(kng)

    unique_labels = np.unique(labels)
    M = len(unique_labels)
    sum_lcc_ratio = 0

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        subG = G.subgraph(indices)
        largest_cc = max(nx.connected_components(subG), key=len)
        LCC_j = len(largest_cc)
        N_j = len(indices)
        lcc_ratio = LCC_j / N_j
        sum_lcc_ratio += lcc_ratio

    return sum_lcc_ratio / M
