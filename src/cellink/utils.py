import numpy as np


def drop_low_variability_columns(arr_list: list, tol=1e-8):
    """
    Drop columns for which the standard deviation is below a specified tolerance in any one of the arrays in arr_list.
    This helps in removing columns with zero or near-zero variability across multiple datasets.

    Parameters
    ----------
    arr_lst : list of np.array
        List of arrays where each array must have the same number of columns.
    tol : float, default=1e-8
        Tolerance threshold below which the standard deviation is considered zero.

    Returns
    -------
    list of np.array
        List of arrays with columns of zero variability removed.
    """
    if not arr_list:
        return []

    # Assert that all arrays have the same number of columns
    n_cols = arr_list[0].shape[1]
    assert all(arr.shape[1] == n_cols for arr in arr_list), "All arrays must have the same number of columns."

    # calculate the stf of each column in each dataset
    remove_cols = []
    column_std = [np.std(arr, axis = 0) for arr in arr_list]
    for data in column_std:
        for col in range(len(data)):
            if data[col] < tol:
                if col not in remove_cols:
                    remove_cols.append(col)
    keep_cols = [i for i in range(len(column_std[0])) if i not in remove_cols]
    # Filter and keep columns in each array
    return [arr[:, keep_cols] for arr in arr_list]

def graph_smoothing(arr, edges, wt):
    """
    Adjust node features towards the weighted average of their neighbors' features.
    
    Parameters
    ----------
    arr : np.array
        Data matrix with shape (n_samples, n_features), where each row corresponds to a node.
    edges : list of lists
        Contains two or three elements:
            edges[0] : list or array of source node indices
            edges[1] : list or array of target node indices
            edges[2] : (optional) list or array of weights corresponding to each edge
    wt : float
        Weight factor for combining original node features with the smoothed features.

    Returns
    -------
    np.array
        Smoothed data matrix of the same shape as 'arr'.
    """
    n_samples, n_features = arr.shape
    adj_list = [[] for _ in range(n_samples)]
    weight_list = [[] for _ in range(n_samples)]

    # Process edges
    for i in range(len(edges[0])):
        src = edges[0][i]
        tgt = edges[1][i]
        adj_list[src].append(tgt)
        
        # Handle weights if provided, otherwise use 1 as default weight
        if len(edges) > 2:
            weight_list[src].append(edges[2][i])
        else:
            weight_list[src].append(1)

    # Compute weighted averages
    centroids = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        if adj_list[i]:  # Check if there are any neighbors
            neighborhood = arr[adj_list[i], :]
            weights = weight_list[i]
            centroids[i] = np.average(neighborhood, axis=0, weights=weights)
        else:
            centroids[i] = arr[i]  # No neighbors, retain original features

    # Combine original node features with their neighborhood averages
    return wt * arr + (1 - wt) * centroids

def cdist_correlation(arr1, arr2):
    """Calculate pair-wise 1 - Pearson correlation between X and Y.

    Parameters
    ----------
    arr1: np.array of shape (n_samples1, n_features)
        First dataset.
    arr2: np.array of shape (n_samples2, n_features)
        Second dataset.

    Returns
    -------
    array-like of shape (n_samples1, n_samples2)
        The (i, j)-th entry is 1 - Pearson correlation between i-th row of arr1 and j-th row of arr2.
    """
    n, p = arr1.shape
    m, p2 = arr2.shape
    assert p2 == p

    arr1 = (arr1.T - np.mean(arr1, axis=1)).T
    arr2 = (arr2.T - np.mean(arr2, axis=1)).T

    arr1 = (arr1.T / np.sqrt(1e-6 + np.sum(arr1 ** 2, axis=1))).T
    arr2 = (arr2.T / np.sqrt(1e-6 + np.sum(arr2 ** 2, axis=1))).T

    return 1 -  arr1 @ arr2.T



    


