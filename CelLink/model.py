import numpy as np
import ot 
import matplotlib.pyplot as plt
from utils import drop_low_variability_columns
from utils import graph_smoothing
from utils import cdist_correlation
import anndata as ad
import scanpy as sc
import scipy
from scipy.optimize import linear_sum_assignment
from matplotlib.lines import Line2D
import umap
import warnings

class Cellink:
     
    def __init__(self, full_ann1, full_ann2, shared_ann1, shared_ann2, partition = True):
         """
         CelLink Input pre-processing 
         -------------------------------
         full_ann1: Anndata, the anndata of modality X with full feature
         full_ann2: Anndata, the anndata of modality X with full feature
         shared_ann1: Anndata, the anndata of modality X with shared feature
         shared_ann2: Anndata, the anndata of modality X with shared feature
         partition: bool, whether use down-samping function to split data into partitions
         """ 
         warnings.filterwarnings("ignore")
         if scipy.sparse.issparse(full_ann1.X):
             full_ann1.X = full_ann1.X.toarray()
         if scipy.sparse.issparse(full_ann2.X):
             full_ann2.X = full_ann2.X.toarray()
         if scipy.sparse.issparse(shared_ann1.X):
             shared_ann1.X = shared_ann1.X.toarray()
         if scipy.sparse.issparse(shared_ann2.X):
             shared_ann1.X = shared_ann2.X.toarray()
    
         self.full_ann1 = full_ann1
         self.full_ann2 = full_ann2
         self.shared_ann1 = shared_ann1
         self.shared_ann2 = shared_ann2
         self.partition = partition 

         # partition cells
         self.partition1 = None
         self.partition2 = None
         self.partition_d1_to_d2 = None

         # graph nodes, edges, weights
         self.edge1 = None
         self.edge2 = None

         # cell_correspondence and imputed results save
         self.cell_correspondence_partition1 = []
         self.cell_correspondence_partition2 = [] 
         self.feature_imputation_partition1 = []
         self.feature_imputation_partition2 = []
         self.arr1_unmatched_cell_id = []
         self.arr2_unmatched_cell_id = []
         self.arr1_wrong_ct = []
         self.arr2_wrong_ct = []
         
         if 'cell_type' in self.full_ann1.obs.columns and 'cell_type' in self.full_ann2.obs.columns:
            print('Cell annotations are provided. Perform Iteratively OT!')
         else:
            print('Cell annotations are not provided. Perform one-time reciprocal UOT!')

     # split data into partitions
    @staticmethod
    def _split(arr, n_batches, seed):
        rng = np.random.RandomState(seed)
        indices = rng.permutation(arr.shape[0])
        batch_size = len(indices) // n_batches
        batches = []
        
        for i in range(n_batches):
            start_index = i * batch_size
            if i == n_batches - 1:
                end_index = len(indices)  # Include all remaining elements in the last batch
            else:
                end_index = start_index + batch_size
            batches.append(indices[start_index:end_index].tolist())

        return batches


    def split_into_batches(self, arr_list, maximum_batch_size=5000, 
                seed = 100, verbose=True):
        '''
        PS: put the modality that has larger cell population into the second position of the arr list.
        '''

        if self.partition:
            n1 = arr_list[0].shape[0]
            n2 = arr_list[1].shape[0]

            if n1 >= maximum_batch_size:
                arr1_batch_size = maximum_batch_size
            else:
                arr1_batch_size = n1

            if n2 >= maximum_batch_size:
                arr2_batch_size = maximum_batch_size 
            else:
                arr2_batch_size = n2
                

            n1_batches = n1 // arr1_batch_size
            n2_batches = n2 // arr2_batch_size

            batch1_sample_indice = Cellink._split(arr_list[0], n1_batches, seed = seed)
            batch2_sample_indice = Cellink._split(arr_list[1], n2_batches, seed = seed)

            batch_d1_to_d2 = []

            b1 = 0
            b2 = 0

            for i in range(max(n1_batches, n2_batches)):
                    batch_d1_to_d2.append((b1, b2))
                    b1 = (b1 + 1) % n1_batches
                    b2 = (b2 + 1) % n2_batches

            if verbose:
                print(('The first modality is split into {} batches, '
                        'and max batch size is {}.').format(
                        n1_batches, len(batch1_sample_indice[-1])
                    ), flush=True)
                
                print(('The second modality is split into {} batches, '
                        'and max batch size is {}.').format(
                        n2_batches, len(batch2_sample_indice[-1])
                    ), flush=True)
                print('Batch to batch correspondence is:\n  {}.'.format(
                        [str(i) + '<->' + str(j) for i, j in batch_d1_to_d2]
                    ), flush=True)
            
            self.partition1 = batch1_sample_indice
            self.partition2 = batch2_sample_indice
            self.partition_d1_to_d2 = batch_d1_to_d2
        
        else:
            batch1_sample_indice = [list(arr_list[0].shape[0])]
            batch2_sample_indice = [list(arr_list[1].shape[0])]
            batch_d1_to_d2 = [(0, 0)]

            n1_batches = 1
            n2_batches = 1

            if verbose:
                print(('The first modality is split into {} batches, '
                        'and maximum batch size is {}.').format(
                        n1_batches, len(batch1_sample_indice[-1])
                    ), flush=True)
                
                print(('The second modality is split into {} batches, '
                        'and maximum batch size is {}.').format(
                        n2_batches, len(batch2_sample_indice[-1])
                    ), flush=True)
                print('Batch to batch correspondence is:\n  {}.'.format(
                        [str(i) + '<->' + str(j) for i, j in batch_d1_to_d2]
                    ), flush=True)

            self.partition1 = batch1_sample_indice
            self.partition2 = batch2_sample_indice
            self.partition_d1_to_d2 = batch_d1_to_d2

    # 
    def construct_graph(
            self, arr1_full_batch, arr2_full_batch,
            n_neighbors=15, metric='correlation', verbose=False
    ):
        """
        Compute k-nearest neighbors of data and return the UMAP graph.

        Parameters
        ----------
        arr1_full_batch: np.array of shape (n_samples1, n_features1)
            Data1 matrix full feature each batch
        arr2_full_batch: np.array of shape (n_samples2, n_features2)
            Data2 matrix full feature each batch
        n_neighbors: int
            Number of neighbors desired
        metric: string, default='correlation'
            Metric used when constructing the initial knn graph
        verbose: bool, default=True
            Whether to print progress

        Returns
        -------
        None, but generate two objects self.edge1 and self.edge 2, which contains:
            rows, cols, vals: list
            Each edge is rows[i], cols[i], and its weight is vals[i]
        """

        arr1 = drop_low_variability_columns(arr_list=[arr1_full_batch, arr1_full_batch])[0]

        if verbose:
            print("Constructing the graph for ann1", flush=True)
        # use scanpy functions to do the graph construction
        arr1 = arr1.astype(np.float32)
        adata = ad.AnnData(arr1)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=None, use_rep='X', metric=metric)
        rows1, cols1 = adata.obsp['connectivities'].nonzero()
        vals1 = adata.obsp['connectivities'][(rows1, cols1)].A1


        arr2 = drop_low_variability_columns(arr_list=[arr2_full_batch, arr2_full_batch])[0]


        if verbose:
            print("Constructing the graph for ann2", flush=True)

        arr2 = arr2.astype(np.float32)
        adata2 = ad.AnnData(arr2)
        sc.pp.neighbors(adata2, n_neighbors=n_neighbors, n_pcs=None, use_rep='X', metric=metric)
        rows2, cols2 = adata2.obsp['connectivities'].nonzero()
        vals2 = adata2.obsp['connectivities'][(rows2, cols2)].A1


        if verbose:
            print("Graphs are constructed!", flush=True)

        self.edges1 = (rows1, cols1, vals1)
        self.edges2 = (rows2, cols2, vals2)

    
    def alignment(self, wt1, wt2, n_neighbors = 10, lambd = 5e-3, matching_ratio = 1, reg = 5e-3, reg_m1 = (20, 0.1), reg_m2 = (0.1, 20), 
                  numItermax = 1000, iterative = True, sparse = False, BOT = True, verbose = True):
        """
        Perform cell-cell alignment (stage I and stage II).

        Parameters
        ----------
        wt1: float, default=0.3
            The shrinkage weight to put on the raw data for arr1.
        wt2: float, default=0.3
            The shrinkage weight to put on the raw data for arr2.
        n_neighbors: int
            Number of neighbors desired
        verbose: bool, default=True
            Whether to print the progress.
        matching_ratio: int
            The mass ratio of cells in arr1 compared with arr2
        lambd: float, default = 1e-1 
            The penalty coefficient of entropy regularization of BOT 
        reg: float, default = 0.05 
            The penalty coefficient of entropy regularization of UOT
        reg_m: tuple
            Each element in the tuple is the penalty coefficient of the KL divergence of the transported mass from one modality and the initial mass from another
        numItermax: int
            Max number of iterations of sinkhorn algorithms
        iterative: bool
            If True, utilize iterative OT, otherwise only balanced OT. 
        
        Returns
        -------
        None
        """
        for batch_id, (b1, b2) in enumerate(self.partition_d1_to_d2):
            if verbose:
                print(
                    'Now at batch {}<->{}...'.format(b1, b2),
                    flush=True
                )

            self.arr1_shared_batch = self.shared_ann1[self.partition1[b1],]
            self.arr2_shared_batch = self.shared_ann2[self.partition2[b2],]
            arr1_full_batch = self.full_ann1[self.partition1[b1],]
            arr2_full_batch = self.full_ann2[self.partition2[b2],]
            
            if arr1_full_batch.shape[1] > 5000:
                sc.pp.highly_variable_genes(arr1_full_batch, n_top_genes=5000)
                arr1_full_batch_f = arr1_full_batch[:, arr1_full_batch.var.highly_variable].copy()
            else:
                arr1_full_batch_f = arr1_full_batch
                
            if arr2_full_batch.shape[1] > 5000:
                sc.pp.highly_variable_genes(arr2_full_batch, n_top_genes=5000)
                arr2_full_batch_f = arr2_full_batch[:, arr2_full_batch.var.highly_variable].copy()
            else:
                arr2_full_batch_f = arr2_full_batch
            
            self.construct_graph(arr1_full_batch = arr1_full_batch_f.X, arr2_full_batch = arr2_full_batch_f.X, n_neighbors = n_neighbors)

            arr1 = self.arr1_shared_batch.X
            arr2 = self.arr2_shared_batch.X

            arr1, arr2 = drop_low_variability_columns(arr_list=[arr1, arr2])

            arr1 = graph_smoothing(arr=arr1, edges=self.edges1, wt=wt1)
            arr2 = graph_smoothing(arr=arr2, edges=self.edges2, wt=wt2)
            
            
            if 'cell_type' in self.arr1_shared_batch.obs.columns and 'cell_type' in self.arr2_shared_batch.obs.columns:
               #print('Cell type annotations are provided. Iteratively OT starts!')
                # stage I : balanced OT
                if (sparse == True & iterative == True) or (not BOT):
                    m11 = {}
                    m21 = {}
                    m12 = np.arange(arr1.shape[0])
                    m22 = np.arange(arr2.shape[0])
                else:
                    m11, m12, wrong_ct_arr1 = self.balanced_ot(arr1, arr2, direction = 1, lambd = lambd, matching_ratio = matching_ratio, numItermax = numItermax, sparse = sparse)
                    m21, m22, wrong_ct_arr2 = self.balanced_ot(arr1, arr2, direction = 2, lambd = lambd, matching_ratio = matching_ratio, numItermax = numItermax, sparse = sparse)
                    
                if(iterative):
                    # stage II: unbalanced OT
                    if len(m12) > 0:
                        print(f"{len(m12)} cells from Modality X are unmatched in Phase I and are realigned in Phase II.")
                        match_result_arr1, unmatched_cell_id_arr1, wrong_ct_all_arr1 = self.iterative_unbalanced_ot(arr1, arr2, bot_match_result = m11, ori_cell_id = m12, reg = reg, 
                                                                                                                reg_m = reg_m1, numItermax = numItermax, direction = 1, sparse = sparse)
                    else:
                        print(f"All cells from Modality X are matched in Phase I.")
                        match_result_arr1 = m11 
                        unmatched_cell_id_arr1 = m12
                        wrong_ct_all_arr1 = wrong_ct_arr1
                    
                    if len(m22) > 0:
                        print(f"{len(m22)} cells from Modality Y are unmatched in Phase I and are realigned in Phase II.")
                        match_result_arr2, unmatched_cell_id_arr2, wrong_ct_all_arr2  = self.iterative_unbalanced_ot(arr1, arr2, bot_match_result = m21, ori_cell_id = m22, reg = reg, 
                                                                                                                reg_m = reg_m2, numItermax = numItermax, direction = 2, sparse = sparse)
                    else:
                        print(f"All cells from Modality Y are matched in Phase I.")
                        match_result_arr2 = m21 
                        unmatched_cell_id_arr2 = m22
                        wrong_ct_all_arr2 = wrong_ct_arr2

                    # stage III: domain transfer and visualization
                    if len(match_result_arr1) == 0 | len(match_result_arr2) == 0:
                        raise ValueError("No cell successfully aligned in modalities, please check the data.")

                    matched1_transport_map = np.vstack(list(match_result_arr1.values()))
                    arr1_impute = matched1_transport_map @ arr2_full_batch.X
                    weights1 = matched1_transport_map.sum(axis = 1)[:, np.newaxis]
                    arr1_impute = arr1_impute / weights1

                    matched2_transport_map = np.vstack(list(match_result_arr2.values()))
                    arr2_impute = matched2_transport_map @ arr1_full_batch.X
                    weights2 = matched2_transport_map.sum(axis = 1)[:, np.newaxis]
                    arr2_impute = arr2_impute / weights2

                    self.cell_correspondence_partition1.append(match_result_arr1)
                    self.cell_correspondence_partition2.append(match_result_arr2)

                    self.feature_imputation_partition1.append(arr1_impute)
                    self.feature_imputation_partition2.append(arr2_impute)

                    self.arr1_unmatched_cell_id.append(unmatched_cell_id_arr1)
                    self.arr2_unmatched_cell_id.append(unmatched_cell_id_arr2)

                    self.arr1_wrong_ct.append(wrong_ct_all_arr1)
                    self.arr2_wrong_ct.append(wrong_ct_all_arr2)
                
                else:
                    if len(match_result_arr1) == 0 | len(match_result_arr2) == 0:
                        raise ValueError("No cell successfully aligned in modalities, please check the data.")
                    
                    matched1_transport_map = np.vstack(list(m11.values()))
                    arr1_impute = matched1_transport_map @ arr2_full_batch.X
                    weights1 = matched1_transport_map.sum(axis = 1)[:, np.newaxis]
                    arr1_impute = arr1_impute / weights1

                    matched2_transport_map = np.vstack(list(m21.values()))
                    arr2_impute = matched2_transport_map @ arr1_full_batch.X
                    weights2 = matched2_transport_map.sum(axis = 1)[:, np.newaxis]
                    arr2_impute = arr2_impute / weights2

                    self.cell_correspondence_partition1.append(m11)
                    self.cell_correspondence_partition2.append(m21)

                    self.feature_imputation_partition1.append(arr1_impute)
                    self.feature_imputation_partition2.append(arr2_impute)

                    self.arr1_unmatched_cell_id.append(m12)
                    self.arr2_unmatched_cell_id.append(m22)

                    self.arr1_wrong_ct.append(wrong_ct_arr1)
                    self.arr2_wrong_ct.append(wrong_ct_arr2)
                    
            else:
                #print('Cell type annotations are NOT provided. One-time reciprocal UOT starts!')
                dist = cdist_correlation(arr1, arr2)
                a = np.ones(shape = (arr1.shape[0],)) 
                b = np.ones(shape = (arr2.shape[0],))
                #adaptive_weights = np.linspace(1.1, 1.5, 5)
                
                matched1_transport_map = ot.unbalanced.sinkhorn_unbalanced(a = a, b = b, M = dist, reg = reg, reg_m = reg_m1, numItermax = numItermax, stopThr=1e-15)
                for i in range(matched1_transport_map.shape[0]):
                    weights = matched1_transport_map[i, :]
                    matched1_transport_map[i, :] = weights
                
                matched2_transport_map = ot.unbalanced.sinkhorn_unbalanced(a = a, b = b, M = dist, reg = reg, reg_m = reg_m2, numItermax = numItermax, stopThr=1e-15)
                for i in range(matched2_transport_map.shape[1]):
                    weights = matched2_transport_map[:, i]
                    matched2_transport_map[:, i] = weights
                
                arr1_impute = matched1_transport_map @ arr2_full_batch.X
                weights1 = matched1_transport_map.sum(axis = 1)[:, np.newaxis]
                arr1_impute = arr1_impute / weights1
                
                arr2_impute = matched2_transport_map.T @ arr1_full_batch.X
                weights2 = matched2_transport_map.sum(axis = 0)[:, np.newaxis]
                arr2_impute = arr2_impute / weights2
                
                self.cell_correspondence_partition1.append(matched1_transport_map)
                self.cell_correspondence_partition2.append(matched2_transport_map.T)

                self.feature_imputation_partition1.append(arr1_impute)
                self.feature_imputation_partition2.append(arr2_impute)
            


    def balanced_ot(self, arr1, arr2, lambd = 5e-3, matching_ratio = 1, numItermax = 1000, direction = 1, sparse = False):
        """
        stage I of celLink: perform balanced optimal transport, filter out the matched cells and retain the unmatched cells.

        Parameters
        -----------------
        arr1: np.array
            The shared-feature arr1
        arr2: np.array
            The shared-feature arr2
        matching_ratio: int
            The mass ratio of cells in arr1 compared with arr2
        lambd: float, default = 1e-1 
            The penalty coefficient of entropy regularization of BOT
        numItermax: int
            Max number of iterations of sinkhorn algorithms
        direction: int, default = 1
            The direction to align between modalities. If direction is 1, align arr2-feature to arr1. If direction is 2, align arr1-feature to arr2.
            For linear sum assignment case (sparse = True and iterative = False), direction value does not matter. 

        Returns
        -----------------
        match_result: dictionary
            keys are the cell-cell correspondence matrix to the matched cells and values are the correspondence vectors.
        ori_cell_id: list,
            Cell ids for unmatched cells
        """
        dist = cdist_correlation(arr1, arr2)

        if sparse == True:
            row_ind, col_ind = linear_sum_assignment(dist)
            ot_sink = np.zeros_like(dist)
            ot_sink[row_ind, col_ind] = 1
        else:
            a = np.ones(shape = (arr1.shape[0],)) * matching_ratio
            b = np.ones(shape = (arr2.shape[0],)) 
            ot_sink = ot.sinkhorn(a, b, dist, lambd, numItermax = numItermax)

        source_ct = []
        predicted_ct = []
        match_result = {}

        if direction == 1:
            target_cell_types = self.arr2_shared_batch.obs['cell_type'] # change into self
            ori_cell_id = np.array(range(arr1.shape[0]))
            for i in range(ot_sink.shape[0]):
                weights = ot_sink[i, :]
                
                weight_distribution = {}
                for cell_type, weight in zip(target_cell_types, weights):
                    if cell_type in weight_distribution:
                        weight_distribution[cell_type] += weight
                    else:
                        weight_distribution[cell_type] = weight
                

                sct = self.arr1_shared_batch.obs['cell_type'].iloc[i] # change into self
                pct = max(weight_distribution, key = weight_distribution.get)
                source_ct.append(sct) 
                predicted_ct.append(pct)

                if sct == pct:
                    match_result[i] = ot_sink[i,:]
                    
            source_ct = np.array(source_ct)
            predicted_ct = np.array(predicted_ct)
            mismatches = source_ct != predicted_ct
            ori_cell_id = np.array(ori_cell_id)[mismatches]
            wrong_ct = predicted_ct[mismatches]

            return match_result, ori_cell_id, wrong_ct
        
        elif direction == 2:
            target_cell_types = self.arr1_shared_batch.obs['cell_type'] # change into self
            ori_cell_id = np.array(range(arr2.shape[0]))
            
            for i in range(ot_sink.shape[1]):
                weights = ot_sink[:, i]
                weight_distribution = {}
                for cell_type, weight in zip(target_cell_types, weights):
                    if cell_type in weight_distribution:
                        weight_distribution[cell_type] += weight
                    else:
                        weight_distribution[cell_type] = weight

                sct = self.arr2_shared_batch.obs['cell_type'].iloc[i] # change into self
                pct = max(weight_distribution, key = weight_distribution.get)
                source_ct.append(sct) 
                predicted_ct.append(pct)

                if sct == pct:
                    match_result[i] = ot_sink[:, i]

            source_ct = np.array(source_ct)
            predicted_ct = np.array(predicted_ct)
            mismatches = source_ct != predicted_ct
            ori_cell_id = np.array(ori_cell_id)[mismatches]
            wrong_ct = predicted_ct[mismatches]

            return match_result, ori_cell_id, wrong_ct
        
        else:
            raise ValueError('Direction must be 1 or 2!')
                

    def iterative_unbalanced_ot(self, arr1, arr2, ori_cell_id, bot_match_result, reg = 5e-3, reg_m = (2, 0.1), numItermax = 1000, sparse = False, direction = 1):
        """
        stage II of celLink: perform iterative unbalanced optimal transport to correct alignments, filter out the matched cells and retain the unmatched cells.

        Parameters
        -----------------
        arr1: np.array
            The shared-feature arr1
        arr2: np.array
            The shared-feature arr2
        ori_cell_id: list
            Cell ids for unmatched cells in phase I
        reg: float, default = 0.05 
            The penalty coefficient of entropy regularization of UOT
        reg_m: tuple
            Each element in the tuple is the penalty coefficient of the KL divergence of the transported mass from one modality and the initial mass from another
        numItermax: int
            Max number of iterations of sinkhorn algorithms
        direction: int, default = 1
            The direction to align between modalities. If direction is 1, align arr1 -> arr2. If direction is 2, align arr2 -> arr1.

        Returns
        -----------------
        match_result: np.array
            Keys are the cell-cell correspondence matrix fo all matched cells and values are the corresponding vectors.
        ori_cell_id: list,
            Cell ids for the unmatched cells after convergence
        """
        
        iter_time = 0
        if direction == 1:
            continue_iter = True
            arr1_refine = arr1[ori_cell_id, :]
            match_result = bot_match_result
            while(continue_iter):
                iter_time += 1
                dist = cdist_correlation(arr1_refine, arr2)
                a = np.ones(shape = (arr1_refine.shape[0],)) 
                b = np.ones(shape = (arr2.shape[0],)) 

                if(sparse):
                    #print('reg_m:', reg_m)
                    ot_fast = ot.unbalanced.lbfgsb_unbalanced(a = a, b = b, M = dist, reg = 0, reg_m = reg_m, reg_div='kl', numItermax = numItermax, stopThr=1e-15)
                else:
                    ot_fast = ot.unbalanced.sinkhorn_unbalanced(a = a, b = b, M = dist, reg = reg, reg_m = reg_m, numItermax = numItermax, stopThr=1e-15)

                source_ct = []
                predicted_ct = []
                target_cell_types = self.arr2_shared_batch.obs['cell_type'] # change into self

                for i in range(ot_fast.shape[0]):
                    weights = ot_fast[i, :]
                    weight_distribution = {}
                    for cell_type, weight in zip(target_cell_types, weights):
                        if cell_type in weight_distribution:
                            weight_distribution[cell_type] += weight
                        else:
                            weight_distribution[cell_type] = weight

                    sct = self.arr1_shared_batch.obs['cell_type'].iloc[ori_cell_id[i]] # change into self
                    pct = max(weight_distribution, key = weight_distribution.get)
                    source_ct.append(sct) 
                    predicted_ct.append(pct)

                    # store the transport map of matched cells by dictionary
                    if sct == pct:
                        match_result[ori_cell_id[i]] = ot_fast[i, :]

                source_ct = np.array(source_ct)
                predicted_ct = np.array(predicted_ct)
                mismatches = source_ct != predicted_ct

                num_mis = sum(mismatches)
                num = len(ori_cell_id)
                
                if num == 0 or num_mis == 0:
                    ori_cell_id = np.array([])
                    ct_acc = 100
                    continue_iter = False
                    wrong_ct = []
                    print(f'iterative unbalanced optimal transport converges after {iter_time} iterations iterations with cell-type matching accuracy {ct_acc}%! \n')
                    print(f'There are {len(ori_cell_id)} unmatched samples and {len(match_result)} matched samples in data{direction}!\n')
                    return match_result, ori_cell_id, wrong_ct
                elif num_mis / num >= 0.99: 
                    ori_cell_id = np.array(ori_cell_id)[mismatches]
                    wrong_ct = predicted_ct[mismatches]
                    ct_acc = round(1 - len(ori_cell_id) / arr1.shape[0], 4) * 100
                    continue_iter = False
                    print(f'iterative unbalanced optimal transport converges after {iter_time} iterations with cell-type matching accuracy {ct_acc}%! \n')
                    print(f'There are {len(ori_cell_id)} unmatched samples and {len(match_result)} matched samples in data{direction}!\n')
                    return match_result, ori_cell_id, wrong_ct
                else:
                    ori_cell_id = np.array(ori_cell_id)[mismatches]
                    arr1_refine = arr1[ori_cell_id, :]

        elif direction == 2:
            continue_iter = True
            arr2_refine = arr2[ori_cell_id, :]
            match_result = bot_match_result

            while(continue_iter):
                iter_time += 1
                dist = cdist_correlation(arr1, arr2_refine)
                a = np.ones(shape = (arr1.shape[0],)) 
                b = np.ones(shape = (arr2_refine.shape[0],)) 

                if(sparse):
                    #print('reg_m:', reg_m)
                    ot_fast = ot.unbalanced.lbfgsb_unbalanced(a = a, b = b, M = dist, reg = 0, reg_m = reg_m, reg_div='kl', numItermax = numItermax, stopThr=1e-15)
                else:
                    ot_fast = ot.unbalanced.sinkhorn_unbalanced(a = a, b = b, M = dist, reg = reg, reg_m = reg_m, numItermax = numItermax, stopThr=1e-15)

                source_ct = []
                predicted_ct = []

                target_cell_types = self.arr1_shared_batch.obs['cell_type'] # change into self

                for i in range(ot_fast.shape[1]):
                    weights = ot_fast[:, i]
                    weight_distribution = {}
                    for cell_type, weight in zip(target_cell_types, weights):
                        if cell_type in weight_distribution:
                            weight_distribution[cell_type] += weight
                        else:
                            weight_distribution[cell_type] = weight

                    sct = self.arr2_shared_batch.obs['cell_type'].iloc[ori_cell_id[i]] # change into self
                    pct = max(weight_distribution, key = weight_distribution.get)
                    source_ct.append(sct) 
                    predicted_ct.append(pct)

                    if sct == pct:
                        match_result[ori_cell_id[i]] = ot_fast[:, i]

                source_ct = np.array(source_ct)
                predicted_ct = np.array(predicted_ct)
                mismatches = source_ct != predicted_ct

                # store the transport map of matched cells store by dictionary
                num_mis = sum(mismatches)
                num = len(ori_cell_id)
                
                if num == 0 or num_mis == 0:
                    ori_cell_id = np.array([])
                    ct_acc = 100
                    continue_iter = False
                    wrong_ct = []
                    print(f'iterative unbalanced optimal transport converges after {iter_time} iterations with cell-type matching accuracy {ct_acc}%! \n')
                    print(f'There are {len(ori_cell_id)} unmatched samples and {len(match_result)} matched samples in data{direction}!\n')
                    return match_result, ori_cell_id, wrong_ct
                elif num_mis / num >= 0.99: 
                    ori_cell_id = np.array(ori_cell_id)[mismatches]
                    wrong_ct = predicted_ct[mismatches]
                    ct_acc = round(1 - len(ori_cell_id) / arr2.shape[0], 4) * 100
                    continue_iter = False
                    print(f'iterative unbalanced optimal transport converges after {iter_time} iterations with cell-type matching accuracy {ct_acc}%! \n')
                    print(f'There are {len(ori_cell_id)} unmatched samples and {len(match_result)} matched samples in data{direction}!\n')
                    return match_result, ori_cell_id, wrong_ct
                else:
                    ori_cell_id = np.array(ori_cell_id)[mismatches]
                    arr2_refine = arr2[ori_cell_id,:]
        else:
            raise ValueError('Direction must be 1 or 2!')
        
        
    def synchronize_imputed_to_initial(self):
        """
        Re-mapp the imputed feature profiles to align with the original data indices, which are shuffled when partitioning.
        
        Returns
        -----------------
        ann1_aligned_ann2: np.array,
            The imputed features of data modality 2 for data modality 1.
        ann1_predict_ct_array: np.array,
            The predicted cell types for data modality 1 following the original data indices.
        ann2_aligned_ann1: np.array,
            The imputed features of data modality 1 for data modality 2.
        ann2_predict_ct_array: np.array,
            The predicted cell types for data modality 2 fllowing the original data indices.
        """
        ann1_predict_ct_array = np.zeros(shape = self.shared_ann1.shape[0], dtype = 'object')
        # the cell index of the ann1_aligned_ann2 is not the same as the original cell index, modify the logic by looping cellink.partition1[i] rather than len(ann1_batch)
        ann1_aligned_ann2 = np.zeros(shape = (self.full_ann1.shape[0], self.full_ann2.shape[1]))
        for i in range(len(self.partition1)):
            cell_id = self.partition1[i]
            match_id = np.array(list(self.cell_correspondence_partition1[i].keys()))
            for j in range(len(cell_id)):
                unmatched_cell_id_ann1 = self.arr1_unmatched_cell_id[i]
                if j in unmatched_cell_id_ann1:
                    ann1_predict_ct_array[cell_id[j]] = self.arr1_wrong_ct[i][np.where(unmatched_cell_id_ann1 == j)[0][0]]
                else:
                    ann1_predict_ct_array[cell_id[j]] = self.shared_ann1.obs['cell_type'].iloc[cell_id[j]]
                    nid = np.where(match_id == j)[0][0]
                    ann1_aligned_ann2[cell_id[j], :] =  self.feature_imputation_partition1[i][nid, :]

        ann2_predict_ct_array = np.zeros(shape = self.shared_ann2.shape[0], dtype = 'object')
        ann2_aligned_ann1 = np.zeros(shape = (self.full_ann2.shape[0], self.full_ann1.shape[1]))
        for i in range(len(self.partition2)):
            cell_id = self.partition2[i]
            match_id = np.array(list(self.cell_correspondence_partition2[i].keys()))
            for j in range(len(cell_id)):
                unmatched_cell_id_ann2 = self.arr2_unmatched_cell_id[i]
                if j in unmatched_cell_id_ann2:
                    ann2_predict_ct_array[cell_id[j]] = self.arr2_wrong_ct[i][np.where(unmatched_cell_id_ann2 == j)[0][0]]
                else:
                    ann2_predict_ct_array[cell_id[j]] = self.shared_ann2.obs['cell_type'].iloc[cell_id[j]]
                    nid = np.where(match_id == j)[0][0]
                    ann2_aligned_ann1[cell_id[j], :] =  self.feature_imputation_partition2[i][nid, :]
                
        return ann1_aligned_ann2, ann1_predict_ct_array, ann2_aligned_ann1, ann2_predict_ct_array
        


    @staticmethod
    def visualize_integration(ann1_full_batch, ann2_full_batch, arr2_imputed, datatype, matched_cellids, direction = 1, loc = "upper right"):
        """
        Joint embed the arr and arr_imputed from alignment. Visualize the joint embedding results. This is done batch-to-batch

        Parameters:
        -------------------
        arr1: np.array
            the original value of arr1
        arr2_imputed: np.array
            the imputed features of arr2 from arr1
        datatype: list
            the data type of arr and arr2_imputed
        direction: int,
            If direction = 1, jointly embed arr1 and arr2_imputed. If direction = 2, jointly embed arr2 and arr1_imputed.
        loc: str,
            The position of the legend in the figure. Its values can be "upper left", "upper right", "bottom left", "bottom right".
        """
        dataall = np.concatenate([ann1_full_batch.X, arr2_imputed], axis = 0)
        ct_array1 = ann1_full_batch.obs['cell_type'].values
        ct_array2 = ann2_full_batch[matched_cellids, :].obs['cell_type'].values
        ct_array_double = np.concatenate([ct_array1, ct_array2], axis = 0)
        color_palette = plt.cm.tab10(np.linspace(0, 1, 10))
        cts = np.unique(ct_array_double)
        num_type = len(cts)
        if num_type > 10:
            repeats = -(-num_type // 10)
            color_palettes = np.tile(color_palette, (repeats, 1))
            colors = color_palettes[:num_type]
        else:
            colors = color_palette[:num_type]
        colorbar = {t: colors[i] for i, t in enumerate(cts)}
        color_points = np.array([colorbar[i] for i in ct_array_double])

        grey = np.array([0.75, 0.75, 0.75, 0.2])[np.newaxis, :]
        ann1_id = np.array(range(0, len(ct_array1)))
        ann2_id = np.array(range(len(ct_array1), len(ct_array_double)))
        color_points1 = color_points.copy()
        color_points1[ann2_id, :] = grey
        color_points2 = color_points.copy()
        color_points2[ann1_id, :] = grey
        embedding = umap.UMAP(n_components=2, n_epochs = 500, n_neighbors = 15, random_state = 30, min_dist = 0.5).fit_transform(dataall)
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].scatter(embedding[:,0], embedding[:,1], color=color_points1, s=5.)
        axs[1].scatter(embedding[:,0], embedding[:,1], color=color_points2, s=5.)

        if direction == 1:
            axs[0].set_title(f'{datatype[0]} highlighted Embeddings')
            axs[1].set_title(f'{datatype[1]} highlighted Embeddings')
        elif direction == 2:
            axs[0].set_title(f'{datatype[1]} highlighted Embeddings')
            axs[1].set_title(f'{datatype[0]} highlighted Embeddings')

        axs[0].set_xlabel('UMAP-1')
        axs[0].set_ylabel('UMAP-2')
        axs[1].set_xlabel('UMAP-1')
        axs[1].set_ylabel('UMAP-2')

        legend_celltype = [Line2D([0], [0], marker='o', color='w', label=f'{t}',
                                    markerfacecolor=c, markersize=10) for t, c in colorbar.items()]

        axs[0].legend(handles = legend_celltype, title = "Cell Types", loc = loc)
        axs[1].legend(handles = legend_celltype, title = "Cell Types", loc = loc)
        plt.show()
        

    

        