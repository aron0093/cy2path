import logging
logging.basicConfig(level = logging.INFO)
import collections
import numpy as np

from scipy.sparse import issparse

from .sampling import sample_state_probability
from .pseudotime import infer_pseudotime
from .simulation import sample_markov_chains

from hausdorff import hausdorff_distance
#import Fred as fred

import networkit
from networkit.community import ParallelLeiden, PLM

from tqdm.auto import tqdm
from joblib import Parallel, delayed

def get_graph(dm):
    """
    Builds a networkit graph from the input.
    :param dm:
    :return: networkit graph
    """

    m, _ = dm.shape
    g = networkit.Graph(m, weighted=True)
    mask_x, mask_y = np.mask_indices(m, np.tril, -1)
    masking_zip = zip(mask_x, mask_y, dm[mask_x, mask_y])

    for nodeA, nodeB, weight in masking_zip:
        if weight == 0:
            continue
        g.addEdge(nodeA, nodeB, weight)

    return g

def infer_cytopath_lineages(data, matrix_key='T_forward', groupby='louvain', recalc_items=False, recalc_matrix=False, self_transitions=False, 
                  init='root_cells', repeat_root=True,  basis='pca', distance='euclidean', #pseudotime_key='pseudotime',
                  resolution=1.0, num_chains=1000, max_iter=1000, tol=1e-5, n_jobs=-1, copy=False):

    # Run analysis using copy of anndata if specified otherwise inplace
    adata = data.copy() if copy else data

    logging.warning('If precomputed items are used, parameters will not be enforced!')

    # Check if state_probability_sampling() has been run
    if 'state_probability_sampling' not in adata.uns.keys() or recalc_items:
        sample_state_probability(adata, matrix_key=matrix_key, recalc_matrix=recalc_matrix, self_transitions=self_transitions, 
                                 init=init, max_iter=max_iter, tol=tol, copy=False)
    else:
        logging.info('Using precomputed state probability sampling')

    # # Check if infer_pseudotime() has been run
    # if pseudotime_key not in adata.obs.columns or recalc_items:
    #     logging.info('Computing pseudotime')
    #     infer_pseudotime(adata, mode=pseudotime_mode, copy=False)
    # else:
    #     logging.info('Using precomputed pseudotime: {}'.format(pseudotime_key)) 
    
    # Check if sample_markov_chains() has been run
    if 'markov_chain_sampling' not in adata.uns.keys() or recalc_items:
        sample_markov_chains(data, matrix_key=matrix_key, recalc_matrix=False, self_transitions=False, 
                         init=init, repeat_root=True, num_chains=num_chains, max_iter=max_iter, 
                         convergence=adata.uns['state_probability_sampling']['sampling_params']['convergence'], 
                         tol=tol, n_jobs=n_jobs, copy=False)

    else:
        logging.info('Using precomputed Markov chains')

    # Compute pariwise distance matrix for simulations
    num_chains = adata.uns['markov_chain_sampling']['sampling_params']['num_chains']
    markov_chains = adata.uns['markov_chain_sampling']['state_indices']

    

    # Computer pairwise distances
    # TODO: precompute pairwise distances between cells and use those when calculating Hausdorff distance
    # Some other metric/measure e.g. Fretchet
    hausdorff_distances = np.zeros((num_chains, num_chains))
    
    # Cell state representation to be used for Hausdorff distance calculations
    if isinstance(basis, str):
        cell_state_repr = adata.obsm['X_{}'.format(basis)]
    elif basis is None:
        if issparse(adata.X):
            cell_state_repr = adata.X.toarray()
        else:
            cell_state_repr = adata.X
    elif isinstance(basis, (collections.Sequence, np.ndarray)) and np.intersect1d(adata.var.index.values, basis).shape[0]>0:
        basis = np.intersect1d(adata.var.index.values, basis)
        gene_locs = adata.var.index.get_indexer(basis)
        if issparse(adata.X):
            cell_state_repr = adata.X.toarray()
        else:
            cell_state_repr = adata.X        
        cell_state_repr = adata.X[:, gene_locs]

    else:
        raise ValueError('Cannot compute hausdorff distances with provided basis key.')

    for i in tqdm(range(num_chains-1, -1, -1), desc='Computing hausdorff distances', unit=' simulations'):
        for j in range(i):
            hausdorff_distances[i, j] = hausdorff_distance(cell_state_repr[markov_chains[i]],
                                                           cell_state_repr[markov_chains[j]],
                                                           distance=distance)
            #frechet_distances[i, j] = fred.discrete_frechet(fred.Curve(cell_state_repr[markov_chains[i]]),
            #                                                fred.Curve(cell_state_repr[markov_chains[j]]))

            hausdorff_distances[j, i] = hausdorff_distances[i, j]

    # Cluster simulations
    # TODO: Implement a heirarchical clustering algorithm with optimal cut selection
    graph = get_graph(hausdorff_distances)
    cluster = networkit.community.detectCommunities(graph, algo=ParallelLeiden(graph, gamma=resolution))
    cluster.compact()

    cluster_labels = np.array(cluster.getVector())
    clusters = list(cluster.getSubsetIds())

    adata.uns['markov_chain_sampling']['lineage_inference_clusters'] = cluster_labels
    adata.uns['markov_chain_sampling']['lineage_inference_params'] = {'basis': basis, 
                                                                      'distance': distance,
                                                                      'resolution': resolution
                                                                      }
    if copy: return adata




    






        

        




