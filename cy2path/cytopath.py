import collections
import logging

import numpy as np
from dtaidistance import clustering, dtw_ndim, preprocessing
from hausdorff import hausdorff_distance
from scipy.cluster import hierarchy
from scipy.sparse import issparse
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score
from tqdm.auto import tqdm

from .sampling import sample_state_probability
from .simulation import sample_markov_chains

logging.basicConfig(level=logging.INFO)
# Citation
# Revant Gupta, Dario Cerletti, Gilles Gut, Annette Oxenius, Manfred Claassen,
# Simulation-based inference of differentiation trajectories from RNA velocity fields,
# Cell Reports Methods,
# Volume 2, Issue 12,
# 2022,
# 100359,
# ISSN 2667-2375,
# https://doi.org/10.1016/j.crmeth.2022.100359.
# (https://www.sciencedirect.com/science/article/pii/S2667237522002569)
# Abstract: Summary
# We report Cytopath, a method for trajectory inference that takes advantage of transcriptional activity information from the RNA velocity of single cells to perform trajectory inference. Cytopath performs this task by defining a Markov chain model, simulating an ensemble of possible differentiation trajectories, and constructing a consensus trajectory. We show that Cytopath can recapitulate the topological and molecular characteristics of the differentiation process under study. In our analysis, we include differentiation trajectories with varying bifurcated, circular, convergent, and mixed topologies studied in single-snapshot as well as time-series single-cell RNA sequencing experiments. We demonstrate the capability to reconstruct differentiation trajectories, assess the association of RNA velocity-based pseudotime with actually elapsed process time, and identify drawbacks in current state-of-the art trajectory inference approaches.
# Keywords: single-cell RNA sequencing; RNA velocity; trajectory inference; simulation-based inference


def compute_Hausdorff_distance(simulations, distance='euclidean'):
    num_chains = len(simulations)
    hausdorff_distances = np.zeros((num_chains, num_chains))
    for i in tqdm(
        range(num_chains - 1, -1, -1),
        desc='Computing hausdorff distances',
        unit=' simulations',
    ):
        for j in range(i):
            hausdorff_distances[i, j] = hausdorff_distance(
                simulations[i], simulations[j], distance=distance
            )
            hausdorff_distances[j, i] = hausdorff_distances[i, j]

    return hausdorff_distances


# Cluster Markov simulations for lineage inference
def cluster_markov_chains(
    adata,
    num_lineages=None,
    method='HDBSCAN',
    distance_func='dtw',
    differencing=False,
    basis='pca',
    max_lineages=10,
    n_jobs=-1,
):
    # Compute pariwise distance matrix for simulations
    num_chains = adata.uns['markov_chain_sampling']['sampling_params']['num_chains']
    markov_chains = adata.uns['markov_chain_sampling']['state_indices']

    # Cell state representation to be used for distance calculations
    if isinstance(basis, str):
        cell_state_repr = adata.obsm['X_{}'.format(basis)]
    elif basis is None:
        if issparse(adata.X):
            cell_state_repr = adata.X.toarray()
        else:
            cell_state_repr = adata.X
    elif (
        isinstance(basis, (collections.Sequence, np.ndarray))
        and np.intersect1d(adata.var.index.values, basis).shape[0] > 0
    ):
        basis = np.intersect1d(adata.var.index.values, basis)
        gene_locs = adata.var.index.get_indexer(basis)
        if issparse(adata.X):
            cell_state_repr = adata.X.toarray()
        else:
            cell_state_repr = adata.X
        cell_state_repr = adata.X[:, gene_locs]

    else:
        raise ValueError('Cannot compute distances with provided basis key.')

    # Cluster simulations
    logging.info('Clustering the samples.')

    simulations = cell_state_repr[markov_chains].astype('double')
    if differencing:
        simulations = preprocessing.differencing(simulations)
    if distance_func == 'dtw':
        distance_func = dtw_ndim.distance_matrix_fast
    elif distance_func == 'hausdorff':
        distance_func = compute_Hausdorff_distance

    def cluster_n_lineages(num_lineages, method, distance_func, simulations):
        if num_lineages > 0:
            if method == 'linkage':
                # Construct linkage tree
                model = clustering.LinkageTree(
                    dists_fun=distance_func, dists_options={}, method='ward'
                )
                cluster_idx = model.fit(simulations)
                cluster_labels = hierarchy.fcluster(
                    model.linkage, num_lineages, criterion='maxclust'
                )

            elif method == 'kmediods':
                model = clustering.KMedoids(distance_func, {}, k=num_lineages)
                cluster_labels = model.fit(simulations)
            else:
                raise ValueError('Invalid method for clustering!')
        else:
            raise ValueError('num_lineages must be greater than 0!')

        return cluster_labels, model

    if method == 'HDBSCAN':
        if num_lineages is not None:
            logging.warning('num_lineages ignored for method HDBSCAN!')

        distances = distance_func(simulations)

        # HDBSCAN
        model = HDBSCAN(
            min_cluster_size=int(num_chains * 0.05),
            metric='precomputed',
            n_jobs=n_jobs,
            allow_single_cluster=True,
        )
        cluster_labels = model.fit_predict(distances)

    elif type(num_lineages) is int and method in ['linkage', 'kmediods']:
        cluster_labels, model = cluster_n_lineages(
            num_lineages, method, distance_func, simulations
        )

    elif num_lineages == 'auto' and method in ['linkage', 'kmediods']:
        if max_lineages > 2:
            # Compute pairwise distance matrix
            distances = distance_func(simulations)

            silhouette_scores = {}
            cluster_labels_dict = {}
            model_dict = {}

            # Iterate over the range of number of lineages
            for i in tqdm(
                range(2, max_lineages + 1),
                desc=f'Finding optimal number of lineages between 2 and {max_lineages}',
                total=max_lineages - 1,
            ):
                cluster_labels_dict[i], model_dict[i] = cluster_n_lineages(
                    num_lineages=i,
                    method=method,
                    distance_func=distance_func,
                    simulations=simulations,
                )
                silhouette_scores[i] = silhouette_score(
                    distances, cluster_labels_dict[i], metric='precomputed'
                )
            # Find the optimal number of lineages
            optimal_num_lineages = max(silhouette_scores, key=silhouette_scores.get)
            cluster_labels = cluster_labels_dict[optimal_num_lineages]
            model = model_dict[optimal_num_lineages]
            logging.info(
                f'Optimal number of lineages is {optimal_num_lineages} '
                f'with silhouette score {silhouette_scores[optimal_num_lineages]:.3g}'
            )
            if 'cytopath' not in adata.uns.keys():
                adata.uns['cytopath'] = {}
            adata.uns['cytopath']['silhouette_scores'] = silhouette_scores
            adata.uns['cytopath']['optimal_num_lineages'] = optimal_num_lineages
        else:
            raise ValueError(
                'max_lineages must be greater than 2 for auto mode with linkage or kmediods!'
            )
    else:
        raise ValueError('Incompatible num_lineages and method specification!')

    return cluster_labels, model


# Minimal Cytopath implementation with no cell fate assignment
def infer_cytopath_lineages(
    data,
    matrix_key='T_forward',
    self_transitions=False,
    init='root_cells',
    recalc_items=False,
    recalc_matrix=False,
    num_lineages=None,
    max_lineages=10,
    method='HDBSCAN',
    distance_func='dtw',
    differencing=False,
    basis='pca',
    num_chains=1000,
    max_iter=1000,
    tol=1e-5,
    n_jobs=-1,
    copy=False,
):
    # Run analysis using copy of anndata if specified otherwise inplace
    adata = data.copy() if copy else data

    logging.warning('If precomputed items are used, parameters will not be enforced!')

    # Check if state_probability_sampling() has been run
    if 'state_probability_sampling' not in adata.uns.keys() or recalc_items:
        sample_state_probability(
            adata,
            matrix_key=matrix_key,
            recalc_matrix=recalc_matrix,
            self_transitions=self_transitions,
            init=init,
            max_iter=max_iter,
            tol=tol,
            copy=False,
        )
    else:
        logging.info('Using precomputed state probability sampling')

    # Check if sample_markov_chains() has been run
    if 'markov_chain_sampling' not in adata.uns.keys() or recalc_items:
        logging.info('Sampling Markov chains')
        sample_markov_chains(
            data,
            matrix_key=matrix_key,
            recalc_matrix=False,
            self_transitions=False,
            init=init,
            repeat_root=True,
            num_chains=num_chains,
            max_iter=max_iter,
            convergence=adata.uns['state_probability_sampling']['sampling_params'][
                'convergence'
            ],
            tol=tol,
            n_jobs=n_jobs,
            copy=False,
        )

    else:
        logging.info('Using precomputed Markov chains')

    adata.uns['cytopath'] = {}

    cluster_labels, model = cluster_markov_chains(
        adata,
        num_lineages=num_lineages,
        method=method,
        distance_func=distance_func,
        differencing=differencing,
        basis=basis,
        max_lineages=max_lineages,
        n_jobs=-1,
    )

    adata.uns['cytopath']['lineage_inference_clusters'] = cluster_labels
    adata.uns['cytopath']['lineage_inference_params'] = {
        'basis': basis,
        'num_lineages': num_lineages,
        'method': method,
    }
    if method == 'linkage':
        adata.uns['cytopath']['lineage_inference_linkage'] = model.linkage

    if copy:
        return adata
