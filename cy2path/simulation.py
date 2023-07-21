import logging
logging.basicConfig(level = logging.INFO)
import numpy as np
import pandas as pd

from scipy.spatial.distance import cosine

from .sampling import iterate_state_probability, check_convergence_criteria
from .utils import check_TPM, check_root_init, estimate_stationary_state

from tqdm.auto import tqdm
from joblib import Parallel, delayed

def extract_nonzero_entries(adata, matrix_key='T_forward'):
        
        # Extract non zero elements and create cumumlative distribution
        trans_matrix_indices = np.split(adata.obsp[matrix_key].indices, adata.obsp[matrix_key].indptr)[1:-1]
        trans_matrix_probabilites = np.split(adata.obsp[matrix_key].data, adata.obsp[matrix_key].indptr)[1:-1]
        
        for i in range(len(trans_matrix_probabilites)):
            trans_matrix_probabilites[i]=np.cumsum(trans_matrix_probabilites[i])
        
        # For increased speed: Save data and indices of nonzero entries in np array--> 210% faster
        trans_matrix_indice_nonzeros = np.zeros((len(trans_matrix_probabilites), 
                                            max(np.count_nonzero(adata.obsp[matrix_key].toarray(), axis=1))), 
                                            dtype=np.int32)
        trans_matrix_probabilites_nonzeros = np.zeros((len(trans_matrix_probabilites), 
                                                    max(np.count_nonzero(adata.obsp[matrix_key].toarray(), axis=1))), 
                                                    dtype=np.float32)
    
        for i in range(len(trans_matrix_probabilites)):
            for j in range(len(trans_matrix_probabilites[i])):
                trans_matrix_indice_nonzeros[i, j] = trans_matrix_indices[i][j]
                trans_matrix_probabilites_nonzeros[i, j] = trans_matrix_probabilites[i][j]
        
        # FIXME: Ensure that cummulative sum reaches 1 precisely (rounding to 3 decimals ensures that this almost never happens)
        trans_matrix_probabilites_nonzeros = np.round(trans_matrix_probabilites_nonzeros, decimals=3)
    
        return trans_matrix_indice_nonzeros, trans_matrix_probabilites_nonzeros

# Sample markov chains
def iterate_markov_chain(adata, matrix_key='T_forward', max_iter=1000, init_state=None, nonzero_entries=None):

    if nonzero_entries is None:
        trans_matrix_indice_nonzeros, trans_matrix_probabilites_nonzeros = extract_nonzero_entries(adata, matrix_key=matrix_key)
    else:
        trans_matrix_indice_nonzeros, trans_matrix_probabilites_nonzeros = nonzero_entries
    
    random_step = np.random.rand(max_iter-1)
    state_indices_ = np.empty(max_iter, dtype=np.int32)
    state_indices_[0] = init_state
    state_transition_probabilities_ = np.empty(max_iter-1, dtype=np.float32)

    for k in range(1, max_iter):
        # Retrieve next cell transition and save the probabilities of the transition and the cluster the cell is in
        state_indices_[k] = trans_matrix_indice_nonzeros[state_indices_[k-1], 
                                                    np.where(random_step[k-1] <= \
                                                    trans_matrix_probabilites_nonzeros[state_indices_[k-1], :])[0][0]]
        state_transition_probabilities_[k-1] = adata.obsp[matrix_key][state_indices_[k-1], state_indices_[k]]

    return state_indices_, state_transition_probabilities_

def sample_markov_chains(data, matrix_key='T_forward', recalc_matrix=False, self_transitions=False, 
                         init='root_cells', repeat_root=True, num_chains=1000, max_iter=1000, 
                         convergence='auto', tol=1e-5, n_jobs=-1, copy=False):

    # Run analysis using copy of anndata if specified otherwise inplace
    adata = data.copy() if copy else data
    check_TPM(adata, matrix_key=matrix_key, recalc_matrix=recalc_matrix, self_transitions=self_transitions)

    # Initialise using root_cells, uniform or custom
    init_state_probability, init_type = check_root_init(adata, init=init)
   
    # Stationary state distribution
    stationary_state_probability = estimate_stationary_state(adata, matrix_key=matrix_key)

    # Create empty arrays for the simulation and sample random numbers for all samples and max steps
    state_transition_probabilities = np.empty((num_chains, max_iter-1), dtype=np.float32)
    state_indices = np.empty((num_chains, max_iter), dtype=np.int32)
    eps_history = np.empty(max_iter)

    trans_matrix_indice_nonzeros, trans_matrix_probabilites_nonzeros = extract_nonzero_entries(adata, matrix_key=matrix_key)

    # Simulate Markov chains
    simulations = Parallel(n_jobs=n_jobs)(delayed(iterate_markov_chain)(adata, 
                                                                         matrix_key=matrix_key, 
                                                                         max_iter=max_iter, 
                                                                         init_state=np.random.choice(adata.shape[0], 1, replace=repeat_root, p=init_state_probability), 
                                                                         nonzero_entries=(trans_matrix_indice_nonzeros, trans_matrix_probabilites_nonzeros)) for i in tqdm(range(num_chains),
                                                                         desc='Iterating Markov chains',  unit=' simulations'))
    # Store Markov chains    
    for i in tqdm(range(num_chains), desc='Storing Markov chains', unit=' simulations'):
        state_transition_probabilities[i] = simulations[i][1]
        state_indices[i] = simulations[i][0]

    # Empirical state history distributions
    state_history_max_iter = np.empty((max_iter, adata.shape[0]), dtype=np.float32)
    for k in tqdm(range(max_iter), desc='Computing empirical state probability distribution'):
        indices, counts = np.unique(state_indices[:, k], return_counts=True)
        probs = counts/counts.sum()
        state_history_max_iter[k, indices] = probs

    # Get convergence criteria to select number of simulation steps
    if convergence == 'auto':
        convergence_check = iterate_state_probability(adata, matrix_key=matrix_key, 
                                              init=init_state_probability, stationary=stationary_state_probability, 
                                              max_iter=max_iter, tol=tol)[-1]
    elif isinstance(convergence, int):
        convergence_check=convergence

    # Convergence of cluster proportions to stationary state
    elif convergence in adata.obs.columns:
        if not pd.api.types.is_categorical_dtype(adata.obs[convergence]):
            logging.warning(f'{convergence} in adata.obs should be categorical.')
        stationary_state_probability_by_cluster = pd.DataFrame({'cluster': adata.obs[convergence].astype('category'),
                                                                'probability': stationary_state_probability}).groupby('cluster').sum()
        cluster_sequences = adata.obs[convergence].astype(str).values[state_indices]
        # calculate cluster proportions for each step
        cluster_proportions = pd.DataFrame(data=np.zeros((len(stationary_state_probability_by_cluster), max_iter),
                                                         dtype=np.float),
                                           index=stationary_state_by_cluster.index)
        for cat in stationary_state_by_cluster.index:
            cluster_proportions.loc[cat, :] = (cluster_sequences == cat).mean(axis=0)
        eps_history = cluster_proportions.apply(lambda x: cosine(x.values.flatten(),
                                                                 stationary_state_by_cluster.values.flatten())).values
        convergence_check = check_convergence_criteria(eps_history, tol)
        if isinstance(convergence_check, int) and convergence_check < max_iter:
            logging.info('Tolerance reached after {} iterations of {}.'.format(convergence_check, max_iter))
        else:
            logging.warning('Max number ({}) of iterations reached.'.format(max_iter))
            convergence_check = max_iter

    state_indices_converged = state_indices[:, :convergence_check]
    state_history_converged = state_history_max_iter[:convergence_check]

    adata.uns['markov_chain_sampling'] = {}
    adata.uns['markov_chain_sampling']['state_indices'] = state_indices_converged
    adata.uns['markov_chain_sampling']['state_history'] = state_history_converged
    adata.uns['markov_chain_sampling']['state_indices_max_iter'] = state_indices
    adata.uns['markov_chain_sampling']['state_history_max_iter'] = state_history_max_iter
    adata.uns['markov_chain_sampling']['state_transition_probabilities_max_iter'] = state_transition_probabilities
    adata.uns['markov_chain_sampling']['init_state_probability'] = init_state_probability
    adata.uns['markov_chain_sampling']['stationary_state_probability'] = stationary_state_probability
    adata.uns['markov_chain_sampling']['sampling_params'] = {'recalc_matrix': recalc_matrix,
                                                             'self_transitions': self_transitions,
                                                             'init_type': init_type,
                                                             'max_iter': max_iter,
                                                             'num_chains': num_chains,
                                                             'convergence': convergence_check,
                                                             'convergence_criterion': convergence,
                                                             'tol': tol,
                                                             'copy': copy}
    if convergence in adata.obs.columns:
        adata.uns['markov_chain_sampling']['stationary_state_by_cluster'] = stationary_state_by_cluster.values
        adata.uns['markov_chain_sampling']['cluster_sequences'] = cluster_sequences[:, :convergence_check]
        adata.uns['markov_chain_sampling']['cluster_proportions'] = cluster_proportions.values[:, :convergence_check].T
        adata.uns['markov_chain_sampling']['cluster_sequences_max_iter'] = cluster_sequences
        adata.uns['markov_chain_sampling']['cluster_proportions_max_iter'] = cluster_proportions.values.T
    if copy: return adata



