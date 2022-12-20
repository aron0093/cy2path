import logging
logging.basicConfig(level = logging.INFO)
import collections
import numpy as np

from scvelo.utils import get_transition_matrix
from scvelo.tools import terminal_states
from scvelo.tools.terminal_states import eigs

from scipy.spatial.distance import cosine
from scipy.sparse import issparse, csr_matrix

from .sampling import iterate_state_probability

from tqdm.auto import tqdm
from joblib import Parallel, delayed

def extract_nonzero_entries(adata, matrix_key='T_forward'):
        
        # Extract non zero elements and create cumumlative distribution
        trans_matrix_indices = np.split(adata.uns[matrix_key].indices, adata.uns[matrix_key].indptr)[1:-1]
        trans_matrix_probabilites = np.split(adata.uns[matrix_key].data, adata.uns[matrix_key].indptr)[1:-1]
        
        for i in range(len(trans_matrix_probabilites)):
            trans_matrix_probabilites[i]=np.cumsum(trans_matrix_probabilites[i])
        
        # For increased speed: Save data and indices of nonzero entries in np array--> 210% faster
        trans_matrix_indice_nonzeros = np.zeros((len(trans_matrix_probabilites), 
                                            max(np.count_nonzero(adata.uns[matrix_key].toarray(), axis=1))), 
                                            dtype=np.int32)
        trans_matrix_probabilites_nonzeros = np.zeros((len(trans_matrix_probabilites), 
                                                    max(np.count_nonzero(adata.uns[matrix_key].toarray(), axis=1))), 
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
        state_transition_probabilities_[k-1] = adata.uns[matrix_key][state_indices_[k-1], state_indices_[k]]

    return state_indices_, state_transition_probabilities_

def sample_markov_chains(data, matrix_key='T_forward', recalc_matrix=True, self_transitions=False, 
                         init='root_cells', repeat_root=True, num_chains=1000, max_iter=1000, 
                         convergence='auto', tol=1e-5, n_jobs=-1, copy=False):

    # Run analysis using copy of anndata if specified otherwise inplace
    adata = data.copy() if copy else data

    # Recalcualte TPM if specified
    if recalc_matrix:
        logging.info('recalc_matrix=True, both TPM and root/end states will be recalculated.')
        adata.uns[matrix_key] = get_transition_matrix(adata, self_transitions=self_transitions)
        terminal_states(adata, self_transitions=self_transitions)
    else:
        try: 
            assert adata.uns[matrix_key].shape
        except: 
            adata.uns[matrix_key] = get_transition_matrix(adata, self_transitions=self_transitions)
            terminal_states(adata, self_transitions=self_transitions)
            logging.warning('Transition probability matrix was not present and was calculated.')

        if not issparse(adata.uns[matrix_key]):
            adata.uns[matrix_key] = csr_matrix(adata.uns[matrix_key])

    # Recalculate root and terminal states if not present and TPM was not recalculated
    try: 
        assert 'root_cells' in adata.obs.columns
        assert 'end_points' in adata.obs.columns
    except: 
        terminal_states(adata, self_transitions=self_transitions)
        logging.warning('Root states were not present and were calculated.')

    # Initialise using root_cells, uniform or custom
    if isinstance(init, str):
        if init=='root_cells':
            init_state_probability = (adata.obs['root_cells']/adata.obs['root_cells'].sum()).values
            init_type = init
        elif init=='uniform':
            init_state_probability = [1/adata.shape[0]]*adata.shape[0]  # uniform probability to start at each cell
            init_type = 'uniform'
    elif isinstance(init, (collections.Sequence, np.ndarray)):
        init_state_probability=init
        init_type = 'custom'
    else:
        raise ValueError('Incorrect initialisation of state probabilities.')
   
    # Stationary state distribution
    stationary_state_probability = eigs(adata.uns[matrix_key])[1]
    if stationary_state_probability.shape[1] == 1:
        stationary_state_probability /= stationary_state_probability.sum()
    else:
        stationary_state_probability = (adata.obs['end_points']/adata.obs['end_points'].sum()).values
        logging.warning('Multiple eigenvalues > 1, falling back to end_points to infer stationary distribution.') 

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
                                                         'tol': tol, 
                                                         'copy': copy}
    if copy: return adata



