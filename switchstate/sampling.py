import logging
logging.basicConfig(level = logging.INFO)
import collections
import numpy as np

from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix

from tqdm.auto import tqdm

from .utils import check_TPM, check_root_init, estimate_stationary_state

def check_convergence_criteria(eps_history, tol=1e-5):
    differences = [abs(eps_history[i+1]-eps_history[i]) for i in range(len(eps_history)-1)]
    # return first element that is under tolerance threshold
    try:
        return 1 + len(differences) - next(i for i, j in enumerate(differences[::-1]) if j > tol)
    except:
        return None

# Update state probabilities by taking dot product with TPM
def iterate_state_probability(adata, matrix_key='T_forward', init=None, stationary=None, max_iter=1000, tol=1e-5):

    # Iterate state probabilities
    state_history_max_iter = np.empty((max_iter, adata.shape[0]))
    eps_history = np.empty(max_iter)

    # Iteratively calculate distribution
    current_state_probability = init
    for i in tqdm(range(max_iter), desc='Iterating state probability distributions'):
        
        state_history_max_iter[i] = current_state_probability
        current_state_probability = csr_matrix.dot(current_state_probability, adata.obsp[matrix_key])
        
        eps = cosine(current_state_probability, stationary)
        eps_history[i] = eps

    # Check if iterations have converged w.r.t. tolerance criteria
    convergence_check = check_convergence_criteria(eps_history, tol=tol)
    if isinstance(convergence_check, int) and convergence_check < max_iter:
        logging.info('Tolerance reached after {} iterations of {}.'.format(convergence_check, max_iter))
    else:
        logging.warning('Max number ({}) of iterations reached.'.format(max_iter))
        convergence_check = max_iter
    
    state_history = state_history_max_iter[:convergence_check]
    return state_history, state_history_max_iter, convergence_check

# Evolve state probabilities 
def sample_state_probability(data, matrix_key='T_forward', recalc_matrix=False, self_transitions=False, 
                            init='root_cells', max_iter=1000, tol=1e-5, copy=False):

    # Run analysis using copy of anndata if specified otherwise inplace
    adata = data.copy() if copy else data
    check_TPM(adata, matrix_key=matrix_key, recalc_matrix=recalc_matrix, self_transitions=self_transitions)

    # Initialise using root_cells, uniform or custom
    init_state_probability, init_type = check_root_init(adata, init=init)
   
    # Stationary state distribution
    stationary_state_probability = estimate_stationary_state(adata, matrix_key=matrix_key)

    # Iterate state probabilities till convergence
    state_history, state_history_max_iter, convergence_check = iterate_state_probability(adata, matrix_key=matrix_key, 
                                              init=init_state_probability, stationary=stationary_state_probability, 
                                              max_iter=max_iter, tol=tol)
        
    adata.uns['state_probability_sampling'] = {}
    adata.uns['state_probability_sampling']['state_history'] = state_history
    adata.uns['state_probability_sampling']['state_history_max_iter'] = state_history_max_iter
    adata.uns['state_probability_sampling']['init_state_probability'] = init_state_probability
    adata.uns['state_probability_sampling']['stationary_state_probability'] = stationary_state_probability
    adata.uns['state_probability_sampling']['sampling_params'] = {'recalc_matrix': recalc_matrix, 
                                                         'self_transitions': self_transitions, 
                                                         'init_type': init_type, 
                                                         'max_iter': max_iter,
                                                         'convergence': convergence_check,
                                                         'tol': tol, 
                                                         'copy': copy}
    if copy: return adata
