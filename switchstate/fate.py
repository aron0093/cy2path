import logging
logging.basicConfig(level = logging.INFO)
import collections
import numpy as np
import pandas as pd

from scvelo.utils import get_transition_matrix
from scvelo.tools import terminal_states

from scipy.sparse import issparse, csr_matrix

from .analytical import calculate_fundamental, calculate_absorbing_probabilities

# Infer the fundamental matrix and fate probabilities
# https://en.wikipedia.org/wiki/Absorbing_Markov_chain#Fundamental_matrix
# TODO: Implement solver based approach as inverting matrix is costly
def calculate_fate_probabilities(adata, matrix_key='T_forward', terminal_state_indices=None):

    # Set terminal states as absorbing states
    T = adata.uns[matrix_key].copy()
    T[terminal_state_indices, :] = 0
    T[terminal_state_indices, terminal_state_indices] = 1

    # Transition states
    transition_state_indices = list(set(np.arange(adata.shape[0])).difference(terminal_state_indices))

    N = calculate_fundamental(T, transition_state_indices)
    B = calculate_absorbing_probabilities(N, T, transition_state_indices, terminal_state_indices)

    return N, T, B, transition_state_indices

# Infer fate probability distribution of each cell with respect to the terminal states
def infer_fate_distribution(data, matrix_key='T_forward', recalc_matrix=False, self_transitions=True, 
                            terminal_state_indices='threshold', terminal_state_probability=0.95, copy=False):

    # Run analysis using copy of anndata if specified otherwise inplace
    adata = data.copy() if copy else data

    # Recalcualte TPM if specified
    if recalc_matrix:
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
    if terminal_state_indices=='threshold':
        try: 
            assert 'end_points' in adata.obs.columns
        except: 
            terminal_states(adata, self_transitions=self_transitions)
            logging.warning('Terminal states were not present and were calculated.')

        terminal_state_indices = np.flatnonzero(adata.obs['end_points']>=terminal_state_probability)

    elif isinstance(terminal_state_indices, (collections.Sequence, np.ndarray)): 
        terminal_state_probability = None
    else:
        raise ValueError('Incorrect initialisation of terminal states indices.')

    # Calculate probability of any transient state ending up in a particular absorbing state
    fundamental_matrix, absorbing_transition_matrix, fate_probabilities_, transition_state_indices = calculate_fate_probabilities(adata, matrix_key=matrix_key, 
                                                                                                       terminal_state_indices=terminal_state_indices)

    # Supplement transient probabilities with absorbing state for convenience
    fate_probabilities = np.zeros((adata.shape[0], len(terminal_state_indices)))
    fate_probabilities[terminal_state_indices, np.arange(len(terminal_state_indices))] = 1
    fate_probabilities[transition_state_indices] = fate_probabilities_                                                                                                   
    
    adata.uns['state_probability_sampling']['fundamental_matrix'] = fundamental_matrix
    adata.uns['state_probability_sampling']['absorbing_transition_matrix'] = absorbing_transition_matrix
    adata.uns['state_probability_sampling']['terminal_state_indices'] = terminal_state_indices
    adata.uns['state_probability_sampling']['transition_state_indices'] = transition_state_indices
    adata.uns['state_probability_sampling']['fate_probabilities'] = fate_probabilities
    adata.uns['state_probability_sampling']['fate_params'] = {'recalc_matrix': recalc_matrix, 
                                                              'self_transitions': self_transitions,
                                                              'terminal_state_probability': terminal_state_probability,
                                                              'copy': copy}

 
    if copy: return adata

# Condense fate probabilities to macrostate
def infer_fate_distribution_macrostates(data, groupby='louvain', matrix_key='T_forward', recalc_matrix=False, self_transitions=True, 
                                        terminal_state_indices='threshold', terminal_state_probability=0.95, copy=False):

    # Run analysis using copy of anndata if specified otherwise inplace
    adata = data.copy() if copy else data

    # Check if grouping key exists
    if groupby not in adata.obs.columns:
        raise ValueError('Groupby key does not exist')

    # Fate probabilities per terminal cell
    try: assert adata.uns['state_probability_sampling']['fate_probabilities'].shape
    except: 
        logging.info('Inferring fate probabilities w.r.t. terminal cells')
        infer_fate_distribution(adata, matrix_key=matrix_key, recalc_matrix=recalc_matrix, self_transitions=self_transitions, 
                            terminal_state_indices=terminal_state_indices, terminal_state_probability=terminal_state_probability, 
                            copy=False)
    fate_probabilities_macro = pd.DataFrame(adata.uns['state_probability_sampling']['fate_probabilities'].T,
                                            index=adata.obs.index.values[adata.uns['state_probability_sampling']['terminal_state_indices']],
                                            columns=adata.obs.index.values
                                           )

    fate_probabilities_macro[groupby] = adata.obs[groupby]
    fate_probabilities_macro = fate_probabilities_macro.groupby(groupby).sum()

    adata.uns['state_probability_sampling']['fate_probabilities_macro'] = fate_probabilities_macro
    adata.uns['state_probability_sampling']['fate_params']['groupby'] = groupby

    if copy: return adata

    

    

    




    






