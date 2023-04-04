import logging
logging.basicConfig(level = logging.INFO)
import collections
import numpy as np
import pandas as pd

from scvelo.utils import get_transition_matrix
from scvelo.tools import terminal_states

from scipy.sparse import issparse, csr_matrix

from .analytical import calculate_fundamental, calculate_absorbing_probabilities
from .utils import check_TPM

# Infer the fundamental matrix and fate probabilities
# https://en.wikipedia.org/wiki/Absorbing_Markov_chain#Fundamental_matrix
# TODO: More efficient approach to matrix inversion
def calculate_fate_probabilities(adata, matrix_key='T_forward', terminal_state_indices=None):

    # Set terminal states as absorbing states
    T = adata.obsp[matrix_key].copy()
    T[terminal_state_indices, :] = 0
    T[terminal_state_indices, terminal_state_indices] = 1

    # Transition states
    transition_state_indices = list(set(np.arange(adata.shape[0])).difference(terminal_state_indices))

    N = calculate_fundamental(T, transition_state_indices)
    B = calculate_absorbing_probabilities(N, T, transition_state_indices, terminal_state_indices)

    return N, T, B, transition_state_indices

# Infer fate probability distribution of each cell with respect to the terminal states
def infer_fate_distribution(data, matrix_key='T_forward', recalc_matrix=False, self_transitions=False, 
                            terminal_state_indices='threshold', terminal_state_probability=0.95, copy=False):

    # Run analysis using copy of anndata if specified otherwise inplace
    adata = data.copy() if copy else data
    check_TPM(adata, recalc_matrix=recalc_matrix, self_transitions=self_transitions)

    if terminal_state_indices=='threshold':
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
    
    adata.uns['cell_fate'] = {}
    adata.uns['cell_fate']['fundamental_matrix'] = fundamental_matrix
    adata.uns['cell_fate']['absorbing_transition_matrix'] = absorbing_transition_matrix
    adata.uns['cell_fate']['terminal_state_indices'] = terminal_state_indices
    adata.uns['cell_fate']['transition_state_indices'] = transition_state_indices
    adata.uns['cell_fate']['fate_probabilities'] = fate_probabilities
    adata.uns['cell_fate']['fate_params'] = {'recalc_matrix': recalc_matrix, 
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
    try: assert adata.uns['cell_fate']['fate_probabilities'].shape
    except: 
        logging.info('Inferring fate probabilities w.r.t. terminal cells')
        infer_fate_distribution(adata, matrix_key=matrix_key, recalc_matrix=recalc_matrix, self_transitions=self_transitions, 
                            terminal_state_indices=terminal_state_indices, terminal_state_probability=terminal_state_probability, 
                            copy=False)
    fate_probabilities_macro = pd.DataFrame(adata.uns['cell_fate']['fate_probabilities'].T,
                                            index=adata.obs.index.values[adata.uns['cell_fate']['terminal_state_indices']],
                                            columns=adata.obs.index.values
                                           )

    fate_probabilities_macro[groupby] = adata.obs[groupby]
    fate_probabilities_macro = fate_probabilities_macro.groupby(groupby).sum()

    adata.uns['cell_fate']['fate_probabilities_macro'] = fate_probabilities_macro
    adata.uns['cell_fate']['fate_params']['groupby'] = groupby

    if copy: return adata

    

    

    




    






