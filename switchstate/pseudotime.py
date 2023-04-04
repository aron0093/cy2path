import numpy as np
from .utils import scale
from .analytical import calculate_expected_steps

def infer_pseudotime(data, mode='weighted_mean', copy=False):

    adata = data.copy() if copy else data

    # Check if state history exists
    try: state_history = adata.uns['state_probability_sampling']['state_history']
    except: raise ValueError('State probability history could not be recovered. Run sample_state_probability() first')

    adata.obs['pseudotime'] = None
    adata.uns['state_probability_sampling']['pseudotime_params'] = {'mode': mode}

    if mode=='max':
        adata.obs['pseudotime'] = state_history.argmax(axis=0)
    if mode=='mean':
        binarize_history = np.where(state_history>0, 1, 0)
        adata.obs['pseudotime'] = binarize_history.mean(axis=0)
    if mode=='weighted_mean':
        _ = np.repeat([np.arange(0, state_history.shape[0])], state_history.shape[1], 0).T
        _ = _*state_history
        adata.obs['pseudotime'] = _.sum(axis=0)/state_history.sum(axis=0)
    if mode=='median':
        raise NotImplementedError()

    adata.obs['pseudotime'] = scale(adata.obs['pseudotime'])

    if copy: return adata

# Add expected steps to .obs
def add_expected_steps(data, copy=False):

    adata = data.copy() if copy else data

    # Check if fundamental matrix exists
    try: N = adata.uns['cell_fate']['fundamental_matrix']
    except: raise ValueError('Fundamental matrix could not be recovered. Run infer_fate_distribution() first')

    transition_state_indices = adata.uns['cell_fate']['transition_state_indices']

    expected_steps = np.zeros(adata.shape[0])
    expected_steps[transition_state_indices] = calculate_expected_steps(N)

    adata.obs['expected_steps'] = expected_steps

    if copy: return adata
    








