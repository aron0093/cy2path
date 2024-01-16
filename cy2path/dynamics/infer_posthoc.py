import numpy as np
from scipy.stats import entropy

import logging
logging.basicConfig(level = logging.INFO)

# Compute conditional using selected states    
def compute_conditionals(adata, use_selected=True):

    if use_selected:
        try: assert 'selected_states' in adata.uns['latent_dynamics']['posthoc_computations'].keys()
        except: raise ValueError('No selected states. Run latent_state_selection() first')
        selected_states = adata.uns['latent_dynamics']['posthoc_computations']['selected_states']
    else:
        selected_states = np.arange(adata.uns['latent_dynamics']['latent_dynamics_params']['num_states'])

    joint_probs = adata.uns['latent_dynamics']['model_outputs']['joint_probabilities']
    # Compute relevant conditionals
    observed_state_probs_mean = joint_probs.mean(0)/joint_probs.mean(0)[selected_states].sum()

    mask = np.ones(observed_state_probs_mean.shape[0], dtype=bool)
    mask[selected_states] = False
    observed_state_probs_mean[mask] = 0

    adata.uns['latent_dynamics']['conditional_probabilities']['state_given_nodes_selected'] = observed_state_probs_mean.sum(1) /\
                                                                                     observed_state_probs_mean.sum(1).sum(0)
    adata.uns['latent_dynamics']['conditional_probabilities']['chain_given_nodes_selected'] = observed_state_probs_mean.sum(0) /\
                                                                                     observed_state_probs_mean.sum(0).sum(0) 
    adata.uns['latent_dynamics']['conditional_probabilities']['chain_given_state_selected'] = (observed_state_probs_mean.sum(-1).T /\
                                                                                     observed_state_probs_mean.sum(-1).sum(1)).T
    adata.uns['latent_dynamics']['conditional_probabilities']['chain_state_given_nodes_selected'] = observed_state_probs_mean /\
                                                                                     observed_state_probs_mean.sum(0, keepdims=True).sum(1, keepdims=True)
    adata.uns['latent_dynamics']['conditional_probabilities']['node_given_chain_state_selected'] = observed_state_probs_mean /\
                                                                                     observed_state_probs_mean.sum(-1, keepdims=True)
    adata.uns['latent_dynamics']['conditional_probabilities']['node_given_state_selected'] = observed_state_probs_mean.sum(1) /\
                                                                                     observed_state_probs_mean.sum(1).sum(-1, keepdims=True)
    
# Select relevant latent states       
def latent_state_selection(adata, states=None, criteria='argmax_joint', min_ratio=0.01):

    # Check if model outputs exists
    try: assert adata.uns['latent_dynamics']['model_outputs']
    except: raise ValueError('Model outputs are missing. Run infer_latent_dynamics() first')

    # Restrict latent state space heuristically (alternative to computational expensive model selection process)    
    if criteria is None:
        selected_states = np.arange(adata.uns['latent_dynamics']['latent_dynamics_params']['num_states'])
    elif criteria == 'argmax_joint':
        selected_states = np.unique(adata.uns['latent_dynamics']['model_outputs']['joint_probabilities'].sum(-1).argmax(1)) 
    elif criteria == 'argmax_smoothing':
        selected_states = np.unique(adata.uns['latent_dynamics']['model_outputs']['log_smoothing'].argmax(1).flatten())
    elif criteria == 'viterbi':
        selected_states = np.unique(adata.uns['latent_dynamics']['model_outputs']['viterbi_path'].flatten())
    
    if states is not None:
        selected_states = np.array(states)


    adata.uns['latent_dynamics']['posthoc_computations'] = {}
    adata.uns['latent_dynamics']['latent_dynamics_params']['criteria'] = criteria
    adata.uns['latent_dynamics']['latent_dynamics_params']['min_ratio'] = min_ratio
    adata.uns['latent_dynamics']['posthoc_computations']['selected_states'] = selected_states
        
    compute_conditionals(adata, use_selected=True)

    # Filter out states with too few cells
    if min_ratio is not None:
        state_ratio = np.empty(adata.uns['latent_dynamics']['latent_dynamics_params']['num_states'])
        states, counts = np.unique(adata.uns['latent_dynamics']['conditional_probabilities']['state_given_nodes_selected'].argmax(0), 
                                   return_counts=True)
        # states, counts = np.unique(adata.uns['latent_dynamics']['model_params']['emission_matrix'].argmax(0), 
        #                            return_counts=True)
        
        for i, state in enumerate(states):
            state_ratio[state] = counts[i]
        state_ratio = state_ratio/adata.shape[0]

        state_ratio = state_ratio[selected_states]
        selected_states = selected_states[state_ratio>=min_ratio]

    if criteria is not None and min_ratio is None:
        if len(selected_states)==adata.uns['latent_dynamics']['latent_dynamics_params']['num_states']:
            logging.warning('Number of kinetic states equals num_states. Consider initialising with more latent states')

    adata.uns['latent_dynamics']['posthoc_computations']['selected_states'] = selected_states

    compute_conditionals(adata, use_selected=True)

# Extract kinetic states
def infer_kinetic_clusters(data, states=None, criteria=None, min_ratio=0.01, copy=False):

    adata = data.copy() if copy else data

    # Check if latent state history exists
    try: assert adata.uns['latent_dynamics']['model_outputs']
    except: raise ValueError('Latent states could not be recovered. Run infer_latent_dynamics() first')

    # Select latent states
    latent_state_selection(adata, states=states, criteria=criteria, min_ratio=min_ratio)

    # Extract kinetic states
    selected_states = adata.uns['latent_dynamics']['posthoc_computations']['selected_states']
    kinetic_states_probs = adata.uns['latent_dynamics']['conditional_probabilities']['state_given_nodes_selected']
                           
    adata.obs['kinetic_states'] = kinetic_states_probs.argmax(0).flatten()
    adata.obs['kinetic_states'] = adata.obs['kinetic_states'].astype('category')

    # Extract kinetic clustering - hierarchical argmax(state) -> lineage 
    # kinetic_clustering_probs = adata.uns['latent_dynamics']['conditional_probabilities']['chain_state_given_nodes_selected']
    # kinetic_clustering_probs = np.take_along_axis(kinetic_clustering_probs, 
    #                                               adata.obs.kinetic_states.values.astype(int).reshape(1, 1,-1), 
    #                                               axis=0)[0]
    
    # adata.obs['lineage_assignment'] = kinetic_clustering_probs.argmax(0).flatten()
    # adata.obs['lineage_assignment'] = adata.obs['lineage_assignment'].astype('category')

    lineage_probs = adata.uns['latent_dynamics']['conditional_probabilities']['chain_given_nodes_selected']
    adata.obs['lineage_assignment'] = lineage_probs.argmax(0).flatten()
    adata.obs['lineage_assignment'] = adata.obs['lineage_assignment'].astype('category')

    adata.obs['kinetic_clustering'] = adata.obs.kinetic_states.astype(str) + '_' + \
                                      adata.obs.lineage_assignment.astype(str)
    adata.obs['kinetic_clustering'] = adata.obs['kinetic_clustering'].astype('category')

    # Transitional entropy
    adata.obs['transitional_entropy'] = entropy(adata.uns['latent_dynamics']['conditional_probabilities']['state_given_nodes_selected'])
    
    # Differentiation entropy
    adata.obs['differentiation_entropy'] = entropy(adata.uns['latent_dynamics']['conditional_probabilities']['chain_given_nodes_selected'])
    
    if copy: return adata

# Infer cellfate
def infer_lineage_probabilities(data, use_selected=True, copy=False):

    adata = data.copy() if copy else data

    if use_selected:
        try: assert 'chain_given_nodes_selected' in adata.uns['latent_dynamics']['conditional_probabilities'].keys()
        except: raise ValueError('Selected state conditional not found. Run compute_conditional(use_selected=True) first.')

        lineage_probs = adata.uns['latent_dynamics']['conditional_probabilities']['chain_given_nodes_selected']
        lineage_probs = lineage_probs/lineage_probs.sum(0)
    else:
        selected_states = np.arange(adata.uns['latent_dynamics']['latent_dynamics_params']['num_states'])
        lineage_probs = adata.uns['latent_dynamics']['conditional_probabilities']['chain_given_nodes']

    adata.uns['latent_dynamics']['posthoc_computations']['lineage_probs'] = lineage_probs

    if copy: return adata

# Infer most likely path in latent space
def infer_latent_paths(data, criteria='argmax_joint', copy=False):

    adata = data.copy() if copy else data

    # Check if latent state history exists
    try: assert adata.uns['latent_dynamics']['model_outputs']
    except: raise ValueError('Latent states could not be recovered. Run infer_latent_dynamics() first')

    # Infer lineages
    selected_states = adata.uns['latent_dynamics']['posthoc_computations']['selected_states']
    selected_states_dict = dict(zip(np.arange(selected_states.shape[0]),
                                    selected_states
                                    ))

    # Assign states to lineages 
    if criteria is None or criteria=='viterbi':
        lineage_paths = adata.uns['latent_dynamics']['model_outputs']['viterbi_path']
    elif criteria=='argmax_smoothing':
        lineage_paths = adata.uns['latent_dynamics']['model_outputs']['log_smoothing'][:, selected_states].argmax(1)
        
        indices = {}
        for state in np.arange(selected_states.shape[0]):
            indices[state] = np.where(lineage_paths==state)
        for state in np.arange(selected_states.shape[0]):
            lineage_paths[indices[state]] = selected_states_dict[state]

    elif criteria=='argmax_joint':
        lineage_paths = adata.uns['latent_dynamics']['model_outputs']['joint_probabilities'].sum(-1)[:, selected_states].argmax(1)

        indices = {}
        for state in np.arange(selected_states.shape[0]):
            indices[state] = np.where(lineage_paths==state)
        for state in np.arange(selected_states.shape[0]):
            lineage_paths[indices[state]] = selected_states_dict[state]

    condensed_lineage_paths = {}
    for lineage in range(adata.uns['latent_dynamics']['latent_dynamics_params']['num_chains']):
        condensed_lineage_paths[lineage] = list(dict.fromkeys(lineage_paths[:, lineage]))

    adata.uns['latent_dynamics']['posthoc_computations']['latent_paths'] = lineage_paths
    adata.uns['latent_dynamics']['posthoc_computations']['condensed_latent_paths'] = condensed_lineage_paths   

    if copy: return adata
    




