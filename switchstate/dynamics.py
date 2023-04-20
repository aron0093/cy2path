import torch
import numpy as np
from scipy.stats import entropy

import logging
logging.basicConfig(level = logging.INFO)

from .utils import check_TPM, log_domain_mean, exponentiate_detach
from .models import IFHMM, SSM, SSM_node

from .models.methods import compute_aic

# Function to store model outputs in anndata
def extract_model_outputs(adata, model):

    # Extract and store model params and inference
    log_observed_state_probs, log_observed_state_probs_, log_hidden_state_probs = model.forward_model()

    # MSM simulation 
    state_history = torch.Tensor(adata.uns['state_probability_sampling']['state_history'])

    log_alpha, log_probs = model.filtering(state_history)
    log_beta, log_gamma = model.smoothing(state_history)
    log_delta, psi, log_max, best_path = model.viterbi(state_history)

    # Model outputs
    adata.uns['latent_dynamics'] = {}
    adata.uns['latent_dynamics']['model_outputs'] = {}

    adata.uns['latent_dynamics']['model_outputs']['latent_state_history'] = exponentiate_detach(log_hidden_state_probs)
    adata.uns['latent_dynamics']['model_outputs']['predicted_state_history'] = exponentiate_detach(log_observed_state_probs)
    adata.uns['latent_dynamics']['model_outputs']['joint_probabilities'] = exponentiate_detach(log_observed_state_probs_)

    adata.uns['latent_dynamics']['model_outputs']['filtering'] = exponentiate_detach(log_alpha)
    adata.uns['latent_dynamics']['model_outputs']['smoothing'] = exponentiate_detach(log_gamma)
    adata.uns['latent_dynamics']['model_outputs']['viterbi'] = exponentiate_detach(log_delta)
    adata.uns['latent_dynamics']['model_outputs']['viterbi_path'] = np.array(best_path).astype(int).T


    # Model params
    adata.uns['latent_dynamics']['model_params'] = {}
    adata.uns['latent_dynamics']['model_params']['chain_weights'] = torch.nn.functional.softmax(model.unnormalized_chain_weights, 
                                                                                dim=-1).cpu().detach().numpy()
    adata.uns['latent_dynamics']['model_params']['emission_matrix'] = torch.nn.functional.softmax(model.unnormalized_emission_matrix, 
                                                                                  dim=-1).cpu().detach().numpy()
    adata.uns['latent_dynamics']['model_params']['latent_transition_matrix'] = torch.nn.functional.softmax(model.unnormalized_transition_matrix, 
                                                                                           dim=-1).cpu().detach().numpy()
    
    # Compute relevant conditionals
    log_observed_state_probs_mean = log_domain_mean(log_observed_state_probs_)

    adata.uns['latent_dynamics']['conditional_probabilities'] = {}
    adata.uns['latent_dynamics']['conditional_probabilities']['state_given_nodes'] = exponentiate_detach(log_observed_state_probs_mean.logsumexp(1) -\
                                                                                     log_observed_state_probs_mean.logsumexp(1).logsumexp(0, keepdims=True))
    adata.uns['latent_dynamics']['conditional_probabilities']['chain_given_nodes'] = exponentiate_detach(log_observed_state_probs_mean.logsumexp(0) -\
                                                                                     log_observed_state_probs_mean.logsumexp(0).logsumexp(0, keepdims=True))
    adata.uns['latent_dynamics']['conditional_probabilities']['chain_given_state'] = exponentiate_detach(log_observed_state_probs_mean.logsumexp(-1) -\
                                                                                     log_observed_state_probs_mean.logsumexp(-1).logsumexp(1, keepdims=True))
    
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
                                                                                     observed_state_probs_mean.sum(0).sum()
    adata.uns['latent_dynamics']['conditional_probabilities']['chain_given_state_selected'] = (observed_state_probs_mean.sum(-1).T /\
                                                                                     observed_state_probs_mean.sum(-1).sum(1)).T

# Select relevant latent states       
def latent_state_selection(adata, states=None, criteria=None, min_ratio=None):

    # Check if model outputs exists
    try: assert adata.uns['latent_dynamics']['model_outputs']
    except: raise ValueError('Model outputs are missing. Run infer_latent_dynamics() first')

    # Restrict latent state space heuristically (alternative to computational expensive model selection process)    
    if criteria is None:
        selected_states = np.arange(adata.uns['latent_dynamics']['latent_dynamics_params']['num_states'])
    elif criteria == 'argmax_joint':
        selected_states = np.unique(adata.uns['latent_dynamics']['model_outputs']['joint_probabilities'].sum(-1).argmax(1)) 
    elif criteria == 'argmax_smoothing':
        selected_states = np.unique(adata.uns['latent_dynamics']['model_outputs']['smoothing'].argmax(1).flatten())
    elif criteria == 'viterbi':
        selected_states = np.unique(adata.uns['latent_dynamics']['model_outputs']['viterbi_path'].flatten())
    
    if states is not None:
        selected_states = np.array(states)

    # Filter out states with too few cells
    if min_ratio is not None:
        state_ratio = adata.uns['latent_dynamics']['conditional_probabilities']['state_given_nodes'].sum(1)/adata.shape[0]
        state_ratio = state_ratio[selected_states]
        selected_states = selected_states[state_ratio>=min_ratio]

    if criteria is not None and min_ratio is None:
        if len(selected_states)==adata.uns['latent_dynamics']['latent_dynamics_params']['num_states']:
            logging.warning('Number of kinetic states equals num_states. Consider initialising with more latent states')

    adata.uns['latent_dynamics']['posthoc_computations'] = {}
    adata.uns['latent_dynamics']['latent_dynamics_params']['criteria'] = criteria
    adata.uns['latent_dynamics']['latent_dynamics_params']['min_ratio'] = min_ratio
    adata.uns['latent_dynamics']['posthoc_computations']['selected_states'] = selected_states

    compute_conditionals(adata, use_selected=True)

# Fit the latent dynamic model
def infer_latent_dynamics(data, model=None, num_states=10, num_chains=1, num_epochs=100, 
                          mode='SSM', regularise_TPM=True, use_gpu=False, verbose=False,
                          precomputed_emissions=None, precomputed_transitions=None,
                          save_model='./model', load_model=None, copy=False, **kwargs):

    adata = data.copy() if copy else data

    # Check if state history exists
    try: state_history = adata.uns['state_probability_sampling']['state_history']
    except: raise ValueError('State probability history could not be recovered. Run sample_state_probability() first')

    if regularise_TPM:
        check_TPM(adata)
        TPM = torch.Tensor(adata.obsp['T_forward'].toarray())
    else:
        TPM = None

    # MSM simulation 
    state_history = torch.Tensor(adata.uns['state_probability_sampling']['state_history'])

    # Initialise the model
    if not model:
        if mode=='SSM':
            model = SSM(num_states, num_chains, adata.shape[0], state_history.shape[0], use_gpu=use_gpu)
        elif mode=='SSM_granular':
            model = SSM_node(num_states, num_chains, adata.shape[0], state_history.shape[0], use_gpu=use_gpu)
        elif mode=='IFHMM':
            model = IFHMM(num_states, num_chains, adata.shape[0], state_history.shape[0], use_gpu=use_gpu)
    else:
        num_states = model.num_states
        num_chains = model.num_chains

    if load_model is not None:
        model.load_state_dict(load_model)

    if precomputed_emissions:
        assert precomputed_emissions.shape==(model.num_states,1,model.num_nodes)
        model.unnormalized_emission_matrix = torch.log(precomputed_emissions)
    
    if precomputed_transitions:
        assert precomputed_transitions.shape==(model.num_states,model.num_states)
        model.unnormalized_transition_matrix = torch.log(precomputed_transitions)

    # Train the model
    loss = model.train(state_history, TPM=TPM, num_epochs=num_epochs, verbose=verbose, **kwargs)
    
    # Model outputs
    extract_model_outputs(adata, model)
    
    # Compute Likelihood P(Data/Model)
    compute_aic(model)

    adata.uns['latent_dynamics']['latent_dynamics_params'] = locals()
    adata.uns['latent_dynamics']['aic'] = model.aic

    if save_model is not None:
        torch.save(model.state_dict(), save_model)

    if copy: return model, adata
    else: return model

# Extract kinetic states
def infer_kinetic_clusters(data, states=None, criteria=None, min_ratio=None, copy=False):

    adata = data.copy() if copy else data

    # Check if latent state history exists
    try: assert adata.uns['latent_dynamics']['model_outputs']
    except: raise ValueError('Latent states could not be recovered. Run infer_latent_dynamics() first')

    # Select latent states
    latent_state_selection(adata, states=states, criteria=criteria, min_ratio=min_ratio)

    # Extract kinetic clustering and other params
    selected_states = adata.uns['latent_dynamics']['posthoc_computations']['selected_states']
    kinetic_states_probs = adata.uns['latent_dynamics']['conditional_probabilities']['state_given_nodes_selected']#[selected_states]
    
    adata.obs['kinetic_states'] = kinetic_states_probs.argmax(0).flatten()#selected_states[kinetic_states_probs.argmax(0).flatten()]
    adata.obs['kinetic_states'] = adata.obs['kinetic_states'].astype('category')

    #TODO: Compute with renormed probabilities
    adata.obs['cellular_entropy'] = entropy(adata.uns['latent_dynamics']['model_outputs']['joint_probabilities'][:, selected_states
                                                                                                                 ].mean(0)).sum(0)
    
    if copy: return adata

# Infer most likely path in latent space
def infer_latent_paths(data, states=None, criteria=None, min_ratio=None, copy=False):

    adata = data.copy() if copy else data

    # Check if latent state history exists
    try: assert adata.uns['latent_dynamics']['model_outputs']
    except: raise ValueError('Latent states could not be recovered. Run infer_latent_dynamics() first')

    # Select latent states
    latent_state_selection(adata, states=states, criteria=criteria, min_ratio=min_ratio)

    # Infer lineages
    #selected_states = adata.uns['latent_dynamics']['posthoc_computations']['selected_states']
    lineage_probs = np.matmul(adata.uns['latent_dynamics']['conditional_probabilities']['chain_given_state_selected'].T, #['selected]
                              adata.uns['latent_dynamics']['conditional_probabilities']['state_given_nodes_selected'])

    adata.obs['lineage'] = lineage_probs.argmax(0).astype(str).flatten()

    # Assign states to lineages 
    # #TODO: selected_states
    if criteria is None or criteria=='viterbi':
        lineage_paths = adata.uns['latent_dynamics']['model_outputs']['viterbi_path']
    elif criteria=='argmax_smoothing':
        lineage_paths = adata.uns['latent_dynamics']['model_outputs']['smoothing'].argmax(1)
    elif criteria=='argmax_joint':
        lineage_paths = adata.uns['latent_dynamics']['model_outputs']['joint_probabilities'].sum(-1).argmax(1)

    condensed_lineage_paths = {}
    for lineage in range(adata.uns['latent_dynamics']['latent_dynamics_params']['num_chains']):
        condensed_lineage_paths[lineage] = list(dict.fromkeys(lineage_paths[:, lineage]))

    adata.uns['latent_dynamics']['posthoc_computations']['lineage_probs'] = lineage_probs
    adata.uns['latent_dynamics']['posthoc_computations']['latent_paths'] = lineage_paths
    adata.uns['latent_dynamics']['posthoc_computations']['condensed_latent_paths'] = condensed_lineage_paths

    if copy: return adata
    




