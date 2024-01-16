import torch
import numpy as np
from scipy.stats import entropy

import logging
logging.basicConfig(level = logging.INFO)

from ..utils import check_TPM, log_domain_mean, exponentiate_detach
from ..models import FHMM, LSSM, NSSM

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
    adata.uns['latent_dynamics']['model_outputs'] = {}

    adata.uns['latent_dynamics']['model_outputs']['latent_state_history'] = exponentiate_detach(log_hidden_state_probs)
    adata.uns['latent_dynamics']['model_outputs']['predicted_state_history'] = exponentiate_detach(log_observed_state_probs)
    adata.uns['latent_dynamics']['model_outputs']['joint_probabilities'] = exponentiate_detach(log_observed_state_probs_)

    adata.uns['latent_dynamics']['model_outputs']['log_filtering'] = log_alpha.detach().cpu().numpy()
    adata.uns['latent_dynamics']['model_outputs']['log_smoothing'] = log_gamma.detach().cpu().numpy()
    adata.uns['latent_dynamics']['model_outputs']['log_viterbi'] = log_delta.detach().cpu().numpy()
    adata.uns['latent_dynamics']['model_outputs']['viterbi_path'] = np.array(best_path).astype(int).T

    # Model params
    adata.uns['latent_dynamics']['model_params'] = {}
    adata.uns['latent_dynamics']['model_params']['chain_weights'] = exponentiate_detach(model.log_chain_weights)
    adata.uns['latent_dynamics']['model_params']['emission_matrix'] = exponentiate_detach(model.log_emission_matrix)
    adata.uns['latent_dynamics']['model_params']['latent_transition_matrix'] = exponentiate_detach(model.log_transition_matrix)
    
    # TODO: Unify output for all model types
    if adata.uns['latent_dynamics']['latent_dynamics_params']['mode'] == 'FHMM':
        adata.uns['latent_dynamics']['model_params']['conditional_latent_transition_matrix'] = \
        adata.uns['latent_dynamics']['model_params']['latent_transition_matrix']
        adata.uns['latent_dynamics']['model_params']['latent_transition_matrix'] *=  adata.uns['latent_dynamics']['model_params']['chain_weights'][:, None, None]
        adata.uns['latent_dynamics']['model_params']['latent_transition_matrix'] = \
        adata.uns['latent_dynamics']['model_params']['latent_transition_matrix'].sum(0)

    # Compute relevant conditionals
    log_observed_state_probs_mean = log_domain_mean(log_observed_state_probs_, use_gpu=model.is_cuda)

    adata.uns['latent_dynamics']['conditional_probabilities'] = {}
    adata.uns['latent_dynamics']['conditional_probabilities']['state_given_nodes'] = exponentiate_detach(log_observed_state_probs_mean.logsumexp(1) -\
                                                                                     log_observed_state_probs_mean.logsumexp(1).logsumexp(0, keepdims=True))
    adata.uns['latent_dynamics']['conditional_probabilities']['chain_given_nodes'] = exponentiate_detach(log_observed_state_probs_mean.logsumexp(0) -\
                                                                                     log_observed_state_probs_mean.logsumexp(0).logsumexp(0, keepdims=True))
    adata.uns['latent_dynamics']['conditional_probabilities']['chain_given_state'] = exponentiate_detach(log_observed_state_probs_mean.logsumexp(-1) -\
                                                                                     log_observed_state_probs_mean.logsumexp(-1).logsumexp(1, keepdims=True))
    adata.uns['latent_dynamics']['conditional_probabilities']['chain_state_given_nodes'] = exponentiate_detach(log_observed_state_probs_mean -\
                                                                                     log_observed_state_probs_mean.logsumexp(0, keepdims=True).logsumexp(1, keepdims=True))
    adata.uns['latent_dynamics']['conditional_probabilities']['node_given_chain_state'] = exponentiate_detach(log_observed_state_probs_mean -\
                                                                                     log_observed_state_probs_mean.logsumexp(-1, keepdims=True))
    adata.uns['latent_dynamics']['conditional_probabilities']['node_given_state'] = exponentiate_detach(log_observed_state_probs_mean.logsumexp(1) -\
                                                                                     log_observed_state_probs_mean.logsumexp(1).logsumexp(-1, keepdims=True))

# Fit the latent dynamic model
def infer_dynamics(data, model=None, num_states=10, num_chains=1, num_epochs=500, 
                   mode='FHMM', restricted=True, use_gpu=False, verbose=False, 
                   precomputed_emissions=None, precomputed_transitions=None,
                   emissions_grad=False, transitions_grad=False,
                   save_model='./model', load_model=None, 
                   copy=False, **kwargs):
    
    ''' 
    Infer dynamics in latent state space.
    
    Parameters
    ----------
    data: AnnData
        AnnData object containing MSM simulation. 
    model: optional (default: None)
        Use initialised model class. Overrides other parameters.
    num_states : int
        Number of hidden states.
    num_chains : int
        Number of hidden Markov chains.
    num_epochs : int
        Number of training epochs.
    mode: {'FHMM', 'LSSM', 'NSSM'} (default: 'FHMM')
        Flavor of latent dynamic model to use.
    restricted: Bool (default: True)
        Set to True for a common latent state space between lineages.
    use_gpu : Bool (default: False)
        Toggle GPU use.
    verbose: Bool (default: False)
        Verbose training report.
    precomputed_emissions: optional (default: None)
        Use precomputed emission matrix parameters.
    precomputed_transitions: optional (default: None)
        Use precomputed latent transition matrix parameters.
    emissions_grad: Bool (default: False)
        Fixed emission matrix parameters (no grad).
    transitions_grad: Bool (default: False)
        Fixed transition matrix parameters (no grad).
    save_model: str (default: './model')
        Path to save trained model.
    load_model: optional (default: None)
        Path to load trained model.
    copy: Bool (default: False)
        Save output inplace or return a copy.

    Return: Returns or updates AnnData object.
    '''

    adata = data.copy() if copy else data

    # Don't save entire anndata in params
    params_ = locals()
    del params_['data']
    del params_['adata']

    # Check if state history exists
    try: state_history = adata.uns['state_probability_sampling']['state_history']
    except: raise ValueError('State probability history could not be recovered. Run sample_state_probability() first')

    # Convert TPM to tensor
    check_TPM(adata)
    TPM = torch.Tensor(adata.obsp['T_forward'].toarray())

    # MSM simulation 
    state_history = torch.Tensor(adata.uns['state_probability_sampling']['state_history'])
    if use_gpu:
        state_history = state_history.cuda()

    # Initialise the model
    if model is None and load_model is None:          
        if mode=='LSSM':
            model = LSSM(num_states, num_chains, adata.shape[0], state_history.shape[0], restricted=restricted, use_gpu=use_gpu)
        elif mode=='NSSM':
            model = NSSM(num_states, num_chains, adata.shape[0], state_history.shape[0], restricted=restricted, use_gpu=use_gpu)
        elif mode=='FHMM':
            model = FHMM(num_states, num_chains, adata.shape[0], state_history.shape[0], restricted=restricted, use_gpu=use_gpu)
    elif model is None and load_model is not None:
        model.load_state_dict(load_model)
    else:
        num_states = model.num_states
        num_chains = model.num_chains

    # TODO: If probabilities are passed don't use softmax
    if precomputed_emissions is not None:
        model.unnormalized_emission_matrix = torch.nn.Parameter(torch.log(precomputed_emissions))
        model.unnormalized_emission_matrix.requires_grad_(emissions_grad)
    
    if precomputed_transitions is not None:
        model.unnormalized_transition_matrix = torch.nn.Parameter(torch.log(precomputed_transitions))
        model.unnormalized_transition_matrix.requires_grad_(transitions_grad)
   
    if use_gpu:
        model.cuda()

    # Train the model
    loss = model.train(state_history, TPM=TPM, num_epochs=num_epochs, verbose=verbose, **kwargs)
    
    # Save params
    adata.uns['latent_dynamics'] = {}
    adata.uns['latent_dynamics']['latent_dynamics_params'] = params_

    # Model outputs
    extract_model_outputs(adata, model)

    # Compute Likelihood P(Data/Model)
    adata.uns['latent_dynamics']['log_likelihood'] = model.log_likelihood.item()

    if save_model is not None:
        torch.save(model.state_dict(), save_model)

    if copy: return model, adata
    else: return model