import torch
import numpy as np

import logging
logging.basicConfig(level = logging.INFO)

from .utils import check_TPM
from .models.IFHMM import IFHMM
from .models.SSM import SSM

# Fit the latent dynamic model
def infer_latent_dynamics(data, model=None, num_states=10, num_chains=1, num_epochs=100, 
                          mode='SSM', regularise_TPM=False, use_gpu=False, verbose=False,
                          copy=False, **kwargs):

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
        elif mode=='IFHMM':
            model = IFHMM(num_states, num_chains, adata.shape[0], state_history.shape[0], use_gpu=use_gpu)
    # Train the model
    loss = model.train(state_history, TPM=TPM, num_epochs=num_epochs, verbose=verbose, **kwargs)

    # Extract and store model params and inference
    log_observed_state_probs, log_observed_state_probs_, log_hidden_state_probs = model.forward_model()

    adata.uns['latent_dynamics'] = {}
    adata.uns['latent_dynamics']['latent_state_history'] = torch.exp(log_hidden_state_probs).cpu().detach().numpy()
    adata.uns['latent_dynamics']['predicted_state_history'] = torch.exp(log_observed_state_probs).cpu().detach().numpy()
    adata.uns['latent_dynamics']['full_predicted_state_history'] = torch.exp(log_observed_state_probs_).cpu().detach().numpy()

    adata.uns['latent_dynamics']['chain_weights'] = torch.nn.functional.softmax(model.unnormalized_chain_weights, dim=-1).cpu().detach().numpy()
    adata.uns['latent_dynamics']['emission_matrix'] = torch.nn.functional.softmax(model.unnormalized_emission_matrix, dim=-1).cpu().detach().numpy()
    adata.uns['latent_dynamics']['latent_transition_matrix'] = torch.nn.functional.softmax(model.unnormalized_transition_matrix, dim=-1).cpu().detach().numpy()

    adata.uns['latent_dynamics']['latent_dynamics_params'] = locals()

    if copy: return model, adata
    else: return model

# Extract kinetic states
def infer_kinetic_clusters(data, copy=False):

    adata = data.copy() if copy else data

    # Check if latent state history exists
    try: assert adata.uns['latent_dynamics']
    except: raise ValueError('Latent state probability history could not be recovered. Run infer_latent_dynamics() first')

    # Extract kinetic clustering
    selected_states = np.unique(adata.uns['latent_dynamics']['full_predicted_state_history'].sum(-1).argmax(1).flatten())

    adata.obs['kinetic_states'] = adata.uns['latent_dynamics']['emission_matrix'][selected_states].argmax(0).astype(str).flatten()
    adata.obs['kinetic_states'] = adata.obs['kinetic_states'].astype('category')
    adata.obs['kinetic_states'] = adata.obs['kinetic_states'].cat.rename_categories(selected_states)

    if selected_states.shape[0]==adata.uns['latent_dynamics']['latent_dynamics_params']['num_states']:
        logging.warning('Number of kinetic states equals num_states. Consider initialising with more latent states')
    
    if copy: return adata

# Infer most likely path in latent space
def infer_latent_path():
    raise NotImplementedError()
    
# Infer cell fate probabilities
def infer_cell_fate():
    raise NotImplementedError()



