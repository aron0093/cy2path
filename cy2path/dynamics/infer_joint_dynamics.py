import os
import torch

import logging
logging.basicConfig(level = logging.INFO)

from .infer_dynamics import infer_dynamics

# Fit the latent dynamic model
def infer_joint_dynamics(*data, models=None, num_states=10, num_chains=1, num_epochs=500, num_epoch_bins=50, mode='FHMM', 
                         restricted=True, use_gpu=False, verbose=False, precomputed_emissions=None, precomputed_transitions=None,
                         emissions_grad=False, transitions_grad=False, save_models='./models/', load_models=None, copy=False, **kwargs):
    
    ''' 
    Infer dynamics in latent state space.
    
    Parameters
    ----------
    data: list of AnnData objects
        AnnData objects containing MSM simulation. 
    models: optional (default: None)
        list of initialised model classes. Overrides other parameters.
    num_states : int
        Number of hidden states.
    num_chains : int
        Number of hidden Markov chains.
    num_epochs : int
        Number of training epochs.
    num_epoch_bins: int
        Number of epochs to train before switching models.
    mode: {'FHMM', 'LSSM', 'NSSM'} (default: 'FHMM')
        Flavor of latent dynamic model to use.
    restricted: Bool (default: True)
        Set to True for a common latent state space between lineages.
    use_gpu : Bool (default: False)
        Toggle GPU use.
    verbose: Bool (default: False)
        Verbose training report.
    precomputed_emissions: optional (default: None)
        list of precomputed emission matrix parameters.
    precomputed_transitions: optional (default: None)
        Use precomputed latent transition matrix parameters.
    emissions_grad: Bool (default: False)
        Fixed emission matrix parameters (no grad).
    transitions_grad: Bool (default: False)
        Fixed transition matrix parameters (no grad).
    save_models: str (default: './model')
        Path to save trained models.
    load_models: optional (default: None)
        Path to load trained models.
    copy: Bool (default: False)
        Save output inplace or return a copy.

    Return: Returns or updates AnnData object.
    '''

    adatas = [adata.copy() if copy else adata for adata in data]

    # Use pre-computed stuff if supplied
    if models is None:
        models = [None]*len(adatas) 

    if load_models is None:
        load_models = [None]*len(adatas)
    
    if save_models is None:
        save_models = [None]*len(adatas)
    elif isinstance(save_models, str):
        os.makedirs(save_models, exist_ok=True)
        save_models_ = [os.path.join(save_models, 'model_{}'.format(num)) \
                       for num in range(len(adatas))]
            
    if precomputed_emissions is None:
        precomputed_emissions = [None]*len(adatas)

    if isinstance(emissions_grad, bool):
        emissions_grad = [emissions_grad]*len(adatas)
    
    if isinstance(transitions_grad, bool):
        transitions_grad = [transitions_grad]*len(adatas)
    
    # Set transitions_grad to True if precomputed_transitions not supplied
    # Default behavior in infer_dynamics, grad is always True if not precomputed
    # grad is False (default) if precomputed unless supplied True
    if precomputed_transitions is None:
        if mode=='FHMM':
            unnormalized_transition_matrix = torch.nn.Parameter(torch.randn(num_chains,
                                                                     num_states,
                                                                     num_states
                                                                    ))
            precomputed_transitions = torch.nn.functional.softmax(unnormalized_transition_matrix, 
                                                                  dim=-1)

        else:
            unnormalized_transition_matrix = torch.nn.Parameter(torch.randn(num_states,
                                                                     num_states
                                                                    ))
            precomputed_transitions = torch.nn.functional.softmax(unnormalized_transition_matrix, 
                                                                  dim=-1)
            
        transitions_grad = [True]*len(adatas)

    # Train models jointly
    cur_bin=0
    while cur_bin < num_epoch_bins:
        for num, adata in enumerate(adatas):
            
            logging.info('Training data {} in round {}.'.format(num, cur_bin))

            if copy:
                model, adata = infer_dynamics(adata, model=models[num], num_states=num_states, 
                                            num_chains=num_chains, num_epochs=int(num_epochs/num_epoch_bins), 
                                            mode=mode, restricted=restricted, 
                                            use_gpu=use_gpu, verbose=verbose, 
                                            precomputed_emissions=precomputed_emissions[num], 
                                            precomputed_transitions=precomputed_transitions, 
                                            emissions_grad=emissions_grad[num], transitions_grad=transitions_grad[num],
                                            save_model=save_models_[num], load_model=load_models[num], 
                                            copy=copy, **kwargs)
                adatas[num] = adata
            else:
                model = infer_dynamics(adata, model=models[num], num_states=num_states, 
                                    num_chains=num_chains, num_epochs=int(num_epochs/num_epoch_bins), 
                                    mode=mode, restricted=restricted, 
                                    use_gpu=use_gpu, verbose=verbose, 
                                    precomputed_emissions=precomputed_emissions[num], 
                                    precomputed_transitions=precomputed_transitions, 
                                    emissions_grad=emissions_grad[num], transitions_grad=transitions_grad[num],
                                    save_model=save_models_[num], load_model=load_models[num], 
                                    copy=copy, **kwargs)
            models[num] = model

        # Update bin
        cur_bin+=1

    if copy: return models, adatas
    else: return models
    
 

