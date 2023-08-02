import gc
from itertools import product

import torch
import numpy as np
import pandas as pd

from scipy.stats import median_test, mannwhitneyu, wilcoxon, ttest_ind

from .dynamics import *

# Run grid search to identify best model parameters
def run_search(adata, regularise_TPM=False, use_gpu=True, verbose=False,
               param_list={'num_states': None, 'num_chains': None, 
                           'num_epochs': None, 'sparsity_weight': None, 
                           'exclusivity_weight': None, 'orthogonality_weight': None,
                           'mode': None}, 
               outfil='gridsearch', **kwargs):
        
    # Construct return df
    fit_values = pd.DataFrame(product(*(param_list[key] for key in param_list)), columns=list(param_list.keys()))
    fit_values[['avg_corrcoef', 'loss', 'kl_divergence', 'log_likelihood', 'exclusivity', 'orthogonality',
                'ratio_1perc_states', 'ratio_5perc_states', 'viterbi_states', 'argmax_states']] = None
    
    # Run grid search
    for idx in fit_values.index.values:

        # Parameters to pass at model initialisation
        init_params = {'num_states': None, 'num_chains': None}
        if 'num_states' in fit_values.columns:
            init_params['num_states'] = fit_values.loc[idx, 'num_states']
        if 'num_chains' in fit_values.columns:
            init_params['num_chains'] = fit_values.loc[idx, 'num_chains']
        if 'mode' in fit_values.columns:
            init_params['mode'] = fit_values.loc[idx, 'mode']

        # Parameters to pass at model training
        train_params = {'num_epochs': None,
                        'sparsity_weight': None, 
                        'exclusivity_weight': None,
                        'orthogonality_weight': None}
        
        if 'num_epochs' in fit_values.columns:
            train_params['num_epochs'] = fit_values.loc[idx, 'num_epochs']
        if 'sparsity_weight' in fit_values.columns:
            train_params['sparsity_weight'] = fit_values.loc[idx, 'sparsity_weight']
        if 'exclusivity_weight' in fit_values.columns:
            train_params['exclusivity_weight'] = fit_values.loc[idx, 'exclusivity_weight']
        if 'orthogonality_weight' in fit_values.columns:
            train_params['orthogonality_weight'] = fit_values.loc[idx, 'orthogonality_weight']

        model, adata_ = infer_latent_dynamics(adata, 
                                              num_states=fit_values.loc[idx, 'num_states'],
                                              num_chains=fit_values.loc[idx, 'num_chains'],
                                              num_epochs=fit_values.loc[idx, 'num_epochs'],
                                              mode=fit_values.loc[idx, 'mode'],
                                              sparsity_weight=fit_values.loc[idx, 'sparsity_weight'],
                                              orthogonality_weight=fit_values.loc[idx, 'orthogonality_weight'],
                                              exclusivity_weight=fit_values.loc[idx, 'exclusivity_weight'],
                                              regularise_TPM=regularise_TPM, 
                                              save_model=None,
                                              use_gpu=use_gpu, verbose=verbose, copy=True, **kwargs)
        
        fit_values.loc[idx, 'avg_corrcoef'] = model.avg_corrcoeff
        fit_values.loc[idx, 'loss'] = model.loss_values[-1]
        fit_values.loc[idx, 'kl_divergence'] = model.divergence_values[-1]
        fit_values.loc[idx, 'log_likelihood'] = model.log_likelihood.item()
        fit_values.loc[idx, 'exclusivity'] = model.exclusivity_values[-1]
        fit_values.loc[idx, 'orthogonality'] = model.orthogonality_values[-1]
        
        state_ratio = np.empty(adata_.uns['latent_dynamics']['latent_dynamics_params']['num_states'])
        states, counts = np.unique(adata_.uns['latent_dynamics']['conditional_probabilities']['state_given_nodes'].argmax(0), 
                                   return_counts=True)

        for i, state in enumerate(states):
            state_ratio[state] = counts[i]
        state_ratio = state_ratio/adata_.shape[0]
        fit_values.loc[idx, 'ratio_1perc_states'] = sum(state_ratio >= 0.01)
        fit_values.loc[idx, 'ratio_1perc_states'] = sum(state_ratio >= 0.05)
        
        infer_kinetic_clusters(adata_, states=None, min_ratio=0.0, criteria='argmax_joint')
        fit_values.loc[idx, 'argmax_states'] = len(adata_.obs.kinetic_states.unique())
        
        infer_kinetic_clusters(adata_, states=None, min_ratio=0.0, criteria='viterbi')
        fit_values.loc[idx, 'viterbi_states'] = len(adata_.obs.kinetic_states.unique())
        
        del model
        model=None
        gc.collect()
        torch.cuda.empty_cache()

        fit_values.to_csv('{}.txt'.format(outfil), sep='\t', index=False) 

    return fit_values

# Use significance tests between successive groups to identify elbow
def estimate_optimial_parameters(fit_values, param='num_states'):

    # p_vals = []
    # for i in range(2, 11):
    #     p_vals.append(wilcoxon(fit_values.loc[loss_df.num_states==i, 'corr'].values,
    #                 fit_values.loc[fit_values.num_states==i+1, 'corr'].values,
    #                 nan_policy='omit')[1])
    # p_vals
    raise NotImplementedError()

        
        
