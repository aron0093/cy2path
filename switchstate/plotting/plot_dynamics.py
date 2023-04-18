import numpy as np
import pandas as pd

import seaborn as sns
from scvelo.plotting import scatter
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm

from ..pseudotime import infer_pseudotime
from .plot_cluster_probabilities import plot_probability_by_clusters

# Plot model loss
def plot_loss(model, figsize=(15,7)):

    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=figsize)

    sns.lineplot(model.loss_values, ax=axs.flat[0])
    axs.flat[0].set_title('Loss')

    sns.lineplot(model.divergence_values, ax=axs.flat[1])
    axs.flat[1].set_title('KL divergence')

    sns.lineplot(model.likelihood_values, ax=axs.flat[2])
    axs.flat[2].set_title('Log likelihood')

    sns.lineplot(model.sparsity_values, ax=axs.flat[3])
    axs.flat[3].set_title('Sparsity regularisation')

    if model.regularisation_values != []:
        sns.lineplot(model.regularisation_values, ax=axs.flat[4])
        axs.flat[4].set_title('TPM regularisation')
    else:
        axs.flat[4].axis('off')

    sns.lineplot(model.exclusivity_values, ax=axs.flat[5])
    axs.flat[5].set_title('Kinetic state exclusivity')

    return fig, axs

# Dynamics report - summary 

# Heatmap with latent state transition matrix
def plot_latent_transition_matrix(adata, states=None, ax=None):

    # Check if latent state TPM exists
    try: assert adata.uns['latent_dynamics']['model_params']
    except: raise ValueError('Latent state TPM not found. Run infer_latent_dynamics() first')

    latent_transition_matrix = adata.uns['latent_dynamics']['model_params']['latent_transition_matrix']
    if states is None:
        states = np.arange(latent_transition_matrix.shape[0])

    latent_transition_matrix = latent_transition_matrix[states][:, states]
    sns.heatmap(latent_transition_matrix, cmap='YlGnBu', 
                vmax=1, vmin=0, annot=True, norm=PowerNorm(gamma=0.25), ax=ax)
    ax.set_xticklabels(states)
    ax.set_yticklabels(states)
    ax.set_title('Hidden state TPM')

    return ax

# Lineplot showing latent state transitions
def plot_latent_transitions(adata, ax=None):

    # Check if latent paths exist
    try: assert 'latent_paths' in adata.uns['latent_dynamics']['posthoc_computations'].keys()
    except: raise ValueError('Latent paths not found. Run infer_latent_paths() first')

    sns.lineplot(adata.uns['latent_dynamics']['posthoc_computations']['latent_paths'].astype(int), ax=ax) 
    ax.set_title('Viterbi decoding per lineage')
    ax.set_yticks(np.unique(adata.uns['latent_dynamics']['posthoc_computations']['latent_paths'].astype(int)))
    
    return ax

# Heatmap P(lineage/state)
def plot_state_assignment(adata, states=None, ax=None):

    # Check if probs exist
    try: assert 'chain_given_state' in adata.uns['latent_dynamics']['conditional_probabilities'].keys()
    except: raise ValueError('Conditionals not found. Run infer_latent_dynamics() first')

    probs = adata.uns['latent_dynamics']['conditional_probabilities']['chain_given_state']
    if states is None:
        states = np.arange(probs.shape[0])

    probs = probs[states]
    sns.heatmap(probs, vmax=1, vmin=0, cmap='YlGnBu', ax=ax)
    ax.set_yticklabels(states)
    ax.set_title('P(lineage/state)')

    return ax

# Plot latent lineages
def plot_latent_paths(adata, color='whitesmoke', basis='umap', ax=None):

    # Check if latent path exists
    try: assert 'condensed_latent_paths' in adata.uns['latent_dynamics']['posthoc_computations'].keys()
    except: raise ValueError('Latent paths not found. Run infer_latent_paths() first')

    lineage_paths = adata.uns['latent_dynamics']['posthoc_computations']['condensed_latent_paths']
    
    scatter(adata, color=color, alpha=0.8, basis=basis, show=False, ax=ax)
    
    states = np.unique(np.concatenate(list(lineage_paths.values())))
    node_coordinates = pd.DataFrame(index=states, columns=np.arange(adata.obsm['X_{}'.format(basis)].shape[1]))
    for state in states:
        node_coordinates.loc[state] = np.mean(adata.obsm['X_{}'.format(basis)][np.where(adata.obs.kinetic_states==str(state))[0]],
                                          axis=0)
        sns.scatterplot(x=[node_coordinates.loc[state][0]], 
                y=[node_coordinates.loc[state][1]],
                s=500, #color=adata.uns['kinetic_states_colors'][state], 
                edgecolors='none',
                ax=ax)

    color_range = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
                   "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
    for lineage, path in lineage_paths.items():
        for step in range(len(path)-1):
            ax.arrow(node_coordinates.loc[path[step], 0],
                     node_coordinates.loc[path[step], 1],
                     node_coordinates.loc[path[step+1], 0] - node_coordinates.loc[path[step], 0],
                     node_coordinates.loc[path[step+1], 1] - node_coordinates.loc[path[step], 1],
                     color=color_range[lineage], length_includes_head=True, 
                     head_width=0.5)
    ax.set_title('Transition graph')
        
    return ax

# Box plots with annotations arranged by a conitnouse quantity (pseudotime, entropy)
def plot_annotation_boxplots(adata, categorical=None, continous=None, hue=None, 
                             order=None, color='salmon', ax=None):

    flierprops = dict(marker='o', markerfacecolor='None', markersize=0.5,  markeredgecolor='black')

    if order is None:
        order = adata.obs.groupby(categorical)[continous].mean().fillna(0).sort_values().index

    sns.boxplot(x=categorical, y=continous, hue=hue, 
                order=order, data=adata.obs, 
                color=color, flierprops=flierprops, ax=ax)

    return ax

# Heatmap with overlap between annotation and kinetic states
def plot_static_kinetic_overlap(adata, cluster_key: str, states=None, ax=None):

    # Check if probs exist
    try: assert 'state_given_nodes' in adata.uns['latent_dynamics']['conditional_probabilities'].keys()
    except: raise ValueError('Conditionals not found. Run infer_latent_dynamics() first')

    probs = adata.uns['latent_dynamics']['conditional_probabilities']['state_given_nodes']
    if states is None:
        states = np.arange(probs.shape[0])

    probs = pd.DataFrame(adata.uns['latent_dynamics']['conditional_probabilities']['state_given_nodes'].T, 
                         index=adata.obs.index)
    probs = probs.loc[:, states]
    probs[cluster_key] = adata.obs[cluster_key]
    probs = probs.groupby(cluster_key).mean()

    sns.heatmap(probs.T.astype(float), annot=True, cmap='YlGnBu', 
                vmax=1, vmin=0, ax=ax)
    ax.set_yticklabels(states)
    ax.set_xlabel('')
    ax.set_title('Annotation - States overlap')

    return ax

# Plot dynamics panel
def plot_dynamics_summary(adata, cluster_key: str, use_selected=True, 
                          basis='umap', rotation=90, figsize=(13,16)):

    # Check if cluster key exists
    if cluster_key not in adata.obs.columns:
        raise KeyError('Key {} for cluster annotations not found in adata.obs.'.format(cluster_key))

    if use_selected:
        try: assert 'selected_states' in adata.uns['latent_dynamics']['posthoc_computations'].keys()
        except: raise ValueError('No selected states. Run latent_state_selection() first')
        selected_states = adata.uns['latent_dynamics']['posthoc_computations']['selected_states']
    else:
        selected_states = None

    fig, axs = plt.subplots(ncols=3, nrows=4, figsize=figsize)

    # Latent TPM
    plot_latent_transition_matrix(adata, states=selected_states, ax=axs.flat[0])

    # Most probable path
    plot_latent_transitions(adata, ax=axs.flat[1])

    # Chain weights
    plot_state_assignment(adata, states=selected_states, ax=axs.flat[2])

    # Kinetic states UMAP
    scatter(adata, color='kinetic_states', legend_loc='on data', 
            basis=basis, alpha=0.8, show=False, ax=axs.flat[3])
    
    # Lineagepath on UMAP
    plot_latent_paths(adata, color='whitesmoke', basis=basis, ax=axs.flat[4])
    
    # Lineage on UMAP
    scatter(adata, color='lineage', legend_loc='on data', 
            basis=basis, alpha=0.8, show=False, ax=axs.flat[5])
    
    # Kinetic states vs pst
    if 'pseudotime' not in adata.obs.columns:
        infer_pseudotime(adata)

    order=adata.obs.groupby("kinetic_states")["pseudotime"].mean().fillna(0).sort_values().index
    plot_annotation_boxplots(adata, categorical='kinetic_states', continous='pseudotime',
                             order=order, color='cyan', ax=axs.flat[6])

    # Annotation vs pst
    order=adata.obs.groupby(cluster_key)["pseudotime"].mean().fillna(0).sort_values().index
    plot_annotation_boxplots(adata, categorical=cluster_key, continous='pseudotime',
                             order=order, color='cyan', ax=axs.flat[7])
    axs.flat[7].set_xticklabels(axs.flat[7].get_xticklabels(), rotation=rotation)

    # Kinetic annotation overlap
    plot_static_kinetic_overlap(adata, cluster_key, states=selected_states, ax=axs.flat[8])
    axs.flat[8].set_xticklabels(axs.flat[8].get_xticklabels(), rotation=rotation)

    # Entropy on UMAP
    scatter(adata, color='cellular_entropy', legend_loc='on data', 
            basis=basis, alpha=0.8, show=False, ax=axs.flat[9])

    # Entropy per annotation
    plot_annotation_boxplots(adata, categorical=cluster_key, continous='cellular_entropy',
                             order=order, color='salmon', ax=axs.flat[10])
    axs.flat[10].set_xticklabels(axs.flat[10].get_xticklabels(), rotation=rotation)

    # Annotation, kinetic entropy overlap
    entropy_df = adata.obs.groupby(['kinetic_states', cluster_key]).sum().reset_index()
    entropy_df = entropy_df.pivot(index='kinetic_states', columns=cluster_key, 
                                  values='cellular_entropy').astype(float).round(2)
    entropy_df = entropy_df.divide(adata.obs.groupby(cluster_key).count().cellular_entropy, axis=1)

    sns.heatmap(entropy_df, 
                cmap='YlGnBu', annot=True, 
                ax=axs.flat[11])
    axs.flat[11].set_title('Entropy of annotation per states')

    return fig, axs 

# Dynamics report - emissions

# Plot argmax latent states, memberships and probability
def plot_latent_summary(adata, use_selected=True, ncols=4, basis='umap', figsize=(5, 3.5)):

    # Check if probs exist
    try: assert 'state_given_nodes' in adata.uns['latent_dynamics']['conditional_probabilities'].keys()
    except: raise ValueError('Conditionals not found. Run infer_latent_dynamics() first')

    probs = adata.uns['latent_dynamics']['conditional_probabilities']['state_given_nodes']

    if use_selected:
        try: assert 'selected_states' in adata.uns['latent_dynamics']['posthoc_computations'].keys()
        except: raise ValueError('No selected states. Run latent_state_selection() first')
        selected_states = adata.uns['latent_dynamics']['posthoc_computations']['selected_states']
    else:
        selected_states = np.arange(probs.shape[0])

    probs = adata.uns['latent_dynamics']['conditional_probabilities']['state_given_nodes']
    probs = probs[selected_states]
    probs = probs/probs.sum(0)

    nrows = max(int(np.ceil((selected_states.shape[0]*2)/ncols)),1)

    figsize = (figsize[0]*ncols,figsize[1]*nrows)
    fig, axs = plt.subplots(ncols=4, nrows=nrows, figsize=figsize)

    for i in range(0, len(axs.flat), 2):
        if i >= selected_states.shape[0]*2:
            axs.flat[i].axis('off')
            axs.flat[i+1].axis('off')
        else:
            scatter(adata, color='silver', basis=basis, ax=axs.flat[i], show=False)
            scatter(adata, color=probs[int(i/2)], color_map='YlGnBu', 
                    title='Hidden state prob. {}'.format(selected_states[int(i/2)]), 
                    basis=basis, vmin=0, vmax=1, ax=axs.flat[i], show=False)
            
            scatter(adata, color='silver', basis=basis, ax=axs.flat[i+1], show=False)
            scatter(adata, color=pd.get_dummies(probs.argmax(0))[int(i/2)], color_map='binary', 
                    title='Hidden state memb. {}'.format(selected_states[int(i/2)]), 
                    basis=basis, ax=axs.flat[i+1], show=False)

    return fig, axs

# Dynamics report - lineages

# Heatmap with latent state history
def plot_latent_state_history(adata, states=None, chain=None, ax=None):
    
    # Check if probs exist
    try: assert 'joint_probabilities' in adata.uns['latent_dynamics']['model_outputs'].keys()
    except: raise ValueError('Probabilities not found. Run infer_latent_dynamics() first')

    probs = adata.uns['latent_dynamics']['model_outputs']['joint_probabilities']
    if states is None:
        states = np.arange(probs.shape[1])
    probs = probs[:, states]

    # Hidden State probabilities chain
    if chain is None:
        probs = probs.sum(-1).sum(-1).T
    else:
        probs = probs.sum(-1)[:,:,chain].T

    sns.heatmap(probs, cmap='Blues', ax=ax)

    return ax

# Plot lineages panel
def plot_lineages_summary(adata, cluster_key: str, use_selected=True, 
                          basis='umap', rotation=90, figsize=(25, 5)):
    
    # Check if cluster key exists
    if cluster_key not in adata.obs.columns:
        raise KeyError('Key {} for cluster annotations not found in adata.obs.'.format(cluster_key))

    if use_selected:
        try: assert 'selected_states' in adata.uns['latent_dynamics']['posthoc_computations'].keys()
        except: raise ValueError('No selected states. Run latent_state_selection() first')
        selected_states = adata.uns['latent_dynamics']['posthoc_computations']['selected_states']
    else:
        selected_states = None

    num_chains = adata.uns['latent_dynamics']['latent_dynamics_params']['num_chains']

    figsize = (figsize[0], figsize[1]*num_chains)
    fig, axs = plt.subplots(ncols=4, nrows=num_chains, figsize=figsize)

    for i in range(num_chains):
    
        # Plot lineage membership
        scatter(adata, color='silver', basis=basis, ax=axs.flat[i*4], show=False)
        scatter(adata, basis=basis, 
                       color=adata.obs.kinetic_states.isin(np.array(adata.uns['latent_dynamics']['posthoc_computations']['condensed_latent_paths'][i]).astype(str)),
                       title='Chain {} cell membership'.format(i), 
                       color_map='binary', show=False, ax=axs.flat[i*4])
    
        # Plot lineage probabilities
        scatter(adata, color='silver', basis=basis, ax=axs.flat[i*4+1], show=False)
        scatter(adata, basis=basis, 
                       color=adata.uns['latent_dynamics']['posthoc_computations']['lineage_probs'][i],
                       #adata.uns['latent_dynamics']['conditional_probabilities']['state_given_nodes'][adata.uns['latent_dynamics']['posthoc_computations']['condensed_latent_paths'][i]].sum(0),
                       title='Chain {} cell probability'.format(i), 
                       color_map='YlGnBu', vmin=0, vmax=1, show=False, ax=axs.flat[i*4+1])

        # Plot hidden state history
        plot_latent_state_history(adata, states=selected_states, chain=i, ax=axs.flat[i*4+2])
        axs.flat[i*4+2].set_title('Hidden state probabilities chain {}'.format(i)) 

        # Observed state probabilities
        adata.uns['state_probability_sampling']['state_history_chain_{}'.format(i)] = adata.uns['latent_dynamics']['model_outputs']['joint_probabilities'].sum(1)[:,i,:]
        plot_probability_by_clusters(adata, cluster_key=cluster_key, 
                                     key='state_history_chain_{}'.format(i),
                                     ax=axs.flat[i*4+3])
        axs.flat[i*4+3].set_title('Observed state probabilities chain {}'.format(i))
        
    return fig, axs






    

