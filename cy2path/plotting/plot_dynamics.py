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

    sns.lineplot(np.log(np.array(model.loss_values)+model.loss_values[-1]/10), 
                 ax=axs.flat[0])
    axs.flat[0].set_title('Log Loss')

    sns.lineplot(model.divergence_values, ax=axs.flat[1])
    axs.flat[1].set_title('KL divergence')

    sns.lineplot(model.likelihood_values, ax=axs.flat[2])
    axs.flat[2].set_title('Log likelihood')

    sns.lineplot(model.sparsity_values, ax=axs.flat[3])
    axs.flat[3].set_title('Sparsity regularisation')

    # if model.regularisation_values != []:
    #     sns.lineplot(model.regularisation_values, ax=axs.flat[4])
    #     axs.flat[4].set_title('TPM regularisation')
    # else:
    #     axs.flat[4].axis('off')

    sns.lineplot(model.orthogonality_values, ax=axs.flat[4])
    axs.flat[4].set_title('Orthogonality regularisation')

    if model.num_chains > 1:
        sns.lineplot(model.exclusivity_values, ax=axs.flat[5])
        axs.flat[5].set_title('Kinetic state exclusivity')
    else:
        axs.flat[5].axis('off')


    return fig, axs

# Plot proportion per kinetic state
def plot_ratio(adata, probability=False,  ax=None):

    if probability:
        state_ratio = adata.uns['latent_dynamics']['conditional_probabilities']['state_given_nodes'].sum(1)/adata.shape[0]
    else:
        state_ratio = np.empty(adata.uns['latent_dynamics']['latent_dynamics_params']['num_states'])
        states, counts = np.unique(adata.uns['latent_dynamics']['conditional_probabilities']['state_given_nodes'].argmax(0), 
                                   return_counts=True)
        
        for i, state in enumerate(states):
            state_ratio[state] = counts[i]
        state_ratio = state_ratio/adata.shape[0]
    state_ratio_args = np.argsort(-1*state_ratio)

    sns.lineplot(x=state_ratio_args.astype(str), y=state_ratio[state_ratio_args], color='black', ax=ax)
    sns.scatterplot(x=state_ratio_args.astype(str), y=state_ratio[state_ratio_args], color='black', ax=ax)
    if adata.uns['latent_dynamics']['latent_dynamics_params']['min_ratio']:
        ax.hlines(adata.uns['latent_dynamics']['latent_dynamics_params']['min_ratio'], 0, 
                  adata.uns['latent_dynamics']['latent_dynamics_params']['num_states']-1,
                linestyle='dotted', color='red')

    ax.set_ylim(0, max(state_ratio)+0.05)
    return ax

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
    ax.set_title('Latent path decoding per lineage')
    ax.set_yticks(np.unique(adata.uns['latent_dynamics']['posthoc_computations']['latent_paths'].astype(int)))
    
    return ax

# Heatmap P(lineage/state)
def plot_state_assignment(adata, states=None, prob_key='chain_given_state', ax=None):

    # Check if probs exist
    try: assert prob_key in adata.uns['latent_dynamics']['conditional_probabilities'].keys()
    except: raise ValueError('Conditionals not found. Run infer_latent_dynamics() first')

    probs = adata.uns['latent_dynamics']['conditional_probabilities'][prob_key]
    if states is None:
        states = np.arange(probs.shape[0])

    probs = probs[states]
    sns.heatmap(probs, vmax=1, vmin=0, cmap='YlGnBu', ax=ax)
    ax.set_yticklabels(states)
    ax.set_title('P(lineage/state)')

    return ax

# Heatmap P(lineage/node)
def plot_node_assignment(adata, prob_key='chain_given_nodes', ordering=None, ax=None):

    # Check if probs exist
    try: assert prob_key in adata.uns['latent_dynamics']['conditional_probabilities'].keys()
    except: raise ValueError('Conditionals not found. Run infer_latent_dynamics() first')

    probs = adata.uns['latent_dynamics']['conditional_probabilities'][prob_key]
    if ordering is not None:
        probs = probs[:, ordering]

    sns.heatmap(probs.T, vmax=1, vmin=0, cmap='YlGnBu', ax=ax)
    ax.set_yticklabels([])
    ax.set_title('P(lineage/node)')

    return ax

# Plot latent lineages
def plot_latent_paths(adata, color='whitesmoke', mode='clusters', basis='umap', ax=None):

    # Check if latent path exists
    try: assert 'condensed_latent_paths' in adata.uns['latent_dynamics']['posthoc_computations'].keys()
    except: raise ValueError('Latent paths not found. Run infer_latent_paths() first')

    lineage_paths = adata.uns['latent_dynamics']['posthoc_computations']['condensed_latent_paths']
    
    scatter(adata, color=color, alpha=0.8, basis=basis, show=False, ax=ax)
    
    color_range = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
                   "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
    for lineage, path in lineage_paths.items():

        node_coordinates = pd.DataFrame(index=path, columns=np.arange(adata.obsm['X_{}'.format(basis)].shape[1]))
        for state in path:
            if mode=='clusters':
                # Chain specific weight of nodes to calculate coords
                state_probs = adata.uns['latent_dynamics']['conditional_probabilities']['node_given_chain_state_selected'][state, lineage]   
            elif mode=='states':
                # Overall coordinate of state
                state_probs = adata.uns['latent_dynamics']['conditional_probabilities']['node_given_state_selected'][state]

            node_coordinates.loc[state] = np.sum(np.multiply(state_probs.reshape(-1,1), adata.obsm['X_{}'.format(basis)]), axis=0)
            sns.scatterplot(x=[node_coordinates.loc[state][0]], 
                        y=[node_coordinates.loc[state][1]],
                        s=500, #color=adata.uns['kinetic_states_colors'][adata.obs.kinetic_states.cat.categories.get_loc(state)], 
                        edgecolors='none',
                        ax=ax)

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
def plot_static_kinetic_overlap(adata, cluster_key: str, prob_key='state_given_nodes', states=None, ax=None):

    # Check if probs exist
    try: assert prob_key in adata.uns['latent_dynamics']['conditional_probabilities'].keys()
    except: raise ValueError('Conditionals not found. Run infer_latent_dynamics() first')

    probs = adata.uns['latent_dynamics']['conditional_probabilities'][prob_key]
    if states is None:
        states = np.arange(probs.shape[0])

    probs = pd.DataFrame(adata.uns['latent_dynamics']['conditional_probabilities'][prob_key].T, 
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
                          basis='umap', rotation=90, figsize=(13,20)):

    # Check if cluster key exists
    if cluster_key not in adata.obs.columns:
        raise KeyError('Key {} for cluster annotations not found in adata.obs.'.format(cluster_key))

    if use_selected:
        try: assert 'selected_states' in adata.uns['latent_dynamics']['posthoc_computations'].keys()
        except: raise ValueError('No selected states. Run latent_state_selection() first')
        selected_states = adata.uns['latent_dynamics']['posthoc_computations']['selected_states']
        plot_node_assignment_key = 'chain_given_nodes_selected'
        plot_state_assignment_key = 'chain_given_state_selected'
        plot_static_kinetic_overlap_key = 'state_given_nodes_selected'
    else:
        selected_states = None

    # Kinetic states vs pst
    if 'pseudotime' not in adata.obs.columns:
        infer_pseudotime(adata)

    fig, axs = plt.subplots(ncols=3, nrows=5, figsize=figsize)

    # Latent TPM
    plot_latent_transition_matrix(adata, states=selected_states, ax=axs.flat[0])

    # Most probable path
    plot_latent_transitions(adata, ax=axs.flat[1])

    # Chain weights
    # ordering = adata.obs.index.get_indexer(adata.obs['pseudotime'].sort_values().index.values)
    # plot_node_assignment(adata, prob_key=plot_node_assignment_key, 
    #                      ordering=ordering, ax=axs.flat[2])
    plot_state_assignment(adata, prob_key=plot_state_assignment_key, 
                          states=selected_states, ax=axs.flat[2])

    # Kinetic states UMAP
    scatter(adata, color='kinetic_states', legend_loc='on data', 
            basis=basis, alpha=0.8, show=False, ax=axs.flat[3])
    
    # Kinetic clustering UMAP
    scatter(adata, color='kinetic_clustering', legend_loc='on data', 
            basis=basis, alpha=0.8, show=False, ax=axs.flat[4])
    
    # Lineagepath on UMAP
    plot_latent_paths(adata, color='whitesmoke', basis=basis, ax=axs.flat[5])

    order=adata.obs.groupby("kinetic_states")["pseudotime"].mean().fillna(0).sort_values().index
    plot_annotation_boxplots(adata, categorical='kinetic_states', continous='pseudotime',
                             order=order, color='cyan', ax=axs.flat[6])

    # Annotation vs pst
    order=adata.obs.groupby(cluster_key)["pseudotime"].mean().fillna(0).sort_values().index
    plot_annotation_boxplots(adata, categorical=cluster_key, continous='pseudotime',
                             order=order, color='cyan', ax=axs.flat[7])
    axs.flat[7].set_xticklabels(axs.flat[7].get_xticklabels(), rotation=rotation)

    # Kinetic annotation overlap
    plot_static_kinetic_overlap(adata, cluster_key, states=selected_states, 
                                prob_key=plot_static_kinetic_overlap_key, ax=axs.flat[8])
    axs.flat[8].set_xticklabels(axs.flat[8].get_xticklabels(), rotation=rotation)

    # Transitional Entropy on UMAP
    scatter(adata, color='transitional_entropy', legend_loc='on data', 
            basis=basis, alpha=0.8, show=False, ax=axs.flat[9])

    # Transitional Entropy per annotation
    plot_annotation_boxplots(adata, categorical=cluster_key, continous='transitional_entropy',
                             order=order, color='salmon', ax=axs.flat[10])
    axs.flat[10].set_xticklabels(axs.flat[10].get_xticklabels(), rotation=rotation)

    # Annotation, kinetic transitional entropy overlap
    t_entropy_df = adata.obs.groupby(['kinetic_states', cluster_key]).count().reset_index()
    t_entropy_df = t_entropy_df.pivot(index='kinetic_states', columns=cluster_key, 
                                  values='transitional_entropy').astype(float).round(2)
    t_entropy_df = t_entropy_df.divide(adata.obs.groupby(cluster_key).count().transitional_entropy, axis=1)

    sns.heatmap(t_entropy_df, 
                cmap='YlGnBu', annot=True, 
                ax=axs.flat[11])
    axs.flat[11].set_title(' Trans. Entropy of annot. per states')

    # Differentiation Entropy on UMAP
    scatter(adata, color='differentiation_entropy', legend_loc='on data', 
            basis=basis, alpha=0.8, show=False, ax=axs.flat[12])

    # Differentiation Entropy per annotation
    plot_annotation_boxplots(adata, categorical=cluster_key, continous='differentiation_entropy',
                             order=order, color='salmon', ax=axs.flat[13])
    axs.flat[13].set_xticklabels(axs.flat[13].get_xticklabels(), rotation=rotation)

    # Annotation, kinetic differentiation entropy overlap
    d_entropy_df = adata.obs.groupby(['kinetic_states', cluster_key]).count().reset_index()
    d_entropy_df = d_entropy_df.pivot(index='kinetic_states', columns=cluster_key, 
                                      values='differentiation_entropy').astype(float).round(2)
    d_entropy_df = d_entropy_df.divide(adata.obs.groupby(cluster_key).count().differentiation_entropy, axis=1)

    sns.heatmap(d_entropy_df, 
                cmap='YlGnBu', annot=True, 
                ax=axs.flat[14])
    axs.flat[14].set_title(' Diff. Entropy of annot. per states')

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
        probs = adata.uns['latent_dynamics']['conditional_probabilities']['state_given_nodes_selected']

    else:
        selected_states = np.arange(probs.shape[0])

    probs = probs[selected_states]
    membership = pd.get_dummies(probs.argmax(0))
    #probs = probs/probs.sum(0)

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
            if int(i/2) in membership.columns:
                scatter(adata, color=membership[int(i/2)], color_map='binary', 
                        title='Hidden state memb. {}'.format(selected_states[int(i/2)]), 
                        basis=basis, ax=axs.flat[i+1], show=False)
            else:
                pass

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

        adata.uns['state_probability_sampling']['state_history_chain_{}'.format(i)] = adata.uns['latent_dynamics']['model_outputs']['joint_probabilities'].sum(1)[:,i,:]
    
        # Plot lineage membership
        scatter(adata, color='silver', basis=basis, ax=axs.flat[i*4], show=False)
        scatter(adata, basis=basis, 
                       color=adata.obs.lineage_assignment==i,
                       title='Chain {} cell membership'.format(i), 
                       color_map='binary', vmin=0, vmax=1, show=False, ax=axs.flat[i*4])
        #plot_simulation(adata, prob_key='state_history_chain_{}'.format(i), color='whitesmoke', basis='umap', ax=axs.flat[i*4])
    
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
        plot_probability_by_clusters(adata, cluster_key=cluster_key, 
                                     key='state_history_chain_{}'.format(i),
                                     ax=axs.flat[i*4+3])
        axs.flat[i*4+3].set_title('Observed state probabilities chain {}'.format(i))
        
    return fig, axs

# Plot simulation
def plot_simulation(adata, prob_key='state_history', color='whitesmoke', basis='umap', ax=None):
    
     # Check if state history exists
    try: assert prob_key in adata.uns['state_probability_sampling'].keys()
    except: raise ValueError('{} not found. Run sampling/dynamics first'.format(prob_key))
    
    probs = adata.uns['state_probability_sampling'][prob_key]
    probs = (probs.T/probs.T.sum(0)).T
    
    scatter(adata, color=color, alpha=0.8, basis=basis, show=False, ax=ax)
    
    node_coordinates = pd.DataFrame(index=np.arange(probs.shape[0]), 
                                    columns=np.arange(adata.obsm['X_{}'.format(basis)].shape[1]))
    for step in node_coordinates.index.values:
        node_coordinates.loc[step] = np.sum(np.multiply(probs[step].reshape(-1,1), 
                                                         adata.obsm['X_{}'.format(basis)]),
                                             axis=0)
        sns.scatterplot(x=[node_coordinates.loc[step][0]], 
                        y=[node_coordinates.loc[step][1]],
                        s=100,
                        color='black',
                        edgecolors='none',
                        ax=ax)
        if step>0:
            ax.arrow(node_coordinates.loc[step-1, 0],
                     node_coordinates.loc[step-1, 1],
                     node_coordinates.loc[step, 0] - node_coordinates.loc[step-1, 0],
                     node_coordinates.loc[step, 1] - node_coordinates.loc[step-1, 1],
                     length_includes_head=True, 
                     color='black',
                     head_width=0.2)
        
    return ax






    

