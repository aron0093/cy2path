import numpy as np

import seaborn as sns
from scvelo.plotting import scatter
from matplotlib import pyplot as plt

from dtaidistance import dtw_barycenter

from tqdm.auto import tqdm

# Plot Markov chains
def plot_cytopath_simulations(adata, key=None, basis='umap', color_map='viridis', ax=None):

    if key is not None:
        try: assert key in adata.obs.columns
        except: raise ValueError('{} not found in .obs'.format(key))
    
    try: assert adata.uns['markov_chain_sampling']['state_indices'].shape
    except: raise ValueError('Simulations could not be recovered.')

    # Compute simulation coordinates
    markov_chains_ = adata.uns['markov_chain_sampling']['state_indices']
    simulation_coordinates = adata.obsm['X_{}'.format(basis)][markov_chains_]

    ax = scatter(adata, basis=basis, color=key,
            color_map=color_map, 
            ax=ax, 
            show=False)
    for simulation in simulation_coordinates:
        ax.plot(simulation[:,0], simulation[:,1], color='grey')

    return ax

# Plot Cytopath lineage coordinates 
def plot_cytopath_lineages(adata, key=None, basis='umap', mode='average', color_map='viridis', ax=None):

    if key is not None:
        try: assert key in adata.obs.columns
        except: raise ValueError('{} not found in .obs'.format(key))
        
    try: assert adata.uns['cytopath']['lineage_inference_clusters'].shape
    except: raise ValueError('Lineage annotations could not be recovered.')

    # Compute trajectory coordinates
    lineages = [l for l in np.unique(adata.uns['cytopath']['lineage_inference_clusters']) if l>=0]
    lineage_coordinates = []
    for i in tqdm(range(len(lineages)), desc='Computing coordinates', unit=' lineage'):
        markov_chain_indices = np.where(adata.uns['cytopath']['lineage_inference_clusters']==lineages[i])[0]
        markov_chains_ = adata.uns['markov_chain_sampling']['state_indices'][markov_chain_indices]

        if mode=='average':
            lineage_coordinate_ = np.median(adata.obsm['X_{}'.format(basis)][markov_chains_], axis=0)
        elif mode=='dba':
            lineage_coordinate_ = dtw_barycenter.dba_loop(adata.obsm['X_{}'.format(basis)][markov_chains_].astype('double'), 
                                                 c=None, max_it=1000, use_c=True)
            
        lineage_coordinates.append(lineage_coordinate_)

    ax= scatter(adata, basis=basis, color=key,
                color_map=color_map, 
                ax=ax, 
                show=False)
    for lineage in lineage_coordinates:
        ax.plot(lineage[:,0], lineage[:,1])
    
    return ax