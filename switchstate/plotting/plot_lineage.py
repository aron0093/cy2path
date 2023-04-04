import numpy as np
import seaborn as sns
from scvelo.plotting import scatter
from matplotlib import pyplot as plt
import collections
from tqdm.auto import tqdm

# Plot cytopath assignment probability
def plot_lineages(adata, keys=None, basis='umap', color_map='viridis', ncols=3, dpi=100, figsize=(6,5), save=None):

    if keys is not None and isinstance(keys, (collections.Sequence, np.ndarray)):
        for key in keys:
            try: assert key in adata.obs.columns
            except: raise ValueError('{} not found in .obs'.format(key))
    elif keys is not None and isinstance(keys, str):
        keys = [keys]
    else:
        keys=[None]
        
    try: assert adata.uns['markov_chain_sampling']['lineage_inference_clusters'].shape
    except: raise ValueError('Lineage annotations could not be recovered.')

    # Compute trajectory coordinates
    lineages = np.unique(adata.uns['markov_chain_sampling']['lineage_inference_clusters'])
    lineage_coordinates = []
    for i in tqdm(range(len(lineages)), desc='Computing coordinates', unit=' lineage'):
        markov_chain_indices = np.where(adata.uns['markov_chain_sampling']['lineage_inference_clusters']==lineages[i])[0]
        markov_chains_ = adata.uns['markov_chain_sampling']['state_indices'][markov_chain_indices]

        lineage_coordinates.append(adata.obsm['X_{}'.format(basis)][markov_chains_].mean(axis=0))

    if len(keys) > ncols:
        nrows=int(np.ceil(len(keys)/ncols))
    else:
        ncols=len(keys)
        nrows=1
    nrows=int(np.ceil(len(keys)/ncols))

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6*ncols, 5*nrows), dpi=dpi)
    if ncols==1 and nrows==1:
        axs=[axs]
    else:
        axs=axs.flat
    
    for i, key in enumerate(keys):
        scatter(adata, basis=basis, color=key, 
                title='{}'.format(key), 
                color_map=color_map, 
                ax=axs[i], 
                show=False)
        for lineage in lineage_coordinates:
            axs[i].plot(lineage[:,0], lineage[:,1])
    
    if save:
        plt.save(save)
    else:
        plt.show()
