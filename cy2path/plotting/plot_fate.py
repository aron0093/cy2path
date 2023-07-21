import numpy as np
import seaborn as sns
from scvelo.plotting import scatter
from matplotlib import pyplot as plt

# Plot fate probability
def plot_fate(adata, key='fate_probabilities_macro', mode='sampling', basis='umap', color_map='viridis', ncols=3, dpi=100, figsize=(6,5), save=None):

    if mode=='sampling':
        upper_key='cell_fate'
    
    try: fate_probabilities = adata.uns[upper_key]['fate_probabilities_macro']
    except: raise ValueError('Fate probabilities were not found.')

    fate_probabilities = fate_probabilities.loc[fate_probabilities.sum(axis=1)>0]

    if fate_probabilities.shape[0] > ncols:
        nrows=int(np.ceil(fate_probabilities.shape[0]/ncols))
    else:
        ncols=fate_probabilities.shape[0]
        nrows=1
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6*ncols, 5*nrows), dpi=dpi)
    if ncols==1 and nrows==1:
        axs=[axs]
    else:
        axs=axs.flat

    for i, idx in enumerate(fate_probabilities.index.values):
        scatter(adata, basis=basis, color=fate_probabilities.loc[idx].astype(float), 
                title='Fate probabilities: {}'.format(idx), color_map=color_map, ax=axs[i], show=False)
    
    if save:
        plt.save(save)
    else:
        plt.show()