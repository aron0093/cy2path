import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def plot_probability_by_clusters(adata, cluster_key: str, key='state_history', plot='heatmap',
                                 cmap='gist_heat_r', agg='sum', figsize=(7, 5),
                                 dpi=100, save=None, show=False, ax=None):
    """
    Plot sampling probabilities summarized by clusters
    :param adata: AnnData object with state probabilities
    :param cluster_key: key in adata.obs
    :param key: key in adata.uns['state_probability_sampling']
    :param plot: type of plot, either 'heatmap' or 'line'
    :param cmap: matplotlib color map
    :param agg: function for aggregating probabilities per cluster
    :param figsize: tuple, figure size
    :param dpi: dpi for plot when saving
    :param save: file name for saving plot
    :return:
    """
    # Check if state history exists
    try:
        state_history = adata.uns['state_probability_sampling'][key].T
    except:
        raise ValueError(
            '{} could not be recovered. Run corresponding function first'.format(key.capitalize().replace('_', ' ')))

    state_history_clusters = pd.DataFrame(state_history, index=adata.obs.index)

    # Check if cluster key exists
    if cluster_key not in adata.obs.columns:
        raise KeyError('Key {} for cluster annotations not found in adata.obs.'.format(cluster_key))
    state_history_clusters[cluster_key] = adata.obs[cluster_key]
    state_history_clusters = state_history_clusters.groupby(cluster_key).aggregate(agg)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Make heatmap or line plot
    if plot == 'heatmap':
        sns.heatmap(state_history_clusters, cmap=cmap, ax=ax)
    elif plot == 'line':
        state_history_clusters = state_history_clusters.reset_index().melt(id_vars=cluster_key,
                                                                           var_name='iteration')
        hue_categories = state_history_clusters[cluster_key].astype('category').cat.categories
        palette = dict(zip(hue_categories, adata.uns['{}_colors'.format(cluster_key)]))
        sns.lineplot(x='iteration', y='value', data=state_history_clusters,
                     hue=cluster_key, palette=palette, ax=ax)
        ax.set_ylabel('{} of probabilities by {}'.format(agg, cluster_key))
        if save is not None:
            plt.savefig(save, dpi=dpi)
        if show:
            plt.show()

    else:
        raise ValueError('Plot must be either "heatmap" or "line".')
