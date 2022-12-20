import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import rgb2hex
import logging
logging.basicConfig(level=logging.INFO)


# Plot heatmap of state history
def plot_persistence_heatmap(adata, key='state_history', metric='cosine', color_map='gist_heat_r', row_color_key=None,
                             vmax=None, figsize=(10, 10), dpi=100, save=None):
    # Check if state history exists
    try:
        state_history = adata.uns['state_probability_sampling'][key]
    except:
        raise ValueError('{} could not be recovered. Run corresponding function first'.format(key.capitalize().replace('_', ' ')))

    row_colors = None
    row_color_map = dict()
    row_color_categories = None

    # Define row colors
    if row_color_key is not None:
        # Check if specified row color key exists in adata.obs
        try:
            row_color_categories = adata.obs[row_color_key].cat.categories
        except KeyError:
            logging.warning('Key {} for row colors not found in adata.obs.'.format(row_color_key))
        except AttributeError:
            logging.info('Converting adata.obs["{}"] to categorical.'.format(row_color_key))
            adata.obs[row_color_key] = adata.obs[row_color_key].astype('category')
            row_color_categories = adata.obs[row_color_key].cat.categories

        # Try using previously define colors in adata
        if row_color_categories is not None:
            try:
                row_color_map = dict(zip(row_color_categories, adata.uns['{}_colors'.format(row_color_key)]))
            except:
                logging.info('Colors not found, creating colors for {} and storing to adata.uns.'.format(row_color_key))
                n_cat = len(row_color_categories)
                if n_cat <= 10:
                    row_color_map = dict(zip(row_color_categories, [rgb2hex(c) for c in cm.tab10.colors]))
                else:
                    row_color_map = dict(zip(row_color_categories, [rgb2hex(c) for c in cm.tab20.colors]))
                adata.uns['{}_colors'.format(row_color_key)] = np.asarray(row_color_map.values())
            row_colors = np.asarray(adata.obs[row_color_key].map(row_color_map))

    g = sns.clustermap(state_history.T, col_cluster=False, metric=metric, cmap=color_map, vmax=vmax,
                       row_colors=row_colors, figsize=figsize)

    # Add a legend for the row colors
    if row_colors is not None:
        for label in row_color_map.keys():
            g.ax_col_dendrogram.bar(0, 0, color=row_color_map[label], label=label, linewidth=0)
        g.ax_col_dendrogram.legend(loc="center", ncol=min(6, len(row_color_categories)))

    if save is not None:
        plt.savefig(save, dpi=dpi)
    else:
        plt.show()
