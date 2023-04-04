import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.distance import cosine, euclidean

# Plot distance between state probability profiles
def plot_distances(adata, key='state_history_max_iter', mode='sampling', metric='cosine', log=False, dpi=100, figsize=(18,5), save=None):
       
    # Check if state history exists
    stationary_state_key = 'stationary_state_probability'
    if mode=='sampling':
        upper_key = 'state_probability_sampling'
    elif mode=='simulation':
        upper_key = 'markov_chain_sampling'
        if adata.uns[upper_key]['sampling_params']['convergence_criterion'] in adata.obs.columns:
            stationary_state_key = 'stationary_state_probability_by_cluster'

    try: data_history = adata.uns[upper_key][key]
    except: raise ValueError('{} could not be recovered. Run corresponding function first'.format(key.capitalize().replace('_', ' ')))

    distance_to_next, distance_to_stationary = [], []
    for i in range(1, data_history.shape[0]):
        if metric=='cosine':
            distance_to_next.append(cosine(data_history[i-1], data_history[i]))
            distance_to_stationary.append(cosine(data_history[i],
                                                 adata.uns[upper_key][stationary_state_key]))
        elif metric=='euclidean':
            distance_to_next.append(euclidean(data_history[i-1], data_history[i]))
            distance_to_stationary.append(euclidean(data_history[i],
                                                    adata.uns[upper_key][stationary_state_key]))

    distance_to_stationary_iterative = [abs(distance_to_stationary[i + 1] - distance_to_stationary[i]) for i in range(len(distance_to_stationary)-1)]
          
    fig, axs = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

    axs[0].set_title('{} distance between iterations'.format(metric.capitalize()))
    axs[0].plot(distance_to_next)

    axs[1].set_title('{} distance to stationary state'.format(metric.capitalize()))
    axs[1].plot(distance_to_stationary)

    axs[2].set_title('{} distance to stationary state: [i] - [i-1]'.format(metric.capitalize()))
    axs[2].plot(distance_to_stationary_iterative)
    
    for ax in axs:
        ax.axvline(x=adata.uns[upper_key]['sampling_params']['convergence'], linestyle='--', color='darkgrey')
        if log:
            ax.set_yscale('log')
            ax.set_ylabel('{} distance (log scale)'.format(metric.capitalize()))
        else:
            ax.set_ylabel('{} distance'.format(metric.capitalize()))
        ax.set_xlabel('Iterations')

    if save:
        plt.save(save)
    else:
        plt.show()


