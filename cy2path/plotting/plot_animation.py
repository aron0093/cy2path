import os
import numpy as np

from scvelo.plotting import scatter

import imageio
from joblib import delayed, Parallel
from tqdm.auto import tqdm

from ..utils import supress_stdout

# Make scatter plots of each iteration
@supress_stdout
def make_frame(i, adata, data_history, basis='umap', key='state_history', dpi=100, figsize=(7,5), color_map='YlGnBu', rescale_color=[0,1], save='./temp/', n_jobs=-1):
    adata.obs['current_state_probability'] = data_history[i]
    if rescale_color:
        scatter(adata, color='current_state_probability', title='Iteration: {}'.format(i), dpi=dpi, figsize=figsize, show=False,
                save=os.path.join(save, '{}.png'.format(i)), basis=basis, color_map=color_map, rescale_color=rescale_color, vmax=rescale_color[1], vmin=rescale_color[0])
    else:
        scatter(adata, color='current_state_probability', title='Iteration: {}'.format(i), dpi=dpi, figsize=figsize, show=False,
        save=os.path.join(save, '{}.png'.format(i)), basis=basis, color_map=color_map, rescale_color=rescale_color)


# Select frames (index based on the original list), replicate, remove frames, add replicated frames
def make_gradient(frames, target_frames, frame_elongations):
    # Gradient length in order to keep target_frames indices w.r.t. original frames
    gradient_length = 0

    # Iterate: select frames, replicate, remove from original list, add replicated frames
    for start_end, elongation in zip(target_frames, frame_elongations):
        start = start_end[0] + gradient_length
        end = start_end[1] + gradient_length

        replicated_frames = np.repeat(frames[start:end], elongation)
        frames = np.delete(frames, range(start, end))
        frames = np.insert(frames, start, replicated_frames)

        gradient_length += len(replicated_frames) - (end - start)

    return frames

# Make animation using individual scatter plots
def plot_animation(adata, basis='umap', key='state_history', mode='sampling', dpi=100, figsize=(7,5), color_map='YlGnBu', 
                   gradient_frames=False, gradient_elongations=False, framerate=0.2,
                   rescale_color=[0,1], save='./temp/', n_jobs=-1, ):
    
    # Check if state history exists
    if mode=='sampling':
        upper_key = 'state_probability_sampling'
    elif mode=='simulation':
        upper_key = 'markov_chain_sampling'

    # Create directory to store frames
    if not os.path.isdir(save):
        os.makedirs(save)

    # Check if state history exists
    try: data_history = adata.uns[upper_key][key]
    except: raise ValueError('{} could not be recovered. Run corresponding function first'.format(key.capitalize().replace('_', ' ')))

    # Make frames
    Parallel(n_jobs=n_jobs)(delayed(make_frame)(i, adata, data_history, 
                                                basis=basis, key=key, dpi=dpi, 
                                                figsize=figsize, color_map=color_map,
                                                rescale_color=rescale_color, 
                                                save=save) for i in tqdm(range(data_history.shape[0]), desc='Creating frame of each iteration', unit=' frames'))

    filenames = [os.path.join(save, '{}.png'.format(i)) for i in range(data_history.shape[0])]
    # Add gradient to selected frames
    if gradient_frames and gradient_elongations:
        filenames = make_gradient(frames=filenames, target_frames=gradient_frames, frame_elongations=gradient_elongations)

    frames = []
    for filename in filenames:
        image = imageio.imread(filename)
        frames.append(image)
        
    kargs = { 'duration': framerate}
    imageio.mimsave(os.path.join(save, 'animation.gif'), frames, 'GIF', **kargs)