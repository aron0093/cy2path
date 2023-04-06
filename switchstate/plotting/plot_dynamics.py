import numpy as np
import pandas as pd

import seaborn as sns
from scvelo.plotting import scatter
from matplotlib import pyplot as plt

# Dynamics report - summary
# Heatmap with latent state transition matrix
def plot_latent_transitions():
    raise NotImplementedError()
    
# Box plots with annotations arranged by pseudotime
def plot_annotation_pseudotime():
    raise NotImplementedError()

# Plot dynamics panel
def plot_dynamics_summary():
    # Latent TPM
    # Most probable path
    # Chain weights
    # Kinetic states UMAP
    # Kinetic states vs pst
    # Annotation vs pst
    raise NotImplementedError()

# Dynamics report - lineages
# Heatmap with latent state history
def plot_latent_state_history():
    raise NotImplementedError()

# Plot lineages panel
def plot_lineages_summary():
    # Cell fate - annot overlap (?)
    # Hidden history
    # Annotation -> observed history
    raise NotImplementedError()   

# Dynamics report - emissions
# Plot argmax latent states, memberships and probability
def plot_latent_summary():
    # Prob
    # Membership
    raise NotImplementedError()

# Heatmap with overlap between annotation and kinetic states
def plot_static_kinetic_overlap():
    raise NotImplementedError()