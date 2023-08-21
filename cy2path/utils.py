import os, contextlib

import logging
logging.basicConfig(level = logging.INFO)

import collections

from scvelo.utils import get_transition_matrix
from scvelo.tools import terminal_states
from scvelo.tools.terminal_states import eigs

import numpy as np
from scipy.sparse import issparse, csr_matrix

import torch
from torch.autograd import Function
from torch.nn.utils.parametrizations import _make_orthogonal

# Scale array to range
def scale(X, min=0, max=1):
    idx = np.isfinite(X)
    if any(idx):
        X = X - X[idx].min() + min
        xmax = X[idx].max()
        X = X / xmax * max if xmax != 0 else X * max
    return X

# Row normalize matrix
def normalize(X):
        if issparse(X):
            return X.multiply(csr_matrix(1.0 / np.abs(X).sum(1)))
        else:
            return X / X.sum(1)

# Supress unwanted print messages
# WARNING: All standard output will be supressed
def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)
    return wrapper

# Check if TPM has been calculated
def check_TPM(adata, matrix_key='T_forward', recalc_matrix=False, self_transitions=False):

    # Recalcualte TPM if specified
    if recalc_matrix:
        adata.obsp[matrix_key] = get_transition_matrix(adata, self_transitions=self_transitions)
        terminal_states(adata, self_transitions=self_transitions)
    else:
        try: 
            assert adata.obsp[matrix_key].shape
        except: 
            adata.obsp[matrix_key] = get_transition_matrix(adata, self_transitions=self_transitions)
            terminal_states(adata, self_transitions=self_transitions)
            logging.warning('Transition probability matrix was not present in .obsp and was calculated.')

        if not issparse(adata.obsp[matrix_key]):
            adata.obsp[matrix_key] = csr_matrix(adata.obsp[matrix_key])

    # Recalculate root and terminal states if not present and TPM was not recalculated
    try: 
        assert 'root_cells' in adata.obs.columns
        assert 'end_points' in adata.obs.columns
    except: 
        terminal_states(adata, self_transitions=self_transitions, random_state=None)
        logging.warning('Root states were not present and were calculated.')

# Check if root states can be initialised
def check_root_init(adata, init='root_cells'):

    # Initialise using root_cells, uniform or custom
    if isinstance(init, str):
        if init=='root_cells':
            init_state_probability = (adata.obs['root_cells']/adata.obs['root_cells'].sum()).values
            init_type = init
        elif init=='uniform':
            init_state_probability = [1/adata.shape[0]]*adata.shape[0]  # uniform probability to start at each cell
            init_type = 'uniform'
    elif isinstance(init, (collections.Sequence, np.ndarray)):
        init_state_probability=init
        init_type = 'custom'
    else:
        raise ValueError('Incorrect initialisation of state probabilities.')

    return init_state_probability, init_type

# Estimate stationary state of TPM
def estimate_stationary_state(adata, matrix_key='T_forward'):

    # Stationary state distribution
    stationary_state_probability = eigs(adata.obsp[matrix_key])[1]
    if stationary_state_probability.shape[1] == 1:
        stationary_state_probability /= stationary_state_probability.sum()
    else:
        stationary_state_probability = (adata.obs['end_points']/adata.obs['end_points'].sum()).values
        logging.warning('Multiple eigenvalues > 1, falling back to end_points to infer stationary distribution.')
    stationary_state_probability = stationary_state_probability.flatten()

    return stationary_state_probability

# Exponentiate and detach pytroch tensor
def exponentiate_detach(tensor):
    return tensor.exp().cpu().detach().numpy()

# Return argmax as one hot encoding in log domain
def log_domain_hardmax(tensor_, dim=0):

    num_cats = tensor_.shape[dim]
    argmax_ = tensor_.argmax(dim=dim)
    log_hardmax = torch.nn.functional.one_hot(argmax_, num_classes=num_cats).log().reshape(tensor_.shape)
    return log_hardmax

# Take mean along axis in log domain
def log_domain_mean(tensor_, dim=0, use_gpu=False):

    log_sum = tensor_.logsumexp(dim)
    if use_gpu:
        log_mean = log_sum - torch.log(torch.Tensor([tensor_.shape[dim]]).cuda())
    else:
        log_mean = log_sum - torch.log(torch.Tensor([tensor_.shape[dim]]))
    return log_mean

# Generalised Jensen-Shannon divergence
class JSDLoss(torch.nn.Module):
    def __init__(self, reduction='sum', use_gpu=False):
        super().__init__()
        self.kl = torch.nn.KLDivLoss(reduction=reduction, log_target=True)
        self.use_gpu = use_gpu

    def forward(self, tensor):
        weight = 1.0/tensor.shape[0]
        centroid = log_domain_mean(tensor, dim=0, use_gpu=self.use_gpu)

        return weight * sum([self.kl(centroid, tensor[i]) for i in range(tensor.shape[0])])
    
# Mutual information from contingency
class MI(torch.nn.Module):

    def __init__(self, use_gpu=False):
        super().__init__()
        self.use_gpu = use_gpu

    def forward(self, contingency_table):

        # Compute the marginal probability distributions
        marginal_x = contingency_table.sum(1)
        marginal_y = contingency_table.sum(0)
        
        # Compute the mutual information
        mutual_info = torch.Tensor([0])
        if self.use_gpu:
            mutual_info = mutual_info.cuda()

        for i in range(contingency_table.shape[0]):
            for j in range(contingency_table.shape[1]):
                if contingency_table[i, j] > 0:
                    mutual_info += contingency_table[i, j] * torch.log2(contingency_table[i, j] /
                                                                    (marginal_x[i] * marginal_y[j]))
        
        return mutual_info[0]

# Gradient reversal from https://github.com/tadeephuy/GradientReversal
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha*grad_output
        return grad_input, None
revgrad = GradientReversal.apply