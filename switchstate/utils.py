import torch
import os, contextlib
import numpy as np
from scipy.sparse import issparse, csr_matrix

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

def log_domain_matmul(log_A, log_B):
    
    '''
    Normally, a matrix multiplication
    computes out_{i,j} = sum_k A_{i,k} x B_{k,j}

    A log domain matrix multiplication
    computes out_{i,j} = logsumexp_k log_A_{i,k} + log_B_{k,j}
    
    Parameters
    ----------
    
    log_A : m x n
    log_B : n x p
    
    Returns
    -------
    output : m x p matrix
    
    '''
    m = log_A.shape[0]
    n = log_A.shape[1]
    p = log_B.shape[1]

    # log_A_expanded = torch.stack([log_A] * p, dim=2)
    # log_B_expanded = torch.stack([log_B] * m, dim=0)
    # fix for PyTorch > 1.5 by egaznep on Github:
    log_A_expanded = torch.reshape(log_A, (m,n,1))
    log_B_expanded = torch.reshape(log_B, (1,n,p))

    elementwise_sum = log_A_expanded + log_B_expanded
    out = torch.logsumexp(elementwise_sum, dim=1)

    return out

def log_domain_hardmax(tensor_, dim=0):

    num_cats = tensor_.shape[dim]
    argmax_ = tensor_.argmax(dim=dim)
    log_hardmax = torch.nn.functional.one_hot(argmax_, num_classes=num_cats).log().reshape(tensor_.shape)
    return log_hardmax

def log_domain_mean(tensor_, dim=0, use_gpu=False):

    log_sum = tensor_.logsumexp(dim)
    if use_gpu:
        log_mean = log_sum - torch.log(torch.Tensor([tensor_.shape[dim]]).cuda())
    else:
        log_mean = log_sum - torch.log(torch.Tensor([tensor_.shape[dim]]))
    return log_mean


