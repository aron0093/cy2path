import numpy as np

# https://en.wikipedia.org/wiki/Absorbing_Markov_chain#Fundamental_matrix

def calculate_fundamental(T, transition_state_indices):
    Q = T[transition_state_indices, :][:, transition_state_indices]
    N = np.linalg.inv(np.eye(Q.shape[0]) - Q.todense())
    return N

def calculate_variance_fundamental(N):
    raise NotImplementedError()

def calculate_expected_steps(N):
    t = N.sum(axis=1).flatten()
    return t

def calculate_variance_expected_steps(N, t):
    raise NotImplementedError()
    
def calculate_transient_probabilities(N):
    H = np.dot((N - np.eye(N.shape[0])), np.linalg.inv(np.diag(np.diag(N))))
    return H
    
def calculate_absorbing_probabilities(N, T, transition_state_indices, terminal_state_indices):
    B = np.dot(N, T[transition_state_indices, :][:, terminal_state_indices].todense())
    return B

