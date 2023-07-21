import torch
from ..utils import log_domain_mean

def log_transform_params(self):

    # Normalise across states 
    self.log_state_init = torch.nn.functional.log_softmax(self.unnormalized_state_init, dim=0)

    # Normalise chain weights
    self.log_chain_weights = torch.nn.functional.log_softmax(self.unnormalized_chain_weights, dim=-1)
    
    # Normalise across chains for each state
    self.log_emission_matrix = torch.nn.functional.log_softmax(self.unnormalized_emission_matrix, dim=-1)

    # Normalise TPM so that they are probabilities (log space)
    self.log_transition_matrix = torch.nn.functional.log_softmax(self.unnormalized_transition_matrix, dim=-1)

def compute_log_likelihood(self):

    prediction, log_observed_state_probs_, log_hidden_state_probs = self.forward_model()

    self.log_likelihood = log_domain_mean(log_domain_mean((log_observed_state_probs_ - \
                           log_observed_state_probs_.logsumexp(-1, keepdims=True)).sum(0), 
                           use_gpu=self.is_cuda), use_gpu=self.is_cuda).logsumexp(0)

def compute_aic(self):

    self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    compute_log_likelihood(self)

    self.aic = 2*(self.num_params - self.log_likelihood).detach().cpu().numpy()




