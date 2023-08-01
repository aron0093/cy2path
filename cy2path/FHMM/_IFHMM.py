import torch

from .methods import log_transform_params
from .trainer import train

from ..utils import log_domain_mean

class IFHMM(torch.nn.Module):
    
    ''' 
    Latent dynamic model with a common latent state space 
    and conditional state transitions per chaon trained on 
    a Markov state simulation of an observed state TPM
    with SGD.
    
    Parameters
    ----------
    num_states : int
        Number of hidden states.
    num_chains : int
        Number of hidden Markov chains.
    num_nodes : int
        Number of observed states in the MSM simulation.
    num_iters : int
        Number of iterations of the MSM simulation.
    restricted: Bool (default: False)
        Condition emission matrix on chains.
    use_gpu : Bool (default: False)
        Toggle GPU use.

    P(node | iter) = sigma_chain sigma_state P(node | state, chain, iter)P(state | chain, iter)P(chain | iter)
    P(state | iter) is parametarised as a HMM i.e. P(state_current | state_previous)

    '''
    
    def __init__(self, num_states, num_chains, num_nodes, num_iters, restricted=True, use_gpu=False):

        super().__init__()
        self.num_nodes = num_nodes
        self.num_chains = num_chains
        self.num_states = num_states
        self.num_iters = num_iters
        self.restricted = restricted
        self.use_gpu = use_gpu
        
        # Initial probability of being in any given hidden state
        self.unnormalized_state_init = torch.nn.Parameter(torch.randn(self.num_states, self.num_chains))

        # Intialise the weights of each node towards each chain
        self.unnormalized_chain_weights = torch.nn.Parameter(torch.randn(self.num_iters,
                                                                         self.num_chains,
                                                                         ))

        # Initialise emission matrix
        if self.restricted:
            em_init = torch.randn(self.num_states, 1, self.num_nodes)
        else:
            em_init = torch.randn(self.num_states, self.num_chains, self.num_nodes)           
        self.unnormalized_emission_matrix = torch.nn.Parameter(em_init)
                
        # Initialise conditional transition probability matrices between hidden states
        self.unnormalized_transition_matrix = torch.nn.Parameter(torch.randn(self.num_chains,
                                                                             self.num_states,
                                                                             self.num_states
                                                                            ))

        # use the GPU
        self.is_cuda = torch.cuda.is_available() and self.use_gpu
        if self.is_cuda: 
            self.cuda()

    def forward_model(self):
    
        log_transform_params(self)
        self.log_transition_matrix_ = self.log_transition_matrix

        # P(l/iter)
        self.log_transition_matrix = (self.log_transition_matrix_ + \
                                     log_domain_mean(self.log_chain_weights, use_gpu=self.is_cuda).unsqueeze(-1).unsqueeze(-1)).logsumexp(0) 

        # P(l)
        # self.log_transition_matrix = (self.log_transition_matrix_ + self.log_chain_weights.unsqueeze(-1).unsqueeze(-1)).logsumexp(0)

        # MSM iteration wise probability calculation
        log_hidden_state_probs = torch.zeros(self.num_iters, self.num_states, self.num_chains)
        log_observed_state_probs_ = torch.zeros(self.num_iters, self.num_states, self.num_chains, self.num_nodes)

        if self.is_cuda: 
            log_hidden_state_probs = log_hidden_state_probs.cuda()
            log_observed_state_probs_ = log_observed_state_probs_.cuda()

        # Initialise at iteration 0
        log_hidden_state_probs[0] = self.log_state_init 
        log_observed_state_probs_[0] = self.log_emission_matrix + \
                                       log_hidden_state_probs[0].unsqueeze(-1) #
        
        for t in range(1, self.num_iters):
            log_hidden_state_probs[t] = (log_hidden_state_probs[t-1].transpose(1,0).unsqueeze(-1) + \
                                        self.log_transition_matrix_).logsumexp(1).transpose(1,0)
            
                                  
            log_observed_state_probs_[t] = self.log_emission_matrix + \
                                           log_hidden_state_probs[t].unsqueeze(-1)
        
        # Joint probabilities

        # P(l/iter)
        log_observed_state_probs_ = log_observed_state_probs_ + \
                                    self.log_chain_weights.unsqueeze(1).unsqueeze(-1)

        # P(l)
        # log_observed_state_probs_ = log_observed_state_probs_ + \
        #                     self.log_chain_weights.repeat(self.num_iters, 1).unsqueeze(1).unsqueeze(-1)

        # Combine lineages                            
        log_observed_state_probs = log_observed_state_probs_.logsumexp(1).logsumexp(1)

        return log_observed_state_probs, log_observed_state_probs_, log_hidden_state_probs
    
    # Conditional hidden state probabilities given observed sequence till current iteration
    def filtering(self, D):

        log_transform_params(self)

        log_alpha = torch.zeros(self.num_iters, self.num_states, self.num_chains)
        log_probs = torch.zeros(self.num_iters, self.num_chains)
        if self.is_cuda: 
            log_alpha = log_alpha.cuda()
            log_probs = log_probs.cuda()
            D = D.cuda()

        # Initialise at iteration 0
        log_alpha[0] = (D.log()[0].unsqueeze(0).unsqueeze(0) + self.log_emission_matrix + \
                        self.log_state_init.unsqueeze(-1)).logsumexp(-1)
        log_probs[0] = log_alpha[0].clone().detach().logsumexp(0)
        log_alpha[0] = log_alpha[0] - log_probs[0].unsqueeze(0)
        
        for t in range(1, self.num_iters):
            log_alpha[t] = (D.log()[t].unsqueeze(0).unsqueeze(0) + self.log_emission_matrix + \
                            (log_alpha[t-1].transpose(1,0).unsqueeze(-1) + \
                            self.log_transition_matrix_).logsumexp(1).transpose(1,0).unsqueeze(-1)).logsumexp(-1)  
            log_probs[t] = log_alpha[t].clone().detach().logsumexp(0)
            log_alpha[t] = log_alpha[t] - log_probs[t].unsqueeze(0)

        return log_alpha, log_probs

    # Conditional hidden state probabilities given an observed sequence
    def smoothing(self, D):

        log_transform_params(self)

        log_alpha, log_probs = self.filtering(D)
        log_beta = torch.zeros(self.num_iters, self.num_states, self.num_chains)
        log_gamma = torch.zeros(self.num_iters, self.num_states, self.num_chains)

        init = torch.tensor([1.0]*self.num_states).log()

        if self.is_cuda: 
            log_beta = log_beta.cuda()
            log_gamma = log_gamma.cuda()
            D = D.cuda()
            init = init.cuda()

        # Initialise at iteration 0
        log_beta[-1] = (self.log_transition_matrix_.permute((1,2,0)) + \
                       (D.log()[-1].unsqueeze(0).unsqueeze(0) + self.log_emission_matrix + \
                       init.unsqueeze(-1).unsqueeze(-1)).logsumexp(-1).unsqueeze(0)).logsumexp(1)
        
        log_gamma[-1] = log_beta[-1] - log_probs.sum(0).unsqueeze(0) + log_alpha[-1] 
        
        for t in range(self.num_iters-1, 0, -1):
            log_beta[t-1] = (self.log_transition_matrix_.permute((1,2,0)) + \
                            (D.log()[-1].unsqueeze(0).unsqueeze(0) + self.log_emission_matrix + \
                            log_beta[t].unsqueeze(-1)).logsumexp(-1).unsqueeze(0)).logsumexp(1)
            
            log_gamma[t-1] = log_beta[t-1] - log_probs.sum(0).unsqueeze(0) + log_alpha[t-1] 

        return log_beta, log_gamma
        
    # Viterbi decoding for best path
    def viterbi(self, D):

        log_transform_params(self)

        log_delta = torch.zeros(self.num_iters, self.num_states, self.num_chains)
        psi = torch.zeros(self.num_iters, self.num_states, self.num_chains)

        if self.is_cuda: 
            log_delta = log_delta.cuda()
            psi = psi.cuda()
            D = D.cuda()

        log_delta[0] = (D.log()[0].unsqueeze(0).unsqueeze(0) + self.log_emission_matrix + \
                        self.log_state_init.unsqueeze(-1)).logsumexp(-1)
        
        for t in range(1, self.num_iters):
            max_val, argmax_val = torch.max(log_delta[t-1].transpose(1,0).unsqueeze(-1) + \
                                            self.log_transition_matrix_, dim=1)

            log_delta[t] = (D.log()[t].unsqueeze(0).unsqueeze(0) + self.log_emission_matrix + \
                           max_val.transpose(1,0).unsqueeze(0).unsqueeze(-1)).logsumexp(-1)
            psi[t] = argmax_val.transpose(1,0)

        # Get the log probability of the best paths
        log_max = log_delta.max(dim=1)[0]

        best_path = []
        for i in range(0, self.num_chains):
            best_path_i = [log_delta[self.num_iters-1, :, i].max(dim=-1)[1].item()]
            for t in range(self.num_iters-1, 0, -1):
                latent_t = psi[t, int(best_path_i[0]), i].item()
                best_path_i.insert(0, latent_t) 

            best_path.append(best_path_i)

        return log_delta, psi, log_max, best_path
    

    # Train the model
    def train(self, D, TPM=None, num_epochs=300, sparsity_weight=1.0, exclusivity_weight=1.0, orthogonality_weight=1e-1,
              optimizer=None, criterion=None, swa_scheduler=None, swa_start=200, verbose=False):
        train(self, D, TPM=TPM, num_epochs=num_epochs, sparsity_weight=sparsity_weight, exclusivity_weight=exclusivity_weight,
              orthogonality_weight=orthogonality_weight, optimizer=optimizer, criterion=criterion, swa_scheduler=swa_scheduler, 
              swa_start=swa_start, verbose=verbose)
            