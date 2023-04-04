import time
import torch
from ..utils import log_domain_mean, JSDLoss, revgrad

class IFHMM(torch.nn.Module):
    
    ''' 
    Independent chain Factorial latent dynamic model
    with a common hidden state space trained on 
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
    sparsity_weight : float (default: 1.0)
        Regularisation weight for sparse latent TPM.
    use_gpu : Bool (default: False)
        Toggle GPU use.

    P(node/iter) = P(node/state, chain, iter)P(state/chain, iter)P(chain/iter)
    P(state/iter) is parametarised as a HMM i.e. P(state_current/state_previous)

    '''
    
    def __init__(self, num_states, num_chains, num_nodes, num_iters, 
                 sparsity_weight = 1.0, use_gpu=False):

        super().__init__()
        self.num_nodes = num_nodes
        self.num_chains = num_chains
        self.num_states = num_states
        self.num_iters = num_iters
        self.sparsity_weight = sparsity_weight
        self.use_gpu = use_gpu
        
        # Initial probability of being in any given hidden state
        self.unnormalized_state_init = torch.nn.Parameter(torch.randn(self.num_states, self.num_chains))

        # Intialise the weights of each node towards each chain
        # Fix lineage likelihood
        self.unnormalized_chain_weights = torch.nn.Parameter(torch.randn(#self.num_nodes,
                                                                         self.num_chains
                                                                         ))

        # Initialise emission matrix
        # Enforce common state space
        self.unnormalized_emission_matrix = torch.nn.Parameter(torch.randn(self.num_states,
                                                                           #self.num_chains,
                                                                           self.num_nodes
                                                                          ))
                
        # Initialise transition probability matrix between hidden states
        self.unnormalized_transition_matrix = torch.nn.Parameter(torch.randn(self.num_states,
                                                                             self.num_states
                                                                            ))

        # use the GPU
        self.is_cuda = torch.cuda.is_available() and self.use_gpu
        if self.is_cuda: 
            self.cuda()

    def forward_model(self):
    
        # Normalise across states 
        log_state_init = torch.nn.functional.log_softmax(self.unnormalized_state_init, dim=0)

        # Normalise chain weights
        log_chain_weights = torch.nn.functional.log_softmax(self.unnormalized_chain_weights, dim=-1)
        
        # Normalise across chains for each state
        log_emission_matrix = torch.nn.functional.log_softmax(self.unnormalized_emission_matrix, dim=-1)

        # Normalise TPM so that they are probabilities (log space)
        log_transition_matrix = torch.nn.functional.log_softmax(self.unnormalized_transition_matrix, dim=1)

        # MSM iteration wise probability calculation
        log_hidden_state_probs = torch.zeros(self.num_iters, self.num_states, self.num_chains)
        log_observed_state_probs_ = torch.zeros(self.num_iters, self.num_states, self.num_chains, self.num_nodes)

        if self.is_cuda: 
            log_hidden_state_probs = log_hidden_state_probs.cuda()
            log_observed_state_probs_ = log_observed_state_probs_.cuda()

        # Initialise at iteration 0
        log_hidden_state_probs[0] = log_state_init 
        log_observed_state_probs_[0] = log_emission_matrix.unsqueeze(-1).permute(0,-1,1) + \
                                       log_hidden_state_probs[0].unsqueeze(-1) #
        
        for t in range(1, self.num_iters):
            log_hidden_state_probs[t] = (log_hidden_state_probs[t-1].transpose(1,0).unsqueeze(-1) + \
                                                          log_transition_matrix.unsqueeze(0)).logsumexp(1).transpose(1,0)                      
            log_observed_state_probs_[t] = log_emission_matrix.unsqueeze(-1).permute(0,-1,1) + \
                                           log_hidden_state_probs[t].unsqueeze(-1)

        # Simple average over lineages                            
        log_observed_state_probs = (log_observed_state_probs_.logsumexp(1) + \
                                    log_chain_weights.repeat(self.num_iters, 1).unsqueeze(-1)).logsumexp(1)

        return log_observed_state_probs, log_observed_state_probs_, log_hidden_state_probs, log_emission_matrix, log_chain_weights

    def train(self, D, TPM=None, num_epochs=300, optimizer=None, criterion=None, swa_scheduler=None, swa_start=200, verbose=False):
        
        '''
        Train the model.
        
        Parameters
        ----------
        D : FloatTensor of shape (MSM_nodes, T_max)
            MSM simulation data.
        TPM : Transition probability matrix
            Used to regularise emission probabilities.
        num_epochs : int (default: 1000)
            Number of training epochs
        optimizer : (default: Adam(lr=0.1))
            Optimizer algorithm.
        criterion : (default: KLDivLoss())
            Loss function. (Default preferred)
        swa_scheduler : (default: None)
            SWA scheduler.
        swa_start : int (default: 200)
            Training epoch to start SWA.
        verbose : bool (default: False)
            Toggle printing of training history
        
        '''

        if self.is_cuda:
            D = D.cuda()
         
        try: assert self.elapsed_epochs
        except: self.elapsed_epochs = 0
        
        try: assert self.loss_values
        except: self.loss_values = []

        try: assert self.reconstruction_values
        except: self.reconstruction_values = []

        try: assert self.sparsity_values
        except: self.sparsity_values = []

        try: assert self.regularisation_values
        except: self.regularisation_values = []

        try: assert self.independence_values
        except: self.independence_values = []
        
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.1)
        self.swa_scheduler = swa_scheduler
        self.criterion = criterion
        if self.criterion is None:
            self.criterion = torch.nn.KLDivLoss(reduction='batchmean', log_target=False)
                
        start_time = time.time()           
        for epoch in range(num_epochs):

            self.optimizer.zero_grad()
            prediction, log_observed_state_probs_, log_hidden_state_probs, log_emission_matrix, log_chain_weights = self.forward_model()

            loss = self.criterion(prediction, D)
            reconstruction = loss.item()
            self.reconstruction_values.append(reconstruction)

            sparsity = self.criterion(torch.nn.functional.log_softmax(self.unnormalized_transition_matrix, dim=1),
                                                                      torch.eye(self.num_states))
            loss += self.sparsity_weight*sparsity
            self.sparsity_values.append(sparsity.item())

            log_chain_observed_state_probs = log_domain_mean(log_observed_state_probs_.logsumexp(1),0) + \
                                             log_chain_weights.unsqueeze(-1)
            log_chain_observed_state_probs = log_chain_observed_state_probs - \
                                             log_chain_observed_state_probs.logsumexp(0).unsqueeze(0)                           
            independence = JSDLoss(reduction='batchmean').forward(log_chain_observed_state_probs)
            loss += revgrad(independence, torch.tensor([1.]))
            self.independence_values.append(independence.item())

            if TPM is not None:
                regularisation =  self.criterion(torch.nn.functional.log_softmax(self.unnormalized_emission_matrix, dim=-1),
                                                 torch.matmul(torch.exp(log_emission_matrix.detach()), TPM))
                loss += regularisation
                self.regularisation_values.append(regularisation)

            loss.backward()
            self.optimizer.step()
            if self.elapsed_epochs > swa_start and self.swa_scheduler is not None:
                swa_scheduler.step()
            
            self.loss_values.append(loss.item())

            self.elapsed_epochs += 1
            if epoch % 10 == 0 or epoch <=10:
                corrcoeffs = []
                outputs = torch.exp(prediction)
                for t in range(self.num_iters):
                    corrcoeffs.append(torch.corrcoef(torch.stack([outputs[t], D[t]]))[1,0])
                avg_corrcoeff = torch.mean(torch.tensor(corrcoeffs))

                # Print Loss
                if verbose:
                    if TPM is not None:
                        print('Time: {:.2f}s. Iteration: {}. Loss: {:.2E}. Recons: {:.2E}. Sparsity: {:.2E}. Reg: {:.2E}. Ind: {:.2E}. Corrcoeff: {:.2f}.'.format(time.time() - start_time,
                                                                                    self.elapsed_epochs, 
                                                                                    self.loss_values[-1],
                                                                                    self.reconstruction_values[-1],
                                                                                    self.sparsity_values[-1],
                                                                                    self.regularisation_values[-1],
                                                                                    self.independence_values[-1],
                                                                                    avg_corrcoeff))
                    else:
                        print('Time: {:.2f}s. Iteration: {}. Loss: {:.2E}. Recons: {:.2E}. Sparsity: {:.2E}. Ind: {:.2E}. Corrcoeff: {:.2f}.'.format(time.time() - start_time,
                                                                                    self.elapsed_epochs, 
                                                                                    self.loss_values[-1],
                                                                                    self.reconstruction_values[-1],
                                                                                    self.sparsity_values[-1],
                                                                                    self.independence_values[-1],
                                                                                    avg_corrcoeff))
                
            
