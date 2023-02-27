import time
import torch
#torch.set_default_dtype(torch.float64)
from ..utils import log_domain_matmul, log_domain_mean

class SSM(torch.nn.Module):
    
    ''' 
    Switch state hidden Markov model trained on 
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
    use_gpu : Bool (default: False)
        Toggle GPU use.

    P(node/iter) = P(node/state, chain, iter)P(chain/state, iter)P(state/iter)
    P(state/iter) is parametarised as a HMM i.e. P(state_current/state_previous)

    '''
    
    def __init__(self, num_states, num_chains, num_nodes, num_iters, 
                 emissions_mask=None, use_gpu=False):

        super().__init__()
        self.num_nodes = num_nodes
        self.num_chains = num_chains
        self.num_states = num_states
        self.num_iters = num_iters
        self.use_gpu = use_gpu
        
        # Initial probability of being in any given hidden state
        self.unnormalized_state_init = torch.nn.Parameter(torch.randn(self.num_states))

        # Intialise the weights of each node towards each chain
        # Removed dependency on iter
        self.unnormalized_chain_weights = torch.nn.Parameter(torch.randn(self.num_states,
                                                                         self.num_chains,
                                                                            
                                                                        ))

        # Initialise emission matrix -> common for either chain
        self.unnormalized_emission_matrix = torch.nn.Parameter(torch.randn(self.num_states,
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
            self.emissions_mask = self.emissions_mask.cuda()

    def forward_model(self):
    
        # Normalise across states 
        log_state_init = torch.nn.functional.log_softmax(self.unnormalized_state_init, dim=0)

        # Normalise chain weights
        log_chain_weights = torch.nn.functional.log_softmax(self.unnormalized_chain_weights, dim=1)
        
        # Normalise across chains for each state
        log_emission_matrix = torch.nn.functional.log_softmax(self.unnormalized_emission_matrix, dim=1) 

        # Normalise TPM so that they are probabilities (log space)
        log_transition_matrix = torch.nn.functional.log_softmax(self.unnormalized_transition_matrix, dim=1)

        # MSM iteration wise probability calculation
        log_hidden_state_probs = torch.zeros(self.num_iters, self.num_states)
        log_observed_state_probs_ = torch.zeros(self.num_iters, self.num_states, self.num_chains, self.num_nodes)

        if self.is_cuda: 
            log_hidden_state_probs = log_hidden_state_probs.cuda()
            log_observed_state_probs_ = log_observed_state_probs_.cuda()

        # Initialise at iteration 0
        log_hidden_state_probs[0] = log_state_init
        log_observed_state_probs_[0] = (torch.permute(log_emission_matrix.repeat(2,1,1), (1,0,-1)) + \
                                                      log_chain_weights.unsqueeze(-1)) + \
                                                      log_hidden_state_probs[[0]].transpose(1,0).unsqueeze(-1)
        
        for t in range(1, self.num_iters):
            log_hidden_state_probs[t] = log_domain_matmul(log_hidden_state_probs[[t-1]], log_transition_matrix)                                   
            log_observed_state_probs_[t] = (torch.permute(log_emission_matrix.repeat(2,1,1), (1,0,-1)) + \
                                                      log_chain_weights.unsqueeze(-1)) + \
                                                      log_hidden_state_probs[[t]].transpose(1,0).unsqueeze(-1)

        log_observed_state_probs = log_observed_state_probs_.logsumexp(1).logsumexp(1)

        return log_observed_state_probs, log_observed_state_probs_, log_hidden_state_probs, log_emission_matrix, log_chain_weights

    def train(self, D, TPM=None, num_epochs=1000, optimizer=None, criterion=None, swa_scheduler=None, swa_start=200):
        
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
        
        '''

        if self.is_cuda:
            D = D.cuda()
         
        try: assert self.elapsed_epochs
        except: self.elapsed_epochs = 0
        
        try: assert self.loss_values
        except: self.loss_values = []

        try: assert self.penality_values
        except: self.penality_values = []

        try: assert self.sparsity_values
        except: self.sparsity_values = []
        
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

            if TPM is not None:
                loss_ = self.criterion(prediction, D) 

                self.penalty_criterion = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)

                log_state_emission_matrix = torch.transpose(log_emission_matrix, 1,0)
                log_transition_emission_matrix = log_domain_matmul(TPM.log(), log_state_emission_matrix.detach())
                penalty = 1000*self.penalty_criterion(log_transition_emission_matrix, log_state_emission_matrix)

                loss = loss_ + penalty
                self.penality_values.append(penalty.item())

            else:
                loss = self.criterion(prediction, D) + self.criterion(torch.nn.functional.log_softmax(self.unnormalized_transition_matrix, dim=1),
                                                                      torch.eye(self.num_states))

            
            sparsity = self.criterion(torch.nn.functional.log_softmax(self.unnormalized_transition_matrix, dim=1),
                                                                      torch.eye(self.num_states))
            loss += sparsity
            self.sparsity_values.append(sparsity.item())

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
                if TPM is not None:
                    print('Time: {:.2f}s. Iteration: {:.2E}. Loss: {:.2E}. Sparsity: {:.2E}. Penalty: {:.2E}. Corrcoeff: {:.2E}.'.format(time.time() - start_time,
                                                                                self.elapsed_epochs, 
                                                                                loss.item(), 
                                                                                sparsity.item(),
                                                                                penalty.item(),
                                                                                avg_corrcoeff))
                else:
                    print('Time: {:.2f}s. Iteration: {:.2E}. Loss: {:.2E}. Sparsity: {:.2E}. Corrcoeff: {:.2E}.'.format(time.time() - start_time,
                                                                                self.elapsed_epochs, 
                                                                                loss.item(),
                                                                                sparsity.item(),
                                                                                avg_corrcoeff))
                
            
