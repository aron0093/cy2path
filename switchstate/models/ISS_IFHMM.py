import time
import torch
#torch.set_default_dtype(torch.float64)
from ..utils import log_domain_matmul, log_domain_mean

class ISS_IFHMM(torch.nn.Module):
    
    ''' 
    Independent chain Factorial hidden Markov model 
    with a chain specific hidden state space trained on 
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

    P(node/iter) = P(node/iter, state, chain)P(state/chain)P(chain/iter)
    P(state/chain) is parametarised as a HMM i.e. P(state_current/state_previous)

    '''
    
    def __init__(self, num_states, num_chains, num_nodes, num_iters, use_gpu=False):

        super().__init__()
        self.num_nodes = num_nodes
        self.num_chains = num_chains
        self.num_states = num_states
        self.num_iters = num_iters
        self.use_gpu = use_gpu

        # Intialise the weights for each chain towards final output
        self.unnormalized_chain_weights = torch.nn.Parameter(torch.randn(
                                                             self.num_iters,
                                                             self.num_chains
                                                             ))
        
        self.unnormalized_state_init = torch.nn.ParameterList()
        self.unnormalized_emission_matrix = torch.nn.ParameterList()
        self.unnormalized_transition_matrix = torch.nn.ParameterList()
        for c in range(self.num_chains):
            # Initial probability of a chain being in any given hidden state
            self.unnormalized_state_init.append(torch.nn.Parameter(torch.randn(1, self.num_states[c])))



            # Initialise emission matrix of states w.r.t. observed states
            self.unnormalized_emission_matrix.append(torch.nn.Parameter(torch.randn(
                                                                self.num_states[c],
                                                                self.num_nodes,
                                                                )))

            # Initialise transition probability matrix between hidden states
            self.unnormalized_transition_matrix.append(torch.nn.Parameter(torch.randn(
                                                                    self.num_states[c],
                                                                    self.num_states[c]
                                                                    )))

        # use the GPU
        self.is_cuda = torch.cuda.is_available() and self.use_gpu
        if self.is_cuda: 
            self.cuda()

    def forward_model(self):
    
        # Normalise across states for each chain
        log_state_init = [torch.nn.functional.log_softmax(self.unnormalized_state_init[c], dim=1) for c in range(self.num_chains)]

        # Normalise chain weights
        log_chain_weights = torch.nn.functional.log_softmax(self.unnormalized_chain_weights, 1)
        
        # Normalise across nodes for each state

        log_emission_matrix = [torch.nn.functional.log_softmax(self.unnormalized_emission_matrix[c], dim=1) for c in range(self.num_chains)]

        # Normalise TPM so that they are probabilities (log space)
        log_transition_matrix = [torch.nn.functional.log_softmax(self.unnormalized_transition_matrix[c], dim=1) for c in range(self.num_chains)]

        # MSM iteration wise probability calculation
        log_hidden_state_probs_ = [torch.zeros(self.num_iters, 1, self.num_states[c]) for c in range(self.num_chains)]
        log_observed_state_probs_ = torch.zeros(self.num_iters, self.num_chains, self.num_nodes)

        if self.is_cuda: 
            log_hidden_state_probs_ = [log_hidden_state_probs_[c].cuda() for c in range(self.num_chains)]
            log_observed_state_probs_ = log_observed_state_probs_.cuda()

        # Initialise at iteration 0
        for c in range(self.num_chains):
            log_hidden_state_probs_[c][0, :, :] = log_state_init[c]
            log_observed_state_probs_[0, c, :] =  log_domain_matmul(log_hidden_state_probs_[c][0, :, :],
                                                                    log_emission_matrix[c]) 
            for t in range(1, self.num_iters):
                log_hidden_state_probs_[c][t, :, :] = log_domain_matmul(log_hidden_state_probs_[c][t-1, :, :], 
                                                                        log_transition_matrix[c])
                log_observed_state_probs_[t, c, :] = log_domain_matmul(log_hidden_state_probs_[c][t, :, :],
                                                                       log_emission_matrix[c])
                #log_observed_state_probs_[t, :self.num_chains, :] = log_emission_matrix[log_hidden_state_probs_[t, :, :].argmax(dim=1)]

        # Predicted log MSM probabilities
        log_observed_state_probs = log_observed_state_probs_  + log_chain_weights.unsqueeze(-1)
        log_observed_state_probs = log_observed_state_probs.logsumexp(1)

        return log_observed_state_probs, log_observed_state_probs_, log_hidden_state_probs_

    def train(self, D, num_epochs=1000, optimizer=None, criterion=None, swa_scheduler=None, swa_start=200):
        
        '''
        Train the model.
        
        Parameters
        ----------
        D : FloatTensor of shape (MSM_nodes, T_max)
            MSM simulation data.
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
            prediction = self.forward_model()[0]
            loss = self.criterion(prediction, D)
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
                print('Time: {:.2f}s. Iteration: {}. Loss: {}. Corrcoeff: {}.'.format(time.time() - start_time,
                                                                                self.elapsed_epochs, 
                                                                                loss.item(), 
                                                                                avg_corrcoeff))
                
            
