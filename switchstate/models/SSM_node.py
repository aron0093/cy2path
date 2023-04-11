import time
import torch
from tqdm.auto import tqdm
from ..utils import log_domain_mean

class SSM_node(torch.nn.Module):
    
    ''' 
    Switch state latent dynamic model 
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
    use_gpu : Bool (default: False)
        Toggle GPU use.

    P(node/iter) = sigma_chain sigma_state P(chain/node, state, iter)P(node/state, iter)P(state/iter)
    P(state/iter) is parametarised as a HMM i.e. P(state_current/state_previous)
    '''

    def __init__(self, num_states, num_chains, num_nodes, num_iters, use_gpu=False):

        super().__init__()
        self.num_nodes = num_nodes
        self.num_chains = num_chains
        self.num_states = num_states
        self.num_iters = num_iters
        self.use_gpu = use_gpu
        
        # Initial probability of being in any given hidden state
        self.unnormalized_state_init = torch.nn.Parameter(torch.randn(self.num_states))

        # Intialise the weights of each node towards each chain
        # Enforce common state space
        self.unnormalized_chain_weights = torch.nn.Parameter(torch.randn(self.num_nodes,
                                                                         self.num_chains,
                                                                        ))

        # Initialise emission matrix
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

    # Simulate MSM using latent model
    def forward_model(self):

        # Normalise across states 
        self.log_state_init = torch.nn.functional.log_softmax(self.unnormalized_state_init, dim=-1)

        # Normalise chain weights
        self.log_chain_weights = torch.nn.functional.log_softmax(self.unnormalized_chain_weights, dim=-1)
        
        # Normalise across chains for each state
        self.log_emission_matrix = torch.nn.functional.log_softmax(self.unnormalized_emission_matrix, dim=-1)

        # Normalise TPM so that they are probabilities (log space)
        self.log_transition_matrix = torch.nn.functional.log_softmax(self.unnormalized_transition_matrix, dim=-1)
        
        # Joint chain, node
        self.log_weights = self.log_emission_matrix.unsqueeze(-1) + \
                           self.log_chain_weights.unsqueeze(0)

        # MSM iteration wise probability calculation
        log_hidden_state_probs = torch.zeros(self.num_iters, self.num_states)
        log_observed_state_probs_ = torch.zeros(self.num_iters, self.num_states, self.num_nodes, self.num_chains)

        if self.is_cuda: 
            log_hidden_state_probs = log_hidden_state_probs.cuda()
            log_observed_state_probs_ = log_observed_state_probs_.cuda()
            
        # Initialise at iteration 0
        log_hidden_state_probs[0] = self.log_state_init
        log_observed_state_probs_[0] = self.log_weights + log_hidden_state_probs[0].unsqueeze(-1).unsqueeze(-1)
        
        for t in range(1, self.num_iters):
            log_hidden_state_probs[t] = (log_hidden_state_probs[t-1].unsqueeze(-1) + \
                                         self.log_transition_matrix.unsqueeze(0)).logsumexp(1)                                   
            log_observed_state_probs_[t] = self.log_weights + log_hidden_state_probs[t].unsqueeze(-1).unsqueeze(-1) 

        log_observed_state_probs = log_observed_state_probs_.logsumexp(1).logsumexp(-1)

        # Reshape output for compatibility with other models
        log_observed_state_probs_ = log_observed_state_probs_.permute(0,1,3,2)

        return log_observed_state_probs, log_observed_state_probs_, log_hidden_state_probs
    
    # Conditional hidden state probabilities given observed sequence till current iteration
    def filtering(self, D):

        log_alpha = torch.zeros(self.num_iters, self.num_states, self.num_chains)
        log_probs = torch.zeros(self.num_iters, self.num_chains)
        if self.is_cuda: 
            log_alpha = log_alpha.cuda()
            log_probs = log_probs.cuda()

        # Initialise at iteration 0
        log_alpha[0] = (D.log()[0].unsqueeze(0).unsqueeze(-1) + self.log_weights + \
                        self.log_state_init.unsqueeze(-1).unsqueeze(-1)).logsumexp(1)
        log_probs[0] = log_alpha[0].clone().detach().logsumexp(0)
        log_alpha[0] = log_alpha[0] - log_probs[0].unsqueeze(0)
        
        for t in range(1, self.num_iters):
            log_alpha[t] = (D.log()[t].unsqueeze(0).unsqueeze(-1) + self.log_weights + \
                            (log_alpha[t-1].transpose(1,0).unsqueeze(-1) + \
                            self.log_transition_matrix.unsqueeze(0)).logsumexp(1).transpose(1,0).unsqueeze(1)).logsumexp(1)   
            log_probs[t] = log_alpha[t].clone().detach().logsumexp(0)
            log_alpha[t] = log_alpha[t] - log_probs[t].unsqueeze(0)

        return log_alpha, log_probs

    # Conditional hidden state probabilities given an observed sequence
    def smoothing(self, D):

        log_alpha, log_probs = self.filtering(D)
        log_beta = torch.zeros(self.num_iters, self.num_states, self.num_chains)
        log_gamma = torch.zeros(self.num_iters, self.num_states, self.num_chains)

        if self.is_cuda: 
            log_beta = log_beta.cuda()
            log_gamma = log_gamma.cuda()

        # Initialise at iteration 0
        log_beta[-1] = (self.log_transition_matrix.unsqueeze(-1) + \
                       (D.log()[-1].unsqueeze(0).unsqueeze(-1) + self.log_weights + \
                       torch.tensor([1.0]*self.num_states).log().unsqueeze(-1).unsqueeze(-1)).logsumexp(1).unsqueeze(0)).logsumexp(1)
        
        log_gamma[-1] = log_beta[-1] - log_probs.sum(0).unsqueeze(0) + log_alpha[-1] 
        
        for t in range(self.num_iters-1, 0, -1):
            log_beta[t-1] = (self.log_transition_matrix.unsqueeze(-1) + \
                            (D.log()[-1].unsqueeze(0).unsqueeze(-1) + self.log_weights + \
                            log_beta[t].unsqueeze(1)).logsumexp(1).unsqueeze(0)).logsumexp(1)
            
            log_gamma[t-1] = log_beta[t-1] - log_probs.sum(0).unsqueeze(0) + log_alpha[t-1] 

        return log_beta, log_gamma
        
    # Viterbi decoding for best path
    def viterbi(self, D):

        log_delta = torch.zeros(self.num_iters, self.num_states, self.num_chains)
        psi = torch.zeros(self.num_iters, self.num_states, self.num_chains)

        if self.is_cuda: 
            log_delta = log_delta.cuda()
            psi = psi.cuda()

        log_delta[0] = (D.log()[0].unsqueeze(0).unsqueeze(-1) + self.log_weights + \
                        self.log_state_init.unsqueeze(-1).unsqueeze(-1)).logsumexp(1)

        for t in range(1, self.num_iters):
            max_val, argmax_val = torch.max(log_delta[t-1].transpose(1,0).unsqueeze(-1) + \
                                            self.log_transition_matrix.unsqueeze(0), dim=1)

            log_delta[t] = (D.log()[t].unsqueeze(0).unsqueeze(-1) + self.log_weights + \
                           max_val.transpose(1,0).unsqueeze(1)).logsumexp(1)
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

    def train(self, D, TPM=None, num_epochs=300, sparsity_weight=1.0,
              optimizer=None, criterion=None, swa_scheduler=None, swa_start=200, 
              verbose=False):
        
        '''
        Train the model.
        
        Parameters
        ----------
        D : FloatTensor of shape (MSM_nodes, T_max)
            MSM simulation data.
        TPM : Transition probability matrix
            Used to regularise emission probabilities.
        num_epochs : int (default: 300)
            Number of training epochs
        sparsity_weight : float (default: 1.0)
            Regularisation weight for sparse latent TPM.
        optimizer : (default: RMSProp(lr=0.2))
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
        
        self.sparsity_weight = sparsity_weight
         
        try: assert self.elapsed_epochs
        except: self.elapsed_epochs = 0
        
        try: assert self.loss_values
        except: self.loss_values = []

        try: assert self.divergence_values
        except: self.divergence_values = []
            
        try: assert self.likelihood_values
        except: self.likelihood_values = []

        try: assert self.sparsity_values
        except: self.sparsity_values = []

        try: assert self.regularisation_values
        except: self.regularisation_values = []

        try: assert self.independence_values
        except: self.independence_values = []
        
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.2)
        self.swa_scheduler = swa_scheduler
        self.criterion = criterion
        if self.criterion is None:
            self.criterion = torch.nn.KLDivLoss(reduction='batchmean', log_target=False)
                
        start_time = time.time()           
        for epoch in tqdm(range(num_epochs), desc='Training dynamic model'):

            self.optimizer.zero_grad()
            prediction, log_observed_state_probs_, log_hidden_state_probs = self.forward_model()
            log_alpha, log_probs = self.filtering(D)
            
            divergence = self.criterion(prediction, D)
            loss = divergence
            self.divergence_values.append(divergence.item())

            likelihood = log_alpha.max(1)[0].logsumexp(0).logsumexp(0)
            #loss = -likelihood
            self.likelihood_values.append(likelihood.item())

            sparsity = self.criterion(self.log_transition_matrix, torch.eye(self.num_states))
            loss += self.sparsity_weight*sparsity
            self.sparsity_values.append(sparsity.item())
                         
            independence = log_domain_mean(self.log_chain_weights.sum(-1)).squeeze()
            loss += 1/independence
            self.independence_values.append(independence.item())

            if TPM is not None:
                regularisation = self.criterion(self.log_emission_matrix, 
                                                torch.matmul(torch.exp(self.log_emission_matrix.detach()), TPM))
                loss += regularisation
                self.regularisation_values.append(regularisation.item())

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
                        print('Time: {:.2f}s. Iter: {}. Loss: {:.2E}. KL: {:.2E}. Likl: {:.2E}. Sparse: {:.2E}. Reg: {:.2E}. Ind: {:.2E}. Corcoef: {:.2f}.'.format(time.time() - start_time,
                                                                                    self.elapsed_epochs, 
                                                                                    self.loss_values[-1],
                                                                                    self.divergence_values[-1],
                                                                                    self.likelihood_values[-1],
                                                                                    self.sparsity_values[-1],
                                                                                    self.regularisation_values[-1],
                                                                                    self.independence_values[-1],
                                                                                    avg_corrcoeff))
                    else:
                        print('Time: {:.2f}s. Iter: {}. Loss: {:.2E}. KL: {:.2E}. Likl: {:.2E}. Sparse: {:.2E}. Ind: {:.2E}. Corcoef: {:.2f}.'.format(time.time() - start_time,
                                                                                    self.elapsed_epochs, 
                                                                                    self.loss_values[-1],
                                                                                    self.divergence_values[-1],
                                                                                    self.likelihood_values[-1],                                                    
                                                                                    self.sparsity_values[-1],
                                                                                    self.independence_values[-1],
                                                                                    avg_corrcoeff))
                
            
