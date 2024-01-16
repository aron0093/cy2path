import time
import torch
from tqdm.auto import tqdm

import logging
logging.basicConfig(level = logging.INFO)

from ..utils import log_domain_mean, JSDLoss
from .methods import compute_log_likelihood   

def train(self, D, TPM=None, num_epochs=300, sparsity_weight=1.0,
          exclusivity_weight=0.0, orthogonality_weight=0.0, TPM_weight=0.0,
          optimizer=None, criterion=None, swa_scheduler=None, 
          swa_start=200, verbose=False):
    
    '''
    Train the model.
    
    Parameters
    ----------
    D : FloatTensor of shape (T_max, MSM_nodes)
        MSM simulation data.
    TPM : Transition probability matrix
        Used to regularise emission probabilities.
    num_epochs : int (default: 300)
        Number of training epochs
    sparsity_weight : float (default: 1.0)
        Regularisation weight for sparse latent TPM.
    orthogonality_weight : float (default: 0.0)
        Regularisation weight for orthogonal EM.
    exclusivity_weight : float (default: 0.0)
        Regularisation weight for exclsuive lineages.
    TPM_weight: float (default: 0.0)
        Regularisation weight for TPM.
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

    # Put tensors on correct device
    identity = torch.ones(self.num_states)
    chain_identity = torch.ones(self.num_chains)
    if self.is_cuda:
        D = D.cuda()
        TPM = TPM.cuda()

        identity = identity.cuda()
        chain_identity = chain_identity.cuda()

    # Store regularisation weights and losses     
    self.sparsity_weight = sparsity_weight
    self.orthogonality_weight = orthogonality_weight
    self.exclusivity_weight = exclusivity_weight
    self.TPM_weight = TPM_weight
        
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

    try: assert self.orthogonality_values
    except: self.orthogonality_values = []

    try: assert self.exclusivity_values
    except: self.exclusivity_values = []

    try: assert self.regularisation_values
    except: self.regularisation_values = []
    
    # Optimizer and loss criteria
    self.optimizer = optimizer
    if self.optimizer is None:
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.2)
    self.swa_scheduler = swa_scheduler
    self.criterion = criterion
    if self.criterion is None:
        self.criterion = torch.nn.KLDivLoss(reduction='batchmean', log_target=False)

    # Train        
    start_time = time.time()           
    for epoch in tqdm(range(num_epochs), desc='Training dynamic model', disable=verbose):

        # Reset gradients
        self.optimizer.zero_grad()

        # Model output
        prediction, log_observed_state_probs_, log_hidden_state_probs = self.forward_model()

        # Sum up loss per chain
        divergence = self.criterion(prediction, D)
        loss = divergence
        self.divergence_values.append(divergence.item())

        # Compute likelihood of model given data
        compute_log_likelihood(self)
        self.likelihood_values.append(self.log_likelihood.item())

        # Regularise latent TPM to be sparse
        sparsity = self.criterion(torch.diag(self.log_transition_matrix), identity)
        loss += self.sparsity_weight*sparsity
        self.sparsity_values.append(sparsity.item())

        # Regularise latent states to be exclusive
        log_observed_state_probs_mean = log_domain_mean(log_observed_state_probs_, use_gpu=self.is_cuda)

        log_nodes_given_state = log_observed_state_probs_mean.logsumexp(1) -\
                                log_observed_state_probs_mean.logsumexp(1).logsumexp(-1, keepdims=True)
        orthogonality = JSDLoss(use_gpu=self.is_cuda)(log_nodes_given_state)   

        loss += self.orthogonality_weight*orthogonality
        self.orthogonality_values.append(orthogonality.item())

        if self.num_chains > 1:

            log_nodes_given_chain = log_observed_state_probs_mean.logsumexp(0) -\
                                    log_observed_state_probs_mean.logsumexp(0).logsumexp(-1, keepdims=True)

            log_chains_given_node = log_observed_state_probs_mean.logsumexp(0) -\
                                    log_observed_state_probs_mean.logsumexp(0).logsumexp(0, keepdims=True)
            
            # Compute lineage x lineage transition matrix and minimize transitions
            right_square = torch.matmul(log_nodes_given_chain.exp(), TPM)
            right_square_left = torch.matmul(right_square, log_chains_given_node.transpose(1,0).exp())

            exclusivity = self.criterion(torch.diag(right_square_left.log()), chain_identity)

            loss += self.exclusivity_weight*exclusivity
            self.exclusivity_values.append(exclusivity.item())

        # Regularise latent states to be consider neighborhood transitions using TPM
        regularisation = self.criterion(log_nodes_given_state, 
                                        torch.matmul(torch.exp(log_nodes_given_state.detach()), 
                                                         TPM))
        loss += self.TPM_weight*regularisation
        self.regularisation_values.append(regularisation.item())

        # Backpropagate
        loss.backward()
        self.optimizer.step()
        if self.elapsed_epochs > swa_start and self.swa_scheduler is not None:
            swa_scheduler.step()
        
        self.loss_values.append(loss.item())
        
        # Print training summary
        self.elapsed_epochs += 1
        if epoch % 100 == 99 or epoch <=9:
            corrcoeffs = []
            outputs = torch.exp(prediction)
            for t in range(self.num_iters):
                corrcoeffs.append(torch.corrcoef(torch.stack([outputs[t], D[t]]))[1,0])
            self.avg_corrcoeff = torch.mean(torch.tensor(corrcoeffs)).item()

            # Print Loss
            if verbose:
                if self.TPM_weight!=0.0 and self.num_chains > 1:
                    logging.info('{:.2f}s. It {} Loss {:.2E} KL {:.2E} Likl {:.2E} Sparse {:.2E} Orth {:.2E} Exl {:.2E} Reg {:.2E} Corcoef {:.2f}'.format(time.time() - start_time,
                                                                                self.elapsed_epochs, 
                                                                                self.loss_values[-1],
                                                                                self.divergence_values[-1],
                                                                                self.likelihood_values[-1],
                                                                                self.sparsity_values[-1],
                                                                                self.orthogonality_values[-1],
                                                                                self.exclusivity_values[-1],
                                                                                self.regularisation_values[-1],
                                                                                self.avg_corrcoeff))
                elif self.num_chains > 1:
                    logging.info('{:.2f}s. It {} Loss {:.2E} KL {:.2E} Likl {:.2E} Sparse {:.2E} Orth {:.2E} Exl {:.2E} Corcoef {:.2f}'.format(time.time() - start_time,
                                                                                self.elapsed_epochs, 
                                                                                self.loss_values[-1],
                                                                                self.divergence_values[-1],
                                                                                self.likelihood_values[-1],                                                    
                                                                                self.sparsity_values[-1],
                                                                                self.orthogonality_values[-1],
                                                                                self.exclusivity_values[-1],
                                                                                self.avg_corrcoeff))
                else:
                     logging.info('{:.2f}s. It {} Loss {:.2E} KL {:.2E} Likl {:.2E} Sparse {:.2E} Orth {:.2E} Corcoef {:.2f}'.format(time.time() - start_time,
                                                                                self.elapsed_epochs, 
                                                                                self.loss_values[-1],
                                                                                self.divergence_values[-1],
                                                                                self.likelihood_values[-1],                                                    
                                                                                self.sparsity_values[-1],
                                                                                self.orthogonality_values[-1],
                                                                                self.avg_corrcoeff))                   
            
        
