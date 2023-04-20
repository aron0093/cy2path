import time
import torch
from tqdm.auto import tqdm

from ..utils import log_domain_mean
from .methods import compute_log_likelihood   

def train(self, D, TPM=None, num_epochs=300, sparsity_weight=1.0,
          exclusivity_weight=1.0, optimizer=None, criterion=None, 
          swa_scheduler=None, swa_start=200, verbose=False):
    
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
    exclusivity_weight : float (default: 1.0)
        Regularisation weight for orthogonal EM.
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
    self.exclusivity_weight = exclusivity_weight
        
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

    try: assert self.exclusivity_values
    except: self.exclusivity_values = []
    
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

        compute_log_likelihood(self)
        self.likelihood_values.append(self.log_likelihood.item())

        identity = torch.eye(self.num_states)
        if self.is_cuda:
            identity = identity.cuda()

        sparsity = self.criterion(self.log_transition_matrix, identity)
        loss += self.sparsity_weight*sparsity
        self.sparsity_values.append(sparsity.item())
                        
        exclusivity = self.criterion(torch.corrcoef(self.log_emission_matrix[:,0].exp()).log(),
                                     identity)
        loss += self.exclusivity_weight*exclusivity
        self.exclusivity_values.append(exclusivity.item())

        if TPM is not None:
            if self.is_cuda:
                TPM = TPM.cuda()
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
            self.avg_corrcoeff = torch.mean(torch.tensor(corrcoeffs)).item()

            # Print Loss
            if verbose:
                if TPM is not None:
                    print('{:.2f}s. It {} Loss {:.2E} KL {:.2E} Likl {:.2E} Sparse {:.2E} Reg {:.2E} Exl {:.2E} Corcoef {:.2f}'.format(time.time() - start_time,
                                                                                self.elapsed_epochs, 
                                                                                self.loss_values[-1],
                                                                                self.divergence_values[-1],
                                                                                self.likelihood_values[-1],
                                                                                self.sparsity_values[-1],
                                                                                self.regularisation_values[-1],
                                                                                self.exclusivity_values[-1],
                                                                                self.avg_corrcoeff))
                else:
                    print('{:.2f}s. It {} Loss {:.2E} KL {:.2E} Likl {:.2E} Sparse {:.2E} Exl {:.2E} Corcoef {:.2f}'.format(time.time() - start_time,
                                                                                self.elapsed_epochs, 
                                                                                self.loss_values[-1],
                                                                                self.divergence_values[-1],
                                                                                self.likelihood_values[-1],                                                    
                                                                                self.sparsity_values[-1],
                                                                                self.exclusivity_values[-1],
                                                                                self.avg_corrcoeff))
            
        
