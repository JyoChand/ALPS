import math
import torch
import torch.nn as nn
import numpy as np


class Denoiser(nn.Module):
    """
    Computes preconditioned psi(x, sigma) required in
    E(x, sigma) = 0.5 ||x - psi(x, sigma)||^2 and the
    vector-Jacobian product for score computation.
    """

    def __init__(self, net):
        """
        Initialize the Denoiser class.

        Args:
            net: conditional network that takes sigma and class label as input
        """
        super().__init__()
        self.net = net

    def forward(self, inputs, sigma):
        """
        Computes psi(x, sigma).

        Args:
            inputs: image whose energy needs to be computed
            sigma: noise level
        """
        y = self.net(inputs, sigma, 0)  # 0: no class label
        return y

    def vjp(self, outputs, inputs, conditioning, vector, precondition):
        """
        Computes vector-Jacobian product for score calculation.

        Args:
            outputs: psi(x)
            inputs: image whose energy needs to be computed
            conditioning: noise level
            vector: x - psi(x)
            precondition: True if one needs to compute preconditioned score
        """
        if precondition:
            vector = vector / conditioning
            grad_JT_eps = torch.autograd.grad(
                outputs=outputs,
                inputs=inputs,
                grad_outputs=vector,
                create_graph=True,
                only_inputs=True
            )[0]
            grad_JT_eps = grad_JT_eps * conditioning  # undo the preconditioning
        else:
            grad_JT_eps = torch.autograd.grad(
                outputs=outputs,
                inputs=inputs,
                grad_outputs=vector,
                create_graph=True,
                only_inputs=True
            )[0]

        return grad_JT_eps

    
#################################################################################################



def giveScore(x, net,sigma,precondition):
        """
        Computes score of E=0.5||x-psi(x,sigma)||^2
        Args:
            
            x: image whose energy needs to be computed
            net: object of the Denoiser class
            vector: x- psi(x)
            precondition: True if one needs to compute preconditioned score
        """
    
        x = x.clone().detach().requires_grad_(True)
        denoised = net(x, sigma)
        eps = x-denoised
        
        #E = 0.5*||x- psi(x)||^2
        E= 0.5*torch.sum((eps).abs()**2,dim =(1,2,3))
        
        JVP = net.vjp(outputs = denoised, inputs = x,  conditioning = sigma, vector = eps,precondition=precondition)
        #computes the gradient of energy
        score = eps-JVP
        
       
        del x
        return E, score

    
#################################################################################################
   
def ALPS(A, net, y, opts, isALPS=True, storeIntermediate=False):
    """
    Annealed Langevin Posterior Sampling
    Implements annealed preconditioned Langevin Dynamics
    Args:
            
            A: object corresponding to the appropriate inverse problem
            net: object of the Denoiser class
            y: measurements
            opts: instance of the Options dataclass, containing configuration.
            isALPS: True for generating samples and False for computing MAP estimate
            storeIntermediate: if True stores intermediate samples 
    """
    device = y.device
    t_steps = giveTsteps(opts.sigma_max,opts.sigma_min,opts.rho,opts.num_steps,device)
    
    Atb = A.adjoint(y).detach()
    
    #initialization
    xtilde = A.PreCondition(Atb/A.eta2, t_steps[0])
    if isALPS:
        n = A.NoiseModulation(torch.randn_like(xtilde),t_steps[0])
        x = xtilde + n.detach() 
    else:
        x = xtilde
    
    

    if storeIntermediate:
        xshape = list(x.shape)
        xshape[0] = len(t_steps+1)
        xsample = torch.zeros(xshape).cpu( )

    for i in range(len(t_steps)):
        for k in range(opts.K):
            t = t_steps[i]
            # denoising
            _, score = giveScore(x, net, t.reshape(-1, 1, 1, 1),precondition=True)
            d = (x-score).detach()
            
            # CG update
            rhs = Atb/A.eta2 + d/t**2
            xtilde = A.PreCondition(rhs, t).detach()
            

            if isALPS and (k<opts.K-1): # not doing it for the last iteration at scale t_i
                # Adding noise to generate posterior samples in p_{t_i}    
                n = A.NoiseModulation(torch.randn_like(x),t)
                x = xtilde + n.detach()
            else:
                x = xtilde
        
        if storeIntermediate:
            xsample[i] = x.detach().cpu()

        # Adding noise to generate posterior samples in p_{t_{i+1}} 
        # No noise was added at the last iteration at scale t_i; now moving to t_{i+1}   
        if(i< len(t_steps)-1):
            n = A.NoiseModulation(torch.randn_like(x),t_steps[i+1])
            x = xtilde + n.detach()

    if storeIntermediate:
        return x, xsample
    else:
        return x
    

def giveTsteps(sigma_max,sigma_min,rho,num_steps,device):
    """
    Annealing schedule
    """
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return t_steps

    
#################################################################################################       