"""Recorded HW Gaussian regression and Gaussian augmentation."""
import math
import torch

def design(kind,device):
    ids=torch.arange(256,device=device);bits=2*((ids[:,None]>>torch.arange(8,device=device))&1).double()-1
    hw=bits.sum(1,keepdim=True)/math.sqrt(8)
    features=hw if kind=='HW' else torch.cat([bits,hw],1)
    return torch.cat([torch.ones((256,1),device=device,dtype=torch.float64),features],1)


@torch.no_grad()
def fit_gaussian(x,y,kind,ratio):
    phi=design(kind,x.device);f=phi[y];z=x.double();penalty=torch.eye(f.shape[1],device=x.device,dtype=torch.float64)*ratio*len(x);penalty[0,0]=0
    beta=torch.linalg.solve(f.T@f+penalty,f.T@z)
    residual=z-f@beta;residual-=residual.mean(0)
    cov=residual.T@residual/max(1,len(x)-1);tau=cov.trace().div(x.shape[1]).clamp_min(1e-6)
    cov=.9*cov+.1*tau*torch.eye(x.shape[1],device=x.device,dtype=torch.float64)
    cov+=1e-6*tau*torch.eye(x.shape[1],device=x.device,dtype=torch.float64)
    return {'means':phi@beta,'cov':cov,'coefficients':beta,'features':phi,'ratio':ratio,'kind':kind}


@torch.no_grad()
def sample_gaussian(state,noise,y):return (state['means'][y]+noise.double()@torch.linalg.cholesky(state['cov']).T).float()

