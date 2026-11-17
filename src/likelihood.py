"""Residual KDE and bitwise Gaussian likelihoods on frozen fitting features.

All selection uses fitting-only folds. Class-independent density constants
cancel in posterior NLL and in key ranking. No attack data enter fitting.
"""
import math
import numpy as np
import torch

RIDGES=(.001,.01,.1)
SHRINKAGES=(.01,.1,.5,.9,1.)
BANDWIDTHS=(.05,.1,.2,.4,.8,1.6,3.2)


def design(basis,device):
    y=torch.arange(256,device=device)
    bits=((y[:,None]>>torch.arange(8,device=device))&1).double()
    if basis=='HW':return torch.stack((torch.ones(256,device=device),(bits.sum(1)-4)/math.sqrt(2)),1)
    assert basis=='bit'
    return torch.cat((torch.ones((256,1),device=device),bits),1)


def fit_mean(x,y,basis='HW',ridge=.01):
    phi=design(basis,x.device);f=phi[y];x=x.double()
    penalty=torch.eye(f.shape[1],device=x.device,dtype=x.dtype)*len(x)*ridge
    penalty[0,0]=0
    beta=torch.linalg.solve(f.T@f+penalty,f.T@x)
    means=phi@beta; residual=x-means[y]; residual=residual-residual.mean(0)
    cov=residual.T@residual/(len(x)-1)
    return dict(means=means,residual=residual,cov=cov,beta=beta,basis=basis,ridge=ridge)


def whiten(state,shrink):
    cov=state['cov'];d=len(cov);v=cov.trace().div(d).clamp_min(1e-6)
    S=(1-shrink)*cov+(shrink+1e-6)*v*torch.eye(d,device=cov.device,dtype=cov.dtype)
    L=torch.linalg.cholesky(S)
    reps=(torch.tensor([(1<<h)-1 for h in range(9)],device=cov.device)
          if state['basis']=='HW' else torch.arange(256,device=cov.device))
    mean=torch.linalg.solve_triangular(L,state['means'][reps].T,upper=False).T
    bank=torch.linalg.solve_triangular(L,state['residual'].T,upper=False).T
    return dict(L=L,means=mean,bank=bank,shrink=shrink,basis=state['basis'])


def scores_gaussian(x,model):
    z=torch.linalg.solve_triangular(model['L'],x.double().T,upper=False).T
    return -.5*((z[:,None,:]-model['means'][None,:,:])**2).sum(-1)


def scores_kde(x,model,bandwidths,chunk=128):
    """Exact full-bank KDE, bounded memory; reuse distances across h values."""
    z=torch.linalg.solve_triangular(model['L'],x.double().T,upper=False).T
    e=model['bank'];enorm=(e*e).sum(1)[None,:]
    outputs=[[] for _ in bandwidths]
    for block in z.split(chunk):
        q=(block[:,None,:]-model['means'][None,:,:]).reshape(-1,z.shape[1])
        d2=((q*q).sum(1)[:,None]+enorm-2*(q@e.T)).clamp_min_(0)
        for parts,h in zip(outputs,bandwidths):
            ll=torch.logsumexp(d2*(-.5/(h*h)),1)-math.log(len(e))
            parts.append(ll.reshape(len(block),model['means'].shape[0]))
    return [torch.cat(parts) for parts in outputs]


def prior_target(y,basis):
    p=(torch.bincount(y,minlength=256).double()+.5)/(len(y)+128)
    if basis=='bit':return p,y
    hw=((torch.arange(256,device=y.device)[:,None]>>torch.arange(8,device=y.device))&1).sum(1)
    return torch.stack([p[hw==h].sum() for h in range(9)]),hw[y]


@torch.no_grad()
def select(x,y,kind):
    assert kind in ['HW-RL','Bit-Gauss','Bitwise-RL']
    basis='HW' if kind=='HW-RL' else 'bit'
    residual_likelihood=kind in ['HW-RL','Bitwise-RL']
    ridges=(.01,) if residual_likelihood else RIDGES
    hs=BANDWIDTHS if residual_likelihood else (None,)
    total=torch.zeros((len(ridges),len(SHRINKAGES),len(hs)),device=x.device,dtype=torch.float64)
    fold=torch.arange(len(x),device=x.device)%5
    for k in range(5):
        tr,va=fold!=k,fold==k
        prior,_=prior_target(y[tr],basis);_,target=prior_target(y[va],basis)
        for ir,ridge in enumerate(ridges):
            mean=fit_mean(x[tr],y[tr],basis,ridge)
            for il,lam in enumerate(SHRINKAGES):
                model=whiten(mean,lam)
                ll=scores_kde(x[va],model,hs) if residual_likelihood else [scores_gaussian(x[va],model)]
                for ih,sigma in enumerate(ll):
                    logpost=(sigma+prior.log()).log_softmax(1)
                    total[ir,il,ih]-=logpost.gather(1,target[:,None]).sum()
    nll=(total/len(x)).cpu().numpy()
    ir,il,ih=np.unravel_index(nll.argmin(),nll.shape)
    state=fit_mean(x,y,basis,ridges[ir]);model=whiten(state,SHRINKAGES[il])
    model['bandwidth']=hs[ih]
    choice=dict(kind=kind,ridge=ridges[ir],shrinkage=SHRINKAGES[il],bandwidth=hs[ih],
                cv_nll=float(nll[ir,il,ih]),grid_nll=nll.tolist(),folds=5,
                fold_rule='arange(n)%5 within frozen fitting feature rows',
                feature_selection='frozen full-fitting POIs and normalization',
                residual_self_match=False,validation_rows_used=0,attack_rows_used=0)
    return model,choice


@torch.no_grad()
def check_numerics(device):
    torch.manual_seed(20261116)
    x=torch.randn(41,5,device=device,dtype=torch.float64)
    y=torch.arange(41,device=device)%256
    state=fit_mean(x,y);m=whiten(state,.1)
    query=torch.randn(7,5,device=device,dtype=torch.float64)
    z=torch.linalg.solve_triangular(m['L'],query.T,upper=False).T
    d2=((z[:,None,None,:]-m['means'][None,:,None,:]-m['bank'][None,None,:,:])**2).sum(-1)
    for h,actual in zip(BANDWIDTHS,scores_kde(query,m,BANDWIDTHS,chunk=3)):
        expected=torch.logsumexp(-d2/(2*h*h),-1)-math.log(len(x))
        assert torch.allclose(actual,expected,atol=1e-8,rtol=1e-10)
    assert torch.allclose(state['residual'].mean(0),torch.zeros(5,device=device,dtype=torch.float64),atol=1e-12)
    phi=design('bit',device)
    assert phi.shape==(256,9) and torch.equal(phi[:,1:].sum(1),(design('HW',device)[:,1]*math.sqrt(2)+4).round())
    return dict(kde_matches_direct_broadcast=True,chunked_exact_bank=True,bit_basis_nine_dimensions=True)
