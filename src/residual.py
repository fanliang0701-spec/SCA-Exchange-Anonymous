"""Whole-vector residual transplant, Equation (9)."""
import torch

def transplant(mu_table,x,y,labels,rng):
    residual=x.double()-mu_table[y];residual=residual-residual.mean(0)
    j=torch.randint(len(x),(len(labels),),device=x.device,generator=rng)
    return (mu_table[labels]+residual[j]).float(),j,residual

