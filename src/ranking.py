"""Zero-based midrank curves, tie tolerance 1e-9."""
import numpy as np
import torch

@torch.no_grad()
def rank_curves(scores,key,orders,device):
 scores=torch.as_tensor(scores,device=device,dtype=torch.float64);relative=scores-scores[:,key,None];out=[]
 for permutation in orders:
  cumulative=relative[torch.as_tensor(permutation,device=device,dtype=torch.long)].cumsum(0)
  rank2=2*(cumulative>1e-9).sum(1)+(cumulative.abs()<=1e-9).sum(1)-1
  out.append(rank2.to(torch.int16).cpu().numpy())
 return np.stack(out).astype(float)/2

