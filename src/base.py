"""Recorded fitting-only preprocessing, replay pools, and base network."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor(a,device,dtype=torch.float32):return torch.as_tensor(a,device=device,dtype=dtype)


def take(ds,ix):
    order=np.argsort(ix);return ds[ix[order]][np.argsort(order)]


def make_pools(x,y,size,seed):
    if size<len(x) or len(x)==0 or y.shape!=(len(x),):raise ValueError('invalid pool inputs')
    rng=torch.Generator(device=x.device).manual_seed(seed+101)
    count=size-len(x);seen=torch.unique(y,sorted=True)
    seq=seen[torch.randperm(len(seen),device=x.device,generator=rng)].repeat((count+len(seen)-1)//len(seen))[:count]
    seq=seq[torch.randperm(count,device=x.device,generator=rng)]
    empirical=torch.randint(len(x),(count,),device=x.device,generator=rng)
    chosen=torch.empty(count,device=x.device,dtype=torch.long)
    for label in seen:
        slots=torch.where(seq==label)[0];rows=torch.where(y==label)[0]
        chosen[slots]=rows[torch.randint(len(rows),(len(slots),),device=x.device,generator=rng)]
    return {'real':(torch.cat([x,x[empirical]]),torch.cat([y,y[empirical]])),
            'balanced':(torch.cat([x,x[chosen]]),torch.cat([y,seq]))},seq


def preprocess(raw,y,device):
    x=tensor(raw,device,torch.float64);labels=tensor(y,device,torch.long)
    hw=((labels[:,None]>>torch.arange(8,device=device))&1).sum(1)
    grand=x.mean(0);between=torch.zeros(x.shape[1],device=device,dtype=torch.float64);within=between.clone()
    for h in range(9):
        rows=x[hw==h]
        if not len(rows):continue
        mean=rows.mean(0);between+=(len(rows)/len(x))*(mean-grand).square();within+=(rows-mean).square().sum(0)/len(x)
    snr=between/within.clamp_min(1e-12);poi=torch.argsort(snr,descending=True,stable=True)[:32].sort().values
    mean=x[:,poi].mean(0);std=x[:,poi].std(0,correction=0).clamp_min(1e-6)
    return ((x[:,poi]-mean)/std).float().cpu().numpy(),{'poi':poi.cpu().numpy(),'mean':mean.cpu().numpy(),'std':std.cpu().numpy()}


def mlp():return nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,64),nn.GELU(),nn.Linear(64,256))

