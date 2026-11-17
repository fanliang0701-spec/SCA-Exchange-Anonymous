"""Independent prefix reconstruction using block triangular matrix products."""
import numpy as np
import torch

@torch.no_grad()
def block_rank2(mapped,key,orders,device,block_size=64):
 a=torch.as_tensor(mapped,device=device,dtype=torch.float64);a=a-a[:,key,None];n,keys=a.shape;blocks=(n+block_size-1)//block_size;triangle=torch.tril(torch.ones((block_size,block_size),device=device,dtype=torch.float64));answer=[]
 for order in orders:
  padded=torch.zeros((blocks*block_size,keys),device=device,dtype=torch.float64);padded[:n]=a[torch.as_tensor(order,device=device,dtype=torch.long)];local=triangle@padded.reshape(blocks,block_size,keys);totals=local[:,-1,:];offset=torch.cat([torch.zeros((1,keys),device=device,dtype=torch.float64),totals[:-1].cumsum(0)],0);prefix=(local+offset[:,None,:]).reshape(-1,keys)[:n];rank2=2*(prefix>1e-9).sum(1)+(prefix.abs()<=1e-9).sum(1)-1;answer.append(rank2.to(torch.int16).cpu().numpy())
 return np.stack(answer)

def cpu_anchor_rank2(mapped,key,orders,rank2):
 relative=np.asarray(mapped,dtype=float)-np.asarray(mapped,dtype=float)[:,key,None];n=len(relative);fixed=[1,2,10,50,100,200,500,1000,2000,5000,10000,20000];checks=0
 for repeat,order in enumerate(orders[:1]):
  zero=np.flatnonzero(rank2[repeat]==0);points=set(k for k in fixed if k<=n)
  if len(zero):points.update(k for k in [int(zero[0]),int(zero[0]+1),int(zero[0]+2)] if 1<=k<=n)
  total=np.zeros(relative.shape[1],dtype=float);previous=0
  for count in sorted(points):
   total+=relative[order[previous:count]].sum(0);previous=count;want=int(2*np.sum(total>1e-9)+np.sum(abs(total)<=1e-9)-1);assert want==int(rank2[repeat,count-1]),(repeat,count,want,rank2[repeat,count-1]);checks+=1
 return checks
