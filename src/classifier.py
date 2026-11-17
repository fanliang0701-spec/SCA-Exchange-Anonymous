"""Recorded classifier fitting and prior correction."""
import time
import torch
from torch import nn
from torch.nn import functional as F
from . import base as b
CHECKPOINTS=[100,250,500,1000,2000]
_GROUP_INDICES={}

def hw_table(device):return torch.tensor([i.bit_count() for i in range(256)],device=device,dtype=torch.long)


def group_logits(z,groups=None):
    if groups is not None:
        return torch.stack([torch.logsumexp(z[:,groups==i],1) for i in range(int(groups.max())+1)],1)
    key=str(z.device)
    if key not in _GROUP_INDICES:
        _GROUP_INDICES[key]=[torch.tensor([j for j in range(256) if j.bit_count()==i],device=z.device) for i in range(9)]
    return torch.stack([torch.logsumexp(z.index_select(1,ix),1) for ix in _GROUP_INDICES[key]],1)


def priors(y):
    p=(torch.bincount(y,minlength=256).double()+.5)/(len(y)+128)
    h=hw_table(y.device);q=torch.stack([p[h==i].sum() for i in range(9)])
    return p,q


def make_model(mode,seed,device):
    if mode not in ['hw9','id','group256']:raise ValueError('unknown training label mode')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed);model=b.mlp()
    if mode=='hw9':
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed+9001);model[4]=nn.Linear(64,9)
    return model.to(device)


def hw_logits(logits,mode):return logits if mode=='hw9' else group_logits(logits)


def metrics(logits,labels,hw_prior,mode):
    h=hw_table(logits.device);target=h[labels];z=hw_logits(logits.double(),mode);lp=z.log_softmax(1)
    ll=lp-hw_prior.log();expanded=ll[:,h];true=expanded.gather(1,labels[:,None])
    ranks=(expanded>true+1e-9).sum(1).double()+((expanded-true).abs().le(1e-9).sum(1).double()-1)/2
    nll=float(F.nll_loss(lp,target));prior_nll=float(-hw_prior[target].log().mean())
    return {'HW_NLL':nll,'HW_accuracy':float((z.argmax(1)==target).double().mean()),'prior_NLL':prior_nll,
      'NLL_gain_over_prior':prior_nll-nll,'mean_log2_ID_rank':float(torch.log2(1+ranks).mean()),
      'HW_macro_accuracy':float(torch.stack([(z.argmax(1)[target==i]==i).double().mean() for i in target.unique()]).mean()),
      'rows':len(labels),'ID_NLL':float(F.cross_entropy(logits.double(),labels)) if mode=='id' else None,'ID_accuracy':float((logits.argmax(1)==labels).double().mean()) if mode=='id' else None}


@torch.no_grad()
def predict(model,x):return torch.cat([model(part) for part in x.split(2048)])


def fit(px,py,x,y,vx,vy,mode,seed,folder,name,steps=2000):
    start=time.monotonic();p,q=priors(py);model=make_model(mode,seed,px.device);h=hw_table(px.device)
    rng=torch.Generator(device=px.device).manual_seed(seed+3101);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    checkpoints=[s for s in CHECKPOINTS+[5000,10000] if s<=steps];history=[];files={};loss_sum=0.
    for step in range(1,steps+1):
        idx=torch.randint(len(px),(256,),device=px.device,generator=rng);logits=model(px[idx])
        loss=F.cross_entropy(logits,py[idx]) if mode=='id' else F.cross_entropy(hw_logits(logits,mode),h[py[idx]])
        opt.zero_grad(set_to_none=True);loss.backward();opt.step();loss_sum+=float(loss.detach())
        if step in checkpoints:
            train=metrics(predict(model,x),y,q,mode);val=metrics(predict(model,vx),vy,q,mode)
            info={'step':step,'optimization_loss_mean':loss_sum/step,'train':train,'validation':val};history.append(info)
            path=folder/f'{name}_step{step}.pt'
            torch.save({'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'ID_prior':p.cpu(),'HW_prior':q.cpu(),'step':step,'seed':seed,'mode':mode},path)
            files[str(step)]={'file':path.name,'sha256':b.sha(path)}
    choices={'final':steps,'HW_rank':min(history,key=lambda r:(r['validation']['mean_log2_ID_rank'],r['validation']['HW_NLL'],r['step']))['step'],
      'HW_nll':min(history,key=lambda r:(r['validation']['HW_NLL'],r['step']))['step']}
    if 2000 in checkpoints:choices['at2000']=2000
    return {'name':name,'mode':mode,'seed':seed,'choices':choices,'history':history,'checkpoints':files,
      'parameters':sum(v.numel() for v in model.parameters()),'seconds':time.monotonic()-start,'pool_counts':torch.bincount(py,minlength=256).cpu().tolist()}

