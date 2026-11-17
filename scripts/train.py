"""Fit a paper method on prepared profiling data, or score a completed model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
METHODS=['Real','RShare-T','Gauss','Bal','Jitter','SMOTE','Mixup-HW','Zaid-Real','Zaid-RShare',
         'FM4-6000','FM4-12000','DDPM16-6000','DDPM16-12000',
         'HW-template-fixed','HW-template-cv','HW-RL','Bit-Gauss','Bitwise-RL']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['fit','evaluate'])
    ap.add_argument('--input',type=Path,required=True,help='Directory produced by prepare.py')
    ap.add_argument('--output',type=Path,required=True,help='Directory for this method run')
    ap.add_argument('--method',choices=METHODS)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--block',choices=['primary','second'],default='primary')
    a=ap.parse_args()
    import numpy as np
    import torch
    from src import base, gaussian, classifier, likelihood, generators, generator_training, aes, ranking, reference_ranks, residual_numpy, hw_selection
    dev=torch.device(a.device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    def sync():
        if dev.type=='cuda':torch.cuda.synchronize(dev)
    def measure(fn):
        sync();t=time.perf_counter();v=fn();sync();return v,time.perf_counter()-t
    def tensor(v,labels=False):return torch.as_tensor(v,device=dev,dtype=torch.long if labels else torch.float32)
    meta=read(a.input/'prepared.json')
    source=a.input/'profiling.npz'
    assert sha(source)==meta['source_sha256']
    q,seed=meta['Q'],meta['seed']
    if a.mode=='fit':
        if not a.method:ap.error('--method is required for fit')
        if a.output.exists():raise FileExistsError('Use a new output directory; completed fits are preserved.')
        available={(r['dataset'],int(r['Q']),r['method']) for r in __import__('csv').DictReader((PACKAGE/'results/configurations.csv').open(encoding='utf-8'))}
        if (meta['dataset'],q,a.method) not in available:ap.error('This method/dataset/budget is not in the paper.')
        with np.load(source) as z:
            arrays={k:z[k] for k in z.files}
        assert len(arrays['x'])==q*4//5 and len(arrays['vx'])==q//5
        assert not np.intersect1d(arrays['train_indices'],arrays['validation_indices']).size
        x,y,vx,vy=[tensor(arrays[k],k in ['y','vy']) for k in ['x','y','vx','vy']]
        a.output.mkdir(parents=True)
        method=a.method
        stages={'preprocess':meta['preprocess_seconds'],'generator_fit':0.,'sampling':0.,'classifier':0.,'selection':0.,'model_save':0.}
        selection={};basis='HW';checkpoint=None
        if method in ['HW-RL','Bit-Gauss','Bitwise-RL']:
            (model,selection),elapsed=measure(lambda:likelihood.select(x,y,method))
            basis=model['basis'];stages['selection']=elapsed
            checkpoint=a.output/'model.npz'
            _,stages['model_save']=measure(lambda:np.savez_compressed(checkpoint,**{k:v.cpu().numpy() if torch.is_tensor(v) else np.array(v) for k,v in model.items() if v is not None}))
        elif method.startswith('HW-template'):
            if method.endswith('cv'):
                (model,selection),elapsed=measure(lambda:hw_selection.tune_cv(hw_selection.common,x,y))
            else:model,elapsed=measure(lambda:gaussian.fit_gaussian(x,y,'HW',.01))
            stages['selection']=elapsed;checkpoint=a.output/'model.npz'
            _,stages['model_save']=measure(lambda:np.savez_compressed(checkpoint,means=model['means'].cpu().numpy(),cov=model['cov'].cpu().numpy()))
        else:
            # Generate the same label sequence and independent random streams as the recorded recipes.
            pool_data,elapsed=measure(lambda:base.make_pools(x,y,20000,q+seed))
            pools,targets=pool_data
            noise=torch.randn((len(targets),32),device=dev,generator=torch.Generator(device=dev).manual_seed(q+seed+8101))
            donor_seed=q+seed+(9101 if meta['dataset']=='AES_GPU' else 48101)
            donors=torch.randint(len(x),(len(targets),),device=dev,generator=torch.Generator(device=dev).manual_seed(donor_seed)).cpu().numpy()
            if method in ['Real','Bal','Zaid-Real']:
                px,py=pools['balanced' if method=='Bal' else 'real'];stages['sampling']=elapsed
            elif method.startswith(('FM4','DDPM16')):
                family,endpoint=method.split('-');endpoint=int(endpoint)
                sync();start=time.perf_counter()
                with torch.random.fork_rng(devices=[dev.index or 0] if dev.type=='cuda' else []):
                    torch.manual_seed(2026111200+seed)
                    model=generators.ConditionalGenerator('fm' if family=='FM4' else 'ddpm',32).to(dev)
                if family=='DDPM16':model.clip_min=x.amin(0).detach();model.clip_max=x.amax(0).detach()
                info=generator_training.train_measured(generators,model,x,y,vx,vy,q+seed+7101,a.output/'generator',
                    steps=endpoint,endpoints=(endpoint,),fit_started=start)
                # Reload the recorded endpoint, not the validation-restored network.
                state=torch.load(a.output/'generator'/f'step{endpoint}.pt',map_location=dev,weights_only=True)
                model.load_state_dict(state);model.eval()
                stages['generator_fit']=info['endpoint_fit_seconds'][str(endpoint)]
                save(a.output/'generator/training.json',info)
                def sample():
                    synthetic=torch.cat([generators.sample(model,model.kind,z,labels,4 if family=='FM4' else 16)
                        for z,labels in zip(noise.split(2048),targets.split(2048))])
                    return torch.cat([x,synthetic]),torch.cat([y,targets])
                (px,py),stages['sampling']=measure(sample)
            elif method=='SMOTE':
                from src.smote import make_smote
                def sample():
                    generated,labels,_=make_smote(x,y,targets,2026113700+seed,k=5)
                    return torch.cat([x,generated]),torch.cat([y,labels])
                (px,py),stages['sampling']=measure(sample)
            elif method=='Mixup-HW':
                def sample():
                    hw=classifier.hw_table(dev);g1=torch.Generator(device=dev).manual_seed(q+seed+9201);g2=torch.Generator(device=dev).manual_seed(q+seed+9202)
                    anchors=torch.empty_like(targets);partners=torch.empty_like(targets)
                    for label in torch.unique(targets,sorted=True):
                        slots=torch.where(targets==label)[0];rows=torch.where(y==label)[0];hrows=torch.where(hw[y]==hw[label])[0]
                        anchors[slots]=rows[torch.randint(len(rows),(len(slots),),device=dev,generator=g1)]
                        partners[slots]=hrows[torch.randint(len(hrows),(len(slots),),device=dev,generator=g2)]
                    lam=tensor(np.random.default_rng(q+seed+9203).beta(.4,.4,len(targets)))[:,None]
                    return torch.cat([x,lam*x[anchors]+(1-lam)*x[partners]]),torch.cat([y,targets])
                (px,py),stages['sampling']=measure(sample)
            else:
                def fit_mean():
                    st=gaussian.fit_gaussian(x,y,'HW',.01)
                    mu=st['means'].cpu().numpy();bank=residual_numpy.center_residuals(arrays['x'],mu[arrays['y']])
                    return st,mu,bank
                (st,mu,bank),stages['selection']=measure(fit_mean)
                def sample():
                    if method in ['RShare-T','Zaid-RShare']:
                        synthetic=tensor(residual_numpy.transplant(mu[targets.cpu().numpy()],bank,donors));labels=targets
                    elif method=='Gauss':synthetic=gaussian.sample_gaussian(st,noise,targets);labels=targets
                    elif method=='Jitter':
                        residual=x.double()-st['means'][y];residual-=residual.mean(0)
                        sd=(.100001*max(float(torch.trace(residual.T@residual/(len(x)-1)))/32,1e-6))**.5
                        bx,by=pools['balanced'];synthetic=(bx[len(x):].double()+sd*noise.double()).float();labels=by[len(x):]
                    else:raise ValueError(method)
                    return torch.cat([x,synthetic]),torch.cat([y,labels])
                (px,py),stages['sampling']=measure(sample)
            assert px.shape==(20000,32) and torch.equal(px[:len(x)],x) and torch.equal(py[:len(y)],y)
            if method.startswith('Zaid'):
                from src.cnn import train_zaid
                _,stages['classifier']=measure(lambda:train_zaid(px,py,2026111300+seed,a.output,method,dev))
                checkpoint=a.output/(method+'.pt')
            else:
                training,stages['classifier']=measure(lambda:classifier.fit(px,py,x,y,vx,vy,'hw9',2026111300+seed,a.output,method,steps=2000))
                save(a.output/'training.json',training);checkpoint=a.output/(method+'_step2000.pt')
        record={'dataset':meta['dataset'],'Q':q,'seed':seed,'method':method,'basis':basis,
            'model_file':checkpoint.name,'model_sha256':sha(checkpoint),'source_sha256':sha(source),
            'selection':selection,'stages':stages,'construction_seconds':sum(stages.values()),
            'attack_rows_used_for_selection':0,'environment':{'python':sys.version.split()[0],'torch':torch.__version__,
                'numpy':np.__version__,'device':torch.cuda.get_device_name(dev) if dev.type=='cuda' else 'CPU','threads':2},
            'timing_scope':'This portable run; not a replacement for the recorded warmed A100 construction measurements.'}
        save(a.output/'fit.json',record)
        print('Fit complete:',method,'construction seconds:',record['construction_seconds'])
    else:
        record=read(a.output/'fit.json');method=record['method'];checkpoint=a.output/record['model_file']
        assert sha(checkpoint)==record['model_sha256'] and sha(source)==record['source_sha256']
        if (a.output/(a.block+'-ranks.npz')).exists():raise FileExistsError('Evaluation already exists')
        with np.load(a.input/(a.block+'.npz')) as z:
            x=tensor(z['x']);candidates=z['candidates'];key=int(z['key']);orders=z['orders']
        if method in ['HW-RL','Bit-Gauss','Bitwise-RL']:
            with np.load(checkpoint) as z:model={k:torch.as_tensor(z[k],device=dev) if z[k].ndim else z[k].item() for k in z.files}
            score=lambda:likelihood.scores_kde(x,model,[model['bandwidth']])[0] if method in ['HW-RL','Bitwise-RL'] else likelihood.scores_gaussian(x,model)
        elif method.startswith('HW-template'):
            with np.load(checkpoint) as z:
                means=torch.as_tensor(z['means'][[(1<<h)-1 for h in range(9)]],device=dev);cov=torch.as_tensor(z['cov'],device=dev)
            score=lambda:aes.gaussian_scores(x.double(),means,cov)
        else:
            cp=torch.load(checkpoint,map_location=dev,weights_only=True)
            if method.startswith('Zaid'):
                from src.cnn import build_zaid
                model=build_zaid(cp['seed'],dev)
            else:model=classifier.make_model('hw9',cp['seed'],dev)
            model.load_state_dict(cp['state']);model.eval()
            score=lambda:classifier.predict(model,x).double().log_softmax(1)-cp['HW_prior'].to(dev).log()
        with torch.no_grad():
            score();durations=[]
            for _ in range(5):sigma,elapsed=measure(score);durations.append(elapsed)
        sigma=sigma.cpu().numpy()
        labels=candidates if record['basis']=='bit' else np.array([i.bit_count() for i in range(256)])[candidates]
        mapped=sigma[np.arange(len(x))[:,None],labels]
        rank2=(2*ranking.rank_curves(mapped,key,orders,dev)).astype(np.int16)
        assert np.array_equal(rank2,reference_ranks.block_rank2(mapped,key,orders,dev))
        np.savez_compressed(a.output/(a.block+'-ranks.npz'),rank2=rank2)
        save(a.output/(a.block+'-evaluation.json'),{'block':a.block,'score_seconds':durations,
            'score_median':float(np.median(durations)),'independent_rank_match':True,
            'endpoints':{str(n):{'SR':float((rank2[:,n-1]==0).mean()),'GE':float(rank2[:,n-1].mean()/2)} for n in [100,1000,10000] if n<=len(x)}})
        print('Evaluation complete:',a.block,method)


if __name__=='__main__':main()
