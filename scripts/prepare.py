"""Prepare one recorded fitting subset from a separately obtained public dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',required=True,choices=['D1','D2','D3','D4','AES_GPU'])
    ap.add_argument('--data',required=True,type=Path,help='AES_PTv2_D*.h5 or AES_GPU.mat')
    ap.add_argument('--budget',type=int,required=True,choices=[2000,5000,10000,20000])
    ap.add_argument('--seed',type=int,default=13,choices=[13,37,73,101,137,211,251,307,401,503])
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',default='cpu',help='cpu or cuda:0; original preparation used CUDA')
    ap.add_argument('--with-attack',action='store_true',help='Also write primary and, when available, second-block evaluation inputs')
    a=ap.parse_args()
    import torch
    from src import base, aes
    torch.set_num_threads(2)
    dev=torch.device(a.device)
    prefix=f'{a.dataset}/Q{a.budget}/s{a.seed}/'
    with np.load(PACKAGE/'data/frozen-splits-and-preprocessing.npz') as z:
        if prefix+'poi' not in z: ap.error('This dataset/budget was not fitted in the paper.')
        ti,vi=[z[prefix+k] for k in ['train_indices','validation_indices']]
        params={k:z[prefix+k] for k in ['poi','mean','std']}
        if a.dataset!='AES_GPU':
            primary=z[a.dataset+'/attack/reserved_replay']
            remaining=np.setdiff1d(z[a.dataset+'/attack/candidates'],primary)
            second=np.random.default_rng(2026111619).permutation(remaining)[:10000]
            assert not np.intersect1d(second,z[a.dataset+'/attack/primary']).size
            assert not np.intersect1d(second,z[a.dataset+'/attack/excluded']).size
    box=aes.sbox(torch.device('cpu')).numpy()
    if a.dataset=='AES_GPU':
        from scipy.io import loadmat
        digest=hashlib.sha256(a.data.read_bytes()).hexdigest()
        if digest!='723fa4b2bc0b65631560b4f121293141e4643bdf47f5d7e3fb4361ef334cfa77':
            raise ValueError('AES_GPU.mat does not match the recorded dataset SHA-256')
        raw=loadmat(a.data);trace=raw['trace'];cipher=raw['cipher'].ravel().astype(int)
        key=int(raw['key'].item());labels=np.argsort(box)[cipher^key]
        xraw,y=trace[ti],labels[ti];vraw,vy=trace[vi],labels[vi]
        with np.load(PACKAGE/'data/gpu-split.npz') as z:primary=z['attack_indices']
        assert not np.intersect1d(primary,np.r_[ti,vi]).size
        def attack(ix):
            return trace[ix],np.argsort(box)[cipher[ix,None]^np.arange(256)],key
    else:
        import h5py
        with h5py.File(a.data,'r') as h:
            g=h[a.dataset+'/Unprotected/Profiling']
            xraw,y=base.take(g['Traces'],ti),base.take(g['Labels'],ti)
            vraw,vy=base.take(g['Traces'],vi),base.take(g['Labels'],vi)
            for ix,lab in [(ti,y),(vi,vy)]:
                meta=base.take(g['MetaData'],ix)
                assert np.array_equal(lab,box[meta['plaintext'][:,0]^meta['key'][:,0]])
        def attack(ix):
            with h5py.File(a.data,'r') as h:
                g=h[a.dataset+'/Unprotected/Attack']
                trace=base.take(g['Traces'],ix);meta=base.take(g['MetaData'],ix)
            assert len(np.unique(meta['key'],axis=0))==1
            return trace,box[meta['plaintext'][:,0,None]^np.arange(256)],int(meta['key'][0,0])
    # Independently recompute the fitting-only transform; apply the recorded transform.
    started=time.perf_counter()
    _,recomputed=base.preprocess(xraw,y,dev)
    assert np.array_equal(recomputed['poi'],params['poi']), 'HW-SNR POIs differ from frozen input'
    for k in ['mean','std']:assert np.allclose(recomputed[k],params[k],atol=1e-9,rtol=1e-9),k
    def transform(raw):return ((raw[:,params['poi']].astype(float)-params['mean'])/params['std']).astype(np.float32)
    x,vx=transform(xraw),transform(vraw)
    elapsed=time.perf_counter()-started
    a.output.mkdir(parents=True,exist_ok=True)
    target=a.output/'profiling.npz'
    if target.exists():raise FileExistsError(target)
    np.savez_compressed(target,x=x,y=y,vx=vx,vy=vy,train_indices=ti,validation_indices=vi,**params)
    metadata={'dataset':a.dataset,'Q':a.budget,'seed':a.seed,'fitting_rows':len(ti),'validation_rows':len(vi),
              'preprocess_seconds':elapsed,'transform_matches_frozen':True,'source_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    (a.output/'prepared.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    if a.with_attack:
        for block,ix in [('primary',primary)]+([] if a.dataset=='AES_GPU' else [('second',second)]):
            raw,candidates,key=attack(ix)
            np.savez_compressed(a.output/(block+'.npz'),x=transform(raw),candidates=candidates,key=np.array(key),indices=ix,
                orders=np.stack([np.random.default_rng(2026116000+r).permutation(len(ix)) for r in range(20)]))
    print('Prepared',a.dataset,a.budget,a.seed,'in',a.output)


if __name__=='__main__':main()
