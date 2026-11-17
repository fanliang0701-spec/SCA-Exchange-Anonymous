"""Combine frozen and newly evaluated likelihood recipes, with cap/fidelity checks."""
import collections,hashlib,io,json,zipfile
import numpy as np
from reanalyze import DATA,ROOT,DATASETS,read,save,first_at,stable_at,winner,joint_envelope,pareto,wilson

STRICT=1-(1-.95)/16

def main():
    assert all((DATA/n).is_file() for n in ['rank-curves.zip','residual-results.zip','bitwise-rl-results.zip','second-block-results.zip'])
    old=read(ROOT/'analysis.json');groups=old['groups'];curves={};subset_curves={}
    with zipfile.ZipFile(DATA/'rank-curves.zip') as z:
        for g in groups:
            sr=[];ge=[]
            for s in g['seeds']:
                ranks=np.load(io.BytesIO(z.read(f"{g['id']}/s{s}.npy")))
                sr.append((ranks==0).mean(0));ge.append(ranks.mean(0)/2)
            sr=np.array(sr);ge=np.array(ge)
            subset_curves[g['id']]=sr
            curves[g['id']+'/SR']=sr.mean(0);curves[g['id']+'/GE']=ge.mean(0)
    newpath=DATA/'residual-results.zip'
    if newpath.exists():
        with zipfile.ZipFile(newpath) as z:
            rows=collections.defaultdict(list)
            for name in z.namelist():
                if not name.endswith('.json') or name=='manifest.json':continue
                r=json.loads(z.read(name));gid=f"{r['dataset']}/Q{r['Q']}/{r['method']}"
                payload=z.read(name[:-5]+'.npy')
                assert hashlib.sha256(payload).hexdigest()==r['rank_npy_sha256']
                ranks=np.load(io.BytesIO(payload));assert ranks.shape==(20,10000)
                rows[gid].append((r,ranks))
            assert len(rows)==38
            for gid,rs in sorted(rows.items()):
                rs.sort(key=lambda x:x[0]['seed']);assert len(rs)==10
                r=rs[0][0];sr=np.array([(rank==0).mean(0) for _,rank in rs]);ge=np.array([rank.mean(0)/2 for _,rank in rs]);ms=sr.mean(0);mg=ge.mean(0)
                cs=np.array([x['fit']['construction_seconds'] for x,_ in rs]);sc=np.array([x['score_median'] for x,_ in rs])
                g=dict(id=gid,dataset=r['dataset'],Q=r['Q'],method=r['method'],name=r['method'],cohort='residual-likelihood',learner='likelihood',
                       n_subsets=10,seeds=[x['seed'] for x,_ in rs],C=float(cs.mean()),C_median=float(np.median(cs)),C_subsets=cs.tolist(),
                       A95=first_at(ms),A95_sustained=stable_at(ms),GE_lt_1=first_at((mg<1).astype(float),1),quality={},
                       score_seconds_10000=float(sc.mean()),score_times_subsets=sc.tolist(),
                       selection=[x['fit']['choice'] for x,_ in rs])
                for a in [100,1000,10000]:
                    k=int(np.count_nonzero(sr[:,a-1]>=.95-1e-12))
                    g['quality'][str(a)]={'SR':float(ms[a-1]),'GE':float(mg[a-1]),'subset_passes':k,'subset_pass_wilson':wilson(k,10)}
                groups.append(g);subset_curves[gid]=sr;curves[gid+'/SR']=ms;curves[gid+'/GE']=mg
    bitpath=DATA/'bitwise-rl-results.zip'
    if bitpath.exists():
        with zipfile.ZipFile(bitpath) as z:
            rows=collections.defaultdict(list)
            for name in z.namelist():
                if not name.endswith('.json') or name=='manifest.json':continue
                r=json.loads(z.read(name));gid=f"{r['dataset']}/Q{r['Q']}/{r['method']}"
                payload=z.read(name[:-5]+'.npy');assert hashlib.sha256(payload).hexdigest()==r['rank_npy_sha256']
                ranks=np.load(io.BytesIO(payload));assert ranks.shape==(20,10000);rows[gid].append((r,ranks))
            assert len(rows)==3
            for gid,rs in sorted(rows.items()):
                rs.sort(key=lambda x:x[0]['seed']);assert len(rs)==10
                r=rs[0][0];sr=np.array([(rank==0).mean(0) for _,rank in rs]);ge=np.array([rank.mean(0)/2 for _,rank in rs]);ms=sr.mean(0);mg=ge.mean(0)
                cs=np.array([x['fit']['construction_seconds'] for x,_ in rs]);sc=np.array([x['score_median'] for x,_ in rs])
                g=dict(id=gid,dataset=r['dataset'],Q=r['Q'],method=r['method'],name=r['method'],cohort='residual-likelihood',learner='likelihood',
                       n_subsets=10,seeds=[x['seed'] for x,_ in rs],C=float(cs.mean()),C_median=float(np.median(cs)),C_subsets=cs.tolist(),
                       A95=first_at(ms),A95_sustained=stable_at(ms),GE_lt_1=first_at((mg<1).astype(float),1),quality={},
                       score_seconds_10000=float(sc.mean()),score_times_subsets=sc.tolist(),selection=[x['fit']['choice'] for x,_ in rs])
                for a in [100,1000,10000]:
                    k=int(np.count_nonzero(sr[:,a-1]>=.95-1e-12));g['quality'][str(a)]={'SR':float(ms[a-1]),'GE':float(mg[a-1]),'subset_passes':k,'subset_pass_wilson':wilson(k,10)}
                groups.append(g);subset_curves[gid]=sr;curves[gid+'/SR']=ms;curves[gid+'/GE']=mg
    for g in groups:
        sr=subset_curves[g['id']];g['Astrict']=first_at(sr.mean(0),STRICT)
        for name,threshold in [('95',.95),('strict',STRICT)]:
            aa=[first_at(row,threshold) for row in sr];values=[a for a in aa if a is not None]
            g['subset_A'+name]=aa
            g['subset_A'+name+'_summary']={'attained':len(values),'censored':10-len(values),'conditional_min_median_max':[min(values),float(np.median(values)),max(values)] if values else None}
    scenarios=[];cap=[];fidelity=[]
    for b in DATASETS:
        bg=[g for g in groups if g['dataset']==b]
        for tp,tt in [(.001,.001),(.02,.001),(.001,.02),(.02,.02),(1/10.87,1/10.87)]:
            scenarios.append(dict(dataset=b,tau_p=tp,tau_t=tt,winner=winner(bg,tp,tt)))
        for s,threshold in [('95',.95),('strict',STRICT)]:
            for limit in [5000,8000,10000]:
                gg=[{**g,'A95':first_at(curves[g['id']+'/SR'][:limit],threshold)} for g in bg]
                cap.append(dict(dataset=b,criterion=s,threshold=threshold,cap=limit,
                    eligible=sum(g['A95'] is not None for g in gg),
                    equal_1ms=winner(gg,.001,.001),profile20_attack1=winner(gg,.02,.001),
                    capture_reported=winner(gg,1/10.87,1/10.87),
                    compute_copies16=winner(gg,1/10.87,1/10.87,B=16),
                    focal=[dict(id=g['id'],A=g['A95']) for g in gg if g['Q']==2000 and g['method'] in ['HW-template-fixed','HW-RL','Bit-Gauss','FM4-6000','DDPM16-6000']]))
        ref=min(g['A95'] for g in bg if g['A95'] is not None)
        allowed=[g for g in bg if g['A95'] is not None and g['A95']<=2*ref]
        fidelity.append(dict(dataset=b,kappa=2,Qmax=20000,Aref=ref,allowed=[g['id'] for g in allowed],
                    equal_price_envelope=joint_envelope(allowed),attack1ms_envelope=joint_envelope(allowed,tau_t=.001),
                    capture_reported=winner(allowed,1/10.87,1/10.87),
                    neural_minimum_kappa={g['id']:g['A95']/ref for g in bg if g['Q']==2000 and g['method'].startswith(('FM4','DDPM16')) and g['A95'] is not None}))
    bootstrap={}
    rng=np.random.default_rng(2026111623)
    weights=rng.multinomial(10,[.1]*10,size=2000)/10
    byid={g['id']:g for g in groups}
    for b in ['D1','AES_GPU']:
        env=joint_envelope([g for g in groups if g['dataset']==b],tau_t=.001)
        a,g=byid[env[-2]['id']],byid[env[-1]['id']]
        attacks=[]
        for candidate in [a,g]:
            vals=subset_curves[candidate['id']];out=[]
            for start in range(0,len(weights),50):
                hit=(weights[start:start+50]@vals)>=.95-1e-12
                out.extend(np.where(hit.any(1),hit.argmax(1)+1,np.nan))
            attacks.append(np.array(out))
        cs=weights@(np.array(g['C_subsets'])-np.array(a['C_subsets']))
        ok=np.isfinite(attacks[0])&np.isfinite(attacks[1])
        draws=(cs+(attacks[1]-attacks[0])*.001)/(a['Q']-g['Q'])
        bootstrap[b]={'from':a['id'],'to':g['id'],'resamples':len(weights),'eligible_resamples':int(ok.sum()),
                      'point':env[-1]['lo'],'ci':np.quantile(draws[ok],[.025,.975]).tolist()}
    scoring=[]
    for b in DATASETS:
        bg=[g for g in groups if g['dataset']==b];chosen=winner(bg,.001,.001);g=byid[chosen['id']]
        scored=chosen['total']+g.get('score_seconds_10000',0)
        alternatives=[x for x in bg if x['id']!=g['id']]
        alt=winner(alternatives,.001,.001)
        scoring.append({'dataset':b,'chosen':g['id'],'charge_full_10000_scoring':g.get('score_seconds_10000'),
                        'chosen_total_plus_full_score':scored,'best_other_with_zero_scoring':alt,
                        'margin_after_full_score':alt['total']-scored})
    cw_rates={'F0':1/9.21,'F1':1/11.23,'F2':1/10.87,'F3':1/12.43,'XMEGA':1/5.21,'median_reference':1/10.87}
    result={'scope':{'groups':len(groups),'subsets_per_group':10,'frozen_groups':149,'new_groups':len(groups)-149,'cap':10000,'requirement':.95,'strict_requirement':STRICT,
                    'capture_rate':{'traces_per_second':10.87,'seconds_per_trace':1/10.87,'source':'median of the five recorded workflow rates','assumption':'same profiling and target rate','target_specific_rates':cw_rates},
                    'attack_uncertainty':'Repeated orders reuse the same block; subset bootstrap does not estimate between-block or between-key variation.'},
            'groups':groups,'scenarios':scenarios,'capture_rate_scenarios':{name:{b:winner([g for g in groups if g['dataset']==b],sec,sec) for b in DATASETS} for name,sec in cw_rates.items()},'cap_sensitivity':cap,'fidelity':fidelity,
            'bootstrap_crossings':bootstrap,'scoring_sensitivity':scoring,
            'equal_price_envelopes':{b:joint_envelope([g for g in groups if g['dataset']==b]) for b in DATASETS},
            'attack_1ms_envelopes':{b:joint_envelope([g for g in groups if g['dataset']==b],tau_t=.001) for b in DATASETS}}
    secondpath=DATA/'second-block-results.zip'
    if secondpath.exists():
        replay=collections.defaultdict(list)
        with zipfile.ZipFile(secondpath) as z:
            for name in z.namelist():
                if not name.endswith('.json') or name=='manifest.json':continue
                r=json.loads(z.read(name));gid=f"{r['dataset']}/Q{r['Q']}/{r['method']}"
                payload=z.read(name[:-5]+'.npy')
                assert hashlib.sha256(payload).hexdigest()==r['rank_npy_sha256']
                ranks=np.load(io.BytesIO(payload));replay[gid].append((r['seed'],(ranks==0).mean(0),ranks.mean(0)/2))
        second=[]
        for gid,rs in sorted(replay.items()):
            rs.sort(key=lambda x:x[0]);assert len(rs)==10
            sr=np.array([r[1] for r in rs]);ge=np.array([r[2] for r in rs]);ms=sr.mean(0)
            second.append({**byid[gid],'A95':first_at(ms),'A95_sustained':stable_at(ms),
                           'Astrict':first_at(ms,STRICT),'subset_A95':[first_at(v) for v in sr]})
            curves[gid+'/second/SR']=ms;curves[gid+'/second/GE']=ge.mean(0)
        assert len(second)==164
        result['second_block']={'groups':second,'scenarios':[],
                  'equal_price_envelopes':{b:joint_envelope([g for g in second if g['dataset']==b]) for b in DATASETS[:4]},
                  'attack_1ms_envelopes':{b:joint_envelope([g for g in second if g['dataset']==b],tau_t=.001) for b in DATASETS[:4]}}
        for b in DATASETS[:4]:
            bg=[g for g in second if g['dataset']==b]
            for tp,tt in [(.001,.001),(.02,.001),(1/10.87,1/10.87)]:
                result['second_block']['scenarios'].append({'dataset':b,'tau_p':tp,'tau_t':tt,'winner':winner(bg,tp,tt)})
    save(ROOT/'extended-analysis.json',result);np.savez_compressed(ROOT/'extended-curves.npz',**curves)
    for b in DATASETS:
        print('\n',b)
        for g in groups:
            if g['dataset']==b and g['method'] in ['HW-RL','Bit-Gauss']:
                print(g['method'],g['Q'],'A95',g['A95'],'strict',g['Astrict'],'C',round(g['C'],4),'score10k',round(g['score_seconds_10000'],4))
        print('equal-price',[(x['method'],x['Q'],round(x['lo']*1000,3)) for x in result['equal_price_envelopes'][b]])
        print('capture',next(s['winner'] for s in scenarios if s['dataset']==b and s['tau_p']==1/10.87))
    print('D1 cap',[(c['criterion'],c['cap'],c['profile20_attack1']) for c in cap if c['dataset']=='D1'])

if __name__=='__main__':main()
