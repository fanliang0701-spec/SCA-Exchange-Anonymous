"""Reanalyze frozen single-byte predictions; no fitting or capture is performed.

The numerical objective is recorded model-construction time plus parameterized
profiling and target acquisition. It does not impute unrecorded scoring times.
All paths in generated scientific records are relative to this package.
"""
from __future__ import annotations

import argparse
import os
import collections
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
DATA = PACKAGE / "data"
ROOT = Path(os.environ.get("SCA_OUTPUT", str(PACKAGE / "results"))).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
DATASETS = ["D1", "D2", "D3", "D4", "AES_GPU"]
REQUIREMENT = 0.95
Q0 = 2000
NAMES = {
    "Real": "Real MLP", "RShare-T": "RShare", "Gauss": "Gaussian augmentation",
    "FM4-6000": "FM4 (6k)", "FM4-12000": "FM4 (12k)",
    "DDPM16-6000": "DDPM16 (6k)", "DDPM16-12000": "DDPM16 (12k)",
    "HW-template-fixed": "HW stochastic", "HW-template-cv": "HW stochastic CV",
    "Zaid-Real": "CNN real", "Zaid-RShare": "CNN RShare",
    "Bal": "Balanced replay", "Jitter": "Jitter", "SMOTE": "SMOTE",
    "Mixup-HW": "HW Mixup",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def member_name(row):
    return f"{row['dataset']}/Q{row['Q']}/{row['method']}/s{row['seed']}"


def first_at(values, threshold=REQUIREMENT):
    indices = np.flatnonzero(np.asarray(values) >= threshold - 1e-12)
    return int(indices[0] + 1) if len(indices) else None


def stable_at(values, threshold=REQUIREMENT):
    return first_at(np.minimum.accumulate(np.asarray(values)[::-1])[::-1], threshold)


def wilson(k, n):
    """95% Wilson interval for the explicitly defined subset-pass event."""
    z = 1.959963984540054
    p = k / n
    den = 1 + z*z/n
    mid = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [0. if k == 0 else max(0., mid-half),
            1. if k == n else min(1., mid+half)]


def cost(g, tau_p, tau_t, A=None, B=1, setup=0.):
    a = g["A95"] if A is None else A
    if a is None:
        return float("inf")
    return (g["Q"]-Q0)*tau_p + a*tau_t + B*g["C"] + setup


def winner(groups, tau_p, tau_t, B=1, fixed_A=None):
    eligible = [g for g in groups if
                (g["A95"] is not None if fixed_A is None
                 else g["quality"][str(fixed_A)]["SR"] >= REQUIREMENT-1e-12)]
    if not eligible:
        return None
    g = min(eligible, key=lambda x: cost(x, tau_p, tau_t, fixed_A, B))
    a = g["A95"] if fixed_A is None else fixed_A
    return {"id": g["id"], "method": g["method"], "Q": g["Q"], "A": a,
            "C": g["C"], "total": cost(g, tau_p, tau_t, a, B)}


def pareto(groups):
    eligible = [g for g in groups if g["A95"] is not None]
    result = []
    for g in eligible:
        v = np.array([g["Q"], g["A95"], g["C"]])
        if not any(np.all((w := np.array([h["Q"], h["A95"], h["C"]])) <= v)
                   and np.any(w < v) for h in eligible if h is not g):
            result.append(g["id"])
    return result


def fixed_envelope(groups, A):
    eligible = [g for g in groups if g["quality"][str(A)]["SR"] >= REQUIREMENT-1e-12]
    knots = {0.}
    for g in eligible:
        for h in eligible:
            if g["Q"] == h["Q"]:
                continue
            x = (h["C"]-g["C"])/(g["Q"]-h["Q"])
            if x > 0:
                knots.add(float(x))
    knots = sorted(knots)
    out = []
    for i, lo in enumerate(knots):
        hi = knots[i+1] if i+1 < len(knots) else None
        probe = (lo+hi)/2 if hi is not None else max(1., lo*2)
        pick = winner(eligible, probe, 0., fixed_A=A)
        if pick is None:
            continue
        if out and out[-1]["id"] == pick["id"]:
            out[-1]["hi"] = hi
        else:
            out.append({"lo": lo, "hi": hi, **{k:v for k,v in pick.items() if k!="total"}})
    return out


def joint_envelope(groups, tau_t=None, B=1, sustained=False):
    """Exact envelope versus profiling price; tau_t=None ties both prices."""
    field="A95_sustained" if sustained else "A95"
    lines=[]
    for g in groups:
        a=g[field]
        if a is None:continue
        slope=g["Q"]-Q0+(a if tau_t is None else 0)
        intercept=B*g["C"]+(0. if tau_t is None else a*tau_t)
        lines.append((g,a,slope,intercept))
    if not lines:return []
    knots={0.}
    for _,_,s,b in lines:
        for _,_,t,c in lines:
            if s!=t and (x:=(c-b)/(s-t))>0:knots.add(float(x))
    knots=sorted(knots);out=[]
    for i,lo in enumerate(knots):
        hi=knots[i+1] if i+1<len(knots) else None
        p=(lo+hi)/2 if hi is not None else max(1.,lo*2)
        g,a,_,_=min(lines,key=lambda v:v[2]*p+v[3])
        if out and out[-1]["id"]==g["id"]:out[-1]["hi"]=hi
        else:out.append({"lo":lo,"hi":hi,"id":g["id"],"method":g["method"],
                         "Q":g["Q"],"A":a,"C":g["C"]})
    return out


def self_check():
    assert first_at([0., .95, .9, 1.]) == 2
    assert stable_at([0., .95, .9, 1.]) == 4
    assert first_at([0., .9]) is None
    assert wilson(0, 10)[0] == 0.
    assert wilson(10, 10)[1] == 1.
    # The original fixed-A comparison is a special case of the joint inequality.
    ca, cg, qa, qg = 8.31, 42.34, 5000, 2000
    crossing = (cg-ca)/(qa-qg)
    assert abs(crossing*1000-11.343333333333334) < 1e-9
    for B in [1, 16]:
        for tp, tt, aa, ag, sa, sg in [(.001,.02,2000,6000,0.,0.),
                                     (.03,.001,2000,6000,2.,.5)]:
            ta=(qa-Q0)*tp+aa*tt+B*ca+sa
            tg=(qg-Q0)*tp+ag*tt+B*cg+sg
            lhs=(qa-qg)*tp+(aa-ag)*tt+(sa-sg)
            assert (tg < ta) == (lhs > B*(cg-ca))


def bootstrap_crossings(groups, cache, n_boot=2000):
    """Reestimate attack requirements and compute differences for fixed pairs.

    Profiling subsets, not attack orders, are resampled. These intervals do
    not include candidate-search uncertainty, new attack blocks, or new keys.
    """
    rng=np.random.default_rng(2026111617)
    draws=rng.integers(0,10,(n_boot,10))
    weights=np.array([np.bincount(d,minlength=10)/10 for d in draws])
    result={}
    with zipfile.ZipFile(cache) as z:
        for b,m in [("D1","DDPM16-6000"),("AES_GPU","FM4-6000")]:
            a=next(g for g in groups if g["dataset"]==b and g["Q"]==5000 and g["method"]=="HW-template-fixed")
            g=next(g for g in groups if g["dataset"]==b and g["Q"]==2000 and g["method"]==m)
            assert a["seeds"]==g["seeds"]
            arrays=[]
            for row in [a,g]:
                arrays.append(np.array([(np.load(io.BytesIO(z.read(f"{b}/Q{row['Q']}/{row['method']}/s{seed}.npy")))==0).mean(0)
                                        for seed in row["seeds"]]))
            attacks=[]
            for values in arrays:
                batches=[]
                for start in range(0,n_boot,50):
                    sr=weights[start:start+50]@values
                    hit=sr>=REQUIREMENT-1e-12
                    batches.append(np.where(hit.any(axis=1),hit.argmax(axis=1)+1,np.nan))
                attacks.append(np.concatenate(batches))
            ca=weights@np.asarray(a["C_subsets"]);cg=weights@np.asarray(g["C_subsets"])
            valid=np.isfinite(attacks[0])&np.isfinite(attacks[1])
            bands=[]
            for B in range(1,17):
                crossings=(B*(cg-ca)+(attacks[1]-attacks[0])*.001)/(a["Q"]-g["Q"])
                point=(B*(g["C"]-a["C"])+(g["A95"]-a["A95"])*.001)/(a["Q"]-g["Q"])
                bands.append({"B":B,"point":point,"ci":np.percentile(crossings[valid],[2.5,97.5]).tolist()})
            result[b]={"from":a["id"],"to":g["id"],"resamples":n_boot,
                       "eligible_resamples":int(valid.sum()),"tau_t":.001,
                       "A_from_ci":np.nanpercentile(attacks[0],[2.5,97.5]).tolist(),
                       "A_to_ci":np.nanpercentile(attacks[1],[2.5,97.5]).tolist(),"bands":bands}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compact-cache", action="store_true")
    args = ap.parse_args()
    self_check()
    source = DATA / "frozen-records.json"
    ids = set(read(DATA / "selected-record-ids.json"))
    rows = [r for r in read(source)["records"] if r["record_id"] in ids]
    assert len(rows) == len(ids)
    grouped = collections.defaultdict(list)
    for r in rows:
        grouped[(r["dataset"], r["Q"], r["method"])].append(r)
    cache = DATA / "rank-curves.zip"
    compact_path = ROOT / "cache/ranks-only.pending.zip"
    groups, curve_arrays, compact_manifest = [], {}, {}
    with zipfile.ZipFile(cache) as z:
        compact_input = "manifest.json" in z.namelist()
        prior_manifest = json.loads(z.read("manifest.json")) if compact_input else {}
        outzip = (zipfile.ZipFile(compact_path, "w", compression=zipfile.ZIP_DEFLATED,
                                  compresslevel=6)
                  if args.compact_cache and not compact_input else None)
        try:
            for (dataset, q, method), rs in sorted(grouped.items()):
                rs = sorted(rs, key=lambda r:r["seed"])
                assert len(rs)==10 and len(set(r["seed"] for r in rs))==10
                subset_sr, subset_ge, score_times = [], [], []
                for r in rs:
                    stem = member_name(r)
                    if compact_input:
                        payload = z.read(stem+".npy")
                        meta = prior_manifest[stem]
                        assert hashlib.sha256(payload).hexdigest()==meta["rank_npy_sha256"]
                        assert r["rank_source"]["sha256"]==meta["source_sha256"]
                        ranks=np.load(io.BytesIO(payload),allow_pickle=False)
                    else:
                        raw = z.read(stem+".npz")
                        assert hashlib.sha256(raw).hexdigest()==r["rank_source"]["sha256"]
                        with zipfile.ZipFile(io.BytesIO(raw)) as inner:
                            payload=inner.read("rank2.npy")
                        ranks=np.load(io.BytesIO(payload),allow_pickle=False)
                        meta={"source_sha256":r["rank_source"]["sha256"],
                              "rank_npy_sha256":hashlib.sha256(payload).hexdigest()}
                        extra=stem+".npz.result.json"
                        if extra in z.namelist():
                            result=json.loads(z.read(extra))
                            value=result.get("scoring_seconds_10000_median")
                            if value is not None:meta["score_seconds_10000"]=value
                        if outzip is not None:
                            outzip.writestr(stem+".npy",payload)
                            compact_manifest[stem]=meta
                    assert ranks.shape==(20,10000) and ranks.dtype==np.int16
                    assert ranks.min()>=0 and ranks.max()<=510
                    for a, rank_values in r["rank2_at_budget"].items():
                        assert np.array_equal(ranks[:,int(a)-1],rank_values),(stem,a)
                    subset_sr.append((ranks==0).mean(axis=0))
                    subset_ge.append(ranks.mean(axis=0)/2)
                    if "score_seconds_10000" in meta:
                        score_times.append(meta["score_seconds_10000"])
                sr=np.asarray(subset_sr); ge=np.asarray(subset_ge)
                mean_sr=sr.mean(axis=0); mean_ge=ge.mean(axis=0)
                gid=f"{dataset}/Q{q}/{method}"
                cs=np.array([r["cost_seconds"]["compute"] for r in rs])
                g={"id":gid,"dataset":dataset,"Q":q,"method":method,
                   "name":NAMES[method],"n_subsets":len(rs),"seeds":[r["seed"] for r in rs],
                   "cohort":rs[0]["cohort"],"learner":rs[0]["learner"],
                   "C":float(cs.mean()),"C_median":float(np.median(cs)),
                   "C_subsets":cs.tolist(),"A95":first_at(mean_sr),
                   "A95_sustained":stable_at(mean_sr),"GE_lt_1":first_at((mean_ge < 1).astype(float),1.),
                   "quality":{},"score_seconds_10000":float(np.mean(score_times)) if score_times else None}
                for a in [100,1000,10000]:
                    k=int(np.count_nonzero(sr[:,a-1]>=REQUIREMENT-1e-12))
                    g["quality"][str(a)]={"SR":float(mean_sr[a-1]),"GE":float(mean_ge[a-1]),
                                          "subset_passes":k,"subset_pass_wilson":wilson(k,len(rs))}
                if g["A95"] is not None:
                    k=int(np.count_nonzero(sr[:,g["A95"]-1]>=REQUIREMENT-1e-12))
                    g["at_A95"]={"SR":float(mean_sr[g["A95"]-1]),
                                  "GE":float(mean_ge[g["A95"]-1]),"subset_passes":k}
                groups.append(g)
                curve_arrays[gid+"/SR"]=mean_sr
                curve_arrays[gid+"/GE"]=mean_ge
        finally:
            if outzip is not None:
                outzip.writestr("manifest.json",json.dumps(compact_manifest,sort_keys=True))
                outzip.close()
    if compact_manifest:
        with zipfile.ZipFile(compact_path) as z:
            assert len(json.loads(z.read("manifest.json")))==len(rows)
            assert z.testzip() is None
        compact_path.replace(cache)
    scenarios=[]
    for b in DATASETS:
        bg=[g for g in groups if g["dataset"]==b]
        for tp,tt in [(0.001,0.001),(0.020,0.001),(0.001,0.020),(0.020,0.020)]:
            for B in [1,16]:
                scenarios.append({"dataset":b,"tau_p":tp,"tau_t":tt,"B":B,
                                  "winner":winner(bg,tp,tt,B)})
    sensitivity=[]
    for b in DATASETS:
        bg=[g for g in groups if g["dataset"]==b]
        for tp,tt in [(0.001,0.001),(0.020,0.001),(0.001,0.020),(0.020,0.020)]:
            eligible=[g for g in bg if g["A95_sustained"] is not None]
            g=min(eligible,key=lambda g:cost(g,tp,tt,A=g["A95_sustained"]))
            sensitivity.append({"dataset":b,"tau_p":tp,"tau_t":tt,"id":g["id"],
                                "Q":g["Q"],"A":g["A95_sustained"],
                                "total":cost(g,tp,tt,A=g["A95_sustained"])})
    result={"scope":{"requirement":REQUIREMENT,"Q0":Q0,"attack_cap":10000,
                      "attainment":"first observed crossing of mean SR; no interpolation",
                      "C":"recorded model-construction time, including preparation and selection",
                      "acquisition":"parameter scenarios; no new physical measurement",
                      "B":"independent-copy compute sensitivity, not measured full-key recovery",
                      "setup":"zero in numerical frontiers; route-specific term in the general model",
                      "inference":"not imputed; available GPU scoring measurements retained separately",
                      "source_evidence_sha256":hashlib.sha256(source.read_bytes()).hexdigest()},
            "records":len(rows),"groups":groups,"scenarios":scenarios,
            "sustained_scenarios":sensitivity,
            "pareto":{b:pareto([g for g in groups if g["dataset"]==b]) for b in DATASETS},
            "equal_price_envelopes":{b:joint_envelope([g for g in groups if g["dataset"]==b]) for b in DATASETS},
            "attack_1ms_envelopes":{b:joint_envelope([g for g in groups if g["dataset"]==b],tau_t=.001) for b in DATASETS},
            "fixed_A_envelopes":{b:{str(a):fixed_envelope([g for g in groups if g["dataset"]==b],a)
                                     for a in [1000,10000]} for b in DATASETS}}
    result["bootstrap_crossings"]=bootstrap_crossings(groups,cache)
    save(ROOT/"analysis.json",result)
    np.savez_compressed(ROOT/"curves.npz",**curve_arrays)
    print("Verified",len(rows),"records and",len(groups),"recipe groups; compact cache bytes",cache.stat().st_size)
    for b in DATASETS:
        print("\n",b)
        for m in ["Real","RShare-T","FM4-6000","DDPM16-6000","HW-template-fixed"]:
            print(m,[(g["Q"],g["A95"],g["A95_sustained"],round(g["C"],4))
                     for g in groups if g["dataset"]==b and g["method"]==m])
        print("equal 1ms",next(x["winner"] for x in scenarios if x["dataset"]==b and
                              x["tau_p"]==.001 and x["tau_t"]==.001 and x["B"]==1))
        print("profile 20ms / attack 1ms",next(x["winner"] for x in scenarios if x["dataset"]==b and
                                             x["tau_p"]==.02 and x["tau_t"]==.001 and x["B"]==1))


if __name__=="__main__":
    main()
