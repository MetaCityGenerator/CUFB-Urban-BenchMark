# -*- coding: utf-8 -*-
"""Benchmark scorer: tests whether a (generated or real) city conforms to the statistical regularities of the Chinese county reference.
Input: the city result.json + spectrum.json (computed with the same analyze pipeline), compared with, in the --ref folder,
national_bounds.csv (must contain share_gt1 / sign_consistency), national_reference_spectrum.json and spectrum_deviation.csv.

Protocol v2 (default):
  * every check belongs to one layer; conformity is computed per layer and weighted by layer -> score in [0,1]:
      invariant  scalar CV < 10%        w=3   interval check [p5,p95]
      weak       scalar CV 10-40%       w=1.5 interval
      signature  scalar CV > 40%        w=0.5 interval (place-specific metrics, low weight)
      assort     morphological assortativity (area/elongation/compactness) w=2   direction check: value > margin*p50 (national), margin=0.25
      height     height structure: height assortativity, Moran's I, commercial height ratio  w=2.5  direction check (all three collapse to no signal after a height shuffle/jitter)
      L3_dir     use pairs with sign consistency >= 0.9  w=3   direction check: sign(log lift) = national direction and |log lift| >= margin*|log p50|
      L3_mixed   other use pairs        w=0.5 interval
      use_height height ratio within use  w=1   interval
      L4         pairs with consistency >= 0.75  w=0.25 interval (other L4 pairs not checked)
      L6         entropy-scale curve    w=3   curve check: mean over 5 radii of |x-p50|/((p95-p5)/2) <= 1
      spectrum   morphology spectrum    w=3   EMD percentile in the national distribution <= 95
  * for direction checks both "no-signal" (right direction but below 25% of the national median magnitude) and "reversed" count as failures.
Protocol v1: all national_bounds rows equally weighted, interval checks only (old behaviour, --protocol v1).

Usage:
  single: python score.py --city <result folder> --ref dataset/output_county
  batch : python score.py --batch <root with many result folders> --ref ... [--out scores.csv]
  AUC   : python score.py --real <root> --perturbed <root> --ref ...   # AUC of real vs perturbed (ranked by score) + county-paired bootstrap 95% CI -> {ref}/auc_{protocol}.csv
"""
import json, csv, os, sys, glob, argparse, ast, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spectrum as SP
from scipy.stats import wasserstein_distance

WEIGHTS={'invariant':3,'weak':1.5,'signature':0.5,'assort':2,'height':2.5,'L3_dir':3,'L3_mixed':0.5,'use_height':1,'L4':0.25,'L6':3,'spectrum':3}
MARGIN=0.25; L3_DIR_MIN=0.9; L4_MIN=0.75; UH_DIR_MIN=0.85

def load_bounds(p):
    B={}
    for r in csv.DictReader(open(p,encoding='utf-8-sig')):
        try:
            d={'p5':float(r['p5']),'p50':float(r['p50']),'p95':float(r['p95']),'cv':float(r.get('cv_pct') or 'nan')}
            for k in ('share_gt1','sign_consistency'):
                v=r.get(k); d[k]=float(v) if v not in (None,'') else None
            B[(r['metric'],r['key'])]=d
        except Exception: pass
    return B

def getpath(d,dotted):
    for k in dotted.split('.'):
        d=d.get(k) if isinstance(d,dict) else None
        if d is None: return None
    return d

def city_value(metric,key,res):
    if metric=='scalar':
        if key.startswith('L6.'):
            e=(res.get('L6_entropy') or {}).get('mean_entropy'); rad=(res.get('L6_entropy') or {}).get('radii')
            if not e or any(x is None for x in e): return None
            if key=='L6.slope_dH_dlnr': return float(np.polyfit(np.log(rad),e,1)[0])
            r=int(key.split('_')[1]); return e[rad.index(r)] if r in rad else None
        if key.startswith('intensity.'):
            _,u,k=key.split('.'); return ((res.get('intensity') or {}).get(u) or {}).get(k)
        return getpath(res,key)
    if metric=='assort': return (res.get('assort') or {}).get(key)
    if metric=='use_height_ratio': return getpath((res.get('use_morph') or {}).get(key,{}),'height_ratio')
    if metric in ('L3','L4'):
        try: a,b=ast.literal_eval(key)
        except Exception: return None
        for x in res.get(metric,[]):
            if metric=='L3' and x.get('a')==a and x.get('b')==b: return x.get('lift')
            if metric=='L4' and x.get('use')==a and x.get('top_road')==b: return x.get('lift')
    return None

def spectrum_dev(city_dir, ref_dir):
    sp=f"{city_dir}/spectrum.json"; rp=f"{ref_dir}/national_reference_spectrum.json"
    if not (os.path.exists(sp) and os.path.exists(rp)): return None,None
    s=json.load(open(sp,encoding='utf-8')); ref=json.load(open(rp,encoding='utf-8'))['ref1']; tot=0
    for k,(lo,hi,nb,_) in SP.BINS1.items():
        if s['d1'].get(k) and ref.get(k):
            c=SP.centers(lo,hi,nb); tot+=wasserstein_distance(c,c,np.array(s['d1'][k]),np.array(ref[k]))/(hi-lo)
    dev=round(float(tot),4); pctl=None; dc=f"{ref_dir}/spectrum_deviation.csv"
    if os.path.exists(dc):
        alld=[float(r['emd_total']) for r in csv.DictReader(open(dc,encoding='utf-8-sig')) if r['emd_total']]
        if alld: pctl=round(float((np.array(alld)<dev).mean()*100),1)
    return dev,pctl

def score_v1(res,B):
    viol=[]; checked=0
    for (metric,key),d in B.items():
        v=city_value(metric,key,res)
        if v is None: continue
        checked+=1
        if v<d['p5'] or v>d['p95']: viol.append({'layer':metric,'metric':metric,'key':key,'value':round(v,3),'p5':d['p5'],'p95':d['p95'],'kind':'low' if v<d['p5'] else 'high'})
    return {'checked':checked,'violations':viol,'conformity':round(1-len(viol)/checked,3) if checked else None}

def score_v2(res,B,city_dir,ref_dir):
    checks={k:[] for k in WEIGHTS}; viol=[]
    def add(layer,ok,detail):
        checks[layer].append(ok)
        if not ok: viol.append(dict(layer=layer,**detail))
    def interval(layer,metric,key,d,v):
        ok=d['p5']<=v<=d['p95']; add(layer,ok,{'metric':metric,'key':key,'value':round(v,3),'p5':d['p5'],'p95':d['p95'],'kind':'ok' if ok else ('low' if v<d['p5'] else 'high')})
    l6_terms=[]
    for (metric,key),d in B.items():
        v=city_value(metric,key,res)
        if v is None: continue
        if metric=='scalar':
            if key.startswith('L6.entropy_'):
                half=(d['p95']-d['p5'])/2; l6_terms.append(abs(v-d['p50'])/half if half>0 else 0); continue
            if key=='L6.slope_dH_dlnr': continue
            if key=='height_moran':
                thr=MARGIN*abs(d['p50']); ok=v>thr
                add('height',ok,{'metric':metric,'key':key,'value':round(v,3),'need':f">{thr:.3f}",'kind':'ok' if ok else ('reversed' if v<0 else 'no-signal')}); continue
            layer='invariant' if d['cv']<10 else 'weak' if d['cv']<40 else 'signature'; interval(layer,metric,key,d,v)
        elif metric=='assort':
            thr=MARGIN*abs(d['p50']); ok=(v>thr) if d['p50']>0 else (v<-thr)
            add('height' if key=='log_h' else 'assort',ok,{'metric':metric,'key':key,'value':round(v,3),'need':f">{thr:.3f}",'kind':'ok' if ok else ('reversed' if v*d['p50']<0 else 'no-signal')})
        elif metric=='L3':
            sc=d.get('sign_consistency') or 0
            if sc>=L3_DIR_MIN and d['p50']>0 and v>0:
                nat=math.log(d['p50']); cty=math.log(max(v,1e-6)); ok=(cty*nat>0) and abs(cty)>=MARGIN*abs(nat)
                add('L3_dir',ok,{'metric':metric,'key':key,'value':round(v,3),'national_p50':d['p50'],'kind':'ok' if ok else ('reversed' if cty*nat<0 else 'no-signal')})
            elif sc>=L3_DIR_MIN and v<=0:
                add('L3_dir',False,{'metric':metric,'key':key,'value':v,'national_p50':d['p50'],'kind':'no-signal'})
            else: interval('L3_mixed',metric,key,d,v)
        elif metric=='use_height_ratio':
            sc=d.get('sign_consistency') or 0
            if sc>=UH_DIR_MIN and d['p50']>0 and v>0:
                nat=math.log(d['p50']); cty=math.log(v); ok=(cty*nat>0) and abs(cty)>=MARGIN*abs(nat)
                add('height',ok,{'metric':metric,'key':key,'value':round(v,3),'national_p50':d['p50'],'kind':'ok' if ok else ('reversed' if cty*nat<0 else 'no-signal')})
            else: interval('use_height',metric,key,d,v)
        elif metric=='L4':
            if (d.get('sign_consistency') or 0)>=L4_MIN: interval('L4',metric,key,d,v)
    if l6_terms:
        m=float(np.mean(l6_terms)); ok=m<=1
        add('L6',ok,{'metric':'L6','key':'curve','value':round(m,3),'kind':'ok' if ok else 'curve-off','note':'mean |x-p50|/half-band over 5 radii'})
    dev,pctl=spectrum_dev(city_dir,ref_dir)
    if pctl is not None:
        ok=pctl<=95; add('spectrum',ok,{'metric':'spectrum','key':'emd_total','value':dev,'percentile':pctl,'kind':'ok' if ok else 'atypical'})
    layers={}; num=0; den=0
    for k,lst in checks.items():
        if not lst: continue
        c=sum(lst)/len(lst); layers[k]={'n':len(lst),'conformity':round(c,3),'weight':WEIGHTS[k]}; num+=WEIGHTS[k]*c; den+=WEIGHTS[k]
    return {'score':round(num/den,4) if den else None,'layers':layers,'violations':[v for v in viol],
            'checked':sum(len(l) for l in checks.values()),'spectrum_emd':dev,'spectrum_percentile':pctl}

def score_city(city_dir,ref_dir,B,protocol='v2'):
    res=json.load(open(f"{city_dir}/result.json",encoding='utf-8'))
    if protocol=='v1':
        out=score_v1(res,B); out['spectrum_emd'],out['spectrum_percentile']=spectrum_dev(city_dir,ref_dir); out['score']=out['conformity']
    else: out=score_v2(res,B,city_dir,ref_dir)
    out['city']=os.path.basename(os.path.normpath(city_dir)); out['protocol']=protocol
    meta=f"{city_dir}/_meta.json"
    if os.path.exists(meta):
        try: out['perturbation']=json.load(open(meta,encoding='utf-8')).get('perturbation')
        except Exception: pass
    if not out.get('perturbation') and '__' in out['city']: out['perturbation']=out['city'].rsplit('__',1)[1]  # naming convention of perturb.py
    return out

def print_report(out):
    print(f"\n===== {out['city']} conformity with the Chinese county reference [{out['protocol']}] =====")
    if out['protocol']=='v2':
        print(f"score = {out['score']:.3f}   {out['checked']} checks, {len(out['violations'])} failed")
        for k,l in out['layers'].items(): print(f"  {k:10s} n={l['n']:3d}  conformity {l['conformity']*100:5.1f}%  w={l['weight']}")
    else: print(f"{out['checked']} checks, {len(out['violations'])} failed, conformity {out['conformity']*100:.1f}%")
    if out.get('spectrum_percentile') is not None:
        p=out['spectrum_percentile']; verdict="typical" if p<70 else ("large deviation" if p<95 else "highly atypical")
        print(f"spectrum deviation EMD={out['spectrum_emd']}  (above {p}% of national counties, {verdict})")
    if out['violations']:
        print("failed checks:")
        for v in sorted(out['violations'],key=lambda x:x['layer']): print(f"  [{v['layer']:10s}] {str(v['key'])[:34]:34} {v.get('value')}  {v['kind']}")

def auc(pos,neg):
    """AUC = P(score_real > score_perturbed), ties count 0.5."""
    pos=np.asarray(pos,float); neg=np.asarray(neg,float)
    if not len(pos) or not len(neg): return None
    gt=(pos[:,None]>neg[None,:]).sum(); eq=(pos[:,None]==neg[None,:]).sum()
    return round(float((gt+0.5*eq)/(len(pos)*len(neg))),4)

def auc_ci(real_by_city,pert_by_city,n_boot=2000,seed=0):
    """95% interval of the AUC by county-paired resampling (real and perturbed cases come from the same counties; the county is the resampling unit); returns (auc, lo, hi)."""
    cities=sorted(set(real_by_city)&set(pert_by_city))
    if len(cities)<2: return None,None,None
    r=np.array([real_by_city[c] for c in cities],float); q=np.array([pert_by_city[c] for c in cities],float)
    rng=np.random.RandomState(seed); vals=[]
    for _ in range(n_boot):
        i=rng.randint(0,len(cities),len(cities)); vals.append(auc(r[i],q[i]))
    return auc(r,q),round(float(np.percentile(vals,2.5)),3),round(float(np.percentile(vals,97.5)),3)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--city'); ap.add_argument('--batch'); ap.add_argument('--real'); ap.add_argument('--perturbed')
    ap.add_argument('--ref',required=True); ap.add_argument('--protocol',choices=['v1','v2'],default='v2'); ap.add_argument('--out',default=None)
    a=ap.parse_args(); B=load_bounds(f"{a.ref}/national_bounds.csv")
    if a.protocol=='v2' and not any(d.get('sign_consistency') is not None for d in B.values()):
        sys.exit("national_bounds.csv has no sign_consistency column: regenerate it with the current benchmark.py aggregate")
    if a.city:
        out=score_city(a.city,a.ref,B,a.protocol); print_report(out)
        json.dump(out,open(f"{a.city}/score.json",'w',encoding='utf-8'),ensure_ascii=False,indent=1); print(f"\n-> {a.city}/score.json"); return
    rows=[]
    def run_root(root,label):
        for d in sorted(glob.glob(f"{root}/*")):
            if not os.path.exists(f"{d}/result.json"): continue
            try: o=score_city(d,a.ref,B,a.protocol)
            except Exception as e: print(f"  [ERR] {os.path.basename(d)}: {str(e)[:80]}"); continue
            rows.append({'city':o['city'],'group':label,'perturbation':o.get('perturbation') or '','score':o['score'],
                         'violations':len(o['violations']),'checked':o['checked'],'spectrum_percentile':o.get('spectrum_percentile'),
                         **{f"conf_{k}":v['conformity'] for k,v in (o.get('layers') or {}).items()}})
    if a.batch: run_root(a.batch,'batch')
    if a.real: run_root(a.real,'real')
    if a.perturbed: run_root(a.perturbed,'perturbed')
    if not rows: sys.exit("no result folders to score")
    keys=sorted({k for r in rows for k in r},key=lambda k:(k not in ('city','group','perturbation','score','violations','checked','spectrum_percentile'),k))
    out=a.out or f"{a.ref}/scores_{a.protocol}.csv"
    with open(out,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"[score] {len(rows)} cities -> {out}")
    if a.real and a.perturbed:
        real=[r['score'] for r in rows if r['group']=='real' and r['score'] is not None]
        per=[r for r in rows if r['group']=='perturbed' and r['score'] is not None]
        print(f"\nreal: {len(real)} counties, median score {np.median(real):.3f} (p5 {np.percentile(real,5):.3f})")
        rb={r['city']:r['score'] for r in rows if r['group']=='real' and r['score'] is not None}
        print(f"{'perturbation':16s} {'n':>4s} {'median':>9s} {'AUC(real>pert)':>14s} {'95% CI (county-paired, 2000)':>22s}")
        arows=[]
        for k in sorted({r['perturbation'] for r in per}):
            sc=[r['score'] for r in per if r['perturbation']==k]
            pb={r['city'].split('__')[0]:r['score'] for r in per if r['perturbation']==k}
            a_,lo,hi=auc_ci(rb,pb); ci=f"[{lo:.3f}, {hi:.3f}]" if lo is not None else '-'
            print(f"{k:16s} {len(sc):4d} {np.median(sc):9.3f} {auc(real,sc):14.3f} {ci:>22s}")
            arows.append({'perturbation':k,'n':len(sc),'median_perturbed':round(float(np.median(sc)),4),'auc':auc(real,sc),'auc_ci_lo':lo,'auc_ci_hi':hi})
        print(f"{'ALL':16s} {len(per):4d} {np.median([r['score'] for r in per]):9.3f} {auc(real,[r['score'] for r in per]):14.3f}")
        ao=os.path.join(os.path.dirname(out) or '.',f"auc_{a.protocol}.csv")
        with open(ao,'w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(arows[0])); w.writeheader(); w.writerows(arows)
        print(f"[auc] -> {ao}")
if __name__=='__main__': main()
