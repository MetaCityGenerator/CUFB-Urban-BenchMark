# -*- coding: utf-8 -*-
"""Technical Validation tools (for the Scientific Data descriptor):
  crosscheck   our morphology metrics vs momepy (building level: rectangularity / elongation / compactness / orientation), our street-orientation entropy vs osmnx (Boeing 2019 implementation)
  sensitivity  analysis parameters at x0.5 / x1.5: median relative change of every metric + Spearman correlation of the county ranking (rank stability)
  bootstrap    R repeated 80% building subsamples without replacement: sampling CI half-width (relative) of building metrics
  heightnoise  height-error propagation: noise added to the CMAB model heights (i.i.d. log-normal sigma=0.15/0.30/0.45, 250 m cell-correlated noise sigma=0.30, 3 m storey rounding),
               only building-dependent blocks recomputed; reports median relative change / signed change / county-rank Spearman of height metrics
Usage (the county list can be dataset/ref_holdout/holdout.txt):
  python validate.py crosscheck  --root dataset/input_county --counties <list> [--n 30] --out dataset/validation
  python validate.py sensitivity --root dataset/input_county --counties <list> [--n 20] --out dataset/validation
  python validate.py bootstrap   --root dataset/input_county --counties <list> [--n 20] [--reps 20] [--frac 0.8] --out dataset/validation
  python validate.py heightnoise --root dataset/input_county --counties <list> [--n 20] [--reps 3] --out dataset/validation
  python validate.py summarize   --out dataset/validation      # only recompute the summary tables (no reruns)
  labelnoise   land-use label-error propagation: 5/10/20% of parcels get another Level1_cn (3 seeds); reports change and direction retention of L3/L6/use_height/FAR (TV10)
"""
import os, sys, json, glob, argparse, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B
from export import row_of

def read_list(path, n, root):
    names=[l.strip() for l in open(path,encoding='utf-8') if l.strip() and not l.startswith('#')]
    names=[c for c in names if os.path.exists(f"{root}/{c}/building.shp")]
    return names[:n] if n else names

# ---------------- crosscheck ----------------
def crosscheck_county(folder, cap=20000, seed=0):
    import momepy, osmnx as ox, networkx as nx, geopandas as gpd
    from scipy.stats import pearsonr, spearmanr
    D=B.load_county(folder); out={'county':os.path.basename(folder)}
    b=D['b']
    if b is not None and len(b)>50:
        bb=b[b.geometry.area>10].reset_index(drop=True)
        if len(bb)>cap: bb=bb.sample(cap,random_state=seed).reset_index(drop=True)
        ours=B.morph_feats(bb)
        ext=pd.DataFrame({
            'rect':momepy.rectangularity(bb.geometry).values,          # A / A(MRR) - same definition
            'elong':momepy.elongation(bb.geometry).values,             # Gil 2012 MBR elongation
            'circ':momepy.circular_compactness(bb.geometry).values,    # A / A(minimum enclosing circle)
            'orient':momepy.orientation(bb.geometry).values,           # deviation from the orthogonal direction, 0-45 deg
        })
        ang=ours['orient'].values; dc=np.minimum(np.minimum(np.abs(ang-0),np.abs(ang-90)),np.abs(ang-180))
        pairs=[('rect',ours['rect'].values,ext['rect'].values),('log_elong',ours['log_elong'].values,ext['elong'].values),
               ('compact',ours['compact'].values,ext['circ'].values),('orient_dev',dc,ext['orient'].values)]
        for k,x,y in pairs:
            m=np.isfinite(x)&np.isfinite(y)
            if m.sum()>50:
                out[f'{k}_pearson']=round(float(pearsonr(x[m],y[m])[0]),4); out[f'{k}_spearman']=round(float(spearmanr(x[m],y[m])[0]),4)
                if k in ('rect','orient_dev'): out[f'{k}_mad']=round(float(np.mean(np.abs(x[m]-y[m]))),4)  # mean absolute difference for identical definitions
        out['n_buildings']=int(len(bb))
    r=D['r']
    if r is not None and len(r)>20:
        rr=r[['geometry']].copy(); rr['length']=rr.geometry.length; r4=rr.to_crs(4326)
        r4=r4[r4.geometry.geom_type.isin(['LineString','MultiLineString'])].explode(index_parts=False).reset_index(drop=True)
        G=momepy.gdf_to_nx(r4[['length','geometry']],approach='primal',multigraph=True,length='mm_len')
        for node in G.nodes: G.nodes[node]['x']=node[0]; G.nodes[node]['y']=node[1]
        G=nx.MultiDiGraph(G); G.graph['crs']='EPSG:4326'
        G=ox.bearing.add_edge_bearings(G)
        ln36=np.log(36)
        out['road_entropy_ours']=None; out['ox_entropy_unweighted']=round(float(ox.bearing.orientation_entropy(G,num_bins=36))/ln36,4)
        out['ox_entropy_weighted']=round(float(ox.bearing.orientation_entropy(G,num_bins=36,weight='length'))/ln36,4)
        # our implementation (36 x 5 deg bins, mod 180, length-weighted) vs a Boeing-style recomputation (36 x 10 deg bins, 0-360 bidirectional, length-weighted)
        bear=[]; wts=[]
        for geom in r.geometry.values:
            ls=geom.geoms if geom.geom_type=='MultiLineString' else [geom]
            for ln in ls:
                c=np.asarray(ln.coords); seg=np.diff(c,axis=0); L=np.sqrt((seg**2).sum(1))
                bear.extend((np.degrees(np.arctan2(seg[:,1],seg[:,0]))%180).tolist()); wts.extend(L.tolist())
        bear=np.array(bear); wts=np.array(wts)
        h,_=np.histogram(bear,bins=36,range=(0,180),weights=wts); p=h/h.sum(); p=p[p>0]
        out['road_entropy_ours']=round(float(-(p*np.log(p)).sum()/ln36),4)
        b360=np.concatenate([bear,bear+180]); w360=np.concatenate([wts,wts])
        h,_=np.histogram(b360,bins=36,range=(0,360),weights=w360); p=h/h.sum(); p=p[p>0]
        out['road_entropy_boeing_style']=round(float(-(p*np.log(p)).sum()/ln36),4)
        out['n_road_edges']=int(G.number_of_edges())
    return out

def cmd_crosscheck(a):
    names=read_list(a.counties,a.n,a.root); os.makedirs(a.out,exist_ok=True); rows=[]
    for i,c in enumerate(names):
        t=time.time()
        try: rows.append(crosscheck_county(f"{a.root}/{c}")); print(f"  [{i+1}/{len(names)}] {c} {time.time()-t:.0f}s",flush=True)
        except Exception as e: print(f"  [ERR] {c}: {str(e)[:100]}",flush=True)
    df=pd.DataFrame(rows); df.to_csv(f"{a.out}/crosscheck_counties.csv",index=False,encoding='utf-8-sig')
    summarize_crosscheck(df,a.out)

def summarize_crosscheck(df,out):
    from scipy.stats import pearsonr, spearmanr
    S={'n_counties':int(len(df))}
    for k in ['rect','log_elong','compact','orient_dev']:
        if f'{k}_pearson' in df:
            S[k]={'pearson_median':round(float(df[f'{k}_pearson'].median()),4),'pearson_p5':round(float(df[f'{k}_pearson'].quantile(.05)),4),
                  'spearman_median':round(float(df[f'{k}_spearman'].median()),4)}
            if f'{k}_mad' in df: S[k]['mad_median']=round(float(df[f'{k}_mad'].median()),4)
    if 'ox_entropy_weighted' in df:
        d=df.dropna(subset=['road_entropy_ours','ox_entropy_weighted'])
        S['road_entropy']={'ours_vs_osmnx_weighted_pearson':round(float(pearsonr(d.road_entropy_ours,d.ox_entropy_weighted)[0]),4),
            'ours_vs_osmnx_weighted_spearman':round(float(spearmanr(d.road_entropy_ours,d.ox_entropy_weighted)[0]),4),
            'boeing_style_vs_osmnx_weighted_mad':round(float(np.mean(np.abs(d.road_entropy_boeing_style-d.ox_entropy_weighted))),4),
            'ours_median':round(float(d.road_entropy_ours.median()),4),'osmnx_weighted_median':round(float(d.ox_entropy_weighted.median()),4),
            'osmnx_unweighted_median':round(float(d.ox_entropy_unweighted.median()),4)}
    json.dump(S,open(f"{out}/crosscheck_summary.json",'w',encoding='utf-8'),ensure_ascii=False,indent=1)
    print(json.dumps(S,ensure_ascii=False,indent=1))

# ---------------- sensitivity ----------------
SWEEP={'bldg_gap':[3.75,11.25],'parcel_gap':[7.5,22.5,30.0],'ring_buf':[10.0,30.0],'moran_k':[4,12],'fractal_res':[4.0,12.0],
       'sun_radius':[30.0,90.0],'sun_cone':[0.3,0.9],'l6_windows':[250,750]}
def flat(res): return {k:v for k,v in row_of(res.get('folder','x'),res,{}).items() if isinstance(v,(int,float)) and k not in ('lat','epsg','n_buildings','n_errors')}

def cmd_sensitivity(a):
    names=read_list(a.counties,a.n,a.root); od=f"{a.out}/sensitivity"; os.makedirs(od,exist_ok=True)
    variants=[('baseline',{})]+[(f"{k}={v}",{k:v}) for k,vs in SWEEP.items() for v in vs]
    for i,c in enumerate(names):
        for tag,pr in variants:
            fp=f"{od}/{c}__{tag}.json"
            if os.path.exists(fp): continue
            try:
                res=B.analyze(f"{a.root}/{c}",None,do_spectrum=False,render_fig=False,outdir=od,params=pr)
                json.dump(res,open(fp,'w',encoding='utf-8'),ensure_ascii=False)
            except Exception as e: print(f"  [ERR] {c} {tag}: {str(e)[:80]}",flush=True)
        print(f"  [{i+1}/{len(names)}] {c}",flush=True)
    summarize_sensitivity(a.out)

def summarize_sensitivity(out):
    from scipy.stats import spearmanr
    od=f"{out}/sensitivity"; recs=[]
    for fp in glob.glob(f"{od}/*__*.json"):
        c,tag=os.path.basename(fp)[:-5].split('__',1); r=json.load(open(fp,encoding='utf-8')); recs.append({'county':c,'variant':tag,**flat(r)})
    df=pd.DataFrame(recs)
    if df.empty: print('no sensitivity results'); return
    base=df[df.variant=='baseline'].set_index('county'); rows=[]
    metrics=[m for m in df.columns if m not in ('county','variant')]
    for tag in sorted(df.variant.unique()):
        if tag=='baseline': continue
        v=df[df.variant==tag].set_index('county'); idx=base.index.intersection(v.index)
        for m in metrics:
            x=base.loc[idx,m].astype(float); y=v.loc[idx,m].astype(float); ok=x.notna()&y.notna()&(x!=0)
            if ok.sum()<5: continue
            rel=((y[ok]-x[ok]).abs()/x[ok].abs()).median(); rho=spearmanr(x[ok],y[ok])[0]
            if rel>1e-9 or abs(rho-1)>1e-9:
                rows.append({'variant':tag,'metric':m,'n':int(ok.sum()),'median_rel_change':round(float(rel),4),'spearman_rank':round(float(rho),4)})
    S=pd.DataFrame(rows).sort_values(['variant','median_rel_change'],ascending=[True,False])
    S.to_csv(f"{out}/sensitivity_summary.csv",index=False,encoding='utf-8-sig')
    print(f"[sensitivity] {base.shape[0]} counties x {df.variant.nunique()-1} variants -> sensitivity_summary.csv")
    top=S[S.median_rel_change>0.05].sort_values('median_rel_change',ascending=False)
    print("(parameter, metric) with median relative change > 5%:"); print(top.head(25).to_string(index=False) if len(top) else "  none")
    print("10 lowest rank correlations:"); print(S.nsmallest(10,'spearman_rank').to_string(index=False))

# ---------------- bootstrap ----------------
def cmd_bootstrap(a):
    names=read_list(a.counties,a.n,a.root); od=f"{a.out}/bootstrap"; os.makedirs(od,exist_ok=True)
    for i,c in enumerate(names):
        for rep in range(a.reps):
            fp=f"{od}/{c}__rep{rep}.json"
            if os.path.exists(fp): continue
            try:
                res=B.analyze(f"{a.root}/{c}",None,do_spectrum=False,render_fig=False,outdir=od,params={'subsample':a.frac,'seed':rep})
                json.dump(res,open(fp,'w',encoding='utf-8'),ensure_ascii=False)
            except Exception as e: print(f"  [ERR] {c} rep{rep}: {str(e)[:80]}",flush=True)
        print(f"  [{i+1}/{len(names)}] {c}",flush=True)
    summarize_bootstrap(a.out)

BUILDING_METRICS=['area_dist.logn_mu','area_dist.logn_sigma','area_dist.hill_alpha','orient.cardinal10','orient.entropy','form_diversity','height_moran','fractal_D','sun_ratio_median',
                  'assort.log_h','assort.log_area','assort.log_elong','assort.compact','setback.median','setback.frontage15','use_height_ratio.\u5546\u670d','use_height_ratio.\u5c45\u4f4f','FAR.\u5c45\u4f4f','FAR.\u5546\u670d','FAR.\u5de5\u4e1a']
def summarize_bootstrap(out):
    od=f"{out}/bootstrap"; recs=[]
    for fp in glob.glob(f"{od}/*__rep*.json"):
        c,tag=os.path.basename(fp)[:-5].split('__',1); r=json.load(open(fp,encoding='utf-8')); recs.append({'county':c,'rep':tag,**flat(r)})
    df=pd.DataFrame(recs)
    if df.empty: print('no bootstrap results'); return
    rows=[]
    for m in [m for m in BUILDING_METRICS if m in df.columns]:
        g=df.groupby('county')[m]; med=g.median(); lo=g.quantile(.025); hi=g.quantile(.975)
        half=((hi-lo)/2); rel=(half/med.abs()).replace([np.inf,-np.inf],np.nan)
        rows.append({'metric':m,'n_counties':int(med.notna().sum()),'median_ci_halfwidth':round(float(half.median()),4),
                     'median_rel_halfwidth':round(float(rel.median()),4),'p90_rel_halfwidth':round(float(rel.quantile(.9)),4)})
    S=pd.DataFrame(rows).sort_values('median_rel_halfwidth',ascending=False)
    S.to_csv(f"{out}/bootstrap_summary.csv",index=False,encoding='utf-8-sig')
    print(f"[bootstrap] {df.county.nunique()} counties x {df.rep.nunique()} reps -> bootstrap_summary.csv"); print(S.to_string(index=False))

# ---------------- height noise ----------------
# CMAB heights are bagging-XGBoost estimates (paper: partition models R2 > 0.8, error < 6 m; street-view check of 2,500 buildings R2 = 0.72).
# No specific error is assumed; instead the error -> metric response is reported: i.i.d. log-normal noise (sigma 0.15/0.30/0.45 ~ relative error 15/30/45%),
# corr: half the variance from noise shared within 250 m cells (spatially correlated error from shared neighbourhood features, which inflates Moran/assortativity), half independent;
# floor3: heights rounded to 3 m storeys (discreteness of real heights).
NOISE={'iid_0.15':('iid',0.15),'iid_0.30':('iid',0.30),'iid_0.45':('iid',0.45),'corr_0.30':('corr',0.30),'floor3':('floor',0.0)}
HEIGHT_METRICS=['height_moran','assort.log_h','sun_ratio_median','use_height_ratio.\u5546\u670d','use_height_ratio.\u5c45\u4f4f','use_height_ratio.\u5de5\u4e1a',
                'FAR.\u5c45\u4f4f','FAR.\u5546\u670d','FAR.\u5de5\u4e1a','form_diversity']
def noisy_loader(orig, kind, sigma, seed, cell=250.0):
    def f(folder):
        D=orig(folder); b=D['b']
        if b is None or 'Height' not in b or not len(b): return D
        rng=np.random.RandomState(seed); h=b['Height'].astype(float).values; n=len(h)
        if kind=='iid': h2=h*np.exp(rng.normal(0,sigma,n))
        elif kind=='corr':
            rp=b.geometry.representative_point(); gx=np.floor(rp.x.values/cell).astype(np.int64); gy=np.floor(rp.y.values/cell).astype(np.int64)
            key=pd.factorize(pd.Series(gx*100003+gy))[0]; s2=sigma/np.sqrt(2)
            h2=h*np.exp(rng.normal(0,s2,key.max()+1)[key]+rng.normal(0,s2,n))
        elif kind=='floor': h2=np.clip(np.round(h/3.0),1,None)*3.0
        else: raise ValueError(kind)
        b=b.copy(); b['Height']=h2; D['b']=b; return D
    return f

def cmd_heightnoise(a):
    names=read_list(a.counties,a.n,a.root); od=f"{a.out}/heightnoise"; os.makedirs(od,exist_ok=True)
    orig=B.load_county
    variants=[('baseline',None)]+[(f"{tag}_s{sd}",(tag,sd)) for tag in NOISE for sd in range(a.reps if NOISE[tag][0]!='floor' else 1)]
    for i,c in enumerate(names):
        for tag,spec in variants:
            fp=f"{od}/{c}__{tag}.json"
            if os.path.exists(fp): continue
            try:
                B.load_county=orig if spec is None else noisy_loader(orig,NOISE[spec[0]][0],NOISE[spec[0]][1],spec[1])
                res=B.analyze(f"{a.root}/{c}",None,full=False,do_spectrum=False,render_fig=False,outdir=od)
                json.dump(res,open(fp,'w',encoding='utf-8'),ensure_ascii=False)
            except Exception as e: print(f"  [ERR] {c} {tag}: {str(e)[:80]}",flush=True)
            finally: B.load_county=orig
        print(f"  [{i+1}/{len(names)}] {c}",flush=True)
    summarize_heightnoise(a.out)

def summarize_heightnoise(out):
    from scipy.stats import spearmanr
    od=f"{out}/heightnoise"; recs=[]
    for fp in glob.glob(f"{od}/*__*.json"):
        c,tag=os.path.basename(fp)[:-5].split('__',1); r=json.load(open(fp,encoding='utf-8'))
        noise=tag.rsplit('_s',1)[0] if tag!='baseline' else 'baseline'
        recs.append({'county':c,'noise':noise,'variant':tag,**flat(r)})
    df=pd.DataFrame(recs)
    if df.empty: print('no heightnoise results'); return
    base=df[df.noise=='baseline'].set_index('county'); rows=[]
    metrics=[m for m in HEIGHT_METRICS if m in df.columns]
    for noise in [k for k in NOISE if k in set(df.noise)]:
        v=df[df.noise==noise].groupby('county')[metrics].mean(); idx=base.index.intersection(v.index)   # average over seeds first
        for m in metrics:
            x=base.loc[idx,m].astype(float); y=v.loc[idx,m].astype(float); ok=x.notna()&y.notna()&(x!=0)
            if ok.sum()<5: continue
            d=(y[ok]-x[ok])/x[ok].abs()
            rows.append({'noise':noise,'metric':m,'n':int(ok.sum()),'median_signed_change':round(float(d.median()),4),
                         'median_abs_change':round(float(d.abs().median()),4),'spearman_rank':round(float(spearmanr(x[ok],y[ok])[0]),4),
                         'baseline_median':round(float(x[ok].median()),4),'noisy_median':round(float(y[ok].median()),4)})
    S=pd.DataFrame(rows)
    S.to_csv(f"{out}/heightnoise_summary.csv",index=False,encoding='utf-8-sig')
    print(f"[heightnoise] {base.shape[0]} counties x {df.noise.nunique()-1} noise types -> heightnoise_summary.csv")
    print(S.to_string(index=False))

# ---------- land-use label-error propagation (TV10): 5/10/20% of parcels get another Level1_cn (another class of the county, drawn by frequency); only land-use-dependent blocks ----------
LABEL_FRAC={'label_0.05':0.05,'label_0.10':0.10,'label_0.20':0.20}
LABEL_METRICS=['L3.\u5c45\u4f4f_\u5c45\u4f4f','L3.\u516c\u7ba1_\u516c\u7ba1','L3.\u5de5\u4e1a_\u5de5\u4e1a','L3.\u516c\u7ba1_\u5c45\u4f4f','L3.\u5c45\u4f4f_\u5de5\u4e1a','L3.\u5546\u670d_\u5c45\u4f4f',
               'L6.entropy_500','L6.entropy_2000','L6.entropy_8000','L6.slope','use_height_ratio.\u5546\u670d','use_height_ratio.\u5c45\u4f4f','FAR.\u5c45\u4f4f','FAR.\u5546\u670d']
def label_loader(orig, frac, seed):
    def f(folder):
        D=orig(folder); lu=D['lu']
        if lu is None or not len(lu): return D
        rng=np.random.RandomState(seed); lab=lu['Level1_cn'].values.copy(); n=len(lab); k=int(round(frac*n))
        idx=rng.choice(n,k,replace=False); pool=lu['Level1_cn'].values
        new=rng.choice(pool,k,replace=True)
        for _ in range(6):                                   # redraw where the original label was drawn, so the label really is wrong
            same=new==lab[idx]
            if not same.any(): break
            new[same]=rng.choice(pool,int(same.sum()),replace=True)
        lab[idx]=new; lu=lu.copy(); lu['Level1_cn']=lab; D['lu']=lu; return D
    return f

def cmd_labelnoise(a):
    names=read_list(a.counties,a.n,a.root); od=f"{a.out}/labelnoise"; os.makedirs(od,exist_ok=True)
    orig=B.load_county
    variants=[('baseline',None)]+[(f"{tag}_s{sd}",(tag,sd)) for tag in LABEL_FRAC for sd in range(a.reps)]
    for i,c in enumerate(names):
        for tag,spec in variants:
            fp=f"{od}/{c}__{tag}.json"
            if os.path.exists(fp): continue
            try:
                B.load_county=orig if spec is None else label_loader(orig,LABEL_FRAC[spec[0]],spec[1])
                res=B.analyze(f"{a.root}/{c}",None,full=True,do_spectrum=False,render_fig=False,outdir=od)
                json.dump(res,open(fp,'w',encoding='utf-8'),ensure_ascii=False)
            except Exception as e: print(f"  [ERR] {c} {tag}: {str(e)[:80]}",flush=True)
            finally: B.load_county=orig
        print(f"  [{i+1}/{len(names)}] {c}",flush=True)
    summarize_labelnoise(a.out)

def summarize_labelnoise(out):
    from scipy.stats import spearmanr
    od=f"{out}/labelnoise"; recs=[]
    for fp in glob.glob(f"{od}/*__*.json"):
        c,tag=os.path.basename(fp)[:-5].split('__',1); r=json.load(open(fp,encoding='utf-8'))
        noise=tag.rsplit('_s',1)[0] if tag!='baseline' else 'baseline'
        recs.append({'county':c,'noise':noise,'variant':tag,**flat(r)})
    df=pd.DataFrame(recs)
    if df.empty: print('no labelnoise results'); return
    base=df[df.noise=='baseline'].set_index('county'); rows=[]
    metrics=[m for m in LABEL_METRICS if m in df.columns]
    for noise in [k for k in LABEL_FRAC if k in set(df.noise)]:
        v=df[df.noise==noise].groupby('county')[metrics].mean(); idx=base.index.intersection(v.index)
        for m in metrics:
            x=base.loc[idx,m].astype(float); y=v.loc[idx,m].astype(float); ok=x.notna()&y.notna()&(x!=0)
            if ok.sum()<5: continue
            d=(y[ok]-x[ok])/x[ok].abs(); row={'noise':noise,'metric':m,'n':int(ok.sum()),'median_signed_change':round(float(d.median()),4),
                 'median_abs_change':round(float(d.abs().median()),4),'spearman_rank':round(float(spearmanr(x[ok],y[ok])[0]),4),
                 'baseline_median':round(float(x[ok].median()),4),'noisy_median':round(float(y[ok].median()),4)}
            if m.startswith('L3.'):   # direction retention: share of counties still on the same side (>1 / <1) as the baseline
                row['sign_kept']=round(float(((x[ok]>1)==(y[ok]>1)).mean()),3)
            rows.append(row)
    S=pd.DataFrame(rows); S.to_csv(f"{out}/labelnoise_summary.csv",index=False,encoding='utf-8-sig')
    print(f"[labelnoise] {base.shape[0]} counties x {df.noise.nunique()-1} error rates -> labelnoise_summary.csv"); print(S.to_string(index=False))

def cmd_summarize(a):
    if os.path.exists(f"{a.out}/crosscheck_counties.csv"): summarize_crosscheck(pd.read_csv(f"{a.out}/crosscheck_counties.csv",encoding='utf-8-sig'),a.out)
    summarize_sensitivity(a.out); summarize_bootstrap(a.out)
    if os.path.isdir(f"{a.out}/heightnoise"): summarize_heightnoise(a.out)
    if os.path.isdir(f"{a.out}/labelnoise"): summarize_labelnoise(a.out)

def main():
    ap=argparse.ArgumentParser(); s=ap.add_subparsers(dest='cmd',required=True)
    for name,fn in [('crosscheck',cmd_crosscheck),('sensitivity',cmd_sensitivity),('bootstrap',cmd_bootstrap),('heightnoise',cmd_heightnoise),('labelnoise',cmd_labelnoise)]:
        p=s.add_parser(name); p.add_argument('--root',required=True); p.add_argument('--counties',required=True); p.add_argument('--n',type=int,default=0)
        p.add_argument('--out',required=True)
        if name=='bootstrap': p.add_argument('--reps',type=int,default=20); p.add_argument('--frac',type=float,default=0.8)
        if name in ('heightnoise','labelnoise'): p.add_argument('--reps',type=int,default=3,help='number of random seeds per noise type')
        p.set_defaults(f=fn)
    p=s.add_parser('summarize'); p.add_argument('--out',required=True); p.set_defaults(f=cmd_summarize)
    a=ap.parse_args(); a.f(a)
if __name__=='__main__': main()
