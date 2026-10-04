# -*- coding: utf-8 -*-
"""Urban-form benchmark - nationwide county-level computation.
Commands: fit (legacy: fits a KMeans morphology typology; not used by analyze) / analyze (all blocks for one county) / runall (all counties, resumable) / aggregate (national bounds)
The analysis is k-free: continuous morphology only, no building clustering (Function is not used). Results are written to result.json in each county folder.
Construction-era strata: analyze/runall --era pre1990|1991_2000|2001_2010|post2010 keeps only buildings of that era (Age field),
  writes result_{era}.json / spectrum_{era}.json; aggregate --strata era rolls them up into national_bounds_by_era.csv.
Stratified aggregation: aggregate --strata type (urban district / county-level city / county) | size (building-count terciles) | era; --holdout lists test counties excluded from the bounds.
Failures: an exception in any metric block is recorded in result.json _errors{block: exception}; a missing key means failed or no data, never silent."""
import pyogrio, geopandas as gpd, pandas as pd, numpy as np, json, os, glob, sys, argparse, time, warnings
warnings.filterwarnings('ignore')
from collections import Counter, defaultdict
import itertools
USES=['\u5c45\u4f4f\u7528\u5730','\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730','\u5de5\u4e1a\u7528\u5730','\u5546\u670d\u7528\u5730','\u9053\u8def\u4e0e\u4ea4\u901a\u8bbe\u65bd\u7528\u5730']
UI={u:i for i,u in enumerate(USES)}
ROAD_RANK=['motorway','trunk','primary','secondary','tertiary','residential','service','living_street','unclassified','other']
RI={r:i for i,r in enumerate(ROAD_RANK)}
RADII=[500,1000,2000,4000,8000]
MORPH_FEATS=['log_area','compact','rect','log_elong','log_h']
# analysis parameters (override with analyze/runall --param key=value; validate.py sensitivity scans these)
PARAMS={
    'bldg_cap':120000,   # maximum number of buildings (random subsample above it)
    'bldg_gap':7.5,      # L1 assortativity: buildings are adjacent if their buffers of this distance intersect (footprint gap < 2x)
    'parcel_gap':15.0,   # L3/L5: parcels are adjacent if their buffers of this distance intersect
    'ring_buf':20.0,     # L4: buffer around the parcel boundary; the highest-ranked intersecting street class is taken
    'moran_k':8,         # k nearest neighbours for height_moran
    'fractal_res':8.0,   # raster resolution for the fractal dimension (m)
    'sun_radius':60.0,   # spacing-to-height ratio: search radius for the nearest neighbour (m)
    'sun_cone':0.6,      # spacing-to-height ratio: northward cone |dx| <= cone*dy (0.6 ~ +-31 deg)
    'l6_windows':500,    # L6: number of random windows
    'setback_max':300.0, # building-street distance: not counted beyond this distance
    'subsample':1.0,     # share of buildings sampled without replacement (< 1 for bootstrap)
    'seed':0,            # random seed for sampling
}
def parse_params(items):
    out=dict(PARAMS)
    for it in items or []:
        k,v=it.split('=',1)
        if k not in out: raise ValueError(f"unknown param {k}; choose from {list(out)}")
        out[k]=type(out[k])(float(v)) if isinstance(out[k],(int,float)) and not isinstance(out[k],bool) else v
    return out
# construction-era bins (Age field: year 1984-2018 or 'AF2018' = built after the 2018 imagery; the pile-up near 1985 is the start of observation, so pre1990 is a lower bound)
ERAS=['pre1990','1991_2000','2001_2010','post2010']
def era_of(age):
    a=pd.Series(age).astype(str)
    yr=pd.to_numeric(a.str.replace('AF','',regex=False),errors='coerce').values
    yr=np.where(a.str.startswith('AF').values, 2019, yr)
    out=np.full(len(yr),None,dtype=object)
    out[yr<=1990]='pre1990'; out[(yr>=1991)&(yr<=2000)]='1991_2000'; out[(yr>=2001)&(yr<=2010)]='2001_2010'; out[yr>=2011]='post2010'
    return out

def auto_epsg(lon,lat): return 32600+int((lon+180)//6)+1

def morph_feats(g):
    area=g.geometry.area.values; peri=g.geometry.length.values
    mrr=g.geometry.minimum_rotated_rectangle(); n=len(g)
    lo=np.empty(n); sh=np.empty(n); rect=np.empty(n); ang=np.empty(n)
    for i,(gm,a) in enumerate(zip(mrr.values,area)):
        try:
            c=np.asarray(gm.exterior.coords)[:4]; e=np.diff(np.vstack([c,c[:1]]),axis=0)
            el=np.sqrt((e**2).sum(1)); k=int(np.argmax(el))
            L=max(el[0],el[1]); S=min(el[0],el[1])
            lo[i]=L; sh[i]=S; rect[i]=a/(L*S) if L*S>0 else np.nan
            ang[i]=np.degrees(np.arctan2(e[k,1],e[k,0]))%180
        except Exception: lo[i]=sh[i]=rect[i]=ang[i]=np.nan
    h=g['Height'].astype(float).values if 'Height' in g else np.full(n,np.nan)
    df=pd.DataFrame({'log_area':np.log10(np.clip(area,10,1e6)),'compact':4*np.pi*area/(peri**2),
        'rect':rect,'log_elong':np.log10(np.clip(lo/sh,1,50)),'log_h':np.log10(np.clip(h,2,300))})
    df['orient']=ang; df['height']=h; df['area']=area; df['long']=lo; df['short']=sh
    return df

def load_county(folder, lu_folder=None):
    """lu_folder: a county folder of the same name under another root whose landuse.shp is used (to swap the land-use source without copying buildings/streets)."""
    out={}
    lp=f"{lu_folder or folder}/landuse.shp"
    if os.path.exists(lp):
        lu=pyogrio.read_dataframe(lp,columns=['Level1_cn']); lu=lu[lu.geometry.notna()&~lu.geometry.is_empty]
    else: lu=None
    # epsg from landuse centroid (4326) else building
    if lu is not None and len(lu):
        c=lu.to_crs(4326).geometry.representative_point(); epsg=auto_epsg(c.x.mean(),c.y.mean())
    else: epsg=None
    bp=f"{folder}/building.shp"
    if os.path.exists(bp):
        b=pyogrio.read_dataframe(bp); b=b[b.geometry.notna()&~b.geometry.is_empty]
        if epsg is None and len(b):
            c=b.to_crs(4326).geometry.representative_point(); epsg=auto_epsg(c.x.mean(),c.y.mean())
    else: b=None
    out['epsg']=epsg
    out['b']=b.to_crs(epsg) if (b is not None and len(b)) else None
    out['lu']=lu.to_crs(epsg) if (lu is not None and len(lu)) else None
    rp=f"{folder}/roads.shp"
    if os.path.exists(rp):
        r=pyogrio.read_dataframe(rp,columns=['fclass']); r=r[r.geometry.notna()&~r.geometry.is_empty]
        out['r']=r.to_crs(epsg) if len(r) else None
    else: out['r']=None
    # latitude
    if lu is not None and len(lu): out['lat']=float(lu.to_crs(4326).geometry.representative_point().y.mean())
    elif b is not None and len(b): out['lat']=float(b.to_crs(4326).geometry.representative_point().y.mean())
    else: out['lat']=None
    return out

def lift_pairs(C,deg,K):
    E=sum(C.values()); twoE=sum(deg.values()); p={k:deg[k]/twoE for k in range(K)} if twoE else {k:0 for k in range(K)}
    rows=[]
    for a,b in itertools.combinations_with_replacement(range(K),2):
        obs=C[(a,b)]; of=obs/E if E else 0; ef=(p[a]**2) if a==b else 2*p[a]*p[b]
        rows.append({'a':a,'b':b,'lift':round(of/ef,3) if ef>0 else None})
    return rows,E

# ---------- FIT ----------
def cmd_fit(a):
    from sklearn.preprocessing import StandardScaler; from sklearn.cluster import KMeans; import joblib
    fols=[d for d in glob.glob(f"{a.root}/*") if os.path.isdir(d) and os.path.exists(f"{d}/building.shp")]
    import random; random.seed(0); random.shuffle(fols); fols=fols[:a.sample_counties]
    rows=[]
    for d in fols:
        try:
            b=pyogrio.read_dataframe(f"{d}/building.shp"); b=b[b.geometry.notna()&~b.geometry.is_empty]
            if len(b)<50: continue
            c=b.to_crs(4326).geometry.representative_point(); epsg=auto_epsg(c.x.mean(),c.y.mean())
            b=b.to_crs(epsg); b=b[b.geometry.area>10]
            if len(b)>a.per_county: b=b.sample(a.per_county,random_state=0)
            df=morph_feats(b)[MORPH_FEATS]; rows.append(df)
        except Exception as e: print("skip",d,e)
    D=pd.concat(rows).replace([np.inf,-np.inf],np.nan).dropna()
    sc=StandardScaler().fit(D); km=KMeans(a.k,n_init=8,random_state=0).fit(sc.transform(D))
    joblib.dump({'scaler':sc,'km':km,'feats':MORPH_FEATS,'k':a.k},a.model)
    print(f"[fit] sample {len(D)} buildings / {len(rows)} counties -> k={a.k} -> {a.model}")

# ---------- ANALYZE one county ----------
def analyze(folder, model=None, full=True, do_spectrum=True, render_fig=True, outdir=None, era=None, params=None, lu_folder=None):
    from scipy.spatial import cKDTree
    P=dict(PARAMS); P.update(params or {})
    if outdir is None: outdir=folder
    D=load_county(folder, lu_folder); res={'folder':os.path.basename(folder),'epsg':D['epsg'],'lat':D['lat']}
    if lu_folder: res['landuse_source']=os.path.basename(os.path.dirname(os.path.normpath(lu_folder)))
    b=D['b']; lu=D['lu']; r=D['r']
    err={}
    def _fail(k,e): err[k]=f"{type(e).__name__}: {str(e)[:100]}"
    mp=f"{folder}/_meta.json"
    if os.path.exists(mp):
        try:
            meta=json.load(open(mp,encoding='utf-8')); res['PAC']=meta.get('PAC'); res['type']=meta.get('\u7c7b\u578b')
        except Exception as e: _fail('meta',e)
    suffix=f"_{era}" if era else ""
    if era:
        if era not in ERAS: raise ValueError(f"era must be one of {ERAS}")
        res['era']=era; full=False   # era strata recompute only building-dependent blocks; L3/L4/L5/L6/road/blocks depend only on parcels and streets, are era-independent and come from the main result.json
        res['_skipped']=['L3','L4','L5_comm','L6_entropy','road','blocks']
        if b is not None and 'Age' in b:
            res['n_buildings_all']=int(len(b)); b=b[era_of(b['Age'].values)==era]
        elif b is not None: _fail('era',ValueError('no Age field')); b=None
    ASSORT_FEATS=['log_h','log_area','log_elong','compact']
    # ---- building morphology (k-free, no clustering) ----
    if b is not None:
        bb=b[b.geometry.area>10].reset_index(drop=True)
        if P['subsample']<1: bb=bb.sample(frac=P['subsample'],random_state=int(P['seed'])).reset_index(drop=True)
        if len(bb)>P['bldg_cap']: bb=bb.sample(int(P['bldg_cap']),random_state=int(P['seed'])).reset_index(drop=True)
        df=morph_feats(bb)
        res['n_buildings']=int(len(bb))
        if do_spectrum:
            try:
                import spectrum as SP
                _sp=SP.spec_from_df(df, os.path.basename(folder)+suffix, D['lat'])
                json.dump(_sp, open(f"{outdir}/spectrum{suffix}.json",'w',encoding='utf-8'), ensure_ascii=False)
                if render_fig: SP.render(_sp, f"{outdir}/spectrum{suffix}.png")
            except Exception as e: _fail('spectrum',e)
        bb=bb.assign(orient=df['orient'].values, height=df['height'].values, area=df['area'].values,
                     log_h=df['log_h'].values, log_area=df['log_area'].values,
                     log_elong=df['log_elong'].values, compact=df['compact'].values)
    else: bb=None; res['n_buildings']=0
    # ---- L1 building-building morphological assortativity (k-free) ----
    if bb is not None and len(bb)>30:
        try:
            buf=gpd.GeoDataFrame(geometry=bb.geometry.buffer(P['bldg_gap']), crs=bb.crs)
            sj=gpd.sjoin(buf,bb[['geometry']],predicate='intersects'); sj=sj[sj.index<sj['index_right']]
            i=sj.index.values; jj=sj['index_right'].values; assort={}
            for f in ASSORT_FEATS:
                v=bb[f].values; x=v[i]; y=v[jj]; mm=np.isfinite(x)&np.isfinite(y); x,y=x[mm],y[mm]
                if len(x)>20:
                    X=np.concatenate([x,y]); Y=np.concatenate([y,x]); assort[f]=round(float(np.corrcoef(X,Y)[0,1]),3)
            res['assort']=assort; res['assort_edges']=int(len(sj))
        except Exception as e: _fail('assort',e)
    # ---- morphology summary metrics ----
    if bb is not None and len(bb)>30:
        ar=bb.area.values; ar=ar[(ar>10)&(ar<1e5)]; la=np.log(ar)
        xmin=np.percentile(ar,80); tail=ar[ar>=xmin]
        res['area_dist']={'logn_mu':round(float(la.mean()),3),'logn_sigma':round(float(la.std()),3),
            'hill_alpha':round(float(1+len(tail)/np.sum(np.log(tail/xmin))),3) if len(tail)>5 else None}
        # orientation
        ang=bb.orient.values; ang=ang[~np.isnan(ang)]
        dc=np.minimum(np.minimum(np.abs(ang-0),np.abs(ang-90)),np.abs(ang-180))
        h,_=np.histogram(ang,bins=36,range=(0,180)); pp=h/h.sum(); pp=pp[pp>0]
        res['orient']={'cardinal10':round(float((dc<=10).mean()),3),'entropy':round(float(-(pp*np.log(pp)).sum()/np.log(36)),3)}
        # form diversity (k-free): normalised entropy of the joint log_area x height distribution
        Hh,_,_=np.histogram2d(bb.log_area.values, bb.height.values, bins=[20,20], range=[[1,4],[2,60]])
        pp2=Hh.ravel(); pp2=pp2[pp2>0]; pp2=pp2/pp2.sum()
        res['form_diversity']=round(float(-(pp2*np.log(pp2)).sum()/np.log(len(pp2))),3) if len(pp2)>1 else None
        # height moran (knn)
        if len(bb)>50:
            rp=bb.geometry.representative_point(); xy=np.c_[rp.x.values,rp.y.values]
            tr=cKDTree(xy); _,nn=tr.query(xy,k=min(int(P['moran_k'])+1,len(bb))); z=bb.height.values-np.nanmean(bb.height.values)
            neigh=nn[:,1:]; den=np.nansum(z**2)
            res['height_moran']=round(float((len(bb)/neigh.size)*(np.nansum(z[:,None]*z[neigh])/den)),3) if den>0 else None
        # fractal (box-count on building bbox raster)
        try:
            tb=bb.total_bounds; RESr=float(P['fractal_res']); W=int((tb[2]-tb[0])/RESr)+1; H=int((tb[3]-tb[1])/RESr)+1
            if 10<W<6000 and 10<H<6000:
                grid=np.zeros((H,W),bool)
                for geom in bb.geometry.values:
                    x0,y0,x1,y1=geom.bounds
                    grid[int((y0-tb[1])/RESr):int((y1-tb[1])/RESr)+1, int((x0-tb[0])/RESr):int((x1-tb[0])/RESr)+1]=True
                sizes=[1,2,4,8,16,32]; cn=[]
                for s in sizes:
                    hh=(H//s)*s; ww=(W//s)*s
                    cn.append(grid[:hh,:ww].reshape(hh//s,s,ww//s,s).any(axis=(1,3)).sum())
                cn=np.array(cn,float); szs=np.array(sizes,float); mk=cn>0
                res['fractal_D']=round(float(-np.polyfit(np.log(szs[mk]),np.log(cn[mk]),1)[0]),3)
        except Exception as e: _fail('fractal',e)
        # sun spacing/height by latitude
        try:
            sel=(bb.height.values>=9)&(bb.height.values<=60)&(bb.area.values>80)
            gg=bb[sel].reset_index(drop=True)
            if len(gg)>30:
                rp=gg.geometry.representative_point(); xy=np.c_[rp.x.values,rp.y.values]; hh=gg.height.values
                tr=cKDTree(xy); ratios=[]
                for i in range(len(gg)):
                    best=None
                    for jx in tr.query_ball_point(xy[i],P['sun_radius']):
                        if jx==i: continue
                        dx=xy[jx,0]-xy[i,0]; dy=xy[jx,1]-xy[i,1]
                        if dy<=3 or abs(dx)>P['sun_cone']*dy: continue
                        d=np.hypot(dx,dy); best=d if best is None else min(best,d)
                    if best and hh[i]>=9: ratios.append(best/hh[i])
                if ratios: res['sun_ratio_median']=round(float(np.median(ratios)),3)
        except Exception as e: _fail('sun',e)
    # ---- L3 parcel-parcel + L5 communities + L6 multiscale ----
    if full and lu is not None and len(lu)>5:
        g=lu.copy(); g['geometry']=g.geometry.make_valid(); g=g[~g.geometry.is_empty&g.geometry.notna()].reset_index(drop=True)   # repair invalid rings (e.g. Shenzhen in the EULUC 2.0 gdb) first; identical to the former buffer(0) for valid geometry
        g['t']=g['Level1_cn']; g['area']=g.geometry.area
        buf=gpd.GeoDataFrame({'t':g['t'].values}, geometry=g.geometry.buffer(P['parcel_gap']), crs=g.crs)
        sj=gpd.sjoin(buf,g[['t','geometry']],predicate='intersects'); sj=sj[sj.index<sj['index_right']]
        types=sorted(g['t'].unique()); C=Counter(); deg=Counter()
        for x,y in zip(sj['t_left'],sj['t_right']): C[tuple(sorted((x,y)))]+=1; deg[x]+=1; deg[y]+=1
        E=sum(C.values()); twoE=sum(deg.values()); p={t:deg[t]/twoE for t in types} if twoE else {}
        rows=[]
        for x,y in itertools.combinations_with_replacement(types,2):
            obs=C[(x,y)]; of=obs/E if E else 0; ef=((p[x]**2) if x==y else 2*p[x]*p[y]) if twoE else 0
            rows.append({'a':x,'b':y,'lift':round(of/ef,3) if ef>0 else None})
        res['L3']=rows; res['L3_edges']=int(E)
        # L5 communities
        try:
            import networkx as nx; from networkx.algorithms.community import louvain_communities
            G=nx.Graph(); G.add_nodes_from(range(len(g)))
            G.add_edges_from(zip(sj.index.tolist(), sj['index_right'].tolist()))
            comms=[c for c in louvain_communities(G,seed=0) if len(c)>=5]; cv=[]
            for c in comms:
                sub=g.iloc[list(c)]; comp=np.zeros(len(USES))
                for u,ar in zip(sub['Level1_cn'],sub['area']):
                    if u in UI: comp[UI[u]]+=ar
                if comp.sum()>0: cv.append([round(float(x/comp.sum()),3) for x in comp])
            res['L5_comm']=cv
        except Exception as e: _fail('L5',e)
        # L6 multiscale entropy
        try:
            rp=g.geometry.representative_point(); xy=np.c_[rp.x.values,rp.y.values]
            W=np.zeros((len(g),len(USES)))
            for i,u in enumerate(g['Level1_cn'].values):
                if u in UI: W[i,UI[u]]=g['area'].values[i]
            tr=cKDTree(xy); rng=np.random.RandomState(0); cen=rng.choice(len(g),min(int(P['l6_windows']),len(g)),replace=False)
            ent=[]
            for rad in RADII:
                e=[]
                for nbi in tr.query_ball_point(xy[cen],rad):
                    comp=W[nbi].sum(0); tot=comp.sum()
                    if tot>0: pp=comp/tot; pp=pp[pp>0]; e.append(-(pp*np.log(pp)).sum())
                ent.append(round(float(np.mean(e)),4) if e else None)
            res['L6_entropy']={'radii':RADII,'mean_entropy':ent}
        except Exception as e: _fail('L6',e)
    # ---- L2 land use -> building morphology (k-free) ----
    if bb is not None and lu is not None and len(bb)>30 and len(lu)>0:
        try:
            pts=gpd.GeoDataFrame({'h':bb.height.values,'la':bb.log_area.values,'le':bb.log_elong.values},
                                 geometry=bb.geometry.representative_point(), crs=bb.crs)
            j=gpd.sjoin(pts,lu[['Level1_cn','geometry']],predicate='within')
            cityh=float(np.nanmedian(bb.height.values)); um={}
            for u,sub in j.groupby('Level1_cn'):
                if len(sub)<20: continue
                um[u]={'n':int(len(sub)),'height_med':round(float(sub['h'].median()),1),
                       'logarea_med':round(float(sub['la'].median()),2),'elong_med':round(float(sub['le'].median()),2),
                       'height_ratio':round(float(sub['h'].median()/cityh),3) if cityh>0 else None}
            res['use_morph']=um
        except Exception as e: _fail('use_morph',e)
    # ---- L4 road enclosure + road metrics + blocks + intensity + setback ----
    if full and lu is not None and r is not None and len(lu)>0 and len(r)>5:
        try:
            keep=set(ROAD_RANK[:-1]); rr=r.copy(); rr['fc']=rr['fclass'].where(rr['fclass'].isin(keep),'other')
            ring=gpd.GeoDataFrame({'Level1_cn':lu['Level1_cn'].values}, geometry=lu.geometry.boundary.buffer(P['ring_buf']), crs=lu.crs)
            sj=gpd.sjoin(ring.reset_index(),rr[['fc','geometry']],predicate='intersects')
            best=defaultdict(lambda:99)
            for idx,fc in zip(sj['index'],sj['fc']): best[idx]=min(best[idx],RI[fc])
            ut=Counter(); un=Counter()
            for i,u in enumerate(lu['Level1_cn']):
                un[u]+=1; bcls=ROAD_RANK[best.get(i,99)] if best.get(i,99)<99 else 'none'; ut[(u,bcls)]+=1
            N=sum(un.values()); tc=Counter()
            for (u,c),n in ut.items(): tc[c]+=n
            rows=[]
            for u in sorted(un):
                for c in ROAD_RANK+['none']:
                    obs=ut[(u,c)]; exp=un[u]*tc[c]/N if N and un[u] and tc[c] else 0
                    rows.append({'use':u,'top_road':c,'lift':round(obs/exp,3) if exp>0 else None})
            res['L4']=rows
        except Exception as e: _fail('L4',e)
    if full and r is not None and len(r)>5:
        try:
            bear=[]; wts=[]
            for geom in r.geometry.values:
                ls=geom.geoms if geom.geom_type=='MultiLineString' else [geom]
                for ln in ls:
                    c=np.asarray(ln.coords); seg=np.diff(c,axis=0); L=np.sqrt((seg**2).sum(1))
                    bear.extend((np.degrees(np.arctan2(seg[:,1],seg[:,0]))%180).tolist()); wts.extend(L.tolist())
            bear=np.array(bear); wts=np.array(wts)
            h,_=np.histogram(bear,bins=36,range=(0,180),weights=wts); pp=h/h.sum(); ppp=pp[pp>0]
            dc=np.minimum(np.minimum(np.abs(bear-0),np.abs(bear-90)),np.abs(bear-180))
            res['road']={'orient_entropy':round(float(-(ppp*np.log(ppp)).sum()/np.log(36)),3),
                'grid_share':round(float(wts[dc<=10].sum()/wts.sum()),3),
                'length_km':round(float(r.geometry.length.sum()/1000),2)}
        except Exception as e: _fail('road',e)
        # blocks via polygonize
        try:
            from shapely.ops import unary_union, polygonize
            polys=[p for p in polygonize(unary_union(r.geometry.values))]
            ba=np.array([p.area for p in polys]); ba=ba[(ba>200)&(ba<2e6)]
            if len(ba): res['blocks']={'median_area':round(float(np.median(ba)),0),'superblock_share':round(float((ba>1e5).mean()),3),'n':int(len(ba))}
        except Exception as e: _fail('blocks',e)
    if r is not None and len(r)>5:
        # setback (building-dependent, also computed for era strata)
        try:
            gc=gpd.GeoDataFrame(geometry=bb.geometry.representative_point(), crs=bb.crs) if bb is not None else None
            if gc is not None and len(gc)>20:
                nr=gpd.sjoin_nearest(gc,r[['geometry']],distance_col='d'); d=nr['d'].values; d=d[d<P['setback_max']]
                res['setback']={'median':round(float(np.median(d)),1),'frontage15':round(float((d<15).mean()),3)}
        except Exception as e: _fail('setback',e)
    # intensity by use (coverage/FAR/floors)
    if bb is not None and lu is not None and len(bb)>30:
        try:
            lu2=lu.copy(); lu2['parea']=lu2.geometry.area; lu2['pid']=range(len(lu2))
            gp=bb.copy(); gp['fa']=bb.geometry.area; gp['fl']=np.clip(np.round(bb.height.values/3),1,None); gp['gfa']=gp['fa']*gp['fl']
            gp['geometry']=bb.geometry.representative_point()
            j=gpd.sjoin(gp[['fa','gfa','fl','geometry']],lu2[['pid','Level1_cn','parea','geometry']],predicate='within')
            ag=j.groupby('pid').agg(fa=('fa','sum'),gfa=('gfa','sum')).reset_index().merge(lu2[['pid','Level1_cn','parea']],on='pid')
            ag=ag[(ag.parea>500)&(ag.parea<5e5)]; ag['cvg']=ag['fa']/ag['parea']; ag['far']=ag['gfa']/ag['parea']; ag=ag[ag['cvg']<=1.5]
            inten={}
            for u,sub in ag.groupby('Level1_cn'):
                inten[u]={'coverage':round(float(sub['cvg'].median()),3),'FAR':round(float(sub['far'].median()),2),'n':int(len(sub))}
            res['intensity']=inten
        except Exception as e: _fail('intensity',e)
    res['_params']={k:v for k,v in P.items() if v!=PARAMS[k]}
    if err: res['_errors']=err
    return res

def cmd_analyze(a):
    out=a.out or a.input; os.makedirs(out,exist_ok=True); sfx=f"_{a.era}" if a.era else ""
    res=analyze(a.input, None, do_spectrum=not a.no_spectrum, render_fig=not a.no_fig, outdir=out, era=a.era, params=parse_params(a.param), lu_folder=f"{a.landuse_root}/{os.path.basename(os.path.normpath(a.input))}" if a.landuse_root else None)
    json.dump(res,open(f"{out}/result{sfx}.json",'w',encoding='utf-8'),ensure_ascii=False)
    print(f"[analyze] {os.path.basename(a.input)} -> {out}/result{sfx}.json  keys={[k for k in res if k not in ('folder','epsg','lat')]}")
    if res.get('_errors'): print(f"          errors: {res['_errors']}")

def cmd_runall(a):
    import time
    fols=sorted([d for d in glob.glob(f"{a.root}/*") if os.path.isdir(d) and os.path.exists(f"{d}/building.shp")])
    if a.shard:   # --shard k/n: process only shard k (0-based), for parallel processes with disjoint work
        k,n=map(int,a.shard.split('/')); fols=fols[k::n]
    os.makedirs(a.out,exist_ok=True)
    sfx=f"_{a.era}" if a.era else ""; params=parse_params(a.param)
    pending=[d for d in fols if a.no_resume or not os.path.exists(f"{a.out}/{os.path.basename(d)}/result{sfx}.json")]
    skipped=len(fols)-len(pending)
    print(f"[runall] {len(fols)} counties (with buildings); skipped (done) {skipped}; pending {len(pending)}",flush=True)
    if skipped and len(pending):
        print(f"         (resume: counties with an existing result.json are skipped; --no-resume forces a rerun)",flush=True)
    done=0; err=0; t0=time.time()
    for d in pending:
        name=os.path.basename(d); od=f"{a.out}/{name}"; os.makedirs(od,exist_ok=True)
        try:
            res=analyze(d,None, do_spectrum=not a.no_spectrum, render_fig=not a.no_fig, outdir=od, era=a.era, params=params, lu_folder=f"{a.landuse_root}/{name}" if a.landuse_root else None)
            json.dump(res,open(f"{od}/result{sfx}.json",'w',encoding='utf-8'),ensure_ascii=False); done+=1
            el=time.time()-t0; eta=el/done*(len(pending)-done)
            print(f"  [{done}/{len(pending)}] {name}  (elapsed {el/60:.1f} min, remaining ~{eta/60:.1f} min)",flush=True)
        except Exception as e:
            err+=1; print(f"  [ERR] {name}: {str(e)[:90]}",flush=True)
    print(f"[runall] newly done {done}, errors {err}, skipped {skipped}; total {skipped+done}/{len(fols)}")

def _load_results(root, suffix, min_n, holdout):
    """Read {root}/*/result{suffix}.json; drop small counties using n in spectrum{suffix}.json; exclude test counties. Returns [(county, result)]."""
    R=[]; bad=0; small=0; held=0
    for f in sorted(glob.glob(f"{root}/*/result{suffix}.json")):
        name=os.path.basename(os.path.dirname(f))
        if name in holdout: held+=1; continue
        try:
            with open(f,encoding='utf-8') as fp: r=json.load(fp)
        except Exception as e:
            bad+=1; print(f"  [skip corrupt] {name}: {str(e)[:60]}",flush=True); continue
        n=r.get('n_buildings'); sp=os.path.join(os.path.dirname(f),f'spectrum{suffix}.json')
        if n is None and os.path.exists(sp):
            try:
                with open(sp,encoding='utf-8') as fp: n=json.load(fp).get('n')
            except Exception: pass
        if n is not None and n < min_n: small+=1; continue
        r['_n']=n; R.append((name,r))
    print(f"[aggregate] result{suffix}.json: read {len(R)} counties (buildings >= {min_n})" + (f"; small skipped {small}" if small else "")
          + (f"; corrupt {bad}" if bad else "") + (f"; held out {held}" if held else ""))
    return R

def _bounds_rows(R, min_cities, stratum=''):
    """R=[(county,result)] -> national_bounds rows. sign: p5>1 attract / p95<1 repel;
    share_gt1 = share of counties with lift > 1 (or assort > 0), sign_consistency = max(share, 1-share)."""
    def collect(getter):
        d=defaultdict(list)
        for _,r in R:
            for key,val in getter(r):
                if val is not None: d[key].append(val)
        return d
    rows=[]
    def emit(name,d,thr=None):
        for key,vals in d.items():
            if len(vals)<max(3,min_cities): continue
            vals=np.array(vals,float); row={'stratum':stratum,'metric':name,'key':str(key),'n':len(vals),
                'p5':round(float(np.percentile(vals,5)),3),'p50':round(float(np.percentile(vals,50)),3),
                'p95':round(float(np.percentile(vals,95)),3),'cv_pct':round(float(np.std(vals)/np.mean(vals)*100),1) if np.mean(vals) else 0,
                'sign':('attract' if np.percentile(vals,5)>1 else 'repel' if np.percentile(vals,95)<1 else 'mixed') if name in ('L3','L4','use_height_ratio') else '',
                'share_gt1':None,'sign_consistency':None}
            if thr is not None:
                g=float((vals>thr).mean()); row['share_gt1']=round(g,3); row['sign_consistency']=round(max(g,1-g),3)
            rows.append(row)
    emit('assort', collect(lambda r:[(f,v) for f,v in (r.get('assort') or {}).items()]), thr=0)
    emit('L3', collect(lambda r:[((rr['a'],rr['b']),rr['lift']) for rr in r.get('L3',[])]), thr=1)
    emit('use_height_ratio', collect(lambda r:[(u,m.get('height_ratio')) for u,m in (r.get('use_morph') or {}).items()]), thr=1)
    emit('L4', collect(lambda r:[((rr['use'],rr['top_road']),rr['lift']) for rr in r.get('L4',[])]), thr=1)
    def sc(r):
        o=[]
        for path in [('area_dist','logn_mu'),('area_dist','logn_sigma'),('area_dist','hill_alpha'),
                     ('orient','cardinal10'),('orient','entropy'),('road','orient_entropy'),('road','grid_share'),
                     ('blocks','median_area'),('blocks','superblock_share'),('setback','median'),('setback','frontage15')]:
            v=r.get(path[0],{}); o.append(('.'.join(path), v.get(path[1]) if isinstance(v,dict) else None))
        for k in ['form_diversity','height_moran','fractal_D','sun_ratio_median']: o.append((k,r.get(k)))
        for u,m in (r.get('intensity') or {}).items(): o.append((f'intensity.{u}.FAR',m.get('FAR'))); o.append((f'intensity.{u}.coverage',m.get('coverage')))
        e=(r.get('L6_entropy') or {}).get('mean_entropy')
        if e and all(x is not None for x in e):
            for rad,v in zip(RADII,e): o.append((f'L6.entropy_{rad}',v))
            o.append(('L6.slope_dH_dlnr',round(float(np.polyfit(np.log(RADII),e,1)[0]),4)))
        return o
    emit('scalar', collect(sc))
    return rows

TYPE_GROUP={'\u5e02\u8f96\u533a':'\u5e02\u8f96\u533a','\u53bf\u7ea7\u5e02':'\u53bf\u7ea7\u5e02','\u53bf':'\u53bf','\u81ea\u6cbb\u53bf':'\u53bf','\u65d7':'\u53bf','\u81ea\u6cbb\u65d7':'\u53bf'}
def cmd_aggregate(a):
    holdout=set()
    if a.holdout:
        with open(a.holdout,encoding='utf-8') as fp: holdout={l.strip() for l in fp if l.strip() and not l.startswith('#')}
    strata=a.strata
    if strata=='none':
        R=_load_results(a.root,'',a.min_n,holdout); rows=_bounds_rows(R,a.min_cities)
        outp=f"{a.out_dir or a.root}/national_bounds.csv"
        if a.bootstrap>0:
            # province-level block bootstrap: neighbouring counties are not independent, so provinces are resampled as blocks; 95% CI of p50 / p5 / p95 / sign_consistency
            prov=defaultdict(list)
            for name,r in R: prov[name.split('_')[0]].append((name,r))
            keys=list(prov); rng=np.random.RandomState(0); acc=defaultdict(lambda:defaultdict(list))
            for bi in range(a.bootstrap):
                samp=[x for k in rng.choice(keys,len(keys),replace=True) for x in prov[k]]
                for row in _bounds_rows(samp,a.min_cities):
                    for c in ('p5','p50','p95','sign_consistency'):
                        if row[c] is not None: acc[(row['metric'],row['key'])][c].append(row[c])
            for row in rows:
                d=acc.get((row['metric'],row['key']),{})
                for c in ('p5','p50','p95','sign_consistency'):
                    v=d.get(c)
                    row[f'{c}_ci_lo']=round(float(np.percentile(v,2.5)),3) if v else None; row[f'{c}_ci_hi']=round(float(np.percentile(v,97.5)),3) if v else None
            print(f"[aggregate] province-level block bootstrap x{a.bootstrap} ({len(keys)} provinces) -> *_ci_lo/*_ci_hi columns")
    elif strata=='era':
        rows=[]
        for era in ERAS:
            R=_load_results(a.root,f'_{era}',a.min_n,holdout)
            if R: rows+=_bounds_rows(R,a.min_cities,stratum=era)
        outp=f"{a.out_dir or a.root}/national_bounds_by_era.csv"
    else:
        R=_load_results(a.root,'',a.min_n,holdout)
        if strata=='type':
            def meta_type(name,r):
                t=r.get('type')
                if t is None and a.input_root:
                    mp=f"{a.input_root}/{name}/_meta.json"
                    if os.path.exists(mp):
                        try: t=json.load(open(mp,encoding='utf-8')).get('\u7c7b\u578b')
                        except Exception: pass
                return TYPE_GROUP.get(t)
            groups=defaultdict(list)
            for name,r in R:
                g=meta_type(name,r)
                if g: groups[g].append((name,r))
            if not groups: print("[aggregate] no type information: result.json has no type field and --input-root was not given"); return
        elif strata=='size':
            ns=np.array([r['_n'] for _,r in R if r.get('_n') is not None],float); q1,q2=np.percentile(ns,[100/3,200/3])
            print(f"[aggregate] size tercile thresholds: small<{q1:.0f}<=medium<{q2:.0f}<=large buildings")
            groups=defaultdict(list)
            for name,r in R:
                n=r.get('_n')
                if n is None: continue
                groups['small' if n<q1 else 'medium' if n<q2 else 'large'].append((name,r))
        else: raise ValueError(strata)
        rows=[]
        for g in sorted(groups):
            print(f"  [{strata}={g}] {len(groups[g])} counties")
            rows+=_bounds_rows(groups[g],a.min_cities,stratum=g)
        outp=f"{a.out_dir or a.root}/national_bounds_by_{strata}.csv"
    if a.out_dir: os.makedirs(a.out_dir,exist_ok=True)
    df=pd.DataFrame(rows)
    if strata=='none' and len(df): df=df.drop(columns=['stratum'])
    df.to_csv(outp,index=False,encoding='utf-8-sig')
    print(f"[aggregate] {len(rows)} metric keys -> {os.path.basename(outp)}")

def main():
    ap=argparse.ArgumentParser(); s=ap.add_subparsers(dest='cmd',required=True)
    p=s.add_parser('fit'); p.add_argument('--root',required=True); p.add_argument('--model',required=True)
    p.add_argument('--k',type=int,default=8); p.add_argument('--sample-counties',type=int,default=150); p.add_argument('--per-county',type=int,default=2000); p.set_defaults(f=cmd_fit)
    p=s.add_parser('analyze'); p.add_argument('--landuse-root',default=None); p.add_argument('--input',required=True); p.add_argument('--out',default=None); p.add_argument('--model',default=None); p.add_argument('--no-spectrum',action='store_true'); p.add_argument('--no-fig',action='store_true'); p.add_argument('--era',choices=ERAS,default=None,help='keep only buildings of this construction era'); p.add_argument('--param',action='append',default=None,help='override an analysis parameter key=value, repeatable'); p.set_defaults(f=cmd_analyze)
    p=s.add_parser('runall'); p.add_argument('--shard',default=None,help='k/n: process only the k-th shard (0-based) of the sorted counties, for parallel processes'); p.add_argument('--landuse-root',default=None,help='use landuse.shp from same-named county folders under another root (e.g. dataset/input_county_euluc2)'); p.add_argument('--root',required=True); p.add_argument('--out',required=True); p.add_argument('--model',default=None); p.add_argument('--no-resume',action='store_true'); p.add_argument('--no-spectrum',action='store_true'); p.add_argument('--no-fig',action='store_true'); p.add_argument('--era',choices=ERAS,default=None); p.add_argument('--param',action='append',default=None); p.set_defaults(f=cmd_runall)
    p=s.add_parser('aggregate'); p.add_argument('--root',required=True); p.add_argument('--min-cities',type=int,default=20); p.add_argument('--min-n',type=int,default=1000)
    p.add_argument('--strata',choices=['none','type','era','size'],default='none',help='stratified aggregation: type=urban district/county-level city/county, era=construction era (run runall --era first), size=building-count terciles')
    p.add_argument('--holdout',default=None,help='list of test counties (one county folder per line) excluded from the bounds'); p.add_argument('--input-root',default=None,help='input root holding _meta.json (for old result.json files without a type field)'); p.add_argument('--out-dir',default=None,help='write the bounds to this folder (default --root)'); p.add_argument('--bootstrap',type=int,default=0,help='number of province-level block-bootstrap resamples (only with --strata none); adds 95% CI columns')
    p.set_defaults(f=cmd_aggregate)
    a=ap.parse_args(); a.f(a)
if __name__=='__main__': main()
