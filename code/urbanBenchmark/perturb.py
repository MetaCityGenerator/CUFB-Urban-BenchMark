# -*- coding: utf-8 -*-
"""Perturbed-county generator - negative and positive controls for the discriminative power of the benchmark.
Perturbs a real county folder in 7 ways and writes county folders in the same format ({building,landuse,roads}.shp + _meta.json)
that can be fed directly to benchmark.py analyze / score.py. Each perturbation breaks specific layers only; the rest should hold:

  use_shuffle     parcel land uses permuted (geometry unchanged)       breaks L3/L4/L5/L6/use_morph/intensity; morphology unchanged
  height_shuffle  building heights permuted                          breaks assort.log_h/height_moran/use_morph/intensity/sun_ratio
  rotate          all three layers rotated by theta in [15, 75] deg   **control**: everything except orient/cardinal/grid_share should hold
  scale           every footprint scaled x k about its centre (0.5)   breaks area_dist/intensity/form_diversity; positions unchanged
  jitter          building positions displaced by N(0, 50 m)          breaks assort/moran/sun_ratio/setback/L2 assignment/fractal
  road_shuffle    street class fclass permuted                        breaks L4 only
  uniform         every building replaced by an equal square (median height)   breaks area_dist/form_diversity/spectrum; positions unchanged

Usage:
  python perturb.py --input dataset/input_county/<county> --out dataset/perturbed [--kinds all|a,b,c] [--seed 0]
  -> dataset/perturbed/<county>__use_shuffle/ ... then
  python benchmark.py analyze --input dataset/perturbed/<county>__use_shuffle --out dataset/perturbed_out/<county>__use_shuffle
  python score.py --city dataset/perturbed_out/<county>__use_shuffle --ref dataset/output_county
Batch: give a root folder as --input plus --n 50 to draw 50 counties at random (with building.shp and at least --min-n buildings)."""
import os, sys, json, glob, argparse, shutil, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geopandas as gpd, pyogrio
from shapely import affinity
from shapely.geometry import box
import benchmark as B

KINDS=['use_shuffle','height_shuffle','rotate','scale','jitter','road_shuffle','uniform']
LAYERS={'building':'building.shp','landuse':'landuse.shp','roads':'roads.shp'}

def read(folder, layer):
    p=f"{folder}/{LAYERS[layer]}"
    if not os.path.exists(p): return None
    g=pyogrio.read_dataframe(p); return g[g.geometry.notna()&~g.geometry.is_empty].reset_index(drop=True)

def write(g, folder, layer):
    os.makedirs(folder, exist_ok=True); g.to_file(f"{folder}/{LAYERS[layer]}", driver='ESRI Shapefile', encoding='utf-8')

def _utm(b, lu):
    """Same as benchmark.load_county: UTM zone from the landuse (else building) representative point."""
    src=lu if (lu is not None and len(lu)) else b
    c=src.to_crs(4326).geometry.representative_point(); return B.auto_epsg(c.x.mean(), c.y.mean())

def perturb(folder, out_root, kinds, seed=0, scale_k=0.5, jitter_m=50.0, uniform_side=15.0):
    name=os.path.basename(folder.rstrip('/\\')); rng=np.random.RandomState(seed)
    b=read(folder,'building'); lu=read(folder,'landuse'); r=read(folder,'roads')
    meta={}
    if os.path.exists(f"{folder}/_meta.json"):
        meta=json.load(open(f"{folder}/_meta.json",encoding='utf-8'))
    epsg=_utm(b,lu) if (b is not None or lu is not None) else None
    made=[]
    for kind in kinds:
        od=f"{out_root}/{name}__{kind}"; bb,ll,rr=b,lu,r
        if kind=='use_shuffle':
            if lu is None: continue
            perm=rng.permutation(len(lu)); ll=lu.copy()
            for c in ['Class','Level1','Level1_cn','Level2','Level2_cn']:
                if c in ll: ll[c]=lu[c].values[perm]
        elif kind=='height_shuffle':
            if b is None: continue
            bb=b.copy(); bb['Height']=b['Height'].values[rng.permutation(len(b))]
        elif kind=='rotate':
            theta=float(rng.uniform(15,75)); ref=(b if b is not None else lu).to_crs(epsg)
            cx,cy=ref.geometry.representative_point().x.mean(), ref.geometry.representative_point().y.mean()
            def rot(g):
                if g is None: return None
                p=g.to_crs(epsg); p['geometry']=p.geometry.rotate(theta, origin=(cx,cy)); return p.to_crs(g.crs)
            bb,ll,rr=rot(b),rot(lu),rot(r); meta_extra={'theta_deg':round(theta,2)}
        elif kind=='scale':
            if b is None: continue
            p=b.to_crs(epsg); p['geometry']=p.geometry.scale(scale_k, scale_k, origin='centroid'); bb=p.to_crs(b.crs)
        elif kind=='jitter':
            if b is None: continue
            p=b.to_crs(epsg); dx=rng.normal(0,jitter_m,len(p)); dy=rng.normal(0,jitter_m,len(p))
            p['geometry']=[affinity.translate(g,x,y) for g,x,y in zip(p.geometry.values,dx,dy)]; bb=p.to_crs(b.crs)
        elif kind=='road_shuffle':
            if r is None: continue
            rr=r.copy(); rr['fclass']=r['fclass'].values[rng.permutation(len(r))]
        elif kind=='uniform':
            if b is None: continue
            p=b.to_crs(epsg); c=p.geometry.representative_point(); h=uniform_side/2
            p['geometry']=[box(x-h,y-h,x+h,y+h) for x,y in zip(c.x.values,c.y.values)]
            p['Height']=float(np.nanmedian(p['Height'].astype(float))); bb=p.to_crs(b.crs)
        else: raise ValueError(kind)
        if os.path.exists(od): shutil.rmtree(od)
        for layer,g in [('building',bb),('landuse',ll),('roads',rr)]:
            if g is not None: write(g,od,layer)
        m=dict(meta); m.update({'perturbation':kind,'source':name,'seed':seed})
        if kind=='rotate': m.update(meta_extra)
        if kind=='scale': m['scale_k']=scale_k
        if kind=='jitter': m['jitter_m']=jitter_m
        if kind=='uniform': m['uniform_side']=uniform_side
        json.dump(m,open(f"{od}/_meta.json",'w',encoding='utf-8'),ensure_ascii=False)
        made.append(od)
    return made

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True,help='a county folder, or a root folder holding many counties (with --n)')
    ap.add_argument('--out',required=True); ap.add_argument('--kinds',default='all')
    ap.add_argument('--n',type=int,default=0,help='draw n counties at random when --input is a root folder'); ap.add_argument('--min-n',type=int,default=1000)
    ap.add_argument('--seed',type=int,default=0); ap.add_argument('--scale-k',type=float,default=0.5)
    ap.add_argument('--jitter-m',type=float,default=50.0); ap.add_argument('--uniform-side',type=float,default=15.0)
    ap.add_argument('--list',default=None,help='write the drawn county names to this file (usable as aggregate --holdout)')
    a=ap.parse_args(); kinds=KINDS if a.kinds=='all' else a.kinds.split(',')
    for k in kinds:
        if k not in KINDS: sys.exit(f"unknown kind {k}; choose from {KINDS}")
    if os.path.exists(f"{a.input}/building.shp") or os.path.exists(f"{a.input}/landuse.shp"): folders=[a.input]
    else:
        cand=sorted(d for d in glob.glob(f"{a.input}/*") if os.path.isdir(d) and os.path.exists(f"{d}/building.shp") and not os.path.basename(d).startswith('_'))
        rng=np.random.RandomState(a.seed); rng.shuffle(cand); folders=[]
        for d in cand:
            if len(folders)>=a.n: break
            try:
                if pyogrio.read_info(f"{d}/building.shp")['features']>=a.min_n: folders.append(d)
            except Exception: pass
        if a.list:
            with open(a.list,'w',encoding='utf-8',newline='\n') as fp: fp.write('\n'.join(os.path.basename(d) for d in folders)+'\n')
            print(f"[perturb] list of drawn counties -> {a.list}")
    os.makedirs(a.out,exist_ok=True)
    for i,d in enumerate(folders):
        made=perturb(d,a.out,kinds,seed=a.seed+i,scale_k=a.scale_k,jitter_m=a.jitter_m,uniform_side=a.uniform_side)
        print(f"[perturb] {os.path.basename(d)}: {len(made)} perturbations -> {a.out}",flush=True)
if __name__=='__main__': main()
