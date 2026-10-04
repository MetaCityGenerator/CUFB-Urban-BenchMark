# -*- coding: utf-8 -*-
"""Building morphology spectrum - per-county figure + data (fixed bins), national reference spectrum + deviation, density-peak map.
Commands:
  one    --input <county folder> [--no-fig]     # one county: write spectrum.json + spectrum.png
  all    --root <input_county> [--no-fig]       # every county, resumable
  reference --root <output_county>             # national mean reference spectrum + per-county EMD/JSD deviation
  peakmap --input <county> --xfeat height --xmin .. --xmax .. [--yfeat log_area --ymin --ymax]  # map buildings under a spectrum peak
Independent of building typology (continuous morphology only). All counties share fixed bins, so spectra can be averaged and compared."""
import sys, os, glob, json, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# CJK font registered by file path (common Windows / Linux / Mac fonts), needed because county names in figure titles are Chinese
import matplotlib.font_manager as fm
_CJK_PATHS=['C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/msyh.ttf','C:/Windows/Fonts/simhei.ttf',
    'C:/Windows/Fonts/simsun.ttc','C:/Windows/Fonts/msyhbd.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc','/System/Library/Fonts/PingFang.ttc','/Library/Fonts/Arial Unicode.ttf']
_set=False
for _p in _CJK_PATHS:
    if os.path.exists(_p):
        try:
            fm.fontManager.addfont(_p); plt.rcParams['font.sans-serif']=[fm.FontProperties(fname=_p).get_name()]; _set=True; break
        except Exception: pass
if not _set:
    for fb in ['Microsoft YaHei','SimHei','Noto Sans CJK SC','WenQuanYi Zen Hei','Arial Unicode MS']:
        if any(fb in f.name for f in fm.fontManager.ttflist): plt.rcParams['font.sans-serif']=[fb]; _set=True; break
plt.rcParams['axes.unicode_minus']=False
SPEC_CMAP=LinearSegmentedColormap.from_list('spec',['#0b0b1a','#1b1b6b','#3a3ad8','#1d9e75','#e0c020','#ff7a18','#ffe8b0'])
# fixed bins (identical for all counties)
BINS1={'log_area':(1,4,50,'log10 footprint area (m²)'),'height':(2,60,50,'height (m)'),
       'compact':(0,1,50,'compactness'),'rect':(0,1,50,'rectangularity'),
       'log_elong':(0,1.6,50,'log10 elongation (point→slab)'),'orient':(0,180,60,'orientation (°)')}
PLANES=[('log_area','height',(1,4),(2,60),30,30),('log_elong','height',(0,1.6),(2,60),30,30),
        ('compact','log_area',(0,1),(1,4),30,30)]
def centers(lo,hi,nb): e=np.linspace(lo,hi,nb+1); return (e[:-1]+e[1:])/2

def spec_from_df(df, name, lat):
    """Spectrum directly from a computed morphology dataframe (reused by benchmark, no second read from disk)."""
    s={'name':name,'n':int(len(df)),'lat':lat,'d1':{},'d2':{}}
    for k,(lo,hi,nb,_) in BINS1.items():
        v=df[k].values; v=v[np.isfinite(v)]
        h,_=np.histogram(v,bins=nb,range=(lo,hi)); s['d1'][k]=(h/h.sum()).round(6).tolist() if h.sum() else None
    for xk,yk,xr,yr,bx,by in PLANES:
        x=df[xk].values; y=df[yk].values; m=np.isfinite(x)&np.isfinite(y)
        H,_,_=np.histogram2d(x[m],y[m],bins=[bx,by],range=[xr,yr])
        s['d2'][f'{xk}__{yk}']=(H/H.sum()).round(6).T.tolist() if H.sum() else None
    return s

def compute(folder, cap=50000):
    D=B.load_county(folder); b=D['b']
    if b is None: return None
    b=b[b.geometry.area>10].reset_index(drop=True)
    if len(b)==0: return None
    if len(b)>cap: b=b.sample(cap,random_state=0).reset_index(drop=True)
    df=B.morph_feats(b)
    return spec_from_df(df, os.path.basename(folder), D['lat'])

def render(s, out):
    fig=plt.figure(figsize=(14,9)); gs=fig.add_gridspec(3,3,hspace=0.45,wspace=0.32)
    keys=list(BINS1.keys())
    for i,k in enumerate(keys):
        ax=fig.add_subplot(gs[i//3,i%3]); lo,hi,nb,lab=BINS1[k]
        d=s['d1'].get(k)
        if d: c=centers(lo,hi,nb); ax.plot(c,d,color='#185FA5',lw=2); ax.fill_between(c,d,alpha=0.15,color='#185FA5')
        ax.set_title(lab,fontsize=10); ax.tick_params(labelsize=7); ax.set_ylabel('freq',fontsize=8)
    for j,(xk,yk,xr,yr,bx,by) in enumerate(PLANES):
        ax=fig.add_subplot(gs[2,j]); g=s['d2'].get(f'{xk}__{yk}')
        if g:
            g=np.array(g); ax.imshow(g,origin='lower',aspect='auto',cmap=SPEC_CMAP,vmin=0,vmax=g.max()*0.9,
                                     extent=[xr[0],xr[1],yr[0],yr[1]])
        ax.set_title(f'{xk} × {yk}',fontsize=9); ax.tick_params(labelsize=7)
    fig.suptitle(f"{s['name']}  (n={s['n']:,}, lat={s['lat']:.1f}) - morphology spectrum",fontsize=13,y=0.995)
    fig.savefig(out,dpi=130,bbox_inches='tight',facecolor='white'); plt.close(fig)

def cmd_one(a):
    s=compute(a.input)
    if s is None: print("no buildings, skipped"); return
    out=a.out or a.input; os.makedirs(out,exist_ok=True)
    json.dump(s,open(f"{out}/spectrum.json",'w',encoding='utf-8'),ensure_ascii=False)
    if not a.no_fig: render(s,f"{out}/spectrum.png")
    print(f"[one] {s['name']} -> {out}/spectrum.json{'' if a.no_fig else ' + spectrum.png'}")

def cmd_all(a):
    import time
    fols=sorted([d for d in glob.glob(f"{a.root}/*") if os.path.isdir(d) and os.path.exists(f"{d}/building.shp")])
    os.makedirs(a.out,exist_ok=True)
    pending=[d for d in fols if a.no_resume or not os.path.exists(f"{a.out}/{os.path.basename(d)}/spectrum.json")]
    skipped=len(fols)-len(pending)
    print(f"[spectrum all] {len(fols)} counties; skipped (done) {skipped}; pending {len(pending)}",flush=True)
    done=0; err=0; t0=time.time()
    for d in pending:
        name=os.path.basename(d); od=f"{a.out}/{name}"
        try:
            s=compute(d)
            if s is None: print(f"  [skip] {name}: no buildings",flush=True); continue
            os.makedirs(od,exist_ok=True)
            json.dump(s,open(f"{od}/spectrum.json",'w',encoding='utf-8'),ensure_ascii=False)
            if not a.no_fig: render(s,f"{od}/spectrum.png")
            done+=1; el=time.time()-t0; eta=el/done*(len(pending)-done)
            print(f"  [{done}/{len(pending)}] {name}  (elapsed {el/60:.1f} min, remaining ~{eta/60:.1f} min)",flush=True)
        except Exception as e:
            err+=1; print(f"  [ERR] {name}: {str(e)[:80]}",flush=True)
    print(f"[spectrum all] newly done {done}, errors {err}, skipped {skipped}; total {skipped+done}/{len(fols)}")

def cmd_reference(a):
    from scipy.stats import wasserstein_distance
    from scipy.spatial.distance import jensenshannon
    holdout=set()
    if a.holdout:
        with open(a.holdout,encoding='utf-8') as fp: holdout={l.strip() for l in fp if l.strip() and not l.startswith('#')}
    od=a.out_dir or a.root; os.makedirs(od,exist_ok=True)
    fs=glob.glob(f"{a.root}/*/spectrum.json"); S=[]
    for f in fs:
        if os.path.basename(os.path.dirname(f)) in holdout: continue
        try:
            with open(f,encoding='utf-8') as fp: s2=json.load(fp)
        except Exception: continue
        if s2.get('n',0) < a.min_n: continue
        S.append(s2)
    print(f"[reference] read {len(S)} counties (buildings >= {a.min_n})")
    ref1={}; ref2={}
    for k,(lo,hi,nb,_) in BINS1.items():
        M=[np.array(s['d1'][k]) for s in S if s['d1'].get(k)]
        ref1[k]=np.mean(M,0) if M else None
    for xk,yk,xr,yr,bx,by in PLANES:
        key=f'{xk}__{yk}'; M=[np.array(s['d2'][key]) for s in S if s['d2'].get(key)]
        ref2[key]=np.mean(M,0) if M else None
    json.dump({'n_counties':len(S),'ref1':{k:(v.round(6).tolist() if v is not None else None) for k,v in ref1.items()},
               'ref2':{k:(v.round(6).tolist() if v is not None else None) for k,v in ref2.items()}},
              open(f"{od}/national_reference_spectrum.json",'w',encoding='utf-8'),ensure_ascii=False)
    # render reference fingerprint
    render({'name':'\u5168\u56fd\u53c2\u8003 NATIONAL REFERENCE','n':sum(s['n'] for s in S),'lat':float(np.mean([s['lat'] for s in S if s['lat']])),
            'd1':{k:(v.tolist() if v is not None else None) for k,v in ref1.items()},
            'd2':{k:(v.tolist() if v is not None else None) for k,v in ref2.items()}}, f"{od}/national_reference_spectrum.png")
    # per-county deviations
    import csv
    rows=[]
    for s in S:
        dev={}
        for k,(lo,hi,nb,_) in BINS1.items():
            if s['d1'].get(k) and ref1[k] is not None:
                c=centers(lo,hi,nb)
                dev[k]=round(float(wasserstein_distance(c,c,np.array(s['d1'][k]),ref1[k]))/(hi-lo),4)  # normalised by the bin range so features are comparable
        twod=[]
        for xk,yk,xr,yr,bx,by in PLANES:
            key=f'{xk}__{yk}'
            if s['d2'].get(key) and ref2[key] is not None:
                p=np.array(s['d2'][key]).ravel()+1e-9; q=ref2[key].ravel()+1e-9
                twod.append(float(jensenshannon(p,q)))
        row={'name':s['name'],'n':s['n'],'lat':round(s['lat'],2) if s['lat'] else None}
        row.update({f'emd_{k}':dev.get(k) for k in BINS1})
        row['emd_total']=round(float(np.nansum([dev.get(k,np.nan) for k in BINS1])),4)
        row['jsd_2d_mean']=round(float(np.mean(twod)),4) if twod else None
        rows.append(row)
    keys=['name','n','lat']+[f'emd_{k}' for k in BINS1]+['emd_total','jsd_2d_mean']
    with open(f"{od}/spectrum_deviation.csv",'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"[reference] -> national_reference_spectrum.json/.png + spectrum_deviation.csv ({len(rows)} counties)")
    rr=sorted([r for r in rows if r['emd_total'] is not None],key=lambda x:-x['emd_total'])
    print("largest deviation (least like the national mean):", [r['name'][:14] for r in rr[:5]])
    print("smallest deviation (most typical):", [r['name'][:14] for r in rr[-5:]])

def cmd_peakmap(a):
    import geopandas as gpd
    D=B.load_county(a.input); b=D['b']
    b=b[b.geometry.area>10].reset_index(drop=True); df=B.morph_feats(b)
    mask=np.ones(len(b),bool)
    sel={a.xfeat:(a.xmin,a.xmax)}
    if a.yfeat: sel[a.yfeat]=(a.ymin,a.ymax)
    for feat,(lo,hi) in sel.items():
        v=df[feat].values; mask&=(v>=lo)&(v<=hi)
    pk=b[mask]
    frac=len(pk)/len(b)
    # spatial concentration: spread of the selected building centres vs the whole county
    allc=b.geometry.representative_point(); pc=pk.geometry.representative_point()
    ext=np.hypot(allc.x.max()-allc.x.min(), allc.y.max()-allc.y.min())
    disp=np.hypot(pc.x.std(), pc.y.std())/ext if len(pk)>5 and ext>0 else None
    outd=a.out or a.input; os.makedirs(outd,exist_ok=True); out=f"{outd}/peak_{a.xfeat}.geojson"
    gpd.GeoDataFrame(geometry=pc, crs=b.crs).to_crs(4326).to_file(out, driver='GeoJSON')
    print(f"[peakmap] selected {len(pk)} buildings ({frac:.1%} of the county), dispersion ratio={disp:.3f} (smaller = more clustered) -> {out}")
    print("  reading: a very small dispersion ratio means the peak sits in one area - probably a local or developer-specific type rather than a county-wide vocabulary")

def main():
    ap=argparse.ArgumentParser(); s=ap.add_subparsers(dest='cmd',required=True)
    p=s.add_parser('one'); p.add_argument('--input',required=True); p.add_argument('--out',default=None); p.add_argument('--no-fig',action='store_true'); p.set_defaults(f=cmd_one)
    p=s.add_parser('all'); p.add_argument('--root',required=True); p.add_argument('--out',required=True); p.add_argument('--no-fig',action='store_true'); p.add_argument('--no-resume',action='store_true'); p.set_defaults(f=cmd_all)
    p=s.add_parser('reference'); p.add_argument('--root',required=True); p.add_argument('--min-n',type=int,default=1000); p.add_argument('--holdout',default=None); p.add_argument('--out-dir',default=None); p.set_defaults(f=cmd_reference)
    p=s.add_parser('peakmap'); p.add_argument('--input',required=True); p.add_argument('--xfeat',required=True); p.add_argument('--xmin',type=float,required=True); p.add_argument('--xmax',type=float,required=True)
    p.add_argument('--yfeat'); p.add_argument('--ymin',type=float); p.add_argument('--ymax',type=float); p.add_argument('--out',default=None); p.set_defaults(f=cmd_peakmap)
    a=ap.parse_args(); a.f(a)
if __name__=='__main__': main()
