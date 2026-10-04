# -*- coding: utf-8 -*-
"""EULUC-China 2.0 (2022, CC BY 4.0, Zenodo 10.5281/zenodo.16794007) -> county landuse.shp with the same schema as the 2018 release.
The 2.0 gdb has a single Class column (0-10), coded as the 2018 Class:
  0 residential | 1 business office, 2 commercial service -> commercial | 3 industrial | 4 transport station, 5 airport -> transport | 6 administration, 7 education, 8 health, 9 sport & culture, 10 park & green -> public administration & services
One nationwide scan (two layers of ~1.15 M parcels each, read sequentially in 200k chunks, representative point -> county, appended per county), about 10 min; no bbox filter (bbox reads of OpenFileGDB are slow).
Usage:
  python euluc2_ingest.py split   --out dataset/input_county_euluc2                 # nationwide -> {out}/{county}/landuse.shp (+ _progress markers)
  python euluc2_ingest.py compare --out dataset/input_county_euluc2 [--counties <list> --n 20]   # compare with 2018 per county: parcel count / area / co-located match / label agreement
  python euluc2_ingest.py link    --out dataset/input_county_euluc2 [--counties <list> --n 20]   # copy building.* roads.* _meta.json in, giving folders ready for analyze"""
import os, sys, glob, json, shutil, argparse, numpy as np, pandas as pd, geopandas as gpd, pyogrio
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
BND=f"{ROOT}/dataset/input/boundary/\u53bf.shp"; IN18=f"{ROOT}/dataset/input_county"
GDB=f"{ROOT}/dataset/input/EULUC_China_20_GDB/EULUC_China_20.gdb"; LAYERS=['EULUC_China_20_1','EULUC_China_20_2']; CH=50000
AEA='+proj=aea +lat_1=25 +lat_2=47 +lon_0=105 +datum=WGS84'
L1={0:(1,'\u5c45\u4f4f\u7528\u5730',101,'\u5c45\u4f4f\u7528\u5730'),1:(3,'\u5546\u670d\u7528\u5730',201,'\u5546\u52a1\u529e\u516c\u7528\u5730'),2:(3,'\u5546\u670d\u7528\u5730',202,'\u5546\u4e1a\u670d\u52a1\u7528\u5730'),3:(2,'\u5de5\u4e1a\u7528\u5730',301,'\u5de5\u4e1a\u7528\u5730'),
    4:(4,'\u9053\u8def\u4e0e\u4ea4\u901a\u8bbe\u65bd\u7528\u5730',401,'\u4ea4\u901a\u573a\u7ad9\u7528\u5730'),5:(4,'\u9053\u8def\u4e0e\u4ea4\u901a\u8bbe\u65bd\u7528\u5730',402,'\u673a\u573a\u8bbe\u65bd\u7528\u5730'),6:(5,'\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730',501,'\u884c\u653f\u529e\u516c\u7528\u5730'),
    7:(5,'\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730',502,'\u6559\u80b2\u79d1\u7814\u7528\u5730'),8:(5,'\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730',503,'\u533b\u7597\u536b\u751f\u7528\u5730'),
    9:(5,'\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730',504,'\u4f53\u80b2\u6587\u5316\u7528\u5730'),10:(5,'\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730',505,'\u516c\u56ed\u4e0e\u7eff\u5730\u7528\u5730')}

def counties():
    b=gpd.read_file(BND); b['folder']=b['\u7701'].astype(str)+'_'+b['\u5e02'].astype(str)+'_'+b['NAME'].astype(str)
    return b[b['\u7701'].notna()&b['NAME'].notna()].to_crs(4326)[['folder','geometry']]

def to_schema(g):
    c=g['Class'].fillna(-1).astype(int); ok=c.isin(L1); g=g[ok].copy(); c=c[ok]
    g['Level1']=c.map(lambda k:L1[k][0]); g['Level1_cn']=c.map(lambda k:L1[k][1]); g['Level2']=c.map(lambda k:L1[k][2]); g['Level2_cn']=c.map(lambda k:L1[k][3])
    g['AREA']=g.geometry.to_crs(AEA).area
    return g[['Class','Level1','Level1_cn','Level2','Level2_cn','AREA','geometry']]

def cmd_split(a):
    """Two stages, resumable: (1) chunks of 50k parcels -> county -> {out}/_chunks/{layer}_{sk}.parquet (atomic, < 300 MB memory);
    (2) read all chunks in order and append per county to landuse.shp. Runs on a machine with little free memory."""
    C=counties(); ch=f"{a.out}/_chunks"; os.makedirs(ch,exist_ok=True); mark=f"{a.out}/_progress/euluc2.done"; os.makedirs(f"{a.out}/_progress",exist_ok=True)
    if os.path.exists(mark) and not a.no_resume: print('already split; use --no-resume to redo'); return
    n=0; kept=0
    for lay in LAYERS:
        total=pyogrio.read_info(GDB,layer=lay)['features']; sk=0
        while sk<total:
            fp=f"{ch}/{lay}_{sk}.parquet"
            if not os.path.exists(fp):
                g=pyogrio.read_dataframe(GDB,layer=lay,skip_features=sk,max_features=CH)
                g=g[g.geometry.notna()&~g.geometry.is_empty]; g=gpd.GeoDataFrame(g,crs=4326)
                pts=gpd.GeoDataFrame(geometry=g.geometry.representative_point(),crs=4326)
                j=gpd.sjoin(pts,C,predicate='within',how='inner'); j=j[~j.index.duplicated()]; del pts
                g2=to_schema(g.loc[j.index]).assign(_folder=j['folder'].values); del g
                g2.to_parquet(fp+'.tmp'); os.replace(fp+'.tmp',fp); kept+=len(g2); del g2
            n+=min(CH,total-sk); sk+=CH; print(f"  [{lay}] {min(sk,total):,}/{total:,}",flush=True)
    # stage 2: append per county (old files cleared first)
    for fol in C.folder:
        for f in glob.glob(f"{a.out}/{fol}/landuse.*"): os.remove(f)
    counts={}
    for fp in sorted(glob.glob(f"{ch}/*.parquet")):
        g2=gpd.read_parquet(fp)
        for fol,sub in g2.groupby('_folder'):
            od=f"{a.out}/{fol}"; os.makedirs(od,exist_ok=True); p=f"{od}/landuse.shp"
            pyogrio.write_dataframe(sub.drop(columns='_folder'),p,append=os.path.exists(p),encoding='utf-8'); counts[fol]=counts.get(fol,0)+len(sub)
        del g2; print(f"  [merge] {os.path.basename(fp)}",flush=True)
    json.dump({'scanned':n,'assigned':sum(counts.values()),'counties':len(counts)},open(mark,'w'))
    print(f"[euluc2 split] {n:,} parcels scanned, {sum(counts.values()):,} assigned to {len(counts)} counties -> {a.out}")

def _names(a):
    fols=sorted(os.path.basename(os.path.normpath(d)) for d in glob.glob(f"{a.out}/*/") if os.path.exists(f"{d}/landuse.shp"))
    if a.counties:
        want=[l.strip() for l in open(a.counties,encoding='utf-8') if l.strip() and not l.startswith('#')]
        if a.n: want=want[:a.n]
        fols=[f for f in want if f in set(fols)]
    return fols

def cmd_compare(a):
    rows=[]
    for fol in _names(a):
        new=pyogrio.read_dataframe(f"{a.out}/{fol}/landuse.shp"); p=f"{IN18}/{fol}/landuse.shp"
        if not os.path.exists(p): rows.append({'folder':fol,'n_2018':0,'n_2022':len(new)}); continue
        old=pyogrio.read_dataframe(p,columns=['Level1_cn']); old=old[old.geometry.notna()&~old.geometry.is_empty].to_crs(4326)
        pts=gpd.GeoDataFrame({'l1_new':new['Level1_cn'].values},geometry=new.geometry.representative_point(),crs=4326)
        j=gpd.sjoin(pts,old[['Level1_cn','geometry']].rename(columns={'Level1_cn':'l1_old'}),predicate='within',how='left'); j=j[~j.index.duplicated()]
        m=j.l1_old.notna(); agree=float((j.l1_new[m]==j.l1_old[m]).mean()) if m.any() else np.nan
        # same geometry? median ratio of 2.0 parcel area to the area of the 2018 parcel it falls in (~1 = same parcel)
        oa=old.geometry.to_crs(AEA).area; ratio=(new['AREA'].values[m.values]/oa.values[j.index_right[m].astype(int)]) if m.any() else np.array([np.nan])
        rows.append({'folder':fol,'n_2018':len(old),'n_2022':len(new),'area_km2_2018':round(oa.sum()/1e6,1),'area_km2_2022':round(new.AREA.sum()/1e6,1),
                     'share_new_inside_old':round(float(m.mean()),3),'median_area_ratio':round(float(np.nanmedian(ratio)),3),'label_agreement':round(agree,3) if agree==agree else None})
        print(f"  {fol}: 2018 {len(old)} / 2022 {len(new)} parcels, inside-old {m.mean():.2f}, area ratio {np.nanmedian(ratio):.2f}, labels agree {agree:.2f}",flush=True)
    df=pd.DataFrame(rows); df.to_csv(f"{a.out}/_compare_2018.csv",index=False,encoding='utf-8-sig')
    d=df.dropna(subset=['label_agreement']) if 'label_agreement' in df else df
    if len(d): print(f"\nmedian: parcel ratio {(d.n_2022/d.n_2018).median():.3f}, inside-old {d.share_new_inside_old.median():.3f}, area ratio {d.median_area_ratio.median():.3f}, label agreement {d.label_agreement.median():.3f}")

def cmd_link(a):
    for fol in _names(a):
        for f in glob.glob(f"{IN18}/{fol}/building.*")+glob.glob(f"{IN18}/{fol}/roads.*")+glob.glob(f"{IN18}/{fol}/_meta.json"): shutil.copy(f,f"{a.out}/{fol}/")
    print('[euluc2 link] done')

def main():
    ap=argparse.ArgumentParser(); s=ap.add_subparsers(dest='cmd',required=True)
    for name,fn in [('split',cmd_split),('compare',cmd_compare),('link',cmd_link)]:
        p=s.add_parser(name); p.add_argument('--out',default=f"{ROOT}/dataset/input_county_euluc2"); p.add_argument('--counties',default=None); p.add_argument('--n',type=int,default=0)
        p.add_argument('--no-resume',action='store_true'); p.set_defaults(f=fn)
    a=ap.parse_args(); a.f(a)
if __name__=='__main__': main()
