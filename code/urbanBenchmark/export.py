# -*- coding: utf-8 -*-
"""Export output_county/*/result.json (+ spectrum.json, _meta) to the county wide table county_metrics.csv.
Usage: python export.py --root dataset/output_county [--input-root dataset/input_county] [--min-n 0] [--en]
      --spectra -> county_spectra.nc; --relations -> county_relations.csv (L3/L4 long table); --manifest -> manifest.json
      --results -> benchmark_results.csv + benchmark_violations.csv (scores, layer conformity and failed checks of the controlled test and generator cases)
Columns: folder, PAC, province, prefecture, county, type, lat, n_buildings, scalar metrics (dotted paths flattened), assort_*, L3_{a}_{b}, FAR_/coverage_ per use,
    L6_entropy_{r}, L6_slope, use_height_ratio_{use}. Missing = the block failed or had no data (see _errors in result.json)."""
import os, sys, glob, json, argparse, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RADII=[500,1000,2000,4000,8000]
SHORT={'\u5c45\u4f4f\u7528\u5730':'\u5c45\u4f4f','\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730':'\u516c\u7ba1','\u5de5\u4e1a\u7528\u5730':'\u5de5\u4e1a','\u5546\u670d\u7528\u5730':'\u5546\u670d','\u9053\u8def\u4e0e\u4ea4\u901a\u8bbe\u65bd\u7528\u5730':'\u4ea4\u901a'}
# release column names (--en): short use labels -> GB 50137-2011 land-use codes; province/prefecture/county headers in English (values stay Chinese; join on PAC)
GB={'\u5c45\u4f4f':'R','\u516c\u7ba1':'A','\u5546\u670d':'B','\u5de5\u4e1a':'M','\u4ea4\u901a':'S'}
def en_columns(cols):
    import re
    pat=re.compile('(?<=[._])('+'|'.join(GB)+')(?=_|$)')
    return [pat.sub(lambda m:GB[m.group(1)],{'\u7701':'province','\u5e02':'prefecture','\u53bf':'county'}.get(c,c)) for c in cols]
SCALARS=[('area_dist','logn_mu'),('area_dist','logn_sigma'),('area_dist','hill_alpha'),('orient','cardinal10'),('orient','entropy'),
         ('road','orient_entropy'),('road','grid_share'),('road','length_km'),('blocks','median_area'),('blocks','superblock_share'),('blocks','n'),
         ('setback','median'),('setback','frontage15')]
TOP=['form_diversity','height_moran','fractal_D','sun_ratio_median','assort_edges']

def row_of(name, r, meta):
    parts=name.split('_'); o={'folder':name,'PAC':r.get('PAC') or meta.get('PAC'),
        '\u7701':parts[0] if len(parts)>0 else None,'\u5e02':parts[1] if len(parts)>1 else None,'\u53bf':parts[2] if len(parts)>2 else None,
        'type':r.get('type') or meta.get('\u7c7b\u578b'),'lat':r.get('lat'),'epsg':r.get('epsg'),'n_buildings':r.get('n_buildings') or r.get('_n')}
    for k in TOP: o[k]=r.get(k)
    for a,b in SCALARS:
        v=r.get(a); o[f'{a}.{b}']=v.get(b) if isinstance(v,dict) else None
    for k,v in (r.get('assort') or {}).items(): o[f'assort.{k}']=v
    for x in r.get('L3',[]): o[f"L3.{SHORT.get(x['a'],x['a'])}_{SHORT.get(x['b'],x['b'])}"]=x['lift']
    for u,m in (r.get('use_morph') or {}).items():
        s=SHORT.get(u,u); o[f'use_height_ratio.{s}']=m.get('height_ratio'); o[f'use_height_med.{s}']=m.get('height_med')
    for u,m in (r.get('intensity') or {}).items():
        s=SHORT.get(u,u); o[f'FAR.{s}']=m.get('FAR'); o[f'coverage.{s}']=m.get('coverage')
    e=(r.get('L6_entropy') or {}).get('mean_entropy')
    if e and all(x is not None for x in e):
        for rad,v in zip(RADII,e): o[f'L6.entropy_{rad}']=v
        o['L6.slope']=round(float(np.polyfit(np.log(RADII),e,1)[0]),4)
    o['n_errors']=len(r.get('_errors') or {})
    return o

def spectra_nc(root, out, min_n=0):
    """Merge {root}/*/spectrum.json into one NetCDF (county_spectra.nc, NetCDF3 via scipy, no netCDF4 needed):
    d1_{feat}(county, {feat}_bin) 1-D histograms (sum to 1), d2_{x}__{y}(county, {y}_bin30, {x}_bin30) 2-D histograms (stored transposed as in spectrum.json),
    coordinates are bin centres; n / lat are county attributes."""
    import xarray as xr, spectrum as SP
    names=[]; N=[]; LAT=[]; D1={k:[] for k in SP.BINS1}; D2={f'{x}__{y}':[] for x,y,*_ in SP.PLANES}
    for f in sorted(glob.glob(f"{root}/*/spectrum.json")):
        try: sj=json.load(open(f,encoding='utf-8'))
        except Exception: continue
        if (sj.get('n') or 0)<min_n: continue
        names.append(os.path.basename(os.path.dirname(f))); N.append(sj.get('n')); LAT.append(sj.get('lat'))
        for k,(lo,hi,nb,_) in SP.BINS1.items(): v=sj['d1'].get(k); D1[k].append(v if v else [np.nan]*nb)
        for x,y,xr_,yr_,bx,by in SP.PLANES: v=sj['d2'].get(f'{x}__{y}'); D2[f'{x}__{y}'].append(v if v else [[np.nan]*bx]*by)
    coords={'county':names}; data={'n':('county',np.array(N,dtype='int32')),'lat':('county',np.array(LAT,dtype='float32'))}
    for k,(lo,hi,nb,lab) in SP.BINS1.items():
        coords[f'{k}_bin']=(f'{k}_bin',SP.centers(lo,hi,nb).astype('float32'),{'long_name':lab,'edges':f'linspace({lo},{hi},{nb+1})'})
        data[f'd1_{k}']=(('county',f'{k}_bin'),np.array(D1[k],dtype='float32'),{'long_name':f'1-D histogram of {lab}, sums to 1'})
    for x,y,xr_,yr_,bx,by in SP.PLANES:   # 2-D planes use 30 bins, a different dimension from the 50-bin 1-D spectra
        for k,(lo,hi),nb in ((x,xr_,bx),(y,yr_,by)):
            coords[f'{k}_bin{nb}']=(f'{k}_bin{nb}',SP.centers(lo,hi,nb).astype('float32'),{'long_name':SP.BINS1[k][3],'edges':f'linspace({lo},{hi},{nb+1})'})
        data[f'd2_{x}__{y}']=(('county',f'{y}_bin{by}',f'{x}_bin{bx}'),np.array(D2[f'{x}__{y}'],dtype='float32'),{'long_name':f'2-D histogram {x} × {y}, sums to 1'})
    ds=xr.Dataset(data,coords=coords,attrs={'title':'CUFB-China county morphology spectra','source':'spectrum.py spec_from_df, ≤50k buildings per county','license':'CC BY 4.0'})
    ds.to_netcdf(out,engine='scipy'); print(f"[spectra] {len(names)} counties -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")

def relations_long(root, out, min_n=0, input_root=None):
    """Long table of L3 (parcel adjacency lift) and L4 (use x highest-ranked enclosing street class lift): county, PAC, layer, a, b, lift. L4 is released only here, not in the wide table."""
    rows=[]
    for f in sorted(glob.glob(f"{root}/*/result.json")):
        try: r=json.load(open(f,encoding='utf-8'))
        except Exception: continue
        if (r.get('n_buildings') or 0)<min_n: continue
        c=os.path.basename(os.path.dirname(f)); pac=r.get('PAC')
        if pac is None and input_root and os.path.exists(f"{input_root}/{c}/_meta.json"):
            try: pac=json.load(open(f"{input_root}/{c}/_meta.json",encoding='utf-8')).get('PAC')
            except Exception: pass
        for x in r.get('L3',[]): rows.append({'county':c,'PAC':pac,'layer':'L3','a':SHORT.get(x['a'],x['a']),'b':SHORT.get(x['b'],x['b']),'lift':x['lift']})
        for x in r.get('L4',[]): rows.append({'county':c,'PAC':pac,'layer':'L4','a':SHORT.get(x['use'],x['use']),'b':x['top_road'],'lift':x['lift']})
    df=pd.DataFrame(rows); df.to_csv(out,index=False,encoding='utf-8-sig')
    print(f"[relations] {df.county.nunique()} counties {len(df)} rows (L3 {int((df.layer=='L3').sum())}, L4 {int((df.layer=='L4').sum())}) -> {out}")

# ---------- benchmark results table: 50 real + 350 perturbed test cases (against the test-excluded reference) + generator cases (against the national reference) ----------
GEN_GROUP=[('josauder','L-system procedural (josauder)'),('mapgen','tensor-field procedural (MapGenerator)'),
           ('citydreamer','learned layout (CityDreamer)'),('drl','RL community planner (DRL-urban-planning)'),('real','real EULUC crop (control)')]
L3_KEYS={'L3_R_R':('\u5c45\u4f4f\u7528\u5730','\u5c45\u4f4f\u7528\u5730'),'L3_A_A':('\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730','\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730'),
         'L3_A_R':('\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730','\u5c45\u4f4f\u7528\u5730'),'L3_R_M':('\u5c45\u4f4f\u7528\u5730','\u5de5\u4e1a\u7528\u5730'),'L3_M_M':('\u5de5\u4e1a\u7528\u5730','\u5de5\u4e1a\u7528\u5730')}
def _l3(res):
    o={k:None for k in L3_KEYS}
    for x in res.get('L3',[]):
        for k,(a,b) in L3_KEYS.items():
            if {x.get('a'),x.get('b')}=={a,b}: o[k]=x.get('lift')
    return o
def _read(p):
    try: return json.load(open(p,encoding='utf-8'))
    except Exception: return {}
def results_table(gate_ref, holdout_out, perturbed_out, gen_root, out_dir):
    """benchmark_results.csv: one row per scored folder (group / case / variant / reference / score / layer conformities / five key L3 lifts);
    benchmark_violations.csv: one row per failed check of generator and real-crop cases (layer / metric / value / bound or direction / kind)."""
    rows=[]; viol=[]
    sc=pd.read_csv(f"{gate_ref}/scores_v2.csv",encoding='utf-8-sig')
    for _,r in sc.iterrows():
        real=r['group']=='real'; folder=f"{holdout_out if real else perturbed_out}/{r['city']}"
        res=_read(f"{folder}/result.json")
        o={'group':'real_holdout' if real else 'perturbed','family':'real county (holdout)' if real else 'perturbed county','case':r['city'].split('__')[0],'variant':None if real else r['perturbation'],
           'source':'CMAB / EULUC / OSM county' if real else f"perturb.py {r['perturbation']}",'reference':'ref_holdout (2,349 counties)','protocol':'v2',
           'score':r['score'],'checked':r['checked'],'violations':r['violations'],'spectrum_percentile':r['spectrum_percentile'],
           'n_buildings':res.get('n_buildings')}
        o.update({c:r[c] for c in sc.columns if c.startswith('conf_')}); o.update(_l3(res)); rows.append(o)
    for d in sorted(glob.glob(f"{gen_root}/*/")):
        name=os.path.basename(os.path.normpath(d)); s=_read(f"{d}/score.json"); res=_read(f"{d}/result.json"); meta=_read(f"{d}/_meta.json")
        if not s: continue
        fam=next((g for k,g in GEN_GROUP if name.startswith(k)),name)
        var=(f"{meta['scale_m_per_unit']:g} m/unit" if meta.get('scale_m_per_unit') else f"{meta['m_per_px']:g} m/px" if meta.get('m_per_px') else meta.get('label'))
        if meta.get('seed') is not None and 'seed' not in str(var): var=f"{var}, seed {meta['seed']}"
        if name.endswith('_randlu'): var=f"{var}, + random land use (control)"
        o={'group':'real_crop' if meta.get('\u7c7b\u578b')=='real_crop' else 'generated','case':name,'variant':var,'source':meta.get('source'),
           'reference':'national (2,399 counties)','protocol':s.get('protocol','v2'),'score':s.get('score'),'checked':s.get('checked'),
           'violations':len(s.get('violations',[])),'spectrum_percentile':s.get('spectrum_percentile'),'n_buildings':res.get('n_buildings')}
        for k,v in (s.get('layers') or {}).items(): o[f'conf_{k}']=v.get('conformity')
        o.update(_l3(res)); o['family']=fam; rows.append(o)
        for v in s.get('violations',[]):
            viol.append({'case':name,'layer':v.get('layer'),'metric':v.get('metric'),'key':v.get('key'),'value':v.get('value'),
                         'p5':v.get('p5'),'p95':v.get('p95'),'national_p50':v.get('national_p50'),'need':v.get('need'),'kind':v.get('kind')})
    df=pd.DataFrame(rows); conf=sorted(c for c in df.columns if c.startswith('conf_'))
    cols=['group','family','case','variant','source','reference','protocol','n_buildings','score','checked','violations','spectrum_percentile']+conf+list(L3_KEYS)
    df=df.reindex(columns=cols); df.to_csv(f"{out_dir}/benchmark_results.csv",index=False,encoding='utf-8-sig')
    vf=pd.DataFrame(viol)
    for zh,en in SHORT.items(): vf['key']=vf['key'].astype(str).str.replace(zh,GB[en],regex=False)   # use-pair keys -> R/A/B/M/S
    vf.to_csv(f"{out_dir}/benchmark_violations.csv",index=False,encoding='utf-8-sig')
    print(f"[results] {df.group.value_counts().to_dict()} -> benchmark_results.csv; {len(vf)} failed checks -> benchmark_violations.csv")

RECORDS=['county_metrics.csv','county_relations.csv','national_bounds.csv','national_bounds_by_type.csv','national_bounds_by_size.csv','national_bounds_by_era.csv',
         'county_spectra.nc','national_reference_spectrum.json','national_reference_spectrum.png','spectrum_deviation.csv','community_vectors.csv',
         'national_archetypes.json','archetype_by_province.csv','data_dictionary.csv','benchmark_results.csv','benchmark_violations.csv']
def manifest(root, version):
    """Data Records manifest: file, bytes, sha256, rows (csv) / entries (json); missing files are marked. data_dictionary.csv is copied from the code folder."""
    import hashlib, shutil
    here=os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(f"{here}/data_dictionary.csv"): shutil.copy(f"{here}/data_dictionary.csv",f"{root}/data_dictionary.csv")
    files=[]
    for nm in RECORDS:
        p=f"{root}/{nm}"
        if not os.path.exists(p): files.append({'file':nm,'status':'missing'}); continue
        h=hashlib.sha256(open(p,'rb').read()).hexdigest(); e={'file':nm,'bytes':os.path.getsize(p),'sha256':h}
        if nm.endswith('.csv'): e['rows']=sum(1 for _ in open(p,encoding='utf-8-sig'))-1
        elif nm.endswith('.json'):
            try:
                j=json.load(open(p,encoding='utf-8')); e['entries']=len(j) if isinstance(j,(list,dict)) else None
            except Exception: pass
        files.append(e)
    m={'dataset':'CUFB-China','version':version,'generated':pd.Timestamp.now().strftime('%Y-%m-%d'),'license':'CC BY 4.0',
       'upstream':{'buildings':'CMAB v7 (Zhang et al. 2025, Sci Data 12:430; Figshare 10.6084/m9.figshare.27992417), CC BY 4.0',
                   'landuse':'EULUC-China 2018 (Gong et al. 2020, Sci Bull 65:182)','roads':'OpenStreetMap via Geofabrik, snapshot 2024-05-29, ODbL','boundary':'county boundaries 2020-02'},
       'files':files}
    json.dump(m,open(f"{root}/manifest.json",'w',encoding='utf-8'),ensure_ascii=False,indent=1)
    print(f"[manifest] {sum(1 for f in files if 'sha256' in f)}/{len(files)} files -> {root}/manifest.json")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--input-root',default=None)
    ap.add_argument('--min-n',type=int,default=0); ap.add_argument('--out',default=None)
    ap.add_argument('--manifest',action='store_true',help='only write manifest.json (sha256 / row counts); do not rebuild the wide table'); ap.add_argument('--version',default='1.0-dev')
    ap.add_argument('--en',action='store_true',help='release column names: uses -> GB 50137 codes R/A/B/M/S, province/prefecture/county headers in English')
    ap.add_argument('--spectra',action='store_true',help='only merge */spectrum.json into county_spectra.nc')
    ap.add_argument('--relations',action='store_true',help='only export the L3/L4 long table county_relations.csv')
    ap.add_argument('--results',action='store_true',help='only export benchmark_results.csv / benchmark_violations.csv (controlled test + generator cases)')
    ap.add_argument('--gate-ref',default='dataset/ref_holdout'); ap.add_argument('--gen-root',default='dataset/generated')
    a=ap.parse_args()
    if a.manifest: manifest(a.root,a.version); return
    if a.spectra: spectra_nc(a.root,a.out or f"{a.root}/county_spectra.nc",a.min_n); return
    if a.relations: relations_long(a.root,a.out or f"{a.root}/county_relations.csv",a.min_n,a.input_root); return
    if a.results:
        ds=os.path.dirname(os.path.normpath(a.gate_ref)); results_table(a.gate_ref,f"{ds}/holdout_out",f"{ds}/perturbed_out",a.gen_root,a.out or a.root); return
    rows=[]
    for f in sorted(glob.glob(f"{a.root}/*/result.json")):
        d=os.path.dirname(f); name=os.path.basename(d)
        try: r=json.load(open(f,encoding='utf-8'))
        except Exception: continue
        sp=f"{d}/spectrum.json"
        if r.get('n_buildings') is None and os.path.exists(sp):
            try: r['_n']=json.load(open(sp,encoding='utf-8')).get('n')
            except Exception: pass
        meta={}
        mp=f"{a.input_root}/{name}/_meta.json" if a.input_root else None
        if mp and os.path.exists(mp):
            try: meta=json.load(open(mp,encoding='utf-8'))
            except Exception: pass
        o=row_of(name,r,meta)
        if o['n_buildings'] is not None and o['n_buildings']<a.min_n: continue
        rows.append(o)
    df=pd.DataFrame(rows); out=a.out or f"{a.root}/county_metrics.csv"
    if a.en: df.columns=en_columns(df.columns)
    df.to_csv(out,index=False,encoding='utf-8-sig')
    print(f"[export] {len(df)} counties x {df.shape[1]} columns -> {out}")
if __name__=='__main__': main()
