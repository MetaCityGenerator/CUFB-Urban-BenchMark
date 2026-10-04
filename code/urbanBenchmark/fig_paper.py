# -*- coding: utf-8 -*-
"""Manuscript figures Fig 1–7 for the CUFB-China data descriptor (English labels only).
  fig1  pipeline schematic + county coverage map (buildings per county)
  fig2  CV ladder: cross-county coefficient of variation of the scalar metrics, tier thresholds 10% / 40%
  fig3  invariants across strata (all / era / county type / size): p5–p95 bars vs the overall band
  fig4  relations: L3 adjacency-lift matrix with sign consistency, building assortativity, relative height by use
  fig5  multi-scale mixing entropy: median curve with bands, slope distribution, strata overlay
  fig6  national reference spectra with cross-county bands + map of spectrum deviation
  fig7  technical validation: momepy/osmnx agreement, parameter sensitivity, subsample CI, height-noise response
  --gis  no figures; writes the map layers behind Fig 1b / Fig 6b (and province-level summaries) as GeoParquet into
         {out}/gis/ so the maps can be drawn in QGIS with the full national base map (Taiwan, nine-dash line)
Usage: python fig_paper.py [--only fig1,fig3] [--out dataset/figures]      (reads dataset/output_county, dataset/validation and the county boundary layer in dataset/input/boundary/)"""
import os, json, argparse, numpy as np, pandas as pd, matplotlib
matplotlib.use('agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
matplotlib.rcParams.update({'font.family':['DejaVu Sans','Arial'],'font.size':10,'axes.titleweight':'bold'})
OUT='dataset/output_county'; VAL='dataset/validation'; BND='dataset/input/boundary/\u53bf.shp'
USE={'\u5c45\u4f4f\u7528\u5730':'R','\u5546\u670d\u7528\u5730':'B','\u5de5\u4e1a\u7528\u5730':'M','\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730':'A','\u9053\u8def\u4e0e\u4ea4\u901a\u8bbe\u65bd\u7528\u5730':'S'}
USE_LONG={'R':'Residential','B':'Commercial','M':'Industrial','A':'Public services','S':'Transport'}
NAMES={'form_diversity':'Form diversity (area × height entropy)','area_dist.logn_mu':'Footprint area, log-mean','area_dist.logn_sigma':'Footprint area, log-SD',
       'area_dist.hill_alpha':'Footprint area, tail exponent α','fractal_D':'Fractal dimension D','road.orient_entropy':'Street orientation entropy',
       'orient.cardinal10':'Cardinal buildings (±10°)','orient.entropy':'Building orientation entropy','sun_ratio_median':'Spacing-to-height ratio',
       'height_moran':"Height Moran's I",'setback.median':'Building–street distance, median (m)','setback.frontage15':'Frontage share (<15 m)','road.grid_share':'Grid share of streets',
       'blocks.median_area':'Block area, median (m²)','blocks.superblock_share':'Superblock share (>10 ha)','L6.slope_dH_dlnr':'Mixing-entropy slope dH/d ln r',
       'intensity.\u5c45\u4f4f\u7528\u5730.FAR':'Est. FAR, residential','intensity.\u5546\u670d\u7528\u5730.FAR':'Est. FAR, commercial','intensity.\u5de5\u4e1a\u7528\u5730.FAR':'Est. FAR, industrial','intensity.\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730.FAR':'Est. FAR, public services',
       'intensity.\u5c45\u4f4f\u7528\u5730.coverage':'Coverage, residential','intensity.\u5546\u670d\u7528\u5730.coverage':'Coverage, commercial','intensity.\u5de5\u4e1a\u7528\u5730.coverage':'Coverage, industrial',
       'assort.log_h':'Height assortativity','assort.log_area':'Area assortativity','assort.log_elong':'Elongation assortativity','assort.compact':'Compactness assortativity',
       'use_height_ratio.\u5c45\u4f4f\u7528\u5730':'Relative height, residential','use_height_ratio.\u5546\u670d\u7528\u5730':'Relative height, commercial','use_height_ratio.\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730':'Relative height, public services',
       'use_height_ratio.\u5de5\u4e1a\u7528\u5730':'Relative height, industrial','use_height_ratio.\u9053\u8def\u4e0e\u4ea4\u901a\u8bbe\u65bd\u7528\u5730':'Relative height, transport'}
TIER_COL={'invariant':'#1c5cab','weak':'#7a9cc6','signature':'#c4501f'}
def tier(cv): return 'invariant' if cv<10 else ('weak' if cv<=40 else 'signature')
def bounds():
    A=pd.read_csv(f'{OUT}/national_bounds.csv',encoding='utf-8-sig'); A['k']=A.metric+'.'+A.key.astype(str); return A
def strata(name):
    S=pd.read_csv(f'{OUT}/national_bounds_by_{name}.csv',encoding='utf-8-sig'); S['k']=S.metric+'.'+S.key.astype(str); return S
def china():
    import geopandas as gpd
    b=gpd.read_file(BND)[['PAC','geometry']]; b['geometry']=b.geometry.simplify(0.01); return b.to_crs('+proj=aea +lat_1=25 +lat_2=47 +lon_0=105 +lat_0=0 +datum=WGS84')
def save(fig,path): fig.savefig(path,dpi=130,bbox_inches='tight'); plt.close(fig); print('->',path)

# ---------------- Fig 1 ----------------
def fig1(out):
    fig=plt.figure(figsize=(18,7.4)); gs=fig.add_gridspec(1,2,width_ratios=[1.25,1.15],wspace=0.02)
    ax=fig.add_subplot(gs[0]); ax.set_xlim(0,12); ax.set_ylim(0,10); ax.set_axis_off(); ax.set_title('(a) Pipeline',loc='left')
    def boxf(x,y,w,h,lines,fc='#eef2f7',ec='#1c5cab',fs=7.8):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.15',fc=fc,ec=ec,lw=1.2))
        ax.text(x+w/2,y+h/2,chr(10).join(lines),ha='center',va='center',fontsize=fs,linespacing=1.4)
    def arrow(x0,y0,x1,y1): ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle='-|>',mutation_scale=12,color='#555',lw=1.1))
    boxf(0.2,8.3,3.7,1.5,['CMAB v7 buildings','23.5 M footprints + height + age','Zhang et al. 2025 · CC BY 4.0'],fc='#fbe9e3',ec='#c4501f')
    boxf(4.15,8.3,3.7,1.5,['EULUC-China 2.0 parcels (2022)','11 classes → 5 Level-1 → R / A / B / M / S','Li et al. 2025 · CC BY 4.0'],fc='#fbe9e3',ec='#c4501f')
    boxf(8.1,8.3,3.7,1.5,['OpenStreetMap roads','Geofabrik 2024-05-29 · ODbL','+ county boundaries 2020'],fc='#fbe9e3',ec='#c4501f')
    for x in (2.05,6.0,9.95): arrow(x,8.3,6.0-(6.0-x)*0.2,7.1)
    boxf(1.5,5.75,9.0,1.35,['split_county.py — 2,904 county units → 2,585 with buildings','per-county UTM zone; buildings, parcels and roads clipped to each county'])
    arrow(6.0,5.75,6.0,4.95)
    boxf(0.2,3.2,11.6,1.75,['benchmark.py analyze — 17 metric blocks per county','morphology (area spectrum, orientation, fractal D, height Moran, solar spacing) · L1 assortativity · L2 use → form','L3 block adjacency · L4 road enclosure · L5 communities · L6 multi-scale mixing · streets, blocks, setback, intensity · spectra'])
    for x in (2.05,6.0,9.95): arrow(x,3.2,x,2.45)
    boxf(0.2,0.8,3.7,1.65,['Data Records','county_metrics.csv · county_relations.csv','county_spectra.nc · archetypes','data dictionary · manifest'],fc='#e6f1ea',ec='#0a7a3c')
    boxf(4.15,0.8,3.7,1.65,['aggregate — 2,399 counties (n ≥ 1,000)','p5 / p50 / p95 · CV tiers · sign consistency','strata (era, type, size) · bootstrap CI','national bounds · reference spectrum'],fc='#e6f1ea',ec='#0a7a3c')
    boxf(8.1,0.8,3.7,1.65,['score.py — conformity score of any','real or generated city, weighted by tier','perturb.py — 7 structural corruptions','validate.py — cross-checks, sensitivity'],fc='#e6f1ea',ec='#0a7a3c')
    import geopandas as gpd
    ax2=fig.add_subplot(gs[1]); b=china(); cm=pd.read_csv(f'{OUT}/county_metrics.csv',encoding='utf-8-sig')[['PAC','n_buildings']]
    g=b.merge(cm,on='PAC',how='left'); g['v']=np.log10(g.n_buildings.where(g.n_buildings>0))
    g[g.v.isna()].plot(ax=ax2,color='#e9e9e9',edgecolor='white',linewidth=0.15)
    g[g.v.notna()].plot(ax=ax2,column='v',cmap='YlGnBu',vmin=2,vmax=5,edgecolor='white',linewidth=0.15,legend=False)
    sm=plt.cm.ScalarMappable(cmap='YlGnBu',norm=plt.Normalize(2,5)); cb=fig.colorbar(sm,ax=ax2,shrink=0.55,pad=0.01,ticks=[2,3,4,5]); cb.ax.set_yticklabels(['100','1,000','10,000','100,000']); cb.set_label('Buildings per county (CMAB v7)')
    ax2.set_axis_off(); n1=int((cm.n_buildings>=1000).sum()); ax2.set_title(f'(b) Coverage: 2,585 counties with buildings, {n1:,} with ≥ 1,000 (grey: none)',loc='left')
    save(fig,f'{out}/fig1_pipeline_coverage.png')

# ---------------- Fig 2 ----------------
def fig2(out):
    A=bounds(); keys=['form_diversity','area_dist.logn_mu','fractal_D','area_dist.logn_sigma','road.orient_entropy','area_dist.hill_alpha','orient.entropy','sun_ratio_median',
        'orient.cardinal10','L6.slope_dH_dlnr','height_moran','setback.median','intensity.\u5c45\u4f4f\u7528\u5730.coverage','intensity.\u5c45\u4f4f\u7528\u5730.FAR','blocks.superblock_share','road.grid_share',
        'intensity.\u5546\u670d\u7528\u5730.FAR','intensity.\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730.FAR','intensity.\u5de5\u4e1a\u7528\u5730.coverage','intensity.\u5de5\u4e1a\u7528\u5730.FAR','setback.frontage15','blocks.median_area']
    S=A[A.metric=='scalar'].set_index('key'); U=A[A.metric=='use_height_ratio'].set_index('key')
    rows=[(NAMES[k],S.loc[k,'cv_pct'],S.loc[k,'p5'],S.loc[k,'p50'],S.loc[k,'p95']) for k in keys]
    for u in ['\u5c45\u4f4f\u7528\u5730','\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730','\u5546\u670d\u7528\u5730','\u5de5\u4e1a\u7528\u5730']: rows.append((NAMES['use_height_ratio.'+u],U.loc[u,'cv_pct'],U.loc[u,'p5'],U.loc[u,'p50'],U.loc[u,'p95']))
    rows.sort(key=lambda r:r[1]); fig,ax=plt.subplots(figsize=(10,9.2))
    y=np.arange(len(rows)); cv=[r[1] for r in rows]; ax.barh(y,cv,color=[TIER_COL[tier(c)] for c in cv],height=0.7)
    for i,r in enumerate(rows): ax.text(r[1]+1.2,i,f'{r[1]:.1f}%   p50 {r[3]:.3g}  [{r[2]:.3g}, {r[4]:.3g}]',va='center',fontsize=8.3,color='#333')
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows],fontsize=9); ax.set_xlim(0,max(cv)*1.55)
    for x,t in ((10,'invariant | weak'),(40,'weak | signature')): ax.axvline(x,color='k',ls='--',lw=0.9); ax.text(x,-1.4,t,ha='center',fontsize=8.5)
    ax.set_ylim(len(rows)-0.4,-2.2)
    ax.set_xlabel('Coefficient of variation across 2,399 counties (%)'); ax.set_title('Cross-county variability sorts the metrics into three tiers of constraint',loc='left')
    from matplotlib.patches import Patch; ax.legend(handles=[Patch(color=TIER_COL[t],label={'invariant':'Invariant (CV < 10%), weight 3','weak':'Weak (10–40%), weight 1.5','signature':'Signature (> 40%), weight 0.5'}[t]) for t in TIER_COL],loc='center right',frameon=False)
    ax.grid(axis='x',alpha=0.25); save(fig,f'{out}/fig2_cv_ladder.png')

# ---------------- Fig 3 ----------------
def fig3(out):
    A=bounds().set_index('k'); E=strata('era'); T=strata('type'); Z=strata('size')
    mets=['scalar.form_diversity','scalar.area_dist.logn_mu','scalar.area_dist.logn_sigma','scalar.area_dist.hill_alpha','scalar.road.orient_entropy','scalar.L6.slope_dH_dlnr','scalar.fractal_D','assort.log_h']
    groups=[('All counties',A,None,'#222')]+[(f'Era {l}',E,s,'#c4501f') for s,l in [('pre1990','≤1990'),('1991_2000','1991–2000'),('2001_2010','2001–2010'),('post2010','>2010')]] \
           +[(f'Type: {l}',T,s,'#1c5cab') for s,l in [('\u5e02\u8f96\u533a','urban district'),('\u53bf\u7ea7\u5e02','county-level city'),('\u53bf','county')]]+[(f'Size: {s}',Z,s,'#0a7a3c') for s in ['small','medium','large']]
    fig,axes=plt.subplots(2,4,figsize=(17,8.4)); axes=axes.ravel(); plt.subplots_adjust(wspace=0.5,hspace=0.45)
    for ax,m in zip(axes,mets):
        k=m.split('.',1)[1]; a=A.loc[m]; ax.axvspan(a.p5,a.p95,color='#dddddd',alpha=0.6,lw=0); ax.axvline(a.p50,color='#999',lw=0.8)
        for i,(lab,D,s,col) in enumerate(groups):
            r=a if s is None else D[(D.k==m)&(D.stratum==s)].iloc[0] if ((D.k==m)&(D.stratum==s)).any() else None
            era=s in ('pre1990','1991_2000','2001_2010','post2010')
            if r is None:
                if era: ax.text(a.p5,i,'era-independent (parcel / road layer)',fontsize=7,va='center',color='#888')
                continue
            if m=='scalar.fractal_D' and era: ax.text(a.p5,i,'not comparable (era subset)',fontsize=7,va='center',color='#c4501f'); continue
            ax.plot([r.p5,r.p95],[i,i],color=col,lw=2.6,solid_capstyle='butt'); ax.plot(r.p50,i,'o',color='white',mec=col,mew=1.8,ms=6,zorder=3)
        ax.set_yticks(range(len(groups))); ax.set_yticklabels([g[0] for g in groups],fontsize=8); ax.invert_yaxis(); ax.set_title(NAMES.get(k,NAMES.get('assort.'+k,k)),fontsize=9.5,loc='left'); ax.grid(axis='x',alpha=0.25)
    fig.suptitle('Invariant-tier metrics keep their p5–p95 band (grey = all counties) across construction era, county type and size',fontweight='bold',y=0.98)
    save(fig,f'{out}/fig3_invariants_strata.png')

# ---------------- Fig 4 ----------------
def fig4(out):
    A=bounds(); L=A[A.metric=='L3'].copy(); L['a']=L.key.str.extract(r"'([^']+)'")[0]; L['b']=L.key.str.extract(r"'[^']+', '([^']+)'")[0]
    order=['\u5c45\u4f4f\u7528\u5730','\u5546\u670d\u7528\u5730','\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730','\u5de5\u4e1a\u7528\u5730','\u9053\u8def\u4e0e\u4ea4\u901a\u8bbe\u65bd\u7528\u5730']; code=[USE[u] for u in order]
    M=np.full((5,5),np.nan); C=np.full((5,5),np.nan)
    for _,r in L.iterrows():
        i,j=order.index(r.a),order.index(r.b); M[i,j]=M[j,i]=r.p50; C[i,j]=C[j,i]=r.sign_consistency
    fig=plt.figure(figsize=(18,6.2)); gs=fig.add_gridspec(1,3,width_ratios=[1.1,1,1],wspace=0.45)
    ax=fig.add_subplot(gs[0]); im=ax.imshow(np.log2(M),cmap='RdBu_r',vmin=-1.5,vmax=1.5)
    for i in range(5):
        for j in range(5):
            if np.isnan(M[i,j]): continue
            ax.text(j,i,f'{M[i,j]:.2f}'+('\n●' if C[i,j]>=0.9 else ''),ha='center',va='center',fontsize=8.5,color='k' if abs(np.log2(M[i,j]))<1 else 'white')
    ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels(code,fontsize=9); ax.set_yticklabels([f'{USE_LONG[c]} ({c})' for c in code],fontsize=8.5)
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',shrink=0.85,pad=0.08); cb.set_label('log₂ median adjacency lift (2,399 counties)'); ax.set_title('(a) Block adjacency lift (L3); ● = same sign in ≥ 90% of counties',loc='left',fontsize=9.5)
    ax=fig.add_subplot(gs[1]); S=A[A.metric=='assort'].set_index('key'); feats=[('log_h','Height'),('log_elong','Elongation'),('log_area','Area'),('compact','Compactness')]
    for i,(k,l) in enumerate(feats):
        r=S.loc[k]; ax.plot([r.p5,r.p95],[i,i],color='#1c5cab',lw=5,solid_capstyle='butt'); ax.plot(r.p50,i,'o',color='white',mec='#1c5cab',mew=2,ms=8,zorder=3); ax.text(r.p95+0.01,i,f'{r.sign_consistency*100:.1f}% > 0',va='center',fontsize=8.5)
    ax.axvline(0,color='k',lw=0.8); ax.set_yticks(range(4)); ax.set_yticklabels([l for _,l in feats]); ax.invert_yaxis(); ax.set_xlim(-0.05,0.65); ax.set_xlabel('Assortativity r of adjacent buildings (< 7.5 m gap), p5–p95'); ax.set_title('(b) Building–building assortativity (L1)',loc='left',fontsize=10); ax.grid(axis='x',alpha=0.25)
    ax=fig.add_subplot(gs[2]); U=A[A.metric=='use_height_ratio'].set_index('key')
    for i,u in enumerate(order):
        if u not in U.index: continue
        r=U.loc[u]; ax.plot([r.p5,r.p95],[i,i],color='#c4501f',lw=5,solid_capstyle='butt'); ax.plot(r.p50,i,'o',color='white',mec='#c4501f',mew=2,ms=8,zorder=3); ax.text(r.p95+0.02,i,f'{r.share_gt1*100:.0f}% > 1',va='center',fontsize=8.5)
    ax.axvline(1,color='k',lw=0.8); ax.set_yticks(range(5)); ax.set_yticklabels([USE_LONG[c] for c in code]); ax.invert_yaxis(); ax.set_xlim(0.6,1.7); ax.set_xlabel('Median height in use ÷ county median (p5–p95)'); ax.set_title('(c) Relative building height by land use (L2)',loc='left',fontsize=10); ax.grid(axis='x',alpha=0.25)
    save(fig,f'{out}/fig4_relations.png')

# ---------------- Fig 5 ----------------
def fig5(out):
    cm=pd.read_csv(f'{OUT}/county_metrics.csv',encoding='utf-8-sig'); R=[500,1000,2000,4000,8000]; cols=[f'L6.entropy_{r}' for r in R]
    d=cm.dropna(subset=cols+['n_buildings']); d=d[d.n_buildings>=1000]; E=d[cols].values
    fig=plt.figure(figsize=(16,5.2)); gs=fig.add_gridspec(1,3,width_ratios=[1.2,0.9,1],wspace=0.3)
    ax=fig.add_subplot(gs[0]); q=lambda p: np.percentile(E,p,axis=0)
    ax.fill_between(R,q(5),q(95),color='#c9d7ea',alpha=0.7,label='p5–p95'); ax.fill_between(R,q(25),q(75),color='#7a9cc6',alpha=0.8,label='p25–p75'); ax.plot(R,q(50),'o-',color='#1c5cab',lw=2,label='median')
    ax.axhline(np.log(5),color='k',ls=':',lw=0.9); ax.text(520,np.log(5)+0.02,'ln 5 (uniform mix of 5 uses)',fontsize=8)
    ax.set_xscale('log'); ax.set_xticks(R); ax.set_xticklabels(['0.5','1','2','4','8']); ax.set_xlabel('Window radius (km)'); ax.set_ylabel('Land-use mixing entropy H(r)'); ax.legend(frameon=False,loc='lower right'); ax.set_title(f'(a) Mixing entropy vs window radius (n = {len(d):,} counties)',loc='left',fontsize=10); ax.grid(alpha=0.25)
    ax=fig.add_subplot(gs[1]); sl=d['L6.slope'].dropna(); ax.hist(sl,bins=40,color='#7a9cc6',edgecolor='white'); p5,p50,p95=np.percentile(sl,[5,50,95])
    for x,c in ((p5,'#999'),(p50,'#1c5cab'),(p95,'#999')): ax.axvline(x,color=c,lw=1.2)
    mono=float(np.mean(np.all(np.diff(E,axis=1)>0,axis=1))); ax.set_xlabel('Slope dH / d ln r'); ax.set_ylabel('Counties'); ax.set_title(f'(b) Slope dH/d ln r: median {p50:.3f} [{p5:.3f}, {p95:.3f}]'+chr(10)+f'{mono*100:.1f}% of counties strictly monotone',loc='left',fontsize=10)
    ax=fig.add_subplot(gs[2]); B=bounds().set_index('k')
    for name,col,labs in [('type','#1c5cab',{'\u5e02\u8f96\u533a':'urban district','\u53bf\u7ea7\u5e02':'county-level city','\u53bf':'county'}),('size','#c4501f',{'small':'small','medium':'medium','large':'large'})]:
        S=strata(name)
        for ls,(s,l) in zip(['-','--',':'],labs.items()):
            y=[S[(S.k==f'scalar.L6.entropy_{r}')&(S.stratum==s)].p50.values[0] for r in R]; ax.plot(R,y,ls=ls,color=col,lw=1.6,label=f'{name}: {l}')
    ax.plot(R,[B.loc[f'scalar.L6.entropy_{r}'].p50 for r in R],'o',color='k',ms=5,label='all counties',zorder=4)
    ax.set_xscale('log'); ax.set_xticks(R); ax.set_xticklabels(['0.5','1','2','4','8']); ax.set_xlabel('Window radius (km)'); ax.set_ylabel('Median H(r)'); ax.legend(frameon=False,fontsize=8); ax.set_title('(c) Median curve by county type and size',loc='left',fontsize=10); ax.grid(alpha=0.25)
    save(fig,f'{out}/fig5_mixing_entropy.png')

# ---------------- Fig 6 ----------------
def fig6(out):
    import xarray as xr, geopandas as gpd
    ds=xr.open_dataset(f'{OUT}/county_spectra.nc',engine='scipy'); ref=json.load(open(f'{OUT}/national_reference_spectrum.json',encoding='utf-8'))['ref1']
    dev=pd.read_csv(f'{OUT}/spectrum_deviation.csv',encoding='utf-8-sig'); keep=set(dev.name); mask=np.array([c in keep for c in ds.county.values])
    feats=[('log_area','Footprint area (log₁₀ m²)'),('height','Height (m)'),('compact','Compactness'),('rect','Rectangularity'),('log_elong','Elongation (log₁₀)'),('orient','Orientation (°)')]
    fig=plt.figure(figsize=(19,8.6)); gs=fig.add_gridspec(2,4,width_ratios=[1,1,1,2.1],wspace=0.3,hspace=0.42)
    for i,(f,lab) in enumerate(feats):
        ax=fig.add_subplot(gs[i//3,i%3]); x=ds[f'{f}_bin'].values; H=ds[f'd1_{f}'].values[mask]
        ax.fill_between(x,np.nanpercentile(H,5,axis=0),np.nanpercentile(H,95,axis=0),color='#c9d7ea',alpha=0.8,label='county p5–p95'); ax.plot(x,ref[f],color='#1c5cab',lw=1.8,label='national reference')
        ax.set_xlabel(lab); ax.set_ylabel('Share of buildings'); ax.set_title(f'({chr(97+i)}) {lab.split(" (")[0]} — median county EMD {dev["emd_"+f].median():.3f}',loc='left',fontsize=9.5); ax.grid(alpha=0.2)
        if i==0: ax.legend(frameon=False,fontsize=8)
    ax=fig.add_subplot(gs[:,3]); b=china(); cm=pd.read_csv(f'{OUT}/county_metrics.csv',encoding='utf-8-sig')[['folder','PAC']]
    g=b.merge(dev.merge(cm,left_on='name',right_on='folder')[['PAC','emd_total']],on='PAC',how='left')
    g[g.emd_total.isna()].plot(ax=ax,color='#ececec',edgecolor='white',linewidth=0.15); g[g.emd_total.notna()].plot(ax=ax,column='emd_total',cmap='magma_r',vmin=0.1,vmax=0.5,edgecolor='white',linewidth=0.15)
    sm=plt.cm.ScalarMappable(cmap='magma_r',norm=plt.Normalize(0.1,0.5)); cb=fig.colorbar(sm,ax=ax,shrink=0.5,pad=0.01); cb.set_label('Total spectrum deviation (EMD, sum over 6 features)')
    ax.set_axis_off(); ax.set_title(f'(g) Deviation from the national reference spectrum\np5 / p50 / p95 = {dev.emd_total.quantile(.05):.3f} / {dev.emd_total.median():.3f} / {dev.emd_total.quantile(.95):.3f}',loc='left',fontsize=9.5)
    save(fig,f'{out}/fig6_spectra.png')

# ---------------- Fig 7 ----------------
def fig7(out):
    cc=pd.read_csv(f'{VAL}/crosscheck_counties.csv',encoding='utf-8-sig'); se=pd.read_csv(f'{VAL}/sensitivity_summary.csv',encoding='utf-8-sig')
    bs=pd.read_csv(f'{VAL}/bootstrap_summary.csv',encoding='utf-8-sig'); hn=pd.read_csv(f'{VAL}/heightnoise_summary.csv',encoding='utf-8-sig')
    fig=plt.figure(figsize=(17,15)); gs=fig.add_gridspec(3,2,wspace=0.3,hspace=0.45)
    # (a) crosscheck
    ax=fig.add_subplot(gs[0,0]); items=[('rect_pearson','Rectangularity vs momepy (Pearson)'),('orient_dev_pearson','Orientation deviation vs momepy (Pearson)'),('log_elong_spearman','Elongation vs momepy (|Spearman|)'),('compact_pearson','Compactness vs momepy circular (Pearson)')]
    vals=[cc[k].abs() for k,_ in items]; ax.boxplot(vals,vert=False,widths=0.55,patch_artist=True,boxprops=dict(fc='#c9d7ea'),medianprops=dict(color='#1c5cab',lw=2))
    ax.set_yticks(range(1,5)); ax.set_yticklabels([l for _,l in items],fontsize=8.5); ax.set_xlim(0.7,1.005); ax.set_xlabel('Per-county correlation (50 counties, ≤ 20,000 buildings each)'); ax.set_title('(a) Building metrics vs momepy',loc='left',fontsize=10); ax.grid(axis='x',alpha=0.25)
    ins=ax.inset_axes([0.06,0.08,0.4,0.5]); ins.scatter(cc.ox_entropy_weighted,cc.road_entropy_ours,s=12,color='#c4501f'); lim=(min(cc.ox_entropy_weighted.min(),cc.road_entropy_ours.min())-0.01,1.0); ins.plot(lim,lim,'k--',lw=0.7); ins.set_xlim(lim); ins.set_ylim(lim)
    r=np.corrcoef(cc.ox_entropy_weighted,cc.road_entropy_ours)[0,1]; ins.set_title(f'Street orientation entropy\nours vs osmnx: r = {r:.2f}',fontsize=7.5); ins.tick_params(labelsize=6.5)
    # (b) sensitivity heatmap
    ax=fig.add_subplot(gs[0,1]); mets=['area_dist.logn_mu','form_diversity','fractal_D','height_moran','sun_ratio_median','assort.log_h','assort.log_area','L3.\u5c45\u4f4f_\u5c45\u4f4f','L3.\u5c45\u4f4f_\u5de5\u4e1a','L3.\u5de5\u4e1a_\u5de5\u4e1a','L6.entropy_500','L6.slope']
    mets=[m for m in mets if m in set(se.metric)]; var=sorted(se.variant.unique(),key=lambda v:(v.split('=')[0],float(v.split('=')[1])))
    H=np.full((len(var),len(mets)),np.nan)
    for i,v in enumerate(var):
        for j,m in enumerate(mets):
            q=se[(se.variant==v)&(se.metric==m)]
            if len(q): H[i,j]=q.median_rel_change.values[0]*100
    H=np.nan_to_num(H,nan=0.0); im=ax.imshow(H,cmap='Reds',vmin=0,vmax=40,aspect='auto')
    SHORT={'L6.entropy_500':'Mixing entropy, r = 500 m','L6.slope':'Mixing-entropy slope','L3.\u5c45\u4f4f_\u5c45\u4f4f':'L3 lift R–R','L3.\u5c45\u4f4f_\u5de5\u4e1a':'L3 lift R–M','L3.\u5de5\u4e1a_\u5de5\u4e1a':'L3 lift M–M'}
    ax.set_xticks(range(len(mets))); ax.set_xticklabels([SHORT.get(m,NAMES.get(m,m)) for m in mets],rotation=40,ha='right',fontsize=8)
    ax.set_yticks(range(len(var))); ax.set_yticklabels([v.replace('=',' = ') for v in var],fontsize=8)
    for i in range(len(var)):
        for j in range(len(mets)):
            if H[i,j]>=1: ax.text(j,i,f'{H[i,j]:.0f}',ha='center',va='center',fontsize=7,color='white' if H[i,j]>25 else 'k')
    cb=fig.colorbar(im,ax=ax,shrink=0.8); cb.set_label('Median relative change (%)'); ax.set_title('(b) Parameter sensitivity (20 counties; defaults × 0.5 / × 1.5); blank < 1%',loc='left',fontsize=10)
    # (c) bootstrap
    SHORT={'use_height_ratio.\u5546\u670d':'Relative height, commercial','use_height_ratio.\u5c45\u4f4f':'Relative height, residential','use_height_ratio.\u5de5\u4e1a':'Relative height, industrial','FAR.\u5c45\u4f4f':'Est. FAR, residential','FAR.\u5546\u670d':'Est. FAR, commercial','FAR.\u5de5\u4e1a':'Est. FAR, industrial','L6.entropy_500':'Mixing entropy, r = 500 m','L6.slope':'Mixing-entropy slope','L3.\u5c45\u4f4f_\u5c45\u4f4f':'L3 lift R–R','L3.\u5c45\u4f4f_\u5de5\u4e1a':'L3 lift R–M','L3.\u5de5\u4e1a_\u5de5\u4e1a':'L3 lift M–M'}
    ax=fig.add_subplot(gs[1,0]); b=bs.sort_values('median_rel_halfwidth'); lab=[SHORT.get(m,NAMES.get(m,m)) for m in b.metric]
    ax.barh(range(len(b)),b.median_rel_halfwidth*100,color='#7a9cc6'); ax.plot(b.p90_rel_halfwidth*100,range(len(b)),'|',color='#c4501f',ms=10,mew=2,label='p90 across counties')
    ax.set_yticks(range(len(b))); ax.set_yticklabels(lab,fontsize=8); ax.set_xlabel('95% CI half-width / value (%), 80% building subsamples × 20'); ax.set_title('(c) Sampling uncertainty (20 counties)',loc='left',fontsize=10); ax.legend(frameon=False,fontsize=8,loc='lower right'); ax.grid(axis='x',alpha=0.25); ax.set_xscale('log'); ax.set_xlim(0.1,120)
    # (d) height noise
    ax=fig.add_subplot(gs[1,1]); sig={'iid_0.15':0.15,'iid_0.30':0.30,'iid_0.45':0.45}
    for m,lab,col in [('height_moran',"Height Moran's I",'#1c5cab'),('assort.log_h','Height assortativity','#7a9cc6'),('sun_ratio_median','Spacing-to-height ratio','#0a7a3c'),('FAR.\u5c45\u4f4f','Est. FAR, residential','#c4501f'),('use_height_ratio.\u5546\u670d','Relative height, commercial','#e0a458'),('form_diversity','Form diversity','#868c88')]:
        q=hn[hn.metric==m].set_index('noise'); xs=[0]+[sig[k] for k in sig]; ys=[0]+[q.loc[k,'median_signed_change']*100 for k in sig]
        ax.plot(xs,ys,'o-',color=col,lw=1.8,label=lab)
        if 'corr_0.30' in q.index: ax.plot(0.30,q.loc['corr_0.30','median_signed_change']*100,'s',color=col,ms=8,mfc='white',mew=1.8)
    ax.axhline(0,color='k',lw=0.8); ax.set_xlabel('Log-normal height noise σ (i.i.d.); open squares = spatially correlated noise, σ = 0.30'); ax.set_ylabel('Median change of the metric (%)'); ax.set_title('(d) Propagation of building-height error (20 counties × 3 seeds)',loc='left',fontsize=10); ax.legend(frameon=False,fontsize=8,ncol=2); ax.grid(alpha=0.25)
    # (e) discriminative power: score strip from benchmark_results.csv (real holdout / 7 perturbations / building generators)
    ax=fig.add_subplot(gs[2,0]); br=pd.read_csv(f'{OUT}/benchmark_results.csv',encoding='utf-8-sig'); rng=np.random.RandomState(0)
    lab={'uniform':'uniform blocks','scale':'footprints × 0.5','use_shuffle':'land-use shuffle','height_shuffle':'height shuffle','jitter':'jitter 50 m','rotate':'rotate','road_shuffle':'road-class shuffle'}
    groups=[('Real counties (50 held out)',br[br.group=='real_holdout'].score.values,'#1c5cab')]
    real=br[br.group=='real_holdout'].score.values; rs=br[br.group=='real_holdout'].set_index('case').score
    def _auc(a,b): a=np.asarray(a,float)[:,None]; b=np.asarray(b,float)[None,:]; return float(((a>b).sum()+0.5*(a==b).sum())/(a.size*b.size))
    for k,l in lab.items():
        q=br[(br.group=='perturbed')&(br.variant==k)]; v=q.score.values
        if not len(v): continue
        auc=_auc(real,v)
        # county-paired bootstrap 95% CI (same method as score.py auc_ci: the same counties are the resampling unit, 2000 resamples, seed 0)
        ps=q.set_index('case').score; cs=[c for c in rs.index if c in ps.index]; r_=rs.loc[cs].values; p_=ps.loc[cs].values; rb=np.random.RandomState(0); bs=[]
        for _ in range(2000):
            i=rb.randint(0,len(cs),len(cs)); bs.append(_auc(r_[i],p_[i]))
        groups.append((f'Perturbed: {l}  (AUC {auc:.2f} [{np.percentile(bs,2.5):.2f}, {np.percentile(bs,97.5):.2f}])',v,'#868c88'))
    for fam,l in [('learned layout (CityDreamer)','CityDreamer, 3 seeds'),('L-system procedural (josauder)','josauder, 3 scales'),('tensor-field procedural (MapGenerator)','MapGenerator, 3 seeds')]:
        q=br[(br.group=='generated')&(br.family==fam)]; q=q[~q.case.str.contains('randlu|x2.5|x10|_s0$',regex=True)] if fam.startswith('learned') else q[~q.case.str.contains('randlu|x2.5|x10',regex=True)]
        groups.append((f'Generated: {l}',q.score.values,'#c4501f'))
    for i,(n,v,c) in enumerate(groups):
        if not len(v): continue
        y=i+rng.uniform(-0.18,0.18,len(v)); ax.scatter(v,y,s=16 if len(v)>5 else 46,color=c,alpha=0.6 if len(v)>5 else 1,edgecolor='none' if len(v)>5 else 'k',zorder=3)
        ax.plot([np.median(v)]*2,[i-0.3,i+0.3],color='k',lw=1.6,zorder=4)
    m=real.min(); ax.axvline(m,color='#1c5cab',ls='--',lw=1); ax.text(m+0.006,len(groups)-0.7,f'lowest held-out county {m:.2f}',color='#1c5cab',fontsize=8,va='top')
    ax.set_yticks(range(len(groups))); ax.set_yticklabels([g[0] for g in groups],fontsize=8); ax.invert_yaxis(); ax.set_xlim(0,1.02); ax.set_xlabel('Conformity score, protocol v2 (bar = median); reference recomputed without the 50 held-out counties'); ax.grid(axis='x',alpha=0.3)
    ax.set_title('(e) Discriminative power (AUC real > perturbed, 95% county-paired bootstrap CI)',loc='left',fontsize=10)
    # (f) land-use label-error propagation (TV10), if the run exists
    ax=fig.add_subplot(gs[2,1]); lp=f'{VAL}/labelnoise_summary.csv'
    if os.path.exists(lp):
        ln=pd.read_csv(lp,encoding='utf-8-sig'); fr={'label_0.05':5,'label_0.10':10,'label_0.20':20}
        for m,l,col in [('L3.\u5c45\u4f4f_\u5c45\u4f4f','L3 lift R–R','#1c5cab'),('L3.\u5de5\u4e1a_\u5de5\u4e1a','L3 lift M–M','#7a9cc6'),('L3.\u5c45\u4f4f_\u5de5\u4e1a','L3 lift R–M','#0a7a3c'),('L3.\u516c\u7ba1_\u5c45\u4f4f','L3 lift A–R','#6fa26f'),('L6.slope','Mixing-entropy slope','#c4501f'),('L6.entropy_500','Mixing entropy, r = 500 m','#e0956f'),('use_height_ratio.\u5546\u670d','Relative height, commercial','#7d5ba6'),('FAR.\u5c45\u4f4f','Est. FAR, residential','#999')]:
            q=ln[ln.metric==m].set_index('noise')
            if not len(q): continue
            xs=[0]+[fr[k] for k in fr if k in q.index]; ys=[0]+[q.loc[k,'median_signed_change']*100 for k in fr if k in q.index]; ax.plot(xs,ys,'o-',color=col,lw=1.8,label=l)
        ax.axhline(0,color='k',lw=0.8); ax.set_xlabel('Share of parcels relabelled to another use (%)'); ax.set_ylabel('Median change of the metric (%)'); ax.legend(frameon=False,fontsize=8,ncol=2); ax.grid(alpha=0.25)
        kept=ln[(ln.noise=='label_0.20')&ln.metric.str.startswith('L3.')].sign_kept
        ax.set_title(f'(f) Land-use label-error propagation (20 counties × 3 seeds); at 20% the lifts keep their side of 1 in {kept.min()*100:.0f}–{kept.max()*100:.0f}% of counties',loc='left',fontsize=10)
    else: ax.set_axis_off(); ax.text(0.5,0.5,'(f) label-noise run pending',ha='center',va='center')
    save(fig,f'{out}/fig7_validation.png')

# ---------------- GIS export (QGIS) ----------------
ARCH_EN={'\u5c45\u4f4f\u4e3b\u5bfc':'residential_dominant','\u516c\u7ba1\u4e3b\u5bfc':'public_dominant','\u5c45\u4f4f + \u516c\u7ba1\u6df7\u5408':'residential_public_mix','\u5de5\u4e1a + \u5c45\u4f4f\u6df7\u5408':'industrial_residential_mix',
         '\u5de5\u4e1a\u4e3b\u5bfc':'industrial_dominant','\u5546\u670d + \u5c45\u4f4f\u6df7\u5408':'commercial_residential_mix'}
def gis(out):
    """Map data behind Fig 1b / Fig 6b -> GeoParquet (original EPSG:4490 coordinates, full resolution): counties + all county metrics, provinces + provincial medians and archetype shares."""
    import geopandas as gpd, sys
    sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from export import en_columns
    g=f'{out}/gis'; os.makedirs(g,exist_ok=True)
    cm=pd.read_csv(f'{OUT}/county_metrics.csv',encoding='utf-8-sig'); cm.columns=en_columns(cm.columns)
    dev=pd.read_csv(f'{OUT}/spectrum_deviation.csv',encoding='utf-8-sig').rename(columns={'name':'folder'}).drop(columns=['n','lat'],errors='ignore')
    cm=cm.merge(dev,on='folder',how='left')
    cm.columns=[c.replace('.','_') for c in cm.columns]
    cm['in_reference']=(cm.n_buildings>=1000).astype(int); cm['log10_n_buildings']=np.log10(cm.n_buildings.where(cm.n_buildings>0))
    b=gpd.read_file(BND).rename(columns={'NAME':'name_zh','\u7701\u4ee3\u7801':'province_code','\u7701':'province_zh','\u5e02\u4ee3\u7801':'prefecture_code','\u5e02':'prefecture_zh','\u7c7b\u578b':'type_zh'})
    b['PAC']=b['PAC'].astype(str).str.zfill(6); cm['PAC']=cm['PAC'].astype('Int64').astype(str).str.zfill(6)
    c=b.merge(cm.drop(columns=['province','prefecture','county'],errors='ignore'),on='PAC',how='left')
    c['has_buildings']=c.n_buildings.notna().astype(int); c['in_reference']=c.in_reference.fillna(0).astype(int)
    # area (Albers equal-area, km2) and building density: absolute counts of large counties (Xinjiang / Inner Mongolia) mislead visually, so maps use density
    c['area_km2']=c.geometry.to_crs('+proj=aea +lat_1=25 +lat_2=47 +lon_0=105 +datum=WGS84').area/1e6
    c['density_per_km2']=c.n_buildings/c['area_km2']; c['log10_density_per_km2']=np.log10(c.density_per_km2.where(c.density_per_km2>0))
    c['reference_status']=np.where(c.in_reference==1,'reference',np.where(c.has_buildings==1,'under_1000','no_buildings'))
    c.to_parquet(f'{g}/cufb_counties.parquet',index=False)
    # provinces: medians over reference counties (north-south split of signature metrics), counts of counties and buildings, shares of the six archetypes
    ref=cm[cm.in_reference==1].copy(); ref['province_zh']=ref.folder.str.split('_').str[0]
    keys=['road_grid_share','orient_cardinal10','road_orient_entropy','height_moran','assort_log_h','FAR_R','setback_median','blocks_median_area','sun_ratio_median','emd_total','L6_slope','fractal_D','form_diversity']
    pv=ref.groupby('province_zh').agg(n_counties=('folder','size'),n_buildings=('n_buildings','sum'),**{f'{k}_median':(k,'median') for k in keys if k in ref}).reset_index()
    ar=pd.read_csv(f'{OUT}/archetype_by_province.csv',encoding='utf-8-sig').rename(columns={'\u7701':'province_zh',**{k:f'arch_{v}' for k,v in ARCH_EN.items()}})
    P=gpd.read_file(BND.replace('\u53bf.shp','\u7701.shp')).rename(columns={'\u7701\u4ee3\u7801':'province_code','\u7701':'province_zh','\u7c7b\u578b':'type_zh'})
    P=P.merge(pv,on='province_zh',how='left').merge(ar,on='province_zh',how='left'); P.to_parquet(f'{g}/cufb_provinces.parquet',index=False)
    open(f'{g}/README.md','w',encoding='utf-8').write(f"""# Map layers for the CUFB-China figures (GeoParquet, EPSG:4490 / CGCS2000, full resolution)

`cufb_counties.parquet` — {len(c):,} county-level polygons (the national boundary layer; the units of Taiwan, Hong Kong and Macao are included but carry no CMAB buildings; one nameless China–DPRK border polygon is dropped in the benchmark) joined on `PAC`
to every column of `county_metrics.csv` (release names: R/A/B/M/S use codes, dots replaced by `_`) and `spectrum_deviation.csv`.
Helper columns: `has_buildings` (0/1), `in_reference` (n ≥ 1000, 0/1), `log10_n_buildings`, `area_km2` (Albers equal-area), `density_per_km2` (= n_buildings / area_km2), `log10_density_per_km2`, `reference_status` (reference / under_1000 / no_buildings).
  * Fig 1b (recommended) = `log10_density_per_km2` (YlGnBu, −2 … 2.5, i.e. 0.01–300 buildings per km²; `has_buildings = 0` in grey) — absolute counts make the large western counties look dense
  * Fig 1b  = `log10_n_buildings` (viridis, 2–5.1; `has_buildings = 0` in grey)
  * Fig 6b  = `emd_total` (magma_r, 0.1–0.5; NaN = not in reference, grey)
  * any other metric can be mapped the same way, e.g. `road_grid_share`, `orient_cardinal10`, `FAR_R`, `assort_log_h`

`cufb_provinces.parquet` — {len(P)} province polygons (incl. Taiwan, Hong Kong and Macao) with, over the reference counties of each province:
`n_counties`, `n_buildings`, `{{metric}}_median` for {len(keys)} signature / invariant metrics, and the six community-archetype shares `arch_*`.
Maps in the descriptor's Usage Notes (grid share 0.58 Shandong vs 0.23–0.27 south; public-dominant communities 50% Xinjiang) come from these columns.

Not included: the nine-dash line and any cartographic base — add them in QGIS. Projection used in the descriptor: Albers equal-area
`+proj=aea +lat_1=25 +lat_2=47 +lon_0=105 +lat_0=0 +datum=WGS84`.
""")
    print(f'-> {g}/cufb_counties.parquet ({len(c)} rows, {c.shape[1]} cols), cufb_provinces.parquet ({len(P)} rows, {P.shape[1]} cols)')

FIGS={'fig1':fig1,'fig2':fig2,'fig3':fig3,'fig4':fig4,'fig5':fig5,'fig6':fig6,'fig7':fig7}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--only',default=None); ap.add_argument('--out',default='dataset/figures'); ap.add_argument('--gis',action='store_true'); a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    if a.gis: gis(a.out); return
    for k in (a.only.split(',') if a.only else FIGS):
        try: FIGS[k](a.out)
        except Exception as e: import traceback; traceback.print_exc(); print('FAILED',k,e)
if __name__=='__main__': main()
