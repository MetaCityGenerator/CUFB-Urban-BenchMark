# -*- coding: utf-8 -*-
"""Manuscript figures for the generated-city case studies (English labels; the journal is English).
  default   Fig 8b: real county vs generated cities (josauder / MapGenerator procedural, CityDreamer learned) — footprints + roads + height, same scale, verdict captions
  --scores  Fig 8a: strip plot of score v2 for real counties / 7 perturbations / 3 generators
  --landuse Fig 8c: community land-use plans (DRL-urban-planning RL agent + heuristics) vs real EULUC crops of the same Beijing communities, with L3 adjacency lifts
Usage: python fig_generated.py [--real dataset/input_county/<county folder>] [--gen dataset/generated] [--out ...] [--scores | --landuse]
Panel sources: dataset/generated/{josauder_mycity, mapgen_s0, citydreamer_s0_z18, citydreamer_s1, citydreamer_s2, drl_*, real_*_crop}/ (building/landuse/roads.shp, score.json, result.json)."""
import os, json, argparse
import geopandas as gpd, numpy as np, matplotlib
matplotlib.use('agg'); import matplotlib.pyplot as plt
from shapely.geometry import box
matplotlib.rcParams['font.family']=['DejaVu Sans','Arial']

def load(folder, crs_to=None, layers=('building','roads')):
    out=[]
    for l in layers:
        g=gpd.read_file(f'{folder}/{l}.shp')
        if crs_to is None:
            c=g.to_crs(4326).geometry.representative_point(); crs_to=32600+int((c.x.mean()+180)//6)+1
        out.append(g.to_crs(crs_to))
    return out
def crop(b,r,half):
    cx,cy=b.geometry.centroid.x.median(),b.geometry.centroid.y.median(); w=box(cx-half,cy-half,cx+half,cy+half)
    return b[b.intersects(w)].clip(w), r[r.intersects(w)].clip(w), w
def scalebar(ax,w,L=500,margin=0.10):
    """Scale bar drawn in the blank margin below the map (ylim extended downward), not over the map."""
    x0,y0,x1,y1=w.bounds; W=x1-x0; H=y1-y0; yb=y0-0.055*H
    ax.set_ylim(y0-margin*H,y1); ax.plot([x0+0.03*W,x0+0.03*W+L],[yb,yb],color='k',lw=3,solid_capstyle='butt')
    ax.text(x0+0.03*W+L+0.02*W,yb,f'{L} m',ha='left',va='center',fontsize=11)
def panel(ax,b,r,w,title,sub=None,hmax=60):
    """One footprint panel: grey streets + buildings coloured by height; two-line title (bold id + regular score/verdict), no paragraph text."""
    r.plot(ax=ax,color='#8a8a8a',linewidth=0.5); b.plot(ax=ax,column='Height',cmap='viridis',vmin=0,vmax=hmax,linewidth=0)
    x0,y0,x1,y1=w.bounds; ax.set_xlim(x0,x1); ax.set_ylim(y0,y1); ax.set_aspect('equal'); ax.set_axis_off()
    ax.set_title(title,fontsize=14,fontweight='bold',pad=22 if sub else 8,loc='left')
    if sub: ax.text(0,1.012,sub,transform=ax.transAxes,ha='left',va='bottom',fontsize=12.5,color='#333')
    scalebar(ax,w)
def score(f): return json.load(open(f'{f}/score.json',encoding='utf-8'))['score']

def panels_figure(real, G, out, dpi=200):
    fig,axes=plt.subplots(2,3,figsize=(21,15.2)); plt.subplots_adjust(wspace=0.06,hspace=0.2,top=0.95,bottom=0.075,left=0.02,right=0.98); axes=axes.ravel()
    b,r=load(real); bb,rr,w=crop(b,r,1000)
    panel(axes[0],bb,rr,w,'(a) Real county — Nankai District, Tianjin','CMAB footprints + OSM streets, 2 × 2 km window')
    b,r=load(f'{G}/josauder_mycity',32650); bb,rr,w=crop(b,r,1000)
    panel(axes[1],bb,rr,w,'(b) L-system procedural — josauder',f'score {score(f"{G}/josauder_mycity"):.2f} · right scale, wrong statistics')
    b,r=load(f'{G}/mapgen_s0',32650); bb,rr,w=crop(b,r,1000)
    panel(axes[2],bb,rr,w,'(c) Tensor-field procedural — MapGenerator',f'score {score(f"{G}/mapgen_s0"):.2f} · too regular, heights random')
    for ax,(d,t,lab) in zip(axes[3:],[('citydreamer_s0_z18','seed 0','(d)'),('citydreamer_s1','seed 1','(e)'),('citydreamer_s2','seed 2','(f)')]):
        b,r=load(f'{G}/{d}',32650); x0,y0,x1,y1=b.total_bounds
        panel(ax,b,r,box(x0,y0,x1,y1),f'{lab} Learned layout — CityDreamer, {t}',f'score {score(f"{G}/{d}"):.2f} · right statistics, wrong scale')
    cax=fig.add_axes([0.32,0.03,0.36,0.013]); sm=plt.cm.ScalarMappable(cmap='viridis',norm=plt.Normalize(0,60)); cb=fig.colorbar(sm,cax=cax,orientation='horizontal'); cb.set_label('Building height (m)',fontsize=12); cb.ax.tick_params(labelsize=11)
    plt.savefig(out,dpi=dpi,bbox_inches='tight'); print('->',out)

def scores_figure(G, ref='dataset/ref_holdout/scores_v2.csv', out=None):
    import pandas as pd
    s=pd.read_csv(ref,encoding='utf-8-sig'); groups=[('Real counties (50 held out)',s[s.group=='real'].score.values,'#1c5cab')]
    lab={'jitter':'Perturbed: jitter 50 m','rotate':'Perturbed: rotate','road_shuffle':'Perturbed: road-class shuffle','height_shuffle':'Perturbed: height shuffle','use_shuffle':'Perturbed: land-use shuffle','scale':'Perturbed: footprints × 0.5','uniform':'Perturbed: uniform blocks'}
    for k in lab: groups.append((lab[k],s[(s.group=='perturbed')&(s.perturbation==k)].score.values,'#868c88'))
    for n,ds in [('CityDreamer (learned) × 3 seeds',['citydreamer_s0_z18','citydreamer_s1','citydreamer_s2']),('josauder (procedural, L-system) × 3 scales',['josauder_mycity','josauder_mycity_s50','josauder_mycity_s200']),('MapGenerator (procedural, tensor field) × 3 seeds',['mapgen_s0','mapgen_s1','mapgen_s2'])]:
        groups.append((n,np.array([score(f'{G}/{d}') for d in ds if os.path.exists(f'{G}/{d}/score.json')]),'#c4501f'))
    fig,ax=plt.subplots(figsize=(10,6.2)); rng=np.random.RandomState(0)
    for i,(n,v,c) in enumerate(groups):
        if not len(v): continue
        y=i+rng.uniform(-0.18,0.18,len(v)); ax.scatter(v,y,s=22 if len(v)>5 else 60,color=c,alpha=0.65 if len(v)>5 else 1,edgecolor='none' if len(v)>5 else 'k',zorder=3)
        ax.plot([np.median(v)]*2,[i-0.3,i+0.3],color='k',lw=1.6,zorder=4)
    m=s[s.group=='real'].score.min(); ax.axvline(m,color='#1c5cab',ls='--',lw=1); ax.text(m+0.005,len(groups)-0.6,f'lowest real county {m:.2f}',color='#1c5cab',fontsize=9,va='top')
    ax.set_yticks(range(len(groups))); ax.set_yticklabels([g[0] for g in groups]); ax.invert_yaxis(); ax.set_xlim(0,1.02); ax.set_xlabel('Conformity score (protocol v2); bar = median'); ax.grid(axis='x',alpha=0.3)
    ax.set_title('Real counties, seven perturbations and three generators',fontweight='bold'); plt.tight_layout(); out=out or f'{G}/fig8a_scores.png'; plt.savefig(out,dpi=120); print('->',out)

USE_COL={'\u5c45\u4f4f\u7528\u5730':'#e15759','\u5546\u670d\u7528\u5730':'#f28e2b','\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730':'#4e79a7','\u5de5\u4e1a\u7528\u5730':'#76b7b2','\u9053\u8def\u4e0e\u4ea4\u901a\u8bbe\u65bd\u7528\u5730':'#bab0ab'}
USE_EN={'\u5c45\u4f4f\u7528\u5730':'Residential (R)','\u5546\u670d\u7528\u5730':'Commercial (B)','\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730':'Public services / green (A)','\u5de5\u4e1a\u7528\u5730':'Industrial (M)','\u9053\u8def\u4e0e\u4ea4\u901a\u8bbe\u65bd\u7528\u5730':'Transport (S)'}
def landuse_figure(G, out=None, dpi=200):
    """Real EULUC crops of the two Beijing communities vs plans from DRL-urban-planning (RL agent, centralized heuristic, GSCA); the three
    direction-consistent L3 lifts (R-R > 1, A-A > 1, A-R < 1 nationally) as the second title line, ✓/✗ against the national direction."""
    S={'\u5c45\u4f4f\u7528\u5730':'R','\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730':'A','\u5546\u670d\u7528\u5730':'B','\u5de5\u4e1a\u7528\u5730':'M','\u9053\u8def\u4e0e\u4ea4\u901a\u8bbe\u65bd\u7528\u5730':'S'}
    def lifts(d):
        r=json.load(open(f'{G}/{d}/result.json',encoding='utf-8')); return {f"{S[x['a']]}-{S[x['b']]}":x['lift'] for x in r.get('L3',[])}
    cols=[('real_{c}_crop','Real land use (EULUC 2.0, 2022)'),('drl_{c}_rl','RL agent (DRL-urban-planning)'),('drl_{c}_rule-centralized','Centralised heuristic'),('drl_{c}_gsca','GSCA heuristic')]
    fig,axes=plt.subplots(2,4,figsize=(22,12.6)); plt.subplots_adjust(wspace=0.08,hspace=0.22,top=0.93,bottom=0.06,left=0.035,right=0.99)
    for i,(c,name) in enumerate([('hlg','Huilongguan, Changping, Beijing'),('dhm','Dahongmen, Fengtai, Beijing')]):
        for j,(pat,title) in enumerate(cols):
            d=pat.format(c=c); ax=axes[i,j]; lu,r=load(f'{G}/{d}',32650,('landuse','roads'))
            lu.plot(ax=ax,color=[USE_COL.get(u,'#ccc') for u in lu['Level1_cn']],edgecolor='white',linewidth=0.4); r.plot(ax=ax,color='#555',linewidth=0.6)
            x0,y0,x1,y1=lu.total_bounds; ax.set_xlim(x0,x1); ax.set_ylim(y0,y1); ax.set_aspect('equal'); ax.set_axis_off()
            L=lifts(d); v=lambda k: L.get(k,float('nan'))
            ok=lambda k,dirn: (k in L) and ((L[k]>1) if dirn=='>' else (L[k]<1))
            txt='    '.join([f"R–R {v('R-R'):.2f} {'✓' if ok('R-R','>') else '✗'}",f"A–A {v('A-A'):.2f} {'✓' if ok('A-A','>') else '✗'}",f"A–R {v('A-R'):.2f} {'✓' if ok('A-R','<') else '✗'}"])
            ax.set_title(f"({chr(97+i*4+j)}) {title}",fontsize=13.5,fontweight='bold',loc='left',pad=22)
            ax.text(0,1.012,txt,transform=ax.transAxes,ha='left',va='bottom',fontsize=12.5,color='#333')
            if j==0: ax.text(-0.04,0.5,name,transform=ax.transAxes,rotation=90,va='center',ha='right',fontsize=13,fontweight='bold')
            scalebar(ax,box(x0,y0,x1,y1))
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=USE_COL[k],label=USE_EN[k]) for k in ['\u5c45\u4f4f\u7528\u5730','\u5546\u670d\u7528\u5730','\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730','\u5de5\u4e1a\u7528\u5730','\u9053\u8def\u4e0e\u4ea4\u901a\u8bbe\u65bd\u7528\u5730']],loc='lower center',ncol=5,frameon=False,fontsize=13,bbox_to_anchor=(0.5,0.0))
    out=out or f'{G}/fig8c_landuse_plans.png'; plt.savefig(out,dpi=dpi,bbox_inches='tight'); print('->',out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--real',default='dataset/input_county/\u5929\u6d25\u5e02_\u5929\u6d25\u5e02_\u5357\u5f00\u533a'); ap.add_argument('--gen',default='dataset/generated')
    ap.add_argument('--out',default=None); ap.add_argument('--scores',action='store_true'); ap.add_argument('--landuse',action='store_true'); ap.add_argument('--dpi',type=int,default=200); a=ap.parse_args(); G=a.gen
    if a.scores: scores_figure(G,out=a.out)
    elif a.landuse: landuse_figure(G,out=a.out,dpi=a.dpi)
    else: panels_figure(a.real,G,a.out or f'{G}/fig8b_generated_vs_real.png',dpi=a.dpi)
if __name__=='__main__': main()
