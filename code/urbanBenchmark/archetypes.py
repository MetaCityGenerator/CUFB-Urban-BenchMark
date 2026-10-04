# -*- coding: utf-8 -*-
"""L5 archetypes: cluster the Louvain-community land-use composition vectors (5 area shares) from every county result.json into national archetypes.
Usage: python archetypes.py --root dataset/output_county [--k 6] [--min-n 1000]
Output (written to --root):
  community_vectors.csv     one row per community: county, 5 use shares, archetype
  national_archetypes.json  per archetype: name, number of communities, share, mean composition
  archetype_by_province.csv province x archetype shares
Naming (Chinese labels; translated to English at release time by make_release.py): largest share >= 0.6 -> "{use} dominant", otherwise the top two -> "{a} + {b} mix". KMeans (seed 0) depends on k; k is fixed and reported for the release."""
import os, glob, json, argparse, numpy as np, pandas as pd
USES=['\u5c45\u4f4f','\u516c\u7ba1','\u5de5\u4e1a','\u5546\u670d','\u4ea4\u901a']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--k',type=int,default=6)
    ap.add_argument('--min-n',type=int,default=1000); ap.add_argument('--min-size',type=int,default=5); a=ap.parse_args()
    from sklearn.cluster import KMeans
    rows=[]
    for f in sorted(glob.glob(f"{a.root}/*/result.json")):
        name=os.path.basename(os.path.dirname(f))
        try: r=json.load(open(f,encoding='utf-8'))
        except Exception: continue
        n=r.get('n_buildings')
        if n is None:
            sp=os.path.join(os.path.dirname(f),'spectrum.json')
            if os.path.exists(sp):
                try: n=json.load(open(sp,encoding='utf-8')).get('n')
                except Exception: pass
        if n is not None and n<a.min_n: continue
        for v in r.get('L5_comm') or []:
            if len(v)==5 and abs(sum(v)-1)<0.05: rows.append([name]+list(v))
    df=pd.DataFrame(rows,columns=['county']+USES); X=df[USES].values
    km=KMeans(a.k,n_init=10,random_state=0).fit(X); df['cluster']=km.labels_
    names={}
    for c in range(a.k):
        m=X[df.cluster==c].mean(0); order=np.argsort(-m)
        names[c]=f"{USES[order[0]]}\u4e3b\u5bfc" if m[order[0]]>=0.6 else f"{USES[order[0]]} + {USES[order[1]]}\u6df7\u5408"
    # de-duplicate identical names (rare): append a number
    seen={}
    for c,nm in list(names.items()):
        if nm in seen: seen[nm]+=1; names[c]=f"{nm}{seen[nm]}"
        else: seen[nm]=1
    df['archetype']=df.cluster.map(names); df['\u7701']=df.county.str.split('_').str[0]
    df[['county','\u7701']+USES+['archetype']].to_csv(f"{a.root}/community_vectors.csv",index=False,encoding='utf-8-sig')
    out=[]
    for c in sorted(names,key=lambda c:-(df.cluster==c).sum()):
        sub=df[df.cluster==c]; m=sub[USES].mean()
        out.append({'archetype':names[c],'n':int(len(sub)),'share':round(len(sub)/len(df),4),**{f'pct_{u}':round(float(m[u]),3) for u in USES}})
    json.dump({'k':a.k,'n_communities':int(len(df)),'n_counties':int(df.county.nunique()),'archetypes':out},
              open(f"{a.root}/national_archetypes.json",'w',encoding='utf-8'),ensure_ascii=False,indent=1)
    prov=pd.crosstab(df['\u7701'],df['archetype'],normalize='index').round(3)
    prov=prov[[o['archetype'] for o in out]]; prov.to_csv(f"{a.root}/archetype_by_province.csv",encoding='utf-8-sig')
    print(f"[archetypes] {len(df):,} communities / {df.county.nunique()} counties -> k={a.k}")
    for o in out: print(f"  {o['archetype']:12s} n={o['n']:6d} share={o['share']:.3f}  "+' '.join(f"{u}={o[f'pct_{u}']:.2f}" for u in USES))
if __name__=='__main__': main()
