# -*- coding: utf-8 -*-
"""Release headers and labels for the CUFB-China data record.

The working tables in dataset/output_county keep the Chinese land-use and archetype labels
produced by the analysis. This script rewrites copies of the released tables:
county_metrics.csv column names -> English + GB 50137-2011 codes R/A/B/M/S;
county_relations.csv use columns a/b and community_vectors.csv use columns -> codes;
archetype names -> English.

Usage: python release_tables.py --record <folder holding 01_county_tables/ and 03_communities/>
Run it only on a copy (e.g. the release folder), never on dataset/output_county.
"""
import os, sys, argparse
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ARCH_EN={'\u5c45\u4f4f\u4e3b\u5bfc':'residential_dominant','\u516c\u7ba1\u4e3b\u5bfc':'public_dominant','\u5c45\u4f4f + \u516c\u7ba1\u6df7\u5408':'residential_public_mix','\u5de5\u4e1a + \u5c45\u4f4f\u6df7\u5408':'industrial_residential_mix',
         '\u5de5\u4e1a\u4e3b\u5bfc':'industrial_dominant','\u5546\u670d + \u5c45\u4f4f\u6df7\u5408':'commercial_residential_mix'}
PROVINCE='\u7701'

def anglicise(droot):
    from export import en_columns, GB
    d=f"{droot}/01_county_tables"; m=pd.read_csv(f"{d}/county_metrics.csv",encoding='utf-8-sig'); m.columns=en_columns(m.columns); m.to_csv(f"{d}/county_metrics.csv",index=False,encoding='utf-8-sig')
    r=pd.read_csv(f"{d}/county_relations.csv",encoding='utf-8-sig'); r['a']=r['a'].replace(GB); r.loc[r.layer=='L3','b']=r.loc[r.layer=='L3','b'].replace(GB); r.to_csv(f"{d}/county_relations.csv",index=False,encoding='utf-8-sig')
    c=f"{droot}/03_communities/community_vectors.csv"; v=pd.read_csv(c,encoding='utf-8-sig'); v=v.rename(columns={**GB,PROVINCE:'province'}); v['archetype']=v['archetype'].replace(ARCH_EN); v.to_csv(c,index=False,encoding='utf-8-sig')
    a=f"{droot}/03_communities/archetype_by_province.csv"; v=pd.read_csv(a,encoding='utf-8-sig'); v=v.rename(columns={**ARCH_EN,PROVINCE:'province'}); v.to_csv(a,index=False,encoding='utf-8-sig')
    j=f"{droot}/03_communities/national_archetypes.json"; t=open(j,encoding='utf-8').read()
    for zh,en in ARCH_EN.items(): t=t.replace(f'"{zh}"',f'"{en}"')
    for zh,en in GB.items(): t=t.replace(f'"{zh}"',f'"{en}"').replace(f'"pct_{zh}"',f'"pct_{en}"')
    open(j,'w',encoding='utf-8').write(t)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--record',required=True); a=ap.parse_args()
    anglicise(a.record); print(f"[release_tables] headers and labels rewritten in {a.record}")
