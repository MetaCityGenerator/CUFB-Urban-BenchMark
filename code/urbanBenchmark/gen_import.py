# -*- coding: utf-8 -*-
"""Convert the output of third-party city generators into the county three-layer format of this benchmark ({out}/building.shp, roads.shp[, landuse.shp] + _meta.json),
ready for benchmark.py analyze + score.py. Supported:
  josauder   josauder/procedural_city_generation (Parish-Mueller style L-system streets + lots + buildings, MPL-2.0)
             reads temp/{name} (street Vertex graph) and temp/{name}_polygons.txt (Polygon2D lots) and reproduces the footprint logic of building_generation/main.py
             (lot -> height from population density -> house/non-house scaling -> randomcut); no land-use layer.
             Scale from the generator itself: floorheight 0.03-0.04 units ~ 3-4 m -> 1 unit = 100 m (--scale to change).
  citydreamer  output of the CityDreamer (Xie et al. CVPR 2024) layout generator (seg.npy + hf.npy written by gen_citydreamer.py):
             buildings = 4-connected components of the BLD_FACADE (2) class -> outline polygons, height = mean of the height field per instance (m, OSM height tag convention);
             streets = skeleton of the ROAD (1) mask -> merged/simplified segments; no land-use layer (the generator only has road/building/green/construction classes).
             --mpp metres per pixel: training rasters are OSM zoom 15, 156543*cos(lat)/2^15 ~ 3.7 m/px at 40N (default), adjustable.
  mapgenerator ProbableTrain/MapGenerator (tensor-field streets; gen_mapgenerator.py runs it in headless Edge and writes mapgen_raw.json):
             buildings = the generator's building lots (shrunk and subdivided blocks, i.e. the footprints of its pseudo-3D buildings); height = the generator's own rule uniform(20, 40) (a decorative random number, imported as metres);
             streets = main/major/minor streamlines -> primary/secondary/residential; no land-use layer. --scale metres per world unit, default 5 (minor spacing dsep = 20 units ~ 100 m blocks).
  drlplan      community plans of DRL-urban-planning (Zheng et al. 2023, Nat Comput Sci; tsinghua-fib-lab): land-use parcels + streets, no buildings.
             input results/{hlg,dhm}.geojson (the RL plans of the paper) or plan.p saved by eval (rule-centralized / rule-decentralized / gsca baselines).
             type codes -> EULUC Level 1: 4 residential -> residential; 5 business, 6 office -> commercial; 7/8 green, 9 school, 10/11 hospital, 12 recreation -> public services
             (EULUC 0505 park & green belongs to 05 public services); 1 feasible (unplanned remainder) dropped; no industry. Coordinates are local metres, no scale needed.
Usage: python gen_import.py josauder --repo dataset/generators/procedural_city_generation --name mycity --out dataset/generated/josauder_mycity [--scale 100] [--seed 0]
      python gen_import.py citydreamer --layout dataset/generated/citydreamer_s0 [--mpp 3.66] [--out <same folder>]
      python gen_import.py mapgenerator --raw dataset/generated/mapgen_s0/mapgen_raw.json --out dataset/generated/mapgen_s0 [--scale 5] [--seed 0]
      python gen_import.py drlplan --plan dataset/generators/DRL-urban-planning/results/hlg.geojson --out dataset/generated/drl_hlg_rl
      python gen_import.py landuse --folder dataset/generated/josauder_mycity --mode blocks      # optional: add a land-use layer from street blocks with random uses, as a control
"""
import os, sys, json, argparse, pickle, numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import unary_union, polygonize

USES=['\u5c45\u4f4f\u7528\u5730','\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730','\u5de5\u4e1a\u7528\u5730','\u5546\u670d\u7528\u5730','\u9053\u8def\u4e0e\u4ea4\u901a\u8bbe\u65bd\u7528\u5730']
ORIGIN=(116.4,39.9)   # generated cities have no location; placed near Beijing so auto_epsg picks UTM 50N; latitude only affects the lat recorded in spectrum

def _meta(out, **kw):
    m={'PAC':None,'\u7c7b\u578b':'generated','source':kw.pop('source'),**kw}
    json.dump(m,open(f"{out}/_meta.json",'w',encoding='utf-8'),ensure_ascii=False,indent=1)

def _to_crs(gdf_xy_m, epsg=32650):
    """Generator coordinates (m, centred on 0,0) -> UTM coordinates with ORIGIN as origin -> saved in that UTM (benchmark re-selects the zone from the centroid; same result)."""
    import pyproj
    tr=pyproj.Transformer.from_crs(4326,epsg,always_xy=True); ox,oy=tr.transform(*ORIGIN)
    from shapely.affinity import translate
    g=gdf_xy_m.copy(); g['geometry']=g.geometry.apply(lambda p: translate(p,ox,oy)); g=g.set_crs(epsg,allow_override=True)
    return g

def josauder(repo, name, out, scale=100.0, seed=0):
    pkg=f"{repo}/procedural_city_generation"; sys.path.insert(0,repo); os.chdir(repo)
    import matplotlib; matplotlib.use('agg')
    from procedural_city_generation.building_generation.cuts import scale as wscale, scaletransform, walls_from_poly, randomcut
    from procedural_city_generation.building_generation.BuildingHeight import BuildingHeight
    from procedural_city_generation.additional_stuff.Singleton import Singleton
    S=Singleton("building_generation"); np.random.seed(seed)
    polylist=pickle.load(open(f"{pkg}/temp/{name}_polygons.txt",'rb'))
    # main.py pops the largest vacant polygon (the outer frame)
    idx=max(range(len(polylist)),key=lambda i:(polylist[i].poly_type=='vacant',len(polylist[i].vertices))); polylist.pop(idx)
    gb=BuildingHeight(name, S.buildingheight_image_name)
    rows=[]
    for poly in polylist:
        if poly.poly_type!='lot': continue
        center=sum(poly.vertices)/len(poly.vertices); h=gb.getBuildingHeight(center)
        house=h<S.house_height
        walls=walls_from_poly(poly)
        f=np.random.uniform(S.scalefactor_min_house,S.scalefactor_max_house) if house else np.random.uniform(S.scalefactor_min_not_house,S.scalefactor_max_not_house)
        walls=scaletransform(walls,f) if not poly.is_convex else wscale(walls,f)
        try: walls=randomcut(walls,house)
        except Exception: pass
        v=np.asarray(walls.vertices)[:,:2]*scale
        P=Polygon(v)
        if not P.is_valid: P=P.buffer(0)
        if P.is_empty or P.area<=0: continue
        if isinstance(P,MultiPolygon): P=max(P.geoms,key=lambda g:g.area)
        rows.append({'merged_id':len(rows),'Height':float(h*scale),'Function':'Residence' if house else 'Other','Age':'NA','Quality':'M','lot_area':float(poly.area*scale*scale),'geometry':P})
    b=gpd.GeoDataFrame(rows,geometry='geometry')
    # roads: Vertex graph -> segments (major / minor)
    verts=pickle.load(open(f"{pkg}/temp/{name}",'rb')); seen=set(); segs=[]; pos={id(v):i for i,v in enumerate(verts)}
    for i,v in enumerate(verts):
        for n in v.neighbours:
            j=pos.get(id(n))
            if j is None: continue
            key=(min(i,j),max(i,j))
            if key in seen: continue
            seen.add(key)
            minor=bool(getattr(v,'minor_road',False) and getattr(n,'minor_road',False))
            segs.append({'fclass':'residential' if minor else 'secondary','geometry':LineString([np.asarray(v.coords)[:2]*scale,np.asarray(n.coords)[:2]*scale])})
    r=gpd.GeoDataFrame(segs,geometry='geometry')
    os.makedirs(out,exist_ok=True)
    _to_crs(b).to_file(f"{out}/building.shp",encoding='utf-8'); _to_crs(r).to_file(f"{out}/roads.shp",encoding='utf-8')
    _meta(out,source='josauder/procedural_city_generation',name=name,scale_m_per_unit=scale,seed=seed,n_buildings=len(b),n_road_segments=len(r),
          note='no land-use layer: generator has none; heights = population-density image × scale; house/non-house from house_height')
    print(f"[josauder] {len(b)} buildings (height p50 {b.Height.median():.1f} m, area p50 {b.geometry.area.median():.0f} m²), {len(r)} road segments -> {out}")

def citydreamer(layout_dir, out=None, mpp=3.66, min_px=4, simplify_px=0.7, road_simplify_px=1.5):
    import cv2
    from skimage.morphology import skeletonize
    from shapely.ops import linemerge
    out=out or layout_dir; seg=np.load(f"{layout_dir}/seg.npy"); hf=np.load(f"{layout_dir}/hf.npy"); H,W=seg.shape
    # buildings: 4-connected components (as in CityDreamer dataset_generator.get_instance_seg_map), outlines -> polygons; y axis flipped to north-up
    bld=(seg==2).astype(np.uint8); n,lab=cv2.connectedComponents(bld,connectivity=4); rows=[]
    for k in range(1,n):
        m=(lab==k).astype(np.uint8)
        if m.sum()<min_px: continue
        cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); c=max(cs,key=cv2.contourArea)
        if len(c)<3: continue
        pts=c[:,0,:].astype(float); pts[:,1]=H-pts[:,1]; P=Polygon(pts*mpp).buffer(0)
        if P.is_empty: continue
        if isinstance(P,MultiPolygon): P=max(P.geoms,key=lambda g:g.area)
        P=P.simplify(simplify_px*mpp)
        if P.is_empty or P.area<=0: continue
        h=float(np.mean(hf[lab==k])); rows.append({'merged_id':k,'Height':max(h,3.0),'Function':'Other','Age':'NA','Quality':'M','geometry':P})
    b=gpd.GeoDataFrame(rows,geometry='geometry')
    # streets: mask skeleton -> pixel segments -> merged -> simplified (removes the 0/45/90 degree pixel-direction bias)
    sk=skeletonize(seg==1); ys,xs=np.nonzero(sk); S=set(zip(ys.tolist(),xs.tolist())); segs=[]
    for y,x in S:
        for dy,dx in ((0,1),(1,0),(1,1),(1,-1)):
            if (y+dy,x+dx) in S: segs.append(LineString([(x*mpp,(H-y)*mpp),((x+dx)*mpp,(H-y-dy)*mpp)]))
    merged=linemerge(unary_union(segs)) if segs else None
    lines=list(merged.geoms) if merged is not None and merged.geom_type=='MultiLineString' else ([merged] if merged is not None else [])
    r=gpd.GeoDataFrame({'fclass':['residential']*len(lines)},geometry=[l.simplify(road_simplify_px*mpp) for l in lines])
    os.makedirs(out,exist_ok=True)
    _to_crs(b).to_file(f"{out}/building.shp",encoding='utf-8'); _to_crs(r).to_file(f"{out}/roads.shp",encoding='utf-8')
    lm={}
    try: lm=json.load(open(f"{layout_dir}/layout_meta.json"))
    except Exception: pass
    _meta(out,source='CityDreamer Unbounded Layout Generator (Xie et al. 2024)',m_per_px=mpp,size_px=int(H),n_buildings=len(b),n_road_lines=len(r),layout=lm,
          note='no land-use layer: generator classes are road/building/green/construction only; heights = raw generated height field (m), no 50–150 m smoothing')
    print(f"[citydreamer] {len(b)} buildings (height p50 {b.Height.median():.1f} m, area p50 {b.geometry.area.median():.0f} m²), {len(r)} road lines ({r.geometry.length.sum()/1000:.1f} km) -> {out}")

def mapgenerator(raw, out, scale=5.0, seed=0):
    d=json.load(open(raw)); rng=np.random.RandomState(seed); rows=[]
    for k,lot in enumerate(d['lots']):
        if len(lot)<3: continue
        P=Polygon([(x*scale,-y*scale) for x,y in lot]).buffer(0)
        if P.is_empty: continue
        if isinstance(P,MultiPolygon): P=max(P.geoms,key=lambda g:g.area)
        rows.append({'merged_id':k,'Height':float(rng.uniform(20,40)),'Function':'Other','Age':'NA','Quality':'M','geometry':P})
    b=gpd.GeoDataFrame(rows,geometry='geometry'); segs=[]
    for key,fc in (('main','primary'),('major','secondary'),('minor','residential')):
        for line in d[key]:
            if len(line)>=2: segs.append({'fclass':fc,'geometry':LineString([(x*scale,-y*scale) for x,y in line])})
    r=gpd.GeoDataFrame(segs,geometry='geometry')
    # world origin is top-left of the screen; shift so the city centroid is 0, then apply the UTM origin
    cx,cy=b.geometry.centroid.x.mean(),b.geometry.centroid.y.mean()
    from shapely.affinity import translate
    b['geometry']=b.geometry.apply(lambda g: translate(g,-cx,-cy)); r['geometry']=r.geometry.apply(lambda g: translate(g,-cx,-cy))
    os.makedirs(out,exist_ok=True)
    _to_crs(b).to_file(f"{out}/building.shp",encoding='utf-8'); _to_crs(r).to_file(f"{out}/roads.shp",encoding='utf-8')
    _meta(out,source='ProbableTrain/MapGenerator (tensor-field roads, block shrink + subdivision lots)',scale_m_per_unit=scale,seed=d.get('seed',seed),
          n_buildings=len(b),n_road_lines=len(r),params=d.get('params'),
          note='no land-use layer; building footprints = generator lots; heights = generator rule uniform(20,40) imported as metres (decorative, spatially random)')
    print(f"[mapgen] {len(b)} buildings (area p50 {b.geometry.area.median():.0f} m²), {len(r)} road lines ({r.geometry.length.sum()/1000:.1f} km) -> {out}")

DRL_MAP={4:'\u5c45\u4f4f\u7528\u5730',5:'\u5546\u670d\u7528\u5730',6:'\u5546\u670d\u7528\u5730',7:'\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730',8:'\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730',9:'\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730',
         10:'\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730',11:'\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730',12:'\u516c\u5171\u7ba1\u7406\u4e0e\u516c\u5171\u670d\u52a1\u8bbe\u65bd\u7528\u5730'}
def drlplan(plan, out, label=None):
    import pickle
    if plan.endswith('.geojson'): g=gpd.read_file(plan)
    else:
        sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(plan))))))
        import types, pandas as pd
        if 'pandas.core.indexes.numeric' not in sys.modules:
            m=types.ModuleType('pandas.core.indexes.numeric'); m.Int64Index=m.Float64Index=m.UInt64Index=m.NumericIndex=pd.Index; sys.modules['pandas.core.indexes.numeric']=m
        P=pickle.load(open(plan,'rb')); g=(P[0] if isinstance(P,list) else P)['gdf']
    g=g.set_crs(None,allow_override=True) if g.crs else g
    poly=g[g.geom_type.isin(['Polygon','MultiPolygon'])].copy(); lines=g[g.geom_type=='LineString'].copy()
    feas=poly[poly['type']==1]; used=poly[poly['type'].isin(DRL_MAP)].copy()
    used['Level1_cn']=used['type'].map(DRL_MAP)
    lu=gpd.GeoDataFrame({'Level1_cn':used['Level1_cn'].values,'drl_type':used['type'].values},geometry=used.geometry.values)
    r=gpd.GeoDataFrame({'fclass':['residential']*len(lines)},geometry=lines.geometry.values)
    os.makedirs(out,exist_ok=True)
    _to_crs(lu).to_file(f"{out}/landuse.shp",encoding='utf-8'); _to_crs(r).to_file(f"{out}/roads.shp",encoding='utf-8')
    _meta(out,source='DRL-urban-planning (Zheng et al. 2023, Nat Comput Sci)',plan=os.path.basename(plan),label=label,n_parcels=len(lu),n_feasible_dropped=int(len(feas)),
          feasible_area_share=round(float(feas.geometry.area.sum()/poly.geometry.area.sum()),3) if len(poly) else None,n_road_lines=len(r),
          use_share={k:round(float(v),3) for k,v in (lu.geometry.area.groupby(lu['Level1_cn']).sum()/lu.geometry.area.sum()).items()},
          note='no building layer (community land-use + road plan); type codes mapped to EULUC Level-1; feasible (unplanned) parcels dropped')
    print(f"[drlplan] {label or plan}: {len(lu)} parcels ({dict(lu['Level1_cn'].value_counts())}), feasible dropped {len(feas)}, {len(r)} road lines -> {out}")

def landuse_from_blocks(folder, mode='blocks', seed=0, p=None):
    """Add a land-use layer to a generated city that has none: polygonize the streets into blocks and assign one of the 5 uses at random (by national area share p).
    Not a generator output; used only to show that random land use is flagged as unlike a city on L3/L6."""
    r=gpd.read_file(f"{folder}/roads.shp"); rng=np.random.RandomState(seed)
    polys=[q for q in polygonize(unary_union(r.geometry.values)) if 200<q.area<2e6]
    p=p or [0.45,0.2,0.2,0.1,0.05]
    lu=gpd.GeoDataFrame({'Level1_cn':rng.choice(USES,len(polys),p=p)},geometry=polys,crs=r.crs)
    lu.to_file(f"{folder}/landuse.shp",encoding='utf-8'); print(f"[landuse] {len(lu)} blocks with random use -> {folder}/landuse.shp")

def main():
    ap=argparse.ArgumentParser(); s=ap.add_subparsers(dest='cmd',required=True)
    q=s.add_parser('josauder'); q.add_argument('--repo',required=True); q.add_argument('--name',default='mycity'); q.add_argument('--out',required=True)
    q.add_argument('--scale',type=float,default=100.0,help='metres per generator unit'); q.add_argument('--seed',type=int,default=0)
    q=s.add_parser('citydreamer'); q.add_argument('--layout',required=True); q.add_argument('--out',default=None); q.add_argument('--mpp',type=float,default=3.66,help='metres per pixel')
    q=s.add_parser('mapgenerator'); q.add_argument('--raw',required=True); q.add_argument('--out',required=True); q.add_argument('--scale',type=float,default=5.0); q.add_argument('--seed',type=int,default=0)
    q=s.add_parser('drlplan'); q.add_argument('--plan',required=True); q.add_argument('--out',required=True); q.add_argument('--label',default=None)
    q=s.add_parser('landuse'); q.add_argument('--folder',required=True); q.add_argument('--mode',default='blocks'); q.add_argument('--seed',type=int,default=0)
    a=ap.parse_args()
    if a.cmd=='josauder': josauder(os.path.abspath(a.repo),a.name,os.path.abspath(a.out),a.scale,a.seed)
    elif a.cmd=='citydreamer': citydreamer(os.path.abspath(a.layout),os.path.abspath(a.out) if a.out else None,a.mpp)
    elif a.cmd=='mapgenerator': mapgenerator(os.path.abspath(a.raw),os.path.abspath(a.out),a.scale,a.seed)
    elif a.cmd=='drlplan': drlplan(os.path.abspath(a.plan),os.path.abspath(a.out),a.label)
    else: landuse_from_blocks(os.path.abspath(a.folder),a.mode,a.seed)
if __name__=='__main__': main()
