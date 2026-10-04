# -*- coding: utf-8 -*-
"""Split the nationwide buildings / landuse / roads layers into one folder per county.

Features:
  - chunked streaming + append writes -> constant memory (large provinces do not blow up).
  - resumable: a marker per (province x layer) in _progress/; finished province-layers are skipped;
    an unfinished province-layer first deletes that layer's county files and rescans, so appends never duplicate.
  - the layers use different CRSs (buildings 3857 / landuse 4326 / roads 4326 / county boundaries CGCS2000); handled automatically.
  - roads have no spatial index and are scanned once nationwide for all pending provinces -> pass every province you need in one call.

Usage:
  # all provinces (one command for every county)
  python split_county.py --provinces all --layers buildings landuse roads --out dataset/input_county
  # selected provinces (names as in the boundary layer)
  python split_county.py --provinces <province> [<province> ...] --out dataset/input_county
  # resume: rerun the same command; finished provinces are skipped
  # force a rerun of a layer: add --no-resume (redone and overwritten)

Output: {out}/{province}_{prefecture}_{county}/{building,landuse,roads}.shp + _meta.json

Province input can be one {province}.shp or several .shp files in the province folder (e.g. CMAB per-natural-city files processed_*_all.shp unzipped as they are;
do not merge them - merged province files exceed the 2-4 GB shapefile limit). The building layer keeps only the five BUILDING_FIELDS unless --keep-all-fields.
"""
import pyogrio, geopandas as gpd, pandas as pd, numpy as np, json, os, re, glob, time, argparse
ROOT = "dataset/input"  # default; pass --root with the absolute path to dataset/input
ROADS = f"{ROOT}/roads/gis_osm_roads_free_1.shp"
CH = 400000  # features per chunk
BUILDING_FIELDS = ['merged_id', 'Height', 'Function', 'Age', 'Quality']  # standard fields of the county building.shp (trimmed CMAB)
KEEP_ALL_FIELDS = False

def san(s): return re.sub(r'[\\/:*?"<>|]+', '_', str(s)).strip()
PROV, CITY = '\u7701', '\u5e02'  # province / prefecture field names in the boundary layer
def folder_of(r): return san(f"{r[PROV]}_{r[CITY]}_{r['NAME']}")

def mark_path(OUT, key): return f"{OUT}/_progress/{key}.done"
def is_done(OUT, key): return os.path.exists(mark_path(OUT, key))
def mark(OUT, key, info=None):
    os.makedirs(f"{OUT}/_progress", exist_ok=True)
    json.dump(info or {}, open(mark_path(OUT, key), 'w', encoding='utf-8'), ensure_ascii=False)

def clear_layer(OUT, folders, outname):
    for fol in folders:
        for f in glob.glob(f"{OUT}/{fol}/{outname}.*"):
            try: os.remove(f)
            except OSError: pass

def read_chunks(path):
    sk = 0
    while True:
        g = pyogrio.read_dataframe(path, skip_features=sk, max_features=CH)
        if len(g) == 0: break
        yield g
        if len(g) < CH: break
        sk += CH

def write_append(gdf, outp):
    pyogrio.write_dataframe(gdf, outp, append=os.path.exists(outp), encoding='utf-8')

def province_files(layer, prov):
    """Province input: {ROOT}/{layer}/{prov}/*.shp (one {prov}.shp or several per-city files)."""
    return sorted(glob.glob(f"{ROOT}/{layer}/{prov}/*.shp"))

def split_layer_province(prov, counties_prov, layer, outname, OUT):
    """Scan buildings/landuse for one province, assign features to counties by representative point, append in chunks. counties_prov is already in the target CRS."""
    paths = province_files(layer, prov)
    if not paths:
        print(f"  [{layer}] {prov}: no data, skipped"); return False
    folders = set(counties_prov['folder'])
    clear_layer(OUT, folders, outname)
    for fol in folders: os.makedirs(f"{OUT}/{fol}", exist_ok=True)
    n = 0
    for g in (chunk for path in paths for chunk in read_chunks(path)):
        g = g[g.geometry.notna() & ~g.geometry.is_empty]
        if len(g) == 0: continue
        if layer == 'buildings' and not KEEP_ALL_FIELDS:
            g = g[[c for c in BUILDING_FIELDS if c in g.columns] + ['geometry']]
        pts = gpd.GeoDataFrame(geometry=g.geometry.representative_point(), crs=g.crs)
        if str(pts.crs) != str(counties_prov.crs): pts = pts.to_crs(counties_prov.crs)
        j = gpd.sjoin(pts, counties_prov[['folder', 'geometry']], predicate='within', how='inner')
        if len(j):
            g2 = g.loc[j.index].assign(_folder=j['folder'].values)
            for fol, sub in g2.groupby('_folder'):
                write_append(sub.drop(columns='_folder'), f"{OUT}/{fol}/{outname}.shp")
        n += len(g)
    print(f"  [{layer}] {prov}: {len(paths)} files, scanned {n}", flush=True)
    return True

def split_roads(counties_todo, OUT):
    """Scan the nationwide roads once, clip to county polygons (4326), append in chunks. counties_todo: counties of pending provinces (4326)."""
    folders = set(counties_todo['folder'])
    clear_layer(OUT, folders, 'roads')
    for fol in folders: os.makedirs(f"{OUT}/{fol}", exist_ok=True)
    n = 0
    for g in read_chunks(ROADS):
        g = g[g.geometry.notna() & ~g.geometry.is_empty]
        if str(g.crs) != 'EPSG:4326': g = g.to_crs(4326)
        j = gpd.sjoin(g[['geometry']], counties_todo[['folder', 'geometry']], predicate='intersects', how='inner')
        if len(j):
            for fol, sub in j.groupby('folder'):
                gg = g.loc[sub.index].copy()
                poly = counties_todo[counties_todo.folder == fol].geometry.iloc[0]
                gg['geometry'] = gg.geometry.intersection(poly)
                gg = gg[~gg.geometry.is_empty & gg.geometry.notna()]
                if len(gg): write_append(gg, f"{OUT}/{fol}/roads.shp")
        n += len(g)
    print(f"  [roads] nationwide scan {n} -> {len(folders)} counties", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--provinces', nargs='+', required=True, help="province names, or all")
    ap.add_argument('--layers', nargs='+', default=['buildings', 'landuse', 'roads'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--root', default=None, help="absolute path to dataset/input (default: ROOT at the top of this file)")
    ap.add_argument('--no-resume', action='store_true', help="ignore existing markers and redo")
    ap.add_argument('--keep-all-fields', action='store_true', help="keep all building fields (default: BUILDING_FIELDS only)")
    a = ap.parse_args()
    global ROOT, ROADS, KEEP_ALL_FIELDS
    KEEP_ALL_FIELDS = a.keep_all_fields
    if a.root: ROOT = a.root; ROADS = f"{ROOT}/roads/gis_osm_roads_free_1.shp"
    OUT = a.out; resume = not a.no_resume
    os.makedirs(OUT, exist_ok=True)

    cty = gpd.read_file(f"{ROOT}/boundary/\u53bf.shp")
    cty = cty[cty['\u7701'].notna()].reset_index(drop=True)
    cty['folder'] = cty.apply(folder_of, axis=1)

    if a.provinces == ['all']:
        have = set(os.listdir(f"{ROOT}/buildings")) | set(os.listdir(f"{ROOT}/landuse"))
        provinces = [p for p in sorted(cty['\u7701'].unique()) if p in have]
    else:
        provinces = a.provinces
    cty = cty[cty['\u7701'].isin(provinces)].reset_index(drop=True)
    print(f"pending: {len(provinces)} provinces / {len(cty)} counties; resume={resume}", flush=True)

    # buildings / landuse: resumable per province x layer
    plan = [('buildings', 'building', 3857), ('landuse', 'landuse', 4326)]
    for layer, outname, epsg in plan:
        if layer not in a.layers: continue
        for prov in provinces:
            key = f"{layer}_{prov}"
            if resume and is_done(OUT, key):
                print(f"  [{layer}] {prov}: done, skipped"); continue
            cp = cty[cty['\u7701'] == prov].to_crs(epsg)
            t = time.time()
            ok = split_layer_province(prov, cp, layer, outname, OUT)
            mark(OUT, key, {'ok': ok, 'sec': round(time.time()-t, 1)})

    # roads: one nationwide scan for all provinces whose roads are not done
    if 'roads' in a.layers:
        todo = [p for p in provinces if not (resume and is_done(OUT, f"roads_{p}"))]
        if todo:
            cc = cty[cty['\u7701'].isin(todo)].to_crs(4326)
            t = time.time(); split_roads(cc, OUT)
            for p in todo: mark(OUT, f"roads_{p}", {'sec': round(time.time()-t, 1)})
        else:
            print("  [roads] all provinces done, skipped")

    # write _meta.json for every county
    for _, r in cty.iterrows():
        od = f"{OUT}/{r['folder']}"
        if os.path.isdir(od):
            json.dump({'PAC': str(r['PAC']), '\u7701': r['\u7701'], '\u5e02': r['\u5e02'], '\u53bf': r['NAME'], '\u7c7b\u578b': r['\u7c7b\u578b']},
                      open(f"{od}/_meta.json", 'w', encoding='utf-8'), ensure_ascii=False)
    print("DONE", flush=True)

if __name__ == '__main__':
    main()
