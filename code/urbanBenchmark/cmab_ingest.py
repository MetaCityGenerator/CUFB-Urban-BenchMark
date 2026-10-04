# -*- coding: utf-8 -*-
"""Put a CMAB province zip (Figshare 27992417) into dataset/input/buildings/{province}/ for split_county.py.

Each CMAB province zip holds one shapefile per natural city: {province}/processed_{StCity_id}_{image_date}_all.shp,
fields merged_id, Height, Function, Age, Quality, Structure, Style, EPSG:3857.
The files are not merged into one province file (large provinces exceed the 2-4 GB shapefile limit); split_county.py reads several .shp per province folder.

Usage:
  python cmab_ingest.py --zip dataset/input/_download/hebei.zip --province <province name as in the boundary layer> --root dataset/input
  # an older merged province file ({province}/{province}.shp) is moved to {province}/_old_merged/ to avoid double counting
Then:
  python split_county.py --provinces <province> --layers buildings --root dataset/input --out dataset/input_county --no-resume
"""
import os, sys, glob, shutil, zipfile, argparse, tempfile
import pyogrio

EXPECT = ['merged_id', 'Height', 'Function', 'Age', 'Quality']

def ingest(zip_path, province, root, keep_zip=True):
    dest = f"{root}/buildings/{province}"
    os.makedirs(dest, exist_ok=True)
    # move old merged province files aside
    old = [p for p in glob.glob(f"{dest}/*") if os.path.isfile(p) and os.path.basename(p).startswith(province)]
    if old:
        bk = f"{dest}/_old_merged"; os.makedirs(bk, exist_ok=True)
        for p in old: shutil.move(p, f"{bk}/{os.path.basename(p)}")
        print(f"[ingest] {len(old)} old merged file(s) -> {bk}")
    tmp = tempfile.mkdtemp(prefix='cmab_', dir=f"{root}/buildings")
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist(); z.extractall(tmp)
    shps = sorted(glob.glob(f"{tmp}/**/*.shp", recursive=True))
    if not shps: sys.exit(f"[ingest] no .shp in zip: {zip_path}")
    tot = 0; moved = 0; bad = []
    for shp in shps:
        info = pyogrio.read_info(shp); fields = list(info['fields'])
        missing = [c for c in EXPECT if c not in fields]
        if missing or str(info['crs']) != 'EPSG:3857': bad.append((os.path.basename(shp), missing, info['crs']))
        tot += info['features']
        stem = os.path.splitext(shp)[0]
        for f in glob.glob(stem + '.*'):
            shutil.move(f, f"{dest}/{os.path.basename(f)}"); moved += 1
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[ingest] {province}: {len(shps)} city files, {tot:,} buildings -> {dest}")
    if bad:
        print(f"[ingest] warning: {len(bad)} file(s) with unexpected fields/CRS:")
        for b in bad[:10]: print("   ", b)
    return tot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', required=True); ap.add_argument('--province', required=True, help="province name exactly as in the boundary layer")
    ap.add_argument('--root', required=True, help="dataset/input")
    a = ap.parse_args()
    ingest(a.zip, a.province, a.root)
    print(f"next: python split_county.py --provinces {a.province} --layers buildings --root {a.root} --out dataset/input_county --no-resume")

if __name__ == '__main__': main()
