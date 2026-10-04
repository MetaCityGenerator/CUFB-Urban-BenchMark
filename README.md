# CUFB-China

Code for **CUFB-China (China Urban-Form Benchmark)**: county-level urban form metrics and cross-county reference bounds for China, and a protocol for comparing real or generated cities with those bounds.

- Data descriptor: *County-level urban form metrics and reference bounds for real and generated cities in China*, Scientific Data ([TODO: DOI]).
- Data record (all released tables, starter county, benchmark cases, validation outputs): Figshare, [TODO: DOI].
- This repository contains the code only. Software archive: Zenodo, [TODO: DOI].

The pipeline starts from three open nationwide layers (CMAB building footprints with model-estimated height and construction era, EULUC-China 2.0 land-use parcels, OpenStreetMap streets) and CTAmap county-level boundaries. It splits them per county, reduces every county to 17 metric blocks, aggregates the counties with at least 1,000 buildings into national and stratified p5/p50/p95 bounds with province-level bootstrap intervals, and scores any county-format folder against those bounds.

## Repository layout

```
code/urbanBenchmark/
  split_county.py      split the nationwide layers into one folder per county
  cmab_ingest.py       place CMAB province archives into the input tree
  euluc2_ingest.py     split EULUC-China 2.0 (gdb) into county landuse.shp
  benchmark.py         analyze (17 metric blocks per county) / runall / aggregate (national and stratified bounds)
  spectrum.py          fixed-bin morphology spectra, national reference spectrum, per-county deviation
  score.py             conformity protocol (v2 default, v1 for comparison), batch scoring, AUC with county-paired bootstrap CI
  export.py            county_metrics.csv, county_relations.csv, county_spectra.nc, benchmark_results.csv, manifest
  archetypes.py        Louvain-community composition vectors and national archetypes (descriptive)
  release_tables.py    English headers and land-use codes for the released copies of the tables
  perturb.py           controlled perturbations for the benchmark test
  validate.py          technical validation: cross-implementation, sensitivity, subsampling, height and label noise
  gen_import.py        convert third-party generator outputs into the county input format
  gen_citydreamer.py   run the CityDreamer layout generator
  gen_mapgenerator.py  run ProbableTrain/MapGenerator headlessly
  fig_paper.py         manuscript figures 1-7
  fig_generated.py     generated-city figures
  run_era.sh  run_gate1.sh  run_validation.sh  run_reaggregate.sh   batch runs used for the paper
examples/run_starter.sh  minimal runnable example on the starter county of the data record
environment.yml          pinned environment used for the released tables
```

## Installation

```bash
conda env create -f environment.yml
conda activate cufb
```
Python 3.12 with geopandas 1.1, shapely 2.1, pyogrio, scipy, scikit-learn, networkx and matplotlib. `validate.py crosscheck` additionally needs momepy and osmnx; the generator converters need their own dependencies (see the docstrings).

## Quick start (about one minute)

Download `02_reference_bounds.zip` and `05_starter.zip` from the Figshare record, unzip both into `data/`, then run from the repository root:

```bash
bash examples/run_starter.sh
```
This analyses the starter county (1,987 buildings), scores it against the national reference and checks that the metric values reproduce the released `result.json`. The two underlying commands are

```bash
python code/urbanBenchmark/benchmark.py analyze --input data/05_starter/<county> --out starter_out
python code/urbanBenchmark/score.py --city starter_out --ref data/02_reference_bounds
```

## Scoring your own city

Write the city as one folder with
- `building.shp`: footprint polygons with a `Height` field in metres;
- `landuse.shp`: parcels with a `Level1_cn` field holding the EULUC-China Level-1 class labels (the five Chinese class names used in `benchmark.USES`); `gen_import.py drlplan` shows how to map other class schemes;
- `roads.shp`: street lines with an OpenStreetMap-style `fclass` field;
- optionally `_meta.json` with provenance and scale.

Layers a generator does not produce can be omitted; the score is then computed on the available layers and the number of checks is reported. Run `benchmark.py analyze` and `score.py` as above. Report the reference and protocol version, the spatial scale and its source, the number of checks and the failed checks together with any score. `gen_import.py` contains converters for josauder/procedural_city_generation, ProbableTrain/MapGenerator, CityDreamer layouts and DRL-urban-planning plans.

Keys that refer to land uses in `result.json`, `national_bounds.csv` and the stratified bounds use the EULUC-China Level-1 labels produced by the analysis; the released `county_metrics.csv` and `county_relations.csv` use the GB 50137-2011 codes R/A/B/M/S instead. `data_dictionary.csv` in the data record maps both.

## Full reproduction

Inputs (not redistributed; see the data descriptor for sources and licences):
- `dataset/input/buildings/{province}/*.shp`: CMAB v7 per-city shapefiles (Figshare 10.6084/m9.figshare.27992417; use `cmab_ingest.py`);
- `dataset/input/landuse/`: EULUC-China 2.0 (Zenodo 10.5281/zenodo.16794007; `euluc2_ingest.py split` writes county landuse.shp directly);
- `dataset/input/roads/gis_osm_roads_free_1.shp`: Geofabrik China extract of OpenStreetMap (2024-05-29);
- `dataset/input/boundary/`: CTAmap county-level polygons, 2020 layer, with its original file name and attribute fields (province, prefecture, `NAME`, type, `PAC`); the expected file name is set in `split_county.py` and `fig_paper.py`.

```bash
B=code/urbanBenchmark
python $B/split_county.py --provinces all --layers buildings landuse roads --root dataset/input --out dataset/input_county   # hours; resumable
python $B/benchmark.py runall --root dataset/input_county --out dataset/output_county          # ~10 s per county; resumable
python $B/benchmark.py aggregate --root dataset/output_county --min-cities 20 --min-n 1000 --bootstrap 200
python $B/benchmark.py aggregate --root dataset/output_county --min-cities 20 --min-n 1000 --strata type
python $B/benchmark.py aggregate --root dataset/output_county --min-cities 20 --min-n 1000 --strata size
bash   $B/run_era.sh                                   # construction-era strata
python $B/spectrum.py reference --root dataset/output_county --min-n 1000
python $B/export.py --root dataset/output_county --input-root dataset/input_county --en
python $B/export.py --root dataset/output_county --relations
python $B/export.py --root dataset/output_county --spectra
python $B/archetypes.py --root dataset/output_county
bash   $B/run_gate1.sh 50 0                            # 50 test counties x 7 perturbations, test-excluded reference, AUC
bash   $B/run_validation.sh 50 20 20 20                # cross-check, sensitivity, subsampling
python $B/validate.py heightnoise --root dataset/input_county --counties dataset/ref_holdout/holdout.txt --n 20 --out dataset/validation
python $B/validate.py labelnoise  --root dataset/input_county --counties dataset/ref_holdout/holdout.txt --n 20 --out dataset/validation
python $B/export.py --root dataset/output_county --results
python $B/fig_paper.py --out dataset/figures
```
Analysis parameters are collected in `PARAMS` in `benchmark.py`; any of them can be overridden with `--param key=value`, and non-default values are recorded in `result.json["_params"]`. All random steps use fixed seeds.

## Citation and licence

Please cite the data descriptor, the data record and this software (see `CITATION.cff`), and the upstream sources: CMAB (Zhang, Zhao & Long 2025, Scientific Data 12, 430), EULUC-China 2.0 (Li et al. 2025, Science Bulletin 70, 3029-3041), OpenStreetMap contributors and CTAmap (Rui, Zhang & Chen 2025, Proc. SPIE 13565).

Code: MIT licence (`LICENSE`). Data record: CC BY 4.0, with redistributed OpenStreetMap geometry under ODbL 1.0.
