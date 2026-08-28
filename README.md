# Wildfire Risk Prediction Model

Predicts high-risk wildfire ignition zones from Landsat-8 NDVI, USGS terrain
data (elevation/slope/aspect), and NOAA/GRIDMET weather, using a tuned
Random Forest classifier.

## Pipeline

1. `src/ingest_landsat.py` — pulls Landsat-8 imagery via Google Earth
   Engine, computes NDVI, exports a raster composite. Splits a multi-month
   pull into one export task per calendar month by default, so a large
   AOI/date-range request doesn't hit GEE's per-task size limits
   (`--single-export` to opt out for a short range).
2. `src/ingest_dem.py` — computes elevation/slope/aspect, either via GEE
   (`--mode gee`) or locally from a downloaded DEM with rasterio
   (`--mode local`). Local mode streams the DEM in row-strips instead of
   loading it into one array, so it scales to DEMs larger than available
   RAM.
3. `src/ingest_weather.py` — pulls GRIDMET weather variables (temp,
   humidity, wind, precip) via GEE.
4. `src/build_feature_table.py` — grids rasters + weather + historical fire
   ignition points into one row-per-(cell, date) CSV, streamed to disk one
   weather-date at a time so memory use stays roughly O(grid cells)
   regardless of the raster size or date range. Also derives simple
   dryness features (trailing precip sum, days-since-rain) from the
   weather time series. Two ways to run it:
   - `--mode local`: reads NDVI/elevation/slope/aspect from rasters
     already on disk (e.g. downloaded from steps 1-2's Drive exports).
   - `--mode gee` + `--mode finalize`: never downloads a raster at all.
     Tiles the AOI into a grid with `ee.Geometry.coveringGrid`, builds the
     NDVI composite and terrain stack as in-memory `ee.Image`s (reusing
     `build_ndvi_composite`/`build_terrain_stack` from steps 1-2 directly,
     without exporting them), and reduces both to per-cell means with
     `reduceRegions` plus a server-side spatial join (`ee.Join.saveAll`)
     for fire ignitions — all inside Earth Engine. The only thing that
     comes back is one row per grid cell (`--sync` for a small grid, or an
     async Drive export otherwise), which `--mode finalize` then joins
     locally with `weather.csv` — a cheap join over a small table, not a
     raster operation.
5. `src/train_model.py` — trains/tunes a RandomForestClassifier
   (RandomizedSearchCV over an expanded parameter grid), reports both a
   spatially/temporally grouped held-out ROC-AUC and a grouped k-fold
   cross-validated ROC-AUC, saves the model plus a `*_metrics.json`
   report (hyperparameters, both AUC estimates, feature importances) for
   reproducibility.
6. `qgis/render_risk_map.py` — loads model predictions into QGIS and
   exports a styled risk heatmap PNG.

Whatever ROC-AUC ends up in `*_metrics.json` for a given run is what that
run actually measured on that data — there's no fixed number baked into
the pipeline, so it'll vary with your AOI, date range, and label quality.

## Setup

```bash
pip install -r requirements.txt
python -c "import ee; ee.Authenticate()"   # one-time browser login
```

Every script that talks to GEE (`ingest_landsat.py`, `ingest_dem.py --mode
gee`, `ingest_weather.py`, `build_feature_table.py --mode gee`) takes a
required `--project <your-earthengine-cloud-project-id>` — `ee.Initialize()`
with no project id fails outright on current `earthengine-api`. Get a
project id by registering at https://code.earthengine.google.com/register.

## Building the feature table without downloading any imagery

```bash
# 1. Grid the AOI + reduce NDVI/terrain to per-cell means + join fire
#    ignitions, entirely inside Earth Engine. For a small AOI/grid, --sync
#    fetches the result directly; drop it to get an async Drive export
#    instead (needed once the grid gets too large for a synchronous pull).
python src/build_feature_table.py --mode gee \
    --region assets/study_area.geojson --start 2025-06-01 --end 2025-09-01 \
    --cell-size-m 250 --fire-labels data/raw/fire_labels.geojson \
    --project your-earthengine-cloud-project-id \
    --out data/processed/cells.csv --sync

# 2. Join that small per-cell table with weather.csv locally -- this step
#    only ever touches rows-per-grid-cell-sized data, not rasters.
python src/build_feature_table.py --mode finalize \
    --cells data/processed/cells.csv --weather data/processed/weather.csv \
    --out data/processed/feature_table.csv
```

## Where QGIS fits in

You don't need QGIS for the ML part — that's all GEE/rasterio/scikit-learn —
but it's genuinely useful in three places, and worth mentioning on a resume
alongside the model itself:

- **Terrain QA / alternative to code**: QGIS's Raster ▸ Analysis ▸
  Slope/Aspect tools do the same thing as `ingest_dem.py --mode local`, if
  you'd rather compute it visually and inspect it interactively.
- **Data QA**: load your NDVI, DEM, and weather rasters as layers in QGIS
  before training, to visually sanity-check alignment, missing data, and
  outliers — much faster than digging through numpy arrays.
- **Final risk map**: `qgis/render_risk_map.py` takes your model's
  `predict_proba` output and renders it as a proper graduated-color risk
  map (with basemap, CRS handling, legend) — the kind of static map image
  that goes well in a report or portfolio, rather than a bare matplotlib
  scatter plot.

You can run QGIS steps either in the desktop GUI or headlessly via
`qgis_process`, and PyQGIS scripts run the same way in either case.

## Suggested resume-worthy extensions

- Try `XGBoost`/`LightGBM` for comparison against the Random Forest.
- Wrap the trained model in a small Flask/FastAPI endpoint for a live demo.
