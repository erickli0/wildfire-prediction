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
4. `src/build_feature_table.py` — grids all rasters + weather + historical
   fire ignition points into one row-per-(cell, date) CSV, streamed to
   disk one weather-date at a time so memory use stays roughly
   O(grid cells) regardless of the raster size or date range. Also derives
   simple dryness features (trailing precip sum, days-since-rain) from the
   weather time series.
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
earthengine authenticate   # one-time
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
