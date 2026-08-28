# Wildfire Risk Prediction Model

Predicts high-risk wildfire ignition zones from Landsat-8 NDVI, USGS terrain
data (elevation/slope/aspect), and NOAA/GRIDMET weather, using a tuned
Random Forest classifier.

## Pipeline

1. `src/ingest_landsat.py` — pulls Landsat-8 imagery via Google Earth
   Engine, computes NDVI, exports a raster composite.
2. `src/ingest_dem.py` — computes elevation/slope/aspect, either via GEE
   (`--mode gee`) or locally from a downloaded DEM with rasterio
   (`--mode local`).
3. `src/ingest_weather.py` — pulls GRIDMET weather variables (temp,
   humidity, wind, precip) via GEE.
4. `src/build_feature_table.py` — grids all rasters + weather + historical
   fire ignition points into one row-per-cell CSV.
5. `src/train_model.py` — trains/tunes a RandomForestClassifier
   (RandomizedSearchCV), reports ROC-AUC on a spatially/temporally
   grouped held-out split, saves the model.
6. `qgis/render_risk_map.py` — loads model predictions into QGIS and
   exports a styled risk heatmap PNG.

## Setup

```bash
pip install earthengine-api geemap rasterio geopandas scikit-learn pandas numpy joblib
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

- Add a temporal feature (drought index, days-since-rain) — ignition risk
  is highly seasonal.
- Try `XGBoost`/`LightGBM` for comparison against the Random Forest.
- Wrap the trained model in a small Flask/FastAPI endpoint for a live demo.
