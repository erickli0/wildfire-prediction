"""
build_feature_table.py

Joins NDVI, terrain (elevation/slope/aspect), weather, and fire-ignition
labels into a feature table ready for scikit-learn. Three modes:

  --mode local (default)
    Reads NDVI/elevation/slope/aspect from local rasters (e.g. downloaded
    from a prior GEE export or QGIS), tiles them into a grid, and joins
    weather + fire labels locally. This is the original approach: it
    requires the source rasters on disk.

  --mode gee
    Never downloads a raster at all. Tiles the AOI into a grid entirely
    inside Earth Engine (`ee.Geometry.coveringGrid`), builds the NDVI
    composite and elevation/slope/aspect stack as in-memory ee.Image
    objects (reusing ingest_landsat.build_ndvi_composite and
    ingest_dem.build_terrain_stack), and reduces them to per-cell means
    with `reduceRegions` -- entirely server-side. Fire ignitions are
    joined onto the same grid server-side too (`ee.Join.saveAll`). The
    only thing that leaves Earth Engine is one row per grid cell (not per
    pixel), either fetched directly for a small grid (--sync) or exported
    as a small CSV to Drive for a large one.

  --mode finalize
    Takes the small per-cell CSV produced by --mode gee (after you've
    downloaded it, if it went to Drive) and does the same local weather
    join / temporal-dryness-feature / cell-date expansion as local mode,
    just keyed by `cell_id` instead of (row, col).

Assumes:
  - weather.csv has one row per date with regional-average weather
    variables (temp/humidity/wind/precip). If it has a 'date' column, the
    feature table is expanded to one row per (grid cell, date) so weather
    actually varies across rows -- without a 'date' column, the period
    mean is broadcast to every cell instead.
  - fire_labels.geojson (or an EE table asset, in --mode gee) has
    point/polygon ignitions with a 'date' field. When weather has dates, a
    (cell, date) row is labeled ignition=1 only if a fire occurred in that
    cell on that date. Without weather dates, a cell is labeled 1 if it
    ever had an ignition (only spatial info is available).

The (cell, date) table is streamed to `--out` one weather-date's worth of
rows at a time rather than built as a single in-memory DataFrame, so
memory use stays roughly O(grid cells) regardless of how many weather
dates or how large the source rasters/grid are.

Usage:
    # local: requires rasters on disk
    python build_feature_table.py --mode local \
        --raster-dir data/processed --weather data/processed/weather.csv \
        --fire-labels data/raw/fire_labels.geojson --grid-size 250 \
        --out data/processed/feature_table.csv

    # gee: no raster ever touches local disk
    python build_feature_table.py --mode gee \
        --region assets/study_area.geojson --start 2025-06-01 --end 2025-09-01 \
        --cell-size-m 250 --fire-labels data/raw/fire_labels.geojson \
        --out data/processed/cells.csv

    # finalize: small local join of the gee-mode output with weather
    python build_feature_table.py --mode finalize \
        --cells data/processed/cells.csv --weather data/processed/weather.csv \
        --out data/processed/feature_table.csv
"""
import argparse
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# shared: temporal features + streaming (cell, date) writer
# ---------------------------------------------------------------------------

def add_temporal_dryness_features(weather_df, precip_col="pr", window_days=7, dry_threshold_mm=1.0):
    """Derive simple fire-weather dryness signals from the regional daily
    weather series: trailing precip sum and days-since-last-rain.

    Ignition risk is highly driven by cumulative dryness rather than a
    single day's weather, so these give the model temporal signal that a
    per-day snapshot alone doesn't.
    """
    weather_df = weather_df.sort_values("date").reset_index(drop=True)
    if precip_col not in weather_df.columns:
        return weather_df

    weather_df[f"{precip_col}_{window_days}day_sum"] = (
        weather_df[precip_col].rolling(window_days, min_periods=1).sum()
    )

    rained = weather_df[precip_col] > dry_threshold_mm
    last_rain_idx = pd.Series(np.where(rained, weather_df.index, np.nan)).ffill()
    weather_df["days_since_rain"] = (weather_df.index - last_rain_idx).fillna(len(weather_df)).astype(float)

    return weather_df


def stream_cell_date_table(base_df, key_fn, ignitions, has_date, weather_path, out_path,
                            required_cols=("ndvi", "elevation", "slope", "aspect")):
    """Cross-join `base_df` (one row per cell) against the weather time
    series and stream the result to `out_path` one weather-date at a time,
    instead of materializing every (cell, date) row in memory at once.

    `key_fn(chunk)` returns, for each row of a per-date chunk, the key used
    to look up that cell's ignition info in `ignitions`.

    Returns (total_rows, ignition_rows).
    """
    weather_df = pd.read_csv(weather_path)
    weather_has_date = "date" in weather_df.columns

    total_rows = 0
    ignition_rows = 0
    wrote_header = False

    def emit(chunk):
        nonlocal total_rows, ignition_rows, wrote_header
        chunk = chunk.dropna(subset=list(required_cols))
        if chunk.empty:
            return
        chunk.to_csv(out_path, mode="a" if wrote_header else "w", header=not wrote_header, index=False)
        wrote_header = True
        total_rows += len(chunk)
        ignition_rows += int(chunk["ignition"].sum())

    if weather_has_date:
        weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.normalize()
        weather_df = add_temporal_dryness_features(weather_df)

        for _, weather_row in weather_df.iterrows():
            chunk = base_df.copy()
            weather_date = weather_row["date"]
            for col, val in weather_row.items():
                if col != "date":
                    chunk[col] = val
            chunk["date"] = weather_date
            keys = key_fn(chunk)
            if has_date:
                chunk["ignition"] = [int(weather_date in ignitions.get(k, set())) for k in keys]
            else:
                chunk["ignition"] = [int(bool(ignitions.get(k, False))) for k in keys]
            emit(chunk)
    else:
        means = weather_df.mean(numeric_only=True)
        chunk = base_df.copy()
        for col, val in means.items():
            chunk[col] = val
        keys = key_fn(chunk)
        chunk["ignition"] = [int(bool(ignitions.get(k, False))) for k in keys]
        emit(chunk)

    return total_rows, ignition_rows


# ---------------------------------------------------------------------------
# --mode local: read rasters from disk
# ---------------------------------------------------------------------------

def rasters_to_grid_df(raster_paths: dict, grid_size: int):
    """Read each raster (opened once, with windowed per-cell reads so
    memory use stays bounded regardless of raster size) and return one row
    per grid cell with columns = raster names + row/col + lon/lat.
    """
    import rasterio

    srcs = {name: rasterio.open(path) for name, path in raster_paths.items()}
    try:
        first = next(iter(srcs.values()))
        transform = first.transform
        height, width = first.height, first.width
        crs = first.crs

        rows = []
        for r0 in range(0, height, grid_size):
            for c0 in range(0, width, grid_size):
                window = rasterio.windows.Window(
                    c0, r0, min(grid_size, width - c0), min(grid_size, height - r0)
                )
                cell = {"row": r0, "col": c0}
                for name, src in srcs.items():
                    block = src.read(1, window=window)
                    valid = block[block != src.nodata] if src.nodata is not None else block
                    cell[name] = float(np.nanmean(valid)) if valid.size else np.nan
                x, y = rasterio.transform.xy(transform, r0 + grid_size // 2, c0 + grid_size // 2)
                cell["lon"], cell["lat"] = x, y
                rows.append(cell)
    finally:
        for src in srcs.values():
            src.close()

    return pd.DataFrame(rows), crs


def find_cell_ignition_dates(grid_df, fire_labels_path, crs, grid_size, transform_res=30):
    """For each unique grid cell, find the set of dates (normalized to
    midnight) on which a fire ignition fell inside that cell.

    Returns (ignitions, has_date):
      - ignitions: {(row, col): set(Timestamp)} if fire labels have a
        'date' column, else {(row, col): True} for any cell that ever had
        an ignition.
      - has_date: whether fire labels carried per-ignition dates.

    Runs once per unique cell (not per cell-date row), so it stays cheap
    even once the output table is expanded across weather dates.
    """
    import geopandas as gpd

    fires = gpd.read_file(fire_labels_path).to_crs(crs)
    has_date = "date" in fires.columns
    if has_date:
        fires["date"] = pd.to_datetime(fires["date"]).dt.normalize()

    cell_deg = grid_size * transform_res / 111_000  # rough deg-per-cell at this scale
    ignitions = {}
    for _, cell in grid_df.iterrows():
        nearby = fires.cx[
            cell.lon - cell_deg / 2 : cell.lon + cell_deg / 2,
            cell.lat - cell_deg / 2 : cell.lat + cell_deg / 2,
        ]
        if len(nearby) == 0:
            continue
        ignitions[(cell.row, cell.col)] = set(nearby["date"]) if has_date else True

    return ignitions, has_date


def run_local(args):
    raster_paths = {
        "ndvi": f"{args.raster_dir}/ndvi.tif",
        "elevation": f"{args.raster_dir}/elevation.tif",
        "slope": f"{args.raster_dir}/slope.tif",
        "aspect": f"{args.raster_dir}/aspect.tif",
    }

    input_bytes = sum(os.path.getsize(p) for p in raster_paths.values() if os.path.exists(p))
    input_bytes += os.path.getsize(args.weather) if os.path.exists(args.weather) else 0
    print(f"Reading {input_bytes / 1e9:.2f} GB of input rasters + weather")

    grid_df, crs = rasters_to_grid_df(raster_paths, args.grid_size)
    ignitions, has_date = find_cell_ignition_dates(grid_df, args.fire_labels, crs, args.grid_size)

    total_rows, ignition_rows = stream_cell_date_table(
        grid_df, lambda df: list(zip(df["row"], df["col"])), ignitions, has_date, args.weather, args.out,
    )

    print(f"Wrote {total_rows} rows to {args.out}")
    if total_rows:
        print(f"Ignition rate: {ignition_rows / total_rows:.4f}")


# ---------------------------------------------------------------------------
# --mode gee: grid + reduce + ignition-join, all server-side
# ---------------------------------------------------------------------------

def build_grid_fc(region, cell_size_m):
    """Tile `region` into a grid of square cells at `cell_size_m` resolution,
    entirely server-side -- no raster ever leaves Earth Engine for this
    step, only the resulting cell polygons (and later, their aggregates).
    """
    import ee

    grid = region.coveringGrid(ee.Projection("EPSG:4326"), cell_size_m)

    def add_centroid(f):
        coords = f.geometry().centroid(1).coordinates()
        return f.set({"cell_id": f.get("system:index"), "lon": coords.get(0), "lat": coords.get(1)})

    return grid.map(add_centroid)


def reduce_rasters_to_grid_gee(region, start, end, grid_fc, dem_asset="USGS/3DEP/10m", scale=30):
    """Compute per-cell mean NDVI/elevation/slope/aspect with reduceRegions,
    server-side, over the whole AOI at once. The composite images backing
    this are never exported or downloaded -- only their per-cell means are.
    """
    import ee
    from ingest_dem import build_terrain_stack
    from ingest_landsat import build_ndvi_composite

    ndvi = build_ndvi_composite(region, start, end).rename("ndvi")
    terrain = build_terrain_stack(region, dem_asset=dem_asset)
    combined = terrain.addBands(ndvi)

    return combined.reduceRegions(collection=grid_fc, reducer=ee.Reducer.mean(), scale=scale)


def attach_ignition_dates_gee(grid_reduced_fc, fire_labels, region):
    """Spatially join fire ignitions onto grid cells, server-side.

    `fire_labels` is either a local GeoJSON path (uploaded via geemap, same
    as local mode -- fire-label vector files are tiny, unlike imagery) or
    an existing EE table asset id. Adds an 'ignition_dates' property: a
    comma-joined string of ignition dates for cells with a fire, empty
    otherwise. If the fire labels have no 'date' property, adds
    'has_ignition' (0/1) instead.

    Returns (joined_fc, has_date).
    """
    import ee
    import geemap

    if fire_labels.endswith((".geojson", ".json")):
        fires_fc = geemap.geojson_to_ee(fire_labels)
    else:
        fires_fc = ee.FeatureCollection(fire_labels)
    fires_fc = fires_fc.filterBounds(region)

    has_date = "date" in fires_fc.first().propertyNames().getInfo()

    joined = ee.Join.saveAll(matchesKey="fire_matches").apply(
        primary=grid_reduced_fc,
        secondary=fires_fc,
        condition=ee.Filter.intersects(leftField=".geo", rightField=".geo", maxError=1),
    )

    def attach(f):
        matches = ee.List(f.get("fire_matches"))
        if has_date:
            dates = matches.map(lambda m: ee.Feature(m).get("date"))
            return f.set("ignition_dates", ee.List(dates).join(","))
        return f.set("has_ignition", matches.size().gt(0))

    return ee.FeatureCollection(joined).map(attach), has_date


def export_grid_table(fc, out_path, keep_props, sync):
    """Hand back the small per-cell aggregate table -- one row per grid
    cell, not per pixel. `sync` fetches it directly (fine for a modest
    grid; convenient for quick iteration, but subject to Earth Engine's
    interactive request limits). Otherwise, starts an async export task
    to Drive, matching the pattern the rest of the pipeline already uses
    for large outputs.
    """
    if sync:
        import geemap

        df = geemap.ee_to_df(fc, columns=keep_props)
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df)} grid cells to {out_path}")
    else:
        import ee

        description = os.path.splitext(os.path.basename(out_path))[0]
        task = ee.batch.Export.table.toDrive(
            collection=fc,
            description=description,
            folder="wildfire_risk",
            fileNamePrefix=description,
            fileFormat="CSV",
            selectors=keep_props,
        )
        task.start()
        print(f"Started export task: {description} (check GEE Tasks tab)")
        print(f"Once it finishes, download the CSV and run --mode finalize on it "
              f"to join it with weather into the final feature table.")


def run_gee(args):
    import ee
    import geemap

    ee.Initialize()
    region = geemap.geojson_to_ee(args.region).geometry()

    grid_fc = build_grid_fc(region, args.cell_size_m)
    reduced_fc = reduce_rasters_to_grid_gee(
        region, args.start, args.end, grid_fc, dem_asset=args.dem_asset, scale=args.scale,
    )
    joined_fc, has_date = attach_ignition_dates_gee(reduced_fc, args.fire_labels, region)

    keep_props = ["cell_id", "lon", "lat", "ndvi", "elevation", "slope", "aspect"]
    keep_props.append("ignition_dates" if has_date else "has_ignition")

    export_grid_table(joined_fc, args.out, keep_props, args.sync)


# ---------------------------------------------------------------------------
# --mode finalize: join the small gee-mode cell table with weather locally
# ---------------------------------------------------------------------------

def parse_ignition_dates(value):
    if pd.isna(value) or value == "":
        return set()
    return set(pd.to_datetime(str(value).split(",")).normalize())


def run_finalize(args):
    cells_df = pd.read_csv(args.cells)
    has_date = "ignition_dates" in cells_df.columns

    if has_date:
        ignitions = {cid: parse_ignition_dates(v) for cid, v in zip(cells_df["cell_id"], cells_df["ignition_dates"])}
    else:
        ignitions = {cid: bool(v) for cid, v in zip(cells_df["cell_id"], cells_df.get("has_ignition", False))}

    total_rows, ignition_rows = stream_cell_date_table(
        cells_df, lambda df: list(df["cell_id"]), ignitions, has_date, args.weather, args.out,
    )

    print(f"Wrote {total_rows} rows to {args.out}")
    if total_rows:
        print(f"Ignition rate: {ignition_rows / total_rows:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "gee", "finalize"], default="local")

    # local mode
    parser.add_argument("--raster-dir", help="Directory with ndvi/elevation/slope/aspect .tif (local mode)")
    parser.add_argument("--grid-size", type=int, default=250, help="Grid cell size in pixels (local mode)")

    # gee mode
    parser.add_argument("--region", help="GeoJSON AOI (gee mode)")
    parser.add_argument("--start", help="NDVI composite start date, YYYY-MM-DD (gee mode)")
    parser.add_argument("--end", help="NDVI composite end date, YYYY-MM-DD (gee mode)")
    parser.add_argument("--cell-size-m", type=int, default=250, help="Grid cell size in meters (gee mode)")
    parser.add_argument("--scale", type=int, default=30, help="reduceRegions scale in meters (gee mode)")
    parser.add_argument("--dem-asset", default="USGS/3DEP/10m", help="EE DEM image asset id (gee mode)")
    parser.add_argument("--sync", action="store_true",
                         help="Fetch the per-cell table directly instead of exporting to Drive "
                              "(gee mode; only for small grids -- subject to EE interactive limits)")

    # finalize mode
    parser.add_argument("--cells", help="Per-cell CSV produced by --mode gee (finalize mode)")

    # shared
    parser.add_argument("--weather", help="Weather CSV (local + finalize modes)")
    parser.add_argument("--fire-labels", help="GeoJSON path or EE table asset id (local + gee modes)")
    parser.add_argument("--out", default="feature_table.csv")
    args = parser.parse_args()

    if args.mode == "local":
        run_local(args)
    elif args.mode == "gee":
        run_gee(args)
    else:
        run_finalize(args)


if __name__ == "__main__":
    main()
