"""
build_feature_table.py

Joins NDVI, terrain (elevation/slope/aspect), weather, and fire-ignition
labels into a feature table ready for scikit-learn.

Assumes:
  - NDVI, elevation, slope, aspect are co-registered rasters (same grid)
  - weather.csv has one row per date with regional-average weather
    variables (temp/humidity/wind/precip). If it has a 'date' column, the
    feature table is expanded to one row per (grid cell, date) so weather
    actually varies across rows -- without a 'date' column, the period
    mean is broadcast to every cell instead.
  - fire_labels.geojson has point/polygon ignitions with a 'date' field.
    When weather has dates, a (cell, date) row is labeled ignition=1 only
    if a fire occurred in that cell on that date. Without weather dates,
    a cell is labeled 1 if it ever had an ignition (only spatial info is
    available).

The (cell, date) table is streamed to `--out` one weather-date's worth of
rows at a time rather than built as a single in-memory DataFrame, so
memory use stays roughly O(grid cells) regardless of how many weather
dates or how large the source rasters are.

Usage:
    python build_feature_table.py --raster-dir data/processed \
        --weather data/processed/weather.csv \
        --fire-labels data/raw/fire_labels.geojson \
        --grid-size 250 \
        --out data/processed/feature_table.csv
"""
import argparse
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


def rasters_to_grid_df(raster_paths: dict, grid_size: int):
    """Read each raster (opened once, with windowed per-cell reads so
    memory use stays bounded regardless of raster size) and return one row
    per grid cell with columns = raster names + row/col + lon/lat.
    """
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


def write_feature_table(grid_df, ignitions, has_date, weather_path, out_path):
    """Stream the (cell, date) feature table to `out_path` one weather
    date at a time, instead of materializing every (cell, date) row in
    memory at once (a cross join of cells x dates can get very large).

    Returns (total_rows, ignition_rows).
    """
    weather_df = pd.read_csv(weather_path)
    weather_has_date = "date" in weather_df.columns

    total_rows = 0
    ignition_rows = 0
    wrote_header = False

    def emit(chunk):
        nonlocal total_rows, ignition_rows, wrote_header
        chunk = chunk.dropna(subset=["ndvi", "elevation", "slope", "aspect"])
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
            chunk = grid_df.copy()
            weather_date = weather_row["date"]
            for col, val in weather_row.items():
                if col != "date":
                    chunk[col] = val
            chunk["date"] = weather_date
            chunk["ignition"] = [
                int(weather_date in ignitions.get((r, c), set()))
                for r, c in zip(chunk["row"], chunk["col"])
            ]
            emit(chunk)
    else:
        means = weather_df.mean(numeric_only=True)
        chunk = grid_df.copy()
        for col, val in means.items():
            chunk[col] = val
        chunk["ignition"] = [
            int(bool(ignitions.get((r, c), False)))
            for r, c in zip(chunk["row"], chunk["col"])
        ]
        emit(chunk)

    return total_rows, ignition_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raster-dir", required=True)
    parser.add_argument("--weather", required=True)
    parser.add_argument("--fire-labels", required=True)
    parser.add_argument("--grid-size", type=int, default=250)
    parser.add_argument("--out", default="feature_table.csv")
    args = parser.parse_args()

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

    total_rows, ignition_rows = write_feature_table(grid_df, ignitions, has_date, args.weather, args.out)

    print(f"Wrote {total_rows} rows to {args.out}")
    if total_rows:
        print(f"Ignition rate: {ignition_rows / total_rows:.4f}")


if __name__ == "__main__":
    main()
