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

Usage:
    python build_feature_table.py --raster-dir data/processed \
        --weather data/processed/weather.csv \
        --fire-labels data/raw/fire_labels.geojson \
        --grid-size 250 \
        --out data/processed/feature_table.csv
"""
import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


def rasters_to_grid_df(raster_paths: dict, grid_size: int):
    """Read each raster, tile it into `grid_size` px blocks, and return one
    row per grid cell with columns = raster names + row/col + lon/lat.
    """
    first_key = next(iter(raster_paths))
    with rasterio.open(raster_paths[first_key]) as ref:
        transform = ref.transform
        height, width = ref.height, ref.width
        crs = ref.crs

    rows = []
    for r0 in range(0, height, grid_size):
        for c0 in range(0, width, grid_size):
            window = rasterio.windows.Window(
                c0, r0, min(grid_size, width - c0), min(grid_size, height - r0)
            )
            cell = {"row": r0, "col": c0}
            for name, path in raster_paths.items():
                with rasterio.open(path) as src:
                    block = src.read(1, window=window)
                    valid = block[block != src.nodata] if src.nodata is not None else block
                    cell[name] = float(np.nanmean(valid)) if valid.size else np.nan
            x, y = rasterio.transform.xy(transform, r0 + grid_size // 2, c0 + grid_size // 2)
            cell["lon"], cell["lat"] = x, y
            rows.append(cell)

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
    even after join_weather() expands the table across dates.
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


def join_weather(grid_df, weather_path):
    """Attach weather to the grid.

    If weather.csv has a 'date' column, cross-join it onto every cell so
    each (cell, date) row carries that date's actual weather -- otherwise
    weather is constant across all rows and the model can't learn
    anything from it. Without a 'date' column, fall back to broadcasting
    the period mean to every cell.
    """
    weather_df = pd.read_csv(weather_path)

    if "date" not in weather_df.columns:
        means = weather_df.mean(numeric_only=True)
        for col, val in means.items():
            grid_df[col] = val
        return grid_df

    weather_df = weather_df.copy()
    weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.normalize()
    return grid_df.merge(weather_df, how="cross")


def label_ignitions(grid_df, ignitions, has_date):
    if has_date:
        grid_df["ignition"] = grid_df.apply(
            lambda r: int(r["date"] in ignitions.get((r.row, r.col), set())), axis=1
        )
    else:
        grid_df["ignition"] = grid_df.apply(
            lambda r: int(bool(ignitions.get((r.row, r.col), False))), axis=1
        )
    return grid_df


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

    grid_df, crs = rasters_to_grid_df(raster_paths, args.grid_size)
    ignitions, has_date = find_cell_ignition_dates(grid_df, args.fire_labels, crs, args.grid_size)

    grid_df = join_weather(grid_df, args.weather)
    grid_df = label_ignitions(grid_df, ignitions, has_date)

    grid_df = grid_df.dropna(subset=["ndvi", "elevation", "slope", "aspect"])
    grid_df.to_csv(args.out, index=False)
    print(f"Wrote {len(grid_df)} rows to {args.out}")
    print(f"Ignition rate: {grid_df['ignition'].mean():.4f}")


if __name__ == "__main__":
    main()
