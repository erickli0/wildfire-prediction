"""
build_feature_table.py

Joins NDVI, terrain (elevation/slope/aspect), weather, and fire-ignition
labels into one row-per-cell feature table ready for scikit-learn.

Assumes:
  - NDVI, elevation, slope, aspect are co-registered rasters (same grid)
  - weather.csv has per-date regional averages (join on date, or refine
    to per-cell if you exported gridded weather instead of a mean)
  - fire_labels.geojson has point/polygon ignitions with a 'date' field

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
from rasterio.features import geometry_mask


def rasters_to_grid_df(raster_paths: dict, grid_size: int):
    """Read each raster, resample/tile to `grid_size` px blocks, and
    return one row per grid cell with columns = raster names + row/col + geometry bounds.
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


def label_ignitions(grid_df, fire_labels_path, crs, grid_size, transform_res=30):
    """Mark grid cells that contain a fire ignition point as label=1."""
    fires = gpd.read_file(fire_labels_path).to_crs(crs)
    grid_df["ignition"] = 0

    cell_deg = grid_size * transform_res / 111_000  # rough deg-per-cell at this scale
    for idx, row in grid_df.iterrows():
        nearby = fires.cx[
            row.lon - cell_deg / 2 : row.lon + cell_deg / 2,
            row.lat - cell_deg / 2 : row.lat + cell_deg / 2,
        ]
        if len(nearby) > 0:
            grid_df.at[idx, "ignition"] = 1

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

    weather_df = pd.read_csv(args.weather)
    weather_means = weather_df.drop(columns=["date"], errors="ignore").mean(numeric_only=True)
    for col, val in weather_means.items():
        grid_df[col] = val

    grid_df = label_ignitions(grid_df, args.fire_labels, crs, args.grid_size)

    grid_df = grid_df.dropna(subset=["ndvi", "elevation", "slope", "aspect"])
    grid_df.to_csv(args.out, index=False)
    print(f"Wrote {len(grid_df)} rows to {args.out}")
    print(f"Ignition rate: {grid_df['ignition'].mean():.4f}")


if __name__ == "__main__":
    main()
