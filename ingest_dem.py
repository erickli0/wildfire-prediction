"""
ingest_dem.py

Two ways to get elevation/slope/aspect, pick whichever fits your workflow:

1) GEE (server-side, scales easily) -> export_dem_products_gee()
2) Local raster via rasterio + numpy, for a DEM you already downloaded
   (e.g. exported from QGIS, or a USGS 3DEP tile) -> compute_local_terrain()

compute_local_terrain() processes the DEM in row-strips rather than
reading it into one array, so a DEM far larger than available RAM (e.g. a
statewide 1m LiDAR-derived tile set) can still be processed on a single
machine -- only a few strips are ever held in memory at once.

Usage:
    python ingest_dem.py --mode gee --region assets/study_area.geojson --out terrain
    python ingest_dem.py --mode local --dem data/raw/dem.tif --out data/processed/
"""
import argparse
import os


def export_dem_products_gee(region_path, out_prefix, scale=30):
    import ee
    import geemap

    ee.Initialize()
    region_fc = geemap.geojson_to_ee(region_path)
    region = region_fc.geometry()

    dem = ee.Image("USGS/3DEP/10m").clip(region)
    slope = ee.Terrain.slope(dem)
    aspect = ee.Terrain.aspect(dem)

    stack = dem.rename("elevation").addBands(slope.rename("slope")).addBands(
        aspect.rename("aspect")
    )

    task = ee.batch.Export.image.toDrive(
        image=stack,
        description=out_prefix,
        folder="wildfire_risk",
        fileNamePrefix=out_prefix,
        region=region,
        scale=scale,
        maxPixels=1e13,
    )
    task.start()
    print(f"Started export task: {out_prefix}")


def compute_local_terrain(dem_path, out_dir, block_rows=1024):
    """Compute slope/aspect from a local DEM using rasterio + numpy.
    (Equivalent to QGIS's Raster > Analysis > Slope/Aspect tools --
    use this if you'd rather do it in code than in the QGIS GUI.)

    Reads and writes the DEM in horizontal strips of `block_rows` rows
    (each with a 1-row halo above/below so gradients stay correct across
    strip boundaries) instead of loading the whole raster into memory, so
    this scales to DEMs of arbitrary size on a single machine.
    """
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(dem_path) as src:
        profile = src.profile
        px, py = src.res
        height, width = src.height, src.width
        profile.update(dtype="float32", count=1)

        slope_path = os.path.join(out_dir, "slope.tif")
        aspect_path = os.path.join(out_dir, "aspect.tif")
        bytes_read = 0

        with rasterio.open(slope_path, "w", **profile) as slope_dst, \
                rasterio.open(aspect_path, "w", **profile) as aspect_dst:
            for r0 in range(0, height, block_rows):
                rows = min(block_rows, height - r0)
                halo_top = 1 if r0 > 0 else 0
                halo_bottom = 1 if r0 + rows < height else 0
                read_window = Window(0, r0 - halo_top, width, rows + halo_top + halo_bottom)

                block = src.read(1, window=read_window).astype("float64")
                bytes_read += block.nbytes

                gy, gx = np.gradient(block, py, px)
                slope = np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))
                # atan2(east, north) of the downhill vector (-gx, gy) is
                # already a compass bearing (0=N, 90=E); wrap -180..180
                # into 0..360.
                aspect = np.degrees(np.arctan2(-gx, gy)) % 360.0

                interior = slice(halo_top, halo_top + rows)
                write_window = Window(0, r0, width, rows)
                slope_dst.write(slope[interior].astype("float32"), 1, window=write_window)
                aspect_dst.write(aspect[interior].astype("float32"), 1, window=write_window)

    print(f"Processed {bytes_read / 1e9:.2f} GB ({height}x{width} px) in strips of {block_rows} rows")
    print(f"Wrote {slope_path}")
    print(f"Wrote {aspect_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gee", "local"], required=True)
    parser.add_argument("--region", help="GeoJSON AOI (gee mode)")
    parser.add_argument("--dem", help="Path to local DEM GeoTIFF (local mode)")
    parser.add_argument("--block-rows", type=int, default=1024,
                         help="Rows per strip when processing a local DEM (local mode)")
    parser.add_argument("--out", default="terrain")
    args = parser.parse_args()

    if args.mode == "gee":
        export_dem_products_gee(args.region, args.out)
    else:
        compute_local_terrain(args.dem, args.out, block_rows=args.block_rows)


if __name__ == "__main__":
    main()
