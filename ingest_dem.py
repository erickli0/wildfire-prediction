"""
ingest_dem.py

Two ways to get elevation/slope/aspect, pick whichever fits your workflow:

1) GEE (server-side, scales easily) -> export_dem_products_gee()
2) Local raster via rasterio + richdem, for a DEM you already downloaded
   (e.g. exported from QGIS, or a USGS 3DEP tile) -> compute_local_terrain()

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


def compute_local_terrain(dem_path, out_dir):
    """Compute slope/aspect from a local DEM using rasterio + numpy.
    (Equivalent to QGIS's Raster > Analysis > Slope/Aspect tools --
    use this if you'd rather do it in code than in the QGIS GUI.)
    """
    import numpy as np
    import rasterio

    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(dem_path) as src:
        elevation = src.read(1).astype("float64")
        profile = src.profile
        px, py = src.res

    gy, gx = np.gradient(elevation, py, px)
    slope = np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))
    aspect = np.degrees(np.arctan2(-gx, gy))
    aspect = np.where(aspect < 0, 90.0 - aspect, 90.0 - aspect)  # compass bearing

    profile.update(dtype="float32", count=1)

    for name, arr in [("slope", slope), ("aspect", aspect)]:
        out_path = os.path.join(out_dir, f"{name}.tif")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr.astype("float32"), 1)
        print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gee", "local"], required=True)
    parser.add_argument("--region", help="GeoJSON AOI (gee mode)")
    parser.add_argument("--dem", help="Path to local DEM GeoTIFF (local mode)")
    parser.add_argument("--out", default="terrain")
    args = parser.parse_args()

    if args.mode == "gee":
        export_dem_products_gee(args.region, args.out)
    else:
        compute_local_terrain(args.dem, args.out)


if __name__ == "__main__":
    main()
