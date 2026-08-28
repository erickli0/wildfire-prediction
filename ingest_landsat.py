"""
ingest_landsat.py

Pulls Landsat-8 Surface Reflectance imagery via Google Earth Engine,
computes NDVI, and exports a mean-composite raster for a given
region/date range to Google Drive (or GCS).

Setup:
    pip install earthengine-api geemap
    earthengine authenticate   # one-time browser auth

Usage:
    python ingest_landsat.py --start 2025-06-01 --end 2025-09-01 \
        --region assets/study_area.geojson --out ndvi_summer2025
"""
import argparse
import ee


def mask_clouds(image):
    """Mask clouds/shadows using the Landsat-8 QA_PIXEL band."""
    qa = image.select("QA_PIXEL")
    cloud_bit = 1 << 3
    shadow_bit = 1 << 4
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(shadow_bit).eq(0))
    return image.updateMask(mask)


def add_ndvi(image):
    ndvi = image.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
    return image.addBands(ndvi)


def build_ndvi_composite(region, start_date, end_date):
    collection = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUD_COVER", 30))
        .map(mask_clouds)
        .map(add_ndvi)
    )
    composite = collection.select("NDVI").mean().clip(region)
    return composite


def export_to_drive(image, region, description, scale=30):
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder="wildfire_risk",
        fileNamePrefix=description,
        region=region,
        scale=scale,
        maxPixels=1e13,
    )
    task.start()
    print(f"Started export task: {description} (check GEE Tasks tab for progress)")
    return task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--region", required=True, help="Path to GeoJSON AOI")
    parser.add_argument("--out", default="ndvi_composite")
    parser.add_argument("--scale", type=int, default=30)
    args = parser.parse_args()

    ee.Initialize()

    import geemap
    region_fc = geemap.geojson_to_ee(args.region)
    region_geom = region_fc.geometry()

    ndvi = build_ndvi_composite(region_geom, args.start, args.end)
    export_to_drive(ndvi, region_geom, args.out, scale=args.scale)


if __name__ == "__main__":
    main()
