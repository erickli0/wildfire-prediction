"""
ingest_landsat.py

Pulls Landsat-8 Surface Reflectance imagery via Google Earth Engine,
computes NDVI, and exports a mean-composite raster for a given
region/date range to Google Drive (or GCS).

By default, a multi-month date range is split into one export task per
calendar month (`--single-export` turns this off). GEE caps how much a
single export task can process (maxPixels, per-task compute time), so
pulling a full fire season over a large AOI needs to be batched rather
than requested as one export -- this is what actually lets the ingestion
side scale to a large multi-GB/TB imagery pull without every export task
failing or timing out.

Setup:
    pip install earthengine-api geemap
    earthengine authenticate   # one-time browser auth

Usage:
    python ingest_landsat.py --start 2025-06-01 --end 2025-09-01 \
        --region assets/study_area.geojson --out ndvi_summer2025
"""
import argparse
import calendar
from datetime import date, timedelta

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


def month_ranges(start_date: str, end_date: str):
    """Split [start_date, end_date) into (month_start, month_end) chunks,
    each spanning at most one calendar month.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    cur = start
    while cur < end:
        days_in_month = calendar.monthrange(cur.year, cur.month)[1]
        month_end = min(date(cur.year, cur.month, days_in_month) + timedelta(days=1), end)
        yield cur.isoformat(), month_end.isoformat()
        cur = month_end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--region", required=True, help="Path to GeoJSON AOI")
    parser.add_argument("--out", default="ndvi_composite")
    parser.add_argument("--scale", type=int, default=30)
    parser.add_argument("--single-export", action="store_true",
                         help="Export one composite for the whole date range instead of "
                              "batching by month (fine for short ranges; large AOI/date-range "
                              "pulls can hit GEE per-task limits)")
    parser.add_argument("--project", required=True, help="Earth Engine Cloud project id")
    args = parser.parse_args()

    ee.Initialize(project=args.project)

    import geemap
    region_fc = geemap.vector_to_ee(args.region)
    region_geom = region_fc.geometry()

    if args.single_export:
        ndvi = build_ndvi_composite(region_geom, args.start, args.end)
        export_to_drive(ndvi, region_geom, args.out, scale=args.scale)
        return

    tasks = []
    for chunk_start, chunk_end in month_ranges(args.start, args.end):
        ndvi = build_ndvi_composite(region_geom, chunk_start, chunk_end)
        description = f"{args.out}_{chunk_start[:7]}"
        tasks.append(export_to_drive(ndvi, region_geom, description, scale=args.scale))
    print(f"Started {len(tasks)} monthly export task(s) covering {args.start} to {args.end}")


if __name__ == "__main__":
    main()
