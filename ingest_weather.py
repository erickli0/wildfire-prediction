"""
ingest_weather.py

Pulls gridded weather data from GRIDMET (via GEE) -- temperature, relative
humidity, wind speed, and precipitation -- and exports a per-cell time
series as CSV for joining into the feature table.

Usage:
    python ingest_weather.py --start 2025-06-01 --end 2025-09-01 \
        --region assets/study_area.geojson --out weather_summer2025.csv
"""
import argparse


VARS = {
    "tmmx": "temp_max",       # max temperature (K)
    "rmin": "humidity_min",   # min relative humidity (%)
    "vs": "wind_speed",       # wind speed (m/s)
    "pr": "precip",           # precipitation (mm)
}


def build_weather_table(region_path, start_date, end_date, scale=4000):
    import ee
    import geemap

    ee.Initialize()
    region_fc = geemap.geojson_to_ee(region_path)
    region = region_fc.geometry()

    collection = (
        ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .select(list(VARS.keys()))
    )

    def reduce_image(image):
        stats = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=scale,
            maxPixels=1e13,
        )
        return ee.Feature(None, stats).set(
            "date", image.date().format("YYYY-MM-dd")
        )

    features = collection.map(reduce_image)
    return ee.FeatureCollection(features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--out", default="weather.csv")
    args = parser.parse_args()

    fc = build_weather_table(args.region, args.start, args.end)

    import geemap
    geemap.ee_export_vector(fc, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
