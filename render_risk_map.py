"""
render_risk_map.py

Run this INSIDE the QGIS Python console (Plugins > Python Console), or
headless with `qgis_process`/`qgis` CLI. Loads your model's predicted-
probability grid (a CSV with lon/lat/risk_score, or a GeoTIFF) and applies
a red-yellow-green graduated color ramp, then exports a PNG map.

Why do this step in QGIS instead of matplotlib:
  - Proper basemap/CRS handling for a professional-looking risk map
  - Graduated/heatmap symbology presets made for exactly this use case
  - Easy to layer roads, structures, or fire perimeters on top for context

Usage (from QGIS Python console):
    exec(open('/path/to/render_risk_map.py').read())
"""
from qgis.core import (
    QgsVectorLayer,
    QgsProject,
    QgsGraduatedSymbolRenderer,
    QgsStyle,
    QgsRendererRange,
    QgsSymbol,
    QgsLayoutExporter,
    QgsPrintLayout,
    QgsLayoutItemMap,
    QgsRectangle,
)

CSV_PATH = "data/processed/predictions.csv"  # needs lon, lat, risk_score columns
OUT_PNG = "outputs/wildfire_risk_map.png"

uri = f"file:///{CSV_PATH}?delimiter=,&xField=lon&yField=lat&crs=EPSG:4326"
layer = QgsVectorLayer(uri, "wildfire_risk", "delimitedtext")
if not layer.isValid():
    raise RuntimeError(f"Failed to load {CSV_PATH} as a point layer")

QgsProject.instance().addMapLayer(layer)

# Graduated color ramp on risk_score, 5 classes, red-yellow-green (reversed
# so red = high risk)
ramp = QgsStyle.defaultStyle().colorRamp("RdYlGn")
ramp.invert()

renderer = QgsGraduatedSymbolRenderer.createRenderer(
    layer,
    "risk_score",
    5,
    QgsGraduatedSymbolRenderer.Mode.EqualInterval,
    QgsSymbol.defaultSymbol(layer.geometryType()),
    ramp,
)
layer.setRenderer(renderer)
layer.triggerRepaint()

# Export a simple print layout to PNG
project = QgsProject.instance()
layout = QgsPrintLayout(project)
layout.initializeDefaults()

map_item = QgsLayoutItemMap(layout)
map_item.setRect(20, 20, 200, 150)
map_item.setExtent(layer.extent())
layout.addLayoutItem(map_item)

exporter = QgsLayoutExporter(layout)
exporter.exportToImage(OUT_PNG, QgsLayoutExporter.ImageExportSettings())
print(f"Exported risk map to {OUT_PNG}")
