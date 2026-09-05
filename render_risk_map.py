"""
render_risk_map.py

Run this INSIDE the QGIS Python console (Plugins > Python Console), or
headless with QGIS's own bundled Python interpreter (see README/session
notes -- `qgis_process` is for Processing algorithms/models, not arbitrary
scripts like this one, so it doesn't apply here). Loads your model's
predicted-probability grid (a CSV with lon/lat/risk_score, or a GeoTIFF)
and renders it as a red-yellow-green heatmap gradient with a title and
legend, then exports a PNG map.

Why do this step in QGIS instead of matplotlib:
  - Proper basemap/CRS handling for a professional-looking risk map
  - Graduated/heatmap symbology presets made for exactly this use case
  - Easy to layer roads, structures, or fire perimeters on top for context

Usage (from QGIS Python console):
    exec(open('/path/to/render_risk_map.py').read())

Usage (headless, via QGIS's bundled python-qgis interpreter):
    python-qgis render_risk_map.py
"""
import os

from qgis.core import (
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsVectorLayer,
    QgsRasterLayer,
    QgsProject,
    QgsHeatmapRenderer,
    QgsStyle,
    QgsUnitTypes,
    QgsLayoutExporter,
    QgsPrintLayout,
    QgsLayoutItemMap,
    QgsLayoutItemLegend,
    QgsLayoutItemLabel,
)
from qgis.PyQt.QtCore import QRectF
from qgis.PyQt.QtGui import QColor, QFont

# The interactive QGIS Python console already has a QgsApplication running;
# a standalone headless script does not, so start (and later stop) one only
# when we're the ones who need it.
_owns_app = QgsApplication.instance() is None
if _owns_app:
    qgs = QgsApplication([], False)
    qgs.initQgis()

CSV_PATH = os.path.abspath("predictions.csv")  # needs lon, lat, risk_score columns
OUT_PNG = os.path.abspath("outputs/wildfire_risk_map.png")

uri = f"file:///{CSV_PATH.replace(os.sep, '/')}?delimiter=,&xField=lon&yField=lat&crs=EPSG:4326"
layer = QgsVectorLayer(uri, "wildfire_risk", "delimitedtext")
if not layer.isValid():
    raise RuntimeError(f"Failed to load {CSV_PATH} as a point layer")

project = QgsProject.instance()

# Points come in as lon/lat (EPSG:4326); reproject the whole map to Web
# Mercator so the OSM basemap tiles (also EPSG:3857) line up without
# distortion, and QGIS reprojects the point layer on the fly to match.
web_mercator = QgsCoordinateReferenceSystem("EPSG:3857")
project.setCrs(web_mercator)

basemap = QgsRasterLayer(
    "type=xyz&url=https://tile.openstreetmap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png&zmax=19&zmin=0",
    "OpenStreetMap",
    "wms",
)
if not basemap.isValid():
    raise RuntimeError("Failed to load OpenStreetMap basemap tiles (check internet access)")

project.addMapLayer(basemap)
project.addMapLayer(layer)

# Heatmap renderer instead of discrete dot symbols: it kernel-smooths the
# risk_score-weighted points into a continuous gradient surface, which
# reads far better than a grid of dots at this cell spacing. Radius is in
# map units (metres, since the project CRS is Web Mercator) and set wide
# enough that adjacent grid cells' kernels overlap and blend rather than
# leaving visible gaps.
ramp = QgsStyle.defaultStyle().colorRamp("RdYlGn")
ramp.invert()

renderer = QgsHeatmapRenderer()
renderer.setWeightExpression("risk_score")
renderer.setRadiusUnit(QgsUnitTypes.RenderMapUnits)
renderer.setRadius(1500)  # grid cells are ~800-1000m apart -- needs >1x spacing to blend
renderer.setColorRamp(ramp)
layer.setRenderer(renderer)
# The heatmap's kernel tails never fully reach zero, so without some
# transparency the "no risk nearby" areas paint as a solid opaque wash
# (the ramp's low-end color) and hide the basemap under the whole AOI,
# not just near the high-risk points.
layer.setOpacity(0.75)
layer.setName("Predicted Wildfire Risk")
layer.triggerRepaint()

# Export a simple print layout to PNG
layout = QgsPrintLayout(project)
layout.initializeDefaults()

# Reproject the point layer's extent into the project CRS (EPSG:3857) and
# pad it out, so the map shows surrounding basemap context instead of
# cropping tight to the outermost dots.
transform = QgsCoordinateTransform(layer.crs(), web_mercator, project)
extent = transform.transformBoundingBox(layer.extent())
extent.scale(1.2)

# Size the map item to fill the entire page instead of a hardcoded rect --
# initializeDefaults()'s default page size doesn't necessarily match a
# fixed 200x150mm box, which left blank page space around a smaller map.
page_size = layout.pageCollection().page(0).pageSize()
map_item = QgsLayoutItemMap(layout)
map_item.attemptSetSceneRect(QRectF(0, 0, page_size.width(), page_size.height()))
# Pin exactly which layers render and in what order (points on top of the
# basemap), independent of the project's layer-tree order.
map_item.setLayers([layer, basemap])
map_item.setKeepLayerSet(True)
map_item.setExtent(extent)
map_item.zoomToExtent(extent)
layout.addLayoutItem(map_item)

# Title, top-left, on a translucent white plate so it stays legible over
# whatever's underneath it on the map.
title = QgsLayoutItemLabel(layout)
title.setText("Wildfire Ignition Risk Map")
title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
title.attemptSetSceneRect(QRectF(8, 8, 150, 14))
title.setBackgroundEnabled(True)
title.setBackgroundColor(QColor(255, 255, 255, 210))
layout.addLayoutItem(title)

# Legend, top-right, restricted to just the risk layer (skip the basemap --
# its own attribution/labelling is already on the tiles) so it shows the
# heatmap's continuous color-ramp swatch.
legend = QgsLayoutItemLegend(layout)
legend.setLinkedMap(map_item)
legend.setTitle("Risk Score")
legend.setAutoUpdateModel(False)
root_group = legend.model().rootGroup()
root_group.clear()
root_group.addLayer(layer)
legend.attemptSetSceneRect(QRectF(page_size.width() - 60, 8, 52, 40))
legend.setBackgroundEnabled(True)
legend.setBackgroundColor(QColor(255, 255, 255, 210))
layout.addLayoutItem(legend)

exporter = QgsLayoutExporter(layout)
exporter.exportToImage(OUT_PNG, QgsLayoutExporter.ImageExportSettings())
print(f"Exported risk map to {OUT_PNG}")

if _owns_app:
    qgs.exitQgis()
