from pathlib import Path
import geopandas as gpd

src = Path(r"data\raw\CH-0222.geojson")        # adjust to your file
dst = Path(r"data\processed\landcover.gpkg")
layer_name = "landcover"

dst.parent.mkdir(parents=True, exist_ok=True)

gdf = gpd.read_file(src)
print("Rows:", len(gdf))
print("CRS:", gdf.crs)
print("Columns:", gdf.columns.tolist())

# GeoJSON is normally WGS84 (EPSG:4326). Set it if the CRS is missing.
if gdf.crs is None:
    gdf = gdf.set_crs(epsg=4326)

# Repair invalid geometries and drop empty ones
gdf["geometry"] = gdf.geometry.make_valid()
gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]

# Convert list/dict columns to text, since they can break QGIS
for col in gdf.columns.drop("geometry"):
    if gdf[col].map(lambda v: isinstance(v, (list, dict))).any():
        gdf[col] = gdf[col].astype(str)

# Optional: reproject to Swiss LV95 to match your CAMELS-CH data
# gdf = gdf.to_crs(epsg=2056)

gdf.to_file(dst, layer=layer_name, driver="GPKG")
print("Saved:", dst, "| exists:", dst.exists(), "| size (MB):", round(dst.stat().st_size / 1e6, 2))