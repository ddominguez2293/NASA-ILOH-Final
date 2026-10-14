# Boundaries.py
# STEP 1 of the pipeline (first script; nothing runs before it).
#
# Purpose : download the Ohio (HUC2 05) and Illinois (HUC4 0712 + 0713) basin
#           boundaries from the USGS WBD and save them as shapefiles.
# Outputs : data/shapes/ohio_river_basin.shp
#           data/shapes/illinois_river_basin.shp   <- AOI used by every R step
#           data/shapes/ohio_illinois_basin.shp    (reference only)
#
# Comment style used in every script:
#   ## N. Section   = major section of the script
#   ### Step        = what the next block of code does
#   #### Note       = why / assumptions / caveats for that step
#   # inline        = short single-line bits next to the code
#
## 1. Imports
#### geopandas is imported twice below (harmless); left as-is
import geopandas as gpd
from pynhd import WaterData
import pandas as pd
import os
import geopandas as gpd
import matplotlib.pyplot as plt

## 2. Output folder
### Create the shapes folder if it doesn't exist
os.makedirs("data/shapes", exist_ok=True)

## 3. Ohio River Basin (HUC2 = 05)
### Pull the HUC2 boundary and save it
#### wbd02 is the WaterData layer that holds HUC2 boundaries
wbd_huc2 = WaterData("wbd02")
ohio = wbd_huc2.byid("huc2", ["05"])
ohio_path = "data/shapes/ohio_river_basin.shp"
ohio.to_file(ohio_path)

## 4. Illinois River Basin (HUC4 = 0712 + 0713)
### Pull both HUC4s, dissolve into one polygon and save it
#### illinois_river_basin.shp is the AOI that every R step reads (basin_shp in Functions_Paths.R)
wbd_huc4 = WaterData("wbd04")
illinois_0712 = wbd_huc4.byid("huc4", ["0712"])
illinois_0713 = wbd_huc4.byid("huc4", ["0713"])
illinois = gpd.GeoDataFrame(pd.concat([illinois_0712, illinois_0713], ignore_index=True)).dissolve()
illinois_path = "data/shapes/illinois_river_basin.shp"
illinois.to_file(illinois_path)

## 5. Combined Ohio + Illinois
### Merge both basins into one shapefile
#### Reference only; no downstream step reads this file
both = gpd.GeoDataFrame(pd.concat([ohio, illinois], ignore_index=True))
both = both.dissolve()
both_path = "data/shapes/ohio_illinois_basin.shp"
both.to_file(both_path)

print("Saved Ohio, Illinois, and merged basin shapefiles in data/shapes")

## 6. Quick-look map
### Load the saved shapefiles
ohio = gpd.read_file("data/shapes/ohio_river_basin.shp")
illinois = gpd.read_file("data/shapes/illinois_river_basin.shp")

### Create a base plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot Ohio River Basin
ohio.plot(ax=ax, color="lightblue", edgecolor="black", label="Ohio River Basin")

# Plot Illinois River Basin
illinois.plot(ax=ax, color="lightgreen", edgecolor="black", label="Illinois River Basin")

# Add labels, title, and legend
plt.title("Ohio and Illinois River Basins")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()