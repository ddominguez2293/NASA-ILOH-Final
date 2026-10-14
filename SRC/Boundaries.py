import geopandas as gpd
from pynhd import WaterData
import pandas as pd
import os
import geopandas as gpd
import matplotlib.pyplot as plt
# Create folder if it doesn't exist
os.makedirs("data/shapes", exist_ok=True)

# Ohio River Basin: HUC2 = 05 (use wbd02 for HUC2 boundaries)
wbd_huc2 = WaterData("wbd02")
ohio = wbd_huc2.byid("huc2", ["05"])
ohio_path = "data/shapes/ohio_river_basin.shp"
ohio.to_file(ohio_path)

# Illinois River Basin: HUC4 = 0712 + 0713, merged into one shapefile
wbd_huc4 = WaterData("wbd04")
illinois_0712 = wbd_huc4.byid("huc4", ["0712"])
illinois_0713 = wbd_huc4.byid("huc4", ["0713"])
illinois = gpd.GeoDataFrame(pd.concat([illinois_0712, illinois_0713], ignore_index=True)).dissolve()
illinois_path = "data/shapes/illinois_river_basin.shp"
illinois.to_file(illinois_path)

# Merge both into one shapefile
both = gpd.GeoDataFrame(pd.concat([ohio, illinois], ignore_index=True))
both = both.dissolve()
both_path = "data/shapes/ohio_illinois_basin.shp"
both.to_file(both_path)

print("Saved Ohio, Illinois, and merged basin shapefiles in data/shapes")

# Load saved shapefiles
ohio = gpd.read_file("data/shapes/ohio_river_basin.shp")
illinois = gpd.read_file("data/shapes/illinois_river_basin.shp")

# Create a base plot
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