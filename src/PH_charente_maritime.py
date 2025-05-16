from APIcalls import *
import json
import numpy as np
from collections import Counter

data = getMeasureDepartment('16', parameter='1302', date_max_prelevement='2025-01-01 00:00:00', date_min_prelevement='2021-08-01 00:00:00')

average_communes = {}
communes = {}
for measure in data['data']:
    if measure['code_commune'] in communes:
        communes[measure['code_commune']].append(measure['resultat_numerique'])
    else:
        communes[measure['code_commune']] = [measure['resultat_numerique']]

average_communes = {key: np.mean(value) for key, value in communes.items()}



import geopandas as gpd
import matplotlib.pyplot as plt
import random
from APIcalls import *
import pprint



try:
    data = getAPIdata('https://geo.api.gouv.fr/departements/16/communes?format=geojson&geometry=contour')
    gdf = gpd.GeoDataFrame.from_features(data['features'])
    # Define a function to determine the color of each geometry
    def get_color(code):
        if code not in average_communes:
            number = 0.0
        else:
            number = average_communes[code]
        min_val = 7.0
        max_val = 9.0
        if min_val <= number <= max_val:
            # Normalize the value to a range between 0 and 1
            normalized_value = (number - min_val) / (max_val - min_val)
            # Interpolate between green and red based on the normalized value
            red = normalized_value
            green = 1 - normalized_value
            blue = 0  # No blue component
            return (red, green, blue)
        else:
            return 'black'  # Default color

    # Apply the function to each geometry to get its color
    gdf['color'] = gdf.apply(lambda row: get_color(row['code']), axis=1)

    pprint.pprint(gdf.head())  # Print the first few rows of the GeoDataFrame
    # Generate random colors for each area
    num_geometries = len(gdf)
    gdf.plot(color=gdf['color'], legend=True)
    plt.show()

except FileNotFoundError:
    print(f"Error: GeoJSON file not found at {gdf}")
except Exception as e:
    print(f"An error occurred: {e}")
