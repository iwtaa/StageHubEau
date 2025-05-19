import pandas as pd
import numpy as np
from APIcalls import *

departement = '17'
date_max_prelevement = '2025-01-01 00:00:00'
date_min_prelevement = '2024-08-01 00:00:00'

communes = getDepartementCommunes(departement)
test = {}
for commune in communes:
    test[commune['code']] = True

criteres = pd.read_csv('src/criteres.csv', sep=',')
parametres = criteres["parametre"].tolist()
criteres_dict = {
    row['parametre']: {key: row[key] for key in row.index if key != 'parametre'}
    for _, row in criteres.iterrows()
}

all_max = criteres['max'].tolist()
data = getMeasureDepartment(departement, parameter=parametres, date_max_prelevement=date_max_prelevement, date_min_prelevement=date_min_prelevement)
if data is None:
    print("No data for department")
    exit(1)
data = data['data']

for measure in data:
    if measure['resultat_numerique'] > criteres_dict[int(measure['code_parametre'])]['max'] and measure['resultat_numerique'] > criteres_dict[int(measure['code_parametre'])]['min']:
        if measure['code_commune'] in test:
            test[measure['code_commune']] = False

import geopandas as gpd
import matplotlib.pyplot as plt
import colorsys
import pprint

try:
    data = getAPIdata(f'https://geo.api.gouv.fr/departements/{departement}/communes?format=geojson&geometry=contour')
    gdf = gpd.GeoDataFrame.from_features(data['features'])
    # Define a function to determine the color of each geometry
    def get_color(code):
        status = test.get(code, None)
        if status is True:
            return "green"  # potable
        elif status is False:
            return "red"    # non-potable
        else:
            return "black"  # unknown

    # Apply the function to each geometry to get its color
    gdf['color'] = gdf.apply(lambda row: get_color(row['code']), axis=1)

    pprint.pprint(gdf.head())  # Print the first few rows of the GeoDataFrame
    # Generate random colors for each area
    num_geometries = len(gdf)
    
    # Create a figure and an axes object
    fig, ax = plt.subplots(1, 1)

    # Plot the GeoDataFrame
    gdf.plot(color=gdf['color'], ax=ax)
    import matplotlib.patches as mpatches
    potable_patch = mpatches.Patch(color='green', label='Potable')
    non_potable_patch = mpatches.Patch(color='red', label='Non Potable')
    ax.legend(handles=[potable_patch, non_potable_patch], loc='upper right')
    plt.title("Potabilite de l'eau du robinet")
    plt.show()

except FileNotFoundError:
    print(f"Error: GeoJSON file not found at {gdf}")
except Exception as e:
    print(f"An error occurred: {e}")
