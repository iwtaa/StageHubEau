import colorsys
from APIcalls import *
import numpy as np
import pandas as pd

import geopandas as gpd
import matplotlib.pyplot as plt
from APIcalls import *
import pprint

criteres = pd.read_csv('src/criteres.csv', sep=',')
parametres = criteres["parametre"].tolist()
criteres_dict = {
    row['parametre']: {key: row[key] for key in row.index if key != 'parametre'}
    for _, row in criteres.iterrows()
}

departement = '17'
date_max_prelevement = '2025-01-01 00:00:00'
date_min_prelevement = '2023-01-01 00:00:00'

data = getAPIdata(f'https://geo.api.gouv.fr/departements/{departement}/communes?format=geojson&geometry=contour')
gdf = gpd.GeoDataFrame.from_features(data['features'])
def get_color(code, min_val, max_val):
    if code not in average_communes:
        return 'black'
    else:
        number = average_communes[code]
    if number < min_val:
        return 'blue'
    elif number > max_val:
        return 'red'
    else:
        # Normalize the value to a range between 0 and 1
        normalized_value = (number - min_val) / (max_val - min_val)
        # Convert the normalized value to a hue value (0=red, 1/6=yellow, 1/3=green)
        hue = (0.33) - (normalized_value * (0.33))
        # Convert the HSL color to RGB
        r, g, b = colorsys.hls_to_rgb(hue, 0.5, 1)  # Use HSL: Hue, Saturation, Lightness
        return (r, g, b)

def plot_geodata(gdf, get_color, min, max, name):
    """
    Plots the GeoDataFrame using the provided get_color lambda to assign a color for each geometry.
    
    Parameters:
      gdf (GeoDataFrame): The geodataframe to plot. Must contain a column 'code' to compute colors.
      get_color (lambda): A lambda function that takes a commune code and returns a color.
    """
    try:
        # Apply the lambda to compute a color for each row based on its 'code'
        gdf['color'] = gdf.apply(lambda row: get_color(row['code'], min, max), axis=1)

        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        # Create the figure and axes
        fig, ax = plt.subplots(1, 1)
        
        # Plot the GeoDataFrame with the assigned colors
        gdf.plot(color=gdf['color'], ax=ax)

        # Define the colormap for the colorbar
        cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ["green", "yellow", "red"])

        # Create the scalar mappable for the colorbar (using fixed vmin and vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min, vmax=max))
        sm._A = []  # Necessary workaround to avoid errors

        # Add and label the colorbar to the figure
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Value (ml)')

        plt.title(f"{name} in {departement}")
        plt.show()

    except Exception as e:
        print(f"An error occurred while plotting: {e}")

for param in parametres:
    parametre = [param]
    min = criteres_dict[param]['min']
    max = criteres_dict[param]['max']
    # Example usage

    data = getMeasureDepartment(departement, parameter=parametre, date_max_prelevement=date_max_prelevement, date_min_prelevement=date_min_prelevement)

    average_communes = {}
    communes = {}
    for measure in data['data']:
        if measure['resultat_numerique'] is not None:
            if measure['code_commune'] in communes:
                
                communes[measure['code_commune']].append(measure['resultat_numerique'])
            else:
                communes[measure['code_commune']] = [measure['resultat_numerique']]

    average_communes = {key: np.mean(value) for key, value in communes.items()}
    plot_geodata(gdf, get_color, min, max, criteres_dict[param]['name'])
