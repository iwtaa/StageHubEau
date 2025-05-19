import colorsys
from APIcalls import *
import numpy as np

import geopandas as gpd
import matplotlib.pyplot as plt
from APIcalls import *
import pprint

departement = '17'
parameter = ['1340']
date_max_prelevement = '2025-01-01 00:00:00'
date_min_prelevement = '2024-08-01 00:00:00'

data = getMeasureDepartment(departement, parameter=parameter, date_max_prelevement=date_max_prelevement, date_min_prelevement=date_min_prelevement)

average_communes = {}
communes = {}
for measure in data['data']:
    if measure['code_commune'] in communes:
        communes[measure['code_commune']].append(measure['resultat_numerique'])
    else:
        communes[measure['code_commune']] = [measure['resultat_numerique']]

average_communes = {key: np.mean(value) for key, value in communes.items()}

data = getAPIdata(f'https://geo.api.gouv.fr/departements/{departement}/communes?format=geojson&geometry=contour')
gdf = gpd.GeoDataFrame.from_features(data['features'])
def get_color(code):
    if code not in average_communes:
        number = 0.0
    else:
        number = average_communes[code]
    min_val = 0.1
    max_val = 50.0
    if number == 0.0:
        return 'black'
    elif number < min_val:
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


def plot_geodata(gdf, get_color):
    """
    Plots the GeoDataFrame using the provided get_color lambda to assign a color for each geometry.
    
    Parameters:
      gdf (GeoDataFrame): The geodataframe to plot. Must contain a column 'code' to compute colors.
      get_color (lambda): A lambda function that takes a commune code and returns a color.
    """
    try:
        # Apply the lambda to compute a color for each row based on its 'code'
        gdf['color'] = gdf.apply(lambda row: get_color(row['code']), axis=1)

        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        # Create the figure and axes
        fig, ax = plt.subplots(1, 1)
        
        # Plot the GeoDataFrame with the assigned colors
        gdf.plot(color=gdf['color'], ax=ax)

        # Define the colormap for the colorbar
        cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ["green", "yellow", "red"])

        # Create the scalar mappable for the colorbar (using fixed vmin and vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.1, vmax=50))
        sm._A = []  # Necessary workaround to avoid errors

        # Add and label the colorbar to the figure
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Value (ml)')

        plt.title("Geopandas Plot")
        plt.show()

    except Exception as e:
        print(f"An error occurred while plotting: {e}")

plot_geodata(gdf, get_color)