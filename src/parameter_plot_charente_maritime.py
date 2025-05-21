import colorsys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from APIcalls import getAPIdata, getMeasureDepartment

def load_criteria(csv_path):
    """Loads criteria from a CSV file and returns a dictionary."""
    criteres = pd.read_csv(csv_path, sep=',')
    criteres_dict = {
        row['parametre']: {key: row[key] for key in row.index if key != 'parametre'}
        for _, row in criteres.iterrows()
    }
    return criteres_dict

def fetch_geodata(departement_code):
    """Fetches GeoJSON data for a given department code."""
    data = getAPIdata(f'https://geo.api.gouv.fr/departements/{departement_code}/communes?format=geojson&geometry=contour')
    gdf = gpd.GeoDataFrame.from_features(data['features'])
    return gdf

def calculate_average_values(data):
    """Calculates the average value for each commune from the API data."""
    communes = {}
    for measure in data['data']:
        if measure['resultat_numerique'] is not None:
            code_commune = measure['code_commune']
            resultat = measure['resultat_numerique']
            if code_commune in communes:
                communes[code_commune].append(resultat)
            else:
                communes[code_commune] = [resultat]
    average_communes = {key: np.mean(value) for key, value in communes.items()}
    return average_communes

def fetch_measurements(departement, parametre, date_min_prelevement, date_max_prelevement):
    """Fetches measurement data from the API for a given department and parameter."""
    data = getMeasureDepartment(departement, parameter=parametre, date_max_prelevement=date_max_prelevement, date_min_prelevement=date_min_prelevement)
    return data

def get_color(value, min_val, max_val):
    """Assigns a color based on the given value and min/max thresholds."""
    if value < min_val:
        return 'blue'
    elif value > max_val:
        return 'red'
    else:
        normalized_value = (value - min_val) / (max_val - min_val)
        hue = (0.33) - (normalized_value * (0.33))
        r, g, b = colorsys.hls_to_rgb(hue, 0.5, 1)
        return (r, g, b)

def plot_geodata(gdf, average_communes, min_val, max_val, plot_title):
    """Plots the GeoDataFrame with colors based on average values and displays a colorbar."""
    def get_geometry_color(code, min_val, max_val):
        if code not in average_communes:
            return 'black'
        else:
            number = average_communes[code]
            return get_color(number, min_val, max_val)

    gdf['color'] = gdf.apply(lambda row: get_geometry_color(row['code'], min_val, max_val), axis=1)

    fig, ax = plt.subplots(1, 1)
    gdf.plot(color=gdf['color'], ax=ax)

    cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ["green", "yellow", "red"])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_val, vmax=max_val))
    sm._A = []

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Value (ml)')

    plt.title(plot_title)
    plt.show()

def save_geodata(gdf, average_communes, min_val, max_val, plot_title, output_path, unit):
    """Saves the GeoDataFrame plot with colors based on average values to a file."""
    def get_geometry_color(code, min_val, max_val):
        if code not in average_communes:
            return 'black'
        else:
            number = average_communes[code]
            return get_color(number, min_val, max_val)

    gdf['color'] = gdf.apply(lambda row: get_geometry_color(row['code'], min_val, max_val), axis=1)

    fig, ax = plt.subplots(1, 1)
    gdf.plot(color=gdf['color'], ax=ax)

    cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ["green", "yellow", "red"])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_val, vmax=max_val))
    sm._A = []

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(unit)

    plt.title(plot_title)
    plt.savefig(output_path)
    plt.close()

def main():
    """Main function to orchestrate the data fetching, processing, and plotting."""
    csv_path = 'src/criteres.csv'
    departement_code = '17'
    date_max_prelevement = '2025-01-01 00:00:00'
    date_min_prelevement = '2020-01-01 00:00:00'

    criteres_dict = load_criteria(csv_path)
    gdf = fetch_geodata(departement_code)

    for param in criteres_dict:
        parametre = [param]
        min_val = criteres_dict[param]['min']
        max_val = criteres_dict[param]['max']
        plot_title = f"{criteres_dict[param]['name']} in {departement_code}"

        data = fetch_measurements(departement_code, parametre, date_min_prelevement, date_max_prelevement)
        average_communes = calculate_average_values(data)
        plot_geodata(gdf, average_communes, min_val, max_val, plot_title)

if __name__ == "__main__":
    main()
