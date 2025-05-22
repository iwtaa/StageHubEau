import colorsys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
import requests
import io

def load_data(plv_path='DIS_PLV_2024.txt', result_path='DIS_RESULT_2024.txt'):
    """Loads and preprocesses data from text files."""
    plv = pd.read_csv(plv_path, sep=',', encoding='latin-1', usecols=['referenceprel', 'inseecommuneprinc'])
    result = pd.read_csv(result_path, sep=',', encoding='latin-1', usecols=['referenceprel', 'cdparametre', 'valtraduite', 'cdunitereferencesiseeaux'])

    plv = plv[plv['referenceprel'].isin(result['referenceprel'])]
    result = result[result['referenceprel'].isin(plv['referenceprel'])]

    result['cdparametre'] = result['cdparametre'].astype(str).str[:-2]
    
    merged = pd.merge(result, plv, on='referenceprel', how='inner')
    
    cdparametre_counts = merged['cdparametre'].value_counts()
    valid_cdparametre = cdparametre_counts[cdparametre_counts >= 250].index
    merged = merged[merged['cdparametre'].isin(valid_cdparametre)]
    
    merged.dropna(inplace=True)
    return merged

def fetch_geodata(departement_code='17'):
    """Fetches GeoJSON data for a given department code."""
    url = f'https://geo.api.gouv.fr/departements/{departement_code}/communes?format=geojson&geometry=contour'
    response = requests.get(url)
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    data = response.json()
    gdf = gpd.GeoDataFrame.from_features(data['features'])
    gdf.rename(columns={'code': 'inseecommuneprinc'}, inplace=True)
    gdf['inseecommuneprinc'] = gdf['inseecommuneprinc'].astype(str)
    return gdf

def get_nom_parametre(code):
    """Fetches the name of a parameter from the API."""
    try:
        response = requests.get(f'https://hubeau.eaufrance.fr/api/v3/parametres/{code}')
        response.raise_for_status()
        data = response.json()
        if data['count'] > 0:
            return data['data'][0]['nom_parametre']
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching parameter {code}: {e}")
        return None

def assign_colors(gdf, average_communes, min_val, max_val):
    """Assigns colors to GeoDataFrame based on average values."""
    def get_color(value, min_val, max_val):
        if value < min_val:
            return 'blue'
        elif value > max_val:
            return 'red'
        else:
            normalized_value = (value - min_val) / (max_val - min_val)
            hue = (0.33) - (normalized_value * (0.33))
            r, g, b = colorsys.hls_to_rgb(hue, 0.5, 1)
            return (r, g, b)

    gdf['color'] = gdf['inseecommuneprinc'].map(lambda code: 'black' if code not in average_communes else get_color(average_communes[code], min_val, max_val))
    return gdf

def plot_and_save(gdf, min_val, max_val, plot_title, output_path, unit):
    """Plots and saves the GeoDataFrame with colors based on average values."""
    fig, ax = plt.subplots(1, 1)
    gdf.plot(color=gdf['color'], ax=ax)

    cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ["green", "yellow", "red"])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_val, vmax=max_val))
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(unit)

    plt.title(plot_title)
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    departement_code = '17'
    
    df_merged = load_data()
    gdf = fetch_geodata(departement_code)

    min_by_parametre = df_merged.groupby('cdparametre')['valtraduite'].min().to_dict()
    max_by_parametre = df_merged.groupby('cdparametre')['valtraduite'].max().to_dict()
    unit_by_parametre = df_merged.groupby('cdparametre')['cdunitereferencesiseeaux'].first().to_dict()

    gdf['inseecommuneprinc'] = gdf['inseecommuneprinc'].astype(str)
    df_merged['inseecommuneprinc'] = df_merged['inseecommuneprinc'].astype(str)
    
    # Filter out parameters with 'SANS OBJET' unit
    valid_parameters = [parametre for parametre in df_merged['cdparametre'].unique() 
                          if unit_by_parametre.get(parametre) != 'SANS OBJET']

    for parametre in tqdm(valid_parameters, desc="Processing parameters"):
        unit = unit_by_parametre[parametre]
        
        # Fetch parameter name and skip if not found
        critere = get_nom_parametre(parametre)
        if critere is None:
            print(f"Parameter {parametre} not found in the API.")
            continue
        
        # Get min and max values, skip if max is 0
        min_val = min_by_parametre[str(parametre)]
        max_val = max_by_parametre[str(parametre)]
        if max_val == 0:
            continue
            
        # Further filter the DataFrame to remove NaN values for the specific parameter
        df_param = df_merged[df_merged['cdparametre'] == parametre].dropna(subset=['valtraduite', 'inseecommuneprinc'])
        
        # Skip if there are fewer than 250 valid entries for this parameter
        if len(df_param) < 250:
            continue
            
        # Calculate average value per commune
        average_by_insee = df_param.groupby('inseecommuneprinc')['valtraduite'].mean().to_dict()
        
        gdf_copy = gdf.copy()
        gdf_copy = assign_colors(gdf_copy, average_by_insee, min_val, max_val)

        plot_title = f"{critere} in {departement_code}"
        output_path = f'maps/{departement_code}_{critere}.png'
        plot_and_save(gdf_copy, min_val, max_val, plot_title, output_path, unit)