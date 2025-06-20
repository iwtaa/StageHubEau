import os
import json
import pandas as pd
from Utilities import *

import geopandas as gpd
import requests
import matplotlib.pyplot as plt
import time

GEO_API_URL = 'https://geo.api.gouv.fr/departements/{}/communes?format=geojson&geometry=contour'
MAX_RETRIES = 3
BACKOFF_FACTOR = 2

path = '/home/iwta/Documents/Univ/StageHubEau/'
cdparams_selected = get_selected_cdparams(path)

def fetch_geodata(departement_code=None):
    if departement_code is None:
        return None
    url = GEO_API_URL.format(departement_code)
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            gdf = gpd.GeoDataFrame.from_features(data['features'])
            gdf.rename(columns={'code': 'inseecommuneprinc'}, inplace=True)
            gdf['inseecommuneprinc'] = gdf['inseecommuneprinc'].astype(str)
            return gdf
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_FACTOR ** attempt)
            else:
                print("Max retries reached. Failed to fetch geodata.")
                return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
    return None

def map_communes_per_department(df, departement, save_path=None):
    col = 'count'
    # Ensure departement code is a string with leading zeros if necessary
    departement_str = str(departement).zfill(2)
    gdf = fetch_geodata(departement_code=departement_str)
    if gdf is None or gdf.empty:
        print(f"No geodata found for department {departement_str}")
        return
    # Ensure both columns are strings for merging
    gdf['inseecommuneprinc'] = gdf['inseecommuneprinc'].astype(str)
    df['inseecommuneprinc'] = df['inseecommuneprinc'].astype(str)
    merged = gdf.merge(df, on='inseecommuneprinc', how='left')
    df_min = merged[col].min()
    df_max = merged[col].max()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    merged.plot(
        column=col,
        cmap='viridis',
        linewidth=0.8,
        ax=ax,
        edgecolor='0.8',
        legend=True,
        legend_kwds={'label': f"Value", 'orientation': "horizontal"},
        missing_kwds={"color": "lightgrey", "edgecolor": "0.8", "label": "Missing values"},
        vmin=df_min,
        vmax=df_max
    )
    
    ax.set_title(f"value", fontsize=15)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    
def main():
    save = False
    dep = 17
    choice = input("Type 'save' if you want to save the plots. Else, shows it by default.").strip().lower()
    if choice == 'save':
        save = True
    for df in load_folder_data(os.path.join(path, 'data/clean')):
        # Interactive prompt to choose between showing or saving the plot
        
        cdparam = df['cdparametre'].iloc[0]
        df_dep = df.groupby('inseecommuneprinc').size().reset_index(name='count')[['inseecommuneprinc', 'count']]
        filename = None
        if save:
            filename = os.path.join(path, 'plots', f"map_{dep}_count_{int(cdparam)}.png")
        map_communes_per_department(df_dep, dep, save_path=filename)
        

if __name__ == "__main__":
    main()