import os
import json
import pandas as pd
from Utilities import *

import geopandas as gpd
import requests
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import sys

GEO_API_URL = 'https://geo.api.gouv.fr/departements/{}/communes?format=geojson&geometry=contour'
MAX_RETRIES = 3
BACKOFF_FACTOR = 2

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

def map_communes_per_department(df, col, df_min, df_max, departement, cdparam, save_path=None):
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
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    print(df_min, df_max)
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
    
    ax.set_title(f"value {get_cdparam_name(cdparam)}", fontsize=15)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)



# Map communes per department for all years, seasons, and overall mean
def main():
    path = os.getcwd()

    save = False
    dep = 17
    choice = input("Type 'save' if you want to save the plots. Else, shows it by default.").strip().lower()
    if choice == 'save':
        save = True
    data_folder = os.path.join(path, 'data/clean')
    
    for df in tqdm(load_folder_data(data_folder), desc="Processing files", file=sys.stdout, total=21):
        
        cdparam = df['cdparametre'].iloc[0]
        df['dateprel'] = pd.to_datetime(df['dateprel'], errors='coerce')
        min = df['valtraduite'].quantile(0.05)
        max = df['valtraduite'].quantile(0.95)
        filename = None
        param_name_short = get_cdparam_nameshort(str(cdparam))
        if param_name_short is None:
            print(f"Warning: get_cdparam_nameshort returned None for cdparam {cdparam}. Using 'unknown_param' as name.")
            name = "unknown_param"
        else:
            name = param_name_short.replace(' ', '_').replace('/', '_').replace('\\', '_')
        for year in range(2016, 2025):
            if save:
                os.makedirs(os.path.join(path, f'plots/maps/{name}'), exist_ok=True)
                filename = os.path.join(path, f'plots/maps/{name}', f"map_{dep}_mean_{name}_{year}.png")
            df_year = df[df['dateprel'].dt.year == year]
            df_year_grouped = df_year.groupby('inseecommuneprinc', as_index=False)['valtraduite'].mean()
            map_communes_per_department(df_year_grouped, 'valtraduite', min, max, dep, cdparam, save_path=filename)
        # Define a function to get season from month
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Hiver'
            elif month in [3, 4, 5]:
                return 'Printemps'
            elif month in [6, 7, 8]:
                return 'Ete'
            else:
                return 'Automne'

        if 'dateprel' in df.columns:
            df['season'] = df['dateprel'].dt.month.apply(get_season)
        else:
            print("Warning: 'dateprel' column not found in dataframe.")
            df['season'] = None

        df_dep = df.groupby('inseecommuneprinc', as_index=False)['valtraduite'].mean()

        for season in ['Hiver', 'Printemps', 'Ete', 'Automne']:
            if save:
                os.makedirs(os.path.join(path, f'plots/maps/{name}'), exist_ok=True)
                filename = os.path.join(path, f'plots/maps/{name}', f"map_{dep}_mean_{name}_{season}.png")
            df_season = df[df['season'] == season].groupby('inseecommuneprinc', as_index=False)['valtraduite'].mean()
            map_communes_per_department(df_season, 'valtraduite', min, max, dep, cdparam, save_path=filename)
        if save:
            filename = os.path.join(path, f'plots/maps/{name}', f"map_{dep}_mean_{name}.png")
        map_communes_per_department(df_dep, 'valtraduite', min, max, dep, cdparam, save_path=filename)
        
        

if __name__ == "__main__":
    main()