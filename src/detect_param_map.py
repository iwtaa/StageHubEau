from Utilities import *
import os
import pandas as pd
from tqdm import tqdm

params = get_selected_cdparams(os.getcwd())

par_sandre_df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'PAR_SANDRE_short.txt'), sep='\t')
criteres_offi_df = pd.read_csv(os.path.join(os.getcwd(), 'src', 'criteresOffi.txt'), sep='\t')

import requests
import geopandas as gpd
import matplotlib.pyplot as plt
import time
import numpy as np
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

import matplotlib.colors as mcolors

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

    # Custom colormap: green for 0, yellow-red for 1-25, red for >25
    def color_mapper(val):
        if np.isnan(val):
            return '#d3d3d3'  # light grey for missing
        if val == 0:
            return '#2ecc40'  # green
        elif 1 <= val < 25:
            # interpolate between yellow and red
            norm = (val - 1) / (24)  # 1 maps to 0, 25 maps to 1
            return mcolors.to_hex(mcolors.LinearSegmentedColormap.from_list('yr', ['#ffff00', '#ff0000'])(norm))
        else:
            return '#ff0000'  # red

    merged['color'] = merged[col].apply(color_mapper)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    merged.plot(
        color=merged['color'],
        linewidth=0.8,
        ax=ax,
        edgecolor='0.8',
        missing_kwds={"color": "lightgrey", "edgecolor": "0.8", "label": "Missing values"},
    )

    # Custom legend
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color='#2ecc40', label='0 (Green)'),
        mpatches.Patch(color='#ffff00', label='1 (Yellow)'),
        mpatches.Patch(color='#ff0000', label='25+ (Red)'),
        mpatches.Patch(color='#d3d3d3', label='Missing')
    ]
    ax.legend(handles=legend_patches, loc='lower left')

    ax.set_title(f"Count of {get_cdparam_name(cdparam)} measurements over European legal limit", fontsize=15)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)

commune_values = {}
commune_entries = {}
def map_by_commune():
    # Reset dictionaries before aggregation
    cdparam = None
    

    for df in load_folder_data(os.path.join(os.getcwd(), 'data', 'clean')):
        over_limit_df = pd.DataFrame()
        commune_values.clear()
        commune_entries.clear()
        if df.empty:
            continue
        cdparam = df['cdparametre'].iloc[0]
        df = df[df['cddept_x'] == '17']

        if cdparam not in criteres_offi_df['CdParametre'].values or df.empty:
            continue
        limit = float(criteres_offi_df[criteres_offi_df['CdParametre'] == cdparam]['Limit_eu'].values[0])
        if limit is None:
            continue
        for insee, group in get_by_commune(df):
            if insee not in commune_values:
                commune_values[insee] = 0
                commune_entries[insee] = 0
            over_limit_df = pd.concat([over_limit_df, group[group['valtraduite'] > limit]])
            for value in group['valtraduite']:
                
                if value > limit:
                    commune_values[insee] += 1
                    
                commune_entries[insee] += 1

        # Compute percentage for each commune
        commune_percent = {
            str(insee): (commune_values[insee] / commune_entries[insee] * 100 if commune_entries[insee] > 0 else None)
            for insee in commune_values
        }
        commune_df = pd.DataFrame(
            [(insee, value) for insee, value in commune_percent.items()],
            columns=['inseecommuneprinc', 'value']
        )
        print(commune_df.head())
        if cdparam is not None:
            map_communes_per_department(commune_df, 'value', 0, 20, 17, cdparam, save_path=os.path.join(os.getcwd(), 'tempmaps', f'{cdparam}_eu_17.png'))
            output_path = os.path.join(os.getcwd(), 'tempmaps', f'{cdparam}_eu_17_breachs.txt')
            with open(output_path, 'w') as f:
                f.write(over_limit_df.to_csv(sep='\t', index=False))
    

map_by_commune()
