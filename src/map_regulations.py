from Utilities import *
import os
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
import requests
import matplotlib.pyplot as plt
import time

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
        except requests.exceptions.RequestException:
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_FACTOR ** attempt)
            else:
                return None
        except Exception:
            return None
    return None

def plot_commune_exceedances(commune_counts_df, value_column, departement_code, parameter_code, output_path=None):
    departement_str = str(departement_code).zfill(2)
    communes_gdf = fetch_geodata(departement_code=departement_str)
    if communes_gdf is None or communes_gdf.empty:
        return
    communes_gdf['inseecommuneprinc'] = communes_gdf['inseecommuneprinc'].astype(str)
    commune_counts_df['inseecommuneprinc'] = commune_counts_df['inseecommuneprinc'].astype(str)
    merged_gdf = communes_gdf.merge(commune_counts_df, on='inseecommuneprinc', how='left')

    def color_by_exceedance(row):
        if pd.isna(row[value_column]):
            return 'white'
        elif row[value_column] == 0:
            return 'green'
        elif row[value_column] <= 3:
            return 'yellow'
        else:
            return 'red'

    merged_gdf['color'] = merged_gdf.apply(color_by_exceedance, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    merged_gdf.plot(
        color=merged_gdf['color'],
        linewidth=0.8,
        ax=ax,
        edgecolor='0.8'
    )

    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color='grey', label='Missing'),
        mpatches.Patch(color='green', label='Count = 0'),
        mpatches.Patch(color='yellow', label='Count ≤ 3'),
        mpatches.Patch(color='red', label='Count > 3')
    ]
    ax.legend(handles=legend_patches, loc='lower left', title="Legend")
    ax.set_title(f"Exceedances count over French legislation of {get_cdparam_name(parameter_code)} measurements (2016-2025)", fontsize=15)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)

selected_parameters = get_selected_cdparams(os.getcwd())
parameter_reference_df = pd.read_csv(r'C:\Users\mberthie\Documents\StageHubEau\data\PAR_SANDRE_short.txt', sep='\t')
official_criteria_df = pd.read_csv(r'C:\Users\mberthie\Documents\StageHubEau\src\criteresOffi.txt', sep='\t')

total_commune_exceedances = {}
all_exceedance_rows = []
for data_df in tqdm(load_folder_data(os.path.join(os.getcwd(), 'data', 'clean')), desc="Processing files", total=139):
    commune_exceedance_counts = {}
    parameter_code = data_df['cdparametre'].iloc[0]
    data_df = data_df[data_df['cddept_x'] == '17']
    if parameter_code not in official_criteria_df['CdParametre'].values or data_df.empty:
        continue
    threshold = float(official_criteria_df[official_criteria_df['CdParametre'] == parameter_code]['Limit_fr'].values[0])
    if threshold is None:
        continue
    exceedance_rows = []
    for commune_code, group in get_by_commune(data_df):
        if commune_code not in commune_exceedance_counts:
            commune_exceedance_counts[commune_code] = 0
        if commune_code not in total_commune_exceedances:
            total_commune_exceedances[commune_code] = 0
        exceedances = group[group['valtraduite'] > threshold]
        if exceedances.empty:
            continue
        commune_exceedance_counts[commune_code] = exceedances['valtraduite'].count()
        total_commune_exceedances[commune_code] += exceedances['valtraduite'].count()
        exceedance_rows.append(exceedances)
    if exceedance_rows:
        exceedance_df = pd.concat(exceedance_rows)
    else:
        exceedance_df = pd.DataFrame()
    commune_counts_df = pd.DataFrame(list(commune_exceedance_counts.items()), columns=['inseecommuneprinc', 'value'])
    plot_commune_exceedances(commune_counts_df, 'value', 17, parameter_code, output_path=os.path.join(os.getcwd(), f'fr_{parameter_code}_17.png'))
    exceedance_df.to_csv(os.path.join(os.getcwd(), f'fr_{parameter_code}_17.txt'), index=False)
    all_exceedance_rows.append(exceedance_df)

if all_exceedance_rows:
    all_exceedances_df = pd.concat(all_exceedance_rows)
else:
    all_exceedances_df = pd.DataFrame()
all_commune_counts_df = pd.DataFrame(list(total_commune_exceedances.items()), columns=['inseecommuneprinc', 'value'])

def plot_total_commune_exceedances(commune_counts_df, value_column, departement_code, output_path=None):
    departement_str = str(departement_code).zfill(2)
    communes_gdf = fetch_geodata(departement_code=departement_str)
    if communes_gdf is None or communes_gdf.empty:
        return
    communes_gdf['inseecommuneprinc'] = communes_gdf['inseecommuneprinc'].astype(str)
    commune_counts_df['inseecommuneprinc'] = commune_counts_df['inseecommuneprinc'].astype(str)
    merged_gdf = communes_gdf.merge(commune_counts_df, on='inseecommuneprinc', how='left')

    def color_by_exceedance(row):
        if pd.isna(row[value_column]):
            return 'white'
        elif row[value_column] == 0:
            return 'green'
        elif row[value_column] <= 3:
            return 'yellow'
        else:
            return 'red'

    merged_gdf['color'] = merged_gdf.apply(color_by_exceedance, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    merged_gdf.plot(
        color=merged_gdf['color'],
        linewidth=0.8,
        ax=ax,
        edgecolor='0.8'
    )

    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color='grey', label='Missing'),
        mpatches.Patch(color='green', label='Count = 0'),
        mpatches.Patch(color='yellow', label='Count ≤ 3'),
        mpatches.Patch(color='red', label='Count > 3')
    ]
    ax.legend(handles=legend_patches, loc='lower left', title="Legend")
    ax.set_title(f"Exceedances count of water quality measurement over French legislation (2016-2025)", fontsize=15)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)

plot_total_commune_exceedances(all_commune_counts_df, 'value', 17, output_path=os.path.join(os.getcwd(), f'fr_17.png'))
all_exceedances_df.to_csv(os.path.join(os.getcwd(), f'fr_17.txt'), index=False)
