import pandas as pd
import requests
import geopandas as gpd
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
from scipy import stats
import numpy as np

DATA_DIR = 'data'
PLOTS_DIR = 'plots'
GEO_API_URL = 'https://geo.api.gouv.fr/departements/{}/communes?format=geojson&geometry=contour'
MAX_RETRIES = 3
BACKOFF_FACTOR = 2
MIN_CDPARAMETRE_COUNT = 500

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {file_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def is_binary_column(df, column_name):
    return set(df[column_name].unique()) <= {0.0, 1.0}

def filter_data(df):
    initial_length = len(df)
    df = df.dropna(subset=['valtraduite'])
    df = df[~df['cdparametre'].isin([cdparametre for cdparametre, group in df.groupby('cdparametre') if is_binary_column(group, 'valtraduite')])]
    df = df[~df['cdparametre'].isin(df['cdparametre'].value_counts()[df['cdparametre'].value_counts() < MIN_CDPARAMETRE_COUNT].index.tolist())]
    return df

def remove_outliers(df, percentile=0.05):
    """
    Removes outliers from 'valtraduite' column in a DataFrame based on percentiles, grouped by 'cdparametre'.

    Args:
        df (pd.DataFrame): The input DataFrame.
        percentile (float): The percentile to use for removing outliers.  Values below this percentile or above 1-this percentile will be removed.

    Returns:
        pd.DataFrame: The DataFrame with outliers removed.
    """
    def filter_group(group):
        lower_bound = group['valtraduite'].quantile(percentile)
        upper_bound = group['valtraduite'].quantile(1 - percentile)
        return group[(group['valtraduite'] >= lower_bound) & (group['valtraduite'] <= upper_bound)]

    df_filtered = df.groupby('cdparametre').apply(filter_group).reset_index(drop=True)
    return df_filtered

def calculate_average_valtraduite(df):
    return df.groupby(['cdparametre', 'inseecommuneprinc'])['valtraduite'].mean().reset_index(name='average_valtraduite')

def calculate_min_max_valtraduite(df):
    return df.groupby('cdparametre')['valtraduite'].agg(['min', 'max']).reset_index().rename(columns={'min': 'min_valtraduite', 'max': 'max_valtraduite'})

def fetch_geodata(departement_code='17'):
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
            if attempt < MAX_RETRIES - 1:
                delay = BACKOFF_FACTOR ** attempt
                time.sleep(delay)
            else:
                return None
        except Exception as e:
            return None
    return None

def plot_parametre(gdf, average_values, min_val, max_val, parametre, unite, lbcourt):
    merged = gdf.merge(average_values, on='inseecommuneprinc', how='left')

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    merged.plot(
        column='average_valtraduite',
        cmap='viridis',
        linewidth=0.8,
        ax=ax,
        edgecolor='0.8',
        legend=True,
        legend_kwds={'label': f"Average {lbcourt} ({unite})", 'orientation': "horizontal"},
        missing_kwds={"color": "lightgrey", "edgecolor": "0.8", "label": "Missing values"},
        vmin=min_val,
        vmax=max_val
    )

    ax.set_title(f'{lbcourt} ({unite})', fontsize=15)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    os.makedirs(PLOTS_DIR, exist_ok=True)
    filename = os.path.join(PLOTS_DIR, f'plot_{lbcourt}.png')
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    file_path = os.path.join(DATA_DIR, 'merged_data.csv')
    try:
        df = load_data(file_path)
        df_filtered = remove_outliers(df)
        df_filtered = filter_data(df_filtered)
        parametre_info = parametre_info = df_filtered.groupby('cdparametre').agg({'cdunitereferencesiseeaux': 'first', 'LbCourtParametre': 'first'}).reset_index()
        if df_filtered.empty:
            print("Filtered DataFrame is empty. Cannot proceed.")
        else:
            average_values = calculate_average_valtraduite(df_filtered)
            min_max_values = calculate_min_max_valtraduite(df_filtered)
            geodata = fetch_geodata()
            geodata['inseecommuneprinc'] = geodata['inseecommuneprinc'].astype('int64')

            if geodata is None:
                print("Failed to fetch geodata. Cannot generate plots.")
            else:
                for cdparametre in tqdm(df_filtered['cdparametre'].unique(), desc="Processing parameters"):
                    average_values_filtered = average_values[average_values['cdparametre'] == cdparametre]
                    min_max_values_filtered = min_max_values[min_max_values['cdparametre'] == cdparametre]
                    if not min_max_values_filtered.empty:
                        min_val = min_max_values_filtered['min_valtraduite'].iloc[0]
                        max_val = min_max_values_filtered['max_valtraduite'].iloc[0]
                        unite = parametre_info[parametre_info['cdparametre'] == cdparametre]['cdunitereferencesiseeaux'].values[0]
                        lbcourt = parametre_info[parametre_info['cdparametre'] == cdparametre]['LbCourtParametre'].values[0]
                        plot_parametre(geodata, average_values_filtered, min_val, max_val, cdparametre, unite, lbcourt)
    except Exception as e:
        print(f"An error occurred: {e}")
