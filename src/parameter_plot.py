import pandas as pd
import requests
import geopandas as gpd
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
import numpy as np
import itertools
import seaborn as sns

# Constants
DATA_DIR = 'data'
PLOTS_DIR = 'plots'
GEO_API_URL = 'https://geo.api.gouv.fr/departements/{}/communes?format=geojson&geometry=contour'
MAX_RETRIES = 3
BACKOFF_FACTOR = 2
MIN_CDPARAMETRE_COUNT = 500
OUTLIER_PERCENTILE = 0.05
DEPARTEMENT_CODE = '17'

# --- Data Loading and Filtering ---

def load_and_prepare_data(file_path):
    """Loads data, removes outliers, and filters the DataFrame."""
    df = load_data(file_path)
    df = remove_outliers(df, percentile=OUTLIER_PERCENTILE)
    df = filter_data(df)
    return df

def load_data(file_path):
    """Loads data from a CSV file."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {file_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def is_binary_column(df, column_name):
    """Checks if a column contains only binary values (0.0 and 1.0)."""
    return set(df[column_name].unique()) <= {0.0, 1.0}

def filter_data(df):
    """Filters out rows with missing 'valtraduite' and rare 'cdparametre' values."""
    df = df.dropna(subset=['valtraduite']).copy()
    
    # Identify binary cdparametre values
    binary_cdparametre = [cdparametre for cdparametre, group in df.groupby('cdparametre') if is_binary_column(group, 'valtraduite')]
    
    # Identify infrequent cdparametre values
    value_counts = df['cdparametre'].value_counts()
    infrequent_cdparametre = value_counts[value_counts < MIN_CDPARAMETRE_COUNT].index.tolist()
    
    # Filter out rows with binary or infrequent cdparametre values
    df = df[~df['cdparametre'].isin(binary_cdparametre + infrequent_cdparametre)].copy()
    
    return df

def remove_outliers(df, percentile=0.05):
    """Removes outliers from 'valtraduite' column based on percentiles, grouped by 'cdparametre'."""
    def filter_group(group):
        lower_bound = group['valtraduite'].quantile(percentile)
        upper_bound = group['valtraduite'].quantile(1 - percentile)
        return group[(group['valtraduite'] >= lower_bound) & (group['valtraduite'] <= upper_bound)]

    df_filtered = df.groupby('cdparametre', group_keys=False).apply(filter_group).reset_index(drop=True)
    return df_filtered

# --- Data Aggregation ---

def calculate_average_valtraduite(df):
    """Calculates the average 'valtraduite' grouped by 'cdparametre' and 'inseecommuneprinc'."""
    return df.groupby(['cdparametre', 'inseecommuneprinc'])['valtraduite'].mean().reset_index(name='average_valtraduite')

def calculate_min_max_valtraduite(df):
    """Calculates the min and max 'valtraduite' for each 'cdparametre'."""
    return df.groupby('cdparametre')['valtraduite'].agg(['min', 'max']).reset_index().rename(columns={'min': 'min_valtraduite', 'max': 'max_valtraduite'})

# --- GeoData Fetching ---

def fetch_geodata(departement_code=DEPARTEMENT_CODE):
    """Fetches GeoJSON data for a given departement code with retry logic."""
    url = GEO_API_URL.format(departement_code)
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            gdf = gpd.GeoDataFrame.from_features(data['features'])
            gdf.rename(columns={'code': 'inseecommuneprinc'}, inplace=True)
            gdf['inseecommuneprinc'] = gdf['inseecommuneprinc'].astype(str)
            return gdf
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                delay = BACKOFF_FACTOR ** attempt
                time.sleep(delay)
            else:
                print("Max retries reached. Failed to fetch geodata.")
                return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
    return None

# --- Plotting ---

def plot_parametre(gdf, average_values, min_val, max_val, parametre, unite, lbcourt):
    """Plots the average 'valtraduite' on a map."""
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

def plot_all_parametre():
    """Generates plots for all unique 'cdparametre' values."""
    file_path = os.path.join(DATA_DIR, 'merged_data.csv')
    try:
        df = load_and_prepare_data(file_path)
        
        # Grouping outside the loop for efficiency
        parametre_info = df.groupby('cdparametre').agg({'cdunitereferencesiseeaux': 'first', 'LbCourtParametre': 'first'}).reset_index()
        
        if df.empty:
            print("Filtered DataFrame is empty. Cannot proceed.")
            return

        average_values = calculate_average_valtraduite(df)
        min_max_values = calculate_min_max_valtraduite(df)
        geodata = fetch_geodata()

        if geodata is None:
            print("Failed to fetch geodata. Cannot generate plots.")
            return
        
        geodata['inseecommuneprinc'] = geodata['inseecommuneprinc'].astype('int64')

        for cdparametre in tqdm(df['cdparametre'].unique(), desc="Processing parameters"):
            # Filtering inside the loop
            average_values_filtered = average_values[average_values['cdparametre'] == cdparametre]
            min_max_values_filtered = min_max_values[min_max_values['cdparametre'] == cdparametre]
            
            if min_max_values_filtered.empty:
                print(f"No min/max values found for cdparametre {cdparametre}. Skipping.")
                continue
            
            min_val = min_max_values_filtered['min_valtraduite'].iloc[0]
            max_val = min_max_values_filtered['max_valtraduite'].iloc[0]
            
            param_info = parametre_info[parametre_info['cdparametre'] == cdparametre]
            
            if param_info.empty:
                print(f"No parameter info found for cdparametre {cdparametre}. Skipping.")
                continue
            
            unite = param_info['cdunitereferencesiseeaux'].values[0]
            lbcourt = param_info['LbCourtParametre'].values[0]
            
            plot_parametre(geodata, average_values_filtered, min_val, max_val, cdparametre, unite, lbcourt)

    except Exception as e:
        print(f"An error occurred: {e}")

# --- Seasonal Analysis ---

def calculate_seasonal_average_valtraduite(df):
    """Calculates the average 'valtraduite' for each season, grouped by 'cdparametre'."""
    def assign_season(date):
        year = date.year
        month = date.month
        if 3 <= month <= 5:
            return f'Spring{year}'
        elif 6 <= month <= 8:
            return f'Summer{year}'
        elif 9 <= month <= 11:
            return f'Autumn{year}'
        else:
            return f'Winter{year}'

    df = df.copy()
    df['dateprel'] = pd.to_datetime(df['dateprel'], errors='coerce')
    df = df.dropna(subset=['dateprel']).copy()
    df['season'] = df['dateprel'].apply(assign_season)
    seasonal_average = df.groupby(['cdparametre', 'season'])['valtraduite'].mean().reset_index(name='average_valtraduite')

    return seasonal_average

def detect_seasonality_fourier(seasonal_average, parametre_info):
    """Computes the Fourier Transform of seasonal averages and plots the power spectrum."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    for cdparametre in seasonal_average['cdparametre'].unique():
        param_seasonal = seasonal_average[seasonal_average['cdparametre'] == cdparametre].copy()
        
        # Sort by season (chronologically)
        season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
        param_seasonal['year'] = param_seasonal['season'].str[-4:].astype(int)
        param_seasonal['season_idx'] = param_seasonal['season'].str[:-4].apply(lambda x: season_order.index(x))
        param_seasonal = param_seasonal.sort_values(['year', 'season_idx'])
        values = param_seasonal['average_valtraduite'].values

        if len(values) < 8:
            print(f"Not enough data for {cdparametre} to assess seasonality (Fourier).")
            continue

        lbcourt = parametre_info[parametre_info['cdparametre'] == cdparametre]['LbCourtParametre'].values[0]

        # Remove mean to focus on periodicity
        values_centered = values - np.mean(values)
        n = len(values_centered)
        freq = np.fft.rfftfreq(n, d=1)  # d=1: one sample per season
        fft_vals = np.fft.rfft(values_centered)
        power = np.abs(fft_vals)**2

        # Ignore the zero frequency (mean)
        if len(power) > 1:
            dominant_idx = np.argmax(power[1:]) + 1
            dominant_freq = freq[dominant_idx]
            if dominant_freq != 0:
                period = 1 / dominant_freq
                period_text = f"Dominant period ≈ {period:.2f} seasons"
            else:
                period_text = "No dominant frequency detected."
        else:
            period_text = "Not enough frequency components."

        # Plot the power spectrum
        plt.figure(figsize=(6, 3))
        plt.stem(freq[1:], power[1:])
        plt.title(f'Fourier Power Spectrum for {lbcourt}')
        plt.xlabel('Frequency (cycles/season)')
        plt.ylabel('Power')
        plt.text(0.5, 0.9, period_text, transform=plt.gca().transAxes, horizontalalignment='center', verticalalignment='top')
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/plot_{lbcourt}_fourierPeriodicity.png")
        plt.close()

def plot_seasonal_parametre(file_path=os.path.join(DATA_DIR, 'merged_data.csv')):
    """Plots seasonal 'valtraduite' data for each 'cdparametre'."""
    try:
        df = load_and_prepare_data(file_path)
        df['dateprel'] = pd.to_datetime(df['dateprel'], errors='coerce')
        df = df.dropna(subset=['dateprel']).copy()
        parametre_info = df.groupby('cdparametre').agg({'cdunitereferencesiseeaux': 'first', 'LbCourtParametre': 'first'}).reset_index()
        seasonal_average = calculate_seasonal_average_valtraduite(df)
        detect_seasonality_fourier(seasonal_average, parametre_info)

        if df.empty:
            print("Filtered DataFrame is empty. Cannot proceed.")
            return

        os.makedirs(PLOTS_DIR, exist_ok=True)
        for cdparametre in tqdm(df['cdparametre'].unique(), desc="Processing parameters"):
            df_parametre = df[df['cdparametre'] == cdparametre]
            seasonal_average_filtered = seasonal_average[seasonal_average['cdparametre'] == cdparametre]
            lbcourt = parametre_info[parametre_info['cdparametre'] == cdparametre]['LbCourtParametre'].values[0]

            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Scatter plot of all entries
            ax.scatter(df_parametre['dateprel'], df_parametre['valtraduite'], label='Data', alpha=0.5)
            
            # Line plot of seasonal averages
            
            seasonal_labels = seasonal_average_filtered['season'].unique()
            seasonal_averages = []
            seasonal_dates = []
            
            season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
            
            # Sort seasons chronologically
            seasonal_labels = sorted(seasonal_labels, key=lambda x: (int(x[-4:]), season_order.index(x[:-4])))
            
            for season in seasonal_labels:
                season_data = seasonal_average_filtered[seasonal_average_filtered['season'] == season]
                if not season_data.empty:
                    seasonal_averages.append(season_data['average_valtraduite'].iloc[0])
                    year = season[len(season)-4:]
                    # Find the middle date for each season
                    if "Winter" in season:
                        seasonal_dates.append(pd.to_datetime(f'{year}-01-15'))
                    elif "Spring" in season:
                        seasonal_dates.append(pd.to_datetime(f'{year}-04-15'))
                    elif "Summer" in season:
                        seasonal_dates.append(pd.to_datetime(f'{year}-07-15'))
                    elif "Autumn" in season:
                        seasonal_dates.append(pd.to_datetime(f'{year}-10-15'))
                else:
                    seasonal_averages.append(None)
                    seasonal_dates.append(None)
            
            # Filter out None values
            seasonal_dates = [date for date in seasonal_dates if date is not None]
            seasonal_averages = [avg for avg in seasonal_averages if avg is not None]
            
            ax.plot(seasonal_dates, seasonal_averages, marker='o', linestyle='-', color='red', label='Seasonal Average')

            ax.set_xlabel('Date')
            ax.set_ylabel('valtraduite')
            ax.set_title(f'Seasonal valtraduite for {lbcourt}')
            ax.legend()

            filename = os.path.join(PLOTS_DIR, f'plot_{lbcourt}_seasons.png')
            plt.savefig(filename)
            plt.close()

    except Exception as e:
        print(f"An error occurred: {e}")

# --- Correlation Analysis ---

def find_pairwise_correlations(df, plot=True):
    """
    Finds pairwise correlations between 'cdparametre' values based on shared dates.
    """
    df = df.copy()
    df['dateprel'] = pd.to_datetime(df['dateprel'], errors='coerce')
    df = df.dropna(subset=['dateprel', 'cdparametre', 'valtraduite']).copy()

    param_codes = df['cdparametre'].unique()
    n_params = len(param_codes)
    results = []

    corr_matrix = pd.DataFrame(index=param_codes, columns=param_codes)

    for param1, param2 in tqdm(itertools.combinations(param_codes, 2), total=n_params * (n_params - 1) // 2, desc="Processing parameter pairs"):
        # Efficiently merge on date
        merged = pd.merge(
            df[df['cdparametre'] == param1][['dateprel', 'valtraduite']].rename(columns={'valtraduite': 'val1'}),
            df[df['cdparametre'] == param2][['dateprel', 'valtraduite']].rename(columns={'valtraduite': 'val2'}),
            on='dateprel',
            how='inner'
        )

        if len(merged) < 2:
            corr = float('nan')
        else:
            corr = merged['val1'].corr(merged['val2'])

        corr_matrix.loc[param1, param2] = corr
        corr_matrix.loc[param2, param1] = corr

        results.append({
            'param1': param1,
            'param2': param2,
            'n_common_dates': len(merged),
            'correlation': corr
        })

    for param in param_codes:
        corr_matrix.loc[param, param] = 1.0

    if plot:
        plot_correlation_matrix(corr_matrix)

    return pd.DataFrame(results)

def plot_correlation_matrix(corr_matrix):
    """Plots the correlation matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix.astype(float), annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Pairwise Correlation Matrix of cdparametre')
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'pairwise_correlation_matrix.png'))
    plt.close()

# --- Main Execution ---

if __name__ == '__main__':
    os.makedirs(PLOTS_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, 'merged_data.csv')
    
    plot_all_parametre()
    plot_seasonal_parametre()
    
    try:
        df = load_and_prepare_data(file_path)
        correlations_df = find_pairwise_correlations(df)
        print(correlations_df.sort_values('correlation', ascending=False))
    except Exception as e:
        print(f"An error occurred during correlation analysis: {e}")

