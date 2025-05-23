import pandas as pd

MIN_CDPARAMETRE_COUNT = 500
OUTLIER_PERCENTILE = 0.05

def load_and_prepare_data(file_path, debug=False):
    """Loads data, removes outliers, and filters the DataFrame."""
    df = load_data(file_path, debug=debug)
    if debug:
        print(f"[DEBUG] Loaded data with {len(df)} rows.")
    df = remove_outliers(df, debug=debug)
    if debug:
        print(f"[DEBUG] Data after outlier removal has {len(df)} rows.")
    df = filter_data(df, debug=debug)
    if debug:
        print(f"[DEBUG] Data after filtering has {len(df)} rows.")
    return df

def load_data(file_path, debug=False):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        if debug:
            print(f"[DEBUG] Successfully loaded file: {file_path}. Rows: {len(df)}.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {file_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def is_binary_column(df, column_name, debug=False):
    """Checks if a column contains only binary values (0.0 and 1.0)."""
    unique_vals = set(df[column_name].unique())
    is_binary = unique_vals <= {0.0, 1.0}
    return is_binary

def filter_data(df, debug=False):
    """Filters out rows with missing 'valtraduite' and rare 'cdparametre' values."""
    init_row_count = len(df)
    df = df.dropna(subset=['valtraduite']).copy()
    na_dropped = init_row_count - len(df)
    if debug:
        print(f"[DEBUG] Dropped {na_dropped} rows due to missing 'valtraduite'.")

    # Identify binary cdparametre
    binary_cdparametre = []
    for cdparametre, group in df.groupby('cdparametre'):
        if is_binary_column(group, 'valtraduite', debug=debug):
            binary_cdparametre.append(cdparametre)
    if debug:
        print(f"[DEBUG] Binary cdparametre found: {binary_cdparametre}")

    # Identify infrequent cdparametre
    value_counts = df['cdparametre'].value_counts()
    infrequent_cdparametre = value_counts[value_counts < MIN_CDPARAMETRE_COUNT].index.tolist()
    if debug:
        print(f"[DEBUG] Infrequent cdparametre (occurrences < {MIN_CDPARAMETRE_COUNT}): {infrequent_cdparametre}")

    to_filter = set(binary_cdparametre + infrequent_cdparametre)
    if debug:
        print(f"[DEBUG] Total cdparametre to filter: {len(to_filter)}")

    before_filter = len(df)
    df = df[~df['cdparametre'].isin(to_filter)].copy()
    filtered_rows = before_filter - len(df)
    if debug:
        print(f"[DEBUG] Filtered out {filtered_rows} rows based on cdparametre criteria.")

    return df

def remove_outliers(df, percentile=OUTLIER_PERCENTILE, debug=False):
    """Removes outliers from 'valtraduite' column based on percentiles, grouped by 'cdparametre'."""
    def filter_group(group):
        count_before = len(group)
        lower_bound = group['valtraduite'].quantile(percentile)
        upper_bound = group['valtraduite'].quantile(1 - percentile)
        filtered = group[(group['valtraduite'] >= lower_bound) & (group['valtraduite'] <= upper_bound)]
        if debug:
            print(f"[DEBUG] cdparametre: {group.name} | Count before: {count_before}, after: {len(filtered)}, lower_bound: {lower_bound}, upper_bound: {upper_bound}")
        return filtered

    df_filtered = df.groupby('cdparametre', group_keys=False).apply(filter_group).reset_index(drop=True)
    if debug:
        print(f"[DEBUG] Total rows after outlier removal: {len(df_filtered)} (from {len(df)} rows)")
    return df_filtered

# --- Data Aggregation ---

def calculate_average_valtraduite(df, debug=False):
    """Calculates the average 'valtraduite' grouped by 'cdparametre' and 'inseecommuneprinc'."""
    result = df.groupby(['cdparametre', 'inseecommuneprinc'])['valtraduite'].mean().reset_index(name='average_valtraduite')
    if debug:
        print(f"[DEBUG] Calculated average valtraduite for {result['cdparametre'].nunique()} unique cdparametre values across {len(result)} groups.")
    return result

def calculate_min_max_valtraduite(df, debug=False):
    """Calculates the min and max 'valtraduite' for each 'cdparametre'."""
    result = df.groupby('cdparametre')['valtraduite'].agg(['min', 'max']).reset_index().rename(columns={'min': 'min_valtraduite', 'max': 'max_valtraduite'})
    if debug:
        print(f"[DEBUG] Calculated min and max for {len(result)} cdparametre groups.")
    return result