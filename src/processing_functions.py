import pandas as pd

MIN_CDPARAMETRE_COUNT = 500
OUTLIER_PERCENTILE = 0.05
PERCENTAGE_THRESHOLD = 25.0

def load_and_prepare_data(file_path, debug=False):
    """Loads data, removes outliers, and filters the DataFrame."""
    df = load_data(file_path, debug=debug)
    df = df[df['valtraduite'].notna()]
    df = remove_cdparametre_with_high_zero_percentage(df, percentage_threshold=PERCENTAGE_THRESHOLD)
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

def remove_cdparametre_with_high_zero_percentage(df, percentage_threshold=25.0):
    def percentage_of_zeros(group):
        total = len(group)
        zeros = len(group[group == 0.0])
        return (zeros / total) * 100 if total > 0 else 0

    zero_percentage = df.groupby('cdparametre')['valtraduite'].apply(percentage_of_zeros)
    cdparametres_to_remove = zero_percentage[zero_percentage > percentage_threshold].index
    df = df[~df['cdparametre'].isin(cdparametres_to_remove)]

    return df