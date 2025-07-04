import os
import json
import pandas as pd
from sklearn.linear_model import LinearRegression

def linear_regression(df, time_col, value_col):
    time = (df[time_col] - df[time_col].min()).dt.days.values.reshape(-1, 1)
    values = df[value_col].values

    model = LinearRegression()
    model.fit(time, values)

    slope = model.coef_[0]
    intercept = model.intercept_

    return slope, intercept

def get_cdparam_nameshort(cdparam):
    path = os.path.join(os.getcwd(), 'data/PAR_SANDRE_short.txt')
    df = pd.read_csv(path, sep='\t', encoding='latin-1')
    try:
        cdparam_int = int(float(cdparam))
    except ValueError:
        return None
    print("cdparam_int:", cdparam_int)
    if cdparam_int in df['CdParametre'].values:
        return df.loc[df['CdParametre'] == cdparam_int, 'LbCourtParametre'].values[0]

def get_cdparam_name(cdparam):
    path = os.path.join(os.getcwd(), 'data/PAR_SANDRE_short.txt')
    df = pd.read_csv(path, sep='\t', encoding='latin-1')
    if cdparam in df['CdParametre'].values:
        return df.loc[df['CdParametre'] == cdparam, 'NomParametre'].values[0]

def get_selected_cdparams(path):
    with open(os.path.join(path, 'extract', 'cdparams_selected.txt'), 'r') as f:
        cdparams_selected = [line.strip() for line in f if line.strip()]
    return cdparams_selected

def get_by_commune(df):
    for insee, group in df.groupby('inseecommuneprinc'):
        yield insee, group

def load_cdparam_jsons(cdparams, path = '/home/iwta/Documents/Univ/StageHubEau/'):
    with open(os.path.join(path, 'data/stats', 'files_paths.json'), 'r') as f:
        list_files = json.load(f)
    files = {entry['cdparam']: entry['files'] for entry in list_files['cdparams']}

    for cdparam in cdparams:
        dfs = []
        for path in files[cdparam]:
            df = pd.read_csv(path, sep='\t')  # Specify the correct separator
            dfs.append(df)
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df['cdparam'] = cdparam
            yield combined_df, cdparam

def load_folder_data(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                full_path = os.path.join(root, file)
                df = pd.read_csv(full_path, sep='\t', encoding='latin-1', on_bad_lines='skip', low_memory=False)
                yield df