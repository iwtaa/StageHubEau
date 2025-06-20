import os
import json
import pandas as pd

def get_selected_cdparams(path):
    with open(os.path.join(path, 'extract', 'cdparams_selected.txt'), 'r') as f:
        cdparams_selected = [line.strip() for line in f if line.strip()]
    return cdparams_selected

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
                df = pd.read_csv(full_path, sep='\t', encoding='latin-1', on_bad_lines='skip')
                yield df