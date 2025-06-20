import glob
import pandas as pd
from tqdm import tqdm
import os
import json
from Utilities import *

def merge_files(file_names, output_file, separator=','):
    list_df = []
    for file in file_names:
        try:
            df = pd.read_csv(file, sep=separator, encoding='latin-1', on_bad_lines='skip')
            list_df.append(df)
        except UnicodeDecodeError:
            print(f"Skipping file due to encoding error: {file}")
    
    if list_df:
        df_merged = pd.concat(list_df, ignore_index=True)
        df_merged.drop_duplicates(inplace=True)
        df_merged.to_csv(output_file, sep='\t', index=False, encoding='latin-1')
    else:
        print("No files were successfully read to merge.")

def merge_com(path):
    file_names = glob.glob(os.path.join(path, 'raw', 'DIS_COM_UDI_*.txt'))
    output_file = os.path.join(path, 'merged', 'DIS_COM_UDI.csv')
    print((output_file))
    merge_files(file_names, output_file)

def merge_results(path):
    file_names = glob.glob(os.path.join(path, 'raw', 'DIS_RESULT_*.txt'))
    short_names = list(set([file_name[-7:-4] for file_name in file_names]))
    for dept in tqdm(short_names, desc='Merging DIS_RESULT'):
        dept_merged_dir = os.path.join(path, 'merged', dept)
        os.makedirs(dept_merged_dir, exist_ok=True)
        file_names_dept = [file_name for file_name in file_names if file_name.endswith(dept + '.txt')]
        all_dept_data = pd.concat([pd.read_csv(file, encoding='latin-1') for file in file_names_dept], ignore_index=True)
        cdparametre_values = all_dept_data['cdparametre'].unique()
        cdparametre_values = [x for x in cdparametre_values if pd.notna(x)]
        grouped = all_dept_data.groupby('cdparametre')
        for cdparametre in cdparametre_values:
            output_file_dept = os.path.join(dept_merged_dir, f'DIS_RESULT_{dept}_{int(cdparametre)}.txt')
            df_cdparametre = grouped.get_group(cdparametre)
            file_exists = os.path.exists(output_file_dept)
            with open(output_file_dept, 'a', encoding='latin-1') as f:
                if not file_exists:
                    header_line = '\t'.join(df_cdparametre.columns.tolist()) + '\n'
                    f.write(header_line)
                df_cdparametre.to_csv(f, mode='a', header=False, sep='\t', index=False, encoding='latin-1', lineterminator='\n')

def merge_plv(path):
    file_names = glob.glob(os.path.join(path, 'raw', 'DIS_PLV_*.txt'))
    short_names = [file_name[:-4][-3:] for file_name in file_names]
    short_names = list(set(short_names))
    for dept in tqdm(short_names, desc='Merging DIS_PLV'):
        file_names_dept = [file_name for file_name in file_names if file_name.endswith(dept + '.txt')]
        output_file_dept = os.path.join(path, 'merged', str(dept), f'DIS_PLV_{dept}.txt')
        merge_files(file_names_dept, output_file_dept)

def analyze(path):
    cdparams_files = {}
    for root, _, files in os.walk(os.path.join(path, 'merged')):
        for file in files:
            if file.startswith('DIS_RESULT'):
                file_path = os.path.join(root, file)
                cdparam = file.split('_')[-1].split('.')[0]
                if cdparam == '0':
                    os.remove(file_path)
                    continue
                if cdparam not in cdparams_files:
                    cdparams_files[cdparam] = []
                cdparams_files[cdparam].append(file_path)

    import matplotlib.pyplot as plt
    
    whole_json = { 'cdparams': [] }
    for cdparam, files in tqdm(cdparams_files.items(), desc='Iterating cdparams'):
        count = {}
        zero_values = 0
        for file in files:
            dept = file.split('_')[-2]
            plv = os.path.join(path, 'merged', dept, 'DIS_PLV_' + dept + '.txt')
            df_plv = pd.read_csv(plv, sep='\t', encoding='latin-1', on_bad_lines='skip')
            df_result = pd.read_csv(file, sep='\t', encoding='latin-1', on_bad_lines='skip')
            df_merged = pd.merge(df_result, df_plv, on='referenceprel', how='left')
            # Now df_merged contains all columns from df_result and df_plv,
            # with the columns from df_plv added based on matching 'referenceprel'.
            # If there's no matching 'referenceprel' in df_plv, the  columns will have NaN values.
            counts = df_merged.groupby('inseecommuneprinc').size().reset_index(name='count')
            zero_values += df_result['valtraduite'].isin([0, 0.0, '0', '0.0']).sum()
            for index, row in counts.iterrows():
                insee = row['inseecommuneprinc']
                c = row['count']
                if insee not in count:
                    count[insee] = 0
                count[insee] += c
        average = sum(count.values()) / len(count) if count else 0
        percentage_zero = (zero_values / sum(count.values())) * 100 if sum(count.values()) > 0 else 0
        cdparam_json = {
            'cdparam': str(cdparam),
            'observations': {
                'average_per_commune': str(average),
                'total_observations': str(sum(count.values())),
                'total_communes': str(len(count)),
                'percentage_zero_values': str(percentage_zero) if 'percentage_zero' in locals() else '0',
                'counts':[{'commune': str(k), 'value': str(v)} for k, v in count.items()]
            }
        }
        stats_file_path = os.path.join(path, 'stats', f'stats_{cdparam}.json')
        os.makedirs(os.path.dirname(stats_file_path), exist_ok=True)
        with open(stats_file_path, 'w') as f:
            json.dump(cdparam_json, f, indent=4)
        cdparam_files_json = {
            'cdparam': str(cdparam),
            'files': files
        }
        whole_json['cdparams'].append(cdparam_files_json)
    with open(os.path.join(path, 'stats', f'files_paths.json'), 'w') as f:
        json.dump(whole_json, f, indent=4)

from tqdm import tqdm
def merge_all(path):
    merged_path = os.path.join(path, 'data/merged')
    cdparams = get_selected_cdparams(path)
    file_names = glob.glob(os.path.join(merged_path, '***/DIS_PLV_*.txt'))
    dfs_plv = []
    for file in tqdm(file_names, desc='Reading DIS_PLV files'):
        df = pd.read_csv(file, sep='\t', encoding='latin-1', on_bad_lines='skip')
        dfs_plv.append(df)
    df_plv = pd.concat(dfs_plv, ignore_index=True)
    df_plv.drop_duplicates(inplace=True)
    
    for cdparametre_data in tqdm(load_cdparam_jsons(cdparams, path), desc='Processing CDPARAMS', total=len(cdparams)):
        df, cdparametre = cdparametre_data
        merged_df = pd.merge(df, df_plv, on='referenceprel', how='left')
        merged_df.drop_duplicates(inplace=True)
        selected_columns = ['cddept_x', 'cdparametre', 'valtraduite', 'inseecommuneprinc', 'dateprel', 'heureprel']
        merged_df = merged_df[selected_columns]
        merged_df = merged_df.dropna(axis=0)
        output_file = os.path.join(path, 'data/clean', f'clean_{cdparametre}.txt')
        merged_df.to_csv(output_file, sep='\t', index=False, encoding='latin-1')

def shorten_param():
    return

if __name__ == '__main__':
    path = '/home/iwta/Documents/Univ/StageHubEau/data'
    #merge_com(path)
    #merge_results(path)
    #merge_plv(path)

    #analyze(path)

    #merge_all('/home/iwta/Documents/Univ/StageHubEau/')

    shorten_param()
    
