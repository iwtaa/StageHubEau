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
    import gc

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

    # Preload all PLV files into memory to reduce disk I/O
    plv_files = {}
    for root, _, files in os.walk(os.path.join(path, 'merged')):
        for file in files:
            if file.startswith('DIS_PLV_') and file.endswith('.txt'):
                dept = file.split('_')[-1].split('.')[0]
                file_path = os.path.join(root, file)
                if dept not in plv_files:
                    try:
                        # Use low_memory=False and dtype optimization
                        plv_files[dept] = pd.read_csv(
                            file_path, sep='\t', encoding='latin-1', on_bad_lines='skip', low_memory=False, dtype=str
                        )
                    except Exception as e:
                        print(f"Error reading PLV file {file_path}: {e}")

    whole_json = {'cdparams': []}

    def process_cdparam(cdparam, files):
        count = {}
        zero_values = 0
        for file in files:
            dept = file.split('_')[-2]
            df_plv = plv_files.get(dept)
            if df_plv is None:
                continue
            try:
                # Use dtype=str to avoid dtype inference overhead
                df_result = pd.read_csv(file, sep='\t', encoding='latin-1', on_bad_lines='skip', low_memory=False, dtype=str)
            except Exception as e:
                print(f"Error reading result file {file}: {e}")
                continue

            # Use merge only on needed columns to reduce memory
            if 'referenceprel' in df_result.columns and 'referenceprel' in df_plv.columns:
                df_merged = pd.merge(
                    df_result[['referenceprel', 'inseecommuneprinc', 'valtraduite']] if 'inseecommuneprinc' in df_result.columns and 'valtraduite' in df_result.columns else df_result,
                    df_plv[['referenceprel', 'inseecommuneprinc']] if 'inseecommuneprinc' in df_plv.columns else df_plv,
                    on='referenceprel', how='left', suffixes=('', '_plv')
                )
            else:
                continue

            # Use value_counts for faster counting
            if 'inseecommuneprinc' in df_merged.columns:
                counts = df_merged['inseecommuneprinc'].value_counts()
                for insee, c in counts.items():
                    if pd.isna(insee):
                        continue
                    count[insee] = count.get(insee, 0) + int(c)
            # Optimize zero value counting
            if 'valtraduite' in df_result.columns:
                zero_values += df_result['valtraduite'].isin(['0', '0.0', 0, 0.0]).sum()

            del df_result, df_merged
            gc.collect()

        total_obs = sum(count.values())
        average = total_obs / len(count) if count else 0
        percentage_zero = (zero_values / total_obs) * 100 if total_obs > 0 else 0
        cdparam_json = {
            'cdparam': str(cdparam),
            'observations': {
                'average_per_commune': str(average),
                'total_observations': str(total_obs),
                'total_communes': str(len(count)),
                'percentage_zero_values': str(percentage_zero),
                'counts': [{'commune': str(k), 'value': str(v)} for k, v in count.items()]
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
        return cdparam_files_json

    # Use ThreadPoolExecutor for parallel processing
    results = []
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [executor.submit(process_cdparam, cdparam, files) for cdparam, files in cdparams_files.items()]
        for f in tqdm(as_completed(futures), total=len(futures), desc='Processing cdparams'):
            result = f.result()
            results.append(result)

    whole_json['cdparams'].extend(results)
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

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
def shorten_param_csv():
    input_filename='data/PAR_20250523_SANDRE.csv'
    output_filename='data/PAR_SANDRE_short.txt'
    delimiter=';'
    indices_to_keep=[0, 1, 6, 7, 10, 184]
    cdparams = get_selected_cdparams('C:/Users/mberthie/Documents/StageHubEau/')

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_filename, delimiter=delimiter, encoding='latin-1', dtype=str)
    df = df.rename(columns={'ï»¿CdParametre': 'CdParametre'})
    df = df.iloc[1:, indices_to_keep]
    df = df[df['CdParametre'].isin(cdparams)]
    print(df.columns.tolist())
    df.to_csv(output_filename, sep='\t', index=False, encoding='latin-1')

if __name__ == '__main__':
    path = os.path.join(os.getcwd(), 'data')
    #merge_com(path)
    #merge_results(path)
    #merge_plv(path)

    #analyze(path)

    #merge_all('C:/Users/mberthie/Documents/StageHubEau/')

    shorten_param_csv()
    
