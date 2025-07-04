import glob
import pandas as pd
from tqdm import tqdm
import os
import json
from Utilities import *
from concurrent.futures import ThreadPoolExecutor, as_completed

def merge_files(file_names, output_file, separator=','):
    dataframes = []
    for file in file_names:
        try:
            df = pd.read_csv(file, sep=separator, encoding='latin-1', on_bad_lines='skip')
            dataframes.append(df)
        except UnicodeDecodeError:
            continue
    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
        merged_df.drop_duplicates(inplace=True)
        merged_df.to_csv(output_file, sep='\t', index=False, encoding='latin-1')

def merge_com(path):
    input_files = glob.glob(os.path.join(path, 'raw', 'DIS_COM_UDI_*.txt'))
    output_file = os.path.join(path, 'merged', 'DIS_COM_UDI.csv')
    merge_files(input_files, output_file)

def merge_results(path):
    input_files = glob.glob(os.path.join(path, 'raw', 'DIS_RESULT_*.txt'))
    departments = list(set([file_name[-7:-4] for file_name in input_files]))
    for dept in tqdm(departments, desc='Merging DIS_RESULT'):
        dept_dir = os.path.join(path, 'merged', dept)
        os.makedirs(dept_dir, exist_ok=True)
        dept_files = [file_name for file_name in input_files if file_name.endswith(dept + '.txt')]
        dept_data = pd.concat([pd.read_csv(file, encoding='latin-1') for file in dept_files], ignore_index=True)
        param_values = [x for x in dept_data['cdparametre'].unique() if pd.notna(x)]
        grouped = dept_data.groupby('cdparametre')
        for param in param_values:
            output_file = os.path.join(dept_dir, f'DIS_RESULT_{dept}_{int(param)}.txt')
            param_data = grouped.get_group(param)
            file_exists = os.path.exists(output_file)
            with open(output_file, 'a', encoding='latin-1') as f:
                if not file_exists:
                    header = '\t'.join(param_data.columns.tolist()) + '\n'
                    f.write(header)
                param_data.to_csv(f, mode='a', header=False, sep='\t', index=False, encoding='latin-1', lineterminator='\n')

def merge_plv(path):
    input_files = glob.glob(os.path.join(path, 'raw', 'DIS_PLV_*.txt'))
    departments = list(set([file_name[:-4][-3:] for file_name in input_files]))
    for dept in tqdm(departments, desc='Merging DIS_PLV'):
        dept_files = [file_name for file_name in input_files if file_name.endswith(dept + '.txt')]
        output_file = os.path.join(path, 'merged', str(dept), f'DIS_PLV_{dept}.txt')
        merge_files(dept_files, output_file)

def analyze(path):
    import gc
    cdparam_files = {}
    for root, _, files in os.walk(os.path.join(path, 'merged')):
        for file in files:
            if file.startswith('DIS_RESULT'):
                file_path = os.path.join(root, file)
                cdparam = file.split('_')[-1].split('.')[0]
                if cdparam == '0':
                    os.remove(file_path)
                    continue
                if cdparam not in cdparam_files:
                    cdparam_files[cdparam] = []
                cdparam_files[cdparam].append(file_path)
    plv_data = {}
    for root, _, files in os.walk(os.path.join(path, 'merged')):
        for file in files:
            if file.startswith('DIS_PLV_') and file.endswith('.txt'):
                dept = file.split('_')[-1].split('.')[0]
                file_path = os.path.join(root, file)
                if dept not in plv_data:
                    try:
                        plv_data[dept] = pd.read_csv(file_path, sep='\t', encoding='latin-1', on_bad_lines='skip', low_memory=False, dtype=str)
                    except Exception:
                        continue
    stats_summary = {'cdparams': []}
    def process_param(param, files):
        commune_counts = {}
        zero_count = 0
        for file in files:
            dept = file.split('_')[-2]
            plv_df = plv_data.get(dept)
            if plv_df is None:
                continue
            try:
                result_df = pd.read_csv(file, sep='\t', encoding='latin-1', on_bad_lines='skip', low_memory=False, dtype=str)
            except Exception:
                continue
            if 'referenceprel' in result_df.columns and 'referenceprel' in plv_df.columns:
                merge_cols = ['referenceprel']
                if 'inseecommuneprinc' in result_df.columns and 'valtraduite' in result_df.columns:
                    merge_cols += ['inseecommuneprinc', 'valtraduite']
                merged_df = pd.merge(
                    result_df[merge_cols] if len(merge_cols) > 1 else result_df,
                    plv_df[['referenceprel', 'inseecommuneprinc']] if 'inseecommuneprinc' in plv_df.columns else plv_df,
                    on='referenceprel', how='left', suffixes=('', '_plv')
                )
            else:
                continue
            if 'inseecommuneprinc' in merged_df.columns:
                counts = merged_df['inseecommuneprinc'].value_counts()
                for insee, count_val in counts.items():
                    if pd.isna(insee):
                        continue
                    commune_counts[insee] = commune_counts.get(insee, 0) + int(count_val)
            if 'valtraduite' in result_df.columns:
                zero_count += result_df['valtraduite'].isin(['0', '0.0', 0, 0.0]).sum()
            del result_df, merged_df
            gc.collect()
        total_obs = sum(commune_counts.values())
        avg_per_commune = total_obs / len(commune_counts) if commune_counts else 0
        zero_percentage = (zero_count / total_obs) * 100 if total_obs > 0 else 0
        param_stats = {
            'cdparam': str(param),
            'observations': {
                'average_per_commune': str(avg_per_commune),
                'total_observations': str(total_obs),
                'total_communes': str(len(commune_counts)),
                'percentage_zero_values': str(zero_percentage),
                'counts': [{'commune': str(k), 'value': str(v)} for k, v in commune_counts.items()]
            }
        }
        stats_file = os.path.join(path, 'stats', f'stats_{param}.json')
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump(param_stats, f, indent=4)
        return {'cdparam': str(param), 'files': files}
    results = []
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [executor.submit(process_param, param, files) for param, files in cdparam_files.items()]
        for f in tqdm(as_completed(futures), total=len(futures), desc='Processing cdparams'):
            results.append(f.result())
    stats_summary['cdparams'].extend(results)
    with open(os.path.join(path, 'stats', 'files_paths.json'), 'w') as f:
        json.dump(stats_summary, f, indent=4)

def merge_all():
    with open('extract/cdparams_selected.txt', 'r') as f:
        selected_params = [int(line.strip()) for line in f if line.strip()]
    param_df = pd.read_csv('data/PAR_SANDRE_short.txt', delimiter='\t')
    base_path = os.getcwd()
    result_files = glob.glob(os.path.join(base_path, 'data', 'merged', '*/DIS_RESULT_*.txt'))
    plv_files = glob.glob(os.path.join(base_path, 'data', 'merged', '*/DIS_PLV_*.txt'))
    plv_dfs = []
    for file in tqdm(plv_files, desc='Reading DIS_PLV files'):
        plv_dfs.append(pd.read_csv(file, sep='\t', encoding='latin-1', on_bad_lines='skip'))
    merged_plv_df = pd.concat(plv_dfs, ignore_index=True)
    merged_plv_df.drop_duplicates(inplace=True)
    for param in tqdm(selected_params, desc='Processing cdparam'):
        param_dfs = []
        for file_name in result_files:
            if f'{param}' in file_name:
                param_dfs.append(pd.read_csv(file_name, delimiter='\t', low_memory=False))
        param_df_merged = pd.concat(param_dfs, ignore_index=True)
        merged_df = pd.merge(param_df_merged, merged_plv_df, on='referenceprel', how='left')
        merged_df.drop_duplicates(inplace=True)
        columns = ['cddept_x', 'cdparametre', 'valtraduite', 'inseecommuneprinc', 'dateprel', 'heureprel']
        merged_df = merged_df[columns]
        merged_df = merged_df.dropna(axis=0)
        output_file = os.path.join(base_path, 'data/clean', f'clean_{param}.txt')
        merged_df['cddept_x'] = merged_df['cddept_x'].astype(str)
        merged_df['cdparametre'] = merged_df['cdparametre'].astype(int)
        merged_df['valtraduite'] = merged_df['valtraduite'].astype(float)
        merged_df['inseecommuneprinc'] = merged_df['inseecommuneprinc'].astype(str)
        merged_df.to_csv(output_file, sep='\t', index=False, encoding='latin-1')

def shorten_param_csv():
    input_file = 'data/PAR_20250523_SANDRE.csv'
    output_file = 'data/PAR_SANDRE_short.txt'
    delimiter = ';'
    columns_indices = [0, 1, 6, 7, 10]
    selected_cdparams = get_selected_cdparams('C:/Users/mberthie/Documents/StageHubEau/')
    df = pd.read_csv(input_file, delimiter=delimiter, encoding='latin-1', dtype=str)
    unit_columns = ['SymUniteMesure1', 'SymUniteMesure2', 'SymUniteMesure3', 'SymUniteMesure4', 'SymUniteMesure5', 'SymUniteMesure6', 'SymUniteMesure7', 'SymUniteMesure8']
    allowed_units = {'µg/L', 'mg/L', 'g/L', 'ng/L'}
    def clean_unit(unit):
        if pd.isna(unit):
            return None
        unit = unit.replace('(', '').replace(')', '').strip()
        for allowed in allowed_units:
            if unit.startswith(allowed[:-2]) and unit.endswith('/L'):
                return allowed
        return None
    cleaned_units = df[unit_columns].applymap(clean_unit)
    df['Unit'] = cleaned_units.apply(lambda row: next((u for u in row if u), None), axis=1)
    df = df.rename(columns={'ï»¿CdParametre': 'CdParametre'})
    df = df.iloc[1:, :]
    df = df[df['CdParametre'].isin(selected_cdparams)]
    columns_to_keep = [df.columns[i] for i in columns_indices] + ['Unit']
    df = df[columns_to_keep]
    df.to_csv(output_file, sep='\t', index=False, encoding='latin-1')

if __name__ == '__main__':
    data_path = os.path.join(os.getcwd(), 'data')
    merge_com(data_path)
    merge_results(data_path)
    merge_plv(data_path)
    analyze(data_path)
    merge_all()
    shorten_param_csv()
