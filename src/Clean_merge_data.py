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

def merge_all():
    print("Creating clean data directory...")
    os.makedirs(os.path.join(os.getcwd(), 'data', 'clean'), exist_ok=True)

    print("Merging DIS_COM_UDI files...")
    input_files = glob.glob(os.path.join(os.getcwd(), 'data', 'raw', 'DIS_COM_UDI_*.txt'))
    output_file = os.path.join(os.getcwd(), 'data', 'DIS_COM_UDI.csv')
    merge_files(input_files, output_file)
    print(f"DIS_COM_UDI merged into {output_file}")

    print("Processing DIS_RESULT files...")
    input_files = glob.glob(os.path.join(os.getcwd(), 'data', 'raw', 'DIS_RESULT_*.txt'))
    for file in tqdm(input_files, desc='Merging DIS_RESULT files'):
        plv = os.path.join(os.getcwd(), 'data', 'raw', 'DIS_PLV_' + '_'.join(file.split('_')[-2:]))
        dtype_dict = {
            'cddept_x': str,
            'cdparametre': str,
            'inseecommuneprinc': str
        }
        df = pd.read_csv(file, sep=',', encoding='latin-1', on_bad_lines='skip', dtype=dtype_dict)
        df_plv = pd.read_csv(plv, sep=',', encoding='latin-1', on_bad_lines='skip', dtype=dtype_dict)
        # Remove trailing '.0' if present in string columns
        for col in ['cddept_x', 'cdparametre', 'inseecommuneprinc']:
            if col in df.columns:
                df[col] = df[col].str.replace(r'\.0$', '', regex=True)
        if 'valtraduite' in df.columns:
            df['valtraduite'] = pd.to_numeric(df['valtraduite'], errors='coerce')

        merged_df = pd.merge(df, df_plv, on='referenceprel', how='left')
        merged_df.drop_duplicates(inplace=True)
        columns = ['cddept_x', 'cdparametre', 'valtraduite', 'inseecommuneprinc', 'dateprel', 'heureprel']
        merged_df = merged_df[columns]

        grouped = merged_df.groupby('cdparametre')
        for param, group in grouped:
            output_file = os.path.join(os.getcwd(), 'data', 'clean', f'DIS_RESULT_{param}.csv')
            if not os.path.exists(output_file):
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                group.to_csv(output_file, sep='\t', index=False, encoding='latin-1')
            else:
                group.to_csv(output_file, mode='a', header=False, sep='\t', index=False, encoding='latin-1')
    print("All merging done.")


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
    merge_all()
