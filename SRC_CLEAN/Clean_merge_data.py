import glob
import pandas as pd
from tqdm import tqdm
import os

def merge_files(file_names, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        first_file = True
        for file in file_names:
            try:
                with open(file, 'r', encoding='latin-1') as infile:
                    header = infile.readline()
                    if first_file:
                        outfile.write(header)
                        first_file = False
                    for line in infile:
                        outfile.write(line)
            except UnicodeDecodeError:
                print(f"Skipping file due to encoding error: {file}")

    # Remove duplicates
    df = pd.read_csv(output_file, sep='\t', encoding='latin-1', on_bad_lines='skip')
    df.drop_duplicates(inplace=True)
    df.to_csv(output_file, sep='\t', index=False)

def merge_com(path):
    file_names = glob.glob(path + r'\raw\DIS_COM_UDI_*.txt')
    output_file = path + r'\merged\DIS_COM_UDI.txt'
    merge_files(file_names, output_file)

def merge_results(path):
    file_names = glob.glob(path + r'\raw\DIS_RESULT_*.txt')
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
    file_names = glob.glob(path + r'\raw\DIS_PLV_*.txt')
    short_names = [file_name[:-4][-3:] for file_name in file_names]
    short_names = list(set(short_names))
    for dept in tqdm(short_names, desc='Merging DIS_PLV'):
        file_names_dept = [file_name for file_name in file_names if file_name.endswith(dept + '.txt')]
        output_file_dept = os.path.join(path, 'merged', str(dept), f'DIS_PLV_{dept}.txt')
        merge_files(file_names_dept, output_file_dept)

if __name__ == '__main__':
    path = r'C:\Users\mberthie\Documents\StageHubEau\data'
    #merge_com(path)
    #merge_results(path)
    #merge_plv(path)

    def iterate_dis_result_files(path):
        cdparams_files = {}
        for root, _, files in os.walk(path):
            for file in tqdm(files, desc='Iterating files'):
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

        for cdparam, files in tqdm(cdparams_files.items(), desc='Iterating cdparams'):
            for file in files:
                dept = file[-12:-9]
                plv = os.path.join(path, dept, 'DIS_PLV_' + dept + '.txt')
                df_plv = pd.read_csv(plv, sep='\t', encoding='latin-1', on_bad_lines='skip')
                df_result = pd.read_csv(file, sep='\t', encoding='latin-1', on_bad_lines='skip')
                df_merged = pd.merge(df_result, df_plv, on='referenceprel', how='left')
                # Now df_merged contains all columns from df_result and df_plv,
                # with the columns from df_plv added based on matching 'referenceprel'.
                # If there's no matching 'referenceprel' in df_plv, the  columns will have NaN values.
                print(df_merged.head())
                break
            break


    merged_path = os.path.join(path, 'merged')
    iterate_dis_result_files(merged_path)
    
