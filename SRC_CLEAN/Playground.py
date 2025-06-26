from Utilities import *
import os
import pandas as pd
from tqdm import tqdm

def amplify_percentage(value):
    return ((((value / 100.0) * 1.5) ** 2 ) / (1.5 ** 2)) * 100.0




params = get_selected_cdparams(os.getcwd())

par_sandre_df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'PAR_SANDRE_short.txt'), sep='\t')
criteres_offi_df = pd.read_csv(os.path.join(os.getcwd(), 'src', 'criteresOffi.txt'), sep='\t')

from map_column_department import map_communes_per_department

for df in load_folder_data(os.path.join(os.getcwd(), 'data', 'upgraded')):
    commune_values = {}
    cdparam = df['cdparametre'].iloc[0]
    df = df[df['cddept_x'] == '17']
    
    if cdparam not in criteres_offi_df['CdParametre'].values or df.empty:
        continue
    limit = float(criteres_offi_df[criteres_offi_df['CdParametre'] == cdparam]['Limit_eu'].values[0])
    if limit == None:
        continue
    for insee, group in tqdm(get_by_commune(df), desc=f"Processing {cdparam}"):
        if insee not in commune_values.keys():
            commune_values[insee] = 0.0
        perc = group['valtraduite'].mean() / limit * 100.0
        commune_values[insee] += perc
        ''' 
        if perc > 60:
            perc = (perc - 60) * 100/40
            commune_values[insee] += amplify_percentage(perc)
        '''
    
    for insee in commune_values:
        if commune_values[insee] > 100.0:
            commune_values[insee] = 100.0

    commune_df = pd.DataFrame(
        [(str(insee), float(value)) for insee, value in commune_values.items()],
        columns=['inseecommuneprinc', 'value']
    )
    print(commune_df.head())
    map_communes_per_department(commune_df, 'value', 0.0, 100.0, 17, cdparam, save_path=os.path.join(os.getcwd(), f'{cdparam}_17.png'))
    output_path = os.path.join(os.getcwd(), f'{cdparam}_17_above_80.txt')
    with open(output_path, 'w') as f:
        for insee, value in commune_values.items():
            if value > 80.0:
                f.write(f"{insee}\t{value}\n")