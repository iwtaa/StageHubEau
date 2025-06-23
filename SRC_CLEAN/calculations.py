import os
import pandas as pd
from Utilities import *
from tqdm import tqdm

if  __name__ == "__main__":
    cwd = os.getcwd()
    cdparams_selected = get_selected_cdparams(cwd)
    print(cdparams_selected)

    for df in load_folder_data(os.path.join(cwd, 'data', 'clean')):
        cdparam = int(df['cdparametre'].iloc[0])
        df['dateprel'] = pd.to_datetime(df['dateprel'], errors='coerce')
        
        df['valtraduite_D'] = None
        for commune, df_group in tqdm(get_by_commune(df), desc=f"Processing {cdparam}", total=len(df['inseecommuneprinc'].unique())):
            mean = df_group['valtraduite'].mean()
            std = df_group['valtraduite'].std()
            df.loc[df_group.index, 'valtraduite_D'] = (df_group['valtraduite'] - mean) / std
        
        df['dateprel'] = pd.to_datetime(df['dateprel'])
        df = df.sort_values(by='dateprel')
        df = df.set_index('dateprel')
        df['valtraduite_DS'] = df['valtraduite_D'].rolling(window='60D').mean()
        df['valtraduite_S'] = df['valtraduite'].rolling(window='60D').mean()
        df['valtraduite_DSX'] = (df['valtraduite_DS'] + df['valtraduite'].mean()) * df['valtraduite_DS'].std()
        
        df.to_csv(os.path.join(cwd, 'data', 'upgraded', f'upgraded_{cdparam}.txt'), sep='\t', index=False, encoding='latin-1')
            
            
    
    