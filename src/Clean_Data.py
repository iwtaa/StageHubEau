import csv
import pandas as pd
'''
PLV Header: ['cddept', 'cdreseau', 'inseecommuneprinc', 'nomcommuneprinc', 'cdreseauamont', 'nomreseauamont', 'pourcentdebit', 'referenceprel', 
         'dateprel', 'heureprel', 'conclusionprel', 'ugelib', 'distrlib', 'moalib', 'plvconformitebacterio', 'plvconformitechimique', 
         'plvconformitereferencebact', 'plvconformitereferencechim']
         0, 2, 3, 7, 8, 9
RESULT Header: ['cddept', 'referenceprel', 'cdparametresiseeaux', 'cdparametre', 'libmajparametre', 'libminparametre', 'libwebparametre', 
         'qualitparam', 'insituana', 'rqana', 'cdunitereferencesiseeaux', 'cdunitereference', 'limitequal', 'refqual', 'valtraduite', 
         'casparam', 'referenceanl']
         0, 1, 3, 5, 11, 12, 14
PAR_20250522_SANDRE_short.csv Header: ['CdParametre', 'NomParametre', 'DfParametre', 'NatParametre', 'LbCourtParametre']
'''

def merge_and_extract_data(output_filename='data/merged_data.csv',
                           dept_index = 17,
                           plv_indices=[0, 2, 3, 8, 9],
                           result_indices=[3, 10, 12, 14],
                           par_indices=[1, 4],
                           join_index_plv=7,
                           join_index_result=1,
                           join_index_par=0,
                           delimiter=','):

    all_merged_data = []

    for year in range(2019, 2025):
        plv_filename = f'data/DIS_PLV_{year}_0{dept_index}.txt'
        result_filename = f'data/DIS_RESULT_{year}_0{dept_index}.txt'
        par_filename='data/PAR_20250522_SANDRE_short.csv'

        plv_data = {}
        result_data = {}
        par_data = {}

        # Read PLV data
        try:
            with open(plv_filename, 'r', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter=delimiter)
                header = next(reader)  # Skip header
                for row in reader:
                    if len(row) > join_index_plv:
                        key = row[join_index_plv]
                        if key not in plv_data:
                            plv_data[key] = []  # Initialize as a list
                        plv_data[key].append([row[i] for i in plv_indices if i < len(row)])
                    else:
                        print(f"Skipping row in {plv_filename} due to insufficient length.")
        except FileNotFoundError:
            print(f"Error: File not found: {plv_filename}")
            continue
        except csv.Error as e:
            print(f"CSV error in {plv_filename}: {e}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred while processing {plv_filename}: {e}")
            continue

        # Read RESULT data
        try:
            with open(result_filename, 'r', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter=delimiter)
                header = next(reader)  # Skip header
                for row in reader:
                    if len(row) > join_index_result:
                        key = row[join_index_result]
                        if key not in result_data:
                            result_data[key] = []  # Initialize as a list
                        result_data[key].append([row[i] for i in result_indices if i < len(row)])
                    else:
                        print(f"Skipping row in {result_filename} due to insufficient length.")
        except FileNotFoundError:
            print(f"Error: File not found: {result_filename}")
            continue
        except csv.Error as e:
            print(f"CSV error in {result_filename}: {e}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred while processing {result_filename}: {e}")
            continue
        
        # Read PAR data
        try:
            with open(par_filename, 'r', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter=';')
                next(reader)  # Skip header
                for row in reader:
                    if row:
                        key = row[0]
                        par_data[key] = [row[i] for i in par_indices if i < len(row)]
                    else:
                        print(f"Skipping row in {par_filename} due to insufficient length.")
        except FileNotFoundError:
            print(f"Error: File not found: {par_filename}")
            return
        except csv.Error as e:
            print(f"CSV error in {par_filename}: {e}")
            return
        except Exception as e:
            print(f"An unexpected error occurred while processing {par_filename}: {e}")
            return

        # Exhaustive merge of data
        for plv_key, plv_values_list in plv_data.items():
            if plv_key in result_data:
                result_values_list = result_data[plv_key]
                for plv_values in plv_values_list:
                    for result_values in result_values_list:
                        par_key = result_values[0]
                        if par_key in par_data:
                            par_values = par_data[par_key]
                            output_row = [plv_values[0]] + [plv_values[1]] + [plv_values[2]] + [plv_values[3]] + [plv_values[4]] + [result_values[0]] + [result_values[1]] + [result_values[2]] + [result_values[3]] + [par_values[0]] + [par_values[1]]
                            all_merged_data.append(output_row)

    # Write all merged data to the output file
    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile, delimiter=delimiter)

            # Write header
            header = ['cddept', 'inseecommuneprinc', 'numcommuneprinc', 'dateprel', 'heureprel', 'cdparametre', 'cdunitereferencesiseeaux', 'limitequal', 'valtraduite', 'NomParametre', 'LbCourtParametre']
            writer.writerow(header)

            # Write data rows
            writer.writerows(all_merged_data)

    except Exception as e:
        print(f"An unexpected error occurred while writing to {output_filename}: {e}")


def display_merged_data(filename='data/merged_data.csv', delimiter=','):
    try:
        df = pd.read_csv(filename, delimiter=delimiter)
        print(df)  # Display the DataFrame
        for col in df.columns:
            print(col)
        print(f"Number of unique 'cdparametre' values: {df['cdparametre'].nunique()}")

    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
    except Exception as e:
        print(f"An unexpected error occurred while reading {filename}: {e}")

def shorten_param_csv():
    input_filename='data/PAR_20250522_SANDRE.csv'
    output_filename='data/PAR_20250522_SANDRE_short.csv'
    delimiter=';'
    indices_to_keep=[0, 1, 6, 7, 10]
    try:
        with open(input_filename, 'r', encoding='utf-8') as infile, \
                open(output_filename, 'w', newline='', encoding='utf-8') as outfile:

            reader = csv.reader(infile, delimiter=delimiter)
            writer = csv.writer(outfile, delimiter=delimiter)

            header = next(reader)
            header_exp = next(reader)

            shortened_row = [header[i] for i in indices_to_keep if i < len(header)]
            writer.writerow(shortened_row)
            shortened_row = [header_exp[i] for i in indices_to_keep if i < len(header_exp)]
            writer.writerow(shortened_row)

            for row in reader:
                shortened_row = [row[i] for i in indices_to_keep if i < len(row)]
                writer.writerow(shortened_row)

    except FileNotFoundError:
        print(f"Error: Input file not found: {input_filename}")
    except csv.Error as e:
        print(f"CSV error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    display_merged_data('data/DIS_RESULT_2024_017.txt', delimiter=',')
    shorten_param_csv()
    merge_and_extract_data()
    display_merged_data()
    display_merged_data('data/PAR_20250522_SANDRE_short.csv', delimiter=';')