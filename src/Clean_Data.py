import csv
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import glob
import numpy as np

def process_plv_file(plv_filename, output_filename, delimiter):
    try:
        with open(plv_filename, 'r', encoding='utf-8') as infile, \
             open(output_filename, 'a', newline='', encoding='utf-8') as outfile:  # Append mode
            reader = csv.reader(infile, delimiter=delimiter)
            writer = csv.writer(outfile, delimiter=delimiter)

            header = next(reader, None)
            if header:
                # Write header only if the file is empty
                outfile.seek(0, 2)  # Go to the end of file
                if outfile.tell() == 0:  # Check if file is empty
                    writer.writerow(header)

                for row in reader:
                    writer.writerow(row)

    except FileNotFoundError:
        print(f"File not found: {plv_filename}")
    except csv.Error as e:
        print(f"CSV error in {plv_filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {plv_filename}: {e}")

def process_result_file(result_filename, output_filename, delimiter):
    try:
        with open(result_filename, 'r', encoding='utf-8') as infile, \
             open(output_filename, 'a', newline='', encoding='utf-8') as outfile:  # Append mode
            reader = csv.reader(infile, delimiter=delimiter)
            writer = csv.writer(outfile, delimiter=delimiter)

            header = next(reader, None)
            if header:
                # Write header only if the file is empty
                outfile.seek(0, 2)  # Go to the end of file
                if outfile.tell() == 0:  # Check if file is empty
                    writer.writerow(header)

                for row in reader:
                    writer.writerow(row)

    except FileNotFoundError:
        print(f"File not found: {result_filename}")
    except csv.Error as e:
        print(f"CSV error in {result_filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {result_filename}: {e}")

def merge_plv_files(output_filename='data/products/merged_plv_data.csv', delimiter=','):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # Clear the output file before appending
    with open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
        pass

    for dept_index in range(0, 96):
        for year in range(2019, 2025):
            plv_filename = f'data/DIS_PLV_{year}_0{dept_index:02d}.txt'  # Format dept_index with leading zero
            process_plv_file(plv_filename, output_filename, delimiter)

def merge_result_files(delimiter=','):
    # Dictionary to store file writers for each cdparametre
    writers = {}

    for dept_index in range(0, 96):
        for year in range(2019, 2025):
            result_filename = f'data/DIS_RESULT_{year}_0{dept_index:02d}.txt'  # Format dept_index with leading zero

            try:
                with open(result_filename, 'r', encoding='utf-8') as infile:
                    reader = csv.reader(infile, delimiter=delimiter)
                    header = next(reader, None)  # Read the header

                    for row in reader:
                        # Assuming cdparametre is in the 4th column (index 3)
                        if len(row) > 3:
                            cdparametre = row[3]
                            output_filename = f'data/products/DIS_PARAM_{cdparametre}.csv'

                            # Get the writer for this cdparametre, create if it doesn't exist
                            if cdparametre not in writers:
                                outfile = open(output_filename, 'a', newline='', encoding='utf-8')  # Append mode
                                writer = csv.writer(outfile, delimiter=delimiter)
                                writers[cdparametre] = {'file': outfile, 'writer': writer}

                                # Write header only if the file is empty
                                outfile.seek(0, 2)  # Go to the end of file
                                if outfile.tell() == 0 and header:  # Check if file is empty
                                    writer.writerow(header)
                            
                            writers[cdparametre]['writer'].writerow(row)
                        else:
                            print(f"Skipping row in {result_filename} due to insufficient length.")

            except FileNotFoundError:
                print(f"File not found: {result_filename}")
            except csv.Error as e:
                print(f"CSV error in {result_filename}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred while processing {result_filename}: {e}")

    # Close all files
    for cdparametre in writers:
        writers[cdparametre]['file'].close()


def merge_and_extract_data(plv_indices=[0, 2, 3, 8, 9],
                           result_indices=[3, 10, 12, 14],
                           par_indices=[1, 4],
                           join_index_plv=7,
                           join_index_result=1,
                           join_index_par=0,
                           delimiter=','):
    plv_filename = f'data/products/merged_plv_data.csv'
    par_filename='data/products/PAR_20250523_SANDRE_short.csv'

    plv_data = {}
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
        return
    except csv.Error as e:
        print(f"CSV error in {plv_filename}: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while processing {plv_filename}: {e}")
        return
    
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
    for cdparametre in tqdm(par_data.keys(), desc="Processing cdparametre"):
        output_filename = f'data/products/merged_{cdparametre}_data.csv'
        
        # Define the header
        header = ['cddept', 'inseecommuneprinc', 'nomcommuneprinc', 'dateprel', 'heureprel', 'cdparametre', 'cdunitereferencesiseeaux', 'limitequal', 'valtraduite', 'NomParametre', 'LbCourtParametre']

        result_filename = f'data/products/DIS_PARAM_{cdparametre}.csv'
        if not os.path.exists(result_filename):
            #print(f"Error: File not found: {result_filename}")
            continue

        try:
            with open(output_filename, 'w', newline='', encoding='utf-8') as outfile:  # Use 'w' to overwrite
                writer = csv.writer(outfile, delimiter=delimiter)
                
                # Write the header
                writer.writerow(header)
                
                result_data = {}
                try:
                    with open(result_filename, 'r', encoding='utf-8') as file:
                        reader = csv.reader(file, delimiter=delimiter)
                        next(reader)  # Skip header
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
                # Write merged data
                for key in result_data.keys():
                    if key in plv_data and cdparametre in par_data:
                        for plv_row in plv_data[key]:
                            for result_row in result_data[key]:
                                merged_row = plv_row + result_row + par_data[cdparametre]
                                writer.writerow(merged_row)
                    else:
                        print(f"No matching data found for key: {key} or cdparametre: {cdparametre}")
        except Exception as e:
            print(f"An unexpected error occurred while writing data to {output_filename}: {e}")

def display_merged_data(filename='data/products/merged_data.csv', delimiter=','):
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
    input_filename='data/PAR_20250523_SANDRE.csv'
    output_filename='data/products/PAR_20250523_SANDRE_short.csv'
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

def list_csv_files(directory):
    all_files = glob.glob(os.path.join(directory, "merged_*.csv"))
    return [f for f in all_files if os.path.basename(f) != "merged_plv_data.csv"]

def count_zero_percentage(df):
    if 'valtraduite' not in df.columns:
        print("Error: 'valtraduite' column not found in DataFrame.")
        return None
    
    valtraduite = df['valtraduite'].fillna(0.0)
    percentage = (valtraduite == 0.0).mean() * 100
    return percentage

def mean(df):
    if 'valtraduite' not in df.columns:
        print("Error: 'valtraduite' column not found in DataFrame.")
        return None
    
    return df['valtraduite'].mean()

def unique_values(df):
    if 'valtraduite' not in df.columns:
        print("Error: 'valtraduite' column not found in DataFrame.")
        return None
    
    return len(df['valtraduite'].unique())


def analyse_csv_files():
    directory = "data/products"  # Replace with the actual directory if needed
    csv_files = list_csv_files(directory)
    file_paths = []
    num_rows_list = []
    zero_percentage_list = []
    mean_list = []
    unique_val_list = []

    for file_path in tqdm(csv_files, desc="Processing files"):
        try:
            df = pd.read_csv(file_path)
            num_rows = len(df)
            file_paths.append(file_path)
            num_rows_list.append(num_rows)
            
            # Calculate and store zero percentage, mean, and unique values
            zero_percentage = count_zero_percentage(df)
            file_mean = mean(df)
            unique_vals = unique_values(df)
            
            zero_percentage_list.append(zero_percentage)
            mean_list.append(file_mean)
            unique_val_list.append(str(unique_vals))  # Store as strings to handle arrays in CSV
            
        except pd.errors.EmptyDataError:
            print(f"File: {file_path} is empty.")
            zero_percentage_list.append(None)
            mean_list.append(None)
            unique_val_list.append(None)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            zero_percentage_list.append(None)
            mean_list.append(None)
            unique_val_list.append(None)
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")
            zero_percentage_list.append(None)
            mean_list.append(None)
            unique_val_list.append(None)

    # Plotting the cumulative bar graph
    if file_paths:
        plt.figure(figsize=(12, 6))
        
        # Sort the data by row number
        sorted_indices = np.argsort(num_rows_list)
        file_paths = [file_paths[i] for i in sorted_indices]
        num_rows_list = [num_rows_list[i] for i in sorted_indices]
        zero_percentage_list = [zero_percentage_list[i] for i in sorted_indices]
        mean_list = [mean_list[i] for i in sorted_indices]
        unique_val_list = [unique_val_list[i] for i in sorted_indices]

        # Calculate cumulative percentage of rows
        total_rows = sum(num_rows_list)
        cumulative_percentage = np.cumsum(num_rows_list) / total_rows * 100

        # Use row number as x-axis values
        x_values = np.arange(1, len(num_rows_list) + 1)

        plt.bar(x_values, cumulative_percentage, color='skyblue', width=0.8)  # Adjust width for larger bars

        plt.xlabel('File (ordered by row number)')
        plt.ylabel('Cumulative Percentage of Rows')
        plt.title('Cumulative Percentage of Rows in CSV files')

        plt.tight_layout()  # Adjust layout to prevent labels from overlapping

        # Save the plot to a PNG file
        plt.savefig('data/products/cumulative_percentage_plot.png')

        plt.show()

        # Create a DataFrame to store the results
        results_df = pd.DataFrame({
            'file_name': file_paths,
            'rows': num_rows_list,
            'zero_percentage': zero_percentage_list,
            'mean': mean_list,
            'unique_val': unique_val_list
        })

        # Save the DataFrame to a CSV file
        results_df.to_csv('data/products/file_row_counts.csv', index=False)

    else:
        print("No CSV files found to plot.")


if __name__ == '__main__':
    #shorten_param_csv()
    #merge_plv_files()
    #merge_result_files()
    #merge_and_extract_data()
    #display_merged_data()
    analyse_csv_files()
