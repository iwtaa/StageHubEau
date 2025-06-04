import itertools
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

class MatrixPairedFileProcessor:
    def __init__(self, files_info, max_memory_mb, calculation_function):
        """
        files_info: dict, {filename: size_in_bytes}
        max_memory_mb: int, memory limit in MB
        """
        self.files_info = files_info
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.loaded_data = {}  # Simulates loaded file data {filename: data_content}
        self.current_memory_usage = 0
        self.processed_pairs_count = 0  # To count unique pairs processed
        self._do_calculation = calculation_function  # Function to process pairs
        self.load_count = 0
        self.matrix = np.zeros((len(files_info), len(files_info)), dtype=np.float32)
    
    def get_matrix(self):
        return np.triu(self.matrix) + np.tril(self.matrix.T, k=-1)

    def print_load_count(self):
        print(f"Total load operations performed: {self.load_count}")

    def _load_file(self, filename):
        if filename not in self.files_info:
            print(f"Error: File {filename} not in provided file info.")
            return False
        file_size = self.files_info[filename]
        if filename not in self.loaded_data:
            if self.current_memory_usage + file_size > self.max_memory_bytes:
                # This should ideally be prevented by higher-level logic
                print(f"Error: Cannot load {filename} (size {file_size}). "
                      f"Current usage {self.current_memory_usage}, limit {self.max_memory_bytes}.")
                return False
            # Simulate loading data
            self.loaded_data[filename] = f"data_for_{filename}"
            self.current_memory_usage += file_size
            print(f"LOAD: {filename} (size: {file_size}), Mem: {self.current_memory_usage}")
        self.load_count += 1
        return True

    def _unload_file(self, filename):
        if filename in self.loaded_data:
            file_size = self.files_info[filename]
            del self.loaded_data[filename]
            self.current_memory_usage -= file_size
            print(f"UNLOAD: {filename} (size: {file_size}), Mem: {self.current_memory_usage}")

    def _unload_all(self):
        # Iterate over a copy of keys if modifying the dictionary
        for filename in list(self.loaded_data.keys()):
            self._unload_file(filename)
        if self.current_memory_usage != 0:  # Sanity check
            print(f"Warning: Memory usage not zero after unload_all: {self.current_memory_usage}")
            self.current_memory_usage = 0

    def _create_bins(self):
        # First Fit Decreasing for bin packing
        # Sort files by size, descending
        sorted_files = sorted(self.files_info.items(), key=lambda item: item[1], reverse=True)
        
        bins = []  # List of lists (each inner list is a bin of filenames)
        bin_sizes = []  # List of total sizes for each bin

        for filename, size in sorted_files:
            if size > self.max_memory_bytes:
                print(f"Warning: File {filename} (size {size}) is larger than max_memory "
                      f"({self.max_memory_bytes}) and cannot be processed in any bin.")
                continue

            placed = False
            for i, current_bin_total_size in enumerate(bin_sizes):
                if current_bin_total_size + size <= self.max_memory_bytes:
                    bins[i].append(filename)
                    bin_sizes[i] += size
                    placed = True
                    break
            if not placed:
                bins.append([filename])
                bin_sizes.append(size)
        
        print(f"Created {len(bins)} bins.")
        # for i, bin_content in enumerate(bins):
        #     print(f"  Bin {i} (size {bin_sizes[i]}): {bin_content}")
        return bins, bin_sizes

    def process_all_file_pairs(self):
        bins, bin_sizes = self._create_bins()
        num_bins = len(bins)
        processed_pairs_tracker = set()  # To ensure each pair (a,b) is done once

        # 1. Intra-bin pairs
        for i in range(num_bins):
            current_bin_files = bins[i]
            print(f"\nProcessing intra-bin pairs for Bin {i} (size {bin_sizes[i]})")
            
            # Load all files in the current bin
            for filename in current_bin_files:
                if not self._load_file(filename):  # Should always succeed by bin construction
                    print(f"Critical error loading file {filename} for intra-bin processing.")
                    self._unload_all()  # Cleanup before potentially skipping
                    continue
            
            for file1_idx in range(len(current_bin_files)):
                for file2_idx in range(file1_idx + 1, len(current_bin_files)):
                    file1_name = current_bin_files[file1_idx]
                    file2_name = current_bin_files[file2_idx]
                    pair = tuple(sorted((file1_name, file2_name)))
                    if pair not in processed_pairs_tracker:
                        file1_global_index = list(self.files_info.keys()).index(file1_name)
                        self.matrix[file2_global_index, file1_global_index] = self._do_calculation(file1_name, file2_name)
                        processed_pairs_tracker.add(pair)
            
            self._unload_all()  # Unload files from this bin

        # 2. Inter-bin pairs
        for i in range(num_bins):
            for j in range(i + 1, num_bins):
                bin_i_files = bins[i]
                bin_j_files = bins[j]
                print(f"\nProcessing inter-bin pairs for Bin {i} and Bin {j}")

                if bin_sizes[i] + bin_sizes[j] <= self.max_memory_bytes:
                    print(f"  Loading Bin {i} and Bin {j} together.")
                    # Load all files from both bins
                    for file1_name in bin_i_files:
                        for file2_name in bin_j_files:
                            pair = tuple(sorted((file1_name, file2_name)))
                            if pair not in processed_pairs_tracker:
                                file1_global_index = list(self.files_info.keys()).index(file1_name)
                                file2_global_index = list(self.files_info.keys()).index(file2_name)
                                self.matrix[file1_global_index, file2_global_index] = self._do_calculation(file1_name, file2_name)
                                self.matrix[file2_global_index, file1_global_index] = self._do_calculation(file1_name, file2_name)
                                processed_pairs_tracker.add(pair)
                    self._unload_all()
                else:
                    # Bins i and j don't fit together. Process file-by-file from one bin against the other.
                    # This part aims to minimize simultaneous memory but might increase load operations.
                    print(f"  Bin {i} (size {bin_sizes[i]}) and Bin {j} (size {bin_sizes[j]}) "
                          f"don't fit together. Processing granularly.")
                    
                    for file1_name in bin_i_files:
                        if not self._load_file(file1_name):
                            continue
                        for file2_name in bin_j_files:
                            # Check if file1 (already loaded) and file2 can fit
                            if self.files_info[file1_name] + self.files_info[file2_name] <= self.max_memory_bytes:
                                if self._load_file(file2_name):  # Try to load file2_name
                                    pair = tuple(sorted((file1_name, file2_name)))
                                    if pair not in processed_pairs_tracker:
                                        file1_global_index = list(self.files_info.keys()).index(file1_name)
                                        file2_global_index = list(self.files_info.keys()).index(file2_name)
                                        self.matrix[file1_global_index, file2_global_index] = self._do_calculation(file1_name, file2_name)
                                        self.matrix[file2_global_index, file1_global_index] = self._do_calculation(file1_name, file2_name)
                                        processed_pairs_tracker.add(pair)
                                    self._unload_file(file2_name)  # Unload file2_name for the next iteration
                                else:
                                    # This case means file1_name is loaded, but file1_name + file2_name > max_memory
                                    # (even though _load_file(file1_name) succeeded).
                                    # The _load_file for file2_name would have printed an error.
                                    print(f"    Skipping pair ({file1_name}, {file2_name}) as they don't fit together with {file1_name} loaded.")
                            else:
                                print(f"    Skipping pair ({file1_name}, {file2_name}) as their combined size exceeds memory limit.")
                        
                        self._unload_file(file1_name)  # Unload file1_name after pairing with all of bin_j_files
                    self._unload_all()  # Final cleanup for this pair of bins

        self.processed_pairs_count = len(processed_pairs_tracker)
        print(f"\nFinished processing. Total unique pairs: {self.processed_pairs_count}")
        expected_pairs = len(list(itertools.combinations(self.files_info.keys(), 2)))
        print(f"Expected total unique pairs: {expected_pairs}")
        if self.processed_pairs_count != expected_pairs:
            print("Warning: Mismatch in processed vs expected pairs. Some files/pairs might have been skipped due to size constraints.")

def calculation(file1_name, file2_name):
    return int(file1_name) + int(file2_name)

if __name__ == "__main__":
    # Example Usage
    files_info = {}
    for i in range(1, 5):
        size_mb = i  # Sizes ranging from 2MB to 100MB
        filename = f"{i}"
        files_info[filename] = size_mb * 1024 * 1024
    print(f"Files info: {files_info}")
    max_memory_mb = 8  # 100 MB limit
    processor = MatrixPairedFileProcessor(files_info, max_memory_mb, calculation)
    processor.process_all_file_pairs()
    processor.print_load_count()
    print(processor.get_matrix())
