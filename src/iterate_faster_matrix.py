import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

class MatrixPairedFileProcessor:
    def __init__(self, files_info, max_memory_mb, calculation_function):
        self.files_info = files_info
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.loaded_data = {}
        self.current_memory_usage = 0
        self._do_calculation = calculation_function
        self.total_memory_loaded = 0
        self.matrix = np.zeros((3, len(files_info), len(files_info)), dtype=np.float32)
        self.saved_loads = 0
        self.file_index_map = {filename: i for i, filename in enumerate(files_info.keys())}

    def get_matrix(self):
        return self.matrix

    def _load_file(self, filename):
        file_size = self.files_info[filename]
        if filename not in self.loaded_data:
            if self.current_memory_usage + file_size > self.max_memory_bytes:
                return False
            try:
                df = pd.read_csv(filename)
            except Exception as e:
                # Handle file loading error if necessary
                return False
            self.loaded_data[filename] = df
            self.current_memory_usage += file_size
            self.total_memory_loaded += file_size
        else:
            self.saved_loads += 1
        return True

    def _unload_file(self, filename):
        if filename in self.loaded_data:
            file_size = self.files_info[filename]
            del self.loaded_data[filename]
            self.current_memory_usage -= file_size

    def _unload_all(self):
        for filename in list(self.loaded_data.keys()):
            self._unload_file(filename)
        self.current_memory_usage = 0

    def process_all_file_pairs(self):
        files_list = sorted(self.files_info.keys(), key=lambda x: self.files_info[x], reverse=True)
        processed_files = set()
        bag_index = 0

        while True:
            bag_memory = 0
            files_to_load = []
            for f in files_list:
                if f in processed_files:
                    continue
                temp_files = [file for file in files_list if file not in processed_files and file != f]
                fmax_size = self.files_info[max(temp_files, key=lambda x: self.files_info[x])] if temp_files else 0
                if bag_memory + self.files_info[f] + fmax_size <= self.max_memory_bytes:
                    bag_memory += self.files_info[f]
                    files_to_load.append(f)
                    processed_files.add(f)
            if not files_to_load:
                break
            for f in files_to_load:
                self._load_file(f)
            pairs = list(itertools.combinations(files_to_load, 2))
            remaining_files = [f for f in files_list if f not in processed_files]
            all_items = len(pairs) + len(remaining_files) * len(files_to_load)
            with tqdm(total=all_items, desc=f"Processing bag {bag_index + 1}") as pbar:
                for file_a, file_b in pairs:
                    index_a = self.file_index_map[file_a]
                    index_b = self.file_index_map[file_b]
                    res = self._do_calculation(self.loaded_data[file_a], self.loaded_data[file_b])
                    if res:
                        cor, lag, sign = res
                        self.matrix[0, index_a, index_b] = cor
                        self.matrix[1, index_a, index_b] = lag
                        self.matrix[2, index_a, index_b] = sign
                        self.matrix[0, index_b, index_a] = cor
                        self.matrix[1, index_b, index_a] = -lag
                        self.matrix[2, index_b, index_a] = sign
                    pbar.update(1)
                for f in remaining_files:
                    if not self._load_file(f):
                        continue
                    df = self.loaded_data[f]
                    for other_file in files_to_load:
                        if not self._load_file(other_file):
                            continue
                        ds = self.loaded_data[other_file]
                        res = self._do_calculation(df, ds)
                        if res:
                            cor, lag, sign = res
                            index_a = self.file_index_map[f]
                            index_b = self.file_index_map[other_file]
                            self.matrix[0, index_a, index_b] = cor
                            self.matrix[1, index_a, index_b] = lag
                            self.matrix[2, index_a, index_b] = sign
                            self.matrix[0, index_b, index_a] = cor
                            self.matrix[1, index_b, index_a] = -lag
                            self.matrix[2, index_b, index_a] = sign
                        pbar.update(1)
                    self._unload_file(f)
            self._unload_all()
            bag_index += 1
