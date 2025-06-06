import itertools
import numpy as np
import pandas as pd

class MatrixPairedFileProcessor:
    def __init__(self, files_info, max_memory_mb, calculation_function):
        self.files_info = files_info
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.loaded_data = {}
        self.current_memory_usage = 0
        self._do_calculation = calculation_function
        self.total_memory_loaded = 0
        self.matrix = np.zeros((len(files_info), len(files_info)), dtype=np.float32)
        self.saved_loads = 0
        self.file_index_map = {filename: i for i, filename in enumerate(files_info.keys())}

    def get_matrix(self):
        return np.triu(self.matrix) + np.tril(self.matrix.T, k=-1)

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
        files_list = sorted(list(self.files_info.keys()), key=lambda x: self.files_info[x], reverse=True)
        processed_files = set()
        remaining_files = [f for f in files_list if f not in processed_files]
        while remaining_files:
            bag_memory = 0
            for f in remaining_files:
                if bag_memory + self.files_info[f] <= self.max_memory_bytes:
                    bag_memory += self.files_info[f]
                    processed_files.add(f)
                    self._load_file(f)
            for file1, file2 in itertools.combinations(self.loaded_data.keys(), 2):
                i1 = self.file_index_map[file1]
                i2 = self.file_index_map[file2]
                res = self._do_calculation(self.loaded_data[file1], self.loaded_data[file2])
                self.matrix[i1, i2] = res
                self.matrix[i2, i1] = res
            remaining_files = [f for f in files_list if f not in self.loaded_data]
            for f in remaining_files:
                if self._load_file(f):
                    for loaded_f in list(self.loaded_data.keys()):
                        if f == loaded_f:
                            continue
                        i1 = self.file_index_map[f]
                        i2 = self.file_index_map[loaded_f]
                        res = self._do_calculation(f, loaded_f)
                        self.matrix[i1, i2] = res
                        self.matrix[i2, i1] = res
                    self._unload_file(f)
            self._unload_all()
            remaining_files = [f for f in files_list if f not in processed_files]