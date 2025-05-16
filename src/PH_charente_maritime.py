from APIcalls import *
import json
import numpy as np
from collections import Counter

data = getMeasureDepartment('17', parameter='1302', date_max_prelevement='2025-01-01 00:00:00', date_min_prelevement='2024-07-01 00:00:00')

results = np.array([])
for measure in data['data']:
    results = np.append(results, measure['resultat_numerique'])

# Use Counter to count occurrences of each value
unique_values, value_counts = np.unique(results, return_counts=True)
# Create a dictionary from the unique values and their counts
counts = dict(zip(unique_values, value_counts))

# Print the counts
for value, count in counts.items():
    print(f"{value}: {count}")

import matplotlib.pyplot as plt

# Ensure unique_values are strings for even spacing
unique_values_str = [str(val) for val in unique_values]

# Create a bar graph with string labels
plt.bar(unique_values_str, value_counts, width=0.5)  # Specify the width of the bars
plt.xlabel('Value')
plt.ylabel('Count')
plt.title('Distribution of PH measures in Charente-Maritime betwen September and December of 2024')
plt.grid(True)
plt.show()