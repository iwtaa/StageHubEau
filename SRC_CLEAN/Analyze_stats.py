import os
import json
import numpy as np
from matplotlib.patches import Patch

path = os.path.join(os.getcwd(), 'data')

stats_dir = os.path.join(path, 'stats')
json_files = [
    f for f in os.listdir(stats_dir)
    if f.endswith('.json') and f != 'files_paths.json'
]

average = {}

# Remove the subplot code and use plt.figure() for each graph
total_communes = {}
percentage_zero_values = {}

# Define thresholds
AVG_THRESHOLD = 40         # Example: only include if average_per_commune >= 10
TOTAL_COMMUNES_THRESHOLD = 0   # Example: only include if total_communes >= 5
ZERO_VALUES_THRESHOLD = 10     # Example: only include if percentage_zero_values <= 50

cdparams_legi = [
    1449,
    6455,
    6274,
    1046,
    1447,
    5440,
    1041,
    1376,
    1369,
    1396,
    1362,
    1388,
    1389,
    1392,
    1382,
    1394,
    1387,
    1386,
    1340,
    1339,
    1385,
]

for filename in json_files:
    with open(os.path.join(stats_dir, filename), 'r') as file:
        content = json.load(file)
        cdparam = content['cdparam']
        avg = content['observations']['average_per_commune']
        total = content['observations']['total_communes']
        zero = content['observations']['percentage_zero_values']

        if (
            (float(avg) >= AVG_THRESHOLD and
            float(total) >= TOTAL_COMMUNES_THRESHOLD and
            float(zero) <= ZERO_VALUES_THRESHOLD)
            or int(cdparam) in cdparams_legi
        ):
            average[cdparam] = avg
            total_communes[cdparam] = total
            percentage_zero_values[cdparam] = zero

import matplotlib.pyplot as plt

# Prepare data for plotting
cdparams = list(average.keys())
avg_values = [float(average[k]) for k in cdparams]
total_values = [float(total_communes[k]) for k in cdparams]
zero_values = [float(percentage_zero_values[k]) for k in cdparams]

# Sort by values
avg_sorted_indices = sorted(range(len(avg_values)), key=lambda k: avg_values[k])
total_sorted_indices = sorted(range(len(total_values)), key=lambda k: total_values[k])
zero_sorted_indices = sorted(range(len(zero_values)), key=lambda k: zero_values[k])

avg_sorted_values = [avg_values[i] for i in avg_sorted_indices]
total_sorted_values = [total_values[i] for i in total_sorted_indices]
zero_sorted_values = [zero_values[i] for i in zero_sorted_indices]

avg_sorted_labels = [cdparams[i] for i in avg_sorted_indices]
total_sorted_labels = [cdparams[i] for i in total_sorted_indices]
zero_sorted_labels = [cdparams[i] for i in zero_sorted_indices]

# Helper to get colors: highlight cdparams_legi
def get_bar_colors(labels, highlight_list, highlight_color, default_color):
    return [highlight_color if int(lbl) in highlight_list else default_color for lbl in labels]

# Average per Commune
plt.figure(figsize=(10, 4))
avg_bar_colors = get_bar_colors(avg_sorted_labels, cdparams_legi, 'orange', 'skyblue')
bars = plt.bar(avg_sorted_labels, avg_sorted_values, color=avg_bar_colors)
plt.title('Average per Commune')
plt.ylabel('Average')
plt.xticks(range(len(avg_sorted_labels)), avg_sorted_labels, rotation=90 if len(cdparams) < 30 else 0, ha='center' if len(cdparams) < 30 else 'right')
plt.ylim([0, max(avg_sorted_values) * 1.1 if avg_sorted_values else 1])
if AVG_THRESHOLD != 0:
    plt.axhline(AVG_THRESHOLD, color='red', linestyle='--', label=f'Threshold = {AVG_THRESHOLD}')
# Custom legend
legend_handles = [
    Patch(color='skyblue', label='Other cdparams'),
    Patch(color='orange', label='cdparams_legi'),
]
if AVG_THRESHOLD != 0:
    legend_handles.append(Patch(color='red', label=f'Threshold = {AVG_THRESHOLD}'))
plt.legend(handles=legend_handles)
plt.xlabel('cdparams')
plt.tight_layout()
plt.show()

# Total Communes
plt.figure(figsize=(10, 4))
total_bar_colors = get_bar_colors(total_sorted_labels, cdparams_legi, 'orange', 'lightgreen')
bars = plt.bar(total_sorted_labels, total_sorted_values, color=total_bar_colors)
plt.title('Total Communes')
plt.ylabel('Total')
plt.xticks(range(len(total_sorted_labels)), total_sorted_labels, rotation=90 if len(cdparams) < 30 else 0, ha='center' if len(cdparams) < 30 else 'right')
plt.ylim([0, max(total_sorted_values) * 1.1 if total_sorted_values else 1])
if TOTAL_COMMUNES_THRESHOLD != 0:
    plt.axhline(TOTAL_COMMUNES_THRESHOLD, color='red', linestyle='--', label=f'Threshold = {TOTAL_COMMUNES_THRESHOLD}')
legend_handles = [
    Patch(color='lightgreen', label='Other cdparams'),
    Patch(color='orange', label='cdparams_legi'),
]
if TOTAL_COMMUNES_THRESHOLD != 0:
    legend_handles.append(Patch(color='red', label=f'Threshold = {TOTAL_COMMUNES_THRESHOLD}'))
plt.legend(handles=legend_handles)
plt.xlabel('cdparams')
plt.tight_layout()
plt.show()

# Percentage Zero Values
plt.figure(figsize=(10, 4))
zero_bar_colors = get_bar_colors(zero_sorted_labels, cdparams_legi, 'orange', 'salmon')
bars = plt.bar(zero_sorted_labels, zero_sorted_values, color=zero_bar_colors)
plt.title('Percentage Zero Values')
plt.ylabel('Percentage')
plt.xticks(range(len(zero_sorted_labels)), zero_sorted_labels, rotation=90 if len(cdparams) < 30 else 0, ha='center' if len(cdparams) < 30 else 'right')
plt.ylim([0, 100])
plt.yticks(range(0, 101, 5))
if ZERO_VALUES_THRESHOLD != 100:
    plt.axhline(ZERO_VALUES_THRESHOLD, color='red', linestyle='--', label=f'Threshold = {ZERO_VALUES_THRESHOLD}')
legend_handles = [
    Patch(color='salmon', label='Other cdparams'),
    Patch(color='orange', label='cdparams_legi'),
]
if ZERO_VALUES_THRESHOLD != 100:
    legend_handles.append(Patch(color='red', label=f'Threshold = {ZERO_VALUES_THRESHOLD}'))
plt.legend(handles=legend_handles)
plt.xlabel('cdparams')
plt.tight_layout()
plt.show()

with open(os.path.join(path, '..', 'cdparams_selected.txt'), 'w') as f:
    for cdparam in cdparams:
        f.write(f"{cdparam}\n")
