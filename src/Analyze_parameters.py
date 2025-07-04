import os
import json
import numpy as np
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

data_dir = os.path.join(os.getcwd(), 'data')
stats_dir = os.path.join(data_dir, 'stats')
json_files = [
    f for f in os.listdir(stats_dir)
    if f.endswith('.json') and f != 'files_paths.json'
]

average_per_commune = {}
commune_counts = {}
zero_value_percentages = {}

AVG_MIN = 10
COMMUNE_COUNT_MIN = 12500
ZERO_PERCENT_MAX = 15
highlighted_cdparams = [1396, 1392, 7073, 1340, 2036]

for file_name in json_files:
    with open(os.path.join(stats_dir, file_name), 'r') as file:
        stats = json.load(file)
        cdparam = stats['cdparam']
        avg = stats['observations']['average_per_commune']
        total = stats['observations']['total_communes']
        zero = stats['observations']['percentage_zero_values']
        if (
            (float(avg) >= AVG_MIN and
             float(total) >= COMMUNE_COUNT_MIN and
             float(zero) <= ZERO_PERCENT_MAX)
            or int(cdparam) in highlighted_cdparams
        ):
            average_per_commune[cdparam] = avg
            commune_counts[cdparam] = total
            zero_value_percentages[cdparam] = zero

cdparam_keys = list(average_per_commune.keys())
avg_values = [float(average_per_commune[k]) for k in cdparam_keys]
commune_counts_values = [float(commune_counts[k]) for k in cdparam_keys]
zero_percent_values = [float(zero_value_percentages[k]) for k in cdparam_keys]

avg_sorted_indices = sorted(range(len(avg_values)), key=lambda k: avg_values[k])
commune_counts_sorted_indices = sorted(range(len(commune_counts_values)), key=lambda k: commune_counts_values[k])
zero_percent_sorted_indices = sorted(range(len(zero_percent_values)), key=lambda k: zero_percent_values[k])

avg_sorted_values = [avg_values[i] for i in avg_sorted_indices]
commune_counts_sorted_values = [commune_counts_values[i] for i in commune_counts_sorted_indices]
zero_percent_sorted_values = [zero_percent_values[i] for i in zero_percent_sorted_indices]

avg_sorted_labels = [cdparam_keys[i] for i in avg_sorted_indices]
commune_counts_sorted_labels = [cdparam_keys[i] for i in commune_counts_sorted_indices]
zero_percent_sorted_labels = [cdparam_keys[i] for i in zero_percent_sorted_indices]

def bar_colors(labels, highlight_list, highlight_color, default_color):
    return [highlight_color if int(lbl) in highlight_list else default_color for lbl in labels]

# Plot 1: Average per Commune
plt.figure(figsize=(10, 4))
avg_bar_colors = bar_colors(avg_sorted_labels, highlighted_cdparams, 'orange', 'skyblue')
plt.bar(avg_sorted_labels, avg_sorted_values, color=avg_bar_colors)
plt.title('Average per Commune')
plt.ylabel('Average')
plt.xticks(range(len(avg_sorted_labels)), avg_sorted_labels, rotation=90, ha='center', fontsize=6)
for i, label in enumerate(plt.gca().get_xticklabels()):
    label.set_y(0.02 if i % 2 == 1 else -0.05)
plt.ylim([0, max(avg_sorted_values) * 1.1 if avg_sorted_values else 1])
if AVG_MIN != 0:
    plt.axhline(AVG_MIN, color='red', linestyle='--', label=f'Threshold = {AVG_MIN}')
legend_items = [
    Patch(color='skyblue', label='Other cdparams'),
    Patch(color='orange', label='Highlighted cdparams'),
]
if AVG_MIN != 0:
    legend_items.append(Patch(color='red', label=f'Threshold = {AVG_MIN}'))
plt.legend(handles=legend_items)
plt.xlabel('cdparams')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'average_per_commune_const.png'))
plt.show()

# Plot 2: Total Communes
plt.figure(figsize=(10, 4))
commune_counts_bar_colors = bar_colors(commune_counts_sorted_labels, highlighted_cdparams, 'orange', 'lightgreen')
plt.bar(commune_counts_sorted_labels, commune_counts_sorted_values, color=commune_counts_bar_colors)
plt.title('Total Communes')
plt.ylabel('Total')
plt.xticks(range(len(commune_counts_sorted_labels)), commune_counts_sorted_labels, rotation=90, ha='center', fontsize=6)
for i, label in enumerate(plt.gca().get_xticklabels()):
    label.set_y(0.02 if i % 2 == 1 else -0.05)
plt.ylim([0, max(commune_counts_sorted_values) * 1.1 if commune_counts_sorted_values else 1])
if COMMUNE_COUNT_MIN != 0:
    plt.axhline(COMMUNE_COUNT_MIN, color='red', linestyle='--', label=f'Threshold = {COMMUNE_COUNT_MIN}')
legend_items = [
    Patch(color='lightgreen', label='Other cdparams'),
    Patch(color='orange', label='Highlighted cdparams'),
]
if COMMUNE_COUNT_MIN != 0:
    legend_items.append(Patch(color='red', label=f'Threshold = {COMMUNE_COUNT_MIN}'))
plt.legend(handles=legend_items)
plt.xlabel('cdparams')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'total_communes_const.png'))
plt.show()

# Plot 3: Percentage Zero Values
plt.figure(figsize=(10, 4))
zero_percent_bar_colors = bar_colors(zero_percent_sorted_labels, highlighted_cdparams, 'orange', 'salmon')
plt.bar(zero_percent_sorted_labels, zero_percent_sorted_values, color=zero_percent_bar_colors)
plt.title('Percentage Zero Values')
plt.ylabel('Percentage')
plt.xticks(range(len(zero_percent_sorted_labels)), zero_percent_sorted_labels, rotation=90, ha='center', fontsize=6)
for i, label in enumerate(plt.gca().get_xticklabels()):
    label.set_y(0.02 if i % 2 == 1 else -0.05)
plt.ylim([0, 100])
plt.yticks(range(0, 101, 5))
if ZERO_PERCENT_MAX != 100:
    plt.axhline(ZERO_PERCENT_MAX, color='red', linestyle='--', label=f'Threshold = {ZERO_PERCENT_MAX}')
legend_items = [
    Patch(color='salmon', label='Other cdparams'),
    Patch(color='orange', label='Highlighted cdparams'),
]
if ZERO_PERCENT_MAX != 100:
    legend_items.append(Patch(color='red', label=f'Threshold = {ZERO_PERCENT_MAX}'))
plt.legend(handles=legend_items)
plt.xlabel('cdparams')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'percentage_zero_values_const.png'))
plt.show()

with open(os.path.join(data_dir, '..', 'cdparams_selected_Const.txt'), 'w') as f:
    for cdparam in cdparam_keys:
        f.write(f"{cdparam}\n")
