import os
import json

path = '/home/iwta/Documents/Univ/StageHubEau/data/'

stats_dir = os.path.join(path, 'stats')
json_files = [
    f for f in os.listdir(stats_dir)
    if f.endswith('.json') and f != 'files_paths.json'
]

average = {}
total_communes = {}
percentage_zero_values = {}

# Define thresholds
AVG_THRESHOLD = 20         # Example: only include if average_per_commune >= 10
TOTAL_COMMUNES_THRESHOLD = 10000   # Example: only include if total_communes >= 5
ZERO_VALUES_THRESHOLD = 15     # Example: only include if percentage_zero_values <= 50

for filename in json_files:
    with open(os.path.join(stats_dir, filename), 'r') as file:
        content = json.load(file)
        cdparam = content['cdparam']
        avg = content['observations']['average_per_commune']
        total = content['observations']['total_communes']
        zero = content['observations']['percentage_zero_values']

        if (
            float(avg) >= AVG_THRESHOLD and
            float(total) >= TOTAL_COMMUNES_THRESHOLD and
            float(zero) <= ZERO_VALUES_THRESHOLD
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

fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Average per Commune
axs[0].bar(avg_sorted_labels, avg_sorted_values, color='skyblue')
axs[0].set_title('Average per Commune')
axs[0].set_ylabel('Average')
axs[0].set_xticks(range(len(avg_sorted_labels)))
axs[0].set_xticklabels(avg_sorted_labels, rotation=45, ha='right')
axs[0].set_ylim([0, max(avg_sorted_values) * 1.1 if avg_sorted_values else 1])

# Total Communes
axs[1].bar(total_sorted_labels, total_sorted_values, color='lightgreen')
axs[1].set_title('Total Communes')
axs[1].set_ylabel('Total')
axs[1].set_xticks(range(len(total_sorted_labels)))
axs[1].set_xticklabels(total_sorted_labels, rotation=45, ha='right')
axs[1].set_ylim([0, max(total_sorted_values) * 1.1 if total_sorted_values else 1])

# Percentage Zero Values
axs[2].bar(zero_sorted_labels, zero_sorted_values, color='salmon')
axs[2].set_title('Percentage Zero Values')
axs[2].set_ylabel('Percentage')
axs[2].set_xticks(range(len(zero_sorted_labels)))
axs[2].set_xticklabels(zero_sorted_labels, rotation=45, ha='right')
axs[2].set_ylim([0, 100])
axs[2].set_yticks(range(0, 101, 5))

if len(cdparams) < 30:
    axs[0].set_xticklabels(avg_sorted_labels, rotation=90, ha='center')
    axs[1].set_xticklabels(total_sorted_labels, rotation=90, ha='center')
    axs[2].set_xticklabels(zero_sorted_labels, rotation=90, ha='center')
if len(cdparams) >= 30:
    for ax in axs:
        ax.set_xticklabels([])
        ax.set_xlabel('cdparams')
else:
    for ax in axs:
        ax.set_xlabel('cdparams')

# Add threshold lines if thresholds are different from default axis limits
if AVG_THRESHOLD != 0:
    axs[0].axhline(AVG_THRESHOLD, color='red', linestyle='--', label=f'Threshold = {AVG_THRESHOLD}')
    axs[0].legend()

if TOTAL_COMMUNES_THRESHOLD != 0:
    axs[1].axhline(TOTAL_COMMUNES_THRESHOLD, color='red', linestyle='--', label=f'Threshold = {TOTAL_COMMUNES_THRESHOLD}')
    axs[1].legend()

if ZERO_VALUES_THRESHOLD != 100:
    axs[2].axhline(ZERO_VALUES_THRESHOLD, color='red', linestyle='--', label=f'Threshold = {ZERO_VALUES_THRESHOLD}')
    axs[2].legend()

plt.tight_layout()
plt.show()

with open(os.path.join(path, '..', 'cdparams_selected.txt'), 'w') as f:
    for cdparam in cdparams:
        f.write(f"{cdparam}\n")