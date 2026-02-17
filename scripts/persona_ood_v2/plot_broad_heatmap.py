"""
Heatmap of category-level mean deltas for broad personas in persona OOD v2 experiment.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load analysis
with open('/Users/oscargilg/Dev/MATS/Preferences/experiments/probe_generalization/persona_ood/v2_analysis.json') as f:
    data = json.load(f)

broad_personas = data['broad']

# Extract personas, means, and categories
personas = [p['name'] for p in broad_personas]
persona_means = [p['mean_delta'] for p in broad_personas]

# Collect all unique categories
all_categories = set()
for persona in broad_personas:
    all_categories.update(persona['category_means'].keys())

all_categories = sorted(all_categories)

# Build matrix: personas x categories
matrix = np.zeros((len(personas), len(all_categories)))
for i, persona in enumerate(broad_personas):
    for j, category in enumerate(all_categories):
        if category in persona['category_means']:
            matrix[i, j] = persona['category_means'][category]['mean']

# Sort rows by overall mean delta (ascending)
sort_idx = np.argsort(persona_means)
matrix_sorted = matrix[sort_idx]
personas_sorted = [personas[i] for i in sort_idx]

# Sort columns by column mean (ascending)
col_means = np.nanmean(matrix_sorted, axis=0)
col_sort_idx = np.argsort(col_means)
matrix_final = matrix_sorted[:, col_sort_idx]
categories_sorted = [all_categories[i] for i in col_sort_idx]

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Create heatmap with diverging colormap centered at 0
vmax = np.nanmax(np.abs(matrix_final))
sns.heatmap(
    matrix_final,
    cmap='RdBu_r',
    center=0,
    vmin=-vmax,
    vmax=vmax,
    xticklabels=categories_sorted,
    yticklabels=personas_sorted,
    cbar_kws={'label': 'Mean preference delta'},
    ax=ax,
    linewidths=0.5,
    linecolor='gray'
)

# Mark cells where shift matches expected direction
for i, persona_data in enumerate(broad_personas):
    persona_idx = personas_sorted.index(persona_data['name'])

    expected_positive = set(persona_data.get('expected_positive', []))
    expected_negative = set(persona_data.get('expected_negative', []))

    for j, category in enumerate(categories_sorted):
        cell_value = matrix_final[persona_idx, j]

        # Mark with asterisk if direction matches expectation
        is_hit = False
        if category in expected_positive and cell_value > 0:
            is_hit = True
        elif category in expected_negative and cell_value < 0:
            is_hit = True

        if is_hit:
            ax.text(
                j + 0.5, persona_idx + 0.7,
                '*',
                ha='center', va='center',
                fontsize=16, fontweight='bold',
                color='black'
            )

ax.set_title('Broad personas: Category-level preference shifts', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Persona (sorted by mean delta)', fontsize=12, fontweight='bold')

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save
output_path = '/Users/oscargilg/Dev/MATS/Preferences/experiments/probe_generalization/persona_ood/assets/plot_021726_broad_persona_heatmap.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved to {output_path}")
