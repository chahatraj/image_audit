import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap

# === Settings ===
flux_path = "../outputs/evals/llama3/flux/llama3_flux_deductive.csv"
sd_path = "../outputs/evals/llama3/sd/llama3_sd_deductive.csv"

flux_save_dir = "../figs/flux"
sd_save_dir = "../figs/sd"

os.makedirs(flux_save_dir, exist_ok=True)
os.makedirs(sd_save_dir, exist_ok=True)

# Likert colors (we'll pick a nice purple-green cmap later)

# === Helper Functions ===
def clean_template_id(value):
    if isinstance(value, str) and '_' in value:
        return value.split('_')[-1]
    return value

def map_template_type(clean_id):
    if clean_id.startswith('1'):
        return 'type1'
    elif clean_id.startswith('2'):
        return 'type2'
    else:
        return 'unknown'

def normalize_filename(name):
    parts = name.split('_template_')
    return parts[0] if len(parts) > 1 else name

def calculate_error_rate(row, metric):
    """
    Calculate error rate for a single row.
    """
    if metric == 'socio_cultural_stereotype':
        error = row.get(3, 0) + row.get(4, 0) + row.get(5, 0)
    else:
        error = row.get(1, 0) + row.get(2, 0) + row.get(3, 0)
    return error

def plot_heatmap(data, title, save_path):
    plt.figure(figsize=(80, 15))
    sns.heatmap(data, cmap="PiYG_r", center=50, annot=True, fmt=".1f", linewidths=0.5, cbar_kws={"label": "Error Rate (%)"})
    plt.title(title, fontsize=26)
    plt.xlabel('Nationality', fontsize=22)
    plt.ylabel('Sub-Aspect', fontsize=22)
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved heatmap: {save_path}")

# === Main Processing Function ===
def process_and_plot_heatmaps(input_path, save_dir, dataset_name):
    df = pd.read_csv(input_path)

    df['clean_templateid'] = df['template_id'].apply(clean_template_id)
    df['template_type'] = df['clean_templateid'].apply(map_template_type)
    df['base_filename'] = df['filename'].apply(normalize_filename)

    evaluation_columns = ['prompt_relevance', 'semantic_accuracy', 'structural_fidelity', 'representational_accuracy', 'socio_cultural_stereotype']

    for metric in evaluation_columns:
        for template_type in ['type1', 'type2']:
            temp = df[df['template_type'] == template_type]

            melted = temp[['sub_aspect', 'nationality_safe', metric]].copy()
            melted['count'] = 1

            # Group and count each score per subaspect-nationality pair
            counts = melted.groupby(['sub_aspect', 'nationality_safe', metric]).count().reset_index()

            # Pivot
            pivot = counts.pivot_table(index=['sub_aspect', 'nationality_safe'], columns=metric, values='count', fill_value=0)

            # Normalize: percentages per subaspect-nationality
            pivot_percent = pivot.div(pivot.sum(axis=1), axis=0) * 100

            # Calculate error rate
            # error_rates = pivot_percent.groupby(['sub_aspect', 'nationality_safe']).apply(lambda x: calculate_error_rate(x, metric))
            # error_rates = error_rates.reset_index(name='error_rate')
            # error_rates = pivot_percent.groupby(['sub_aspect', 'nationality_safe']).agg(lambda x: calculate_error_rate(x, metric)).rename(columns={0: 'error_rate'}).reset_index()
            error_rates = pivot_percent.reset_index()
            error_rates['error_rate'] = error_rates.apply(lambda row: calculate_error_rate(row, metric), axis=1)
            error_rates = error_rates[['sub_aspect', 'nationality_safe', 'error_rate']]



            # Pivot for heatmap (sub_aspect as rows, nationality as columns)
            heatmap_data = error_rates.pivot(index='sub_aspect', columns='nationality_safe', values='error_rate')

            # Save heatmap
            metric_folder = os.path.join(save_dir, metric)
            os.makedirs(metric_folder, exist_ok=True)
            save_path = os.path.join(metric_folder, f"{dataset_name}_{metric}_{template_type}_subaspect_vs_nationality_heatmap.png")

            plot_heatmap(
                heatmap_data,
                title=f"{dataset_name.upper()} | {metric.replace('_', ' ').title()} | {template_type.upper()}",
                save_path=save_path
            )

# === Run for both datasets ===
process_and_plot_heatmaps(flux_path, flux_save_dir, dataset_name='flux')
process_and_plot_heatmaps(sd_path, sd_save_dir, dataset_name='sd')
