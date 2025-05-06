import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Settings ===
flux_path = "../outputs/evals/llama3/flux/llama3_flux_deductive.csv"
sd_path = "../outputs/evals/llama3/sd/llama3_sd_deductive.csv"

flux_save_dir = "../figs/flux"
sd_save_dir = "../figs/sd"

# Make sure output folders exist
os.makedirs(flux_save_dir, exist_ok=True)
os.makedirs(sd_save_dir, exist_ok=True)

# Likert labels mapping
likert_labels = {
    'prompt_relevance': [
        "Completely Irrelevant", "Mostly Irrelevant", "Partially Relevant", 
        "Mostly Relevant", "Perfectly Relevant"
    ],
    'semantic_accuracy': [
        "Nonsensical", "Mostly Implausible", "Partially Coherent", 
        "Mostly Coherent", "Fully Coherent"
    ],
    'structural_fidelity': [
        "Severely Flawed", "Major Errors", "Noticeable Errors", 
        "Slightly Flawed", "Structurally Flawless"
    ],
    'representational_accuracy': [
        "Completely Misrepresented", "Strong Misrepresentation", "Some Misrepresentation", 
        "Minor Misrepresentation", "Accurate Representation"
    ],
    'socio_cultural_stereotype': [
        "Not Stereotypical", "Subtly Stereotypical", "Moderately Stereotypical", 
        "Strongly Stereotypical", "Highly Stereotypical"
    ]
}

# Color palette for scores 1-5
likert_colors = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]

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

def plot_percentage_bars(df, group_by, metric, save_dir, dataset_name):
    # Create subfolder for each metric
    metric_folder = os.path.join(save_dir, metric)
    os.makedirs(metric_folder, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(24, 12), sharey=True)
    fig.suptitle(f"{dataset_name.upper()} | {metric.replace('_', ' ').title()} - Grouped by {group_by.title()}", fontsize=22)

    for i, template_type in enumerate(['type1', 'type2']):
        temp = df[df['template_type'] == template_type]

        # Group by group_by + metric
        melted = temp[[group_by, metric]].copy()
        melted['count'] = 1

        # Group and count how many 1/2/3/4/5 per group
        counts = melted.groupby([group_by, metric]).count().reset_index()

        # Pivot to wide format: rows = group, columns = 1-5
        pivot = counts.pivot(index=group_by, columns=metric, values='count').fillna(0)

        # Normalize to percentage across 1-5 for each group
        pivot_percent = pivot.div(pivot.sum(axis=1), axis=0) * 100

        # Sort index
        pivot_percent = pivot_percent.sort_index()

        # Plot
        ax = axes[i]
        left = [0] * len(pivot_percent)

        labels = likert_labels[metric]

        for score, color, label in zip([1,2,3,4,5], likert_colors, labels):
            if score in pivot_percent.columns:
                ax.barh(
                    pivot_percent.index,
                    pivot_percent[score],
                    left=left,
                    color=color,
                    label=label
                )
                left += pivot_percent[score]

        ax.set_xlim(0, 100)
        ax.set_xlabel('Percentage (%)', fontsize=16)
        ax.set_title(f"{template_type.upper()}", fontsize=18)
        ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Shared legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=14)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    # Save plot
    filename = f"{dataset_name}_{metric}_{group_by}.png"
    filepath = os.path.join(metric_folder, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"✅ Saved: {filepath}")

# === Processing Function ===
def process_and_plot(input_path, save_dir, dataset_name):
    df = pd.read_csv(input_path)

    # Create new columns
    df['clean_templateid'] = df['template_id'].apply(clean_template_id)
    df['template_type'] = df['clean_templateid'].apply(map_template_type)
    df['base_filename'] = df['filename'].apply(normalize_filename)

    # NO averaging! Keep raw rows
    # Plotting
    grouping_vars = ['aspect', 'sub_aspect', 'continent', 'region', 'gender_safe', 'nationality_safe']
    evaluation_columns = list(likert_labels.keys())

    for metric in evaluation_columns:
        for group_by in grouping_vars:
            plot_percentage_bars(df, group_by, metric, save_dir, dataset_name)

# === Run for both datasets ===
process_and_plot(flux_path, flux_save_dir, dataset_name='flux')
process_and_plot(sd_path, sd_save_dir, dataset_name='sd')
