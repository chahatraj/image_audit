import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import matplotlib.colors as mcolors

# === Settings ===
flux_path = "../outputs/evals/llama3/flux/llama3_flux_deductive.csv"
sd_path = "../outputs/evals/llama3/sd/llama3_sd_deductive.csv"

flux_save_dir = "../figs/flux"
sd_save_dir = "../figs/sd"

os.makedirs(flux_save_dir, exist_ok=True)
os.makedirs(sd_save_dir, exist_ok=True)

# Likert labels
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

likert_colors = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]

# === Helper Functions ===
def clean_template_id(value):
    if isinstance(value, str) and '_' in value:
        return value.split('_')[-1]
    return value

def get_text_color(background_color):
    rgb = mcolors.to_rgb(background_color)
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return 'white' if luminance < 0.5 else 'black'

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

# === Main Plotting Function ===
def plot_professional_percentage_bars(df, metric, save_dir, dataset_name):
    metric_folder = os.path.join(save_dir, metric)
    os.makedirs(metric_folder, exist_ok=True)

    fig = plt.figure(figsize=(36, 20))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.2, 1], wspace=0.05)

    ax1 = fig.add_subplot(gs[0, 0])  # Type1
    ax2 = fig.add_subplot(gs[0, 1])  # Aspect labels
    ax3 = fig.add_subplot(gs[0, 2])  # Type2

    # Prepare data
    melted = df[['aspect', 'sub_aspect', 'template_type', metric]].copy()
    melted = melted[melted['aspect'] != 'activity']
    melted['count'] = 1
    counts = melted.groupby(['template_type', 'aspect', 'sub_aspect', metric]).count().reset_index()

    # Pivot
    pivot = counts.pivot_table(index=['template_type', 'aspect', 'sub_aspect'], columns=metric, values='count', fill_value=0)

    # Normalize
    pivot_percent = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_percent = pivot_percent.sort_index(level=[1,2])  # aspect then sub_aspect

    # Prepare plotting
    type1 = pivot_percent.loc['type1']
    type2 = pivot_percent.loc['type2']

    subaspects = [sub for asp, sub in type1.index]
    aspects = [asp for asp, sub in type1.index]

    # Compute rows where aspect changes
    aspect_positions = []
    prev = None
    for idx, asp in enumerate(aspects):
        if asp != prev:
            aspect_positions.append(idx)
            prev = asp
    aspect_positions.append(len(aspects))  # Add final end

    # Alternate background shading
    for i in range(len(aspect_positions)-1):
        start = aspect_positions[i]
        end = aspect_positions[i+1]
        if i % 2 == 0:
            ax1.axhspan(start-0.5, end-0.5, color="#f0f0f0", zorder=-1)
            ax3.axhspan(start-0.5, end-0.5, color="#f0f0f0", zorder=-1)

    if metric == 'socio_cultural_stereotype':
        plotting_colors = likert_colors[::-1]  # reversed
    else:
        plotting_colors = likert_colors

    # Plot bars Type1
    left = [0] * len(type1)

    for score, color in zip([1,2,3,4,5], plotting_colors):
        if score in type1.columns:
            bar = ax1.barh(subaspects, type1[score], left=left, color=color, edgecolor="none")
            
            # Add percentage labels
            for idx, value in enumerate(type1[score]):
                if value > 1:
                    x = left[idx] + value / 2
                    y = idx
                    text_color = get_text_color(color)  # NEW
                    ax1.text(x, y, f"{value:.0f}%", ha='center', va='center', fontsize=14, color=text_color)


            left = [l + p for l, p in zip(left, type1[score])]

    # Plot bars Type2
    left = [0] * len(type2)

    for score, color in zip([1,2,3,4,5], plotting_colors):
        if score in type2.columns:
            bar = ax3.barh(subaspects, type2[score], left=left, color=color, edgecolor="none")
            
            # Add percentage labels
            for idx, value in enumerate(type2[score]):
                if value > 1:
                    x = left[idx] + value / 2
                    y = idx
                    text_color = get_text_color(color)  # NEW
                    ax3.text(x, y, f"{value:.0f}%", ha='center', va='center', fontsize=14, color=text_color)


            left = [l + p for l, p in zip(left, type2[score])]

    # Style type1 and type2 plots
    for ax in [ax1, ax3]:
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        # ax.set_xlabel('Percentage (%)', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.grid(axis='x', linestyle='--', alpha=0.7)

    ax1.set_yticks(range(len(subaspects)))
    ax1.set_yticklabels(subaspects, fontsize=24, ha='right')
    ax3.set_yticks(range(len(subaspects)))
    ax3.set_yticklabels(subaspects, fontsize=24, ha='left')

    ax1.yaxis.tick_left()
    ax1.yaxis.set_label_position("left")
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")

    # Aspect labels in the center
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_xlim(0, 1)
    ax2.axis('off')

    for i in range(len(aspect_positions)-1):
        start = aspect_positions[i]
        end = aspect_positions[i+1]
        mid = (start + end - 1) / 2
        aspect_name = aspects[start]
        ax2.text(0.5, mid, aspect_name, fontsize=24, fontweight='bold',
                 ha='center', va='center', rotation=0)

    # Legend
    handles = [patches.Patch(color=c, label=l) for c, l in zip(likert_colors, likert_labels[metric])]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=28)

    # Title
    fig.suptitle(f"{dataset_name.upper()} | {metric.replace('_', ' ').title()}", fontsize=30, y=0.95)

    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    save_path = os.path.join(metric_folder, f"{dataset_name}_{metric}_subaspect_with_aspect.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved: {save_path}")

# === Processing Function ===
def process_and_plot(input_path, save_dir, dataset_name):
    df = pd.read_csv(input_path)

    df['clean_templateid'] = df['template_id'].apply(clean_template_id)
    df['template_type'] = df['clean_templateid'].apply(map_template_type)
    df['base_filename'] = df['filename'].apply(normalize_filename)

    evaluation_columns = list(likert_labels.keys())

    for metric in evaluation_columns:
        plot_professional_percentage_bars(df, metric, save_dir, dataset_name)

# === Run for both datasets ===
process_and_plot(flux_path, flux_save_dir, dataset_name='flux')
process_and_plot(sd_path, sd_save_dir, dataset_name='sd')
