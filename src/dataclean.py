import pandas as pd

# Paths to your files
flux_path = "../outputs/evals/llama3/flux/llama3_flux_deductive.csv"
sd_path = "../outputs/evals/llama3/sd/llama3_sd_deductive.csv"

# Read the CSVs
flux_df = pd.read_csv(flux_path)
sd_df = pd.read_csv(sd_path)

# Function to extract the part after the last underscore
def clean_template_id(value):
    if isinstance(value, str) and '_' in value:
        return value.split('_')[-1]  # split at underscores, take last part
    return value

# Create new column 'clean_templateid'
flux_df['clean_templateid'] = flux_df['template_id'].apply(clean_template_id)
sd_df['clean_templateid'] = sd_df['template_id'].apply(clean_template_id)

# Save back to the same files (overwrite)
flux_df.to_csv(flux_path, index=False)
sd_df.to_csv(sd_path, index=False)

print("✅ Corrected: New columns saved after extracting content after the last underscore!")
