import os
import json
import pandas as pd
import torch
import random
from diffusers import DiffusionPipeline, StableDiffusion3Pipeline
from torch import autocast
import argparse

# ========== ARGUMENTS ==========
parser = argparse.ArgumentParser(description="Generate images using Flux or Stable Diffusion.")
parser.add_argument("--model", choices=["flux", "sd"], required=True, help="Model to use: 'flux' or 'sd'")
args = parser.parse_args()
model_choice = args.model

def save_log(log_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as log_file:
        json.dump(log_dict, log_file, indent=4)

# ========== LOAD TEMPLATES AND NATIONALITIES ==========
templates_df = pd.read_csv("../data/templates.csv")
templates_df = templates_df[:1]  # Limit for testing

with open("../data/geoscheme_short.json", "r") as f:
    geoscheme = json.load(f)

def build_nationality_lookup(geojson):
    lookup = {}
    for continent, continent_data in geojson.items():
        for region, region_data in continent_data.items():
            continent_code = region_data.get("continent", "")
            for nationality in region_data["nationality_terms"]:
                lookup[nationality] = {
                    "continent": continent_code,
                    "continent_name": continent,
                    "region": region,
                }
    return lookup

nationality_lookup = build_nationality_lookup(geoscheme)
all_nationalities = sorted(nationality_lookup.keys())[:3]  # limit for testing
all_genders = ["male", "female"]

# ========== MODEL LOADING ==========
if model_choice == "flux":
    pipeline = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        cache_dir="/scratch/craj/model_cache/flux-lora-real",
        torch_dtype=torch.float16,
    )
elif model_choice == "sd":
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        cache_dir="/scratch/craj/model_cache/sd_35_large",
        torch_dtype=torch.float16,
    )
pipeline.enable_attention_slicing()
pipeline.to("cuda")

# ========== PATHS ==========
log_file_path = f"../outputs/generated_images/{model_choice}_generated_images_log.json"
if os.path.exists(log_file_path):
    with open(log_file_path, "r") as f:
        generated_images_log = json.load(f)
else:
    generated_images_log = {}

root_output_dir = f"../outputs/generated_images/{model_choice}/"
os.makedirs(root_output_dir, exist_ok=True)

# ========== GENERATE & SAVE ==========
def generate_and_save_image(prompt, save_dir, filename, log_key):
    image_path = os.path.join(save_dir, filename)

    if log_key in generated_images_log:
        print(f"Skipping already generated image: {log_key}")
        return None

    os.makedirs(save_dir, exist_ok=True)

    try:
        with autocast("cuda"):
            generator = torch.Generator("cuda").manual_seed(random.randint(0, 1000000))
            image = pipeline(prompt, generator=generator, num_inference_steps=20).images[0]
        image.save(image_path)
        print(f"Saved: {image_path}")
        return log_key, image_path
    except Exception as e:
        print(f"Error generating image for prompt '{prompt}': {str(e)}")
        return None

# ========== MAIN LOOP ==========
template_columns = [col for col in templates_df.columns if col.startswith("template_")]
generated_paths = {}

for _, row in templates_df.iterrows():
    aspect = row["aspect"].replace(" ", "_").lower()
    sub_aspect = row["sub_aspect"].replace(" ", "_").lower()

    for template_col in template_columns:
        raw_template = row[template_col]
        template_id = template_col

        for nationality in all_nationalities:
            if nationality not in nationality_lookup:
                continue

            info = nationality_lookup[nationality]
            continent = info["continent"]
            continent_name = info["continent_name"].replace(" ", "_").lower()
            region = info["region"].replace(" ", "_").lower()
            nationality_safe = nationality.replace(" ", "_").replace("/", "-").lower()

            for gender in all_genders:
                gender_safe = gender.lower()
                prompt = raw_template.replace("[nationality]", nationality).replace("[gender]", gender)

                filename = f"{aspect}_{sub_aspect}_{continent}_{region}_{gender_safe}_{nationality_safe}_{template_id}.png"
                save_dir = os.path.join(root_output_dir, template_id)
                log_key = filename

                # ✅ Check before calling generate
                if log_key in generated_images_log:
                    print(f"⏩ Skipping already generated image: {log_key}")
                    continue

                result = generate_and_save_image(prompt, save_dir, filename, log_key)
                if result:
                    key, path = result
                    generated_paths[key] = path
                    generated_images_log[key] = path

                    if len(generated_paths) % 50 == 0:
                        print("🔄 Saving intermediate log...")
                        save_log(generated_images_log, log_file_path)

# ========== SAVE LOG ==========
print("✅ Final log save...")
save_log(generated_images_log, log_file_path)

print("Finished. Total new images generated:", len(generated_paths))
