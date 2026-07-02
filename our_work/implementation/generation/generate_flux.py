import os
from datetime import datetime

import pandas as pd
import torch
from tqdm import tqdm
from diffusers import FluxPipeline

# =====================================================
# Configuration
# =====================================================

PROMPT_FILE = "../prompts/pilot_prompts.csv"
OUTPUT_DIR = "../../results/pilot"

MODEL_ID = "black-forest-labs/FLUX.1-schnell"

NUM_IMAGES = 20
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 0.0
MAX_SEQUENCE_LENGTH = 256

DEVICE_DTYPE = torch.bfloat16

# =====================================================
# Load Prompt CSV
# =====================================================

prompts_df = pd.read_csv(PROMPT_FILE)

prompts_df = prompts_df[prompts_df["include"].str.lower() == "yes"]

print(f"Loaded {len(prompts_df)} prompt pairs.")

# =====================================================
# Load Model
# =====================================================

print("Loading FLUX.1 Schnell...")

pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DEVICE_DTYPE
)

pipe.enable_model_cpu_offload()

print("Model loaded.")

# =====================================================
# Prepare Outputs
# =====================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

metadata = []

error_log = os.path.join(OUTPUT_DIR, "error_log.txt")

# =====================================================
# Generation Loop
# =====================================================

for _, row in prompts_df.iterrows():

    pair_id = row["pair_id"]

    prompt_list = [
        row["negative"],
        row["positive"]
    ]

    for prompt in prompt_list:

        prompt_folder = os.path.join(OUTPUT_DIR, prompt.replace(" ", "_"))
        os.makedirs(prompt_folder, exist_ok=True)

        print(f"\nGenerating: {prompt}")

        # Seed resets for every prompt
        for seed in tqdm(range(NUM_IMAGES)):

            try:

                generator = torch.Generator("cpu").manual_seed(seed)

                image = pipe(
                    prompt=prompt,
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    max_sequence_length=MAX_SEQUENCE_LENGTH,
                    generator=generator
                ).images[0]

                filename = f"{prompt.replace(' ','_')}_{seed:03d}.png"

                image_path = os.path.join(prompt_folder, filename)

                image.save(image_path)

                metadata.append({
                    "pair_id": pair_id,
                    "prompt": prompt,
                    "seed": seed,
                    "filename": filename,
                    "model": "FLUX.1-schnell",
                    "steps": NUM_INFERENCE_STEPS,
                    "guidance_scale": GUIDANCE_SCALE,
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:

                with open(error_log, "a") as f:
                    f.write(
                        f"[{datetime.now()}] "
                        f"Prompt={prompt} "
                        f"Seed={seed} "
                        f"Error={str(e)}\n"
                    )

                print(f"Skipped seed {seed}")

                continue

# =====================================================
# Save Metadata
# =====================================================

metadata_df = pd.DataFrame(metadata)

metadata_df.to_csv(
    os.path.join(OUTPUT_DIR, "metadata.csv"),
    index=False
)

print("\nGeneration Complete.")
print(f"Images saved to: {OUTPUT_DIR}")