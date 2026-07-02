import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline

# =====================================================
# Configuration
# =====================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

PROMPT_FILE = PROJECT_ROOT / "our_work" / "implementation" / "prompts" / "pilot_prompts.csv"
OUTPUT_DIR = PROJECT_ROOT / "our_work" / "results" / "pilot"

MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"

# Use 1 first to verify everything works.
NUM_IMAGES = 1

NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 4.5

HEIGHT = 1024
WIDTH = 1024

DTYPE = torch.float16

# =====================================================
# Load Prompt CSV
# =====================================================

prompts_df = pd.read_csv(PROMPT_FILE)
prompts_df = prompts_df[prompts_df["include"].str.lower() == "yes"]

print(f"Loaded {len(prompts_df)} prompt pairs.")

# =====================================================
# Load Model
# =====================================================

print("Loading Stable Diffusion 3.5 Medium...")

pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
)

# -----------------------------------------------------
# Memory Optimizations (for Colab T4)
# -----------------------------------------------------

pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing()

print("Model loaded.")

# =====================================================
# Prepare Output
# =====================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

metadata = []

error_log = OUTPUT_DIR / "error_log.txt"

# =====================================================
# Generation
# =====================================================

for _, row in prompts_df.iterrows():

    pair_id = row["pair_id"]

    adjectives = [
        row["negative"],
        row["positive"]
    ]

    for adjective in adjectives:

        prompt = (
            f"A realistic candid photograph of a single adult person who appears "
            f"{adjective}, standing naturally, neutral background, DSLR photo."
        )

        folder_name = adjective.replace(" ", "_").replace("/", "_")

        folder = OUTPUT_DIR / folder_name
        folder.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating: {adjective}")

        for seed in tqdm(range(NUM_IMAGES), leave=False):

            try:

                generator = torch.Generator("cpu").manual_seed(seed)

                image = pipe(
                    prompt=prompt,
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=generator,
                ).images[0]

                filename = f"{folder_name}_{seed:03d}.png"

                image.save(folder / filename)

                metadata.append({
                    "pair_id": pair_id,
                    "prompt": adjective,
                    "full_prompt": prompt,
                    "seed": seed,
                    "filename": filename,
                    "model": MODEL_ID,
                    "steps": NUM_INFERENCE_STEPS,
                    "guidance_scale": GUIDANCE_SCALE,
                    "height": HEIGHT,
                    "width": WIDTH,
                    "timestamp": datetime.now().isoformat(timespec="seconds")
                })

            except Exception as e:

                with open(error_log, "a", encoding="utf-8") as f:
                    f.write(
                        f"{datetime.now().isoformat(timespec='seconds')} | "
                        f"{adjective} | "
                        f"Seed {seed} | "
                        f"{str(e)}\n"
                    )

                print(f"Skipped seed {seed}")

# =====================================================
# Save Metadata
# =====================================================

metadata_df = pd.DataFrame(metadata)

metadata_df.to_csv(
    OUTPUT_DIR / "metadata.csv",
    index=False
)

print("\nGeneration Complete.")
print(f"Saved to {OUTPUT_DIR}")