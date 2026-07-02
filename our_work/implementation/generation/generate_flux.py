import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from diffusers import AutoPipelineForText2Image

# =====================================================
# Configuration
# =====================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

PROMPT_FILE = PROJECT_ROOT / "our_work" / "implementation" / "prompts" / "pilot_prompts.csv"
OUTPUT_DIR = PROJECT_ROOT / "our_work" / "results" / "pilot"

MODEL_ID = "stabilityai/sdxl-turbo"

NUM_IMAGES = 20          # Change to 1 for smoke test
NUM_INFERENCE_STEPS = 1  # SDXL Turbo recommendation
GUIDANCE_SCALE = 0.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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

print("Loading SDXL Turbo...")

pipe = AutoPipelineForText2Image.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    variant="fp16"
)

pipe.to(DEVICE)

print("Model loaded.")

# =====================================================
# Prepare Output
# =====================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

metadata = []

error_log = OUTPUT_DIR / "error_log.txt"

# =====================================================
# Generation Loop
# =====================================================

for _, row in prompts_df.iterrows():

    pair_id = row["pair_id"]

    prompts = [
        row["negative"],
        row["positive"]
    ]

    for prompt in prompts:

        folder = OUTPUT_DIR / prompt.replace(" ", "_")
        folder.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating: {prompt}")

        for seed in tqdm(range(NUM_IMAGES)):

            try:

                generator = torch.Generator(device=DEVICE).manual_seed(seed)

                image = pipe(
                    prompt=prompt,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=generator
                ).images[0]

                filename = f"{prompt.replace(' ','_')}_{seed:03d}.png"

                image.save(folder / filename)

                metadata.append({
                    "pair_id": pair_id,
                    "prompt": prompt,
                    "seed": seed,
                    "filename": filename,
                    "model": MODEL_ID,
                    "steps": NUM_INFERENCE_STEPS,
                    "guidance_scale": GUIDANCE_SCALE,
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:

                with open(error_log, "a") as f:
                    f.write(
                        f"{datetime.now()} | "
                        f"{prompt} | "
                        f"Seed {seed} | "
                        f"{e}\n"
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