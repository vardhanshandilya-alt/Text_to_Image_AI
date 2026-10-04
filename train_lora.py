"""
LoRA Fine-tuning Script for Stable Diffusion.
Fixed version: Properly uses the VAE to encode images into latent space
before adding noise, ensuring correct LDM training.
"""

import argparse
import os
from PIL import Image
import torch
from torchvision import transforms
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model

# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "Lykon/dreamshaper-8"  # Use the same base model as generation
DATA_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "caption.txt"))
IMAGE_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "images"))
OUTPUT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "models", "lora_output"))

EPOCHS = 3
LR = 1e-4
IMAGE_SIZE = 512

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("WARNING: Training on CPU is extremely slow. A GPU is recommended.")

    # ==============================
    # LOAD MODEL
    # ==============================
    print(f"Loading pipeline from {MODEL_NAME}...")
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    pipe = pipe.to(device)

    # Freeze VAE and Text Encoder to save memory
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    unet = pipe.unet
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    scheduler = pipe.scheduler
    vae = pipe.vae

    # ==============================
    # APPLY LoRA
    # ==============================
    print("Applying LoRA to UNet...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.1,
        bias="none"
    )
    unet = get_peft_model(unet, lora_config)

    # ==============================
    # LOAD DATA
    # ==============================
    def load_data(file_path):
        data = []
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found.")
            return data
            
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if "|" in line:
                    img, cap = line.strip().split("|", 1)
                    data.append((img.strip(), cap.strip()))
        return data

    dataset = load_data(DATA_PATH)
    if not dataset:
        print("No data found. Exiting.")
        return

    # ==============================
    # IMAGE TRANSFORM
    # ==============================
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    optimizer = torch.optim.AdamW(unet.parameters(), lr=LR)

    # ==============================
    # TRAINING LOOP
    # ==============================
    print("Starting training...")
    for epoch in range(EPOCHS):
        total_loss = 0

        for img_name, caption in dataset:
            img_path = os.path.join(IMAGE_DIR, img_name)

            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} not found. Skipping.")
                continue

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {img_name}: {e}. Skipping.")
                continue

            # 1. Transform Image
            pixel_values = transform(image).unsqueeze(0).to(device)

            # 2. Tokenize text
            inputs = tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt"
            ).to(device)

            # 3. Get Text Embeddings
            with torch.no_grad():
                text_embeddings = text_encoder(**inputs).last_hidden_state

            # 4. Convert Image to Latent Space using VAE (CRITICAL FIX)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # 5. Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # 6. Predict noise
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings
            ).sample

            # 7. Compute Loss & Optimize
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

    # ==============================
    # SAVE MODEL
    # ==============================
    print(f"Saving LoRA weights to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    unet.save_pretrained(OUTPUT_DIR)
    print("✅ LoRA training complete!")

if __name__ == "__main__":
    train()
