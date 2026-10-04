"""
Generator Engine for SmartPrompt AI.
Handles Stable Diffusion inference with strict 4GB VRAM optimizations.
"""

import contextlib
import os
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from config import MODEL_ID, INFERENCE_STEPS, GUIDANCE_SCALE, IMAGE_SIZE
from utils import clear_memory
import streamlit as st

HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

@st.cache_resource
def load_pipeline():
    """
    Loads the SD pipeline once and caches it to prevent reloading.
    Optimized for 4GB VRAM using FP16 and attention slicing.
    """
    try:
        clear_memory()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load in float16 to save memory
        pipe_kwargs = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "use_safetensors": True,
        }
        if HF_TOKEN:
            pipe_kwargs["use_auth_token"] = HF_TOKEN

        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            **pipe_kwargs
        )

        if device == "cuda":
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            final_sigmas_type="sigma_min",
        )
        pipe.scheduler = scheduler
        pipe = pipe.to(device)

        return pipe
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def generate_image(
    positive_prompt: str,
    negative_prompt: str,
    num_inference_steps: int = INFERENCE_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
):
    """
    Generates an image from the prompt.
    """
    pipe = load_pipeline()
    if not pipe:
        return None

    clear_memory()
    device = pipe.device
    autocast_context = torch.autocast(device.type) if device.type == "cuda" else contextlib.nullcontext()

    try:
        with torch.inference_mode():
            with autocast_context:
                image = pipe(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=IMAGE_SIZE,
                    width=IMAGE_SIZE,
                ).images[0]

        clear_memory()
        return image
    except Exception as e:
        st.error(f"Generation error: {e}")
        return None