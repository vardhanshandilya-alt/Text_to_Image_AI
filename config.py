"""
Configuration parameters for SmartPrompt AI.
Includes models, UI settings, prompt dictionaries, and negative prompts.
"""

# ========================================
# MODEL SETTINGS
# ========================================
# SDXL supports 256 tokens + higher quality. Optimized for 4GB VRAM with FP16.
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
INFERENCE_STEPS = 28
GUIDANCE_SCALE = 7.5
IMAGE_SIZE = 768

# ========================================
# QUALITY ENHANCEMENT
# ========================================
QUALITY_PREFIX = "high quality, high detail, hyper realistic, professional photography, masterpiece, "
QUALITY_SUFFIX = "8k resolution, sharp focus, intricate details, cinematic lighting, professional composition"

# ========================================
# NEGATIVE PROMPT
# ========================================
UNIVERSAL_NEGATIVE = (
    "ugly, blurry, low quality, bad anatomy, bad proportions, malformed, distorted, deformed, "
    "watermark, text, signature, lowres, jpeg artifacts, cropped, out of frame, worst quality, "
    "gross proportions, bad hands, extra limbs, missing limbs, mutated, disfigured"
)
