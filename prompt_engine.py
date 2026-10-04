"""Prompt Engine"""
from config import UNIVERSAL_NEGATIVE, QUALITY_PREFIX, QUALITY_SUFFIX
from ai_enhancer import enhance_prompt
from utils import format_prompt_component

def build_final_prompt(user_input: str) -> dict:
    """Build optimized positive and negative prompts (SDXL supports 256 tokens)."""
    enhanced_prompt, _ = enhance_prompt(user_input, fantasy_level=3, preserve_strict=True)
    
    # Combine with quality boosters (no truncation needed for SDXL)
    final_positive = f"{QUALITY_PREFIX}{enhanced_prompt}, {QUALITY_SUFFIX}"
    final_positive = format_prompt_component(final_positive)
    
    # Negative prompt - keep full but clean
    final_negative = format_prompt_component(UNIVERSAL_NEGATIVE)
    
    return {"positive": final_positive, "negative": final_negative}
