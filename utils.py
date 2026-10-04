"""
Utility functions for SmartPrompt AI.
Handles memory management and generic helpers.
"""

import torch
import gc

def clear_memory():
    """
    Aggressively clears GPU and CPU memory.
    Crucial for 4GB VRAM environments.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def format_prompt_component(component: str) -> str:
    """
    Cleans up string components for the prompt.
    """
    if not component:
        return ""
    return component.strip().strip(",")
