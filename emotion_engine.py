"""
Emotion Engine for SmartPrompt AI.
Uses rule-based NLP mapping to infer emotional context from simple prompts
and injects atmospheric modifiers. Designed for 0 VRAM overhead.
"""

import re
from typing import Dict

# Keyword to Mood mapping
EMOTION_KEYWORDS = {
    "sadness": ["sad", "alone", "cry", "lonely", "depressed", "grief", "heartbreak", "loss", "tears"],
    "happiness": ["happy", "joy", "smile", "laugh", "excited", "celebration", "cheerful", "glad"],
    "fear": ["scary", "fear", "terrified", "horror", "monster", "creepy", "spooky", "nightmare", "dread"],
    "mystery": ["mystery", "secret", "hidden", "unknown", "fog", "shadow", "enigma", "curious"],
    "romance": ["love", "kiss", "romantic", "couple", "passion", "heart", "intimate", "wedding"],
    "anger": ["angry", "rage", "furious", "mad", "wrath", "hate", "fight", "war"]
}

# Auto-assigned mood modifiers based on detected emotion
EMOTION_MODIFIERS = {
    "sadness": "melancholic, cinematic, emotional atmosphere, somber, tearful",
    "happiness": "uplifting, bright, joyful atmosphere, vibrant colors, sunny",
    "fear": "eerie, ominous, dark atmosphere, suspenseful, shadowy, dread",
    "mystery": "mysterious, hazy, foggy, intriguing lighting, secretive",
    "romance": "romantic atmosphere, warm glow, soft focus, passionate",
    "anger": "intense, chaotic, high contrast, sharp angles, fierce"
}

def analyze_emotion(prompt: str) -> Dict[str, str]:
    """
    Analyzes the user prompt and returns the detected emotion and its modifiers.
    Returns:
        dict: {"detected_emotion": str, "modifiers": str}
    """
    prompt_lower = prompt.lower()
    
    # Tokenize simply by splitting non-alphanumeric
    words = set(re.findall(r'\b\w+\b', prompt_lower))
    
    detected_emotion = "neutral"
    max_matches = 0
    
    for emotion, keywords in EMOTION_KEYWORDS.items():
        matches = len(words.intersection(set(keywords)))
        if matches > max_matches:
            max_matches = matches
            detected_emotion = emotion
            
    if detected_emotion == "neutral":
        return {"detected_emotion": "Neutral", "modifiers": ""}
        
    return {
        "detected_emotion": detected_emotion.capitalize(),
        "modifiers": EMOTION_MODIFIERS[detected_emotion]
    }
