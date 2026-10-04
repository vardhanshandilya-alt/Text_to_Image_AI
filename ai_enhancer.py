"""
AI Enhancer for SmartPrompt AI.
Expands a user's prompt into a richer, more accurate image generation prompt
by preserving intent, adding descriptive detail, and appending contextual
modifiers for higher quality results.
"""

import re
from typing import Tuple

# Core subject mapping to richer visual descriptions
SUBJECT_ENHANCEMENTS = {
    "dog": "a highly detailed dog, realistic fur texture, expressive eyes",
    "cat": "a beautiful detailed cat, piercing eyes, soft fur, elegant posture",
    "man": "a highly detailed man, sharp facial features, realistic skin texture, expressive",
    "woman": "a stunningly detailed woman, beautiful facial features, realistic skin texture, graceful",
    "car": "a highly detailed sleek car, metallic reflections, realistic textures, automotive photography",
    "house": "a beautifully detailed house, realistic architectural rendering, inviting",
    "city": "a sprawling detailed city, complex architecture, bustling streets",
    "forest": "a dense detailed forest, lush vegetation, ancient trees",
    "space": "a vast detailed cosmic space, nebula clouds, sparkling stars, awe-inspiring",
    "boy": "a highly detailed young boy, expressive face, realistic features",
    "girl": "a highly detailed young girl, expressive face, realistic features",
}

# Contextual prompt enrichments for environments, lighting, and mood cues
ENVIRONMENT_ENHANCEMENTS = {
    "sunset": "warm golden hour lighting, dramatic clouds, rich shadows",
    "sunrise": "soft dawn lighting, misty horizon, pastel sky",
    "night": "night scene, moody shadows, neon reflections, deep contrast",
    "rain": "rain-soaked surfaces, glistening reflections, atmospheric wet streets",
    "snow": "snow-covered landscape, delicate falling snow, cold blue lighting",
    "forest": "lush foliage, dappled light, textured moss, immersive greenery",
    "city": "urban details, bustling streets, glowing signs, cinematic depth",
    "desert": "sandy dunes, warm sunset glow, wind-swept textures",
    "ocean": "sparkling water, coastal waves, reflective surface, sea breeze",
}

ADJECTIVE_ENHANCEMENTS = {
    "vibrant": "rich saturated colors, bold contrast, vivid detail",
    "mysterious": "moody atmosphere, subtle fog, hidden details",
    "epic": "monumental scale, dramatic composition, cinematic presence",
    "dark": "shadowy lighting, dramatic contrast, intense atmosphere",
    "romantic": "soft glow, warm hues, intimate composition",
    "magical": "glowing particles, enchanting aura, fantastical elements",
    "realistic": "life-like textures, accurate anatomy, natural lighting",
    "vintage": "film grain, muted tones, classic analog styling",
}

QUALITY_BOOSTERS = "ultra detailed, masterpiece, high quality, 8k resolution, professional composition"


def normalize_prompt(prompt: str) -> str:
    """Normalize whitespace and remove duplicate commas."""
    prompt = re.sub(r"\s+", " ", prompt.strip())
    prompt = re.sub(r",\s*", ", ", prompt)
    prompt = re.sub(r"(,\s*)+", ", ", prompt)
    return prompt.strip(" ,")


def dedupe_prompt(prompt: str) -> str:
    """Remove repeated comma-separated clauses while preserving order."""
    parts = [part.strip() for part in prompt.split(",") if part.strip()]
    seen = set()
    result = []
    for part in parts:
        normalized = part.lower()
        if normalized not in seen:
            seen.add(normalized)
            result.append(part)
    return ", ".join(result)


def enhance_subject(prompt: str) -> str:
    """Enhance the prompt subject with richer descriptions when a known noun appears."""
    prompt = prompt.strip()
    for key, enhancement in SUBJECT_ENHANCEMENTS.items():
        pattern = rf"\b{re.escape(key)}\b"
        if re.search(pattern, prompt, flags=re.IGNORECASE):
            return re.sub(pattern, enhancement, prompt, count=1, flags=re.IGNORECASE)
    return prompt


def apply_contextual_enhancements(prompt: str) -> str:
    """Add environment and adjective details based on user keywords."""
    enriched = prompt
    for keyword, enhancement in ENVIRONMENT_ENHANCEMENTS.items():
        if re.search(rf"\b{re.escape(keyword)}\b", prompt, flags=re.IGNORECASE):
            if enhancement.lower() not in enriched.lower():
                enriched = f"{enriched}, {enhancement}"
    for keyword, enhancement in ADJECTIVE_ENHANCEMENTS.items():
        if re.search(rf"\b{re.escape(keyword)}\b", prompt, flags=re.IGNORECASE):
            if enhancement.lower() not in enriched.lower():
                enriched = f"{enriched}, {enhancement}"
    return enriched


def apply_fantasy_level(base_prompt: str, level: int) -> str:
    """Injects fantasy and style modifiers based on the 0-10 scale."""
    if level == 0:
        return f"{base_prompt}, grounded in reality, hyper-realistic"
    if level <= 3:
        return f"{base_prompt}, subtle artistic enhancement, slightly stylized, cinematic detail"
    if level <= 5:
        return f"{base_prompt}, magical atmosphere, glowing accents, ethereal light"
    if level <= 8:
        return f"{base_prompt}, epic fantasy world, majestic details, otherworldly atmosphere"
    return f"{base_prompt}, surreal dreamlike universe, physics-defying forms, cosmic scale, masterpiece"


def enhance_prompt(prompt: str, fantasy_level: int, preserve_strict: bool = True) -> Tuple[str, str]:
    """Main enhancement function that preserves the user's intent while enriching the prompt.

    Args:
        prompt: original user text
        fantasy_level: 0-10 fantasy/intensity scale
        preserve_strict: when True, avoid replacing user's core nouns with mapped enhancements
    """
    original = normalize_prompt(prompt)
    # Optionally avoid subject substitution to strictly preserve user wording
    if preserve_strict:
        enhanced = original
    else:
        enhanced = enhance_subject(original)

    enhanced = apply_contextual_enhancements(enhanced)
    enhanced = apply_fantasy_level(enhanced, fantasy_level)
    enhanced = f"{enhanced}, {QUALITY_BOOSTERS}"
    enhanced = dedupe_prompt(normalize_prompt(enhanced))

    summary = "Preserved user prompt and enriched it with descriptive context"
    return enhanced, summary
