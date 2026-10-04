# 🎨 SmartPrompt AI

SmartPrompt AI is an advanced, production-grade text-to-image generation system powered by Stable Diffusion and custom Rule-Based Natural Language Processing. It bridges the gap between simple human imagination and the complex prompt engineering required by latent diffusion models.

## 🚀 Project Vision
"Any user should be able to express imagination, emotions, fantasy, artistic style, cinematic feeling, atmosphere, or abstract thoughts in natural language, and the system should generate an image that closely matches the intent and emotional meaning of the text."

This is not just a wrapper around Stable Diffusion. It is an **intelligent creative companion** featuring a semantic enhancement layer, an emotion detection engine, and dynamic style/fantasy controls.

---

## ✨ Core Features

* **🧠 Emotion Understanding Engine:** Analyzes user input to infer emotions (sadness, joy, fear, mystery) and automatically applies appropriate cinematic lighting and atmospheric modifiers.
* **⚡ AI Prompt Enhancer:** Transforms basic prompts (e.g., "a dog") into rich, highly detailed descriptions without the overhead of running a secondary LLM, preserving precious GPU VRAM.
* **🪄 Fantasy Control System:** Dynamically scales reality from 0 (hyper-realistic) to 10 (surreal, mind-bending cosmic scales).
* **🎨 Artistic Control:** Instantly apply complex styles like Cyberpunk, Gothic, Watercolor, or Anime with perfectly tuned positive and negative prompts.
* **🚀 Faster, high-quality rendering:** Uses `Lykon/dreamshaper-8` with a DPMSolver scheduler and optimized attention/vae slicing for quicker generation without compromising image fidelity.
* **🎛️ Streamlit UI with Smart Prompt Controls:** Elegant frontend with style, composition, mood, lighting, and fantasy level controls, plus prompt preview and download support.
* **📚 Future-Ready Architecture:** Clean, modular codebase ready for SDXL upgrades, ControlNet integration, or RLHF.
* **🏋️ LoRA Fine-Tuning Script:** Includes a mathematically correct LoRA training script (fixing VAE latent space encoding bugs found in many tutorials) ready for custom dataset fine-tuning.

---

## 🏗️ Architecture

```text
project/
│
├── src/
│   ├── app.py               # Streamlit GUI & Application Logic
│   ├── generator.py         # SD Pipeline & Memory Management
│   ├── prompt_engine.py     # Master Prompt Compiler
│   ├── ai_enhancer.py       # Semantic Enrichment Layer
│   ├── emotion_engine.py    # Rule-based Emotion Detection
│   ├── config.py            # System Modifiers & Constants
│   ├── utils.py             # Memory Utilities
│
├── training/
│   ├── train_lora.py        # Corrected PEFT LoRA training loop
│   ├── prepare_dataset.py   # Dataset formatting utilities
│
├── models/                  # Output directory for LoRA weights
├── data/                    # Dataset directory
├── outputs/                 # Generated images
├── requirements.txt         # Project dependencies
└── README.md
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- Nvidia GPU (Minimum 4GB VRAM)
- CUDA Toolkit installed

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/yourusername/text_to_image_ai.git
cd text_to_image_ai

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Hugging Face Token
If you are using a private model or the model requires authentication, set your token before running the app.

Windows PowerShell:
```powershell
$env:HUGGINGFACE_TOKEN = "YOUR_TOKEN_HERE"
streamlit run src/app.py
```

Linux/Mac:
```bash
export HUGGINGFACE_TOKEN="YOUR_TOKEN_HERE"
streamlit run src/app.py
```

Alternatively, you can login with the Hugging Face CLI:
```bash
python -m huggingface_hub login
```

---

## 🎮 Usage

Start the SmartPrompt AI web interface:

```bash
streamlit run src/app.py
```

1. Enter your concept in the **Imagination** box (e.g., "A lone knight standing before a glowing portal").
2. Select an **Artistic Style**, **Composition**, **Mood**, and **Lighting** from the sidebar.
3. Adjust the **Fantasy Level** slider to set realism vs. surrealism.
4. Click **Generate Image** and watch the AI enhance your prompt and render the masterpiece.
5. Expand the **Prompt details** panel to inspect the final positive and negative prompts, then download the image directly.

---

## 🧑‍💻 AI Engineering Highlights (Resume Worthy)

* **Memory-Conscious Intelligence:** Designed a zero-VRAM Rule-Based Expert System for NLP to perform text enhancement, dedicating 100% of the 4GB VRAM budget to the Stable Diffusion generation process.
* **Interactive Streamlit Frontend:** Built a polished UI with guided prompts, creative controls, and prompt preview for a more engaging user experience.
* **Latent Space Training Correction:** Identified and resolved a critical flaw in pixel-space noising during custom LoRA training by correctly implementing VAE latent encoding (`vae.encode(pixel_values).latent_dist.sample() * scaling_factor`).
* **Modular Pipeline Design:** Decoupled the UI, the Prompt Intelligence Engine, and the Image Generation Pipeline to allow for seamless future model swapping (e.g., upgrading to SDXL) without altering the prompt logic.

---
*Built with ❤️ using PyTorch, Diffusers, and Streamlit.*
