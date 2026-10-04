"""Professional Text-to-Image Generator"""
import io
import os
import streamlit as st
from generator import generate_image
from prompt_engine import build_final_prompt
from utils import clear_memory

st.set_page_config(
    page_title="SmartPrompt AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root { color-scheme: dark; }
        body, .stApp { background: linear-gradient(135deg, #0a0e27 0%, #12192f 100%); }
        .block-container { padding: 2rem; max-width: 1400px; margin: 0 auto; }
        .stButton>button {
            border-radius: 12px; padding: 0.9rem 2rem; font-weight: 700; font-size: 1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important; border: none !important; transition: all 0.3s ease;
        }
        .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important; }
        .stTextArea>div>div>textarea {
            background: rgba(255,255,255,0.05) !important; border: 1.5px solid rgba(102, 126, 234, 0.3) !important;
            color: #eef2ff !important; font-size: 1.05rem; padding: 1.2rem !important; border-radius: 12px;
        }
        .header-title { font-size: 3.5rem; font-weight: 800; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0.5rem 0; }
        .header-subtitle { font-size: 1.25rem; color: #b0b8d4; margin: 1rem 0 2rem; font-weight: 400; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="header-title">SmartPrompt AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="header-subtitle">Describe anything. AI brings it to life.</p>', unsafe_allow_html=True)

col_prompt, col_controls = st.columns([4, 1])
with col_prompt:
    user_input = st.text_area(
        "Describe your vision:",
        height=120,
        placeholder="Example: A lonely traveler walking through an endless desert beneath a sky filled with galaxies",
        label_visibility="collapsed"
    )
with col_controls:
    st.write(" ")
    st.write(" ")
    st.write(" ")
    generate_btn = st.button("Generate", use_container_width=True, key="generate_btn")

control_cols = st.columns([1, 1, 10])
with control_cols[0]:
    if st.button("Clear Session", use_container_width=True):
        st.session_state.clear()
        st.rerun()
with control_cols[1]:
    if st.button("Exit App", use_container_width=True):
        clear_memory()
        st.warning("Session terminated...")
        os._exit(0)

if generate_btn:
    if not user_input.strip():
        st.error("Please describe what you want to see.")
    else:
        st.divider()
        prompt_data = build_final_prompt(user_input)
        with st.spinner("Generating your image..."):
            image = generate_image(prompt_data["positive"], prompt_data["negative"])
        if image:
            st.session_state["latest_image"] = image
            st.session_state["latest_prompt"] = prompt_data
            st.session_state["user_description"] = user_input
        else:
            st.error("Generation failed. Try refining your description.")

st.divider()

if "latest_image" in st.session_state:
    res_col1, res_col2 = st.columns([2, 1])
    with res_col1:
        st.markdown("### Your Creation")
        st.image(st.session_state["latest_image"], use_column_width=True)
        buffer = __import__("io").BytesIO()
        st.session_state["latest_image"].save(buffer, format="PNG")
        st.download_button(
            label="Download Image",
            data=buffer.getvalue(),
            file_name="smartprompt_creation.png",
            mime="image/png",
            use_container_width=True
        )
    with res_col2:
        st.markdown("### Details")
        st.markdown(f"**Your description:**\n{st.session_state['user_description']}")
        with st.expander("Generated Prompt", expanded=False):
            st.code(st.session_state["latest_prompt"]["positive"], language="text")
        with st.expander("Quality Filter", expanded=False):
            st.caption("Negative prompt (removes artifacts):")
            st.code(st.session_state["latest_prompt"]["negative"], language="text")
else:
    st.info("Enter your description and click Generate")
