import streamlit as st
import pandas as pd
import torch
import os
import json
import requests
import numpy as np
from io import BytesIO
from datetime import date
from PIL import Image
from fpdf import FPDF
from supabase import create_client
from transformers import CLIPModel, CLIPProcessor
from auth import show_login_form, logout_user # <-- NEW IMPORT

# --- CONFIG ---
APP_VERSION = "2.0.0 (Secure)"
IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)
LOGO_PATH = os.path.join(IMAGE_DIR, "sally_mustang_logo.jpg")

# Define tattoo attributes for dropdowns
TATTOO_COLOR_TYPES = ["Black & Grey", "Full Color"]

# --- SECURELY INITIALIZE SUPABASE ---
try:
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    supabase = create_client(supabase_url, supabase_key)
except (KeyError, AttributeError):
    st.error("Supabase credentials are not set in st.secrets. Please add them.")
    st.stop()

# --- OPTIONAL LIBRARIES ---
try:
    from streamlit_cropper import st_cropper
    CROP_AVAILABLE = True
except ImportError:
    CROP_AVAILABLE = False

# --- DATA & MODEL LOADING ---
@st.cache_data(ttl=300)
def load_settings():
    """Fetches settings from the 'app_settings' table in Supabase."""
    response = supabase.table("app_settings").select("setting_name, setting_value").execute()
    settings_data = {item['setting_name']: item['setting_value'] for item in response.data}
    defaults = {
        "artists": ["Tally", "Alex", "Jay"],
        "styles": ["Line", "Color", "Realism"],
        "placements": ["Arm", "Leg", "Torso", "Back"],
        "model_variant": "openai/clip-vit-base-patch32"
    }
    for key, value in defaults.items():
        if key not in settings_data:
            supabase.table("app_settings").insert({"setting_name": key, "setting_value": value}).execute()
            settings_data[key] = value
    return settings_data

settings = load_settings()

@st.cache_resource
def load_clip_model():
    """Loads the CLIP model and processor. Cached for performance."""
    model = CLIPModel.from_pretrained(settings["model_variant"])
    processor = CLIPProcessor.from_pretrained(settings["model_variant"])
    return model, processor

with st.spinner("Loading AI model..."):
    model, processor = load_clip_model()

@st.cache_data(ttl=600)
def load_data_from_supabase():
    """Fetches tattoo data from Supabase, including pre-calculated embeddings."""
    response = supabase.table("tattoos").select("artist, style, price, time_hours, image_url, embedding, size_cm, placement, color_type").execute()
    df = pd.DataFrame(response.data)
    if not df.empty and 'embedding' in df.columns and pd.notna(df['embedding']).any():
        df['embedding'] = df['embedding'].apply(lambda x: np.array(json.loads(x)) if pd.notna(x) else None)
    else:
        df['embedding'] = None
    return df

# --- HELPER FUNCTIONS ---
def save_settings(key, value):
    # (Implementation is the same)
def generate_pdf_report(uploaded_image_path, top_matches, price_range, currency):
    # (Implementation is the same)
@st.cache_data(ttl=3600)
def get_live_rates(base="ZAR"):
    # (Implementation is the same)

# --- APP PAGES (Implementations are the same, just defined here) ---
def page_quote_tattoo():
    # (Full implementation is the same as your file)
def page_quote_history():
    # (Full implementation is the same as your file)
def page_supabase_upload():
    # (Full implementation is the same as your file)
def page_batch_upload():
    # (Full implementation is the same as your file)
def page_settings():
    # (Full implementation is the same as your file)


# --- MAIN APP LOGIC ---
def main_app():
    """This function runs the main application after the user has logged in."""
    if os.path.exists(LOGO_PATH):
        st.sidebar.image(LOGO_PATH, width=100)
    st.sidebar.title("Navigation")
    
    st.sidebar.write(f"Logged in as: {st.session_state.get('user_email', '')}")
    st.sidebar.button("Logout", on_click=logout_user)
    st.sidebar.markdown("---")
    
    pages = {
        "Quote Tattoo": page_quote_tattoo,
        "Quote History": page_quote_history,
        "Upload Single Tattoo": page_supabase_upload,
        "Batch Upload": page_batch_upload,
        "Settings": page_settings,
    }
    
    choice = st.sidebar.radio("Go to", list(pages.keys()), key="page_choice")
    pages[choice]()

    st.sidebar.markdown("---")
    st.sidebar.info(f"App Version: {APP_VERSION}")

# --- APP ENTRY POINT ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if st.session_state['authenticated']:
    main_app()
else:
    show_login_form()
