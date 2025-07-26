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
from auth import show_login_form, logout_user

# --- CONFIG ---
APP_VERSION = "4.2.3 (Stable Quoting)"
IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)
LOGO_PATH = os.path.join(IMAGE_DIR, "sally_mustang_logo.jpg")
TATTOO_COLOR_TYPES = ["Black & Grey", "Full Color"]

# --- SECURELY INITIALIZE SUPABASE ---
try:
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    supabase = create_client(supabase_url, supabase_key)
except (KeyError, AttributeError):
    st.error("Supabase credentials are not set in st.secrets.")
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
    response = supabase.table("app_settings").select("setting_name, setting_value").execute()
    settings_data = {item['setting_name']: item['setting_value'] for item in response.data}
    defaults = {
        "artists": ["Tally", "Alex", "Jay"], "styles": ["Line", "Color", "Realism"],
        "placements": ["Arm", "Leg", "Torso", "Back"], "model_variant": "openai/clip-vit-base-patch32"
    }
    for key, value in defaults.items():
        if key not in settings_data:
            supabase.table("app_settings").insert({"setting_name": key, "setting_value": value}).execute()
            settings_data[key] = value
    return settings_data

settings = load_settings()

@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained(settings["model_variant"])
    processor = CLIPProcessor.from_pretrained(settings["model_variant"])
    return model, processor

with st.spinner("Loading AI model..."):
    model, processor = load_clip_model()

@st.cache_data(ttl=600)
def load_data_from_supabase():
    response = supabase.table("tattoos").select("artist, style, price, time_hours, image_url, embedding, size_cm, placement, color_type").execute()
    df = pd.DataFrame(response.data)
    if not df.empty and 'embedding' in df.columns and pd.notna(df['embedding']).any():
        df['embedding'] = df['embedding'].apply(lambda x: np.array(json.loads(x)) if pd.notna(x) else None)
    else:
        df['embedding'] = None
    return df

# --- HELPER FUNCTIONS ---
def save_settings(key, value):
    supabase.table("app_settings").update({"setting_value": value}).eq("setting_name", key).execute()
    st.cache_data.clear()

@st.cache_data(ttl=3600)
def get_live_rates(base="ZAR"):
    try:
        response = requests.get(f"https://api.exchangerate.host/latest?base={base}")
        response.raise_for_status()
        return response.json().get("rates", {})
    except requests.RequestException:
        return {"USD": 0.055, "EUR": 0.051}

def generate_pdf_report(uploaded_image_path, top_matches, price_range, currency):
    pdf = FPDF()
    pdf.add_page()
    if os.path.exists(LOGO_PATH):
        pdf.image(LOGO_PATH, x=10, y=8, w=30)
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 15, "Tattoo Quote Report", ln=True, align="C")
    pdf.ln(15)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Date: {date.today()}", ln=True)
    pdf.cell(0, 8, f"Estimated Price Range (ZAR): R{price_range[0]} - R{price_range[1]}", ln=True)
    if currency != "ZAR":
        rates = get_live_rates("ZAR")
        converted_range = (price_range[0] * rates.get(currency, 1), price_range[1] * rates.get(currency, 1))
        pdf.cell(0, 8, f"Converted Price ({currency}): {converted_range[0]:.2f} - {converted_range[1]:.2f}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Your Reference Image:", ln=True)
    if os.path.exists(uploaded_image_path):
        pdf.image(uploaded_image_path, w=80)
    pdf.ln(5)
    pdf.cell(0, 10, "Our Closest Matches:", ln=True)
    for _, row in top_matches.iterrows():
         pdf.multi_cell(0, 8, f"- Artist: {row['artist']}, Style: {row['style']}, Size: {row.get('size_cm', 'N/A')} cm\n  Placement: {row.get('placement', 'N/A')}, Color: {row.get('color_type', 'N/A')}\n  Price: R{row['price']}, Time: {row.get('time_hours', 'N/A')} hrs")
    output_path = os.path.join(IMAGE_DIR, "quote_report.pdf")
    pdf.output(output_path)
    return output_path

# --- APP PAGES ---
def page_quote_tattoo():
    st.header("Quote Your Tattoo")
    
    def clear_uploader():
        if "quote_uploader" in st.session_state:
            st.session_state.quote_uploader = None

    # --- FIX: MOVED BUTTON TO THE TOP ---
    col_a, col_b = st.columns([3, 1])
    with col_a:
        customer_name = st.text_input("Customer Name (Optional)")
    with col_b:
        st.write("") 
        st.write("") 
        st.button("✨ Start New Quote", on_click=clear_uploader, use_container_width=True)
    
    uploaded_img = st.file_uploader(
        "Upload a clear image of the tattoo or reference design", 
        type=["jpg", "jpeg", "png"],
        key="quote_uploader"
    )
    
    if uploaded_img:
        st.markdown("---")
        st.subheader("Filtering Options")
        size_range = st.slider("Filter by Size (cm)", min_value=1.0, max_value=40.0, value=(5.0, 15.0))
        col1, col2 = st.columns(2)
        with col1:
            artist_filter = st.selectbox("Artist", ["All"] + settings.get("artists", []))
            placement_filter = st.selectbox("Body Placement", ["All"] + settings.get("placements", []))
        with col2:
            color_filter = st.selectbox("Color Type", ["All"] + TATTOO_COLOR_TYPES)
            compare_count = st.slider("Number of Tattoos to Compare", 1, 10, 3, key="compare_slider")
        
        tattoo_data = load_data_from_supabase()
        if not tattoo_data.empty and 'size_cm' in tattoo_data.columns and tattoo_data['size_cm'].notna().any():
            tattoo_data = tattoo_data[(tattoo_data['size_cm'] >= size_range[0]) & (tattoo_data['size_cm'] <= size_range[1])]
        if artist_filter != "All": tattoo_data = tattoo_data[tattoo_data['artist'] == artist_filter]
        if placement_filter != "All": tattoo_data = tattoo_data[tattoo_data['placement'] == placement_filter]
        if color_filter != "All": tattoo_data = tattoo_data[tattoo_data['color_type'] == color_filter]
        
        if tattoo_data.empty or tattoo_data[tattoo_data['embedding'].notna()].empty:
            st.warning("No reference tattoos found for the selected filters.")
            return
        
        image = Image.open(uploaded_img).convert("RGB")
        if CROP_AVAILABLE: image = st_cropper(
