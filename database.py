import streamlit as st
import pandas as pd
import numpy as np
import json
from supabase import create_client

# Initialize Supabase client using secrets
try:
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    supabase = create_client(supabase_url, supabase_key)
except (KeyError, AttributeError):
    pass

@st.cache_data(ttl=300)
def load_settings():
    response = supabase.table("app_settings").select("setting_name, setting_value").execute()
    settings_data = {item['setting_name']: item['setting_value'] for item in response.data}
    defaults = {
        "artists": ["Tally", "Alex", "Jay"], "styles": ["Line", "Color", "Realism"],
        "placements": ["Arm", "Leg", "Torso", "Back"], "model_variant": "openai/clip-vit-base-patch32",
        "complicated_placements": {"Neck": 15, "Ribs": 20}
    }
    for key, value in defaults.items():
        if key not in settings_data:
            supabase.table("app_settings").insert({"setting_name": key, "setting_value": value}).execute()
            settings_data[key] = value
    return settings_data

@st.cache_data(ttl=600)
def load_data_from_supabase():
    response = supabase.table("tattoos").select("artist, style, price, time_hours, image_url, embedding, size_cm, placement, color_type, ai_caption").execute()
    df = pd.DataFrame(response.data)
    if not df.empty and 'embedding' in df.columns and pd.notna(df['embedding']).any():
        df['embedding'] = df['embedding'].apply(lambda x: np.array(json.loads(x)) if pd.notna(x) else None)
    else:
        df['embedding'] = None
    return df

def save_settings(key, value):
    supabase.table("app_settings").update({"setting_value": value}).eq("setting_name", key).execute()
    st.cache_data.clear()