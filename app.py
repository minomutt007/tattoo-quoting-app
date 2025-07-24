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

# --- CONFIG ---
APP_VERSION = "2.3.1"
IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)
LOGO_PATH = os.path.join(IMAGE_DIR, "sally_mustang_logo.jpg")

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

@st.cache_data(ttl=300) # Cache settings for 5 minutes
def load_settings():
    """Fetches settings from the 'app_settings' table in Supabase."""
    response = supabase.table("app_settings").select("setting_name, setting_value").execute()
    
    if not response.data:
        default_settings = {
            "artists": ["Tally", "Alex", "Jay"],
            "styles": ["Line", "Color", "Realism"],
            "model_variant": "openai/clip-vit-base-patch32"
        }
        records_to_insert = [{"setting_name": k, "setting_value": v} for k, v in default_settings.items()]
        supabase.table("app_settings").insert(records_to_insert).execute()
        return default_settings

    return {item['setting_name']: item['setting_value'] for item in response.data}

settings = load_settings()

@st.cache_resource
def load_clip_model():
    """Loads the CLIP model and processor. Cached for performance."""
    model = CLIPModel.from_pretrained(settings["model_variant"])
    processor = CLIPProcessor.from_pretrained(settings["model_variant"])
    return model, processor

with st.spinner("Loading AI model..."):
    model, processor = load_clip_model()

@st.cache_data(ttl=600) # Cache tattoo data for 10 minutes
def load_data_from_supabase():
    """Fetches tattoo data from Supabase, including pre-calculated embeddings."""
    response = supabase.table("tattoos").select("artist, style, price, time_hours, image_url, embedding").execute()
    df = pd.DataFrame(response.data)
    
    if not df.empty and 'embedding' in df.columns and pd.notna(df['embedding']).any():
        df['embedding'] = df['embedding'].apply(lambda x: np.array(json.loads(x)) if pd.notna(x) else None)
    else:
        df['embedding'] = None

    return df

# --- HELPER FUNCTIONS ---

def save_settings(key, value):
    """Saves a specific setting to the Supabase table."""
    supabase.table("app_settings").update({"setting_value": value}).eq("setting_name", key).execute()
    st.cache_data.clear()

@st.cache_data(ttl=3600) # Cache exchange rates for 1 hour
def get_live_rates(base="ZAR"):
    try:
        response = requests.get(f"https://api.exchangerate.host/latest?base={base}")
        response.raise_for_status()
        return response.json().get("rates", {})
    except requests.RequestException:
        return {"USD": 0.055, "EUR": 0.051}

def convert_price(price_zar, currency, rates):
    return price_zar * rates.get(currency, 1.0)

def generate_pdf_report(uploaded_image_path, top_matches, price_range, currency, converted_range):
    """Generates a PDF report for the tattoo quote."""
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
        pdf.cell(0, 8, f"Converted Price ({currency}): {converted_range[0]:.2f} - {converted_range[1]:.2f}", ln=True)
    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Your Reference Image:", ln=True)
    if os.path.exists(uploaded_image_path):
        pdf.image(uploaded_image_path, w=80)
    pdf.ln(5)

    pdf.cell(0, 10, "Our Closest Matches:", ln=True)
    for _, row in top_matches.iterrows():
         pdf.multi_cell(0, 8, f"- Artist: {row['artist']}, Style: {row['style']}\n  Price: R{row['price']}, Time: {row.get('time_hours', 'N/A')} hrs")
    
    output_path = os.path.join(IMAGE_DIR, "quote_report.pdf")
    pdf.output(output_path)
    return output_path

# --- APP PAGES ---

def page_quote_tattoo():
    st.header("Quote Your Tattoo")
    tattoo_data = load_data_from_supabase()

    if tattoo_data[tattoo_data['embedding'].notna()].empty:
        st.info("No reference tattoos found in the database. Please upload tattoos and run the embedding generation script.")
        return

    uploaded_img = st.file_uploader("Upload a clear image of the tattoo or reference design", type=["jpg", "jpeg", "png"])
    
    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        
        if CROP_AVAILABLE:
            st.write("Crop the image to focus on the main design:")
            image = st_cropper(image, realtime_update=True, box_color="#0015FF")
        
        st.image(image, caption="Your Tattoo Reference", use_container_width=True)

        inputs = processor(images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            query_embedding = model.get_image_features(**inputs).squeeze(0)

        db = tattoo_data[tattoo_data['embedding'].notna()].copy()
        db_embeddings = torch.tensor(np.stack(db['embedding'].values))
        similarities = torch.nn.functional.cosine_similarity(query_embedding, db_embeddings)
        db['similarity'] = similarities.tolist()

        st.markdown("---")
        col1, col2 = st.columns(2)
        compare_count = col1.slider("Number of tattoos to compare", 1, 10, 3)
        currency = col2.selectbox("Display Currency", ["ZAR", "USD", "EUR"])

        top_matches = db.sort_values("similarity", ascending=False).head(compare_count)
        
        min_price, max_price = top_matches["price"].min(), top_matches["price"].max()
        rates = get_live_rates("ZAR")
        converted_range = (convert_price(min_price, currency, rates), convert_price(max_price, currency, rates))

        st.subheader("Price Estimate & Top Matches")
        st.metric("Estimated Price Range (ZAR)", f"R{min_price} - R{max_price}")
        if currency != "ZAR":
            st.metric(f"Converted Range ({currency})", f"{converted_range[0]:.2f} - {converted_range[1]:.2f}")

        temp_path = os.path.join(IMAGE_DIR, "temp_uploaded.png")
        image.save(temp_path)
        pdf_path = generate_pdf_report(temp_path, top_matches, (min_price, max_price), currency, converted_range)
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download Quote Report (PDF)", f, file_name="Tattoo_Quote.pdf")

        for _, match in top_matches.iterrows():
            st.markdown("---")
            st.write(f"**Similarity Score:** {match['similarity']:.2f}")
            # --- THIS IS THE LINE THAT WAS CHANGED ---
            st.image(match["image_url"], caption=f"Artist: {match['artist']}, Style: {match['style']}, Time: {match['time_hours']} hrs", use_container_width=True)

def page_supabase_upload():
    st.header("📸 Upload New Tattoo")
    
    uploaded_file = st.file_uploader("1. Upload Finished Tattoo Image", type=["jpg", "jpeg", "png"])
    
    cropped_img = None
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        if CROP_AVAILABLE:
            st.write("2. Crop the image (optional)")
            cropped_img = st_cropper(img, realtime_update=True, box_color="#0015FF", aspect_ratio=None)
            st.image(cropped_img, caption="Cropped Image Preview", use_container_width=True)
        else:
            st.image(img, caption="Image Preview", use_container_width=True)

    with st.form("upload_form"):
        st.write("3. Enter Tattoo Details")
        artist = st.selectbox("Artist", settings.get("artists", []))
        style  = st.selectbox("Style", settings.get("styles", []))
        price  = st.number_input("Final Price (R)", min_value=0, step=100)
        time_hours = st.number_input("Time Taken (in hours)", min_value=0.5, step=0.5)
        
        submitted = st.form_submit_button("Save to Database")

        if submitted:
            image_to_upload = cropped_img if cropped_img else (Image.open(uploaded_file) if uploaded_file else None)

            if not all([artist, style, price > 0, time_hours > 0, image_to_upload]):
                st.error("Please upload an image and fill out all fields.")
                return
            
            with st.spinner("Uploading and saving..."):
                buffer = BytesIO()
                image_to_upload.save(buffer, format="PNG")
                file_content = buffer.getvalue()

                original_filename = uploaded_file.name if uploaded_file else "cropped_image.png"
                file_name = f"{artist.replace(' ', '_')}_{date.today()}_{original_filename}"
                
                supabase.storage.from_("tattoo-images").upload(file_name, file_content, file_options={"upsert": "true"})
                image_url = supabase.storage.from_("tattoo-images").get_public_url(file_name)

                supabase.table("tattoos").insert({
                    "artist": artist, "style": style, "price": price,
                    "time_hours": time_hours, "image_url": image_url
                }).execute()
            
                st.success(f"Successfully saved tattoo by {artist}!")
                st.image(image_to_upload, caption="Uploaded Tattoo")

def page_settings():
    st.header("⚙️ App Settings (Stored in Supabase)")

    st.subheader("Manage Artists")
    artists = settings.get("artists", [])
    new_artist = st.text_input("Add New Artist", key="new_artist")
    if st.button("Add Artist"):
        if new_artist and new_artist not in artists:
            artists.append(new_artist)
            save_settings("artists", artists)
            st.success(f"Added '{new_artist}'")
            st.rerun()

    artist_to_remove = st.selectbox("Remove Artist", ["-"] + artists, key="remove_artist")
    if st.button("Remove Artist"):
        if artist_to_remove != "-":
            artists.remove(artist_to_remove)
            save_settings("artists", artists)
            st.success(f"Removed '{artist_to_remove}'")
            st.rerun()
    st.markdown("---")

    st.subheader("Manage Styles")
    styles = settings.get("styles", [])
    new_style = st.text_input("Add New Style", key="new_style")
    if st.button("Add Style"):
        if new_style and new_style not in styles:
            styles.append(new_style)
            save_settings("styles", styles)
            st.success(f"Added '{new_style}'")
            st.rerun()

    style_to_remove = st.selectbox("Remove Style", ["-"] + styles, key="remove_style")
    if st.button("Remove Style"):
        if style_to_remove != "-":
            styles.remove(style_to_remove)
            save_settings("styles", styles)
            st.success(f"Removed '{style_to_remove}'")
            st.rerun()

# --- MAIN APP NAVIGATION ---
def main():
    if os.path.exists(LOGO_PATH):
        st.sidebar.image(LOGO_PATH, width=100)
    st.sidebar.title("Navigation")
    
    pages = {
        "Quote Tattoo": page_quote_tattoo,
        "Upload Tattoo": page_supabase_upload,
        "Settings": page_settings,
    }
    
    choice = st.sidebar.radio("Go to", list(pages.keys()))
    pages[choice]()

    st.sidebar.markdown("---")
    st.sidebar.info(f"App Version: {APP_VERSION}")

if __name__ == "__main__":
    main()
