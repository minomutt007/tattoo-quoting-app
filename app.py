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
APP_VERSION = "4.3.0 (Checkbox Reset)"
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
    
    customer_name = st.text_input("Customer Name (Optional)")
    
    # --- NEW CHECKBOX METHOD TO PREVENT ERRORS ---
    if st.checkbox("Button needs to be slected to Quote and reselected for each new quote"):
        uploaded_img = st.file_uploader(
            "Upload a clear image of the tattoo or reference design", 
            type=["jpg", "jpeg", "png"]
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
            if CROP_AVAILABLE: image = st_cropper(image, realtime_update=True, box_color="#0015FF")
            st.image(image, caption="Your Tattoo Reference", use_container_width=True)

            inputs = processor(images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                query_embedding = model.get_image_features(**inputs).squeeze(0)
            db = tattoo_data[tattoo_data['embedding'].notna()].copy()
            db_embeddings = torch.tensor(np.stack(db['embedding'].values))
            similarities = torch.nn.functional.cosine_similarity(query_embedding, db_embeddings)
            db['similarity'] = similarities.tolist()
            top_matches = db.sort_values("similarity", ascending=False).head(compare_count)
            
            if top_matches.empty:
                st.warning("Could not find any matching tattoos with the selected criteria.")
                return

            min_price, max_price = top_matches["price"].min(), top_matches["price"].max()
            st.markdown("---")
            st.subheader("Final Quote")
            colA, colB = st.columns(2)
            final_min_price = colA.number_input("Final Minimum Price (R)", value=int(min_price))
            final_max_price = colB.number_input("Final Maximum Price (R)", value=int(max_price))
            final_price_range_str = f"R{final_min_price} - R{final_max_price}"

            if st.button("💾 Save Quote"):
                with st.spinner("Saving quote..."):
                    buffer = BytesIO()
                    image.save(buffer, format="PNG")
                    ref_image_content = buffer.getvalue()
                    ref_image_name = f"{customer_name.replace(' ', '_') or 'quote'}_{date.today()}_{uploaded_img.name}"
                    supabase.storage.from_("quote-references").upload(ref_image_name, ref_image_content, file_options={"upsert": "true"})
                    ref_image_url = supabase.storage.from_("quote-references").get_public_url(ref_image_name)
                    supabase.table("quotes").insert({"customer_name": customer_name if customer_name else "N/A", "reference_image_url": ref_image_url, "final_price_range": final_price_range_str}).execute()
                    st.success(f"Quote for {customer_name or 'N/A'} saved successfully!")
                
            with st.expander("Show AI Top Matches"):
                for _, match in top_matches.iterrows():
                    st.markdown("---")
                    caption_text = (f"Artist: {match['artist']}, Style: {match['style']}, Size: {match.get('size_cm', 'N/A')} cm, Placement: {match.get('placement', 'N/A')}, Color: {match.get('color_type', 'N/A')}, Time: {match['time_hours']} hrs")
                    st.image(match["image_url"], caption=caption_text, use_container_width=True)

def page_quote_history():
    st.header("📜 Quote History")
    response = supabase.table("quotes").select("*").order("quote_date", desc=True).execute()
    quotes_df = pd.DataFrame(response.data)
    search_term = st.text_input("Search by Customer Name")
    if search_term:
        quotes_df = quotes_df[quotes_df['customer_name'].str.contains(search_term, case=False, na=False)]
    if quotes_df.empty:
        st.info("No saved quotes found.")
        return
    for _, row in quotes_df.iterrows():
        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(row['reference_image_url'])
        with col2:
            st.write(f"**Customer:** {row['customer_name']}")
            st.write(f"**Date:** {pd.to_datetime(row['quote_date']).strftime('%Y-%m-%d')}")
            st.write(f"**Quoted Range:** {row['final_price_range']}")

def page_supabase_upload():
    st.header("📸 Upload Single Tattoo")
    st.info("For uploading multiple tattoos at once, please use the 'Batch Upload' page.")
    uploaded_file = st.file_uploader("1. Upload Finished Tattoo Image", type=["jpg", "jpeg", "png"], key="single_uploader")
    cropped_img = None
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        if CROP_AVAILABLE:
            st.write("2. Crop the image (optional)")
            cropped_img = st_cropper(img, realtime_update=True, box_color="#0015FF")
            st.image(cropped_img, caption="Cropped Image Preview", use_container_width=True)
        else:
            st.image(img, caption="Image Preview", use_container_width=True)
    with st.form("upload_form"):
        st.write("3. Enter Tattoo Details")
        artist = st.selectbox("Artist", settings.get("artists", []))
        style  = st.selectbox("Style", settings.get("styles", []))
        size_cm   = st.number_input("Approximate Size (cm)", min_value=1.0, step=0.5)
        placement = st.selectbox("Body Placement", settings.get("placements", []))
        color_type = st.selectbox("Color Type", TATTOO_COLOR_TYPES)
        price  = st.number_input("Final Price (R)", min_value=0, step=100)
        time_hours = st.number_input("Time Taken (in hours)", min_value=0.5, step=0.5)
        submitted = st.form_submit_button("Save to Database")
        if submitted:
            image_to_upload = cropped_img if cropped_img else (Image.open(uploaded_file) if uploaded_file else None)
            if not all([artist, style, size_cm > 0, placement, color_type, price > 0, time_hours > 0, image_to_upload]):
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
                supabase.table("tattoos").insert({"artist": artist, "style": style, "price": price, "time_hours": time_hours, "image_url": image_url, "size_cm": size_cm, "placement": placement, "color_type": color_type}).execute()
                st.success(f"Successfully saved tattoo by {artist}!")
                st.image(image_to_upload, caption="Uploaded Tattoo")

def page_batch_upload():
    st.header("📦 Batch Upload Tattoos")
    uploaded_files = st.file_uploader("1. Upload all tattoo images for this session", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch_uploader")
    if uploaded_files:
        if 'batch_files' not in st.session_state or st.session_state.batch_files != uploaded_files:
            st.session_state.batch_files = uploaded_files
            st.session_state.details = [{} for _ in uploaded_files]
            st.session_state.current_index = 0
            st.session_state.cropped_images = [None] * len(uploaded_files)
        total_files = len(st.session_state.batch_files)
        current_index = st.session_state.current_index
        if current_index < total_files:
            st.write(f"--- \n### 2. Crop and Enter Details for Image {current_index + 1} of {total_files}")
            current_file = st.session_state.batch_files[current_index]
            img = Image.open(current_file).convert("RGB")
            col1, col2 = st.columns(2)
            with col1:
                if CROP_AVAILABLE:
                    cropped_img = st_cropper(img, realtime_update=True, box_color="#0015FF", key=f"crop_{current_index}")
                    st.image(cropped_img, caption=f"Cropped Preview: {current_file.name}", use_container_width=True)
                else:
                    cropped_img = img
                    st.image(img, caption=f"Image: {current_file.name}", use_container_width=True)
            with col2:
                with st.form(f"detail_form_{current_index}", clear_on_submit=True):
                    details = {}
                    details['artist'] = st.selectbox("Artist", settings.get("artists", []), key=f"art_{current_index}")
                    details['style']  = st.selectbox("Style", settings.get("styles", []), key=f"sty_{current_index}")
                    details['size_cm'] = st.number_input("Size (cm)", min_value=1.0, step=0.5, key=f"siz_{current_index}")
                    details['placement'] = st.selectbox("Placement", settings.get("placements", []), key=f"pla_{current_index}")
                    details['color_type'] = st.selectbox("Color", TATTOO_COLOR_TYPES, key=f"col_{current_index}")
                    details['price'] = st.number_input("Price (R)", min_value=0, step=100, key=f"pri_{current_index}")
                    details['time_hours'] = st.number_input("Time (hrs)", min_value=0.5, step=0.5, key=f"tim_{current_index}")
                    if st.form_submit_button("✅ Save and Next"):
                        st.session_state.details[current_index] = details
                        st.session_state.cropped_images[current_index] = cropped_img
                        st.session_state.current_index += 1
                        st.rerun()
        else:
            st.success("All details entered! Ready to upload to Supabase.")
            if st.button("🚀 Save All to Supabase"):
                progress_bar = st.progress(0, text="Uploading...")
                for i in range(total_files):
                    original_file = st.session_state.batch_files[i]
                    detail = st.session_state.details[i]
                    cropped_image = st.session_state.cropped_images[i]
                    with st.spinner(f"Uploading {original_file.name}..."):
                        buffer = BytesIO()
                        cropped_image.save(buffer, format="PNG")
                        file_content = buffer.getvalue()
                        file_name = f"{detail['artist'].replace(' ', '_')}_{date.today()}_{original_file.name}"
                        supabase.storage.from_("tattoo-images").upload(file_name, file_content, file_options={"upsert": "true"})
                        image_url = supabase.storage.from_("tattoo-images").get_public_url(file_name)
                        detail['image_url'] = image_url
                        supabase.table("tattoos").insert(detail).execute()
                    progress_bar.progress((i + 1) / total_files, text=f"Uploaded {original_file.name}")
                st.success("Batch upload complete! Remember to run the embedding script.")
                st.session_state.clear()

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
    st.markdown("---")
    st.subheader("Manage Body Placements")
    placements = settings.get("placements", [])
    new_placement = st.text_input("Add New Body Placement", key="new_placement")
    if st.button("Add Placement"):
        if new_placement and new_placement not in placements:
            placements.append(new_placement)
            save_settings("placements", placements)
            st.success(f"Added '{new_placement}'")
            st.rerun()
    placement_to_remove = st.selectbox("Remove Body Placement", ["-"] + placements, key="remove_placement")
    if st.button("Remove Placement"):
        if placement_to_remove != "-":
            placements.remove(placement_to_remove)
            save_settings("placements", placements)
            st.success(f"Removed '{placement_to_remove}'")
            st.rerun()

# --- MAIN APP LOGIC ---
def main_app():
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
