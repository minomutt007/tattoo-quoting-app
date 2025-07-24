
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
APP_VERSION = "2.1.0" # Updated version
IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True) # Ensure image dir exists
LOGO_PATH = os.path.join(IMAGE_DIR, "sally_mustang_logo.jpg")

# --- SECURELY INITIALIZE SUPABASE ---
try:
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    supabase = create_client(supabase_url, supabase_key)
except KeyError:
    st.error("Supabase credentials are not set in st.secrets. Please add them.")
    st.stop()

from storage3.exceptions import StorageApiError

# --- CONFIG ---
APP_VERSION = "1.2.1"
CSV_PATH = "tattoos.csv"
SETTINGS_PATH = "settings.json"
IMAGE_DIR = "images"
LOGS_PATH = "match_logs.csv"
# Securely access the Supabase credentials from secrets.toml
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]
supabase = create_client(supabase_url, supabase_key)

LOGO_PATH = os.path.join(IMAGE_DIR, "sally_mustang_logo.jpg")

os.makedirs(IMAGE_DIR, exist_ok=True)

try:
    from streamlit_cropper import st_cropper
    CROP_AVAILABLE = True
except ImportError:
    CROP_AVAILABLE = False


# --- DATA & MODEL LOADING ---

@st.cache_data(ttl=300) # Cache settings for 5 minutes
def load_settings():
    # Fetches settings from the 'app_settings' table in Supabase.
    response = supabase.table("app_settings").select("setting_name, setting_value").execute()
    
    if not response.data:
        # If the table is empty, create default settings
        default_settings = {
            "artists": ["Tally", "Alex", "Jay"],
            "styles": ["Line", "Color", "Realism"],
            "model_variant": "openai/clip-vit-base-patch32"
        }
        # Insert them into the table
        records_to_insert = [{"setting_name": k, "setting_value": v} for k, v in default_settings.items()]
        supabase.table("app_settings").insert(records_to_insert).execute()
        return default_settings

    # Convert the list of records into a single settings dictionary
    settings_dict = {item['setting_name']: item['setting_value'] for item in response.data}
    return settings_dict
  
@st.cache_data
def load_settings():
    if os.path.exists(SETTINGS_PATH):
        return json.load(open(SETTINGS_PATH))
    settings = {
        "artists": ["Tally", "Alex", "Jay", "Lee", "Emilia"],
        "styles": ["Line", "Linework"],
        "archived_artists": [],
        "model_variant": "openai/clip-vit-base-patch32"
    }
    json.dump(settings, open(SETTINGS_PATH, "w"), indent=2)
    return settings

# This line should be right after the function definition
settings = load_settings()

@st.cache_resource
def load_clip_model():
    # Loads the CLIP model and processor. Cached for performance.
    model = CLIPModel.from_pretrained(settings["model_variant"])
    processor = CLIPProcessor.from_pretrained(settings["model_variant"])
    return model, processor

with st.spinner("Loading AI model..."):
    model, processor = load_clip_model()

@st.cache_data(ttl=600) # Cache Supabase data for 10 minutes
def load_data_from_supabase():
    # Fetches all tattoo data directly from Supabase, including pre-calculated embeddings.
    response = supabase.table("tattoos").select("artist, style, price, time_hours, image_url, embedding").execute()
    df = pd.DataFrame(response.data)
    
    # Convert string embedding back to numpy array
    if not df.empty and 'embedding' in df.columns and df['embedding'].iloc[0] is not None:
        df['embedding'] = df['embedding'].apply(lambda x: np.array(json.loads(x)))
    else:
        # Create an empty embedding column if none exist
        df['embedding'] = pd.Series([None] * len(df))

    return df

# --- HELPER FUNCTIONS ---

def save_settings(key, value):
    """Saves a specific setting to the Supabase table."""
    supabase.table("app_settings").update({"setting_value": value}).eq("setting_name", key).execute()
    st.cache_data.clear() # Clear cache to force a reload of settings

@st.cache_data(ttl=3600) # Cache exchange rates for 1 hour
def get_live_rates(base="ZAR"):
    try:
        response = requests.get(f"https://api.exchangerate.host/latest?base={base}")
        return response.json().get("rates", {})
    except requests.RequestException:
        return {"USD": 0.055, "EUR": 0.051} # Fallback rates

@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame(columns=["filename","artist","style","price","time"])

@st.cache_data
def load_logs():
    return pd.read_csv(LOGS_PATH, parse_dates=["date"]) if os.path.exists(LOGS_PATH) else pd.DataFrame(columns=["date","artist"])

def save_settings(s):
    json.dump(s, open(SETTINGS_PATH, "w"), indent=2)
    st.cache_data.clear()

def save_data(df):
    df.to_csv(CSV_PATH, index=False)
    st.cache_data.clear()

def save_logs(df):
    df.to_csv(LOGS_PATH, index=False)
    st.cache_data.clear()

data = load_data()
logs = load_logs()

@st.cache_data(ttl=3600)
def get_live_rates(base="ZAR"):
    try:
        response = requests.get(f"https://api.exchangerate.host/latest?base={base}")
        if response.status_code == 200:
            return response.json().get("rates", {})
        else:
            return {"USD": 0.055, "EUR": 0.051}
    except Exception:
        return {"USD": 0.055, "EUR": 0.051}

def convert_price(price_zar, currency, rates):
    return price_zar * rates.get(currency, 1)

def generate_pdf_report(uploaded_image_path, top_matches, price_range, currency, converted_range):
    # Generates a PDF report for the tattoo quote.
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    if os.path.exists(LOGO_PATH):
        pdf.image(LOGO_PATH, x=10, y=8, w=30)
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 15, "Tattoo Quote Report", ln=True, align="C")
    pdf.ln(15)

    # Main Info
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Date: {date.today()}", ln=True)
    pdf.cell(0, 8, f"Estimated Price Range (ZAR): R{price_range[0]} - R{price_range[1]}", ln=True)
    if currency != "ZAR":
        pdf.cell(0, 8, f"Converted Price ({currency}): {converted_range[0]:.2f} - {converted_range[1]:.2f}", ln=True)
    pdf.ln(10)

    # User's Image
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Your Reference Image:", ln=True)
    if os.path.exists(uploaded_image_path):
        pdf.image(uploaded_image_path, w=80)
    pdf.ln(5)

    # Top Matches
    pdf.cell(0, 10, "Our Closest Matches:", ln=True)
    for _, row in top_matches.iterrows():
         pdf.multi_cell(0, 8, f"- Artist: {row['artist']}, Style: {row['style']}\n  Price: R{row['price']}, Time: {row.get('time_hours', 'N/A')} hrs")
    
    output_path = os.path.join(IMAGE_DIR, "quote_report.pdf")
    pdf.output(output_path)
    return output_path

# --- APP PAGES ---

def generate_pdf_report(image_path, top_matches, price_range, currency, converted_range):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Tattoo Quote Report", ln=True, align="C")
    pdf.ln(10)
    if os.path.exists(LOGO_PATH):
        pdf.image(LOGO_PATH, x=80, y=25, w=50)
        pdf.ln(40)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Date: {date.today()}", ln=True)
    pdf.ln(5)
    pdf.cell(0, 10, f"Price Range: R{price_range[0]} - R{price_range[1]}", ln=True)
    if currency != "ZAR":
        pdf.cell(0, 10, f"Converted Price Range: {currency} {converted_range[0]:.2f} - {currency} {converted_range[1]:.2f}", ln=True)
    pdf.ln(10)
    pdf.cell(0, 10, "Top Matches:", ln=True)
    for i, row in top_matches.iterrows():
        pdf.multi_cell(0, 10, f"{row['artist']} - {row['style']} | Price: R{row['price']} | Time: {row['time']} hrs")
    if image_path and os.path.exists(image_path):
        pdf.ln(10)
        pdf.image(image_path, w=100)
    output_path = os.path.join(IMAGE_DIR, "quote_report.pdf")
    pdf.output(output_path)
    return output_path

def quote_tattoo():
    st.header("Quote Tattoo")
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True, width=200)

def page_quote_tattoo():
    st.header("Quote Your Tattoo")
    tattoo_data = load_data_from_supabase()

    if tattoo_data[tattoo_data['embedding'].notna()].empty:
        st.error("No reference tattoos with embeddings found in the database. Please run the embedding generation script.")
        return

    uploaded_img = st.file_uploader("Upload a clear image of the tattoo or reference design", type=["jpg", "jpeg", "png"])
    
    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        
        if CROP_AVAILABLE:
            st.write("Crop the image to focus on the main design:")
            image = st_cropper(image, realtime_update=True, box_color="#0015FF")
        
        st.image(image, caption="Your Tattoo Reference", use_container_width=True)

        # --- Process and Compare ---
        inputs = processor(images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            query_embedding = model.get_image_features(**inputs).squeeze(0)

        # Filter data to only include tattoos with embeddings
        db = tattoo_data[tattoo_data['embedding'].notna()].copy()
        
        # Calculate similarity
        db_embeddings = torch.tensor(np.stack(db['embedding'].values))
        similarities = torch.nn.functional.cosine_similarity(query_embedding, db_embeddings)
        db['similarity'] = similarities.tolist()

        # --- Display Results ---
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

        # PDF Download Button
        temp_path = os.path.join(IMAGE_DIR, "temp_uploaded.png")
        image.save(temp_path)
        pdf_path = generate_pdf_report(temp_path, top_matches, (min_price, max_price), currency, converted_range)
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download Quote Report (PDF)", f, file_name="Tattoo_Quote.pdf")

        for _, match in top_matches.iterrows():
            st.markdown("---")
            st.write(f"**Similarity Score:** {match['similarity']:.2f}")
            st.image(match["image_url"], caption=f"Artist: {match['artist']}, Style: {match['style']}", use_container_width=True)

def page_supabase_upload():
    st.header("📸 Upload New Tattoo")
    with st.form("upload_form", clear_on_submit=True):
        artist = st.selectbox("Artist", settings["artists"])
        style  = st.selectbox("Style", settings["styles"])
        price  = st.number_input("Final Price (R)", min_value=0, step=100)
        time_hours = st.number_input("Time Taken (in hours)", min_value=0.5, step=0.5)
        uploaded_file = st.file_uploader("Upload Finished Tattoo Image", type=["jpg", "jpeg", "png"])
        
        submitted = st.form_submit_button("Save to Database")

        if submitted:
            if not all([artist, style, price, time_hours, uploaded_file]):
                st.error("Please fill out all fields and upload an image.")
                return
            
            with st.spinner("Uploading and saving..."):
                # 1. Upload image to Supabase Storage
                file_content = uploaded_file.getvalue()
                file_name = f"{artist.replace(' ', '_')}_{date.today()}_{uploaded_file.name}"
                
                bucket = supabase.storage.from_("tattoo-images")
                bucket.upload(file_name, file_content, file_options={"upsert": "true"})
                image_url = bucket.get_public_url(file_name)

                # 2. Insert record into Supabase Table
                supabase.table("tattoos").insert({
                    "artist": artist,
                    "style": style,
                    "price": price,
                    "time_hours": time_hours,
                    "image_url": image_url
                }).execute()
            
                st.success(f"Successfully saved tattoo by {artist}!")
                st.image(file_content, caption="Uploaded Tattoo")

def page_settings():
    st.header("⚙️ App Settings (Stored in Supabase)")

    # --- Manage Artists ---
    st.subheader("Manage Artists")
    artists = settings.get("artists", [])
    new_artist = st.text_input("Add New Artist", key="new_artist")
    if st.button("Add Artist"):
        if new_artist and new_artist not in artists:
            artists.append(new_artist)
            save_settings("artists", artists) # Save the updated list
            st.success(f"Added '{new_artist}'")
            st.rerun()
        else:
            st.warning("Artist name cannot be empty or already exist.")

    artist_to_remove = st.selectbox("Remove Artist", ["-"] + artists, key="remove_artist")
    if st.button("Remove Artist"):
        if artist_to_remove != "-":
            artists.remove(artist_to_remove)
            save_settings("artists", artists) # Save the updated list
            st.success(f"Removed '{artist_to_remove}'")
            st.rerun()
    st.markdown("---")

    # --- Manage Styles ---
    st.subheader("Manage Styles")
    styles = settings.get("styles", [])
    new_style = st.text_input("Add New Style", key="new_style")
    if st.button("Add Style"):
        if new_style and new_style not in styles:
            styles.append(new_style)
            save_settings("styles", styles) # Save the updated list
            st.success(f"Added '{new_style}'")
            st.rerun()
        else:
            st.warning("Style name cannot be empty or already exist.")

    style_to_remove = st.selectbox("Remove Style", ["-"] + styles, key="remove_style")
    if st.button("Remove Style"):
        if style_to_remove != "-":
            styles.remove(style_to_remove)
            save_settings("styles", styles) # Save the updated list
            st.success(f"Removed '{style_to_remove}'")
            st.rerun()

# --- Main App Navigation ---
def main():
    st.sidebar.image(LOGO_PATH, width=100)
    st.sidebar.title("Navigation")
    
    pages = {
        "Quote Tattoo": page_quote_tattoo,
        "Upload Tattoo": page_supabase_upload,
        "Settings": page_settings,
    }
    
    choice = st.sidebar.radio("Go to", list(pages.keys()))
    pages[choice]() # Run the selected page function

    st.sidebar.markdown("---")
    st.sidebar.info(f"App Version: {APP_VERSION}")

if __name__ == "__main__":
    main()

    if img:
        image = Image.open(img).convert("RGB")
        temp_path = os.path.join(IMAGE_DIR, "uploaded_image.png")
        image.save(temp_path)

        if CROP_AVAILABLE:
            st.write("Crop the tattoo image to focus on the design:")
            cropped_img = st_cropper(image, realtime_update=True, box_color="#0015FF", aspect_ratio=None)
            st.image(cropped_img, caption="Cropped Tattoo Image", use_container_width=True)
            image = cropped_img

        inputs = processor(images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            q_feats = model.get_image_features(**inputs)

        df = load_data()
        df = df[~df["artist"].isin(settings.get("archived_artists", []))]
        if artist_filter != "All":
            df = df[df["artist"] == artist_filter]

        feats, rows = [], []
        for _, r in df.iterrows():
            path = os.path.join(IMAGE_DIR, r["filename"])
            if os.path.exists(path):
                ref = Image.open(path).convert("RGB")
                in_ref = processor(images=ref, return_tensors="pt", padding=True)
                with torch.no_grad():
                    feats.append(model.get_image_features(**in_ref).squeeze(0))
                rows.append(r)

        if feats:
            sims = torch.nn.functional.cosine_similarity(q_feats, torch.stack(feats))
            dfm = pd.DataFrame(rows)
            dfm["sim"] = sims.tolist()
            dfm.sort_values("sim", ascending=False, inplace=True)
            top_matches = dfm.head(compare_count)
            min_price = top_matches["price"].min()
            max_price = top_matches["price"].max()

            st.subheader("Top Matches")
            st.write(f"**Price Range:** R{min_price} - R{max_price}")
            converted_range = (convert_price(min_price, currency, rates), convert_price(max_price, currency, rates))
            if currency != "ZAR":
                st.write(f"**Converted Price Range:** {currency} {converted_range[0]:.2f} - {currency} {converted_range[1]:.2f}")

            for _, match in top_matches.iterrows():
                st.markdown(f"### {match['artist']} - {match['style']}")
                st.write(f"Price: R{match['price']} ({currency} {convert_price(match['price'], currency, rates):.2f})")
                st.write(f"Time: {match['time']} hrs")
                st.image(os.path.join(IMAGE_DIR, match["filename"]), use_container_width=True)
                st.markdown("---")

            if st.button("📥 Download Quote Report (PDF)"):
                pdf_path = generate_pdf_report(temp_path, top_matches, (min_price, max_price), currency, converted_range)
                with open(pdf_path, "rb") as f:
                    st.download_button("Download PDF", f, file_name="tattoo_quote_report.pdf")

            new_log = pd.DataFrame({"date":[pd.to_datetime(date.today())], "artist":[top_matches.iloc[0]['artist']]})
            save_logs(pd.concat([logs, new_log], ignore_index=True))
        else:
            st.warning("No samples available.")

def supabase_upload():
    st.header("📸 Upload Tattoo & Save Quote (DEV)")
    artist = st.selectbox("Artist", ["Select an artist"] + settings["artists"])
    style  = st.selectbox("Style",  ["Select a style"]  + settings["styles"])
    price  = st.number_input("Estimated Price (R)", min_value=0)
    time_est = st.text_input("Time Estimate (e.g. 3 hours)")
    up = st.file_uploader("Upload Tattoo Image", type=["jpg","jpeg","png"])
    if st.button("Save to Supabase"):
        if artist == "Select an artist" or style == "Select a style":
            st.error("Please choose both artist and style.")
            return
        if not up:
            st.error("Please upload an image.")
            return
        temp_path = os.path.join(IMAGE_DIR, up.name)
        with open(temp_path, "wb") as f:
            f.write(up.getbuffer())
        bucket = supabase.storage.from_("tattoo-images")
        with open(temp_path, "rb") as f:
            bucket.upload(up.name, f, file_options={"upsert": True})
        url = bucket.get_public_url(up.name)
        supabase.table("tattoos").insert({
            "artist": artist,
            "style": style,
            "price": price,
            "time_estimate": time_est,
            "image_url": url
        }).execute()
        st.success("Saved to Supabase! (overwritten if existed)")
        st.image(Image.open(temp_path), use_container_width=True)
        os.remove(temp_path)

def saved_tattoos():
    st.markdown("---")
    st.header("🖼️ Saved Tattoos")
    try:
        response = supabase.table("tattoos").select("*").order("created_at", desc=True).execute()
        if response.data:
            for row in response.data:
                st.subheader(f"{row['artist']} — {row['style']}")
                st.image(row["image_url"], use_container_width=True)
                st.write(f"💰 Price: R{row['price']}")
                st.write(f"⏱️ Time Estimate: {row['time_estimate']}")
                st.markdown("---")
        else:
            st.info("No tattoos saved yet.")
    except Exception as e:
        st.error(f"Error fetching tattoos: {e}")


def settings_page():
    st.header("App Settings")

    # Artists Management
    st.subheader("Manage Artists")
    new_artist = st.text_input("Add New Artist")
    if st.button("Add Artist") and new_artist.strip():
        if new_artist not in settings["artists"]:
            settings["artists"].append(new_artist)
            save_settings(settings)
            st.success(f"Artist '{new_artist}' added!")
        else:
            st.warning(f"Artist '{new_artist}' already exists.")

    selected_artist = st.selectbox("Select Artist to Remove", ["None"] + settings["artists"])
    if st.button("Remove Artist") and selected_artist != "None":
        settings["artists"].remove(selected_artist)
        save_settings(settings)
        st.success(f"Artist '{selected_artist}' removed!")

    st.markdown("---")

    # Styles Management
    st.subheader("Manage Styles")
    new_style = st.text_input("Add New Style")
    if st.button("Add Style") and new_style.strip():
        if new_style not in settings["styles"]:
            settings["styles"].append(new_style)
            save_settings(settings)
            st.success(f"Style '{new_style}' added!")
        else:
            st.warning(f"Style '{new_style}' already exists.")

    selected_style = st.selectbox("Select Style to Remove", ["None"] + settings["styles"])
    if st.button("Remove Style") and selected_style != "None":
        settings["styles"].remove(selected_style)
        save_settings(settings)
        st.success(f"Style '{selected_style}' removed!")


def reports_page():
    st.header("Match Reports (DEV)")
    df = load_data()
    if df.empty:
        st.info("No data available to generate reports.")
        return

    st.subheader("Summary Statistics")
    st.write(f"**Total Tattoos in DB:** {len(df)}")

    if 'artist' in df.columns:
        top_artist = df['artist'].value_counts().idxmax()
        st.write(f"**Most Quoted Artist:** {top_artist}")

    if 'style' in df.columns:
        top_style = df['style'].value_counts().idxmax()
        st.write(f"**Most Popular Style:** {top_style}")

    if 'price' in df.columns:
        st.write(f"**Average Price:** R{df['price'].mean():.2f}")

    if 'time' in df.columns:
        st.write(f"**Average Time:** {df['time'].mean():.1f} hrs")



def main():
    st.sidebar.markdown(f"**Version:** {APP_VERSION}")
    st.sidebar.markdown("---")
    pages=["Quote Tattoo","Supabase Upload","Saved Tattoos","Settings","Reports"]
    choice=st.sidebar.radio("Navigate",pages)
    if choice=="Quote Tattoo":
        quote_tattoo()
    elif choice=="Supabase Upload":
        supabase_upload()
    elif choice=="Saved Tattoos":
        saved_tattoos()
    elif choice=="Settings":
        settings_page()
    else:
        reports_page()

if __name__=="__main__":
    main()
