
import streamlit as st
import pandas as pd
import torch
import os
import json
import requests
from datetime import date, timedelta
from PIL import Image
from fpdf import FPDF
from supabase import create_client
from transformers import CLIPModel, CLIPProcessor
from storage3.exceptions import StorageApiError

# --- CONFIG ---
APP_VERSION = "1.2.1"
CSV_PATH = "tattoos.csv"
SETTINGS_PATH = "settings.json"
IMAGE_DIR = "images"
LOGS_PATH = "match_logs.csv"
SUPABASE_URL = "https://ryessoqfbdbgluzedegt.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ5ZXNzb3FmYmRiZ2x1emVkZWd0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI2NjgzMzYsImV4cCI6MjA2ODI0NDMzNn0.GRHnX0uMnIRZOLLTJhZ-Onek5YZmniweA4OjDBq8OzM"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

LOGO_PATH = os.path.join(IMAGE_DIR, "sally_mustang_logo.jpg")

os.makedirs(IMAGE_DIR, exist_ok=True)

try:
    from streamlit_cropper import st_cropper
    CROP_AVAILABLE = True
except ImportError:
    CROP_AVAILABLE = False

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

settings = load_settings()

@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained(settings["model_variant"])
    processor = CLIPProcessor.from_pretrained(settings["model_variant"])
    return model, processor

with st.spinner("Loading CLIP model..."):
    model, processor = load_clip_model()

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

    img = st.file_uploader("Upload Tattoo Image", type=["jpg","jpeg","png"])
    artist_filter = st.selectbox("Filter by Artist", ["All"] + settings["artists"])
    compare_count = st.slider("Number of similar tattoos to compare", 1, 5, 3)
    currency = st.selectbox("Currency", ["ZAR", "USD", "EUR"])
    rates = get_live_rates("ZAR")

    if img:
        image = Image.open(img).convert("RGB")
        temp_path = os.path.join(IMAGE_DIR, "uploaded_image.png")
        image.save(temp_path)

        if CROP_AVAILABLE:
            st.write("Crop the tattoo image to focus on the design:")
            cropped_img = st_cropper(image, realtime_update=True, box_color='#FF0004', aspect_ratio=None)
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
    st.header("🖼️ Saved Tattoos (DEV)")
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
    st.header("App Settings (DEV)")

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
