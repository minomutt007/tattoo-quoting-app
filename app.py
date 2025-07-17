import streamlit as st
import pandas as pd
import torch
import os
import json
from datetime import date, timedelta
from PIL import Image
from supabase import create_client
from transformers import CLIPModel, CLIPProcessor
from storage3.exceptions import StorageApiError  # Import for duplicate error handling

# --- CONFIG ---
CSV_PATH = "tattoos.csv"
SETTINGS_PATH = "settings.json"
IMAGE_DIR = "images"
LOGS_PATH = "match_logs.csv"
SUPABASE_URL = "https://ryessoqfbdbgluzedegt.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ5ZXNzb3FmYmRiZ2x1emVkZWd0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI2NjgzMzYsImV4cCI6MjA2ODI0NDMzNn0.GRHnX0uMnIRZOLLTJhZ-Onek5YZmniweA4OjDBq8OzM"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Ensure image directory exists
os.makedirs(IMAGE_DIR, exist_ok=True)

# Optional image cropper
try:
    from streamlit_cropper import st_cropper
    CROP_AVAILABLE = True
except ImportError:
    CROP_AVAILABLE = False

# ----------------------
# Data & Model Loading
# ----------------------
@st.cache_data
def load_settings():
    if os.path.exists(SETTINGS_PATH):
        settings = json.load(open(SETTINGS_PATH))
        # ensure model_variant key exists
        if not settings.get("model_variant"):
            settings["model_variant"] = "openai/clip-vit-base-patch32"
            json.dump(settings, open(SETTINGS_PATH, "w"), indent=2)
        return settings

    settings = {
        "artists": [], 
        "styles": [], 
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

with st.spinner("Loading CLIP model (may take a moment)..."):
    model, processor = load_clip_model()

@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame(
        columns=["filename","artist","style","price","time"]
    )

@st.cache_data
def load_logs():
    return pd.read_csv(LOGS_PATH, parse_dates=["date"]) if os.path.exists(LOGS_PATH) else pd.DataFrame(
        columns=["date","artist"]
    )

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

# ----------------------
# Page Functions
# ----------------------
def quote_tattoo():
    st.header("Quote Tattoo")
    img = st.file_uploader("Upload Tattoo Image", type=["jpg","jpeg","png"])
    artist_filter = st.selectbox("Filter by Artist", ["All"] + settings["artists"])
    if img:
        image = Image.open(img).convert("RGB")
        if CROP_AVAILABLE:
            st.markdown("#### Crop Tattoo Region")
            image = st_cropper(image, box_color="blue", realtime_update=True)
        st.image(image, use_container_width=True)
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
            best = dfm.iloc[0]
            st.subheader("Best Match")
            st.write(f"Artist: {best['artist']}")
            st.write(f"Style: {best['style']}")
            st.write(f"Price: R{best['price']}")
            st.write(f"Time: {best['time']} hrs")
            st.image(os.path.join(IMAGE_DIR, best["filename"]), use_container_width=True)

            new_log = pd.DataFrame({"date":[pd.to_datetime(date.today())], "artist":[best['artist']]})
            save_logs(pd.concat([logs, new_log], ignore_index=True))
        else:
            st.warning("No samples available.")

def supabase_upload():
    st.header("📸 Upload Tattoo & Save Quote")
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
        try:
            bucket.upload(up.name, open(temp_path, "rb"))
        except StorageApiError as e:
            if getattr(e, "error", None) == "Duplicate":
                bucket.update(up.name, open(temp_path, "rb"))
            else:
                raise

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

def settings_page():
    st.header("App Settings")
    st.subheader("➕ Add Artist")
    new_artist = st.text_input("New Artist Name", "")
    if st.button("Add Artist"):
        if new_artist and new_artist not in settings["artists"]:
            settings["artists"].append(new_artist)
            save_settings(settings)
            st.success(f"Artist '{new_artist}' added.")
        elif new_artist:
            st.warning("That artist already exists.")

    st.subheader("➕ Add Tattoo Style")
    new_style = st.text_input("New Style", "")
    if st.button("Add Style"):
        if new_style and new_style not in settings["styles"]:
            settings["styles"].append(new_style)
            save_settings(settings)
            st.success(f"Style '{new_style}' added.")
        elif new_style:
            st.warning("That style already exists.")
    st.markdown("---")
    st.subheader("🛠️ Manage Artists")
    if settings["artists"]:
        artist_to_manage = st.selectbox("Select an artist to manage", settings["artists"])
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Archive Artist"):
                settings["artists"].remove(artist_to_manage)
                settings.setdefault("archived_artists", []).append(artist_to_manage)
                save_settings(settings)
                st.success(f"Artist '{artist_to_manage}' archived.")
        with col2:
            if st.button("Delete Artist"):
                settings["artists"] = [a for a in settings["artists"] if a != artist_to_manage]
                settings["archived_artists"] = [a for a in settings.get("archived_artists", []) if a != artist_to_manage]
                save_settings(settings)
                st.success(f"Artist '{artist_to_manage}' deleted.")
    else:
        st.info("No artists to manage.")
    st.markdown("---")
    st.subheader("📦 Archived Artists")
    archived = settings.get("archived_artists", [])
    if archived:
        st.write(archived)
    else:
        st.write("No archived artists.")

def reports_page():
    st.header("Match Reports")
    presets = ["Custom", "Today", "Yesterday", "Last 7 Days", "Last 14 Days"]
    preset = st.selectbox("Quick Range", presets)
    today = date.today()
    ranges = {"Today":(today,today),"Yesterday":(today - timedelta(1),today - timedelta(1)),"Last 7 Days":(today - timedelta(7),today),"Last 14 Days":(today - timedelta(14),today)}
    start,end = ranges.get(preset,(today - timedelta(7),today))
    dates = st.date_input("Select Date Range",[start,end])
    if len(dates)==2:
        d0,d1 = dates
        lf = load_logs()
        lf['date'] = pd.to_datetime(lf['date']).dt.date
        rpt = lf[(lf['date']>=d0)&(lf['date']<=d1)].groupby('artist').size().reset_index(name='matches')
        if not rpt.empty:
            st.dataframe(rpt)
            st.bar_chart(rpt.set_index('artist'))
            csv = rpt.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV",csv,"match_report.csv","text/csv")
        else:
            st.info("No matches found.")

def main():
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
