import streamlit as st
import pandas as pd
import torch
import os
from PIL import Image
from io import BytesIO
from transformers import CLIPModel, CLIPProcessor
from database import load_data_from_supabase, load_settings, supabase
from utils import generate_pdf_report, get_live_rates

# --- CONFIG ---
CROP_AVAILABLE = True
try:
    from streamlit_cropper import st_cropper
except ImportError:
    CROP_AVAILABLE = False
TATTOO_COLOR_TYPES = ["Black & Grey", "Full Color"]
IMAGE_DIR = "images"

# --- Load Model and Settings ---
settings = load_settings()
model, processor = CLIPModel.from_pretrained(settings["model_variant"]), CLIPProcessor.from_pretrained(settings["model_variant"])


st.header("Quote Your Tattoo")

customer_name = st.text_input("Customer Name (Optional)")

if st.checkbox("Start a New Quote"):
    uploaded_img = st.file_uploader(
        "Upload a clear image of the tattoo or reference design",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_img:
        st.markdown("---")
        st.subheader("Filtering Options")
        size_range = st.slider("Filter by Size (cm)", 1.0, 40.0, (5.0, 15.0))
        col1, col2 = st.columns(2)
        with col1:
            artist_filter = st.selectbox("Artist", ["All"] + settings.get("artists", []))
            placement_filter = st.selectbox("Body Placement", ["All"] + settings.get("placements", []))
        with col2:
            color_filter = st.selectbox("Color Type", ["All"] + TATTOO_COLOR_TYPES)
            compare_count = st.slider("Number to Compare", 1, 10, 3)

        tattoo_data = load_data_from_supabase()

        if not tattoo_data.empty and 'size_cm' in tattoo_data.columns and tattoo_data['size_cm'].notna().any():
            tattoo_data = tattoo_data[tattoo_data['size_cm'].between(size_range[0], size_range[1])]
        if artist_filter != "All": tattoo_data = tattoo_data[tattoo_data['artist'] == artist_filter]
        if placement_filter != "All": tattoo_data = tattoo_data[tattoo_data['placement'] == placement_filter]
        if color_filter != "All": tattoo_data = tattoo_data[tattoo_data['color_type'] == color_filter]

        if tattoo_data.empty or tattoo_data[tattoo_data['embedding'].notna()].empty:
            st.warning("No reference tattoos found for the selected filters.")
        else:
            image = Image.open(uploaded_img).convert("RGB")
            if CROP_AVAILABLE:
                image = st_cropper(image, realtime_update=True, box_color="#0015FF")
            st.image(image, caption="Your Tattoo Reference", use_container_width=True)

            inputs = processor(images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                query_embedding = model.get_image_features(**inputs).squeeze(0)
            db = tattoo_data[tattoo_data['embedding'].notna()].copy()
            db_embeddings = torch.tensor(np.stack(db['embedding'].values))
            similarities = torch.nn.functional.cosine_similarity(query_embedding, db_embeddings)
            db['similarity'] = similarities.tolist()
            top_matches = db.sort_values("similarity", ascending=False).head(compare_count)

            if not top_matches.empty:
                min_price, max_price = top_matches["price"].min(), top_matches["price"].max()
                min_time, max_time = top_matches["time_hours"].min(), top_matches["time_hours"].max()

                complicated_placements = settings.get("complicated_placements", {})
                time_increase_percent = complicated_placements.get(placement_filter, 0)

                if time_increase_percent > 0:
                    min_time *= (1 + time_increase_percent / 100)
                    max_time *= (1 + time_increase_percent / 100)

                st.markdown("---")
                st.subheader("Final Quote")
                
                if time_increase_percent > 0:
                    st.warning(f"Note: A {time_increase_percent}% time increase has been added for the selected body part: **{placement_filter}**.")

                col_price, col_time = st.columns(2)
                with col_price:
                    st.metric("Estimated Price Range (ZAR)", f"R{min_price} - R{max_price}")
                with col_time:
                    st.metric("Estimated Time Range", f"{min_time:.1f} - {max_time:.1f} hrs")

                st.subheader("Manual Quote Adjustment")
                colA, colB = st.columns(2)
                final_min_price = colA.number_input("Final Minimum Price (R)", value=int(min_price))
                final_max_price = colB.number_input("Final Maximum Price (R)", value=int(max_price))
                final_min_time = colA.number_input("Final Minimum Time (hrs)", value=float(min_time), step=0.5)
                final_max_time = colB.number_input("Final Maximum Time (hrs)", value=float(max_time), step=0.5)
                final_price_range_str = f"R{final_min_price} - R{final_max_price}"
                final_time_range_str = f"{final_min_time:.1f} - {final_max_time:.1f} hrs"

                if st.button("💾 Save Quote"):
                    with st.spinner("Saving quote..."):
                        buffer = BytesIO()
                        image.save(buffer, format="PNG")
                        ref_image_content = buffer.getvalue()
                        ref_image_name = f"{customer_name.replace(' ', '_') or 'quote'}_{date.today()}_{uploaded_img.name}"
                        supabase.storage.from_("quote-references").upload(ref_image_name, ref_image_content, file_options={"upsert": "true"})
                        ref_image_url = supabase.storage.from_("quote-references").get_public_url(ref_image_name)
                        
                        supabase.table("quotes").insert({
                            "customer_name": customer_name if customer_name else "N/A",
                            "reference_image_url": ref_image_url,
                            "final_price_range": final_price_range_str,
                            "final_time_range": final_time_range_str
                        }).execute()
                        st.success(f"Quote for {customer_name or 'N/A'} saved successfully!")

                with st.expander("Show AI Top Matches"):
                    for _, match in top_matches.iterrows():
                        st.markdown("---")
                        caption_text = (f"Artist: {match['artist']}, Style: {match['style']}, Size: {match.get('size_cm', 'N/A')} cm, "
                                        f"Placement: {match.get('placement', 'N/A')}, Color: {match.get('color_type', 'N/A')}, "
                                        f"Time: {match['time_hours']} hrs")
                        st.image(match["image_url"], caption=caption_text, use_container_width=True)
            else:
                st.warning("Could not find any matching tattoos with the selected criteria.")