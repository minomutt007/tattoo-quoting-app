import streamlit as st
from PIL import Image
from datetime import date
from io import BytesIO
from database import supabase, load_settings

CROP_AVAILABLE = True
try:
    from streamlit_cropper import st_cropper
except ImportError:
    CROP_AVAILABLE = False
TATTOO_COLOR_TYPES = ["Black & Grey", "Full Color"]
settings = load_settings()

st.header("📦 Batch Upload Tattoos")

uploaded_files = st.file_uploader("1. Upload all tattoo images for this session", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch_uploader")

if uploaded_files:
    if 'batch_files' not in st.session_state or st.session_state.batch_files != uploaded_files:
        st.session_state.batch_files = uploaded_files
        st.session_state.details = [{} for _ in uploaded_files]
        st.session_state.current_index = 0
        st.session_state.cropped_images = [None] * len(uploaded_files)
    
    current_index = st.session_state.current_index
    if current_index < len(uploaded_files):
        st.write(f"--- \n### 2. Crop and Enter Details for Image {current_index + 1} of {len(uploaded_files)}")
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
                details['ai_caption'] = st.text_area("AI Training Description", height=150, placeholder="e.g., A black and grey realism tattoo...", key=f"cap_{current_index}")
                
                if st.form_submit_button("✅ Save and Next"):
                    st.session_state.details[current_index] = details
                    st.session_state.cropped_images[current_index] = cropped_img
                    st.session_state.current_index += 1
                    st.rerun()
    else:
        st.success("All details entered! Ready to upload to Supabase.")
        if st.button("🚀 Save All to Supabase"):
            progress_bar = st.progress(0, text="Uploading...")
            for i in range(len(st.session_state.batch_files)):
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
                progress_bar.progress((i + 1) / len(st.session_state.batch_files), text=f"Uploaded {original_file.name}")
            st.success("Batch upload complete! Remember to run the embedding script.")
            st.session_state.clear()