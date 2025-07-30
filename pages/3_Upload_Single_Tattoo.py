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
    
    generated_caption = (f"A {style} tattoo of a size around {size_cm} cm, in {color_type}, "
                         f"on the {placement.lower()}, by artist {artist}.")
    ai_caption = st.text_area("AI Training Description", value=generated_caption, height=150)

    submitted = st.form_submit_button("Save to Database")
    if submitted:
        image_to_upload = cropped_img if cropped_img else (Image.open(uploaded_file) if uploaded_file else None)
        if not all([artist, style, size_cm > 0, placement, color_type, price > 0, time_hours > 0, image_to_upload]):
            st.error("Please upload an image and fill out all fields.")
        else:
            with st.spinner("Uploading and saving..."):
                buffer = BytesIO()
                image_to_upload.save(buffer, format="PNG")
                file_content = buffer.getvalue()
                original_filename = uploaded_file.name
                file_name = f"{artist.replace(' ', '_')}_{date.today()}_{original_filename}"
                supabase.storage.from_("tattoo-images").upload(file_name, file_content, file_options={"upsert": "true"})
                image_url = supabase.storage.from_("tattoo-images").get_public_url(file_name)
                
                supabase.table("tattoos").insert({
                    "artist": artist, "style": style, "price": price, "time_hours": time_hours, 
                    "image_url": image_url, "size_cm": size_cm, "placement": placement, 
                    "color_type": color_type, "ai_caption": ai_caption
                }).execute()
                st.success(f"Successfully saved tattoo by {artist}!")
                st.image(image_to_upload, caption="Uploaded Tattoo")