
import streamlit as st
import pandas as pd
import torch
import json
import numpy as np
from io import BytesIO
from supabase import create_client
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

# --- SETUP ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except KeyError:
    print("ERROR: Supabase credentials are not set in st.secrets. Please add them.")
    st.stop()

# Load Model
print("Loading CLIP model...")
MODEL_ID = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL_ID)
processor = CLIPProcessor.from_pretrained(MODEL_ID)
print("Model loaded.")

# --- SCRIPT LOGIC ---
def generate_and_update_embeddings():
    # 1. Fetch records
    print("Fetching tattoos from database...")
    response = supabase.table("tattoos").select("id, image_url, embedding").execute()
    df = pd.DataFrame(response.data)
    
    # --- FIX: Handle empty table ---
    if df.empty:
        print("The 'tattoos' table is empty. Nothing to process.")
        return

    # 2. Filter for rows where embedding is None
    missing_embeddings = df[df['embedding'].isnull()]
    
    if missing_embeddings.empty:
        print("All tattoos already have embeddings. Nothing to do!")
        return

    print(f"Found {len(missing_embeddings)} tattoos to process.")
    bucket_name = "tattoo-images"

    for index, row in missing_embeddings.iterrows():
        try:
            print(f"Processing ID: {row['id']}...")
            
            # 3. Download image using Supabase client
            file_path = row['image_url'].split(f'/{bucket_name}/')[1]
            response_bytes = supabase.storage.from_(bucket_name).download(file_path)
            
            # 4. Generate embedding
            image = Image.open(BytesIO(response_bytes)).convert("RGB")
            inputs = processor(images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                embedding = model.get_image_features(**inputs).squeeze(0).numpy()
            
            # 5. Convert embedding to a JSON string list
            embedding_json = json.dumps(embedding.tolist())
            
            # 6. Update the record in Supabase
            supabase.table("tattoos").update({"embedding": embedding_json}).eq("id", row["id"]).execute()
            print(f"  -> Successfully updated ID: {row['id']}")

        except Exception as e:
            print(f"  -> FAILED to process ID: {row['id']}. Reason: {e}")

if __name__ == "__main__":
    generate_and_update_embeddings()