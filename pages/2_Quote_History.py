import streamlit as st
import pandas as pd
from database import supabase

st.header("📜 Quote History")

# --- Function to delete a quote ---
def delete_quote(quote_id):
    try:
        supabase.table("quotes").delete().eq("id", quote_id).execute()
        st.success("Quote deleted successfully!")
        # Clear the search box after deletion if needed
        if 'search_term' in st.session_state:
            st.session_state.search_term = ""
    except Exception as e:
        st.error(f"Failed to delete quote: {e}")

# --- Main Page Logic ---
response = supabase.table("quotes").select("*").order("quote_date", desc=True).execute()
quotes_df = pd.DataFrame(response.data)

search_term = st.text_input("Search by Customer Name", key="search_term")
if search_term:
    quotes_df = quotes_df[quotes_df['customer_name'].str.contains(search_term, case=False, na=False)]

if quotes_df.empty:
    st.info("No saved quotes found.")
else:
    for index, row in quotes_df.iterrows():
        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(row['reference_image_url'], use_column_width=True)
        
        with col2:
            st.write(f"**Customer:** {row['customer_name']}")
            st.write(f"**Date:** {pd.to_datetime(row['quote_date']).strftime('%Y-%m-%d')}")
            st.write(f"**Quoted Price Range:** {row['final_price_range']}")
            if 'final_time_range' in row and row['final_time_range']:
                st.write(f"**Quoted Time Range:** {row['final_time_range']}")
            
            # --- NEW FEATURE: DELETE BUTTON ---
            if st.button("🗑️ Delete Quote", key=f"delete_{row['id']}", type="primary"):
                delete_quote(row['id'])
                st.rerun() # Rerun the script to refresh the list