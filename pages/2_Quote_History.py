import streamlit as st
import pandas as pd
from database import supabase

st.header("📜 Quote History")

response = supabase.table("quotes").select("*").order("quote_date", desc=True).execute()
quotes_df = pd.DataFrame(response.data)

search_term = st.text_input("Search by Customer Name")
if search_term:
    quotes_df = quotes_df[quotes_df['customer_name'].str.contains(search_term, case=False, na=False)]

if quotes_df.empty:
    st.info("No saved quotes found.")
else:
    for _, row in quotes_df.iterrows():
        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(row['reference_image_url'])
        with col2:
            st.write(f"**Customer:** {row['customer_name']}")
            st.write(f"**Date:** {pd.to_datetime(row['quote_date']).strftime('%Y-%m-%d')}")
            st.write(f"**Quoted Price Range:** {row['final_price_range']}")
            if 'final_time_range' in row and row['final_time_range']:
                st.write(f"**Quoted Time Range:** {row['final_time_range']}")