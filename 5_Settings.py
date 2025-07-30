import streamlit as st
from database import load_settings, save_settings, supabase

settings = load_settings()

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
st.markdown("---")

st.subheader("Manage Complicated Placements")
complicated_placements = settings.get("complicated_placements", {})
col1, col2 = st.columns(2)
with col1:
    placement_to_add = st.selectbox("Select Placement", placements)
with col2:
    percent_increase = st.number_input("Time Increase (%)", min_value=0, max_value=200, value=10, step=5)
if st.button("Add/Update Complicated Placement"):
    complicated_placements[placement_to_add] = percent_increase
    save_settings("complicated_placements", complicated_placements)
    st.success(f"Set {placement_to_add} to a {percent_increase}% time increase.")
    st.rerun()
if complicated_placements:
    st.write("Current Complicated Placements:")
    for placement, pct in complicated_placements.items():
        col_a, col_b = st.columns([3, 1])
        col_a.write(f"- **{placement}**: +{pct}% time")
        if col_b.button("Remove", key=f"remove_comp_{placement}"):
            del complicated_placements[placement]
            save_settings("complicated_placements", complicated_placements)
            st.success(f"Removed {placement}.")
            st.rerun()
st.markdown("---")

st.subheader("Artist Price Adjustment")
selected_artist = st.selectbox("Select Artist to Adjust Prices", artists)
adjustment_type = st.radio(
    "Adjustment Type",
    ["Percentage (%) Increase", "Fixed Amount (R) Increase"],
    horizontal=True
)
if adjustment_type == "Percentage (%) Increase":
    adj_value = st.number_input("Increase by Percentage (%)", min_value=0.0, step=0.5, format="%.1f")
else:
    adj_value = st.number_input("Increase by Fixed Amount (R)", min_value=0, step=50)
if st.button("Apply Price Adjustment"):
    st.session_state.adjustment_details = {
        "artist": selected_artist, "type": adjustment_type, "value": adj_value
    }
if 'adjustment_details' in st.session_state:
    details = st.session_state.adjustment_details
    artist = details['artist']
    adj_type = details['type']
    value = details['value']
    st.warning(f"**Confirmation Needed:** You are about to increase all prices for **{artist}** by **{value}{'%' if '%' in adj_type else ' R'}**. This action cannot be undone.")
    if st.button("Confirm and Update Prices"):
        with st.spinner(f"Updating prices for {artist}..."):
            response = supabase.table("tattoos").select("id, price").eq("artist", artist).execute()
            tattoos_to_update = response.data
            progress_bar = st.progress(0, text="Starting update...")
            for i, tattoo in enumerate(tattoos_to_update):
                old_price = tattoo['price']
                if '%' in adj_type:
                    new_price = old_price * (1 + (value / 100))
                else:
                    new_price = old_price + value
                supabase.table("tattoos").update({"price": round(new_price)}).eq("id", tattoo["id"]).execute()
                progress_bar.progress((i + 1) / len(tattoos_to_update), text=f"Updated tattoo {i+1}/{len(tattoos_to_update)}")
        st.success(f"Successfully updated {len(tattoos_to_update)} tattoos for {artist}!")
        del st.session_state.adjustment_details
        st.cache_data.clear()
        st.rerun()