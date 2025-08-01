import streamlit as st
import os
from auth import show_login_form, logout_user

# --- CONFIG ---
APP_VERSION = "5.1.0 (CSS Fix)"
LOGO_PATH = os.path.join("images", "sally_mustang_logo.jpg")

st.set_page_config(page_title="Tattoo Quoting App", layout="wide")

# --- CSS TO HIDE/SHOW THE SIDEBAR NAVIGATION ---
def hide_pages_nav():
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] ul {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)

def show_pages_nav():
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] ul {
                display: block;
            }
        </style>
    """, unsafe_allow_html=True)

# --- MAIN APP LOGIC ---
def main_app():
    """This function runs the main application after the user has logged in."""
    show_pages_nav() # <-- Show the navigation links

    if os.path.exists(LOGO_PATH):
        st.sidebar.image(LOGO_PATH, width=100)
    
    st.sidebar.write(f"Logged in as: {st.session_state.get('user_email', '')}")
    st.sidebar.button("Logout", on_click=logout_user)
    
    st.title("Welcome to the Tattoo Quoting App")
    st.write("Please select a page from the sidebar to get started.")

    st.sidebar.info(f"App Version: {APP_VERSION}")

# --- APP ENTRY POINT ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if st.session_state['authenticated']:
    main_app()
else:
    hide_pages_nav() # <-- Hide the navigation links
    show_login_form()