import streamlit as st
import os
from supabase import create_client

try:
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    supabase = create_client(supabase_url, supabase_key)
except (KeyError, AttributeError):
    pass

def login_user(email, password):
    try:
        user = supabase.auth.sign_in_with_password({"email": email, "password": password})
        st.session_state['authenticated'] = True
        st.session_state['user_email'] = user.user.email
        st.rerun()
    except Exception as e:
        st.error(f"Login failed. Please check your credentials.")

def logout_user():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state['authenticated'] = False
    st.rerun()

def show_login_form():
    if os.path.exists("images/sally_mustang_logo.jpg"):
        st.image("images/sally_mustang_logo.jpg", width=200)
    st.header("Tattoo Quoting App Login")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if email and password:
                login_user(email, password)
            else:
                st.warning("Please enter both email and password.")