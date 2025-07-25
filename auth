import streamlit as st
from supabase import create_client

# Initialize Supabase client
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]
supabase = create_client(supabase_url, supabase_key)

def login_user(email, password):
    """Logs in a user and updates session state."""
    try:
        user = supabase.auth.sign_in_with_password({"email": email, "password": password})
        st.session_state['authenticated'] = True
        st.session_state['user'] = user
        st.success("Logged in successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"Login failed: {e}")

def logout_user():
    """Logs out the user and clears session state."""
    if 'authenticated' in st.session_state:
        del st.session_state['authenticated']
    if 'user' in st.session_state:
        del st.session_state['user']
    st.rerun()

def show_login_form():
    """Displays the login form."""
    st.header("Login")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if email and password:
                login_user(email, password)
            else:
                st.warning("Please enter both email and password.")
