import streamlit as st
from supabase import create_client

# --- SECURELY INITIALIZE SUPABASE ---
try:
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    supabase = create_client(supabase_url, supabase_key)
except (KeyError, AttributeError):
    # This helps avoid errors during local development if secrets aren't set up
    st.error("Supabase credentials are not set in st.secrets. Please add them.")
    st.stop()

def login_user(email, password):
    """Logs in a user and updates the session state."""
    try:
        user_response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        st.session_state['authenticated'] = True
        st.session_state['user_email'] = user_response.user.email
        st.success("Logged in successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"Login failed. Please check your credentials.")

def logout_user():
    """Logs out the user and clears the session state."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state['authenticated'] = False
    st.rerun()

def show_login_form():
    """Displays the login form."""
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