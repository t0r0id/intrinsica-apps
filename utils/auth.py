"""
Authentication utilities for Intrinsica Apps
"""

import streamlit as st

def check_authentication():
    """Check if user is authenticated"""
    return st.session_state.get('authenticated', False)

def login_page():
    """Display login page"""
    st.title("🔐 Intrinsica Apps - Login")
    st.markdown("Please enter your credentials to access the application suite.")
    
    with st.form("login_form"):
        st.subheader("Authentication Required")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            if username == "intrinsica_admin" and password == "Market-Report83!":
                st.session_state['authenticated'] = True
                st.session_state['username'] = username
                st.success("✅ Login successful! Redirecting...")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
                st.warning("Please check your credentials and try again.")

def logout_sidebar():
    """Add logout option to sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"👤 **Logged in as:** {st.session_state.get('username', 'Unknown')}")
        if st.button("🚪 Logout"):
            st.session_state['authenticated'] = False
            st.session_state.pop('username', None)
            st.rerun()