#!/usr/bin/env python3
"""
Intrinsica Apps - Main Application
Multi-page Streamlit application combining Competitor Identification and Radar Tracker
"""

import streamlit as st
from utils.auth import check_authentication, login_page, logout_sidebar

# Page configuration
st.set_page_config(
    page_title="Intrinsica Apps",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application entry point"""
    
    # Check authentication first
    if not check_authentication():
        login_page()
        return
    
    # Add logout option to sidebar
    logout_sidebar()
    
    # Main content
    st.title("🚀 Intrinsica Apps")
    st.markdown("Welcome to the Intrinsica application suite!")
    
    st.markdown("""
    ### Available Applications:
    
    **🎯 Competitor Identification**
    - Find competitors using hybrid search (BM25 + Semantic Similarity)
    - Analyze business unit descriptions and company comparisons
    - Interactive visualizations and detailed competitor insights
    
    **📊 Radar Tracker** 
    - Track key issues and hypotheses for companies
    - Search and filter through conference insights
    - View radar reports and development timelines
    
    ### Getting Started
    Use the sidebar navigation to switch between applications.
    """)
    
    # Instructions
    with st.expander("ℹ️ Instructions"):
        st.markdown("""
        1. **Select an application** from the sidebar navigation
        2. **Configure your search criteria** using the sidebar controls
        3. **Explore the results** using the interactive tabs and visualizations
        4. **Click links** to view detailed reports in the Intrinsica platform
        """)

if __name__ == "__main__":
    main()