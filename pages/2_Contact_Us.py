# pages/2_Contact_Us.py
import streamlit as st

st.set_page_config(
    page_title="Contact Us",
    layout="wide",
    page_icon="✉️",
    initial_sidebar_state="collapsed"
)

from components.nav_bar import render_nav_bar
render_nav_bar()
# --- Display Custom Top Navigation (Visual Consistency) ---
st.markdown("""
<style>
    .top-nav-container { background-color: #f0f2f6; padding: 10px 0px; border-bottom: 1px solid #cccccc; margin-bottom: 20px; text-align: center; }
    .top-nav-container a { margin: 0 15px; text-decoration: none; color: #333; font-weight: 500; font-size: 1.1em; }
    .top-nav-container a:hover { color: #007bff; }
</style>
<div class="top-nav-container">
    <a href="/" target="_self">Home</a>
    <a href="/Financial_Dashboard" target="_self">Financial Dashboard</a>
    <a href="/Contact_Us" target="_self">Contact Us</a>
</div>
""", unsafe_allow_html=True)
# selected_page = top_navigation_bar() # If using streamlit-option-menu


# --- Contact Us Page Content ---
st.title("Contact Us")
st.markdown("This page is currently under construction. Please check back later for contact information.")

st.subheader("Get in Touch (Placeholder)")
st.write("Email: contact@financialinsights.example.com (Placeholder)")
st.write("Phone: +1 (555) 123-4567 (Placeholder)")

st.markdown("---")
st.info("We appreciate your interest!")