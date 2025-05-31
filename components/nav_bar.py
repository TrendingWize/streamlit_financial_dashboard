import streamlit as st

# one-shot CSS
st.markdown(
    """
    <style>
    .nav-bar {display:flex;gap:1.5rem;padding:8px 12px;margin-bottom:16px;
              background:#F2F6FF;border-bottom:1px solid #E0E6F0;}
    </style>
    """,
    unsafe_allow_html=True,
)

def render_nav_bar() -> None:
    """Call once at the very top of **every page** (after st.set_page_config)."""
    # flex container for links
    st.markdown('<div class="nav-bar">', unsafe_allow_html=True)
    st.page_link("app.py",                        label="🏠 Home")
    st.page_link("pages/1_Financial_Dashboard.py", label="📊 Financial Dashboard")
    st.page_link("pages/2_Contact_Us.py",          label="✉️ Contact Us")
    st.markdown("</div>", unsafe_allow_html=True)
