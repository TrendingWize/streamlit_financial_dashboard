# pages/sec_filing_analysis.py
import streamlit as st

# ── Page configuration (must be first Streamlit command) ─────────
# FIRST command on the page
st.set_page_config(page_title="Financial Insights - Home",
                   layout="wide", page_icon="🏠")

from components.nav_bar import render_nav_bar
render_nav_bar()

# ── Top navigation bar (shared across pages) ─────────────────────
from components.nav_bar import render_nav_bar
render_nav_bar()

# ── Page body: run the filing-analysis component ─────────────────
from components.sec_filing_analysis import sec_filing_analysis_tab_content
sec_filing_analysis_tab_content()
