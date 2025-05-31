import streamlit as st

# Import navigation, page content components, and global styles
#from components.navigation import R_custom_navigation
from components.company_profile import company_profile_tab_content
from components.income_statement import income_statement_tab_content
from components.balance_sheet import balance_sheet_tab_content
from components.cash_flow import cash_flow_tab_content
# from styles import load_global_css # Global CSS is loaded once in app.py

# --- Page Configuration ---
# Must be the first Streamlit command in the page file.
st.set_page_config(
    page_title="Financial Dashboard",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="collapsed"  # Sidebar can be used for filters or options
)

# load_global_css() # Global CSS is loaded once in app.py, so not needed here.

# --- Display Custom Top Navigation ---
# This renders the consistent navigation bar defined in components/navigation.py.
#R_custom_navigation()

# --- Main Title for this page ---
st.title("📊 Financial Statements Dashboard")
st.markdown("Analyze key financial statements, company profiles, and cash flow dynamics.") # Added a subtitle

# --- Ensure global_selected_symbol is initialized ---
# This is a fallback if app.py didn't initialize it or if the page is run directly.
if 'global_selected_symbol' not in st.session_state:
    st.session_state.global_selected_symbol = "AAPL" # Default symbol

# --- Search Box and Button for Company Symbol ---
# This section allows users to input a company symbol to fetch data for.
with st.container():
    # Using columns for a cleaner layout of the search input and button.
    search_col1, search_col2 = st.columns([3, 1]) # Input takes more space

    with search_col1:
        current_input_symbol = st.text_input(
            label="Enter Company Symbol (e.g., AAPL, MSFT, NVDA):",
            value=st.session_state.global_selected_symbol, # Uses the global symbol
            key="symbol_search_input_dashboard_page", # Unique key for this input
            label_visibility="collapsed", # Hides the label, placeholder is used instead
            placeholder="Enter Symbol (e.g., AAPL)"
        ).upper() # Convert input to uppercase for consistency

    with search_col2:
        if st.button("Search Company", key="search_button_dashboard_page", use_container_width=True):
            if current_input_symbol:
                st.session_state.global_selected_symbol = current_input_symbol # Update the global symbol
                # Streamlit will rerun, and components will use the new symbol.
            else:
                st.warning("Please enter a company symbol to search.")

# Visual separator
st.markdown("---")

# --- Tabs for Different Financial Statements and Profile ---
# Uses the globally selected symbol to display content in tabs.
symbol_to_display = st.session_state.global_selected_symbol

# Define the tabs
tab_profile, tab_is, tab_bs, tab_cf = st.tabs([
    "👤 Company Profile",
    "💰 Income Statement",
    "🏦 Balance Sheet",
    "🌊 Cash Flow Statement"
])

# Content for "Company Profile" Tab
with tab_profile:
    if symbol_to_display:
        st.header(f"👤 Company Profile: {symbol_to_display}")
        company_profile_tab_content(symbol_to_display)
    else:
        # This case should ideally not be hit if global_selected_symbol has a default
        st.info("Please enter a company symbol in the search bar above.")

# Content for "Income Statement" Tab
with tab_is:
    if symbol_to_display:
        st.header(f"💰 Income Statement: {symbol_to_display}")
        income_statement_tab_content(symbol_to_display)
    else:
        st.info("Please enter a company symbol to view the Income Statement.")

# Content for "Balance Sheet" Tab
with tab_bs:
    if symbol_to_display:
        st.header(f"🏦 Balance Sheet: {symbol_to_display}")
        balance_sheet_tab_content(symbol_to_display)
    else:
        st.info("Please enter a company symbol to view the Balance Sheet.")

# Content for "Cash Flow Statement" Tab
with tab_cf:
    if symbol_to_display:
        st.header(f"🌊 Cash Flow Statement: {symbol_to_display}")
        cash_flow_tab_content(symbol_to_display)
    else:
        st.info("Please enter a company symbol to view the Cash Flow Statement.")