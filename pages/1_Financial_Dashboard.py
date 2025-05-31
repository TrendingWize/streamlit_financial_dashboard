import streamlit as st
st.set_page_config(page_title="Financial Dashboard", layout="wide", page_icon="📈")
from components.nav_bar import render_nav_bar
render_nav_bar()

# heavy component imports AFTER the page-config / nav-bar
from components.company_profile import company_profile_tab_content
from components.income_statement import income_statement_tab_content
from components.balance_sheet import balance_sheet_tab_content
from components.cash_flow import cash_flow_tab_content
from components.similar_companies_tab import similar_companies_tab_content
from components.sec_filing_analysis import sec_filing_analysis_tab_content
from components.ai_analysis_tab  import ai_analysis_tab_content
from components.ask_ai_tab import ask_ai_tab_content

#links = [
#    st.page_link("pages/Home.py", label="📊 Dashboard", icon=None),
#    st.page_link("pages/SEC_Filing_Analysis.py", label="📑 SEC Filing Analysis", icon=None),
#]
#st.markdown('<div class="top-links">' + "".join(links) + '</div>', unsafe_allow_html=True)

st.title("📊 Financial Statements & AI Insights Dashboard")

if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = "AAPL"

with st.form(key="symbol_search_form_dashboard_main"): # Renamed key
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        current_input_symbol = st.text_input(
            "Search Company Symbol:", 
            value=st.session_state.selected_symbol,
            key="symbol_search_input_dashboard_main", 
            label_visibility="collapsed",
            placeholder="Enter Symbol (e.g., AAPL, MSFT)"
        ).upper()
    with search_col2:
        search_submitted = st.form_submit_button("🔍 Search", use_container_width=True)

    if search_submitted:
        if current_input_symbol:
            st.session_state.selected_symbol = current_input_symbol
        else:
            st.warning("Please enter a symbol.")
st.markdown("---")

symbol_to_display = st.session_state.selected_symbol

tabs = st.tabs(
    [
        "👤 Profile",
        "💰 Income Stmt",
        "🏦 Balance Sheet",
        "🌊 Cash Flow",
        "🔗 Similar Cos",
        "📑 SEC Filing Analysis",
        "🤖 AI Analysis",      # NEW
        "💬 Ask AI",           # NEW
    ]
)

(
    tab_profile,
    tab_is,
    tab_bs,
    tab_cf,
    tab_sc,
    tab_sec,
    tab_ai,                  # NEW
    tab_ask,                 # NEW
) = tabs

with tab_profile:
    if symbol_to_display:
        company_profile_tab_content(symbol_to_display)
    else:
        st.info("Please enter a symbol and click Search.")

with tab_is:
    if symbol_to_display:
        income_statement_tab_content(symbol_to_display)
    else:
        st.info("Please enter a symbol and click Search.")

with tab_bs:
    if symbol_to_display:
        balance_sheet_tab_content(symbol_to_display)
    else:
        st.info("Please enter a symbol and click Search.")

with tab_cf:
    if symbol_to_display:
        cash_flow_tab_content(symbol_to_display)
    else:
        st.info("Please enter a symbol and click Search.")

with tab_sc:
    similar_companies_tab_content()

with tab_sec:
    sec_filing_analysis_tab_content()          # runs the generator / viewer
with tab_ai:
    ai_analysis_tab_content()             # handles its own symbol input
with tab_ask:
    ask_ai_tab_content()            