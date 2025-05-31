# components/ai_analysis.py
import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    get_neo4j_driver, 
    get_nearest_aggregate_similarities, 
    fetch_financial_details_for_companies,
    format_value # For displaying financial values nicely
)
# --- Helper Functions (Keep or refine from previous versions) ---
def format_value_report(value, currency="", is_percent=False, is_ratio=False, is_cagr=False, decimals=2):
    if value is None or pd.isna(value): return "N/A"
    if is_percent or is_cagr: return f"{value:.{decimals}f}%"
    if is_ratio: return f"{value:.{decimals}f}"
    if isinstance(value, (int, float)):
        num_str = ""
        if abs(value) >= 1_000_000_000_000: num_str = f"{value / 1_000_000_000_000:.{decimals}f}T"
        elif abs(value) >= 1_000_000_000: num_str = f"{value / 1_000_000_000:.{decimals}f}B"
        elif abs(value) >= 1_000_000: num_str = f"{value / 1_000_000:.{decimals}f}M"
        elif abs(value) >= 1_000: num_str = f"{value / 1_000:.{decimals}f}K"
        else: num_str = f"{value:,.{decimals}f}"
        return f"{currency}{num_str}" if currency and num_str else num_str
    return str(value)

def classification_indicator(classification_text, display_type="icon_text"):
    if not classification_text: return ""
    color_map = {"Positive": "#28a745", "Neutral": "#ffc107", "Negative": "#dc3545"}
    icon_map = {"Positive": "📈", "Neutral": "📊", "Negative": "📉"} # Alternative: "✓", "–", "✗"
    
    color = color_map.get(classification_text, "#6c757d")
    icon = icon_map.get(classification_text, "❓")

    if display_type == "icon_text":
        return f"<span style='color:{color}; font-weight:bold;'>{icon} {classification_text}</span>"
    elif display_type == "text_color":
        return f"<span style='color:{color};'>{classification_text}</span>"
    elif display_type == "icon_only":
        return f"<span style='color:{color}; font-size: 1.2em;'>{icon}</span>"
    return classification_text # Fallback

def create_trend_chart_report(df_values, y_col, title, y_axis_title, is_percent=False, currency_symbol="$"):
    if df_values.empty or y_col not in df_values.columns:
        # st.caption(f"{title} data not available.") # Avoid cluttering with N/A for charts
        return None
    
    fig = px.line(df_values, x='period', y=y_col, markers=True, line_shape="spline")
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        yaxis_title=y_axis_title,
        xaxis_title=None,
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=12, color="#333")
    )
    if is_percent:
        fig.update_yaxes(ticksuffix="%")
    else:
        fig.update_yaxes(tickprefix=currency_symbol, tickformat=".2s") # SI format (B, M, K)
    
    chart_config = {'displayModeBar': False}
    return fig, chart_config

def create_pie_chart_report(data_list, names_col, values_col, title):
    if not data_list: return None
    df = pd.DataFrame(data_list)
    fig = px.pie(df, names=names_col, values=values_col, hole=0.4)
    fig.update_traces(textposition='outside', textinfo='percent+label', hoverinfo='label+percent+value')
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        legend_title_text="Segments",
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=12, color="#333")
    )
    chart_config = {'displayModeBar': False}
    return fig, chart_config


@st.cache_data(ttl="1h")
def load_ai_analysis_data_report(symbol_for_file: str = "AAPL"):
    """Load pre-generated AI analysis JSON for a ticker, if available.
    Looks in a `sample_data/` directory at project root for files named `<SYMBOL>_1.json`.
    If the file is missing it returns `None` so the caller can decide to generate the report
    on‑the‑fly or show a friendly message.
    """
    from pathlib import Path
    # Determine project root (= two levels up from this file)
    project_root = Path(__file__).resolve().parents[1]
    sample_dir = project_root / "sample_data"
    file_path = sample_dir / f"{symbol_for_file.upper()}_1.json"
    if file_path.exists():
        try:
            with file_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        except (ValueError, json.JSONDecodeError):
            st.warning(f"❗ The analysis file for {symbol_for_file.upper()} is corrupted.")
            return None
    else:
        # Silent fallback: let caller decide next steps
        return None
    except FileNotFoundError:
        st.error(f"Error: Analysis file for {symbol_for_file} ({file_path}) not found.")
        return None
    except Exception as e:
        st.error(f"Error loading analysis file {file_path}: {e}")
        return None

# --- Main AI Analysis Tab Content ---
def ai_analysis_tab_content(selected_symbol_from_app):
    st.markdown("""
    <style>
        .report-header {text-align: center; margin-bottom: 30px;}
        .report-section {margin-bottom: 40px; padding: 20px; background-color: #f9f9f9; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);}
        .report-section h2 { color: #004085; border-bottom: 2px solid #004085; padding-bottom: 10px; margin-bottom:20px; font-size: 1.8em;}
        .report-section h3 { color: #333; margin-top: 20px; margin-bottom:10px; font-size: 1.4em;}
        .metric-card-ai {text-align: center; padding: 15px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: #fff;}
        .metric-card-ai .label {font-size: 0.9em; color: #555; margin-bottom: 5px;}
        .metric-card-ai .value {font-size: 1.5em; font-weight: bold; color: #007bff;}
        .explanation-text {font-style: italic; color: #555; margin-top: 5px; margin-bottom:15px; line-height: 1.6;}
        .list-item { margin-bottom: 8px; }
        .key-item { margin-bottom: 10px; }
        .positive-item { color: #28a745; font-weight: bold;} /* Green */
        .negative-item { color: #dc3545; font-weight: bold;} /* Red */
        .neutral-item { color: #6c757d; } /* Grey */
    </style>
    """, unsafe_allow_html=True)

    json_data = load_ai_analysis_data_report(selected_symbol_from_app)
    if not json_data:
        if selected_symbol_from_app != "AAPL": # Default if primary fails
            st.info(f"AI analysis for {selected_symbol_from_app} not found. Displaying AAPL as an example.")
            json_data = load_ai_analysis_data_report("AAPL")
            if not json_data: return # Stop if AAPL also fails
        else: return


    meta = json_data.get("metadata", {})
    analysis = json_data.get("analysis", {})
    currency = meta.get("currency", "$")

    # --- Report Header ---
    st.markdown(f"<div class='report-header'><h1>AI Financial Analysis: {meta.get('company_name', selected_symbol_from_app)} ({meta.get('ticker')})</h1>"
                f"<h4>As of {meta.get('as_of_date', 'N/A')} | Generated: {meta.get('analysis_generation_date', 'N/A')}</h4></div>", 
                unsafe_allow_html=True)

    # --- Investment Thesis Summary (Executive Summary) ---
    inv_thesis = analysis.get("investment_thesis_summary", {})
    if inv_thesis:
        st.markdown("<div class='report-section'><h2>Executive Summary: Investment Thesis</h2>", unsafe_allow_html=True)
        rec_color = {"Buy": "#28a745", "Hold": "#ffc107", "Sell": "#dc3545"}.get(inv_thesis.get("overall_recommendation", "N/A"), "#6c757d")
        
        cols_thesis = st.columns([1,1,1])
        cols_thesis[0].markdown(f"<div class='metric-card-ai'><div class='label'>Overall Recommendation</div><div class='value' style='color:{rec_color};'>{inv_thesis.get('overall_recommendation', 'N/A').upper()}</div></div>", unsafe_allow_html=True)
        cols_thesis[1].metric("Price Target", format_value_report(inv_thesis.get('price_target'), currency=currency, decimals=2))
        cols_thesis[2].metric("Time Horizon", inv_thesis.get('time_horizon', 'N/A'))
        
        st.markdown(f"<p class='explanation-text'><strong>Final Justification:</strong> {inv_thesis.get('final_justification', 'N/A')} {classification_indicator(inv_thesis.get('final_justification_classification'))}</p>", unsafe_allow_html=True)

        cols_pos_neg = st.columns(2)
        with cols_pos_neg[0]:
            st.markdown("<h3>Key Positives</h3>", unsafe_allow_html=True)
            for item in inv_thesis.get("key_investment_positives", []):
                st.markdown(f"<div class='key-item positive-item'>✓ {item.get('item_text', '')}</div>", unsafe_allow_html=True)
        with cols_pos_neg[1]:
            st.markdown("<h3>Key Risks</h3>", unsafe_allow_html=True)
            for item in inv_thesis.get("key_investment_risks", []):
                st.markdown(f"<div class='key-item negative-item'>✗ {item.get('item_text', '')}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True) # Close report-section


    # --- Financial Performance ---
    perf = analysis.get("financial_performance", {})
    if perf:
        st.markdown("<div class='report-section'><h2>Financial Performance Review</h2>", unsafe_allow_html=True)
        
        # Revenues & Margins
        st.markdown("<h3>Revenue & Profitability Trends</h3>", unsafe_allow_html=True)
        rev_data = perf.get("revenues", {}).get("values")
        gm_data = perf.get("gross_margin", {}).get("values")
        nm_data = perf.get("net_margin", {}).get("values")

        chart_cols1 = st.columns(3 if rev_data and gm_data and nm_data else 1) # Adjust columns based on available data
        
        if rev_data:
            with chart_cols1[0]:
                fig, config = create_trend_chart_report(pd.DataFrame(rev_data), "value", "Annual Revenues", "Value", currency_symbol=currency)
                if fig: st.plotly_chart(fig, use_container_width=True, config=config)
                st.markdown(f"<p class='explanation-text'>{perf.get('revenues', {}).get('explanation', '')} {classification_indicator(perf.get('revenues', {}).get('classification'))}</p>", unsafe_allow_html=True)

        if gm_data:
            with chart_cols1[1 % len(chart_cols1)]: # Cycle through columns
                fig, config = create_trend_chart_report(pd.DataFrame(gm_data), "value_percent", "Gross Margin %", "%", is_percent=True)
                if fig: st.plotly_chart(fig, use_container_width=True, config=config)
                st.markdown(f"<p class='explanation-text'>{perf.get('gross_margin', {}).get('explanation', '')} {classification_indicator(perf.get('gross_margin', {}).get('classification'))}</p>", unsafe_allow_html=True)

        if nm_data:
            with chart_cols1[2 % len(chart_cols1)]:
                fig, config = create_trend_chart_report(pd.DataFrame(nm_data), "value_percent", "Net Margin %", "%", is_percent=True)
                if fig: st.plotly_chart(fig, use_container_width=True, config=config)
                st.markdown(f"<p class='explanation-text'>{perf.get('net_margin', {}).get('explanation', '')} {classification_indicator(perf.get('net_margin', {}).get('classification'))}</p>", unsafe_allow_html=True)

        # Key Profitability Metrics (EBITDA, Operating Income, Net Income)
        st.markdown("<h3>Core Profitability Metrics</h3>", unsafe_allow_html=True)
        profit_metrics = perf.get("profitability_metrics", {})
        ebitda_data = profit_metrics.get("ebitda", {}).get("values")
        op_income_data = profit_metrics.get("operating_income", {}).get("values")
        net_income_data = profit_metrics.get("net_income", {}).get("values")

        chart_cols2 = st.columns(3 if ebitda_data and op_income_data and net_income_data else 1)
        if ebitda_data:
            with chart_cols2[0]:
                fig, config = create_trend_chart_report(pd.DataFrame(ebitda_data), "value", "EBITDA", "Value", currency_symbol=currency)
                if fig: st.plotly_chart(fig, use_container_width=True, config=config)
                st.markdown(f"<p class='explanation-text'>{profit_metrics.get('ebitda', {}).get('explanation', '')} {classification_indicator(profit_metrics.get('ebitda', {}).get('classification'))}</p>", unsafe_allow_html=True)
        # ... Add similar blocks for Operating Income and Net Income in chart_cols2[1] and chart_cols2[2] ...
        if op_income_data:
             with chart_cols2[1 % len(chart_cols2)]:
                fig, config = create_trend_chart_report(pd.DataFrame(op_income_data), "value", "Operating Income", "Value", currency_symbol=currency)
                if fig: st.plotly_chart(fig, use_container_width=True, config=config)
                st.markdown(f"<p class='explanation-text'>{profit_metrics.get('operating_income', {}).get('explanation', '')} {classification_indicator(profit_metrics.get('operating_income', {}).get('classification'))}</p>", unsafe_allow_html=True)
        if net_income_data:
             with chart_cols2[2 % len(chart_cols2)]:
                fig, config = create_trend_chart_report(pd.DataFrame(net_income_data), "value", "Net Income", "Value", currency_symbol=currency)
                if fig: st.plotly_chart(fig, use_container_width=True, config=config)
                st.markdown(f"<p class='explanation-text'>{profit_metrics.get('net_income', {}).get('explanation', '')} {classification_indicator(profit_metrics.get('net_income', {}).get('classification'))}</p>", unsafe_allow_html=True)


        # CAGR Cards
        st.markdown("<h3>Key Growth Rates (CAGR)</h3>", unsafe_allow_html=True)
        cagr_cols = st.columns(3) # Revenue, Net Income, FCF
        rev_cagr = perf.get("revenue_cagr_calculated", {})
        ni_cagr = profit_metrics.get("net_income_cagr_calculated", {})
        fcf_cagr = perf.get("cash_generation", {}).get("fcf_cagr_calculated", {})

        if rev_cagr: cagr_cols[0].metric("Revenue CAGR (5Y)", format_value_report(rev_cagr.get("rate_percent"), is_cagr=True), help=rev_cagr.get("calculation_note"))
        if ni_cagr: cagr_cols[1].metric("Net Income CAGR (5Y)", format_value_report(ni_cagr.get("rate_percent"), is_cagr=True), help=ni_cagr.get("calculation_note"))
        if fcf_cagr: cagr_cols[2].metric("FCF CAGR (5Y)", format_value_report(fcf_cagr.get("rate_percent"), is_cagr=True), help=fcf_cagr.get("calculation_note"))
        
        st.markdown("</div>", unsafe_allow_html=True) # Close Financial Performance section


    # --- Revenue Segmentation ---
    rev_seg = analysis.get("revenue_segmentation", {})
    if rev_seg:
        st.markdown("<div class='report-section'><h2>Revenue Segmentation (FY {})</h2>".format(rev_seg.get("latest_fiscal_year", "")), unsafe_allow_html=True)
        seg_cols = st.columns(2)
        with seg_cols[0]:
            geo_data = rev_seg.get("geographic_breakdown")
            if geo_data:
                fig, config = create_pie_chart_report(geo_data, "region", "revenue_percentage", "Geographic Revenue Mix")
                if fig: st.plotly_chart(fig, use_container_width=True, config=config)
        with seg_cols[1]:
            prod_data = rev_seg.get("product_breakdown")
            if prod_data:
                fig, config = create_pie_chart_report(prod_data, "segment_name", "revenue_percentage", "Product Revenue Mix")
                if fig: st.plotly_chart(fig, use_container_width=True, config=config)
        
        if rev_seg.get("segmentation_trends_commentary"):
            st.markdown(f"<p class='explanation-text'>{rev_seg['segmentation_trends_commentary']} {classification_indicator(rev_seg.get('segmentation_trends_commentary_classification'))}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Valuation Summary ---
    val_sum = analysis.get("valuation_analysis", {}).get("valuation_summary", {})
    if val_sum:
        st.markdown("<div class='report-section'><h2>Valuation Snapshot</h2>", unsafe_allow_html=True)
        val_cols = st.columns(4)
        val_cols[0].metric("Current Price", format_value_report(meta.get('last_price'), currency=currency, decimals=2))
        val_cols[1].metric("Fair Value (Base)", format_value_report(val_sum.get('fair_value_estimate_base'), currency=currency, decimals=2))
        val_cols[2].metric("vs. Fair Value", val_sum.get('current_price_vs_fair_value', 'N/A'))
        
        fv_low = val_sum.get('fair_value_estimate_low')
        fv_high = val_sum.get('fair_value_estimate_high')
        if fv_low is not None and fv_high is not None:
            val_cols[3].markdown(f"<div class='metric-card-ai'><div class='label'>Fair Value Range</div><div class='value'>{format_value_report(fv_low, currency=currency, decimals=2)} - {format_value_report(fv_high, currency=currency, decimals=2)}</div></div>", unsafe_allow_html=True)

        st.markdown(f"<p class='explanation-text'>{val_sum.get('summary_commentary', '')} {classification_indicator(val_sum.get('summary_commentary_classification'))}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Risk Factors ---
    risks = analysis.get("risk_factors", {})
    if risks:
        st.markdown("<div class='report-section'><h2>Key Risk Factors</h2>", unsafe_allow_html=True)
        risk_cols = st.columns(2) # Display risks in two columns
        col_idx = 0
        for risk_category, risk_items_list in risks.items():
            if risk_items_list and isinstance(risk_items_list, list):
                with risk_cols[col_idx % 2]:
                    st.markdown(f"<h3>{risk_category.replace('_risks','').replace('_',' ').title()}</h3>", unsafe_allow_html=True)
                    for item in risk_items_list:
                        st.markdown(f"<li class='list-item'>{item.get('item_text', '')} {classification_indicator(item.get('classification'), 'icon_only')}</li>", unsafe_allow_html=True)
                col_idx +=1
        st.markdown("</div>", unsafe_allow_html=True)


    st.markdown("---")
    st.caption("AI-generated analysis for informational purposes. Conduct your own research before making investment decisions.")


# Standalone testing
if __name__ == '__main__':
    import os # Make sure os is imported for standalone
    st.set_page_config(layout="wide")
    # Ensure AAPL_1.json is in the project root when running this directly or adjust path in load_ai_analysis_data_report
    ai_analysis_tab_content("AAPL") 