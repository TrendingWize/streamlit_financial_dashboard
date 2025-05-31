# components/ai_analysis_tab.py
# Python Standard Library Imports
import json
import os # Ensure os is imported here for sys.path logic
import sys # Ensure sys is imported here for sys.path logic
from typing import Any, List, Dict 

# Third-party imports
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Add project root to sys.path (if analysis_pipeline.py is in project root) ---
# This should be done ONCE at the module level
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from analysis_pipeline import generate_ai_report, initialize_analysis_resources
    PIPELINE_IMPORTED_SUCCESSFULLY = True
except ImportError as e_pipeline:
    PIPELINE_IMPORTED_SUCCESSFULLY = False
    ERROR_PIPELINE_IMPORT = str(e_pipeline)

# Local utility imports (after sys.path modification if utils.py is also in project root)
try:
    from utils import get_neo4j_driver
    UTILS_IMPORTED_SUCCESSFULLY = True
except ImportError as e_utils:
    UTILS_IMPORTED_SUCCESSFULLY = False
    ERROR_UTILS_IMPORT = str(e_utils)
    
    
# --- Helper Functions (should be defined before display_ai_analysis_dashboard) ---
def ai_tab_css(): # Make sure this is defined
    st.markdown("""
    <style>
        .metric-card-ai {
            background-color: #FFFFFF; border: 1px solid #e0e7ff; border-radius: 10px;
            padding: 18px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            text-align: center; margin-bottom: 15px; height: 120px; 
            display: flex; flex-direction: column; justify-content: center;
        }
        .metric-card-ai .stMetricLabel { 
            font-size: 0.95em; color: #4A5568; font-weight: 500; margin-bottom: 8px; line-height: 1.2;
        }
        .metric-card-ai .stMetricValue { 
            font-size: 1.8em; color: #2563eb; font-weight: 700; line-height: 1.1;
        }
        .section-header-ai {
            font-size: 1.75em; font-weight: 700; color: #1e3a8a;
            margin-top: 30px; margin-bottom: 20px;
            border-bottom: 3px solid #3b82f6; padding-bottom: 8px;
        }
        .subsection-header-ai {
            font-size: 1.3em; font-weight: 600; color: #1d4ed8;
            margin-top: 20px; margin-bottom: 12px;
        }
        .explanation-box {
            background-color: #eff6ff; padding: 12px 15px; border-radius: 8px; 
            color: #1e40af; border-left: 4px solid #3b82f6;
            margin-bottom: 15px; font-size: 0.95em;
        }
        .sentiment-positive { color: #16a34a; font-weight: bold; }
        .sentiment-negative { color: #dc2626; font-weight: bold; }
        .sentiment-neutral { color: #d97706; font-weight: bold; }
        .kpi-item { margin-bottom: 8px; padding-left: 5px;}
        .kpi-item strong { color: #374151; }
        .stDataFrame table { font-size: 0.9em; }
        .stDataFrame th { background-color: #eef2ff; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

def get_sentiment_html_enhanced(classification_text, text_prefix=""):
    if not classification_text or not isinstance(classification_text, str): return ""
    icon = ""; style_class = ""
    cl_lower = classification_text.lower()
    if any(s in cl_lower for s in ["positive", "bullish", "outperform", "favorable"]): icon, style_class = "✅", "sentiment-positive"
    elif any(s in cl_lower for s in ["negative", "bearish", "underperform", "unfavorable", "risks"]): icon, style_class = "❌", "sentiment-negative"
    elif any(s in cl_lower for s in ["neutral", "mixed", "cautiously"]): icon, style_class = "⚠️", "sentiment-neutral"
    return f"<span class='{style_class}'>{icon} {text_prefix}{classification_text}</span>" if icon else f"<span>({classification_text})</span>"

def format_currency_short(value):
    if pd.isna(value) or not isinstance(value, (int, float)): return "N/A"
    if abs(value) >= 1_000_000_000_000: return f"${value/1_000_000_000_000:.2f}T"
    if abs(value) >= 1_000_000_000: return f"${value/1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000: return f"${value/1_000_000:.2f}M"
    if abs(value) >= 1_000: return f"${value/1_000:.1f}K"
    return f"${value:,.2f}"

def format_generic_value(value, key_context=""):
    if pd.isna(value) or value is None: return "N/A" # Added explicit None check
    val_str = str(value)
    key_lower = key_context.lower()
    if isinstance(value, (int, float)):
        if any(s in key_lower for s in ["price", "value", "amount", "capitalization", "ebitda", "income", "debt", "equity", "assets", "cash"]) and \
           not any(s in key_lower for s in ["percent", "rate", "margin", "yield", "ratio", "cagr", "beta", "multiple", "growth", "coverage", "turnover", "cycle"]): # More specific exclusion
            return format_currency_short(value)
        elif any(s in key_lower for s in ["percent", "rate", "margin", "yield", "cagr", "growth"]):
            return f"{value:.2f}%"
        elif isinstance(value, float):
            return f"{value:.2f}" # Default for other floats
    return val_str # Return original string if not a number or no specific format matched

# --- Recursive Section Renderer (MUST BE DEFINED BEFORE display_ai_analysis_dashboard) ---
def render_generic_section(data_to_render: Any, current_path_keys: List[str]):
    if data_to_render is None: return # Added None check

    if isinstance(data_to_render, list):
        if not data_to_render:
            st.markdown(f"<div class='kpi-item'>  •  No items listed.</div>", unsafe_allow_html=True)
            return
        
        # Special handling for peers_comparison.comparison_table
        if current_path_keys and current_path_keys[-1] == "comparison_table" and all(isinstance(item, dict) for item in data_to_render):
            try:
                df_comparison = pd.DataFrame(data_to_render)
                st.dataframe(df_comparison, use_container_width=True, hide_index=True)
            except Exception as e:
                st.caption(f"Could not render comparison table: {e}")
                for item_dict in data_to_render: # Fallback to list rendering
                    st.markdown(f"<div class='kpi-item'>▪️ {json.dumps(item_dict)}</div>", unsafe_allow_html=True)
            return # Handled comparison_table

        for item_dict in data_to_render: # Assuming list of dicts for most other cases
            if isinstance(item_dict, dict):
                summary_parts = []
                item_classification_html = ""
                if "item_text" in item_dict: summary_parts.append(item_dict["item_text"])
                elif "name" in item_dict and "ticker" in item_dict: summary_parts.append(f"{item_dict['name']} ({item_dict.get('ticker', 'N/A')})")
                elif "segment_name" in item_dict: summary_parts.append(f"{item_dict['segment_name']}: {item_dict.get('revenue_percentage', 'N/A')}% ({format_currency_short(item_dict.get('revenue_amount'))})")
                elif "region" in item_dict: summary_parts.append(f"{item_dict['region']}: {item_dict.get('revenue_percentage', 'N/A')}% ({format_currency_short(item_dict.get('revenue_amount'))})")
                else:
                    for k_item, v_item in item_dict.items():
                        if k_item != "classification": summary_parts.append(f"{str(k_item).replace('_',' ').title()}: {format_generic_value(v_item, k_item)}")
                if "classification" in item_dict: item_classification_html = get_sentiment_html_enhanced(item_dict["classification"])
                st.markdown(f"<div class='kpi-item'>▪️ {', '.join(summary_parts)} {item_classification_html}</div>", unsafe_allow_html=True)
            elif isinstance(item_dict, str):
                st.markdown(f"<div class='kpi-item'>▪️ {item_dict}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='kpi-item'>▪️ {str(item_dict)}</div>", unsafe_allow_html=True)
        return

    if isinstance(data_to_render, dict):
        for key, value in data_to_render.items():
            if value == "__PROCESSED__": continue
            display_key = str(key).replace("_", " ").title()
            new_path_keys = current_path_keys + [key]

            if key == "values" and isinstance(value, list) and value and isinstance(value[0], dict) and "period" in value[0]:
                parent_section_title = str(current_path_keys[-1]).replace("_"," ").title() if current_path_keys else "Trend"
                df_values = pd.DataFrame(value)
                possible_value_cols = ['value', 'value_percent', 'growth_percent', 'rate_percent']
                value_col_name = next((col for col in possible_value_cols if col in df_values.columns and pd.api.types.is_numeric_dtype(df_values[col])), None)
                if value_col_name and 'period' in df_values.columns:
                    try:
                        if all(isinstance(p, str) and p.isdigit() and len(p) == 4 for p in df_values['period']):
                            df_values['period_num'] = pd.to_numeric(df_values['period'])
                            df_values = df_values.sort_values(by='period_num').drop(columns=['period_num'])
                    except: pass
                    chart_is_line = len(df_values) >= 5
                    chart_kwargs = {"x": 'period', "y": value_col_name, "title": parent_section_title}
                    if chart_is_line: chart_kwargs["markers"] = True; fig = px.line(df_values, **chart_kwargs)
                    else: chart_kwargs["text_auto"] = True; fig = px.bar(df_values, **chart_kwargs)
                    is_percent_chart_val = any(s in value_col_name.lower() for s in ["percent", "rate", "margin"])
                    fig.update_layout(title_x=0.5, yaxis_title=value_col_name.replace("_"," ").title(), 
                                      yaxis_ticksuffix="%" if is_percent_chart_val else None, 
                                      yaxis_tickformat=None if is_percent_chart_val else "$,.2s",
                                      height=350, margin=dict(t=50,b=20,l=20,r=20), xaxis_title=None)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown(f"**{display_key}:**")
                    st.dataframe(df_values, hide_index=True, use_container_width=True)
                continue

            is_text_block = any(key.endswith(suffix) for suffix in ["explanation", "commentary", "summary", "justification", "overview", "assumptions", "note"])
            if is_text_block and isinstance(value, str):
                classification_html = ""
                base_key_for_class = key
                for suffix in ["explanation", "commentary", "summary", "justification", "overview", "assumptions", "note"]:
                    if key.endswith(suffix): base_key_for_class = key[:-len(suffix)]; break
                possible_class_keys = [f"{base_key_for_class}classification", f"{base_key_for_class}_classification"]
                if "cagr_calculated" in key: possible_class_keys.append(key.replace("_calculated", "_classification"))
                found_class_val = None
                for class_key_attempt in possible_class_keys:
                    if class_key_attempt in data_to_render and data_to_render[class_key_attempt] != "__PROCESSED__":
                        found_class_val = data_to_render[class_key_attempt]
                        data_to_render[class_key_attempt] = "__PROCESSED__"; break 
                if found_class_val: classification_html = get_sentiment_html_enhanced(found_class_val)
                st.markdown(f"**{display_key}:** {classification_html}", unsafe_allow_html=True)
                st.markdown(f"<div class='explanation-box'>{value}</div>", unsafe_allow_html=True)
                continue
            
            if key.endswith("classification") and value == "__PROCESSED__": continue

            # Specific handling for structures like *_cagr_calculated
            if "_cagr_calculated" in key and isinstance(value, dict):
                st.markdown(f"<div class='subsection-header-ai'>{display_key}</div>", unsafe_allow_html=True)
                for k_cagr, v_cagr in value.items():
                    if k_cagr == "calculation_note" and v_cagr:
                         st.markdown(f"<div class='explanation-box'><em>Note:</em> {v_cagr}</div>", unsafe_allow_html=True)
                    elif k_cagr != "classification":
                         st.markdown(f"<div class='kpi-item'><strong>{str(k_cagr).replace('_',' ').title()}:</strong> {format_generic_value(v_cagr, k_cagr)}</div>", unsafe_allow_html=True)
                cagr_classification = value.get("classification")
                if cagr_classification:
                    st.markdown(f"<div class='kpi-item'>{get_sentiment_html_enhanced(cagr_classification, 'Overall Assessment: ')}</div>", unsafe_allow_html=True)
                continue

            if isinstance(value, (dict, list)):
                if value and not (key == "values" and isinstance(value, list)): 
                    st.markdown(f"<div class='subsection-header-ai'>{display_key}</div>", unsafe_allow_html=True)
                render_generic_section(value, new_path_keys)
            else:
                val_str = format_generic_value(value, key)
                st.markdown(f"<div class='kpi-item'><strong>{display_key}:</strong> {val_str}</div>", unsafe_allow_html=True)
        return
    
    st.markdown(f"{str(data_to_render)}")


# --- New Helper for plotting a group of related time series charts ---
def plot_financial_performance_charts_row(
    df_dict: Dict[str, pd.DataFrame], # Dict where key is metric name, value is DataFrame with 'period' and value col
    chart_titles: Dict[str, str],    # Dict mapping metric name to display chart title
    value_col_names: Dict[str, str], # Dict mapping metric name to its value column name (e.g., 'value', 'value_percent')
    is_percent_dict: Dict[str, bool] # Dict mapping metric name to boolean if it's a percentage
):
    active_charts_data = []
    for metric_name, df in df_dict.items():
        if df is not None and not df.empty and 'period' in df.columns and value_col_names.get(metric_name) in df.columns:
            if pd.api.types.is_numeric_dtype(df[value_col_names[metric_name]]):
                 active_charts_data.append({
                     "metric_name": metric_name,
                     "df": df,
                     "title": chart_titles.get(metric_name, metric_name.replace("_"," ").title()),
                     "value_col": value_col_names.get(metric_name),
                     "is_percent": is_percent_dict.get(metric_name, False)
                 })
    
    if not active_charts_data:
        # st.caption("No data for this chart group.") # Can be too noisy
        return

    num_charts_to_plot = len(active_charts_data)
    cols = st.columns(num_charts_to_plot)

    for i, chart_data in enumerate(active_charts_data):
        with cols[i]:
            df_plot = chart_data["df"].copy() # Work with a copy
            value_col = chart_data["value_col"]

            # Ensure periods are sorted (CRUCIAL for line charts to connect correctly)
            try:
                if all(isinstance(p, str) and p.isdigit() and len(p) == 4 for p in df_plot['period']):
                    df_plot['period_num'] = pd.to_numeric(df_plot['period'])
                    df_plot = df_plot.sort_values(by='period_num').drop(columns=['period_num'])
                # Add more sophisticated date parsing/sorting here if 'period' can be "YYYY-MM-DD", "Q1 2023" etc.
                # Example for "YYYY-MM-DD":
                # elif all(isinstance(p, str) and re.match(r'^\d{4}-\d{2}-\d{2}$', p) for p in df_plot['period']):
                #    df_plot['period_dt'] = pd.to_datetime(df_plot['period'])
                #    df_plot = df_plot.sort_values(by='period_dt').drop(columns=['period_dt'])
            except Exception as e_sort:
                st.caption(f"Note: Could not sort periods for {chart_data['title']}. Chart might appear unsorted. Error: {e_sort}")


            chart_is_line = len(df_plot) >= 4 # Adjusted heuristic for line chart
            chart_kwargs = {"x": 'period', "y": value_col, "title": chart_data["title"]}

            if chart_is_line:
                chart_kwargs["markers"] = True
                fig = px.line(df_plot, **chart_kwargs)
            else:
                chart_kwargs["text_auto"] = True
                fig = px.bar(df_plot, **chart_kwargs)
            
            yaxis_title = value_col.replace("_"," ").title()
            yaxis_ticksuffix = "%" if chart_data["is_percent"] else None
            yaxis_tickformat = None if chart_data["is_percent"] else "$,.2s"

            fig.update_layout(title_x=0.5, title_font_size=14, 
                              yaxis_title=yaxis_title, 
                              yaxis_ticksuffix=yaxis_ticksuffix, yaxis_tickformat=yaxis_tickformat,
                              height=320, margin=dict(t=40,b=20,l=10,r=10), xaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

# --- Main Display Function for the Dashboard Content ---
def display_ai_analysis_dashboard(analysis_data: Dict[str, Any], metadata: Dict[str, Any], company_symbol: str):
    ai_tab_css()
    # ... (Dashboard Overview Cards section remains the same) ...
    st.markdown("<div class='section-header-ai'>Dashboard Overview</div>", unsafe_allow_html=True)
    val_summary_data = analysis_data.get("valuation_analysis", {}).get("valuation_summary", {})
    current_price_val = metadata.get("last_price")
    card_metrics_config = [
        {"label": "Fair Value (Base Est.)", "value_path": ["valuation_analysis", "valuation_summary", "fair_value_estimate_base"], "type": "currency"},
        {"label": "Current Price", "value": current_price_val, "type": "currency"},
        {"label": "Recommendation", "value_path": ["investment_thesis_summary", "overall_recommendation"], "type": "text_strong"},
        {"label": "P/E (TTM)", "value_path": ["fundamental_analysis", "valuation_ratios", "pe_ratio_ttm"], "type": "number"},
        {"label": "Revenue CAGR (Hist.)", "value_path": ["growth_prospects", "historical_growth_summary", "revenue_cagr", "rate_percent"], "type": "percent"},
        {"label": "ROE (TTM)", "value_path": ["fundamental_analysis", "profitability_ratios", "roe_ttm"], "type": "percent"},
    ]
    active_cards_list = [] # Logic to populate active_cards_list remains same
    for card_conf in card_metrics_config:
        val = card_conf.get("value") 
        if val is None and "value_path" in card_conf: 
            temp_val = analysis_data
            try:
                for p_key in card_conf["value_path"]: temp_val = temp_val[p_key]
                val = temp_val
            except: val = None
        if val is not None and str(val).strip() != "" and str(val).lower() != 'n/a':
            active_cards_list.append({"label": card_conf["label"], "value": val, "type": card_conf["type"]})
    if active_cards_list: # Card display logic remains same
        cols = st.columns(len(active_cards_list))
        for i, card_info in enumerate(active_cards_list):
            with cols[i]:
                val_str = str(card_info["value"]) 
                if card_info["type"] == "currency" and isinstance(card_info["value"], (int, float)): val_str = f"${card_info['value']:,.2f}"
                elif card_info["type"] == "percent" and isinstance(card_info["value"], (int, float)): val_str = f"{card_info['value']:.1f}%"
                elif card_info["type"] == "number" and isinstance(card_info["value"], (int, float)): val_str = f"{card_info['value']:.2f}"
                elif card_info["type"] == "text_strong": val_str = f"<strong>{card_info['value']}</strong>"
                st.markdown(f"""<div class="metric-card-ai"><div class="stMetricLabel">{card_info['label']}</div><div class="stMetricValue">{val_str}</div></div>""", unsafe_allow_html=True)
    else: st.info("Key overview metrics not available.")


    # --- Financial Performance Section (Modified for grouped charts) ---
    fin_perf_data = analysis_data.get("financial_performance", {})
    if fin_perf_data:
        st.markdown(f"<div class='section-header-ai'>Financial Performance ({company_symbol})</div>", unsafe_allow_html=True)

        # --- Chart Group 1: Key Absolute Values ---
        st.markdown("<div class='subsection-header-ai'>Key Performance Indicators (Absolute Values)</div>", unsafe_allow_html=True)
        abs_metrics_df_dict = {}
        abs_metrics_titles = {}
        abs_metrics_value_cols = {}
        abs_metrics_is_percent = {}

        for metric_key, display_title in [
            ("revenues", "Revenue"), 
            ("profitability_metrics.grossProfit", "Gross Profit"), # Assuming grossProfit is nested like this, or adjust
            ("profitability_metrics.operating_income", "Operating Income"),
            ("profitability_metrics.net_income", "Net Income")
        ]:
            data_path = metric_key.split('.')
            current_data = fin_perf_data
            valid_path = True
            for p_key in data_path:
                if isinstance(current_data, dict) and p_key in current_data:
                    current_data = current_data[p_key]
                else:
                    valid_path = False; break
            
            if valid_path and isinstance(current_data, dict) and current_data.get("values"):
                df = pd.DataFrame(current_data["values"])
                if 'value' in df.columns: # Assuming primary value column is 'value'
                    abs_metrics_df_dict[metric_key] = df
                    abs_metrics_titles[metric_key] = display_title
                    abs_metrics_value_cols[metric_key] = 'value'
                    abs_metrics_is_percent[metric_key] = False
        
        plot_financial_performance_charts_row(abs_metrics_df_dict, abs_metrics_titles, abs_metrics_value_cols, abs_metrics_is_percent)
        
        # Display explanations for this group if available
        for metric_key_orig, _ in [("revenues", ""), ("profitability_metrics.grossProfit", ""), ("profitability_metrics.operating_income", ""), ("profitability_metrics.net_income", "")]:
            data_path = metric_key_orig.split('.')
            current_data_for_expl = fin_perf_data
            valid_path_expl = True
            for p_key in data_path:
                if isinstance(current_data_for_expl, dict) and p_key in current_data_for_expl: current_data_for_expl = current_data_for_expl[p_key]
                else: valid_path_expl = False; break
            if valid_path_expl and isinstance(current_data_for_expl, dict):
                expl = current_data_for_expl.get("explanation")
                classification = current_data_for_expl.get("classification")
                if expl:
                    st.markdown(f"**{metric_key_orig.split('.')[-1].replace('_',' ').title()}:** {get_sentiment_html_enhanced(classification)}", unsafe_allow_html=True)
                    st.markdown(f"<div class='explanation-box'>{expl}</div>", unsafe_allow_html=True)


        # --- Chart Group 2: Key Margins ---
        st.markdown("<div class='subsection-header-ai'>Key Profitability Margins (%)</div>", unsafe_allow_html=True)
        margin_metrics_df_dict = {}
        margin_metrics_titles = {}
        margin_metrics_value_cols = {}
        margin_metrics_is_percent = {}

        for metric_key, display_title in [
            ("gross_margin", "Gross Margin"),
            ("ebitda_margin", "EBITDA Margin"),
            ("net_margin", "Net Margin"),
            ("profitability_metrics.roic", "ROIC") # Example if ROIC is here and has "values"
        ]:
            data_path = metric_key.split('.') # Handles potential nesting like profitability_metrics.roic
            current_data = fin_perf_data
            valid_path = True
            for p_key in data_path:
                if isinstance(current_data, dict) and p_key in current_data: current_data = current_data[p_key]
                else: valid_path = False; break

            if valid_path and isinstance(current_data, dict) and current_data.get("values"):
                df = pd.DataFrame(current_data["values"])
                if 'value_percent' in df.columns: # Margins usually have 'value_percent'
                    margin_metrics_df_dict[metric_key] = df
                    margin_metrics_titles[metric_key] = display_title
                    margin_metrics_value_cols[metric_key] = 'value_percent'
                    margin_metrics_is_percent[metric_key] = True
        
        plot_financial_performance_charts_row(margin_metrics_df_dict, margin_metrics_titles, margin_metrics_value_cols, margin_metrics_is_percent)

        # Display explanations for margins
        for metric_key_orig, _ in [("gross_margin", ""), ("ebitda_margin", ""), ("net_margin", ""), ("profitability_metrics.roic", "")]:
            data_path = metric_key_orig.split('.')
            current_data_for_expl = fin_perf_data; valid_path_expl = True
            for p_key in data_path:
                if isinstance(current_data_for_expl, dict) and p_key in current_data_for_expl: current_data_for_expl = current_data_for_expl[p_key]
                else: valid_path_expl = False; break
            if valid_path_expl and isinstance(current_data_for_expl, dict):
                expl = current_data_for_expl.get("explanation")
                classification = current_data_for_expl.get("classification")
                if expl:
                    st.markdown(f"**{metric_key_orig.split('.')[-1].replace('_',' ').title()}:** {get_sentiment_html_enhanced(classification)}", unsafe_allow_html=True)
                    st.markdown(f"<div class='explanation-box'>{expl}</div>", unsafe_allow_html=True)

        # --- You can add more chart groups here for other financial performance aspects ---
        # e.g., Growth Rates (Revenue Growth, EBITDA Growth, EPS Growth)
        # e.g., Cash Flow metrics (Operating Cash Flow, Free Cash Flow)

        # Fallback to render_generic_section for any remaining parts of financial_performance
        # Be careful not to re-render what's already charted above.
        # This requires a more complex state or passing already processed keys.
        # For now, this example focuses on specific chart groups.
        # You might need to create a list of keys already handled and skip them in render_generic_section.

    # --- Render other main analysis sections using the generic renderer ---
    sections_in_order_config = [
        # ("financial_performance", "Financial Performance"), # Handled specially above
        ("fundamental_analysis", "Fundamental Ratios (TTM)"),
        ("valuation_analysis", "Valuation Analysis"),
        ("growth_prospects", "Growth Prospects & Outlook"),
        ("competitive_position", "Competitive Landscape"),
        ("peers_comparison", "Peer Group Benchmarking"),
        ("revenue_segmentation", "Revenue Segmentation Insights"),
        ("risk_factors", "Key Risk Factors"),
        ("scenario_analysis", "Scenario Analysis (Valuation)"),
        ("shareholder_returns_analysis", "Shareholder Returns & Capital Allocation"),
        ("investment_thesis_summary", "Investment Thesis & Recommendation"),
    ]

    for section_key, section_display_title in sections_in_order_config:
        section_data_content = analysis_data.get(section_key)
        if section_data_content:
            st.markdown(f"<div class='section-header-ai'>{section_display_title}</div>", unsafe_allow_html=True)
            with st.container():
                render_generic_section(section_data_content, [section_key])
    
    st.markdown("---")
    if st.checkbox("Show Full Raw AI Analysis JSON", value=False, key="show_raw_ai_json_cb_v4"):
        st.json(analysis_data)
        
def ai_analysis_tab_content(): # This is the function imported by 1_Financial_Dashboard.py
    st.header("🤖 AI Financial Analysis & Insights")

    if not PIPELINE_IMPORTED_SUCCESSFULLY:
        st.error(f"Critical Error: Could not import the analysis pipeline. Please check setup. Details: {ERROR_PIPELINE_IMPORT}")
        return
    if not UTILS_IMPORTED_SUCCESSFULLY:
        st.error(f"Critical Error: Could not import utilities (e.g., for Neo4j). Details: {ERROR_UTILS_IMPORT}")
        return

    try:
        # initialize_analysis_resources is from analysis_pipeline
        initialize_analysis_resources()
    except Exception as e:
        st.error(f"Error initializing analysis resources: {e}")
        return

    st_cached_neo4j_driver = get_neo4j_driver() # from utils

    # ... (rest of your input collection, button logic, spinner, report display logic) ...
    # (This part which calls generate_ai_report and display_ai_analysis_dashboard)

    input_cols = st.columns([2, 1, 1.2]) 
    symbol = input_cols[0].text_input("Enter Stock Symbol:", value="AMZN", key="ai_symbol_final_v2").upper() # New key
    period = input_cols[1].selectbox("Select Period:", ["annual", "quarter"], key="ai_period_final_v2", index=0) # New key
    
    if 'ai_report_data' not in st.session_state: st.session_state.ai_report_data = None
    if 'generating_report' not in st.session_state: st.session_state.generating_report = False

    if input_cols[2].button("✨ Generate Analysis", key="ai_generate_button_final_v2", type="primary", use_container_width=True, # New key
                      disabled=st.session_state.generating_report):
        if not symbol: st.warning("Please enter a symbol.")
        else:
            st.session_state.generating_report = True
            st.session_state.ai_report_data = None 
            st.rerun()

    if st.session_state.generating_report:
        with st.spinner(f"🧠 Generating AI analysis for {symbol} ({period})... This may take a minute or two."):
            # generate_ai_report is from analysis_pipeline
            report_data = generate_ai_report(symbol, period, 0, st_cached_neo4j_driver)
            st.session_state.ai_report_data = report_data
            st.session_state.generating_report = False
            st.rerun()

    report_data_to_display = st.session_state.ai_report_data
    if report_data_to_display:
        st.markdown("---")
        # ... (your existing logic to display errors or the dashboard) ...
        if isinstance(report_data_to_display, dict) and report_data_to_display.get("status", "").startswith("error_"):
            st.error(f"Could not generate report for {symbol}: {report_data_to_display.get('message', 'Unknown error')}")
            if st.checkbox("Show error details", key="ai_error_details_cb_final_v2"): st.json(report_data_to_display)
        
        elif isinstance(report_data_to_display, dict) and "analysis" in report_data_to_display:
            analysis_json_data = report_data_to_display.get("analysis", {})
            metadata_json_str = report_data_to_display.get("metadata_json")
            metadata_dict = {}
            
            if isinstance(metadata_json_str, str):
                try: metadata_dict = json.loads(metadata_json_str)
                except: metadata_dict = report_data_to_display.get("metadata", {})
            elif isinstance(report_data_to_display.get("metadata"), dict):
                 metadata_dict = report_data_to_display.get("metadata", {})

            company_name_display = metadata_dict.get('company_name', symbol) if metadata_dict.get('company_name') else symbol
            st.subheader(f"Comprehensive Analysis: {company_name_display} ({metadata_dict.get('ticker', symbol)})")
            
            if metadata_dict:
                with st.expander("View Report Metadata & FMP Snapshot", expanded=False):
                    # ... (metadata display logic) ...
                    pass # Replace with your metadata display
            
            display_ai_analysis_dashboard(analysis_json_data, metadata_dict, symbol) # Defined above
        else:
            st.info("Report available, but 'analysis' key not found or data is in an unexpected format.")
            if st.checkbox("Show raw report data", key="ai_raw_data_cb_final_v2"): st.json(report_data_to_display)