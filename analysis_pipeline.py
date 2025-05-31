# analysis_pipeline.py (formerly load_annual_analysis.py - heavily condensed for this example)
from __future__ import annotations
import datetime
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime as dt_class, timedelta
from functools import lru_cache, wraps
from typing import Any, Dict, List, Optional
from openai import OpenAI, APIError, APIConnectionError, RateLimitError, BadRequestError, OpenAIError
import requests
import pandas as pd
import streamlit as st
try:
    from neo4j import GraphDatabase, Driver, unit_of_work
except ImportError:
    GraphDatabase = None; Driver = None; unit_of_work = None
    print("WARNING: neo4j driver not installed.")


# OpenAI is optional.  Install the package *and* set OPENAI_API_KEY to enable.
try:
    pass # Already imported above
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]
    OpenAIError = Exception  # type: ignore[assignment]

# --------------------------------------------------------------------------- #
# Logging & global constants
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",
)
logger = logging.getLogger("annual")

DEFAULT_MAX_WORKERS = os.cpu_count() or 4
SYMBOL_PROCESSING_WORKERS = 1 # Set as needed, start with 1 for easier debugging

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
symbol='AAL' # Can be a list of symbols later
period='annual' # 'annual' or 'quarter'
NEO4J_URI_ASK_AI_TAB       = st.secrets.get("NEO4J_URI", os.getenv("NEO4J_URI","neo4j+s://f9f444b7.databases.neo4j.io")) 
NEO4J_USERNAME_ASK_AI_TAB  = st.secrets.get("NEO4J_USERNAME", os.getenv("NEO4J_USERNAME",  "neo4j"))
NEO4J_PASSWORD_ASK_AI_TAB  = st.secrets.get("NEO4J_PASSWORD", os.getenv("NEO4J_PASSWORD",  "BhpVJhR0i8NxbDPU29OtvveNE8Wx3X2axPI7mS7zXW0"))
period_back = 0


@dataclass(slots=True, frozen=True)
class Config:
    fmp_key: str = field(default_factory=lambda: _must_get("FMP_API_KEY"))
    openai_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    user_agent: str = "TrendingWizeAnalysis/2.0"
    price_look_forward: int = 7
    retry_attempts: int = 3
    retry_delay: int = 5
    retry_backoff: float = 1.5
    # Neo4j config can be added here if preferred, or kept as global constants
    neo4j_uri: str = NEO4J_URI
    neo4j_user: str = NEO4J_USER
    neo4j_password: str = NEO4J_PASSWORD


APP_CONFIG = None
FDM_MODULE_INSTANCE = None
OPENAI_CLIENT_INSTANCE = None
NEO4J_DRIVER_INSTANCE = None # This should be managed by Streamlit's lifecycle

# Function to initialize shared resources if not already done
@st.cache_resource(show_spinner="Initializing OpenAI Client for Report Pipeline...")
def initialize_analysis_resources():
    global APP_CONFIG, FDM_MODULE_INSTANCE, OPENAI_CLIENT_INSTANCE, NEO4J_DRIVER_INSTANCE

    if APP_CONFIG is None:
        # Load FMP_API_KEY and OPENAI_API_KEY from Streamlit secrets or os.environ
        # For this example, let's assume they are set in os.environ as in your original script
        NEO4J_URI_ASK_AI_TAB       = st.secrets.get("NEO4J_URI", os.getenv("NEO4J_URI","neo4j+s://f9f444b7.databases.neo4j.io")) 
        NEO4J_USERNAME_ASK_AI_TAB  = st.secrets.get("NEO4J_USERNAME", os.getenv("NEO4J_USERNAME",  "neo4j"))
        NEO4J_PASSWORD_ASK_AI_TAB  = st.secrets.get("NEO4J_PASSWORD", os.getenv("NEO4J_PASSWORD",  "BhpVJhR0i8NxbDPU29OtvveNE8Wx3X2axPI7mS7zXW0"))    
        APP_CONFIG = Config() # Your Config class from the script

    if FDM_MODULE_INSTANCE is None and APP_CONFIG:
        FDM_MODULE_INSTANCE = FinancialDataModule(config=APP_CONFIG)

    if OPENAI_CLIENT_INSTANCE is None and APP_CONFIG and APP_CONFIG.openai_key and OpenAI:
        try:
            OPENAI_CLIENT_INSTANCE = OpenAI(api_key=APP_CONFIG.openai_key)
            logger.info("OpenAI client initialized for analysis pipeline.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client in pipeline: {e}")
            OPENAI_CLIENT_INSTANCE = None # Ensure it's None if init fails
    
    # NEO4J_DRIVER_INSTANCE should ideally be managed by Streamlit's @st.cache_resource
    # and passed into generate_ai_report if needed, or re-fetched inside.
    # For now, let's assume process_symbol_logic can use a globally available or passed driver.


# --- Main function to be called by Streamlit ---
def generate_ai_report(
    symbol_to_process: str,
    report_period: str, # 'annual' or 'quarter'
    period_back_offset: int = 0,
    st_neo4j_driver: Optional[Driver] = None # Allow passing Streamlit cached driver
) -> Optional[Dict[str, Any]]:
    """
    Generates an AI financial analysis report for a given symbol and period.
    Returns the full analysis report dictionary or None on failure.
    """
    initialize_analysis_resources() # Ensure global instances are ready

    if not APP_CONFIG or not FDM_MODULE_INSTANCE:
        logger.error("Analysis pipeline resources (Config/FDM) not initialized.")
        return {"status": "error_init", "message": "Core analysis modules not initialized."}
    
    # If st_neo4j_driver is provided by Streamlit, use it.
    # Otherwise, process_symbol_logic might try to get its own if neo4j_driver_instance is set globally (less ideal here)
    # Best to ensure process_symbol_logic accepts a driver instance.
    
    # --- Modify process_symbol_logic to accept period_type and use global FDM & OpenAI ---
    # You would need to adjust your `process_symbol_logic` slightly:
    # 1. Remove its own FDM/OpenAI/Neo4j driver instantiation if they are global/passed.
    # 2. Accept `report_period` and `period_back_offset` and use them.
    # 3. The original `process_symbol_logic` seems to use a global `period` variable.
    #    This should be passed as `report_period` to `get_financial_statements` etc.

    # For this example, let's simulate the call structure.
    # You will need to integrate this with your actual `process_symbol_logic`.
    # The key is that `process_symbol_logic` should now use the initialized FDM and OpenAI client.

    logger.info(f"Calling process_symbol_logic for {symbol_to_process}, period: {report_period}, back: {period_back_offset}")

    # Make sure process_symbol_logic uses the global/passed instances:
    # fdm_module_instance = FDM_MODULE_INSTANCE
    # openai_client_instance = OPENAI_CLIENT_INSTANCE
    # neo4j_driver_instance = st_neo4j_driver (or your globally managed one)
    # app_config = APP_CONFIG
    # And uses `report_period` instead of global `period`
    
    # Placeholder for the actual call to your refactored process_symbol_logic
    # This assumes `process_symbol_logic` is modified to work with these parameters
    # and uses the globally initialized FDM and OpenAI client.
    try:
        # You'll need to adjust the call to your `process_symbol_logic`
        # to ensure it uses the `report_period` and `period_back_offset`
        # and the globally initialized or passed `FDM_MODULE_INSTANCE`,
        # `OPENAI_CLIENT_INSTANCE`, and `st_neo4j_driver`.

        # If process_symbol_logic is directly callable and adapted:
        report_data = process_symbol_logic(
            symbol_to_process=symbol_to_process,
            current_period_back_val=period_back_offset, # Ensure this matches param name in process_symbol_logic
            fdm_module_instance=FDM_MODULE_INSTANCE,    # Pass initialized FDM
            openai_client_instance=OPENAI_CLIENT_INSTANCE, # Pass initialized OpenAI
            neo4j_driver_instance=st_neo4j_driver,       # Pass Neo4j driver
            app_config=APP_CONFIG                       # Pass config
            # You might need to also explicitly pass `report_period` if process_symbol_logic
            # uses a global `period` variable internally.
        )
        # Ensure process_symbol_logic itself uses the `report_period` (annual/quarter)
        # when calling fdm_module.get_financial_statements(..., period_param=report_period, ...)

        if isinstance(report_data, dict) and "analysis" in report_data and isinstance(report_data["analysis"], dict):
            return report_data # Return the full report
        else:
            logger.error(f"Analysis generation failed or returned unexpected format for {symbol_to_process}. Report data: {str(report_data)[:500]}")
            if isinstance(report_data, dict): return report_data # Return error dict
            return {"status": "error_unexpected_report_format", "symbol": symbol_to_process, "message": "Unexpected report format from processing."}

    except Exception as e:
        logger.error(f"Exception during generate_ai_report for {symbol_to_process}: {e}", exc_info=True)
        return {"status": "error_exception", "symbol": symbol_to_process, "message": str(e)}
        
        
def _must_get(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise EnvironmentError(f"Required environment variable '{name}' is missing.")
    return val


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def retry(
    *,
    max_attempts: int = 3,
    delay: int = 2,
    backoff: float = 1.4,
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if attempt == max_attempts:
                        logger.error("%s failed after %s attempts", func.__name__, attempt)
                        raise
                    pause = delay * (backoff ** (attempt - 1)) + random.random()
                    logger.warning(
                        "%s attempt %s/%s failed: %s – retrying in %.1fs",
                        func.__name__,
                        attempt,
                        max_attempts,
                        exc,
                        pause,
                    )
                    time.sleep(pause)
            return None # Should be unreachable if max_attempts >= 1
        return wrapper
    return decorator


def _redact(text: str, secret: str | None) -> str:
    return text.replace(secret, "***") if secret else text


# --------------------------------------------------------------------------- #
# FinancialDataModule
# --------------------------------------------------------------------------- #


class FinancialDataModule:
    def __init__(self, config: Optional[Config] = None) -> None:
        self.cfg = config or Config()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.cfg.user_agent})
        self._openai: Optional[OpenAI] = None
        if self.cfg.openai_key and OpenAI is not None:
            self._openai = OpenAI(api_key=self.cfg.openai_key)
            logger.debug("OpenAI client initialised")
        elif self.cfg.openai_key:
            logger.warning("OPENAI_API_KEY set but `openai` package not installed")

    @retry()
    def _get(self, url: str) -> Any | None:
        redacted_url = _redact(url, self.cfg.fmp_key)
        logger.debug("GET %s", redacted_url)
        resp = self.session.get(url, timeout=20)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            logger.error("HTTPError for %s: %s, Response: %s", redacted_url, e, resp.text)
            if 400 <= resp.status_code < 500 and resp.status_code != 429 : # Do not retry client errors other than 429
                 raise # Propagate immediately
            raise # Let retry handler deal with 429 or 5xx

        if not resp.content:
            logger.warning("Empty response from %s", redacted_url)
            return None
        try:
            return resp.json()
        except json.JSONDecodeError:
            logger.error("Non‑JSON body returned from %s: %s", redacted_url, resp.text[:500])
            return None

    @staticmethod
    def _pct_change(curr: Optional[float], prev: Optional[float]) -> Optional[float]:
        if curr is None or prev is None or prev == 0:
            return None
        return curr / prev - 1

    @staticmethod
    def _cagr(first: Optional[float], last: Optional[float], years: int) -> Optional[float]:
        if first is None or first == 0 or last is None or years <= 0:
            return None
        # Handle potential negative numbers if they are not meaningful for CAGR
        if first < 0 and last > 0 or first > 0 and last < 0: # Sign change, CAGR not meaningful
             return None
        if first < 0 and last < 0: # Both negative, take absolute for calculation logic
            return ((abs(last) / abs(first)) ** (1 / years) - 1) * (-1 if last < first else 1)
        return (last / first) ** (1 / years) - 1


    @staticmethod
    def _safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
        if num is None or den is None or den == 0:
            return None
        try:
            return num / den
        except ZeroDivisionError: # Should be caught by den == 0
            return None

    @retry()
    def get_financial_statements(
        self,
        symbol: str,
        statement: str,
        period_param: str, # Renamed to avoid conflict with global 'period'
        limit: int,
    ) -> Optional[List[Dict[str, Any]]]:
        url = (
            f"https://financialmodelingprep.com/api/v3/{statement}/{symbol}"
            f"?period={period_param}&limit={limit}&apikey={self.cfg.fmp_key}"
        )
        return self._get(url)

    @retry()
    def get_first_historical_close_price(
        self,
        symbol: str,
        target_date_str: str,
        look_forward_days: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        if not target_date_str:
            return None
        look_forward_days = (
            look_forward_days if look_forward_days is not None else self.cfg.price_look_forward
        )
        # Use aliased dt_class for strptime
        start = dt_class.strptime(target_date_str[:10], "%Y-%m-%d").date()
        end = start + timedelta(days=look_forward_days)
        url = (
            "https://financialmodelingprep.com/api/v3/historical-price-full/"
            f"{symbol}?from={start}&to={end}&apikey={self.cfg.fmp_key}"
        )
        data = self._get(url)
        if not data or "historical" not in data or not data["historical"]: # Check if historical is empty
            return None
        # FMP historical data is typically sorted ascending by date.
        # We want the first available price ON or AFTER the target_date.
        # So, we iterate normally (oldest to newest in response) and take the first one.
        # If the API returns descending, then reversed() would be correct.
        # Based on typical API behavior, non-reversed is usually what's needed for "first after date".
        # Let's assume the API returns ascending.
        for rec in data["historical"]: # Iterate from earliest to latest in the response
            if rec.get("close") is not None:
                return {"date": rec["date"], "close": rec["close"]}
        return None


    @retry()
    def get_latest_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={self.cfg.fmp_key}"
        data = self._get(url)
        if not data or not isinstance(data, list) or len(data) == 0:
            return None
        row = data[0]
        ts = row.get("timestamp")
        # Use aliased dt_class for fromtimestamp
        date_str = dt_class.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else "N/A"
        return {"date": date_str, "close": row.get("price")}

    @lru_cache(maxsize=128)
    def get_company_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={self.cfg.fmp_key}"
        data = self._get(url)
        if not data or not isinstance(data, list) or len(data) == 0:
            return None
        r = data[0]
        return {
            "company_name": r.get("companyName"),
            "exchange": r.get("exchangeShortName"),
            "industry": r.get("industry"),
            "sector": r.get("sector"),
            "beta": r.get("beta"),
            "currency": r.get("currency"),
            # Add other fields if needed by OpenAI prompt that are available in profile
            "description": r.get("description"),
            "ipoDate": r.get("ipoDate"),
            "website": r.get("website"),
            "ceo": r.get("ceo"),
            "fullTimeEmployees": r.get("fullTimeEmployees"),
            "image": r.get("image"),
            "marketCap": r.get("mktCap"), # Note: mktCap is often more current than in income statement
        }

    def process_batch(self, symbol: str) -> Optional[Dict[str, List[str]]]:
        if not self._openai:
            logger.info("OpenAI disabled – skipping competitor discovery")
            return {symbol: []}
        prompt = (
            "List five competitor tickers ONLY in JSON:\n"
            f'{{ "{symbol}": ["TICK1", "TICK2", "TICK3", "TICK4", "TICK5"] }}'
        )
        try:
            chat = self._openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a US equity research assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            content = chat.choices[0].message.content
            if content:
                 return json.loads(content)
            return {symbol: []} # Should not happen with json_object
        except (OpenAIError, json.JSONDecodeError, AttributeError) as exc:
            logger.warning("OpenAI peer lookup failed for %s: %s", symbol, exc)
            return {symbol: []}

    def get_peer_metrics(self, symbol: str, peers: List[str]) -> List[Dict[str, Any]]:
        tickers = [symbol, *peers]
        out: List[Dict[str, Any]] = []
        # Ensure DEFAULT_MAX_WORKERS is at least 1
        max_workers_for_pool = max(1, DEFAULT_MAX_WORKERS)

        with ThreadPoolExecutor(max_workers=max_workers_for_pool) as pool:
            fut_quotes = {
                t: pool.submit(
                    self._get, f"https://financialmodelingprep.com/api/v3/quote/{t}?apikey={self.cfg.fmp_key}"
                )
                for t in tickers
            }
            fut_income = {
                t: pool.submit(
                    self.get_financial_statements, t, "income-statement", "annual", 4
                )
                for t in tickers
            }
            fut_bs = {
                t: pool.submit(
                    self.get_financial_statements, t, "balance-sheet-statement", "annual", 4
                )
                for t in tickers
            }

        for tkr in tickers:
            try:
                quote_data = fut_quotes[tkr].result()
                income_data = fut_income[tkr].result()
                bs_data = fut_bs[tkr].result() # Changed variable name
                
                row = {}
                if quote_data and isinstance(quote_data, list) and len(quote_data) > 0:
                    row = quote_data[0]

                def _nth(field: str, n: int, default=None):
                    try:
                        return income_data[n][field] if income_data and len(income_data) > n else default
                    except (IndexError, KeyError, TypeError):
                        return default

                def _bs(field: str, n: int = 0, default=None): # Added n for bs_data
                    try:
                        return bs_data[n][field] if bs_data and len(bs_data) > n else default
                    except (IndexError, KeyError, TypeError):
                        return default

                rev_0, rev_1, rev_3 = _nth("revenue", 0), _nth("revenue", 1), _nth("revenue", 3)
                ebitda_0, ebitda_1 = _nth("ebitda", 0), _nth("ebitda", 1)
                eps_0, eps_1 = _nth("epsdiluted", 0), _nth("epsdiluted", 1)

                eps_growth_yoy = self._pct_change(eps_0, eps_1)
                peg_ratio = (
                    self._safe_div(row.get("pe"), (eps_growth_yoy * 100))
                    if row.get("pe") is not None and eps_growth_yoy and eps_growth_yoy > 0
                    else None
                )
                
                shareholder_equity_0 = _bs("totalStockholdersEquity", 0)
                total_assets_0 = _bs("totalAssets", 0)
                cash_0 = _bs("cashAndCashEquivalents", 0, 0.0) or 0.0
                total_debt_0 = _bs("totalDebt", 0) or ((_bs("longTermDebt", 0) or 0) + (_bs("shortTermDebt", 0) or 0))

                gross_margin = self._safe_div(_nth("grossProfit", 0), rev_0)
                ebitda_margin = self._safe_div(ebitda_0, rev_0)
                net_income_0 = _nth("netIncome", 0)
                net_margin = self._safe_div(net_income_0, rev_0)

                roe = self._safe_div(net_income_0, shareholder_equity_0)

                ebit_0 = _nth("operatingIncome", 0) # operatingIncome is typically EBIT
                if ebit_0 is None: ebit_0 = ebitda_0 # Fallback to EBITDA if op income is missing
                
                income_before_tax_0 = _nth("incomeBeforeTax", 0)
                income_tax_exp_0 = _nth("incomeTaxExpense", 0)
                tax_rate = (
                    self._safe_div(income_tax_exp_0, income_before_tax_0) if income_before_tax_0 else None
                )
                
                nopat = (
                    ebit_0 * (1 - tax_rate)
                    if (ebit_0 is not None and tax_rate is not None)
                    else None
                )
                invested_capital = (
                    (total_debt_0 or 0) + (shareholder_equity_0 or 0) - cash_0
                    if shareholder_equity_0 is not None
                    else None
                )
                roic = self._safe_div(nopat, invested_capital)

                asset_turnover = self._safe_div(rev_0, total_assets_0)
                debt_to_ebitda = self._safe_div(total_debt_0, ebitda_0)
                interest_exp_0 = _nth("interestExpense", 0)
                interest_coverage = self._safe_div(
                    ebit_0, abs(interest_exp_0) if interest_exp_0 else None # Use abs for interest expense
                )
                revenue_cagr_3y = self._cagr(rev_3, rev_0, 3)

                out.append({
                    "symbol": tkr, "price": row.get("price"), "pe": row.get("pe"),
                    "revenue_growth_yoy": self._pct_change(rev_0, rev_1),
                    "revenue_cagr_3y": revenue_cagr_3y,
                    "ebitda_growth_yoy": self._pct_change(ebitda_0, ebitda_1),
                    "eps_diluted_growth_yoy": eps_growth_yoy,
                    "peg_ratio": peg_ratio, "gross_margin": gross_margin,
                    "ebitda_margin": ebitda_margin, "net_margin": net_margin,
                    "roe": roe, "roic": roic, "asset_turnover": asset_turnover,
                    "debt_to_ebitda": debt_to_ebitda, "interest_coverage": interest_coverage,
                })
            except Exception as e:
                logger.error(f"Error processing peer metrics for {tkr}: {e}", exc_info=True)
                out.append({"symbol": tkr, "error": str(e)}) # Add placeholder for errored ticker
        return out

    @retry()
    def get_revenue_product_segmentation(
        self, symbol: str, period_param: str # Renamed
    ) -> Optional[List[Dict[str, Any]]]:
        url = (
            "https://financialmodelingprep.com/api/v4/revenue-product-segmentation"
            f"?symbol={symbol}&period={period_param}&apikey={self.cfg.fmp_key}"
        )
        return self._get(url)

    @retry()
    def get_revenue_geographic_segmentation(
        self, symbol: str, period_param: str # Renamed
    ) -> Optional[List[Dict[str, Any]]]:
        url = (
            "https://financialmodelingprep.com/api/v4/revenue-geographic-segmentation"
            f"?symbol={symbol}&period={period_param}&apikey={self.cfg.fmp_key}"
        )
        return self._get(url)

    @retry()
    def get_outstanding_shares(
        self, symbol: str
    ) -> Optional[List[Dict[str, Any]]]:
        # This endpoint seems to be deprecated or changed.
        # Using /api/v3/historical-price-full/{symbol}/shares for historical shares
        # For latest, it's in company profile or financial statements.
        # The prompt uses income_data[period_back].get("weightedAverageShsOut")
        # This function might not be strictly needed if that's sufficient.
        # Let's try to find a V4 equivalent or adjust.
        # FMP docs suggest /v3/enterprise-values/{symbol}?limit=1 for latest sharesOutstanding
        # Or stick to weightedAverageShsOut from income statement as primary.
        # For historical trend, if needed:
        url = (
            f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}/shares"
            f"?apikey={self.cfg.fmp_key}"
        )
        data = self._get(url)
        if data and "historicalStockSplits" in data: # This is not shares, it's splits
            # This endpoint is incorrect for shares.
            # Let's return an empty list and rely on income statement for shares.
            logger.warning(f"get_outstanding_shares endpoint for {symbol} might be incorrect, relying on income statement.")
            return []
        # A different endpoint might be needed for historical outstanding shares trend
        # /api/v4/shares_float?symbol=AAPL
        # This gives shares float, not total outstanding directly
        return [] # Placeholder, as the original endpoint might be problematic.

    @retry()
    def get_treasury_rate(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        # Use aliased dt_class for utcnow
        today = dt_class.utcnow().date()
        if end_date is None:
            end_date = today.strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
        url = (
            "https://financialmodelingprep.com/api/v4/treasury"
            f"?from={start_date}&to={end_date}&apikey={self.cfg.fmp_key}"
        )
        data = self._get(url)
        if not data or not isinstance(data, list) or len(data) == 0:
            return None
        
        latest = None
        latest_dt = dt_class.min.date() # Initialize with a very old date

        for row in data:
            if "year10" in row and row.get("date"):
                try:
                    current_dt = dt_class.strptime(row["date"], "%Y-%m-%d").date()
                    if current_dt >= latest_dt: # Use >= to prefer more recent if dates are same
                        latest_dt = current_dt
                        latest = row
                except ValueError:
                    continue # Skip malformed dates
        
        if not latest:
            return None
        return {"date": latest["date"], "year10": latest["year10"]}

    @retry()
    def _get_sector_ratio_data( # Renamed from get_sector_pe_ratio
        self,
        exchange: str,
        sector_name: str, # Renamed from sector to avoid conflict
        anchor_date: Optional[str | dt_class.date] = None, # Use aliased dt_class
    ) -> Optional[List[Dict[str, Any]]]:
        target_date: Optional[dt_class.date] = None
        if anchor_date is None:
            target_date = dt_class.utcnow().date()
        elif isinstance(anchor_date, str):
            try:
                target_date = dt_class.strptime(anchor_date[:10], "%Y-%m-%d").date()
            except ValueError:
                logger.error("Invalid anchor_date string format for sector P/E: %s", anchor_date)
                return None
        elif isinstance(anchor_date, dt_class.date): # Use aliased dt_class
            target_date = anchor_date
        else:
            logger.error("Invalid type for anchor_date for sector P/E: %s (%s)", type(anchor_date), anchor_date)
            return None

        if not target_date: return None # Should not happen with above logic
        date_str = target_date.strftime("%Y-%m-%d")

        if not sector_name:
            logger.warning("Sector name is None or empty for sector P/E request.")
            return None
        if not exchange:
            logger.warning("Exchange is None or empty for sector P/E request.")
            return None
        
        # FMP endpoint for historical sector P/E seems to be /api/v4/sector_pe?date=YYYY-MM-DD
        # The URL in original code might be outdated or for a different version.
        # Let's try the documented v4 one. It might require iterating if no data for exact date.
        # For now, stick to the provided URL structure and adapt if it fails.
        # The URL /stable/historical-sector-pe seems like a bulk data endpoint.
        # Let's assume it works as in the original code.
        url = (
            f"https://financialmodelingprep.com/stable/historical-sector-pe" # Changed to v3 as /stable/ might be internal
            f"?sector={requests.utils.quote(sector_name)}" # URL Encode sector name
            f"&date={date_str}" # FMP often uses 'date' for single day, or from/to for range
            # f"&from={date_str}&to={date_str}" # Using from/to as in original
            f"&exchange={exchange}"
            f"&apikey={self.cfg.fmp_key}"
        )
        # If the above URL doesn't work, try with from/to
        # url = (
        #     f"https://financialmodelingprep.com/api/v3/historical-sector-pe"
        #     f"?sector={requests.utils.quote(sector_name)}"
        #     f"&from={date_str}&to={date_str}" 
        #     f"&exchange={exchange}&apikey={self.cfg.fmp_key}"
        # )
        logger.info(f"Fetching Sector P/E URL: {_redact(url, self.cfg.fmp_key)}")
        data = self._get(url)
        if data and isinstance(data, list) and len(data) > 0:
            logger.info("Got Sector P/E for Sector=%s Exchange=%s Date=%s", sector_name, exchange, date_str)
            return data
        else:
            logger.warning("No Sector P/E data found for Sector=%s Exchange=%s Date=%s. Data: %s",
                        sector_name, exchange, date_str, str(data)[:200])
            return None

    @retry()
    def get_industry_pe_ratio( # Renamed from industry_pe_ratio
        self,
        exchange: str,
        industry_name: str, # Renamed from industry
        anchor_date: Optional[str | dt_class.date] = None, # Use aliased dt_class
    ) -> Optional[List[Dict[str, Any]]]:
        target_date: Optional[dt_class.date] = None # Use aliased dt_class
        if anchor_date is None:
            target_date = dt_class.utcnow().date() # Use aliased dt_class
        elif isinstance(anchor_date, str):
            try:
                target_date = dt_class.strptime(anchor_date[:10], "%Y-%m-%d").date() # Use aliased dt_class
            except ValueError:
                logger.error("Invalid anchor_date string format for industry P/E: %s", anchor_date)
                return None
        elif isinstance(anchor_date, dt_class.date): # Use aliased dt_class
            target_date = anchor_date
        else:
            logger.error("Invalid type for anchor_date for industry P/E: %s (%s)", type(anchor_date), anchor_date)
            return None
        
        if not target_date: return None
        date_str = target_date.strftime("%Y-%m-%d")

        if not industry_name:
            logger.warning("Industry name is None or empty for industry P/E request.")
            return None
        if not exchange:
            logger.warning("Exchange is None or empty for industry P/E request.")
            return None
        
        # Similar to sector, using v3 and date parameter, or from/to
        url = (
            f"https://financialmodelingprep.com/stable/historical-industry-pe"
            f"?industry={requests.utils.quote(industry_name)}" # URL Encode industry name
            f"&date={date_str}"
            # f"&from={date_str}&to={date_str}"
            f"&exchange={exchange}"
            f"&apikey={self.cfg.fmp_key}"
        )
        logger.info(f"Fetching Industry P/E URL: {_redact(url, self.cfg.fmp_key)}")
        data = self._get(url)
        if data and isinstance(data, list) and len(data) > 0:
            logger.info("Got Industry P/E for Industry=%s Exchange=%s Date=%s", industry_name, exchange, date_str)
            return data
        else:
            logger.warning("No Industry P/E data found for Industry=%s Exchange=%s Date=%s. Data: %s",
                        industry_name, exchange, date_str, str(data)[:200])
            return None

    def execute_all(self, symbol_param: str, current_period_back: int = 0, current_period_type: str = 'annual') -> Dict[str, Any]: # Renamed params
        t0 = time.perf_counter()
        logger.info("execute_all(%s, period_back=%s, period_type=%s) – start", symbol_param, current_period_back, current_period_type)
    
        profile = self.get_company_profile(symbol_param)
        if not profile:
            raise RuntimeError(f"Could not retrieve profile for {symbol_param}")
    
        exchange = profile.get("exchange")
        company_sector = profile.get("sector") # Renamed to avoid conflict
        company_industry = profile.get("industry") # Renamed to avoid conflict
    
        max_workers_for_pool = max(1, DEFAULT_MAX_WORKERS)
        with ThreadPoolExecutor(max_workers=max_workers_for_pool) as pool:
            fut_income = pool.submit(
                self.get_financial_statements,
                symbol_param, "income-statement", current_period_type, 5 + current_period_back,
            )
            fut_balance_sheet = pool.submit(
                self.get_financial_statements,
                symbol_param, "balance-sheet-statement", current_period_type, 5 + current_period_back,
            )
            fut_cash_flow = pool.submit(
                self.get_financial_statements,
                symbol_param, "cash-flow-statement", current_period_type, 5 + current_period_back,
            )
            fut_product_seg = pool.submit(
                self.get_revenue_product_segmentation, symbol_param, current_period_type
            )
            fut_geo_seg = pool.submit(
                self.get_revenue_geographic_segmentation, symbol_param, current_period_type
            )
            # fut_shares = pool.submit(self.get_outstanding_shares, symbol_param) # Potentially remove or fix
            fut_peers = pool.submit(self.process_batch, symbol_param)
    
            income_data = fut_income.result()
            if not income_data or not isinstance(income_data, list) or len(income_data) <= current_period_back:
                raise RuntimeError(f"No/Insufficient income statements for {symbol_param} (need {current_period_back+1}, got {len(income_data) if income_data else 0})")
    
            target_income_statement = income_data[current_period_back]
            filing_date = target_income_statement.get("fillingDate")
            if not filing_date: # Also check 'date' as a fallback
                filing_date = target_income_statement.get("date")
            if not filing_date:
                 raise RuntimeError(f"Missing fillingDate/date in income statement for {symbol_param} at period_back {current_period_back}")


            fut_price_now = pool.submit(
                self.get_first_historical_close_price,
                symbol_param, filing_date, self.cfg.price_look_forward,
            )
            # FMP treasury takes from/to, so using filing_date for both gets the rate for that specific day or nearest available
            fut_treasury = pool.submit(self.get_treasury_rate, filing_date, filing_date)

            fut_sector_pe = None
            if exchange and company_sector:
                fut_sector_pe = pool.submit(self._get_sector_ratio_data,
                                exchange=exchange, anchor_date=filing_date, sector_name=company_sector)
            else:
                logger.warning(f"Skipping sector PE for {symbol_param} due to missing exchange ({exchange}) or sector ({company_sector})")

            fut_industry_pe = None
            if exchange and company_industry:
                fut_industry_pe = pool.submit(self.get_industry_pe_ratio,
                                exchange=exchange, anchor_date=filing_date, industry_name=company_industry)
            else:
                logger.warning(f"Skipping industry PE for {symbol_param} due to missing exchange ({exchange}) or industry ({company_industry})")

        balance_sheet_data = fut_balance_sheet.result()
        cash_flow_data = fut_cash_flow.result()
        peers_dict = fut_peers.result() or {symbol_param: []} # Ensure it's a dict
        peers_list = peers_dict.get(symbol_param, []) # Get list of peers for the symbol

        fillingDate_price = fut_price_now.result()
        peer_metrics = self.get_peer_metrics(symbol_param, peers_list)
        
        product_seg = fut_product_seg.result() or []
        # Filter product_seg for the relevant year if possible (matching calendarYear)
        # This is complex as product_seg might not align perfectly with period_back logic directly
        # For now, take the latest available set or filter if 'date' or 'calendarYear' is in product_seg items
        # Example: if product_seg items have 'date', filter by target_income_statement['calendarYear']
        # Assuming product_seg is a list of dicts, each for a year/quarter
        
        geo_seg = fut_geo_seg.result() or []
        # shares_hist = fut_shares.result() # Potentially remove

        # Update profile with potentially more accurate/recent shares outstanding from income statement
        profile['sharesOutstanding'] = target_income_statement.get("weightedAverageShsOutDil") or \
                                       target_income_statement.get("weightedAverageShsOut")
        profile['marketCap'] = (fillingDate_price.get('close') * profile['sharesOutstanding']
                                if fillingDate_price and fillingDate_price.get('close') and profile.get('sharesOutstanding')
                                else profile.get('marketCap')) # Use calculated if possible

        treasury_rate = fut_treasury.result()
    
        sector_pe_data = fut_sector_pe.result() if fut_sector_pe else None
        industry_pe_data = fut_industry_pe.result() if fut_industry_pe else None
    
        def safe_get_first(pe_data_list): # Renamed var
            return pe_data_list[0] if pe_data_list and isinstance(pe_data_list, list) and len(pe_data_list) > 0 else None
    
        payload: Dict[str, Any] = {
            "symbol": symbol_param,
            "profile": profile, # Profile now includes more details
            "treasury_rate_10y": treasury_rate,
            "sector_pe_ratio": safe_get_first(sector_pe_data),
            "industry_pe_ratio": safe_get_first(industry_pe_data),
            "fillingDate_price": fillingDate_price, # Price at/after filling date
            "current_price_info": self.get_latest_price(symbol_param), # Add truly current price for context
            "income_statement": income_data[current_period_back:] if income_data else [],
            "balance_sheet": balance_sheet_data[current_period_back:] if balance_sheet_data else [],
            "cash_flow_statement": cash_flow_data[current_period_back:] if cash_flow_data else [], # Renamed key
            "peers": peers_list,
            "peer_metrics": peer_metrics, # Includes the main symbol as the first item
            # Filter segmentation data if possible (e.g., by calendarYear of target_income_statement)
            "product_segmentation": product_seg, # Renamed key & provide all years for now
            "geographic_segmentation": geo_seg, # Renamed key & provide all years
            # "shares_history": shares_hist, # Potentially remove
            "metadata_package": { # Add some context about this data package
                "fmp_filling_date": filing_date,
                "fmp_calendar_year": target_income_statement.get("calendarYear"),
                "fmp_period": target_income_statement.get("period"),
                "period_back_offset": current_period_back,
                "data_fetch_timestamp": dt_class.utcnow().isoformat()
            }
        }
        logger.info("execute_all(%s) – done in %.2fs", symbol_param, time.perf_counter() - t0)
        return payload

    @staticmethod
    def _cli() -> None: # pragma: no cover
        import argparse, pprint, sys
        p = argparse.ArgumentParser(description="Annual data fetcher")
        p.add_argument("symbol", help="Ticker symbol, e.g. AAPL")
        p.add_argument("--period-back", type=int, default=0, help="Years/Quarters back from most recent filing")
        p.add_argument("--period-type", type=str, default="annual", choices=["annual", "quarter"], help="Period type for financials")
        args = p.parse_args()
        try:
            fmp = FinancialDataModule()
            data = fmp.execute_all(args.symbol, args.period_back, args.period_type)
            # To save to file:
            # with open(f"{args.symbol}_fmp_data.json", "w") as f:
            #    json.dump(data, f, indent=4)
            # logger.info(f"Data saved to {args.symbol}_fmp_data.json")
            pprint.pprint(data)
        except Exception as exc:
            logger.error("CLI Error: %s", exc, exc_info=True)
            sys.exit(1)

# --- Neo4j Helper Functions ---
def get_neo4j_driver(config: Config) -> Optional[Driver]:
    if not GraphDatabase:
        logger.error("Neo4j driver not available.")
        return None
    try:
        driver = GraphDatabase.driver(config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password))
        driver.verify_connectivity()
        logger.info(f"Successfully connected to Neo4j at {config.neo4j_uri}")
        return driver
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}", exc_info=True)
        return None

@unit_of_work(timeout=10) # type: ignore
def get_latest_bs_filling_date_neo4j(tx, symbol_param: str) -> Optional[str]:
    # Query to get the fillingDate of the most recent BalanceSheet node for the symbol
    # This assumes you are also storing raw FMP BalanceSheet data in Neo4j
    # If not, this query needs to change or this check needs to be rethought.
    # The prompt asks to compare with BalanceSheet.fillingDate
    query = (
        "MATCH (bs:BalanceSheet {symbol: $symbol_param}) "
        "WHERE bs.fillingDate IS NOT NULL "
        "RETURN bs.fillingDate AS latestDate "
        "ORDER BY bs.fillingDate DESC LIMIT 1"
    )
    result = tx.run(query, symbol_param=symbol_param)
    record = result.single()
    if record and record["latestDate"]:
        # Neo4j date object needs conversion to string if not already
        date_obj = record["latestDate"]
        if hasattr(date_obj, 'iso_format'): # Neo4j Date object
            return date_obj.iso_format()
        return str(date_obj) # Fallback
    return None

@unit_of_work(timeout=10) # type: ignore
def get_analysis_from_neo4j(tx, symbol_param: str, filling_date_str: str) -> Optional[Dict[str, Any]]:
    query = (
        "MATCH (ar:AnalysisReport {symbol: $symbol_param, fillingDate: date($filling_date_str)}) "
        "RETURN ar"
    )
    result = tx.run(query, symbol_param=symbol_param, filling_date_str=filling_date_str)
    record = result.single()
    if record and record["ar"]:
        # The properties of the node 'ar' form the dictionary
        # Need to handle Neo4j specific types like Date, DateTime if they are not auto-converted by driver
        report_data = dict(record["ar"])
        # Ensure metadata and analysis are dictionaries (they should be from Cypher load)
        if 'metadata' in report_data and isinstance(report_data['metadata'], str):
            try: report_data['metadata'] = json.loads(report_data['metadata'])
            except json.JSONDecodeError: pass # Keep as string if not valid JSON
        if 'analysis' in report_data and isinstance(report_data['analysis'], str):
            try: report_data['analysis'] = json.loads(report_data['analysis'])
            except json.JSONDecodeError: pass
        
        # Convert Neo4j DateTime/Date objects in metadata back to ISO strings if necessary
        if 'metadata' in report_data and isinstance(report_data['metadata'], dict):
            for key, val in report_data['metadata'].items():
                if hasattr(val, 'iso_format'): # Check if it's a Neo4j Date/DateTime object
                    report_data['metadata'][key] = val.iso_format()
        if hasattr(report_data.get('analysis_generated_at'), 'iso_format'):
             report_data['analysis_generated_at'] = report_data['analysis_generated_at'].iso_format()

        return report_data
    return None

@unit_of_work(timeout=30) # type: ignore
def save_analysis_to_neo4j(tx, symbol_param: str, analysis_report_data: Dict[str, Any]):
    # analysis_report_data is the JSON from OpenAI plus your added fields.
    
    metadata_block = analysis_report_data.get("metadata", {}) # This is a dict
    analysis_block = analysis_report_data.get("analysis", {})   # This is a dict
    fmp_data_snapshot_block = analysis_report_data.get("fmp_data_for_analysis", {}) # This is a dict

    filling_date_str = metadata_block.get("fillingDate") 

    if not filling_date_str:
        logger.error(f"Cannot save analysis for {symbol_param} to Neo4j: missing fillingDate in metadata.")
        # Optionally raise an error here to prevent proceeding
        raise ValueError(f"Missing fillingDate in metadata for symbol {symbol_param} during Neo4j save.")

    # Merge Company Node
    company_cypher = (
        "MERGE (c:Company {symbol: $symbol_param}) "
        "ON CREATE SET c.companyName = $companyName, c.exchange = $exchange, c.sector = $sector, c.industry = $industry, c.lastUpdated = datetime() "
        "ON MATCH SET c.companyName = COALESCE(c.companyName, $companyName), c.exchange = COALESCE(c.exchange, $exchange), "
        "c.sector = COALESCE(c.sector, $sector), c.industry = COALESCE(c.industry, $industry), c.lastUpdated = datetime()"
    )
    company_params = {
        "symbol_param": symbol_param,
        "companyName": metadata_block.get("company_name"),
        "exchange": metadata_block.get("exchange"),
        "sector": metadata_block.get("sector"),
        "industry": metadata_block.get("industry")
    }
    # logger.debug(f"Executing Company Cypher with params: {company_params}") # Optional debug
    tx.run(company_cypher, **company_params)

    # Prepare parameters for AnalysisReport, serializing dicts to JSON strings
    params_for_ar = {
        "symbol_param": symbol_param, # Used in MERGE (ar) and MATCH (c)
        "filling_date_str": filling_date_str, # Used in MERGE (ar)
        
        "metadata_json_str": json.dumps(metadata_block), 
        "analysis_json_str": json.dumps(analysis_block), 
        "fmp_data_snapshot_json_str": json.dumps(fmp_data_snapshot_block, default=str), 

        "prompt_tokens": analysis_report_data.get("prompt_tokens"),
        "completion_tokens": analysis_report_data.get("completion_tokens"),
        "total_tokens": analysis_report_data.get("total_tokens"),
        "analysis_generated_at_str": analysis_report_data.get("analysis_generated_at"), 
        "model_used": analysis_report_data.get("model_used"),
        "symbol_processing_duration": analysis_report_data.get("symbol_processing_duration"),
        "calendarYear": analysis_report_data.get("calendarYear"),
    }

    # ---- START DETAILED LOGGING ----
    logger.info("--- Preparing to save AnalysisReport. Parameter types: ---")
    for key, value in params_for_ar.items():
        logger.info(f"Param: {key}, Type: {type(value)}, Value (first 100 chars if str): {str(value)[:100] if isinstance(value, str) else value}")
    logger.info("--- End of parameter types ---")
    # ---- END DETAILED LOGGING ----

    query_ar = (
        "MERGE (ar:AnalysisReport {symbol: $symbol_param, fillingDate: date($filling_date_str)}) "
        "ON CREATE SET "
        "  ar.metadata_json = $metadata_json_str, "
        "  ar.analysis_json = $analysis_json_str, "
        "  ar.fmp_data_snapshot_json = $fmp_data_snapshot_json_str, "
        "  ar.prompt_tokens = $prompt_tokens, "
        "  ar.completion_tokens = $completion_tokens, "
        "  ar.total_tokens = $total_tokens, "
        "  ar.analysis_generated_at = datetime($analysis_generated_at_str), "
        "  ar.model_used = $model_used, "
        "  ar.symbol_processing_duration = $symbol_processing_duration, "
        "  ar.calendarYear = $calendarYear, "
        "  ar.lastUpdated = datetime() "
        "ON MATCH SET "
        "  ar.metadata_json = $metadata_json_str, "
        "  ar.analysis_json = $analysis_json_str, "
        "  ar.fmp_data_snapshot_json = $fmp_data_snapshot_json_str, "
        "  ar.prompt_tokens = $prompt_tokens, "
        "  ar.completion_tokens = $completion_tokens, "
        "  ar.total_tokens = $total_tokens, "
        "  ar.analysis_generated_at = datetime($analysis_generated_at_str), "
        "  ar.model_used = $model_used, "
        "  ar.symbol_processing_duration = $symbol_processing_duration, "
        "  ar.calendarYear = $calendarYear, "
        "  ar.lastUpdated = datetime() "
        "WITH ar "
        "MATCH (c:Company {symbol: $symbol_param}) " # $symbol_param is used here
        "MERGE (c)-[:HAS_ANALYSIS_REPORT]->(ar) "
        "RETURN ar"
    )
    tx.run(query_ar, **params_for_ar) # This is where the error occurs during commit
    logger.info(f"Analysis for {symbol_param} (fillingDate: {filling_date_str}) saved to Neo4j (queued in transaction).") # Clarified logging
    return True
    

def process_symbol_logic(
    symbol_to_process: str,
    current_period_back_val: int,
    fdm_module_instance: FinancialDataModule,
    openai_client_instance: Optional[OpenAI],
    neo4j_driver_instance: Optional[Driver],
    app_config: Config
):
    start_ts = time.time()
    logger.info(f"Starting analysis for symbol: {symbol_to_process}, period_back: {current_period_back_val}")

    prospective_fmp_filling_date_str = None

    # STEP 1: Determine prospective_fmp_filling_date_str
    try:
        target_statements = fdm_module_instance.get_financial_statements(
            symbol=symbol_to_process, statement="income-statement",
            period_param=period, limit=1 + current_period_back_val
        )
        if target_statements and len(target_statements) > current_period_back_val:
            target_statement_for_date = target_statements[current_period_back_val]
            date_val = target_statement_for_date.get("fillingDate") or target_statement_for_date.get("date")
            if date_val: prospective_fmp_filling_date_str = date_val[:10]
        
        if prospective_fmp_filling_date_str:
            logger.info(f"Prospective FMP filling date for {symbol_to_process} is {prospective_fmp_filling_date_str}")
        else:
            logger.warning(f"Could not determine prospective FMP filling date for {symbol_to_process}. Cache check will be less targeted or skipped.")
    except Exception as e_date_fetch:
        logger.error(f"Error during initial FMP call for filling date for {symbol_to_process}: {e_date_fetch}.")
        prospective_fmp_filling_date_str = None # Ensure it's None

    # STEP 2: Check Neo4j cache
    if neo4j_driver_instance: # Check if driver is available before attempting to use it
        # Define transformer inside the conditional block to ensure neo4j_driver_instance is valid for its scope
        def transform_and_check_analysis_report(record_cursor_ar):
            single_record_ar = record_cursor_ar.single()
            if not single_record_ar or not single_record_ar["ar"]: return None
            
            report_node_data = dict(single_record_ar["ar"])
            metadata_dict = {}
            analysis_dict = {} # Initialize analysis_dict

            if 'metadata_json' in report_node_data and isinstance(report_node_data['metadata_json'], str):
                try: metadata_dict = json.loads(report_node_data['metadata_json'])
                except json.JSONDecodeError as e: logger.error(f"Cache: Error decoding metadata_json for {symbol_to_process}: {e}"); return None
            else: logger.warning(f"Cache: Missing or invalid metadata_json for {symbol_to_process}."); return None

            analysis_as_of_date_str = metadata_dict.get("as_of_date")
            report_node_filling_date_obj = report_node_data.get("fillingDate")
            report_node_filling_date_str = report_node_filling_date_obj.iso_format()[:10] if report_node_filling_date_obj and hasattr(report_node_filling_date_obj, 'iso_format') else None

            logger.info(f"Cache Candidate for {symbol_to_process}: Node.fillingDate={report_node_filling_date_str}, Metadata.as_of_date={analysis_as_of_date_str}")

            # Core comparison: Prospective FMP fillingDate vs. cached analysis's as_of_date
            if prospective_fmp_filling_date_str and analysis_as_of_date_str and \
               prospective_fmp_filling_date_str == analysis_as_of_date_str:
                logger.info(f"CACHE HIT: Prospective FMP Date ({prospective_fmp_filling_date_str}) matches Cached Analysis As-Of-Date ({analysis_as_of_date_str}).")
                
                # Reconstruct the full report object
                full_cached_report = {
                    "metadata": metadata_dict,
                    "analysis": {}, # Initialize
                    # Include other top-level fields from the node if they were stored directly
                    "prompt_tokens": report_node_data.get("prompt_tokens"),
                    "completion_tokens": report_node_data.get("completion_tokens"),
                    "total_tokens": report_node_data.get("total_tokens"),
                    "analysis_generated_at": report_node_data.get("analysis_generated_at").iso_format() if hasattr(report_node_data.get('analysis_generated_at'), 'iso_format') else None,
                    "model_used": report_node_data.get("model_used"),
                    "symbol_processing_duration_total": report_node_data.get("symbol_processing_duration"), # Assuming 'symbol_processing_duration' was the key on the node
                    "fmp_data_for_analysis": {} # Placeholder, or load if stored
                }

                if 'analysis_json' in report_node_data and isinstance(report_node_data['analysis_json'], str):
                    try: full_cached_report['analysis'] = json.loads(report_node_data['analysis_json'])
                    except json.JSONDecodeError as e: logger.error(f"Cache: Error decoding analysis_json for {symbol_to_process}: {e}"); # Keep analysis as {}
                
                # If fmp_data_snapshot_json was stored, load it
                if 'fmp_data_snapshot_json' in report_node_data and isinstance(report_node_data['fmp_data_snapshot_json'], str):
                    try: full_cached_report['fmp_data_for_analysis'] = json.loads(report_node_data['fmp_data_snapshot_json'])
                    except json.JSONDecodeError as e: logger.error(f"Cache: Error decoding fmp_data_snapshot_json for {symbol_to_process}: {e}");

                return full_cached_report
            
            logger.info(f"CACHE MISS or Date Mismatch for {symbol_to_process}.")
            return None

        cypher_for_cache_check = ""
        params_for_cache_check = {"symbol_param": symbol_to_process}
        if prospective_fmp_filling_date_str:
            cypher_for_cache_check = "MATCH (ar:AnalysisReport {symbol: $symbol_param, fillingDate: date($date_param)}) RETURN ar LIMIT 1"
            params_for_cache_check["date_param"] = prospective_fmp_filling_date_str
        else: # Fallback if prospective date couldn't be determined, get latest and check its as_of_date
            cypher_for_cache_check = "MATCH (ar:AnalysisReport {symbol: $symbol_param}) RETURN ar ORDER BY ar.fillingDate DESC LIMIT 1"

        existing_analysis_report = neo4j_driver_instance.execute_query(
            query_=cypher_for_cache_check, database_=None, 
            result_transformer_=transform_and_check_analysis_report, **params_for_cache_check
        )

        if existing_analysis_report: # If transformer returned a valid, reconstructed report
            logger.info(f"Using cached analysis report for {symbol_to_process}.")
            return existing_analysis_report # THIS IS THE CACHE HIT RETURN
    else:
        logger.info("Neo4j driver not available or prospective FMP date unknown. Skipping cache check.")


    logger.info(f"Proceeding with full FMP data fetch and OpenAI analysis for {symbol_to_process} (Prospective FMP Date: {prospective_fmp_filling_date_str or 'Unknown'}).")
    
    fmp_company_data = None
    actual_fmp_filling_date_str = None 

    try:
        # STEP 3: Fetch full FMP data
        fmp_company_data = fdm_module_instance.execute_all(
            symbol_param=symbol_to_process,
            current_period_back=current_period_back_val,
            current_period_type=period
        )
        
        date_val_pkg = (fmp_company_data.get("metadata_package", {}).get("fmp_filling_date") or
                       (fmp_company_data["income_statement"][0].get("fillingDate") if fmp_company_data.get("income_statement") and fmp_company_data["income_statement"][0] else None) or
                       (fmp_company_data["income_statement"][0].get("date") if fmp_company_data.get("income_statement") and fmp_company_data["income_statement"][0] else None))
        if not date_val_pkg:
            logger.error(f"CRITICAL: No FMP filling date in full package for {symbol_to_process}.")
            return {"status": "fmp_error_no_date_full_pkg", "symbol": symbol_to_process, "fmp_data": fmp_company_data or {}}
        actual_fmp_filling_date_str = date_val_pkg[:10]
        logger.info(f"Actual FMP filling date from full package for {symbol_to_process} is {actual_fmp_filling_date_str}")

        # Optional: Save raw FMP data to disk
        # ... (code to save fmp_company_data to file) ...

        # STEP 4: OpenAI Analysis
        if not openai_client_instance:
            logger.warning(f"OpenAI client not available. Skipping OpenAI for {symbol_to_process}.")
            return {"status": "data_only_no_openai", "symbol": symbol_to_process, "fmp_data": fmp_company_data}

        question = f"Perform a detailed {period} financial and fundamental analysis for ({symbol_to_process}) company using the provided data."
        instructions = """
        Perform a detailed annual financial and fundamental analysis for a company using the provided data."
        # Use the instruction from your current __main__ block
        instruction = f'''You are a financial analyst specializing in detailed financial analysis and valuation.
        Utilize the provided **annual** financial data (income statements, balance sheets, cash flow statements), peer metrics, and company profile (found in the user message) to perform a thorough **annual** financial and fundamental analysis.
        The analysis must address the company’s growth trajectory, operational efficiency, and valuation against peers.
        Be precise, unbiased, and objective in your analysis. Avoid being overly bullish or bearish.
        Consider last_price currency. All textual content within the JSON should use standard **ASCII** characters only; **avoid** non-standard Unicode characters (e.g \u2013, \u2019 etc...).
        For valuation models requiring assumptions (like DCF discount rate, growth rates, etc.) that are not explicitly provided, state the reasonable assumptions you are making. Base calculations *only* on the provided data context and your financial expertise.
        **Rely in your response only on context provided to you**.
        **Strictly return a single JSON object that conforms precisely to the following JSON structure. Do not include any other text, markdown, or explanations outside the JSON structure.**
        **do not add new line character (\n)**
        JSON:
        {
  "metadata": {
    "company_name": "Full legal name of the company. (Guidance: Fill from 'metadata.company_name' in the provided data. If data is missing, set as an empty value of this type.)",
    "ticker": "Stock ticker symbol. (Guidance: Fill from 'metadata.ticker' in the provided data. If data is missing, set as an empty value of this type.)",
    "exchange": "Stock exchange where the ticker is listed. (Guidance: Fill from 'metadata.exchange' in the provided data. If data is missing, set as an empty value of this type.)",
    "industry": "Specific industry classification (e.g., Semiconductors). (Guidance: Fill from 'metadata.industry' in the provided data. If data is missing, set as an empty value of this type.)",
    "sector": "Broader sector classification (e.g., Technology). (Guidance: Fill from 'metadata.sector' in the provided data. If data is missing, set as an empty value of this type.)",
    "currency": "Reporting currency (e.g., USD). (Guidance: Fill from 'metadata.currency' in the provided data. If data is missing, set as an empty value of this type.)",
    "as_of_date": "Date of last price/data used for snapshot ratios/analysis (YYYY-MM-DD). (Guidance: Fill from 'metadata.as_of_date' in the provided data. If data is missing, set as an empty value of this type.)",
    "last_price": "Last closing stock price as of 'as_of_date'. (Guidance: Fill from 'metadata.last_price' in the provided data. If data is missing, set as an empty value of this type.)",
    "data_range_years": "Period covered by time-series analysis (e.g., 2021–2025). (Guidance: Fill from 'metadata.data_range_years' in the provided data. If data is missing, set as an empty value of this type.)",
    "analysis_generation_date": "Timestamp when this analysis report was generated. (Guidance: Fill from 'metadata.analysis_generation_date' in the provided data. If data is missing, set as an empty value of this type.)",
    "sec_filing_link": "Link to the latest relevant SEC filing (e.g., 10-K). (Guidance: Fill from 'metadata.sec_filing_link' in the provided data. If data is missing, set as an empty value of this type.)",
    "fillingDate": "fillingDate from source data as (YYY-MM-DD)""
  },
  "analysis": {
    "financial_performance": {
      "revenues": {
        "values": [
          {
            "period": "The periods label (e.g., '2025', 'FY2025', '2025-01-26').",
            "value": "The metric's value for all provided periods. Null if not available."
          }
        ],
        "explanation": "Narrative analysis of the trend over the period. (Guidance: Fill from 'analysis.financial_performance.revenues.explanation' in the provided data. If data is missing, set as an empty value of this type.)",
        "classification": "Sentiment classification of the associated analysis, explanation, or data point regarding its implication for the company. (Guidance: Fill from 'analysis.financial_performance.revenues.classification' in the provided data. If data is missing, set as an empty value of this type.)"
      },
      "revenue_growth_rates": {
        "values": [
          {
            "period": "The periods label (e.g., '2025', 'FY2025', '2025-01-26').",
            "growth_percent": "Year-over-year growth rate as a percentage (e.g., 15.2 for 15.2% growth). Null if not available."
          }
        ],
        "explanation": "Narrative analysis of the growth rate trend over the period. (Guidance: Fill from 'analysis.financial_performance.revenue_growth_rates.explanation' in the provided data. If data is missing, set as an empty value of this type.)",
        "classification": "Sentiment classification. (Guidance: Fill from 'analysis.financial_performance.revenue_growth_rates.classification' in the provided data. If data is missing, set as an empty value of this type.)"
      },
      "revenue_cagr_calculated": {
        "rate_percent": "",
        "period_years": "",
        "start_year": "",
        "start_value": "",
        "end_year": "",
        "end_value": "",
        "calculation_note": "",
        "classification": ""
      },
      "gross_margin": {
        "values": [
          {
            "period": "The period label.",
            "value_percent": "The metric's value as a percentage (e.g., 74.5 for 74.5%). Null if not available."
          }
        ],
        "explanation": "Narrative analysis of the percentage trend over the period. (Guidance: Fill from 'analysis.financial_performance.gross_margin.explanation' in the provided data. If data is missing, set as an empty value of this type.)",
        "classification": "Sentiment classification. (Guidance: Fill from 'analysis.financial_performance.gross_margin.classification' in the provided data. If data is missing, set as an empty value of this type.)"
      },
      "ebitda_margin": {
        "values": [
          {
            "period": "The periods label (e.g., '2025', 'FY2025', '2025-01-26').",
            "value_percent": "The metric's value as a percentage (e.g., 74.5 for 74.5%) for all periods"
          }
        ],
        "explanation": "Field 'explanation' at path 'analysis.financial_performance.ebitda_margin.explanation'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "classification": "Field 'classification' at path 'analysis.financial_performance.ebitda_margin.classification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
      },
      "net_margin": {
        "values": [
          {
            "period": "The periods labels (e.g., '2025', '2026', '2027' etc...).",
            "value_percent": "values"
          }
        ],
        "explanation": "Field 'explanation' at path 'analysis.financial_performance.net_margin.explanation'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "classification": "Field 'classification' at path 'analysis.financial_performance.net_margin.classification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
      },
      "profitability_metrics": {
        "ebitda": {
          "values": [
            {
              "period": "The period labels (e.g., '2025', '2026', '2027' etc...).",
              "value": "The metric's value for the all periods."
            }
          ],
          "explanation": "Field 'explanation' at path 'analysis.financial_performance.profitability_metrics.ebitda.explanation'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "classification": "Field 'classification' at path 'analysis.financial_performance.profitability_metrics.ebitda.classification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
        },
        "operating_income": {
          "values": [
            {
              "period": "The period label (e.g., '2020', '2021', '2022').",
              "value": "The metric's value for the all periods.. Null if not available."
            }
          ],
          "explanation": "Field 'explanation' at path 'analysis.financial_performance.profitability_metrics.operating_income.explanation'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "classification": "Field 'classification' at path 'analysis.financial_performance.profitability_metrics.operating_income.classification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
        },
        "net_income": {
          "values": [
            {
              "period": "The period label (e.g., '2025', 'FY2025', '2025-01-26').",
              "value": "The metric's value for the period. Null if not available."
            }
          ],
          "explanation": "Field 'explanation' at path 'analysis.financial_performance.profitability_metrics.net_income.explanation'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "classification": "Field 'classification' at path 'analysis.financial_performance.profitability_metrics.net_income.classification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
        },
        "eps_diluted": {
          "values": [
            {
              "period": "The period label (e.g., '2025', 'FY2025', '2025-01-26').",
              "value": "The metric's value for the period. Null if not available."
            }
          ],
          "explanation": "Field 'explanation' at path 'analysis.financial_performance.profitability_metrics.eps_diluted.explanation'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "classification": "Field 'classification' at path 'analysis.financial_performance.profitability_metrics.eps_diluted.classification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
        },
        "net_income_cagr_calculated": {
          "rate_percent": "",
          "period_years": "",
          "start_year": "",
          "start_value": "",
          "end_year": "",
          "end_value": "",
          "calculation_note": "",
          "classification": ""
        },
        "eps_diluted_cagr_calculated": {
          "rate_percent": "",
          "period_years": "",
          "start_year": "",
          "start_value": "",
          "end_year": "",
          "end_value": "",
          "calculation_note": "",
          "classification": ""
        },
        "roa": {
          "values": [
            {
              "period": "The period label.",
              "value_percent": "The metric's value as a percentage (e.g., 15.2 for 15.2%). Null if not available."
            }
          ],
          "explanation": "Field 'explanation' at path 'analysis.financial_performance.profitability_metrics.roa.explanation'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "classification": "Field 'classification' at path 'analysis.financial_performance.profitability_metrics.roa.classification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
        },
        "roic": {
          "values": [
            {
              "period": "The period label.",
              "value_percent": "The metric's value as a percentage (e.g., 15.2 for 15.2%). Null if not available."
            }
          ],
          "explanation": "Field 'explanation' at path 'analysis.financial_performance.profitability_metrics.roic.explanation'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "classification": "Field 'classification' at path 'analysis.financial_performance.profitability_metrics.roic.classification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
        }
      },
      "profitability_summary": "Overall commentary on profitability trends and drivers, referencing calculated CAGRs. (Guidance: Fill from 'analysis.financial_performance.profitability_summary' in the provided data. If data is missing, set as an empty value of this type.)",
      "profitability_summary_classification": "Sentiment classification. (Guidance: Fill from 'analysis.financial_performance.profitability_summary_classification' in the provided data. If data is missing, set as an empty value of this type.)",
      "debt_and_liquidity": {
        "current_ratio": {
          "values": [
            {
              "period": "The period label.",
              "value": "The metric's value for the period. Null if not available."
            }
          ],
          "explanation": "Field 'explanation' at path 'analysis.financial_performance.debt_and_liquidity.current_ratio.explanation'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "classification": "Field 'classification' at path 'analysis.financial_performance.debt_and_liquidity.current_ratio.classification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
        },
        "debt_to_equity": {
          "values": [
            {
              "period": "The period label.",
              "value": "The metric's value for the period. Null if not available."
            }
          ],
          "explanation": "Field 'explanation' at path 'analysis.financial_performance.debt_and_liquidity.debt_to_equity.explanation'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "classification": "Field 'classification' at path 'analysis.financial_performance.debt_and_liquidity.debt_to_equity.classification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
        },
        "interest_coverage_ratio": {
          "values": [
            {
              "period": "The period label.",
              "value": "The metric's value for the period. Null if not available."
            }
          ],
          "explanation": "Field 'explanation' at path 'analysis.financial_performance.debt_and_liquidity.interest_coverage_ratio.explanation'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "classification": "Field 'classification' at path 'analysis.financial_performance.debt_and_liquidity.interest_coverage_ratio.classification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
        }
      },
      "cash_generation": {
        "operating_cash_flow": {
          "values": [
            {
              "period": "The period label (e.g., '2025', 'FY2025', '2025-01-26').",
              "value": "The metric's value for the period. Null if not available."
            }
          ],
          "explanation": "Field 'explanation' at path 'analysis.financial_performance.cash_generation.operating_cash_flow.explanation'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "classification": "Field 'classification' at path 'analysis.financial_performance.cash_generation.operating_cash_flow.classification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
        },
        "free_cash_flow": {
          "values": [
            {
              "period": "The period label (e.g., '2025', 'FY2025', '2025-01-26').",
              "value": "The metric's value for the period. Null if not available."
            }
          ],
          "explanation": "Field 'explanation' at path 'analysis.financial_performance.cash_generation.free_cash_flow.explanation'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "classification": "Field 'classification' at path 'analysis.financial_performance.cash_generation.free_cash_flow.classification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
        },
        "fcf_cagr_calculated": {
          "rate_percent": "",
          "period_years": "",
          "start_year": "",
          "start_value": "",
          "end_year": "",
          "end_value": "",
          "calculation_note": "",
          "classification": ""
        },
        "fcf_margin": {
          "values": [
            {
              "period": "The period label.",
              "value_percent": "The metric's value as a percentage (e.g., 74.5 for 74.5%). Null if not available."
            }
          ],
          "explanation": "Field 'explanation' at path 'analysis.financial_performance.cash_generation.fcf_margin.explanation'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "classification": "Field 'classification' at path 'analysis.financial_performance.cash_generation.fcf_margin.classification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
        },
        "capex_as_percent_of_revenue": {
          "values": [
            {
              "period": "The period label.",
              "value_percent": "The metric's value as a percentage. Null if not available."
            }
          ],
          "explanation": "Field 'explanation' at path 'analysis.financial_performance.cash_generation.capex_as_percent_of_revenue.explanation'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "classification": "Field 'classification' at path 'analysis.financial_performance.cash_generation.capex_as_percent_of_revenue.classification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
        }
      }
    },
    "fundamental_analysis": {
      "market_capitalization": "Field 'market_capitalization' at path 'analysis.fundamental_analysis.market_capitalization'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
      "enterprise_value_ttm": "Field 'enterprise_value_ttm' at path 'analysis.fundamental_analysis.enterprise_value_ttm'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
      "beta": "Field 'beta' at path 'analysis.fundamental_analysis.beta'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
      "shares_outstanding": "Field 'shares_outstanding' at path 'analysis.fundamental_analysis.shares_outstanding'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
      "valuation_ratios": {
        "pe_ratio_ttm": "Field 'pe_ratio_ttm' at path 'analysis.fundamental_analysis.valuation_ratios.pe_ratio_ttm'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "price_to_sales_ratio_ttm": "Field 'price_to_sales_ratio_ttm' at path 'analysis.fundamental_analysis.valuation_ratios.price_to_sales_ratio_ttm'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "price_to_book_ratio_ttm": "Field 'price_to_book_ratio_ttm' at path 'analysis.fundamental_analysis.valuation_ratios.price_to_book_ratio_ttm'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "ev_to_sales_ttm": "Field 'ev_to_sales_ttm' at path 'analysis.fundamental_analysis.valuation_ratios.ev_to_sales_ttm'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "ev_to_ebitda_ttm": "Field 'ev_to_ebitda_ttm' at path 'analysis.fundamental_analysis.valuation_ratios.ev_to_ebitda_ttm'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "earnings_yield_ttm": "Field 'earnings_yield_ttm' at path 'analysis.fundamental_analysis.valuation_ratios.earnings_yield_ttm'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "free_cash_flow_yield_ttm": "Field 'free_cash_flow_yield_ttm' at path 'analysis.fundamental_analysis.valuation_ratios.free_cash_flow_yield_ttm'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "peg_ratio_ttm": "Trailing Twelve Months Price/Earnings to Growth Ratio. (Guidance: Fill from 'analysis.fundamental_analysis.valuation_ratios.peg_ratio_ttm' in the provided data. If data is missing, set as an empty value of this type.)"
      },
      "profitability_ratios": {
        "roe_ttm": "Field 'roe_ttm' at path 'analysis.fundamental_analysis.profitability_ratios.roe_ttm'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "roa_ttm": "Field 'roa_ttm' at path 'analysis.fundamental_analysis.profitability_ratios.roa_ttm'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "roic_ttm": "Field 'roic_ttm' at path 'analysis.fundamental_analysis.profitability_ratios.roic_ttm'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "gross_margin_ttm": "Trailing Twelve Months Gross Profit Margin. (Guidance: Fill from 'analysis.fundamental_analysis.profitability_ratios.gross_margin_ttm' in the provided data. If data is missing, set as an empty value of this type.)",
        "ebitda_margin_ttm": "Trailing Twelve Months EBITDA Margin. (Guidance: Fill from 'analysis.fundamental_analysis.profitability_ratios.ebitda_margin_ttm' in the provided data. If data is missing, set as an empty value of this type.)",
        "net_margin_ttm": "Trailing Twelve Months Net Income Margin. (Guidance: Fill from 'analysis.fundamental_analysis.profitability_ratios.net_margin_ttm' in the provided data. If data is missing, set as an empty value of this type.)"
      },
      "liquidity_and_solvency_ratios": {
        "debt_to_equity_ttm": "Field 'debt_to_equity_ttm' at path 'analysis.fundamental_analysis.liquidity_and_solvency_ratios.debt_to_equity_ttm'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "debt_to_assets_ttm": "Field 'debt_to_assets_ttm' at path 'analysis.fundamental_analysis.liquidity_and_solvency_ratios.debt_to_assets_ttm'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "net_debt_to_ebitda_ttm": "Field 'net_debt_to_ebitda_ttm' at path 'analysis.fundamental_analysis.liquidity_and_solvency_ratios.net_debt_to_ebitda_ttm'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "current_ratio_ttm": "Field 'current_ratio_ttm' at path 'analysis.fundamental_analysis.liquidity_and_solvency_ratios.current_ratio_ttm'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "interest_coverage_ttm": "Field 'interest_coverage_ttm' at path 'analysis.fundamental_analysis.liquidity_and_solvency_ratios.interest_coverage_ttm'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
      },
      "efficiency_ratios": {
        "days_sales_outstanding_ttm": "Field 'days_sales_outstanding_ttm' at path 'analysis.fundamental_analysis.efficiency_ratios.days_sales_outstanding_ttm'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "days_inventory_on_hand_ttm": "Field 'days_inventory_on_hand_ttm' at path 'analysis.fundamental_analysis.efficiency_ratios.days_inventory_on_hand_ttm'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "days_payables_outstanding_ttm": "Field 'days_payables_outstanding_ttm' at path 'analysis.fundamental_analysis.efficiency_ratios.days_payables_outstanding_ttm'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "cash_conversion_cycle_ttm": "Field 'cash_conversion_cycle_ttm' at path 'analysis.fundamental_analysis.efficiency_ratios.cash_conversion_cycle_ttm'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "asset_turnover_ttm": "Trailing Twelve Months Asset Turnover. (Guidance: Fill from 'analysis.fundamental_analysis.efficiency_ratios.asset_turnover_ttm' in the provided data. If data is missing, set as an empty value of this type.)"
      },
      "growth_metrics_ttm": {
        "revenue_growth_yoy_ttm": "TTM Year-over-Year Revenue Growth Rate (%). (Guidance: Fill from 'analysis.fundamental_analysis.growth_metrics_ttm.revenue_growth_yoy_ttm' in the provided data. If data is missing, set as an empty value of this type.)",
        "ebitda_growth_yoy_ttm": "TTM Year-over-Year EBITDA Growth Rate (%). (Guidance: Fill from 'analysis.fundamental_analysis.growth_metrics_ttm.ebitda_growth_yoy_ttm' in the provided data. If data is missing, set as an empty value of this type.)",
        "eps_diluted_growth_yoy_ttm": "TTM Year-over-Year Diluted EPS Growth Rate (%). (Guidance: Fill from 'analysis.fundamental_analysis.growth_metrics_ttm.eps_diluted_growth_yoy_ttm' in the provided data. If data is missing, set as an empty value of this type.)"
      },
      "industry_sector_comparison": {
        "sector_pe_average": "Field 'sector_pe_average' at path 'analysis.fundamental_analysis.industry_sector_comparison.sector_pe_average'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "industry_pe_average": "Field 'industry_pe_average' at path 'analysis.fundamental_analysis.industry_sector_comparison.industry_pe_average'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "commentary": "Field 'commentary' at path 'analysis.fundamental_analysis.industry_sector_comparison.commentary'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "commentary_classification": "Field 'commentary_classification' at path 'analysis.fundamental_analysis.industry_sector_comparison.commentary_classification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
      }
    },
    "growth_prospects": {
      "historical_growth_summary": {
        "revenue_cagr": {
          "rate_percent": "",
          "period_years": "",
          "start_year": "",
          "start_value": "",
          "end_year": "",
          "end_value": "",
          "calculation_note": "",
          "classification": ""
        },
        "net_income_cagr": {
          "rate_percent": "",
          "period_years": "",
          "start_year": "",
          "start_value": "",
          "end_year": "",
          "end_value": "",
          "calculation_note": "",
          "classification": ""
        },
        "eps_diluted_cagr": {
          "rate_percent": "",
          "period_years": "",
          "start_year": "",
          "start_value": "",
          "end_year": "",
          "end_value": "",
          "calculation_note": "",
          "classification": ""
        },
        "fcf_cagr": {
          "rate_percent": "",
          "period_years": "",
          "start_year": "",
          "start_value": "",
          "end_year": "",
          "end_value": "",
          "calculation_note": "",
          "classification": ""
        }
      },
      "future_drivers_commentary": "Field 'future_drivers_commentary' at path 'analysis.growth_prospects.future_drivers_commentary'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
      "future_drivers_commentary_classification": "Field 'future_drivers_commentary_classification' at path 'analysis.growth_prospects.future_drivers_commentary_classification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
      "analyst_growth_expectations": {
        "source": "Field 'source' at path 'analysis.growth_prospects.analyst_growth_expectations.source'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "eps_growth_next_year_percent": "Field 'eps_growth_next_year_percent' at path 'analysis.growth_prospects.analyst_growth_expectations.eps_growth_next_year_percent'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "eps_growth_5yr_annualized_percent": "Field 'eps_growth_5yr_annualized_percent' at path 'analysis.growth_prospects.analyst_growth_expectations.eps_growth_5yr_annualized_percent'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "revenue_growth_next_year_percent": "Field 'revenue_growth_next_year_percent' at path 'analysis.growth_prospects.analyst_growth_expectations.revenue_growth_next_year_percent'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
      }
    },
    "competitive_position": {
      "market_share_overview": "Field 'market_share_overview' at path 'analysis.competitive_position.market_share_overview'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
      "market_share_overview_classification": "Field 'market_share_overview_classification' at path 'analysis.competitive_position.market_share_overview_classification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
      "competitive_advantages": "Field 'competitive_advantages' at path 'analysis.competitive_position.competitive_advantages'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
      "competitive_advantages_classification": "Field 'competitive_advantages_classification' at path 'analysis.competitive_position.competitive_advantages_classification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
      "key_competitors": [
        {
          "name": "Field 'name' at path 'analysis.competitive_position.key_competitors.name'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "ticker": "Field 'ticker' at path 'analysis.competitive_position.key_competitors.ticker'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
        }
      ]
    },
    "peers_comparison": {
      "peer_group_used": [
        ""
      ],
      "comparison_table": [
        {
          "metric_name": "Field 'metric_name' at path 'analysis.peers_comparison.comparison_table.metric_name'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "company_value": "Field 'company_value' at path 'analysis.peers_comparison.comparison_table.company_value'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "peer_values": "Field 'peer_values' at path 'analysis.peers_comparison.comparison_table.peer_values'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "unit": "Field 'unit' at path 'analysis.peers_comparison.comparison_table.unit'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
        }
      ],
      "relative_positioning_summary": "Field 'relative_positioning_summary' at path 'analysis.peers_comparison.relative_positioning_summary'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
      "relative_positioning_summary_classification": "Field 'relative_positioning_summary_classification' at path 'analysis.peers_comparison.relative_positioning_summary_classification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
    },
    "revenue_segmentation": {
      "latest_fiscal_year": "Field 'latest_fiscal_year' at path 'analysis.revenue_segmentation.latest_fiscal_year'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
      "geographic_breakdown": [
        {
          "region": "Field 'region' at path 'analysis.revenue_segmentation.geographic_breakdown.region'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "revenue_amount": "Field 'revenue_amount' at path 'analysis.revenue_segmentation.geographic_breakdown.revenue_amount'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "revenue_percentage": "Field 'revenue_percentage' at path 'analysis.revenue_segmentation.geographic_breakdown.revenue_percentage'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
        }
      ],
      "product_breakdown": [
        {
          "segment_name": "Field 'segment_name' at path 'analysis.revenue_segmentation.product_breakdown.segment_name'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "revenue_amount": "Field 'revenue_amount' at path 'analysis.revenue_segmentation.product_breakdown.revenue_amount'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "revenue_percentage": "Field 'revenue_percentage' at path 'analysis.revenue_segmentation.product_breakdown.revenue_percentage'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
        }
      ],
      "segmentation_trends_commentary": "Field 'segmentation_trends_commentary' at path 'analysis.revenue_segmentation.segmentation_trends_commentary'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
      "segmentation_trends_commentary_classification": "Field 'segmentation_trends_commentary_classification' at path 'analysis.revenue_segmentation.segmentation_trends_commentary_classification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
    },
    "risk_factors": {
      "macroeconomic_risks": [
        {
          "item_text": "",
          "classification": ""
        }
      ],
      "industry_competitive_risks": [
        {
          "item_text": "",
          "classification": ""
        }
      ],
      "operational_execution_risks": [
        {
          "item_text": "",
          "classification": ""
        }
      ],
      "financial_risks": [
        {
          "item_text": "",
          "classification": ""
        }
      ],
      "regulatory_legal_risks": [
        {
          "item_text": "",
          "classification": ""
        }
      ],
      "esg_related_risks": [
        {
          "item_text": "",
          "classification": ""
        }
      ]
    },
    "valuation_analysis": {
      "dcf_valuation": {
        "key_assumptions": {
          "forecast_period_years": "Field 'forecast_period_years' at path 'analysis.valuation_analysis.dcf_valuation.key_assumptions.forecast_period_years'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "risk_free_rate_percent": "Field 'risk_free_rate_percent' at path 'analysis.valuation_analysis.dcf_valuation.key_assumptions.risk_free_rate_percent'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "beta_used": "Field 'beta_used' at path 'analysis.valuation_analysis.dcf_valuation.key_assumptions.beta_used'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "equity_risk_premium_percent": "Field 'equity_risk_premium_percent' at path 'analysis.valuation_analysis.dcf_valuation.key_assumptions.equity_risk_premium_percent'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "wacc_percent": "Field 'wacc_percent' at path 'analysis.valuation_analysis.dcf_valuation.key_assumptions.wacc_percent'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "terminal_growth_rate_percent": "Field 'terminal_growth_rate_percent' at path 'analysis.valuation_analysis.dcf_valuation.key_assumptions.terminal_growth_rate_percent'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "revenue_growth_assumptions": "Field 'revenue_growth_assumptions' at path 'analysis.valuation_analysis.dcf_valuation.key_assumptions.revenue_growth_assumptions'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
          "margin_assumptions": "Field 'margin_assumptions' at path 'analysis.valuation_analysis.dcf_valuation.key_assumptions.margin_assumptions'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
        },
        "intrinsic_value_per_share": "Field 'intrinsic_value_per_share' at path 'analysis.valuation_analysis.dcf_valuation.intrinsic_value_per_share'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
      },
      "comparable_analysis_valuation": {
        "peer_group_used": [
          ""
        ],
        "key_multiple_used": "Field 'key_multiple_used' at path 'analysis.valuation_analysis.comparable_analysis_valuation.key_multiple_used'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "multiple_value": "Field 'multiple_value' at path 'analysis.valuation_analysis.comparable_analysis_valuation.multiple_value'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "implied_metric_value": "Field 'implied_metric_value' at path 'analysis.valuation_analysis.comparable_analysis_valuation.implied_metric_value'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "intrinsic_value_per_share": "Field 'intrinsic_value_per_share' at path 'analysis.valuation_analysis.comparable_analysis_valuation.intrinsic_value_per_share'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
      },
      "valuation_summary": {
        "fair_value_estimate_low": "Field 'fair_value_estimate_low' at path 'analysis.valuation_analysis.valuation_summary.fair_value_estimate_low'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "fair_value_estimate_base": "Field 'fair_value_estimate_base' at path 'analysis.valuation_analysis.valuation_summary.fair_value_estimate_base'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "fair_value_estimate_high": "Field 'fair_value_estimate_high' at path 'analysis.valuation_analysis.valuation_summary.fair_value_estimate_high'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "current_price_vs_fair_value": "Field 'current_price_vs_fair_value' at path 'analysis.valuation_analysis.valuation_summary.current_price_vs_fair_value'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "summary_commentary": "Field 'summary_commentary' at path 'analysis.valuation_analysis.valuation_summary.summary_commentary'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "summary_commentary_classification": "Field 'summary_commentary_classification' at path 'analysis.valuation_analysis.valuation_summary.summary_commentary_classification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
      }
    },
    "scenario_analysis": {
      "base_case_value_per_share": "Field 'base_case_value_per_share' at path 'analysis.scenario_analysis.base_case_value_per_share'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
      "bull_case": {
        "key_driver_changes": "Field 'key_driver_changes' at path 'analysis.scenario_analysis.bull_case.key_driver_changes'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "value_per_share": "Field 'value_per_share' at path 'analysis.scenario_analysis.bull_case.value_per_share'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
      },
      "bear_case": {
        "key_driver_changes": "Field 'key_driver_changes' at path 'analysis.scenario_analysis.bear_case.key_driver_changes'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
        "value_per_share": "Field 'value_per_share' at path 'analysis.scenario_analysis.bear_case.value_per_share'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
      }
    },
    "shareholder_returns_analysis": {
      "dividend_yield_annualized_percent": "Field 'dividend_yield_annualized_percent' at path 'analysis.shareholder_returns_analysis.dividend_yield_annualized_percent'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
      "dividend_payout_ratio_ttm_percent": "Field 'dividend_payout_ratio_ttm_percent' at path 'analysis.shareholder_returns_analysis.dividend_payout_ratio_ttm_percent'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
      "share_repurchase_yield_ttm_percent": "Field 'share_repurchase_yield_ttm_percent' at path 'analysis.shareholder_returns_analysis.share_repurchase_yield_ttm_percent'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
      "total_shareholder_yield_ttm_percent": "Field 'total_shareholder_yield_ttm_percent' at path 'analysis.shareholder_returns_analysis.total_shareholder_yield_ttm_percent'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
      "capital_allocation_commentary": "Field 'capital_allocation_commentary' at path 'analysis.shareholder_returns_analysis.capital_allocation_commentary'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
      "capital_allocation_commentary_classification": "Field 'capital_allocation_commentary_classification' at path 'analysis.shareholder_returns_analysis.capital_allocation_commentary_classification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
    },
    "investment_thesis_summary": {
      "overall_recommendation": "Field 'overall_recommendation' at path 'analysis.investment_thesis_summary.overall_recommendation'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
      "price_target": "Field 'price_target' at path 'analysis.investment_thesis_summary.price_target'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
      "time_horizon": "Field 'time_horizon' at path 'analysis.investment_thesis_summary.time_horizon'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
      "key_investment_positives": [
        {
          "item_text": "",
          "classification": ""
        }
      ],
      "key_investment_risks": [
        {
          "item_text": "",
          "classification": ""
        }
      ],
      "final_justification": "Field 'final_justification' at path 'analysis.investment_thesis_summary.final_justification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type.",
      "final_justification_classification": "Field 'final_justification_classification' at path 'analysis.investment_thesis_summary.final_justification_classification'. Fill this field using the provided data. Do not skip. If data is missing, return an empty value for its type."
    }
  }
}"""
        fmp_company_data_string = json.dumps(fmp_company_data, ensure_ascii=False, default=str)
        
        logger.info(f"Requesting OpenAI analysis for {symbol_to_process} (FMP fillingDate: {actual_fmp_filling_date_str})...")
        generated_analysis_json = None 
        response_obj_for_metadata = None # To store the successful response object

        try:
            response = openai_client_instance.chat.completions.create(
                model=getattr(app_config, 'openai_model', 'gpt-4o-mini'),
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": f"{period.capitalize()} Company financial information (JSON):\n{fmp_company_data_string}\n\nQuestion: {question}"}
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=8192
            )
            response_obj_for_metadata = response # Save for metadata if successful
            message_content = response.choices[0].message.content
            if message_content:
                try: 
                    generated_analysis_json = json.loads(message_content)
                    logger.info(f"Successfully parsed OpenAI JSON response for {symbol_to_process}.")
                except json.JSONDecodeError as jde:
                    logger.error(f"OpenAI: Invalid JSON for {symbol_to_process}: {jde}. Resp: {message_content[:200]}");
                    generated_analysis_json = {"error_openai_json": str(jde), "raw_response": message_content}
            else:
                logger.error(f"OpenAI: Empty response for {symbol_to_process}. Finish: {response.choices[0].finish_reason}");
                generated_analysis_json = {"error_openai_empty": f"Empty. Reason: {response.choices[0].finish_reason}"}

        except (APIError, APIConnectionError, RateLimitError, BadRequestError) as e_api:
            logger.error(f"OpenAI API error for {symbol_to_process}: {e_api}", exc_info=False)
            generated_analysis_json = {"error_openai_api": str(e_api)}
        except Exception as e_openai_proc:
            logger.error(f"OpenAI processing error for {symbol_to_process}: {e_openai_proc}", exc_info=True)
            generated_analysis_json = {"error_openai_processing": str(e_openai_proc)}
        
        if generated_analysis_json is None:
             generated_analysis_json = {"error_openai_unknown": "Analysis was not populated after call attempt."}

        # Add metadata to generated_analysis_json
        if isinstance(generated_analysis_json, dict):
            if not any(k.startswith("error_") for k in generated_analysis_json) and response_obj_for_metadata:
                # Only add these if OpenAI call was successful and we have the response object
                generated_analysis_json['prompt_tokens'] = response_obj_for_metadata.usage.prompt_tokens
                generated_analysis_json['completion_tokens'] = response_obj_for_metadata.usage.completion_tokens
                generated_analysis_json['total_tokens'] = response_obj_for_metadata.usage.total_tokens
                generated_analysis_json['model_used'] = response_obj_for_metadata.model
            
            generated_analysis_json['analysis_generated_at'] = dt_class.now(datetime.timezone.utc).isoformat()
            generated_analysis_json['symbol_processing_duration_total'] = time.time() - start_ts
            
            current_meta = generated_analysis_json.get('metadata', {})
            if not isinstance(current_meta, dict): current_meta = {}
            current_meta['ticker'] = symbol_to_process
            current_meta['fillingDate'] = actual_fmp_filling_date_str # For Neo4j node key
            current_meta['as_of_date'] = actual_fmp_filling_date_str  # For cache comparison consistency
            current_meta['calendarYear'] = fmp_company_data.get("metadata_package", {}).get("fmp_calendar_year")
            generated_analysis_json['metadata'] = current_meta
            
            generated_analysis_json['fmp_data_for_analysis'] = fmp_company_data
        
        # ... (Optional save generated_analysis_json to disk) ...

        # STEP 5: Save new analysis to Neo4j (only if no errors from OpenAI)
        if neo4j_driver_instance and isinstance(generated_analysis_json, dict) and \
           not any(k.startswith("error_") for k in generated_analysis_json): # Check for any error key
            try:
                with neo4j_driver_instance.session(database_=None) as session:
                    session.execute_write(
                        save_analysis_to_neo4j,
                        symbol_param=symbol_to_process,
                        analysis_report_data=generated_analysis_json
                    )
            except Exception as e_neo_save:
                logger.error(f"Error saving NEW analysis to Neo4j for {symbol_to_process}: {e_neo_save}", exc_info=True)
                if isinstance(generated_analysis_json, dict): generated_analysis_json['error_neo4j_save'] = str(e_neo_save)
        
        if isinstance(generated_analysis_json, dict) and not any(k.startswith("error_") for k in generated_analysis_json):
             logger.info(f"SUCCESS (new analysis by OpenAI): {symbol_to_process} processed in {time.time() - start_ts:.2f} seconds.")
        else:
             logger.warning(f"ISSUES processing {symbol_to_process}. Final result object: {str(generated_analysis_json)[:200]}...")
        return generated_analysis_json

    except RuntimeError as e_runtime:
        logger.error(f"RUNTIME ERROR for {symbol_to_process} during full FMP fetch: {e_runtime}", exc_info=True)
        return {"status": "runtime_error_fmp_full", "symbol": symbol_to_process, "error": str(e_runtime), "fmp_data_on_error": fmp_company_data}
    except Exception as e_main:
        logger.error(f"OVERALL FAILED for {symbol_to_process} in main block: {e_main}", exc_info=True)
        return {"status": "overall_failure_main_process", "symbol": symbol_to_process, "error": str(e_main)}

