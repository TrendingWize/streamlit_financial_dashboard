# load_nio_to_auradb.py
import json
import os
from neo4j import GraphDatabase, Driver
# Ensure you have the neo4j driver: pip install neo4j

# --- AuraDB Connection Details ---
# Replace with your AuraDB credentials and URI
# It's recommended to use environment variables for these in production
AURA_URI = os.environ.get("NEO4J_URI", "neo4j+s://f9f444b7.databases.neo4j.io") 
AURA_USER = os.environ.get("NEO4J_USER", "neo4j")
AURA_PASSWORD = os.environ.get("NEO4J_PASSWORD", "BhpVJhR0i8NxbDPU29OtvveNE8Wx3X2axPI7mS7zXW0")

# Path to your NIO.json file
NIO_JSON_FILE_PATH = "NIO.json"


def create_constraints(driver: Driver):
    """Creates necessary constraints in the database."""
    queries = [
        "CREATE CONSTRAINT company_symbol_unique IF NOT EXISTS FOR (c:Company) REQUIRE c.symbol IS UNIQUE;",
        "CREATE CONSTRAINT analysis_report_key IF NOT EXISTS FOR (ar:AnalysisReport) REQUIRE (ar.symbol, ar.fillingDate) IS NODE KEY;"
    ]
    with driver.session() as session:
        for query in queries:
            try:
                print(f"Executing constraint query: {query}")
                session.run(query)
                print(f"Successfully executed: {query}")
            except Exception as e:
                if "already exists" in str(e).lower() or "no change" in str(e).lower() :
                    print(f"Constraint likely already exists or no change needed for: {query}")
                else:
                    print(f"Error executing constraint query '{query}': {e}")

def load_nio_analysis_report(driver: Driver, report_parameter: dict):
    """
    Loads a single analysis report into Neo4j using a parameter.
    'report_parameter' is the Python dictionary loaded from NIO.json.
    Nested structures (metadata, analysis) will be stored as JSON strings.
    """
    # Prepare parameters for the Cypher query, serializing nested dicts to JSON strings
    params = {
        "ticker": report_parameter.get("raw_content", {}).get("metadata", {}).get("ticker"),
        "company_name": report_parameter.get("raw_content", {}).get("metadata", {}).get("company_name"),
        "exchange": report_parameter.get("raw_content", {}).get("metadata", {}).get("exchange"),
        "industry": report_parameter.get("raw_content", {}).get("metadata", {}).get("industry"),
        "sector": report_parameter.get("raw_content", {}).get("metadata", {}).get("sector"),
        "currency": report_parameter.get("raw_content", {}).get("metadata", {}).get("currency"),
        "fillingDate": report_parameter.get("raw_content", {}).get("metadata", {}).get("fillingDate"),
        
        # Serialize metadata and analysis to JSON strings
        "metadata_str": json.dumps(report_parameter.get("raw_content", {}).get("metadata", {})),
        "analysis_str": json.dumps(report_parameter.get("raw_content", {}).get("analysis", {})),
        
        "prompt_tokens": report_parameter.get("prompt_tokens"),
        "completion_tokens": report_parameter.get("completion_tokens"),
        "total_tokens": report_parameter.get("total_tokens"),
        "analysis_generated_at": report_parameter.get("analysis_generated_at"), # Should be ISO string
        "model_used": report_parameter.get("model_used"),
        "symbol_processing_duration": report_parameter.get("symbol_processing_duration"),
        "calendarYear": report_parameter.get("calendarYear")
    }

    # Check for None values that might cause issues with Cypher if not handled
    # (e.g., if a field is optional and missing from JSON)
    for key, value in list(params.items()): # Iterate over a copy for modification
        if value is None:
            # Decide how to handle None: either remove or set to a default like empty string for strings
            # For Cypher, setting a property to null is fine if the type allows it.
            # But if a required field for MERGE like 'ticker' or 'fillingDate' is None, it's an issue.
            if key in ["ticker", "fillingDate"]:
                print(f"CRITICAL ERROR: Required field '{key}' is None. Cannot proceed with loading.")
                return # Or raise an exception
            # params[key] = "" # Example: set None string-like fields to empty string
    
    if not params["ticker"] or not params["fillingDate"]:
        print("Error: Ticker or fillingDate is missing in the report metadata. Cannot load.")
        return

    cypher_query = """
    // Merge the Company node
    MERGE (company:Company {symbol: $ticker})
    ON CREATE SET
        company.companyName = $company_name,
        company.exchange = $exchange,
        company.industry = $industry,
        company.sector = $sector,
        company.currency = $currency,
        company.lastUpdated = datetime()
    ON MATCH SET
        company.companyName = COALESCE(company.companyName, $company_name),
        company.exchange = COALESCE(company.exchange, $exchange),
        company.industry = COALESCE(company.industry, $industry),
        company.sector = COALESCE(company.sector, $sector),
        company.currency = COALESCE(company.currency, $currency),
        company.lastUpdated = datetime()

    // Merge the AnalysisReport node
    MERGE (analysis:AnalysisReport {
        symbol: $ticker,
        fillingDate: date($fillingDate) // Use the passed fillingDate string
    })
    ON CREATE SET
        analysis.metadata_json = $metadata_str,      // Store as JSON string
        analysis.analysis_json = $analysis_str,      // Store as JSON string
        analysis.prompt_tokens = $prompt_tokens,
        analysis.completion_tokens = $completion_tokens,
        analysis.total_tokens = $total_tokens,
        analysis.analysis_generated_at = datetime($analysis_generated_at),
        analysis.model_used = $model_used,
        analysis.symbol_processing_duration = $symbol_processing_duration,
        analysis.calendarYear = $calendarYear,
        analysis.lastUpdated = datetime()
    ON MATCH SET
        analysis.metadata_json = $metadata_str,
        analysis.analysis_json = $analysis_str,
        analysis.prompt_tokens = $prompt_tokens,
        analysis.completion_tokens = $completion_tokens,
        analysis.total_tokens = $total_tokens,
        analysis.analysis_generated_at = datetime($analysis_generated_at),
        analysis.model_used = $model_used,
        analysis.symbol_processing_duration = $symbol_processing_duration,
        analysis.calendarYear = $calendarYear,
        analysis.lastUpdated = datetime()

    // Create relationship
    MERGE (company)-[:HAS_ANALYSIS_REPORT]->(analysis)

    RETURN company.symbol AS symbol_loaded, analysis.fillingDate AS report_date_loaded
    """
    with driver.session() as session:
        result = session.run(cypher_query, **params) # Unpack the params dictionary
        summary = result.consume()
        print(f"Loading summary: Nodes created: {summary.counters.nodes_created}, Relationships created: {summary.counters.relationships_created}")
        # To see the returned values, if any:
        # for record in result:
        #     print(f"Loaded report for symbol: {record['symbol_loaded']}, with fillingDate: {record['report_date_loaded']}")
        print(f"Successfully loaded analysis report for {params['ticker']}.")


if __name__ == "__main__":
    try:
        with open(NIO_JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            nio_report_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {NIO_JSON_FILE_PATH} not found. Make sure it's in the same directory as this script.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {NIO_JSON_FILE_PATH}.")
        exit(1)

    driver = None
    try:
        driver = GraphDatabase.driver(AURA_URI, auth=(AURA_USER, AURA_PASSWORD))
        driver.verify_connectivity()
        print(f"Successfully connected to AuraDB instance at {AURA_URI.split('@')[-1] if '@' in AURA_URI else AURA_URI}.")

        create_constraints(driver)
        load_nio_analysis_report(driver, nio_report_data)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if driver:
            driver.close()
            print("Neo4j connection closed.")