"""Reusable wrappers for LLM and Neo4j access used across the dashboard.

All heavy objects are cached so pages import functions instead of re‑implementing
boilerplate.

Usage
-----
from utils.openai_helpers import ask_ai
answer = ask_ai("Why did TSLA's free cash‑flow drop in 2023?")
"""

from __future__ import annotations
import os, streamlit as st
from functools import lru_cache
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI

# --- Secrets & configuration -------------------------------------------------

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
NEO4J_URI      = st.secrets.get("NEO4J_URI")      
NEO4J_USERNAME = st.secrets.get("NEO4J_USERNAME") 
NEO4J_PASSWORD = st.secrets.get("NEO4J_PASSWORD")

# -----------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatOpenAI:
    """Return a cached LLM client."""
    return ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1, api_key=OPENAI_API_KEY)


@st.cache_resource(show_spinner=False)
def get_neo4j_graph() -> Neo4jGraph:
    """Return a cached Neo4jGraph connection used by LangChain."""
    return Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
    )


def ask_ai(question: str, **kwargs) -> str:
    """High‑level helper for the "Ask AI" tabs."""
    llm = get_llm()
    graph = get_neo4j_graph()
    cypher = graph.build_cypher_prompt(question)
    response = llm.invoke(cypher, **kwargs)
    return response
