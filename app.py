"""Member 5 integrated Streamlit dashboard. Run: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="EV Battery Life & Range Predictor", layout="wide")

from m5integration.paths import ensure_repo_on_path
ensure_repo_on_path()
from dashboard import overview, battery_health, range_analysis, agent_simulation, literature_assistant

PAGES = {"Overview": overview, "Battery Health & SOH": battery_health, "EV Range": range_analysis,
         "Multi-Agent Simulation": agent_simulation, "Literature Assistant (RAG)": literature_assistant}

page = st.sidebar.radio("Navigate", list(PAGES))
for name, mod in PAGES.items():
    pass
try:
    PAGES[page].render()
except Exception as exc:  # one failing module must not crash the dashboard
    st.error(f"This page failed to render: {type(exc).__name__}: {exc}")
