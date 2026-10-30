import streamlit as st
from m5integration import member2_adapter as m2, member3_adapter as m3, member4_adapter as m4
from m5integration.paths import MASTER_TABLE
from rag.vector_store import VectorStore


@st.cache_resource
def _predictor():
    return m2.load_predictor()


def module_status() -> list[dict]:
    rows = [{"Module": "Member 1 - data", "Path": "master_table.parquet",
             "Status": "Present (read-only)" if MASTER_TABLE.exists() else "MISSING",
             "Note": "Upstream dependency; never modified."}]
    pred, err = _predictor()
    fe_ok, fe_err = m2.feature_extraction_available()
    rows.append({"Module": "Member 2 - SOH", "Path": "member2_modeling/",
                 "Status": "Loaded (experimental models)" if pred else f"UNAVAILABLE: {err}",
                 "Note": "Learned SOH underperforms persistence; RUL not approved. "
                         + ("Feature extraction OK." if fe_ok else f"Feature extraction blocked: {fe_err}")})
    try:
        n = len(m3.list_vehicles())
        rows.append({"Module": "Member 3 - range", "Path": "member3/range_calculator.py",
                     "Status": f"Importable, {n} vehicle configs", "Note": "All vehicle configs are partially sourced/illustrative."})
    except Exception as e:
        rows.append({"Module": "Member 3 - range", "Path": "member3/", "Status": f"UNAVAILABLE: {e}", "Note": ""})
    ok, e4 = m4.availability()
    rows.append({"Module": "Member 4 - agents", "Path": "ev_member4/",
                 "Status": "Importable" if ok else f"UNAVAILABLE: {e4}",
                 "Note": "Rule-based BMS; degradation coefficients are placeholders; LLM text optional."})
    n_src = len(VectorStore().sources())
    rows.append({"Module": "Member 5 - RAG", "Path": "rag/", "Status": f"{n_src} source(s) indexed",
                 "Note": "Empty corpus until PDFs are uploaded." if not n_src else "Local TF-IDF retrieval."})
    return rows


def render():
    st.title("Autonomous Multi-Agent EV Battery Life & Range Predictor")
    st.caption("Member 5 integration dashboard. Status below is computed live, not hard-coded.")
    st.dataframe(module_status(), width="stretch", hide_index=True)
    st.subheader("Chemistry support")
    st.table([
        {"Chemistry": "NMC", "SOH model": "Experimental LightGBM", "Range": "Yes (SOH %)", "Agent simulation": "Yes (placeholder degradation)"},
        {"Chemistry": "LFP", "SOH model": "Experimental LightGBM (conditional labels)", "Range": "Yes (SOH %)", "Agent simulation": "Yes (placeholder degradation)"},
        {"Chemistry": "Sodium-ion", "SOH model": "None", "Range": "Only with a user-supplied SOH", "Agent simulation": "Yes (placeholder degradation)"},
        {"Chemistry": "Solid-state", "SOH model": "None", "Range": "Only with a user-supplied SOH", "Agent simulation": "Yes (placeholder degradation)"},
    ])
    st.subheader("Key limitations")
    st.markdown("- Learned SOH models are **experimental** and underperform the persistence baseline; observed SOH is the default anchor.\n"
                "- **RUL is not approved** and is never used or shown as a number.\n"
                "- Range uses illustrative/partially sourced vehicle parameters; it is not a validated range.\n"
                "- Agent simulation is a scenario simulation with placeholder degradation coefficients, not a hardware BMS.\n"
                "- Cell-level Ah is never converted into pack kWh.")
