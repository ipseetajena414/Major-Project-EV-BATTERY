import streamlit as st
from m5integration import member2_adapter as m2
from dashboard.common import badge


@st.cache_resource
def _predictor():
    return m2.load_predictor()


@st.cache_data
def _index():
    return m2.list_cycles_index()


def render():
    st.header("Battery Health and SOH (Member 2)")
    chem = st.selectbox("Chemistry", ["NMC", "LFP", "Na-ion", "Solid-State"])
    if chem not in m2.SUPPORTED:
        badge("bad", f"SOH prediction unsupported for {chem}. {m2.UNSUPPORTED_REASONS[chem]}")
        return
    pred, err = _predictor()
    if pred is None:
        badge("bad", f"Member 2 predictor unavailable: {err}")
        return
    ok, ferr = m2.feature_extraction_available()
    if not ok:
        badge("bad", f"Feature extraction unavailable (Member 2 build_and_train import failed): {ferr}. "
                     "Install the pinned requirements (torch) to enable this page.")
        return
    idx = _index()
    sub = idx[idx.chemistry == chem]
    dataset = st.selectbox("Dataset", sorted(sub.dataset_id.unique()))
    sub = sub[sub.dataset_id == dataset]
    cell = st.selectbox("Cell", sorted(sub.cell_id.unique()))
    cyc = sub[sub.cell_id == cell].sort_values("cycle_id")
    cycle_id = st.selectbox("Observed cycle", cyc.cycle_id.tolist())
    m1_flag = m2.member1_quality_flag(cell, int(cycle_id), dataset)
    st.caption(f"Dataset: {dataset} | Cell: {cell} | Cycle: {cycle_id} | Member 1 quality_flag: {m1_flag}")
    if st.button("Compute health result"):
        with st.spinner("Extracting real cycle features and calling Member 2..."):
            contract, e = m2.health_for_cycle(pred, cell, int(cycle_id), chem, dataset_id=dataset)
        if e:
            badge("bad", e)
        else:
            st.session_state["health_contract"] = contract
            st.session_state["health_label"] = f"{chem} | {dataset} | {cell} | cycle {cycle_id}"
            st.session_state["health_m1_flag"] = m1_flag  # Member 1's flag, kept outside the Member 2 contract
    c = st.session_state.get("health_contract")
    if not c:
        return
    st.subheader(st.session_state.get("health_label", ""))
    soh = c.get("soh", {})
    kind = "ok" if c.get("status") in m2.OK_STATUSES else "bad"
    badge("warn" if kind == "ok" else "bad", f"Member 2 status: {c.get('status')} (experimental; not validated)")
    a, b = st.columns(2)
    a.metric("Observed SOH (anchor)", _fmt(soh.get("current_observed_soh_pct"), "%"))
    b.metric("Predicted SOH - EXPERIMENTAL", _fmt(soh.get("predicted_soh_pct"), "%"))
    st.write("**Model status:**", soh.get("model_status"), "| **Validation:**", soh.get("validation_status"))
    badge("info", "RUL: unavailable. Member 2's RUL output is not approved for downstream use and is not displayed as a number.")
    cap = c.get("capacity") or {}
    if cap:
        st.write("**Cell-level capacity (not pack kWh):**", cap)
    st.write("**Member 1 quality_flag (original, not part of the Member 2 contract):**", st.session_state.get("health_m1_flag"))
    show_quality(c.get("quality") or {})
    if c.get("errors"):
        st.write("**errors**")
        st.json(c["errors"])
    with st.expander("Full Member 2 contract"):
        st.json(c)


def _fmt(v, unit):
    return "n/a" if v is None else f"{v:.2f} {unit}"


def show_quality(q: dict):
    """Display the nested Member 2 ``quality`` object exactly as delivered (schema unchanged)."""
    st.write("**Member 2 quality**")
    st.write("quality_flags:", q.get("quality_flags") or "none")
    st.write("warnings:", q.get("warnings") or "none")
    st.write("checks_passed:", q.get("checks_passed") or "none")
    st.write("checks_not_performed:", q.get("checks_not_performed") or "none")
    st.write("label_provenance:", q.get("label_provenance"))
