import streamlit as st
from m5integration import member3_adapter as m3, member4_adapter as m4
from dashboard.common import badge, SIM_WARNING


def render():
    st.header("Multi-Agent Simulation (Member 4)")
    ok, err = m4.availability()
    if not ok:
        badge("bad", f"Member 4 module unavailable: {err}. This page is disabled.")
        return
    badge("warn", SIM_WARNING)
    o = m4.options()
    st.caption("Explanations are templated unless GOOGLE_API_KEY is set; " + ("LLM text ENABLED (never overrides rules)." if m4.llm_enabled() else "LLM text disabled."))
    t1, t2 = st.tabs(["Single charging request", "Lifecycle simulation"])
    with t1:
        st.markdown("**Driver Persona request (scenario input)**")
        a, b, c, d = st.columns(4)
        crate = a.slider("Requested C-rate", 0.1, 3.0, 2.5, 0.1)
        tsoc = b.slider("Target SOC %", 50, 100, 100)
        msoc = c.slider("Discharge floor SOC %", 0, 40, 5)
        amb = d.slider("Ambient C", -10, 50, 40)
        scen = st.selectbox("BMS policy scenario", o["scenarios"], index=o["scenarios"].index("full_protection"))
        dec = m4.evaluate_request(crate, tsoc, msoc, amb, scen)
        st.markdown("**BMS Supervisor response (rule-based)**")
        badge({"approve": "ok", "modify": "warn", "reject": "bad"}[dec["decision"]], f"Decision: {dec['decision'].upper()}")
        st.write(f"**Applied:** {dec['c_rate']}C, SOC window {dec['min_soc']:.0f}-{dec['target_soc']:.0f}%")
        st.write("**Reasons:**", dec["reason"] or "No restriction needed.")
        with st.expander("Limits used"):
            st.json(dec["limits"])
    with t2:
        chem = st.selectbox("Chemistry", o["chemistries"])
        climate = st.selectbox("Climate", o["climates"])
        veh = st.selectbox("Vehicle", m3.list_vehicles(), index=m3.list_vehicles().index("tata_curvv_ev_55kwh_partial_reference.json"))
        cyc = st.selectbox("Drive cycle", ["udds_city.csv", "hwfet_highway.csv"])
        if st.button("Compare all 5 policies"):
            with st.spinner("Running LangGraph simulations..."):
                st.session_state["m4_cmp"] = m4.compare_scenarios(chem, climate, veh, cyc)
        res = st.session_state.get("m4_cmp")
        if res:
            st.dataframe([{"Policy": r["scenario"], "Years to EOL (sim)": r["years_to_eol"], "EOL reached": r["eol_reached"],
                           "Final SOH %": r["final_soh"], "Final range km (illustrative)": r["final_range_km"],
                           "Fast charges": r["fast_charge_count"]} for r in res], width="stretch", hide_index=True)
            import pandas as pd
            curves = pd.DataFrame({r["scenario"]: pd.Series([h["soh"] for h in r["history"]], index=[h["year"] for h in r["history"]]) for r in res})
            st.line_chart(curves)
            st.caption("Simulated SOH vs years. Degradation coefficients are placeholders; absolute years are not predictions.")
