import streamlit as st
from m5integration import member2_adapter as m2, member3_adapter as m3
from dashboard.common import badge, ILLUSTRATIVE_WARNING


def _soh_choice():
    contract = st.session_state.get("health_contract")
    opts = ["Manual SOH (user-provided scenario value)"]
    if contract:
        opts = ["Observed SOH from Battery Health page (default anchor)",
                "EXPERIMENTAL predicted SOH from Battery Health page"] + opts
    pick = st.radio("SOH source", opts)
    if pick.startswith("Observed"):
        return m2.soh_for_range(contract, "observed")
    if pick.startswith("EXPERIMENTAL"):
        return m2.soh_for_range(contract, "predicted")
    v = st.slider("SOH (%)", 50.0, 100.0, 90.0, 0.5)
    return {"allowed": True, "soh_pct": v, "label": "User-provided scenario SOH"}


def _m1_flag(soh):
    """Member 1 flag only applies to SOH that came from the Battery Health selection."""
    return st.session_state.get("health_m1_flag") if soh.get("label", "").startswith(("Observed", "EXPERIMENTAL")) else None


def render():
    st.header("Real-World EV Range (Member 3 RangeCalculator)")
    badge("warn", ILLUSTRATIVE_WARNING)
    vehicle = st.selectbox("Vehicle configuration", m3.list_vehicles())
    info = m3.vehicle_info(vehicle)
    st.caption(f"{info['name']} | pack {info['pack_kwh']} kWh | reference range {info['rated_range_km_reference']} km | "
               f"provenance: {info['provenance_status']}")
    cycle = st.selectbox("Drive cycle", [c for c in m3.list_cycles() if c != "example_cycle.csv"])
    st.caption(m3.CYCLE_NOTES.get(cycle, ""))
    chem = st.selectbox("Chemistry (context only)", ["NMC", "LFP", "Sodium-ion", "Solid-State"])
    if chem in ("Sodium-ion", "Solid-State"):
        badge("warn", f"No SOH model exists for {chem}; only a manual scenario SOH can be used.")
    soh = _soh_choice()
    ambient = st.slider("Ambient temperature (C)", -20, 45, 25)
    override = st.checkbox("Override HVAC power (otherwise Member 3's empirical HVAC model is used)")
    hvac = st.slider("HVAC power (kW)", 0.0, 6.0, 1.0, 0.1) if override else None
    st.caption("Driving style and auxiliary power are not inputs of Member 3's estimate(); they are not offered here.")
    try:
        r = m3.estimate(vehicle, cycle, soh, ambient, hvac, quality_flag=_m1_flag(soh))
    except m3.RangeBlocked as e:
        badge("bad", f"Range blocked: {e}")
        return
    except Exception as e:
        badge("bad", f"Range calculation failed: {type(e).__name__}: {e}")
        return
    c = st.columns(4)
    c[0].metric("Range (illustrative)", f"{r['estimated_range_km']:.0f} km")
    c[1].metric("Consumption", f"{r['consumption_kwh_per_100km']:.1f} kWh/100 km")
    c[2].metric("Usable pack energy", f"{r['usable_energy_kwh']:.1f} kWh")
    c[3].metric("Applied SOH", f"{r['soh_pct']:.1f} %")
    st.write(f"**Member 1 quality_flag passed to Member 3:** {r.get('quality_flag')}")
    st.write(f"**SOH source:** {r['_applied_soh_source']} | **HVAC:** {r['hvac_power_kw']:.2f} kW ({r['hvac_mode']}) | "
             f"**Cycle:** {r['_cycle']} | **Quality status:** {r['quality_status']}")
    with st.expander("Assumptions and limitations"):
        st.write(info["assumptions"])
        st.write(r["limitations"])
    st.subheader("Scenario comparison")
    rows = []
    def add(label, **kw):
        a = dict(vehicle=vehicle, cycle=cycle, soh=soh, ambient_c=ambient, hvac_power_kw=hvac); a.update(kw)
        a.setdefault("quality_flag", _m1_flag(a["soh"]))
        try:
            x = m3.estimate(**a)
            rows.append({"Scenario": label, "Range km": round(x["estimated_range_km"], 1),
                         "kWh/100km": round(x["consumption_kwh_per_100km"], 1)})
        except Exception as e:
            rows.append({"Scenario": label, "Range km": None, "kWh/100km": str(e)[:60]})
    add("City (UDDS)", cycle="udds_city.csv"); add("Highway (HWFET)", cycle="hwfet_highway.csv")
    for t in (-10, 25, 40):
        add(f"Ambient {t} C", ambient_c=t)
    contract = st.session_state.get("health_contract")
    if contract:
        for src in ("observed", "predicted"):
            s = m2.soh_for_range(contract, src)
            if s["allowed"]:
                add(f"{s['label']} ({s['soh_pct']:.1f}%)", soh=s)
    st.dataframe(rows, width="stretch", hide_index=True)
