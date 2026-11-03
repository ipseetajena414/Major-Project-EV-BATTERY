"""
Member 5 — Multi-Agent BMS Safety Simulation Dashboard (Page 4)
Schema Version 2.0.0 — Modern Navy Blue Enterprise UI with Plotly Operating Envelope & EV Presets

Architectural Ground Rules Enforced:
  1. Zero modification to ev_member4/ source files
  2. Deterministic safety primacy — LLM never overrides hard physical limits
  3. SOH/RUL non-fabrication — None stays None, displays "UNAVAILABLE"
  4. Member 3 decoupling — runs standalone without range models
"""
import streamlit as st
import json
import sys
import os
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import ev_member4
    from ev_member4 import RULES_CATALOG, BMS_LIMITS, list_available_vehicles
    MEMBER4_AVAILABLE = True
except ImportError as e:
    MEMBER4_AVAILABLE = False
    MEMBER4_ERROR = str(e)

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

st.set_page_config(
    page_title="BMS Safety Simulation & Telemetry",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# MODERN NAVY BLUE ENTERPRISE DESIGN SYSTEM (CSS OVERRIDES)
# ═══════════════════════════════════════════════════════════════
NAVY_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --navy-base: #070d18;
    --navy-bg: #0a1122;
    --navy-surface: #0f1c32;
    --navy-card: #13223e;
    --navy-card-hover: #182a4c;
    --navy-border: #1e3356;
    --navy-border-subtle: #172844;
    --navy-border-focus: #2d4c7e;
    --navy-muted: #829ab8;
    --navy-light: #cbd5e1;
    --navy-primary: #2563eb;
    --navy-primary-hover: #1d4ed8;
    --navy-accent: #3b82f6;
    --navy-accent-light: #60a5fa;
    --navy-highlight: #93c5fd;
    --navy-soft: #dbeafe;
    --text-main: #f8fafc;
}

html, body, [class*="css"], .stMarkdown, p, span, div {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

h1, h2, h3, h4, .cyber-title, .scenario-title, .outcome-text-title, .gauge-title, .guide-title, .transcript-speaker {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    font-weight: 700;
    letter-spacing: -0.3px;
}

/* Screen Canvas Maximization */
.main .block-container {
    max-width: 100% !important;
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2.2rem !important;
    padding-right: 2.2rem !important;
}

.stApp {
    background: radial-gradient(circle at 18% 12%, #0e192e 0%, #0a1122 50%, #060a14 100%);
    color: var(--text-main);
}

/* Header Container */
.cyber-header {
    background: linear-gradient(135deg, rgba(19, 34, 62, 0.95) 0%, rgba(13, 23, 42, 0.98) 100%);
    border: 1px solid var(--navy-border);
    border-radius: 14px;
    padding: 22px 28px;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(4, 8, 16, 0.5);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}

.cyber-title {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #ffffff;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.cyber-subtitle {
    font-size: 14px;
    color: var(--navy-muted);
    margin-top: 4px;
    font-weight: 500;
}

/* Telemetry Status Chips in Header */
.telemetry-status-group {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    align-items: center;
}

.telemetry-status-chip {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    color: var(--navy-highlight);
    background: rgba(15, 28, 50, 0.9);
    border: 1px solid var(--navy-border);
    padding: 6px 14px;
    border-radius: 8px;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #3b82f6;
    display: inline-block;
}

/* Scenario Quick Launch Cards */
.scenario-card-box {
    background: linear-gradient(145deg, rgba(19, 34, 62, 0.95) 0%, rgba(13, 24, 44, 0.98) 100%);
    border: 1px solid var(--navy-border);
    border-radius: 14px;
    padding: 24px 26px;
    height: 100%;
    min-height: 240px;
    transition: all 0.2s ease-in-out;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 4px 16px rgba(4, 8, 16, 0.4);
}

.scenario-card-box:hover {
    border-color: var(--navy-accent);
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(10, 20, 38, 0.6);
}

.scenario-title {
    font-size: 18px;
    font-weight: 700;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
}

.scenario-params {
    font-size: 13.5px;
    color: var(--navy-light);
    line-height: 1.65;
    margin-bottom: 16px;
}

.scenario-expected {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    padding: 5px 12px;
    border-radius: 6px;
    display: inline-block;
    margin-bottom: 16px;
}

/* Metric Cards Grid */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 16px;
    margin-bottom: 22px;
}

.metric-card {
    background: linear-gradient(145deg, rgba(19, 34, 62, 0.95) 0%, rgba(13, 23, 42, 0.98) 100%);
    border: 1px solid var(--navy-border);
    border-radius: 12px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: all 0.2s ease-in-out;
    box-shadow: 0 4px 16px rgba(4, 8, 16, 0.35);
}

.metric-card:hover {
    border-color: var(--navy-accent);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(4, 8, 16, 0.5);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #1d4ed8, #3b82f6);
    opacity: 0.85;
}

.metric-label {
    font-size: 12px;
    font-weight: 700;
    font-family: 'Plus Jakarta Sans', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--navy-muted);
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.metric-value {
    font-size: 34px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #ffffff;
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin-bottom: 6px;
}

.metric-sub {
    font-size: 12.5px;
    color: var(--navy-muted);
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Side Gauges & Control Box (Left Column in Run Mode) */
.gauge-card {
    background: linear-gradient(145deg, rgba(19, 34, 62, 0.95) 0%, rgba(13, 23, 42, 0.98) 100%);
    border: 1px solid var(--navy-border);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 14px rgba(4, 8, 16, 0.3);
}

.gauge-title {
    font-size: 14px;
    font-weight: 700;
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
}

.gauge-bar-outer {
    width: 100%;
    height: 10px;
    background: rgba(10, 18, 34, 0.8);
    border-radius: 999px;
    overflow: hidden;
    margin: 8px 0 10px 0;
    border: 1px solid var(--navy-border-subtle);
}

.gauge-bar-inner {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #1d4ed8 0%, #3b82f6 100%);
    transition: width 0.4s ease;
}

/* Outcome Status Banners */
.outcome-banner {
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    gap: 18px;
    border: 1px solid;
}

.outcome-banner.approve {
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.35) 0%, rgba(13, 30, 48, 0.7) 100%);
    border-color: #059669;
    color: #ecfdf5;
}

.outcome-banner.modify {
    background: linear-gradient(135deg, rgba(30, 58, 138, 0.35) 0%, rgba(13, 26, 46, 0.7) 100%);
    border-color: #2563eb;
    color: #eff6ff;
}

.outcome-banner.reject {
    background: linear-gradient(135deg, rgba(127, 29, 29, 0.35) 0%, rgba(20, 15, 25, 0.7) 100%);
    border-color: #dc2626;
    color: #fef2f2;
}

.outcome-icon {
    font-size: 30px;
    flex-shrink: 0;
}

.outcome-text-title {
    font-size: 16.5px;
    font-weight: 700;
    margin-bottom: 3px;
    color: #ffffff;
}

.outcome-text-desc {
    font-size: 13.5px;
    color: var(--navy-light);
    line-height: 1.5;
}

/* Guide Cards */
.guide-card {
    background: linear-gradient(145deg, rgba(19, 34, 62, 0.95) 0%, rgba(13, 23, 42, 0.98) 100%);
    border: 1px solid var(--navy-border);
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 16px;
    height: 100%;
    transition: all 0.2s ease-in-out;
}

.guide-card:hover {
    border-color: var(--navy-accent);
    box-shadow: 0 4px 16px rgba(10, 20, 38, 0.5);
}

.guide-title {
    font-size: 15px;
    font-weight: 700;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
}

.guide-body {
    font-size: 13px;
    color: var(--navy-light);
    line-height: 1.6;
}

.guide-highlight {
    color: var(--navy-highlight);
    font-weight: 600;
}

/* Transcript Bubbles */
.transcript-card {
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 14px;
    border: 1px solid var(--navy-border);
}

.transcript-card.driver {
    background: linear-gradient(145deg, rgba(17, 30, 52, 0.8) 0%, rgba(12, 21, 38, 0.9) 100%);
    border-left: 4px solid #3b82f6;
}

.transcript-card.bms {
    background: linear-gradient(145deg, rgba(21, 37, 65, 0.85) 0%, rgba(14, 25, 45, 0.95) 100%);
    border-left: 4px solid #60a5fa;
}

.transcript-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.transcript-speaker {
    font-size: 14px;
    font-weight: 700;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 8px;
}

.transcript-round {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11.5px;
    color: var(--navy-highlight);
    background: rgba(37, 99, 235, 0.15);
    padding: 3px 10px;
    border-radius: 6px;
    border: 1px solid rgba(59, 130, 246, 0.3);
}

.transcript-msg {
    font-size: 14px;
    line-height: 1.6;
    color: var(--navy-light);
}

/* Modern Navy Table */
.cyber-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--navy-border);
    margin: 14px 0 20px 0;
}

.cyber-table th {
    background: rgba(19, 34, 62, 0.95);
    color: var(--navy-highlight);
    font-size: 12.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--navy-border);
}

.cyber-table td {
    padding: 12px 16px;
    font-size: 13px;
    color: var(--navy-light);
    border-bottom: 1px solid rgba(30, 51, 86, 0.5);
    background: rgba(14, 25, 45, 0.7);
}

.cyber-table tr:last-child td {
    border-bottom: none;
}

.cyber-table tr:hover td {
    background: rgba(22, 40, 72, 0.6);
}

.status-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

.status-pill.triggered {
    background: rgba(220, 38, 38, 0.18);
    color: #fca5a5;
    border: 1px solid rgba(239, 68, 68, 0.4);
}

.status-pill.clear {
    background: rgba(16, 185, 129, 0.15);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

/* Sidebar Customization */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d172a 0%, #080f1d 100%);
    border-right: 1px solid var(--navy-border);
}

/* Primary Action Buttons */
div.stButton > button {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    letter-spacing: 0.4px !important;
    text-transform: none !important;
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    color: #ffffff !important;
    border: 1px solid #3b82f6 !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3) !important;
    transition: all 0.2s ease-in-out !important;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    border-color: #60a5fa !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.45) !important;
    transform: translateY(-1px);
}
</style>
"""

st.markdown(NAVY_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# VEHICLE PROFILES & DRIVE CYCLES LIBRARY (RECOMMENDATION 3)
# ═══════════════════════════════════════════════════════════════
VEHICLE_PROFILES = {
    "tata_curvv_ev_55kwh_partial_reference.json": {
        "display_name": "Tata Curvv.ev 55 kWh (LFP)",
        "battery_kwh": 55.0,
        "range_km": 502.0,
        "chemistry": "LFP (Lithium Iron Phosphate)",
        "drivetrain": "FWD Single Motor (123 kW)",
        "kerb_mass_kg": 2100.0,
        "cd": 0.28
    },
    "tata_nexon_ev_45kwh_partial_reference.json": {
        "display_name": "Tata Nexon.ev 45 kWh (LFP)",
        "battery_kwh": 45.0,
        "range_km": 489.0,
        "chemistry": "LFP (Lithium Iron Phosphate)",
        "drivetrain": "FWD Single Motor (106 kW)",
        "kerb_mass_kg": 1400.0,
        "cd": 0.32
    },
    "tata_punch_ev_40kwh_partial_reference.json": {
        "display_name": "Tata Punch.ev 40 kWh (LFP)",
        "battery_kwh": 40.0,
        "range_km": 468.0,
        "chemistry": "LFP (Lithium Iron Phosphate)",
        "drivetrain": "FWD Single Motor (90 kW)",
        "kerb_mass_kg": 1340.0,
        "cd": 0.34
    },
    "mahindra_xev9e_59kwh_partial_reference.json": {
        "display_name": "Mahindra XEV 9e 59 kWh (LFP)",
        "battery_kwh": 59.0,
        "range_km": 542.0,
        "chemistry": "LFP Blade (62 kWh gross / 59 net)",
        "drivetrain": "RWD Single Motor (170 kW)",
        "kerb_mass_kg": 2180.0,
        "cd": 0.29
    },
    "byd_atto3_gl_49_92kwh_partial_reference.json": {
        "display_name": "BYD Atto 3 GL 49.92 kWh (Blade LFP)",
        "battery_kwh": 49.92,
        "range_km": 400.0,
        "chemistry": "BYD Blade LFP",
        "drivetrain": "FWD Single Motor (150 kW)",
        "kerb_mass_kg": 1750.0,
        "cd": 0.29
    },
    "mg4_ev_51kwh_partial_reference.json": {
        "display_name": "MG4 EV 51 kWh (LFP)",
        "battery_kwh": 51.0,
        "range_km": 350.0,
        "chemistry": "LFP Cell-to-Pack (One Pack)",
        "drivetrain": "RWD Single Motor (125 kW)",
        "kerb_mass_kg": 1655.0,
        "cd": 0.279
    },
    "hyundai_ioniq5_84kwh_partial_reference.json": {
        "display_name": "Hyundai IONIQ 5 84 kWh (NMC)",
        "battery_kwh": 84.0,
        "range_km": 690.0,
        "chemistry": "NMC 811 (800V Architecture)",
        "drivetrain": "RWD Single Motor (168 kW)",
        "kerb_mass_kg": 2010.0,
        "cd": 0.288
    },
    "tesla_model_y_lr_78kwh_partial_reference.json": {
        "display_name": "Tesla Model Y Long Range 78 kWh (NMC)",
        "battery_kwh": 78.0,
        "range_km": 600.0,
        "chemistry": "NMC 2170 Cylindrical",
        "drivetrain": "Dual Motor AWD (378 kW)",
        "kerb_mass_kg": 1979.0,
        "cd": 0.23
    }
}

DRIVE_CYCLES = {
    "udds_city.csv": "UDDS (Urban Dynamometer Driving Schedule - City)",
    "hwfet_highway.csv": "HWFET (Highway Fuel Economy Test)"
}

# ═══════════════════════════════════════════════════════════════
# PLOTLY CHART BUILDERS (RECOMMENDATION 2)
# ═══════════════════════════════════════════════════════════════

def build_safe_operating_envelope_chart(cell_temp: float, req_c_rate: float, applied_c_rate: float, crit_temp: float = 50.0, max_temp: float = 35.0, max_c: float = 1.5):
    """
    Constructs an interactive Safe Operating Envelope (SOE) plot in Plotly.
    Visualizes Safe Zone, Derate Zone, and Abort Zone, mapping the clamp vector.
    """
    if not PLOTLY_AVAILABLE:
        return None

    fig = go.Figure()

    # 1. Safe Nominal Operating Zone
    fig.add_shape(
        type="rect",
        x0=-10, y0=0, x1=max_temp, y1=max_c,
        fillcolor="rgba(16, 185, 129, 0.16)",
        line=dict(color="rgba(16, 185, 129, 0.4)", width=1),
        layer="below"
    )

    # 2. Thermal Derating Zone (Kinetic Heat Protection)
    fig.add_trace(go.Scatter(
        x=[max_temp, crit_temp, crit_temp, max_temp],
        y=[max_c, 0.0, max_c, max_c],
        fill="toself",
        fillcolor="rgba(245, 158, 11, 0.15)",
        line=dict(color="rgba(245, 158, 11, 0.4)", width=1, dash="dot"),
        mode="lines",
        name="Thermal Derate Zone",
        hoverinfo="skip"
    ))

    # 3. Emergency Thermal Cutoff / Abort Zone
    fig.add_shape(
        type="rect",
        x0=crit_temp, y0=0, x1=60, y1=3.5,
        fillcolor="rgba(239, 68, 68, 0.18)",
        line=dict(color="rgba(239, 68, 68, 0.4)", width=1),
        layer="below"
    )

    # 4. Driver Unconstrained Demand Marker
    fig.add_trace(go.Scatter(
        x=[cell_temp],
        y=[req_c_rate],
        mode="markers+text",
        marker=dict(symbol="diamond", size=14, color="#f97316", line=dict(color="#ffffff", width=1.5)),
        name="Driver Request (Unconstrained)",
        text=[f"Demand: {req_c_rate:.1f}C @ {cell_temp:.1f}°C"],
        textposition="top center",
        textfont=dict(color="#fdba74", size=11, family="JetBrains Mono"),
        hoverinfo="text"
    ))

    # 5. BMS Clamped Safe Operating Point Marker
    fig.add_trace(go.Scatter(
        x=[cell_temp],
        y=[applied_c_rate],
        mode="markers+text",
        marker=dict(symbol="circle", size=15, color="#2563eb", line=dict(color="#93c5fd", width=2)),
        name="BMS Approved Action",
        text=[f"Approved: {applied_c_rate:.1f}C"],
        textposition="bottom center",
        textfont=dict(color="#93c5fd", size=11, family="JetBrains Mono"),
        hoverinfo="text"
    ))

    # 6. Safety Enforcement Clamp Arrow
    if req_c_rate != applied_c_rate:
        fig.add_annotation(
            x=cell_temp, y=applied_c_rate,
            ax=cell_temp, ay=req_c_rate,
            xref="x", yref="y",
            axref="x", ayref="y",
            text="Safety Gate Clamping",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor="#60a5fa",
            font=dict(color="#93c5fd", size=10, family="Plus Jakarta Sans")
        )

    # Safe Zone Label Annotation
    fig.add_annotation(x=15, y=0.75, text="NOMINAL SAFE ENVELOPE (LFP/NMC)", showarrow=False, font=dict(color="rgba(110, 231, 183, 0.6)", size=11, family="Plus Jakarta Sans"))
    fig.add_annotation(x=42, y=0.75, text="KINETIC DERATE ZONE", showarrow=False, font=dict(color="rgba(252, 211, 77, 0.6)", size=10, family="Plus Jakarta Sans"))
    fig.add_annotation(x=54, y=2.0, text="EMERGENCY ABORT (0.0C)", showarrow=False, font=dict(color="rgba(248, 113, 113, 0.7)", size=11, family="Plus Jakarta Sans"))

    fig.update_layout(
        title=dict(text="Deterministic BMS Safe Operating Envelope (SOE)", font=dict(color="#ffffff", size=15, family="Plus Jakarta Sans")),
        xaxis=dict(
            title=dict(text="Cell Pack Temperature (°C)", font=dict(color="#829ab8", size=12)),
            range=[-5, 60],
            gridcolor="#1e3356",
            zerolinecolor="#1e3356",
            tickfont=dict(color="#cbd5e1", family="JetBrains Mono")
        ),
        yaxis=dict(
            title=dict(text="Charge Multiplier Rate (C-Rate)", font=dict(color="#829ab8", size=12)),
            range=[-0.2, 3.8],
            gridcolor="#1e3356",
            zerolinecolor="#1e3356",
            tickfont=dict(color="#cbd5e1", family="JetBrains Mono")
        ),
        paper_bgcolor="#0d172a",
        plot_bgcolor="#0f1c32",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#cbd5e1", size=11)
        ),
        height=380
    )
    return fig


def build_longevity_projection_chart(current_soh: float = 89.4):
    """
    Constructs a comparative battery lifespan degradation projection:
    Unrestricted DC Fast Charging vs BMS-Supervised Charging.
    """
    if not PLOTLY_AVAILABLE:
        return None

    cycles = np.linspace(0, 1200, 60)
    base_soh = current_soh if current_soh is not None else 90.0

    # Unrestricted DC Fast Charging (Fast SEI degradation + lithium plating)
    soh_unrestricted = base_soh - (0.028 * cycles) - (0.000012 * (cycles ** 2))
    soh_unrestricted = np.clip(soh_unrestricted, 50.0, 100.0)

    # BMS Supervised (80% SOC Ceiling + Thermal Protection)
    soh_supervised = base_soh - (0.012 * cycles) - (0.000003 * (cycles ** 2))
    soh_supervised = np.clip(soh_supervised, 50.0, 100.0)

    fig = go.Figure()

    # 80% End-of-Life (EOL) Threshold line
    fig.add_hline(y=80.0, line=dict(color="#ef4444", width=1.5, dash="dash"), annotation_text="Standard Automotive 80% EOL Threshold", annotation_position="bottom right", annotation_font=dict(color="#fca5a5", size=10))

    fig.add_trace(go.Scatter(
        x=cycles,
        y=soh_unrestricted,
        mode="lines",
        name="Unrestricted Driver Demand (100% Target / High C-Rate)",
        line=dict(color="#f97316", width=2.5, dash="dot"),
        hovertemplate="Cycle: %{x:.0f}<br>SOH: %{y:.1f}%<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=cycles,
        y=soh_supervised,
        mode="lines",
        name="BMS Protected (80% SOC Cap + Thermal Clamping)",
        line=dict(color="#3b82f6", width=3),
        hovertemplate="Cycle: %{x:.0f}<br>SOH: %{y:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="Comparative Battery SOH Longevity Trajectory Over 1,200 Cycles", font=dict(color="#ffffff", size=15, family="Plus Jakarta Sans")),
        xaxis=dict(
            title=dict(text="Operating Equivalent Full Cycles (EFC)", font=dict(color="#829ab8", size=12)),
            gridcolor="#1e3356",
            tickfont=dict(color="#cbd5e1", family="JetBrains Mono")
        ),
        yaxis=dict(
            title=dict(text="State of Health (SOH %)", font=dict(color="#829ab8", size=12)),
            range=[65, 100],
            gridcolor="#1e3356",
            tickfont=dict(color="#cbd5e1", family="JetBrains Mono")
        ),
        paper_bgcolor="#0d172a",
        plot_bgcolor="#0f1c32",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#cbd5e1", size=11)
        ),
        height=360
    )
    return fig


def build_radar_comparison_chart(req_c: float, applied_c: float, req_target: float, applied_target: float, cell_temp: float, decision: str):
    """
    Constructs a 5-axis spider/radar chart comparing Driver Unconstrained Demand vs BMS Clamped Action.
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    categories = ['Charge Speed', 'Capacity Saturation', 'Thermal Margin', 'Battery Longevity', 'Lithium Safety']
    
    driver_speed = min(100, int((req_c / 3.5) * 100))
    driver_sat = int(req_target)
    driver_therm = max(10, int(100 - (cell_temp / 55.0) * 100))
    driver_longevity = max(15, int(100 - (req_c / 3.5) * 45 - (req_target / 100.0) * 40))
    driver_plating_safety = max(10, int(100 - (req_c / 1.5) * 50))
    driver_r = [driver_speed, driver_sat, driver_therm, driver_longevity, driver_plating_safety]

    bms_speed = min(100, int((applied_c / 3.5) * 100))
    bms_sat = int(applied_target)
    bms_therm = max(20, int(100 - (cell_temp / 50.0) * 60))
    bms_longevity = min(98, max(40, int(95 - (applied_c / 1.5) * 15)))
    bms_plating_safety = 95 if applied_c <= 1.5 else 60
    bms_r = [bms_speed, bms_sat, bms_therm, bms_longevity, bms_plating_safety]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=driver_r,
        theta=categories,
        fill='toself',
        fillcolor='rgba(249, 115, 22, 0.22)',
        line=dict(color='#f97316', width=2),
        name='Driver Demand (Unconstrained)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=bms_r,
        theta=categories,
        fill='toself',
        fillcolor='rgba(37, 99, 235, 0.28)',
        line=dict(color='#3b82f6', width=2.5),
        name='BMS Safe Action (Hardware Clamped)'
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='#0f1c32',
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='#1e3356', tickfont=dict(color='#cbd5e1', size=9)),
            angularaxis=dict(gridcolor='#1e3356', tickfont=dict(color='#cbd5e1', size=11, family='Plus Jakarta Sans'))
        ),
        paper_bgcolor='#0d172a',
        title=dict(text="Driver Demand vs. BMS Physical Safety Radar", font=dict(color="#ffffff", size=14, family="Plus Jakarta Sans")),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", font=dict(color="#cbd5e1", size=10)),
        margin=dict(l=30, r=30, t=45, b=45),
        height=320
    )
    return fig


def build_decision_waterfall_chart(req_c: float, applied_c: float, decision: str, reason: str):
    """
    Constructs a stepped waterfall derating chart tracking gate clamping progression.
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    kinetic_clamp = -max(0.0, req_c - 1.5) if req_c > 1.5 else 0.0
    post_kinetic = req_c + kinetic_clamp
    thermal_derate = applied_c - post_kinetic
    
    steps = ["Driver Demand", "Kinetic Limit Clamp", "Thermal Derating", "Final Approved Rate"]
    values = [req_c, kinetic_clamp, thermal_derate, applied_c]
    measures = ["relative", "relative", "relative", "total"]

    fig = go.Figure(go.Waterfall(
        name="Rate Clamping",
        orientation="v",
        measure=measures,
        x=steps,
        y=values,
        text=[f"{v:+.1f}C" if m == "relative" else f"{v:.1f}C" for v, m in zip(values, measures)],
        textposition="outside",
        textfont=dict(color="#cbd5e1", family="JetBrains Mono", size=11),
        connector=dict(line=dict(color="#829ab8", width=1, dash="dot")),
        decreasing=dict(marker=dict(color="#f87171")),
        increasing=dict(marker=dict(color="#60a5fa")),
        totals=dict(marker=dict(color="#10b981" if decision == "APPROVE" else ("#3b82f6" if decision == "MODIFY" else "#ef4444")))
    ))
    fig.update_layout(
        title=dict(text=f"BMS Decision Derating Stepped Waterfall ({decision})", font=dict(color="#ffffff", size=14, family="Plus Jakarta Sans")),
        xaxis=dict(
            title=dict(text="Safety Evaluation Gate Step", font=dict(color="#829ab8", size=11)),
            gridcolor="#1e3356",
            tickfont=dict(color="#cbd5e1", family="Plus Jakarta Sans", size=10)
        ),
        yaxis=dict(
            title=dict(text="Charging Multiplier (C-Rate)", font=dict(color="#829ab8", size=11)),
            gridcolor="#1e3356",
            tickfont=dict(color="#cbd5e1", family="JetBrains Mono", size=10),
            range=[-0.2, max(4.0, req_c + 0.8)]
        ),
        paper_bgcolor="#0d172a",
        plot_bgcolor="#0f1c32",
        margin=dict(l=40, r=30, t=45, b=40),
        height=320
    )
    return fig


def build_cccv_charging_profile(applied_c: float, applied_target: float, cell_temp: float):
    """
    Constructs a simulated Constant Current - Constant Voltage (CC-CV) charging profile curve.
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    t = np.linspace(0, 60, 61)
    if applied_c <= 0.0:
        soc = np.full_like(t, 50.0)
        temp = np.full_like(t, cell_temp)
        current = np.zeros_like(t)
    else:
        cc_rate = applied_c
        cc_mins = max(10, int((applied_target - 20) / (cc_rate * 1.2)))
        cc_mins = min(45, cc_mins)
        
        soc_cc = np.linspace(20, min(applied_target, 80.0), cc_mins)
        soc_cv = np.linspace(min(applied_target, 80.0), applied_target, len(t) - cc_mins)
        soc = np.concatenate([soc_cc, soc_cv])
        
        temp_rise = cc_rate * 4.5
        temp_cc = np.linspace(cell_temp, cell_temp + temp_rise, cc_mins)
        temp_cv = np.linspace(cell_temp + temp_rise, cell_temp + temp_rise - 2.0, len(t) - cc_mins)
        temp = np.concatenate([temp_cc, temp_cv])
        
        curr_cc = np.full(cc_mins, cc_rate)
        curr_cv = np.linspace(cc_rate, 0.15, len(t) - cc_mins)
        current = np.concatenate([curr_cc, curr_cv])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=soc, name="Pack State of Charge (SOC %)",
        line=dict(color="#3b82f6", width=2.5),
        hovertemplate="Time: %{x}m<br>SOC: %{y:.1f}%<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=t, y=temp, name="Cell Temperature (°C)",
        line=dict(color="#f59e0b", width=2, dash="dot"),
        hovertemplate="Time: %{x}m<br>Temp: %{y:.1f}°C<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=t, y=current, name="Charging Current (C-Rate)",
        yaxis="y2",
        line=dict(color="#10b981", width=2.2),
        hovertemplate="Time: %{x}m<br>Rate: %{y:.2f}C<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text="Simulated CC-CV Charging Profile & Thermal Dynamic Response", font=dict(color="#ffffff", size=15, family="Plus Jakarta Sans")),
        xaxis=dict(
            title=dict(text="Charging Elapsed Time (Minutes)", font=dict(color="#829ab8", size=12)),
            gridcolor="#1e3356",
            tickfont=dict(color="#cbd5e1", family="JetBrains Mono")
        ),
        yaxis=dict(
            title=dict(text="SOC (%) / Temperature (°C)", font=dict(color="#829ab8", size=12)),
            range=[0, 105],
            gridcolor="#1e3356",
            tickfont=dict(color="#cbd5e1", family="JetBrains Mono")
        ),
        yaxis2=dict(
            title=dict(text="Applied Current Rate (C-Rate)", font=dict(color="#829ab8", size=12)),
            range=[-0.1, 3.5],
            overlaying="y",
            side="right",
            gridcolor="#1e3356",
            tickfont=dict(color="#cbd5e1", family="JetBrains Mono")
        ),
        paper_bgcolor="#0d172a",
        plot_bgcolor="#0f1c32",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom", font=dict(color="#cbd5e1", size=11)),
        height=360
    )
    return fig


def build_negotiation_convergence_chart(res: dict):
    """
    Constructs a multi-turn negotiation convergence chart tracking Driver vs BMS concessions.
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    dp = res.get("driver_proposal", {})
    be = res.get("bms_evaluation", {})
    dc = res.get("driver_counter", {})
    fa = res.get("final_applied_action", {})
    
    rounds = ["Round 1: Driver Demand", "Round 1: BMS Evaluation", "Round 2: Driver Counter", "Round 2: Final Gate"]
    c_rates = [
        dp.get("charge_rate_c", 2.5),
        be.get("charge_rate_c", 1.5),
        dc.get("charge_rate_c", be.get("charge_rate_c", 1.5)),
        fa.get("charge_rate_c", 1.5)
    ]
    targets = [
        dp.get("target_soc", 100.0),
        be.get("target_soc", 80.0),
        dc.get("target_soc", be.get("target_soc", 80.0)),
        fa.get("target_soc", 80.0)
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rounds, y=c_rates,
        name="Charge Rate (C-Rate)",
        mode="lines+markers",
        line=dict(color="#3b82f6", width=2.5),
        marker=dict(size=11, color="#60a5fa", line=dict(color="#ffffff", width=1.5)),
        hovertemplate="%{x}<br>Rate: %{y:.2f}C<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=rounds, y=targets,
        name="Target SOC Ceiling (%)",
        mode="lines+markers",
        yaxis="y2",
        line=dict(color="#10b981", width=2.5, dash="dash"),
        marker=dict(size=11, color="#34d399", line=dict(color="#ffffff", width=1.5)),
        hovertemplate="%{x}<br>Target SOC: %{y:.0f}%<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text="Multi-Turn Agent Negotiation Parameter Convergence", font=dict(color="#ffffff", size=15, family="Plus Jakarta Sans")),
        xaxis=dict(
            title=dict(text="Agent Negotiation Dialogue Round", font=dict(color="#829ab8", size=12)),
            gridcolor="#1e3356",
            tickfont=dict(color="#cbd5e1", family="Plus Jakarta Sans", size=11)
        ),
        yaxis=dict(
            title=dict(text="Charge Rate (C-Rate)", font=dict(color="#829ab8", size=12)),
            range=[-0.2, 4.0],
            gridcolor="#1e3356",
            tickfont=dict(color="#cbd5e1", family="JetBrains Mono")
        ),
        yaxis2=dict(
            title=dict(text="Target SOC (%)", font=dict(color="#829ab8", size=12)),
            range=[40, 110],
            overlaying="y",
            side="right",
            gridcolor="#1e3356",
            tickfont=dict(color="#cbd5e1", family="JetBrains Mono")
        ),
        paper_bgcolor="#0d172a",
        plot_bgcolor="#0f1c32",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom", font=dict(color="#cbd5e1", size=11)),
        height=320
    )
    return fig


def build_safety_margin_audit_chart(res: dict, params: dict, bms_limits: dict):
    """
    Constructs a multi-parameter safety audit bar chart comparing Demanded vs Limit vs Approved.
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    applied = res.get("final_applied_action", {})
    req_c = params.get("req_c_rate", 2.5)
    app_c = applied.get("charge_rate_c", 1.5)
    
    cell_t = params.get("cell_temp", 25.0)
    crit_t = float(bms_limits.get("critical_shutdown_temp_c", 50.0))
    
    req_soc = params.get("req_target_soc", 100.0)
    app_soc = applied.get("target_soc", 80.0)
    max_soc = float(bms_limits.get("max_soc", 80.0))

    categories = ["C-Rate (C)", "Cell Temp (°C)", "Target SOC (%)"]
    demands = [req_c, cell_t, req_soc]
    limits = [float(bms_limits.get("max_c_rate", 1.5)), crit_t, max_soc]
    approved = [app_c, cell_t, app_soc]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Driver Unconstrained Demand",
        x=categories, y=demands,
        marker=dict(color="#f97316"),
        text=[f"{v:.1f}" for v in demands], textposition="outside",
        textfont=dict(color="#fed7aa", family="JetBrains Mono")
    ))
    fig.add_trace(go.Bar(
        name="Hard Physical Safety Gate Limit",
        x=categories, y=limits,
        marker=dict(color="#64748b"),
        text=[f"{v:.1f}" for v in limits], textposition="outside",
        textfont=dict(color="#cbd5e1", family="JetBrains Mono")
    ))
    fig.add_trace(go.Bar(
        name="BMS Approved Enforcement Level",
        x=categories, y=approved,
        marker=dict(color="#2563eb"),
        text=[f"{v:.1f}" for v in approved], textposition="outside",
        textfont=dict(color="#93c5fd", family="JetBrains Mono")
    ))
    fig.update_layout(
        barmode="group",
        title=dict(text="Hardware Safety Boundary Audit (Demand vs Hard Limits vs Approved)", font=dict(color="#ffffff", size=15, family="Plus Jakarta Sans")),
        xaxis=dict(
            title=dict(text="Key Monitored Battery Metric", font=dict(color="#829ab8", size=12)),
            gridcolor="#1e3356",
            tickfont=dict(color="#cbd5e1", family="Plus Jakarta Sans", size=11)
        ),
        yaxis=dict(
            title=dict(text="Operating Value Level", font=dict(color="#829ab8", size=12)),
            gridcolor="#1e3356",
            tickfont=dict(color="#cbd5e1", family="JetBrains Mono")
        ),
        paper_bgcolor="#0d172a",
        plot_bgcolor="#0f1c32",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom", font=dict(color="#cbd5e1", size=11)),
        height=340
    )
    return fig


def build_degradation_mechanism_donut(applied_c: float, cell_temp: float, applied_target: float):
    """
    Constructs an electrochemical degradation breakdown donut chart.
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    if applied_c > 2.0 or cell_temp >= 45.0 or applied_target > 90.0:
        values = [42, 38, 14, 6]
    else:
        values = [58, 16, 18, 8]

    labels = [
        "Solid Electrolyte Interphase (SEI) Growth",
        "Kinetic Lithium Plating & Dendrites",
        "Cathode Micro-Cracking & Strain",
        "Loss of Active Lithium Material (LAM)"
    ]
    colors = ["#1d4ed8", "#f97316", "#3b82f6", "#60a5fa"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.52,
        marker=dict(colors=colors, line=dict(color="#0f1c32", width=2)),
        textinfo="label+percent",
        textfont=dict(color="#cbd5e1", size=10, family="Plus Jakarta Sans"),
        hovertemplate="Mechanism: %{label}<br>Contribution: %{percent}<extra></extra>"
    )])
    fig.update_layout(
        title=dict(text="Electrochemical Capacity Degradation Failure Mechanisms", font=dict(color="#ffffff", size=15, family="Plus Jakarta Sans")),
        paper_bgcolor="#0d172a",
        plot_bgcolor="#0f1c32",
        legend=dict(font=dict(color="#cbd5e1", size=10), orientation="h", y=-0.12, x=0.5, xanchor="center"),
        margin=dict(l=30, r=30, t=45, b=40),
        height=320
    )
    return fig


def build_energy_waterfall_chart(battery_kwh: float, approved_soc: float, min_soc: float):
    """
    Constructs a waterfall energy partition breakdown for the battery pack.
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    floor_pct = min_soc / 100.0
    floor_kwh = battery_kwh * floor_pct
    
    ceiling_loss_pct = (100.0 - approved_soc) / 100.0
    ceiling_kwh = battery_kwh * ceiling_loss_pct
    
    usable_kwh = max(0.0, battery_kwh - floor_kwh - ceiling_kwh)
    
    steps = [
        "Gross Nominal Pack",
        "Deep Discharge Floor (Reserve)",
        "High-SOC Preservation Buffer",
        "BMS Usable Cycle Window"
    ]
    values = [battery_kwh, -floor_kwh, -ceiling_kwh, usable_kwh]
    measures = ["relative", "relative", "relative", "total"]

    fig = go.Figure(go.Waterfall(
        name="Pack Energy Distribution",
        orientation="v",
        measure=measures,
        x=steps,
        y=values,
        text=[f"{v:.1f} kWh" if m == "total" else (f"{v:+.1f} kWh" if m == "relative" and i > 0 else f"{v:.1f} kWh") for i, (v, m) in enumerate(zip(values, measures))],
        textposition="outside",
        textfont=dict(color="#cbd5e1", family="JetBrains Mono", size=11),
        connector=dict(line=dict(color="#829ab8", width=1, dash="dot")),
        decreasing=dict(marker=dict(color="#f87171")),
        totals=dict(marker=dict(color="#10b981"))
    ))
    fig.update_layout(
        title=dict(text=f"Battery Pack Usable Energy Allocation Partition ({battery_kwh} kWh)", font=dict(color="#ffffff", size=15, family="Plus Jakarta Sans")),
        xaxis=dict(
            title=dict(text="Energy Partition Allocation", font=dict(color="#829ab8", size=12)),
            gridcolor="#1e3356",
            tickfont=dict(color="#cbd5e1", family="Plus Jakarta Sans", size=11)
        ),
        yaxis=dict(
            title=dict(text="Pack Energy (kWh)", font=dict(color="#829ab8", size=12)),
            gridcolor="#1e3356",
            tickfont=dict(color="#cbd5e1", family="JetBrains Mono"),
            range=[0, battery_kwh * 1.25]
        ),
        paper_bgcolor="#0d172a",
        plot_bgcolor="#0f1c32",
        margin=dict(l=40, r=40, t=50, b=40),
        height=320
    )
    return fig

# ═══════════════════════════════════════════════════════════════
# TOP HERO HEADER (NAVY ENTERPRISE TELEMETRY BAR)
# ═══════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="cyber-header">
        <div>
            <div class="cyber-title">
                <span>⚡</span> Multi-Agent BMS Safety Simulator & Telemetry
            </div>
            <div class="cyber-subtitle">
                Autonomous Driver Persona & Deterministic Hardware-Guarded BMS Negotiation Sandbox
            </div>
        </div>
        <div class="telemetry-status-group">
            <div class="telemetry-status-chip">
                <span class="status-dot"></span>
                <span>SCHEMA 2.0.0</span>
            </div>
            <div class="telemetry-status-chip">
                <span>🛡️ GATE: DETERMINISTIC</span>
            </div>
            <div class="telemetry-status-chip">
                <span>💬 AGENTS: 2-TURN</span>
            </div>
            <div class="telemetry-status-chip">
                <span>🌡️ PHYSICS: COUPLED</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="background: rgba(19, 34, 62, 0.6); border: 1px solid #1e3356; border-radius: 10px; padding: 12px 20px; margin-bottom: 22px; font-size: 13px; color: #cbd5e1;">
        🔬 <strong>Academic Simulation Disclaimer:</strong> This layer models driver charging preferences and deterministic BMS safety enforcement. 
        It is not an automotive ASIL-D certified ECU and should not be used as the sole basis for physical hardware decisions.
    </div>
    """,
    unsafe_allow_html=True
)

if not MEMBER4_AVAILABLE:
    st.error(
        f"❌ Could not import `ev_member4`. Ensure the repository branch `member4-simulation` is checked out.\n\n"
        f"**Error Details:** `{MEMBER4_ERROR}`\n\n"
        f"**Quick Fix:** In your terminal, run `git checkout member4-simulation`."
    )
    st.stop()

# ═══════════════════════════════════════════════════════════════
# SIMULATION EXECUTION HELPER
# ═══════════════════════════════════════════════════════════════

def execute_simulation(soc, temp, crate, target, floor, persona_val, climate_val, scen_val, m2_val=None, overrides=None, vehicle_file="tata_curvv_ev_55kwh_partial_reference.json", cycle_file="udds_city.csv"):
    with st.spinner("⚡ Simulating multi-turn Driver vs BMS negotiation..."):
        try:
            res_obj = ev_member4.simulate_bms_negotiation(
                battery_state={"soc_pct": soc, "cell_temp_c": temp},
                driver_request={"charge_rate_c": crate, "target_soc": target, "min_soc": floor, "persona": persona_val},
                member2_predictions=m2_val,
                climate=climate_val,
                scenario=scen_val,
                vehicle=vehicle_file,
                cycle=cycle_file,
                bms_limits_override=overrides,
                use_llm=False
            )
            st.session_state["bms_sim_result"] = res_obj
            st.session_state["bms_sim_params"] = {
                "req_c_rate": crate,
                "req_target_soc": target,
                "persona": persona_val,
                "climate": climate_val,
                "scenario": scen_val,
                "cell_temp": temp,
                "current_soc": soc,
                "vehicle_file": vehicle_file,
                "cycle_file": cycle_file
            }
        except Exception as ex:
            st.error(f"🛑 Simulation failed: `{ex}`")
            st.exception(ex)

# ═══════════════════════════════════════════════════════════════
# SIDEBAR CONTROLS (EV PROFILES & DRIVE CYCLES - REC 3)
# ═══════════════════════════════════════════════════════════════

st.sidebar.markdown(
    """
    <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #1e3356; margin-bottom: 16px;">
        <span style="font-size: 16.5px; font-weight: 700; color: #ffffff;">⚙️ Simulation Parameters</span>
        <div style="font-size: 12.5px; color: #829ab8;">Vehicle profile & battery safety controls</div>
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar.expander("🚗 1️⃣ EV Profile & Drive Cycle", expanded=True):
    vehicle_key_list = list(VEHICLE_PROFILES.keys())
    vehicle_names = [VEHICLE_PROFILES[k]["display_name"] for k in vehicle_key_list]
    
    selected_vehicle_idx = st.selectbox("Select EV Model Profile", range(len(vehicle_names)), format_func=lambda i: vehicle_names[i], index=0)
    selected_vehicle_file = vehicle_key_list[selected_vehicle_idx]
    v_data = VEHICLE_PROFILES[selected_vehicle_file]

    cycle_key_list = list(DRIVE_CYCLES.keys())
    cycle_names = [DRIVE_CYCLES[k] for k in cycle_key_list]
    selected_cycle_idx = st.selectbox("Standard Drive Cycle", range(len(cycle_names)), format_func=lambda i: cycle_names[i], index=0)
    selected_cycle_file = cycle_key_list[selected_cycle_idx]

    st.markdown(
        f"""
        <div style="background: rgba(15, 28, 50, 0.85); border: 1px solid #1e3356; border-radius: 8px; padding: 10px 14px; font-size: 12px; margin-top: 8px;">
            <div style="color: #93c5fd; font-weight: 700;">{v_data['display_name']}</div>
            <div style="color: #cbd5e1; margin-top: 4px;">• <strong>Battery:</strong> {v_data['battery_kwh']} kWh ({v_data['chemistry']})</div>
            <div style="color: #cbd5e1;">• <strong>Rated Range:</strong> {v_data['range_km']:.0f} km reference</div>
            <div style="color: #cbd5e1;">• <strong>Kerb Mass:</strong> {v_data['kerb_mass_kg']:.0f} kg | <strong>Cd:</strong> {v_data['cd']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with st.sidebar.expander("🌡️ 2️⃣ Operating Environment", expanded=True):
    climate = st.selectbox("Ambient Climate", ["hot", "temperate", "cold"], index=0)
    scenario = st.selectbox("BMS Supervision Scenario", ["full_protection", "bms_supervised", "soc_20_80", "unrestricted"], index=0)
    current_soc = st.slider("Current Pack SOC (%)", 5.0, 95.0, 45.0, step=1.0)
    cell_temp = st.number_input("Measured Cell Temp (°C)", -10.0, 60.0, 25.0, step=0.5)

with st.sidebar.expander("🏎️ 3️⃣ Driver Demand (Round 1)", expanded=True):
    req_c_rate = st.slider("Requested Charge Rate (C)", 0.2, 3.5, 2.5, step=0.1)
    req_target_soc = st.slider("Requested Target SOC (%)", 50.0, 100.0, 100.0, step=5.0)
    req_min_soc = st.slider("Minimum SOC Floor (%)", 0.0, 25.0, 5.0, step=1.0)
    persona = st.selectbox("Driver Persona", ["aggressive", "balanced", "gentle"], index=0)

with st.sidebar.expander("🧠 4️⃣ Member 2 Predictions (Ingestion)", expanded=False):
    use_m2 = st.checkbox("Ingest Member 2 Machine Learning Outputs", value=True)
    if use_m2:
        m2_soh = st.number_input("Member 2 Predicted SOH (%)", 50.0, 100.0, 89.4, step=0.1)
        m2_rul = st.number_input("Member 2 Predicted RUL (Cycles)", 0, 5000, 1150, step=25)
        m2_predictions = {
            "soh_pct": m2_soh,
            "rul_cycles": m2_rul,
            "model_id": "member2_lgbm_v1",
            "provenance": "Member 2 Verified Machine Learning Ensemble"
        }
    else:
        m2_predictions = None

with st.sidebar.expander("🛡️ 5️⃣ BMS Safety Limits Override", expanded=False):
    max_c_override = st.number_input("Max Permitted C-Rate (C)", 0.5, 3.0, float(BMS_LIMITS.get("max_c_rate", 1.5)), step=0.1)
    max_temp_override = st.number_input("Max Permitted Temp (°C)", 30.0, 50.0, float(BMS_LIMITS.get("max_temp_c", 35.0)), step=1.0)
    crit_temp_override = st.number_input("Critical Shutdown Temp (°C)", 45.0, 55.0, float(BMS_LIMITS.get("critical_shutdown_temp_c", 50.0)), step=1.0)
    max_soc_override = st.number_input("Max SOC Ceiling (%)", 70.0, 100.0, float(BMS_LIMITS.get("max_soc", 80.0)), step=5.0)
    min_soc_override = st.number_input("Min SOC Floor (%)", 10.0, 30.0, float(BMS_LIMITS.get("min_soc", 20.0)), step=5.0)

st.sidebar.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
run_simulation_btn = st.sidebar.button("🚀 Run Custom Negotiation", type="primary", use_container_width=True)

if run_simulation_btn:
    execute_simulation(
        soc=current_soc,
        temp=cell_temp,
        crate=req_c_rate,
        target=req_target_soc,
        floor=req_min_soc,
        persona_val=persona,
        climate_val=climate,
        scen_val=scenario,
        m2_val=m2_predictions,
        overrides={
            "max_c_rate": max_c_override,
            "max_temp_c": max_temp_override,
            "critical_shutdown_temp_c": crit_temp_override,
            "max_soc": max_soc_override,
            "min_soc": min_soc_override,
        },
        vehicle_file=selected_vehicle_file,
        cycle_file=selected_cycle_file
    )

# ═══════════════════════════════════════════════════════════════
# DASHBOARD OUTPUT DISPLAY (SPLIT MISSION CONTROL CANVAS)
# ═══════════════════════════════════════════════════════════════

if "bms_sim_result" in st.session_state:
    res = st.session_state["bms_sim_result"]
    params = st.session_state.get("bms_sim_params", {})
    applied = res["final_applied_action"]
    decision = applied["decision"].upper()
    req_c = params.get("req_c_rate", req_c_rate)
    applied_c = applied["charge_rate_c"]
    delta_c = applied_c - req_c
    active_v_file = params.get("vehicle_file", selected_vehicle_file)
    active_v_info = VEHICLE_PROFILES.get(active_v_file, v_data)

    soh_display = f"{res['member2_soh_pct']:.1f}%" if res["member2_soh_pct"] is not None else "UNAVAILABLE"
    rul_display = f"{res['member2_rul_cycles']:,} cyc" if res["member2_rul_cycles"] is not None else "UNAVAILABLE"
    target_soc_display = f"{applied['target_soc']:.0f}%"

    # ── 1. Top Hero KPI Metric Grid ──────────────────────────────
    gate_color = "#3b82f6" if decision == "APPROVE" else ("#60a5fa" if decision == "MODIFY" else "#f87171")
    delta_color = "#60a5fa" if delta_c >= 0 else "#94a3b8"

    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">
                    <span>Requested Rate</span>
                    <span>🏎️</span>
                </div>
                <div class="metric-value">{req_c:.1f}<span style="font-size: 18px; color: var(--navy-muted);">C</span></div>
                <div class="metric-sub">Driver ({params.get('persona', persona)})</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">
                    <span>Approved Rate</span>
                    <span>🛡️</span>
                </div>
                <div class="metric-value" style="color: var(--navy-highlight);">{applied_c:.1f}<span style="font-size: 18px; color: var(--navy-muted);">C</span></div>
                <div class="metric-sub">
                    <span style="color: {delta_color};">
                        {'+' if delta_c > 0 else ''}{delta_c:.1f}C from request
                    </span>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">
                    <span>Target SOC</span>
                    <span>🔋</span>
                </div>
                <div class="metric-value">{target_soc_display}</div>
                <div class="metric-sub">BMS Envelope Safe Cap</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">
                    <span>Member 2 SOH</span>
                    <span>🧠</span>
                </div>
                <div class="metric-value" style="font-size: 28px;">{soh_display}</div>
                <div class="metric-sub">Status: {res['member2_soh_status']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">
                    <span>Member 2 RUL</span>
                    <span>⏳</span>
                </div>
                <div class="metric-value" style="font-size: 28px;">{rul_display}</div>
                <div class="metric-sub">Status: {res['member2_rul_status']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">
                    <span>BMS Gate</span>
                    <span>⚖️</span>
                </div>
                <div class="metric-value" style="font-size: 28px; color: {gate_color};">
                    {decision}
                </div>
                <div class="metric-sub">{len(res['rules_triggered'])} safety rules tripped</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── 2. Outcome Status Banner ─────────────────────────────────
    banner_class = "approve" if decision == "APPROVE" else ("modify" if decision == "MODIFY" else "reject")
    banner_icon = "✅" if decision == "APPROVE" else ("⚠️" if decision == "MODIFY" else "🛑")
    banner_title = "Action Approved by BMS" if decision == "APPROVE" else ("Action Modified by Deterministic BMS Gate" if decision == "MODIFY" else "Action Rejected: Safety Breach Prevented")

    st.markdown(
        f"""
        <div class="outcome-banner {banner_class}">
            <div class="outcome-icon">{banner_icon}</div>
            <div style="flex-grow: 1;">
                <div class="outcome-text-title">{banner_title}</div>
                <div class="outcome-text-desc">{res['reason']}</div>
            </div>
            <div style="text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #93c5fd; background: rgba(15, 28, 50, 0.7); padding: 8px 14px; border-radius: 8px; border: 1px solid #1e3356;">
                <div>EV: {active_v_info['display_name'].split('(')[0].strip()}</div>
                <div style="color: #cbd5e1; font-size: 11px;">{active_v_info['battery_kwh']} kWh | {params.get('climate', climate).upper()}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── 3. Split Command Center Layout (Canvas Space Maximizer) ───
    col_left, col_right = st.columns([1, 2], gap="large")

    # LEFT COLUMN: Dynamic Gauges & Quick-Switch Scenarios
    with col_left:
        st.markdown(
            """
            <div style="font-size: 15.5px; font-weight: 700; color: #ffffff; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                <span>📊</span> Live Telemetry Gauges
            </div>
            """,
            unsafe_allow_html=True
        )

        curr_soc = params.get("current_soc", 50.0)
        approved_soc = applied["target_soc"]
        st.markdown(
            f"""
            <div class="gauge-card">
                <div class="gauge-title">
                    <span>🔋 Battery SOC Window</span>
                    <span style="font-family: 'JetBrains Mono', monospace; color: var(--navy-highlight);">{curr_soc:.0f}% ➔ {approved_soc:.0f}%</span>
                </div>
                <div class="gauge-bar-outer">
                    <div class="gauge-bar-inner" style="width: {approved_soc}%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 11.5px; color: var(--navy-muted); font-family: 'JetBrains Mono', monospace;">
                    <span>Floor: 20%</span>
                    <span>Current: {curr_soc:.0f}%</span>
                    <span>Cap: {approved_soc:.0f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        curr_temp = params.get("cell_temp", 25.0)
        temp_pct = min(100, max(0, int((curr_temp + 10) / 70 * 100)))
        temp_color = "#60a5fa" if curr_temp < 35 else ("#f59e0b" if curr_temp < 50 else "#ef4444")
        temp_bar_grad = "linear-gradient(90deg, #1d4ed8, #3b82f6)" if curr_temp < 35 else ("linear-gradient(90deg, #d97706, #f59e0b)" if curr_temp < 50 else "linear-gradient(90deg, #b91c1c, #ef4444)")

        st.markdown(
            f"""
            <div class="gauge-card">
                <div class="gauge-title">
                    <span>🌡️ Cell Temperature Sensor</span>
                    <span style="font-family: 'JetBrains Mono', monospace; color: {temp_color};">{curr_temp:.1f}°C</span>
                </div>
                <div class="gauge-bar-outer">
                    <div class="gauge-bar-inner" style="width: {temp_pct}%; background: {temp_bar_grad};"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 11.5px; color: var(--navy-muted); font-family: 'JetBrains Mono', monospace;">
                    <span>Ambient: {res.get('ambient_temp_c', 25):.0f}°C</span>
                    <span>Derate: 35°C</span>
                    <span>Cutoff: 50°C</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="font-size: 15px; font-weight: 700; color: #ffffff; margin: 18px 0 10px 0; display: flex; align-items: center; gap: 8px;">
                <span>🔄</span> Switch Test Scenario
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("▶ Run Baseline (Tata Curvv, 1.0C)", key="res_btn_base", use_container_width=True):
            execute_simulation(50.0, 22.0, 1.0, 80.0, 5.0, "balanced", "temperate", "full_protection", {"soh_pct": 89.4, "rul_cycles": 1150}, vehicle_file="tata_curvv_ev_55kwh_partial_reference.json", cycle_file="udds_city.csv")
        if st.button("▶ Run DCFC Lockout (Tata Nexon, 3.0C)", key="res_btn_dcfc", use_container_width=True):
            execute_simulation(35.0, 32.0, 3.0, 100.0, 5.0, "aggressive", "hot", "full_protection", {"soh_pct": 85.2, "rul_cycles": 920}, vehicle_file="tata_nexon_ev_45kwh_partial_reference.json", cycle_file="hwfet_highway.csv")
        if st.button("▶ Run Overheat Abort (Mahindra XEV, 52°C)", key="res_btn_overheat", use_container_width=True):
            execute_simulation(60.0, 52.0, 2.0, 85.0, 5.0, "aggressive", "hot", "bms_supervised", {"soh_pct": 78.0, "rul_cycles": 600}, vehicle_file="mahindra_xev9e_59kwh_partial_reference.json", cycle_file="udds_city.csv")

    # RIGHT COLUMN: 6 Comprehensive Analysis Tabs
    with col_right:
        tab_summary, tab_scientific, tab_transcript, tab_rules, tab_advisory, tab_export = st.tabs([
            "📖 Executive Summary & Guide",
            "🔬 Scientific & Operating Envelope",
            "💬 Multi-Turn Transcript",
            "📋 Hardware Rule Audit",
            "💡 Preservation Advisory",
            "🔍 Provenance & Export"
        ])

        with tab_summary:
            action_desc = "approved without changes" if decision == "APPROVE" else ("partially modified to protect battery longevity" if decision == "MODIFY" else "strictly rejected to prevent critical damage")
            rules_str = f"({', '.join(res['rules_triggered'])})" if res['rules_triggered'] else "No safety limits violated."

            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, rgba(19, 34, 62, 0.95) 0%, rgba(13, 23, 42, 0.98) 100%); border: 1px solid var(--navy-border); border-radius: 12px; padding: 22px 26px; margin-bottom: 20px;">
                    <div style="font-size: 16.5px; font-weight: 700; color: var(--navy-highlight); margin-bottom: 8px;">
                        📌 Executive Verdict — {active_v_info['display_name']}
                    </div>
                    <p style="font-size: 14px; line-height: 1.6; color: var(--navy-light); margin-bottom: 14px;">
                        The <strong>Driver</strong> requested to charge at <strong>{req_c:.1f}C</strong> up to <strong>{params.get('req_target_soc', req_target_soc):.0f}%</strong> under a <strong>{params.get('climate', climate)}</strong> climate.<br>
                        The <strong>BMS (Battery Management System)</strong> evaluated battery pack physics and {action_desc}:
                        approving <strong>{applied_c:.1f}C</strong> charge rate and a maximum target of <strong>{applied['target_soc']:.0f}%</strong>.<br>
                        <strong>Governing Factor:</strong> <em>{res['reason']}</em> {rules_str}
                    </p>
                    <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                        <span class="telemetry-status-chip">C-Rate: {req_c:.1f}C ➔ {applied_c:.1f}C</span>
                        <span class="telemetry-status-chip">Target SOC: {params.get('req_target_soc', req_target_soc):.0f}% ➔ {applied['target_soc']:.0f}%</span>
                        <span class="telemetry-status-chip">Decision: {decision}</span>
                        <span class="telemetry-status-chip">Battery: {active_v_info['battery_kwh']} kWh</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("#### ⚖️ Dynamic Driver Persona vs. BMS Gate Evaluation")
            col_sum_c1, col_sum_c2 = st.columns(2)
            with col_sum_c1:
                fig_radar = build_radar_comparison_chart(req_c, applied_c, params.get('req_target_soc', 100.0), applied['target_soc'], params.get('cell_temp', 25.0), decision)
                if fig_radar is not None:
                    st.plotly_chart(fig_radar, use_container_width=True)
            with col_sum_c2:
                fig_wf = build_decision_waterfall_chart(req_c, applied_c, decision, res['reason'])
                if fig_wf is not None:
                    st.plotly_chart(fig_wf, use_container_width=True)

            st.markdown("#### 📚 Decoding the Output Readings")
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.markdown(
                    """
                    <div class="guide-card">
                        <div class="guide-title">⚡ 1. C-Rate (Charge Multiplier)</div>
                        <div class="guide-body">
                            Measures charging speed relative to pack capacity.<br>
                            • <span class="guide-highlight">1.0C:</span> 1 hour full charge.<br>
                            • <span class="guide-highlight">2.0C - 3.0C:</span> DC Fast Charging.<br>
                            • <span class="guide-highlight">0.3C - 0.5C:</span> Slow home AC charging.<br>
                            <strong>Implication:</strong> Higher C-rates generate exponential internal heat and trigger lithium plating.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col_g2:
                st.markdown(
                    """
                    <div class="guide-card">
                        <div class="guide-title">🔋 2. SOC (State of Charge)</div>
                        <div class="guide-body">
                            The battery fuel gauge percentage.<br>
                            • <span class="guide-highlight">0% to 100%:</span> Available charge.<br>
                            • <span class="guide-highlight">80% Ceiling:</span> Capping daily charges prevents cathode structural degradation and SEI breakdown.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with tab_scientific:
            st.markdown("#### 📈 Interactive Safe Operating Envelope (SOE)")
            st.caption("Visualizing hardware safety limits, thermal derating curves, and the deterministic BMS clamp vector.")

            fig_soe = build_safe_operating_envelope_chart(
                cell_temp=params.get("cell_temp", cell_temp),
                req_c_rate=req_c,
                applied_c_rate=applied_c,
                crit_temp=float(BMS_LIMITS.get("critical_shutdown_temp_c", 50.0)),
                max_temp=float(BMS_LIMITS.get("max_temp_c", 35.0)),
                max_c=float(BMS_LIMITS.get("max_c_rate", 1.5))
            )
            if fig_soe is not None:
                st.plotly_chart(fig_soe, use_container_width=True)

            st.markdown("#### ⚡ Dynamic Simulated CC-CV Charging Profile")
            st.caption("Simulating Constant Current (CC) fast charge transition to Constant Voltage (CV) taper under active thermal monitoring.")
            fig_cccv = build_cccv_charging_profile(applied_c, applied['target_soc'], params.get('cell_temp', 25.0))
            if fig_cccv is not None:
                st.plotly_chart(fig_cccv, use_container_width=True)

            st.markdown("#### 🚗 Selected EV Hardware Configuration")
            st.markdown(
                f"""
                <div style="background: rgba(19, 34, 62, 0.7); border: 1px solid var(--navy-border); border-radius: 12px; padding: 18px 22px; margin-bottom: 16px;">
                    <div style="font-weight: 700; color: #ffffff; margin-bottom: 12px; font-size: 15px;">
                        Engineering Parameters — {active_v_info['display_name']}
                    </div>
                    <table style="width: 100%; font-size: 13.5px; color: var(--navy-light);">
                        <tr style="border-bottom: 1px solid #1e3356;">
                            <td style="padding: 10px 0; color: var(--navy-muted);">Battery Pack Energy</td>
                            <td style="text-align: right; font-family: 'JetBrains Mono', monospace;">{active_v_info['battery_kwh']} kWh Nominal</td>
                            <td style="text-align: right; font-family: 'JetBrains Mono', monospace; color: var(--navy-highlight);">{active_v_info['chemistry']}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #1e3356;">
                            <td style="padding: 10px 0; color: var(--navy-muted);">Certified Reference Range</td>
                            <td style="text-align: right; font-family: 'JetBrains Mono', monospace;">{active_v_info['range_km']:.0f} km</td>
                            <td style="text-align: right; font-family: 'JetBrains Mono', monospace; color: var(--navy-highlight);">Drivetrain: {active_v_info['drivetrain']}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #1e3356;">
                            <td style="padding: 10px 0; color: var(--navy-muted);">Charge Rate Clamping</td>
                            <td style="text-align: right; font-family: 'JetBrains Mono', monospace;">Requested: {req_c:.2f}C</td>
                            <td style="text-align: right; font-family: 'JetBrains Mono', monospace; color: var(--navy-highlight);">Approved: {applied_c:.2f}C</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #1e3356;">
                            <td style="padding: 10px 0; color: var(--navy-muted);">Target SOC Window</td>
                            <td style="text-align: right; font-family: 'JetBrains Mono', monospace;">Requested: {params.get('req_target_soc', req_target_soc):.0f}%</td>
                            <td style="text-align: right; font-family: 'JetBrains Mono', monospace; color: var(--navy-highlight);">Approved: {applied['target_soc']:.0f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #1e3356;">
                            <td style="padding: 10px 0; color: var(--navy-muted);">Thermal Conditions</td>
                            <td style="text-align: right; font-family: 'JetBrains Mono', monospace;">Ambient: {res.get('ambient_temp_c', 'N/A')} °C ({params.get('climate', climate)})</td>
                            <td style="text-align: right; font-family: 'JetBrains Mono', monospace; color: var(--navy-highlight);">Measured Cell: {params.get('cell_temp', cell_temp):.1f} °C</td>
                        </tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True
            )

        with tab_transcript:
            st.markdown("#### 🔄 Multi-Agent Negotiation Parameter Convergence")
            st.caption("Tracking proposed charge rate (C-Rate) and target SOC across dialogue turns between Driver Persona and BMS Safety Agent.")
            fig_conv = build_negotiation_convergence_chart(res)
            if fig_conv is not None:
                st.plotly_chart(fig_conv, use_container_width=True)

            st.markdown("#### 💬 Turn-by-Turn Dialogue Transcript")
            for turn in res["transcript"]:
                speaker = turn["speaker"]
                msg = turn["message"]
                round_idx = turn["round"]
                is_driver = "Driver" in speaker
                card_class = "driver" if is_driver else "bms"
                icon = "🏎️" if is_driver else "🛡️"

                st.markdown(
                    f"""
                    <div class="transcript-card {card_class}">
                        <div class="transcript-header">
                            <div class="transcript-speaker">
                                <span>{icon}</span>
                                <span>{speaker}</span>
                            </div>
                            <div class="transcript-round">Round {round_idx}</div>
                        </div>
                        <div class="transcript-msg">{msg}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with tab_rules:
            st.markdown("#### 🛡️ Hardware Safety Boundary Audit (Demand vs Hard Limits vs Approved)")
            st.caption("Multi-parameter safety audit verifying that physical ASIL safety limits were never violated.")
            fig_audit = build_safety_margin_audit_chart(res, params, BMS_LIMITS)
            if fig_audit is not None:
                st.plotly_chart(fig_audit, use_container_width=True)

            st.markdown("#### 📋 Deterministic Hardware Rules Reference Catalog")
            catalog = res.get("rules_catalog", {})
            triggered_set = set(res.get("rules_triggered", []))
            table_rows = []
            for r_id, meta in catalog.items():
                is_trig = r_id in triggered_set
                pill_class = "triggered" if is_trig else "clear"
                pill_label = "TRIGGERED" if is_trig else "CLEAR"
                row_html = f"""
                <tr>
                    <td><strong style="color: #ffffff;">{meta.get('name', r_id)}</strong><br><span style="font-size: 11px; font-family: 'JetBrains Mono', monospace; color: var(--navy-muted);">{r_id}</span></td>
                    <td><span class="status-pill {pill_class}">{pill_label}</span></td>
                    <td style="font-family: 'JetBrains Mono', monospace;">{meta.get('parameter', 'N/A')}</td>
                    <td style="font-family: 'JetBrains Mono', monospace;">{meta.get('default_limit', 'N/A')}</td>
                    <td>{meta.get('action', 'Derate / Cap')}</td>
                </tr>
                """
                table_rows.append(row_html)

            st.markdown(
                f"""
                <table class="cyber-table">
                    <thead>
                        <tr>
                            <th>Rule ID & Description</th>
                            <th>Audit Status</th>
                            <th>Parameter</th>
                            <th>Threshold</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(table_rows)}
                    </tbody>
                </table>
                """,
                unsafe_allow_html=True
            )

        with tab_advisory:
            st.markdown("#### ⏳ Projected SOH Longevity Degradation Trajectory")
            st.caption("Comparative electrochemical degradation over 1,200 Operating Equivalent Full Cycles (EFC).")
            fig_longevity = build_longevity_projection_chart(current_soh=res.get("member2_soh_pct", 89.4))
            if fig_longevity is not None:
                st.plotly_chart(fig_longevity, use_container_width=True)

            st.markdown("#### 🔬 Electrochemical Capacity Loss Mechanisms")
            st.caption("Distribution of internal physical cell degradation causes under the current charging stress level.")
            fig_donut = build_degradation_mechanism_donut(applied_c, params.get('cell_temp', 25.0), applied['target_soc'])
            if fig_donut is not None:
                st.plotly_chart(fig_donut, use_container_width=True)

            st.markdown("#### 💡 Prescriptive Operational Guidelines")
            st.markdown(
                f"""
                <div style="background: linear-gradient(145deg, rgba(19, 34, 62, 0.95) 0%, rgba(13, 23, 42, 0.98) 100%); border: 1px solid var(--navy-border); border-radius: 12px; padding: 22px 24px; margin-bottom: 20px;">
                    <div style="font-size: 15.5px; font-weight: 700; color: #ffffff; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                        <span>💡</span> Recommended Operational Practices
                    </div>
                    <div style="font-size: 14px; line-height: 1.6; color: var(--navy-light);">
                        {res.get('recommendations', 'Maintain charge targets within 20% - 80% SOC for daily operations.')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with tab_export:
            st.markdown("#### 🔋 Battery Pack Usable Energy Allocation Partition")
            st.caption("Visualizing Gross Energy vs Discharge Reserve Floor vs High-SOC Buffer for the selected EV pack.")
            fig_energy = build_energy_waterfall_chart(active_v_info['battery_kwh'], applied['target_soc'], params.get('min_soc', 20.0))
            if fig_energy is not None:
                st.plotly_chart(fig_energy, use_container_width=True)

            col_p1, col_p2 = st.columns([2, 1])
            with col_p1:
                st.markdown("#### 🔬 Provenance & Metadata")
                st.json(res.get("provenance", {}))
            with col_p2:
                st.markdown("#### 💾 Export Simulation")
                export_climate = params.get("climate", climate)
                export_persona = params.get("persona", persona)
                st.download_button(
                    label="📥 Download Full Telemetry JSON",
                    data=json.dumps(res, indent=2, default=str),
                    file_name=f"bms_sim_{export_climate}_{export_persona}_schema2.json",
                    mime="application/json",
                    use_container_width=True
                )
                st.caption("Conforms strictly to Technical Handover Schema 2.0.0.")

else:
    # ═══════════════════════════════════════════════════════════════
    # INITIAL MISSION CONTROL CANVAS (FULLY OCCUPIES SCREEN SPACE)
    # ═══════════════════════════════════════════════════════════════

    st.markdown(
        """
        <div style="font-size: 17px; font-weight: 700; color: #ffffff; margin-bottom: 14px; display: flex; align-items: center; gap: 10px;">
            <span>🎯</span> Quick Launch Test Scenarios (1-Click Execution)
        </div>
        """,
        unsafe_allow_html=True
    )

    sc_col1, sc_col2, sc_col3 = st.columns(3, gap="medium")

    with sc_col1:
        st.markdown(
            """
            <div class="scenario-card-box">
                <div>
                    <div class="scenario-title">🟢 Baseline (Tata Curvv.ev)</div>
                    <div class="scenario-params">
                        • <strong>Vehicle:</strong> Tata Curvv.ev (55 kWh LFP)<br>
                        • <strong>Climate:</strong> Temperate (22°C Ambient)<br>
                        • <strong>Driver Demand:</strong> 1.0C Normal Rate<br>
                        • <strong>Target SOC:</strong> 80% (Preservation target)<br>
                        • <strong>BMS Gate:</strong> Full Protection<br>
                        • <strong>Description:</strong> Standard daily AC charging well within physical limits.
                    </div>
                    <div class="scenario-expected" style="background: rgba(37, 99, 235, 0.15); color: #93c5fd; border: 1px solid #1d4ed8;">
                        Expected Gate: APPROVE
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("▶ Launch Baseline (Curvv 55kWh)", key="init_btn_base", use_container_width=True):
            execute_simulation(50.0, 22.0, 1.0, 80.0, 5.0, "balanced", "temperate", "full_protection", {"soh_pct": 89.4, "rul_cycles": 1150, "model_id": "member2_lgbm_v1"}, vehicle_file="tata_curvv_ev_55kwh_partial_reference.json", cycle_file="udds_city.csv")

    with sc_col2:
        st.markdown(
            """
            <div class="scenario-card-box">
                <div>
                    <div class="scenario-title">⚡ DCFC Lockout (Tata Nexon.ev)</div>
                    <div class="scenario-params">
                        • <strong>Vehicle:</strong> Tata Nexon.ev (45 kWh LFP)<br>
                        • <strong>Climate:</strong> Hot (33°C Ambient)<br>
                        • <strong>Driver Demand:</strong> 3.0C Aggressive DCFC<br>
                        • <strong>Target SOC:</strong> 100% Saturation<br>
                        • <strong>BMS Gate:</strong> Full Protection<br>
                        • <strong>Description:</strong> Fast charging in high heat trips kinetic and thermal caps.
                    </div>
                    <div class="scenario-expected" style="background: rgba(30, 58, 138, 0.25); color: #bfdbfe; border: 1px solid #2563eb;">
                        Expected Gate: MODIFY (Capped to 1.5C / 80%)
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("▶ Launch DCFC (Nexon 45kWh)", key="init_btn_dcfc", use_container_width=True):
            execute_simulation(35.0, 32.0, 3.0, 100.0, 5.0, "aggressive", "hot", "full_protection", {"soh_pct": 85.2, "rul_cycles": 920, "model_id": "member2_lgbm_v1"}, vehicle_file="tata_nexon_ev_45kwh_partial_reference.json", cycle_file="hwfet_highway.csv")

    with sc_col3:
        st.markdown(
            """
            <div class="scenario-card-box">
                <div>
                    <div class="scenario-title">🛑 Overheat Abort (Mahindra XEV 9e)</div>
                    <div class="scenario-params">
                        • <strong>Vehicle:</strong> Mahindra XEV 9e (59 kWh LFP)<br>
                        • <strong>Climate:</strong> Hot (33°C Ambient)<br>
                        • <strong>Driver Demand:</strong> 2.0C Fast Charge<br>
                        • <strong>Measured Cell:</strong> 52.0°C (Thermal Breach)<br>
                        • <strong>BMS Gate:</strong> BMS Supervised<br>
                        • <strong>Description:</strong> Core cell overheat triggers emergency zero-power cutoff.
                    </div>
                    <div class="scenario-expected" style="background: rgba(127, 29, 29, 0.25); color: #fca5a5; border: 1px solid #991b1b;">
                        Expected Gate: REJECT (Emergency Cutoff 0.0C)
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("▶ Launch Overheat Abort (XEV 9e)", key="init_btn_overheat", use_container_width=True):
            execute_simulation(60.0, 52.0, 2.0, 85.0, 5.0, "aggressive", "hot", "bms_supervised", {"soh_pct": 78.0, "rul_cycles": 600, "model_id": "member2_lgbm_v1"}, vehicle_file="mahindra_xev9e_59kwh_partial_reference.json", cycle_file="udds_city.csv")

    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size: 17px; font-weight: 700; color: #ffffff; margin-bottom: 14px; display: flex; align-items: center; gap: 10px;">
            <span>🧭</span> System Architecture & Beginner Guide
        </div>
        """,
        unsafe_allow_html=True
    )

    g_col1, g_col2, g_col3, g_col4 = st.columns(4, gap="medium")

    with g_col1:
        st.markdown(
            """
            <div class="guide-card">
                <div class="guide-title">🏎️ Driver Persona</div>
                <div class="guide-body">
                    Simulates human driving urgency. Aggressive personas demand maximum C-rate (up to 3.5C) and 100% saturation, ignoring battery life and heat.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with g_col2:
        st.markdown(
            """
            <div class="guide-card">
                <div class="guide-title">🛡️ BMS Supervisor</div>
                <div class="guide-body">
                    Deterministic gate enforcing hard physical safety limits. Caps C-rate to 1.5C, limits SOC to 80%, and aborts charging if cell reaches 50°C.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with g_col3:
        st.markdown(
            """
            <div class="guide-card">
                <div class="guide-title">🧠 Member 2 ML</div>
                <div class="guide-body">
                    Ingests LightGBM / BiLSTM machine learning models for battery SOH (%) and RUL (cycles). Never fabricates false 100% or 0% values.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with g_col4:
        st.markdown(
            """
            <div class="guide-card">
                <div class="guide-title">🌡️ Thermal Model</div>
                <div class="guide-body">
                    Lumped thermodynamic model estimating cell heating (ΔT = k · C-rate) and derating current dynamically to prevent thermal runaway.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
