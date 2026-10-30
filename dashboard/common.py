import streamlit as st

ILLUSTRATIVE_WARNING = ("Vehicle parameters are partially sourced / illustrative and uncalibrated. Results are "
                        "illustrative estimates, NOT validated real-world range or manufacturer specifications.")
SIM_WARNING = ("Simulated scenario: inputs are user-chosen scenario values and degradation coefficients are Member 4 "
               "placeholders. This is not measured battery data and not real BMS behaviour.")


def badge(kind: str, text: str):
    {"ok": st.success, "warn": st.warning, "bad": st.error, "info": st.info}[kind](text)
