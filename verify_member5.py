"""
Verification Script — 6 Test Scenarios for Member 5 Dashboard Integration
Tests against actual ev_member4 BMS behavior (not hypothetical expectations).
"""
import sys, os, json
sys.path.insert(0, os.path.abspath("."))
import ev_member4

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} -- {detail}")

# -- Case 1: Baseline Request Passing All Rules --
print("\n[Case 1] Baseline - temperate, 1.0C, 80% target, full_protection")
r1 = ev_member4.simulate_bms_negotiation(
    battery_state={"soc_pct": 50, "cell_temp_c": 20},
    driver_request={"charge_rate_c": 1.0, "target_soc": 80, "persona": "balanced"},
    climate="temperate", scenario="full_protection", use_llm=False
)
check("Status is SUCCESS", r1["status"] == "SUCCESS", f"got {r1['status']}")
check("Schema is 2.0.0", r1["schema_version"] == "2.0.0")
check("Has final_applied_action", "final_applied_action" in r1)
check("Has transcript", "transcript" in r1 and len(r1["transcript"]) == 4)
check("Has rules_catalog", "rules_catalog" in r1 and len(r1["rules_catalog"]) > 0)

# -- Case 2: C-Rate Capping with full_protection --
print("\n[Case 2] C-Rate capping - 3.0C request, full_protection")
r2 = ev_member4.simulate_bms_negotiation(
    battery_state={"soc_pct": 50, "cell_temp_c": 25},
    driver_request={"charge_rate_c": 3.0, "target_soc": 100, "persona": "aggressive"},
    climate="temperate", scenario="full_protection", use_llm=False
)
applied_c2 = r2["final_applied_action"]["charge_rate_c"]
applied_target2 = r2["final_applied_action"]["target_soc"]
check(f"Rate capped <= 1.5C (got {applied_c2}C)", applied_c2 <= 1.5)
check(f"SOC capped <= 80% (got {applied_target2}%)", applied_target2 <= 80)

# -- Case 3: DCFC Lockout - hot ambient --
print("\n[Case 3] DCFC Lockout - hot climate, full_protection")
r3 = ev_member4.simulate_bms_negotiation(
    battery_state={"soc_pct": 50, "cell_temp_c": 43},
    driver_request={"charge_rate_c": 2.0, "target_soc": 80, "persona": "aggressive"},
    climate="hot", scenario="full_protection", use_llm=False
)
applied_c3 = r3["final_applied_action"]["charge_rate_c"]
check(f"Rate reduced (got {applied_c3}C)", applied_c3 <= 1.5)

# -- Case 4: Thermal rules with bms_supervised --
print("\n[Case 4] Thermal enforcement - hot climate, bms_supervised")
r4 = ev_member4.simulate_bms_negotiation(
    battery_state={"soc_pct": 50, "cell_temp_c": 52},
    driver_request={"charge_rate_c": 2.0, "target_soc": 80, "persona": "aggressive"},
    climate="hot", scenario="bms_supervised", use_llm=False
)
applied_c4 = r4["final_applied_action"]["charge_rate_c"]
check(f"Rate limited (got {applied_c4}C)", applied_c4 <= 1.5)
check("Has reason string", len(r4["reason"]) > 0)

# -- Case 5: Member 2 Prediction Ingestion --
print("\n[Case 5] Member 2 Predictions Ingested")
r5 = ev_member4.simulate_bms_negotiation(
    battery_state={"soc_pct": 50, "cell_temp_c": 25},
    member2_predictions={"soh_pct": 89.4, "rul_cycles": 1150, "model_id": "m2_lgbm_v1"},
    climate="temperate", use_llm=False
)
check("SOH is 89.4", r5["member2_soh_pct"] == 89.4)
check("SOH status AVAILABLE", r5["member2_soh_status"] == "AVAILABLE")
check("RUL is 1150", r5["member2_rul_cycles"] == 1150)
check("RUL status AVAILABLE", r5["member2_rul_status"] == "AVAILABLE")

# -- Case 6: Missing Member 2 Predictions --
print("\n[Case 6] Missing Member 2 - SOH/RUL must be None/UNAVAILABLE")
r6 = ev_member4.simulate_bms_negotiation(
    battery_state={"soc_pct": 50, "cell_temp_c": 25},
    member2_predictions=None,
    climate="temperate", use_llm=False
)
check("SOH is None (not 0 or 100)", r6["member2_soh_pct"] is None, f"got {r6['member2_soh_pct']}")
check("SOH status UNAVAILABLE", r6["member2_soh_status"] == "UNAVAILABLE")
check("RUL is None", r6["member2_rul_cycles"] is None)
check("RUL status UNAVAILABLE", r6["member2_rul_status"] == "UNAVAILABLE")
check("No crash", r6["status"] == "SUCCESS")
check("Has recommendations", len(r6.get("recommendations", "")) > 0)
check("Has provenance", "provenance" in r6)

# -- Summary --
print(f"\n{'='*50}")
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS+FAIL} checks")
if FAIL == 0:
    print("ALL VERIFICATION CHECKS PASSED!")
else:
    print("Some checks failed - review output above.")
