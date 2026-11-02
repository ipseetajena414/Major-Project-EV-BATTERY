# Member 4 — Multi-Agent Simulation & BMS Advisory (Schema 2.0.0)

A Driver Persona agent (fast charge demand to 100%, high C-rate preference) and a deterministic BMS Safety Supervisor agent (thermal, C-rate, SOC limits, and catastrophic cutoff) negotiate charging and driving actions across a multi-turn LangGraph workflow.

```
driver_proposes (R1) 
    -> bms_evaluates (R1) 
    -> driver_responds (R2 Counter / Concession) 
    -> bms_final_verifies (R2 Deterministic Gate) 
    -> apply_action 
    -> update_battery_state 
    -> check_end_of_life 
    -> (loop)
```

## Architectural & Scientific Principles

1. **Deterministic Safety Enforcement**: Safety rules (`RULES_CATALOG` in `config.py`) take absolute precedence. LLM agents generate explanatory reasoning and dialogue, but NEVER override hard physical constraints. If cell temperature exceeds critical thresholds ($T_{cell} \ge 50.0^\circ\text{C}$), charging is unconditionally aborted (`0.0C`).
2. **Strict SOH/RUL Non-Fabrication**: Member 4 does NOT independently predict SOH, RUL, capacity, or range. If Member 2 ML predictions are provided, their values and provenance are recorded. If absent, SOH and RUL remain explicitly `None` with status `"UNAVAILABLE"`. Multi-year degradation curves are strictly isolated as comparative hypothetical simulations with disclaimers.
3. **Complete Negotiation Transcripts**: Every negotiation step logs `driver_message`, `bms_message`, `requested_action`, `applied_action`, `decision`, `reason`, `rules_triggered`, and `negotiation_round`.
4. **Member 5 Handover-Ready**: Root-level package exports, zero broken relative imports, and a stable versioned schema (`2.0.0`).

---

## Quickstart

### 1. Setup Environment
```bash
# In ev_member4 directory
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API Key (Optional)
Gemini LLM dialogue is optional. When `GOOGLE_API_KEY` is present in `.env`, the agents generate natural-language dialogue. When omitted or `use_llm=False`, the system uses high-performance deterministic templated dialogue.
```powershell
# In PowerShell:
$env:GOOGLE_API_KEY="your-api-key"
# Or copy .env.example to .env
Copy-Item .env.example .env
```

### 3. Run Test Suite (15 Unit Tests)
```bash
python -m unittest discover -s tests -v
```
All 15 tests verify:
- Missing SOH/RUL non-fabrication (`UNAVAILABLE`).
- Member 2 prediction ingestion.
- Catastrophic thermal shutdown ($0.0\text{C}$ cutoff at $\ge 50^\circ\text{C}$).
- Extreme heat fast charge lockout ($T_{cell} > 35^\circ\text{C}$).
- SOC and C-rate ceilings.
- Full dialogue retention in history records.
- 2-round counter-proposal negotiation loop.
- Package importability from root/external directories.
- Simulation reproducibility.

### 4. Run Batch Simulations
```bash
python run_simulation.py hot                            # Hot climate (20 runs: 4 chemistries x 5 strategies)
python run_simulation.py temperate                      # Temperate climate
python run_simulation.py cold                           # Cold climate
python run_simulation.py all                            # All 60 simulations
```
Outputs are exported to `results/<climate>/` (`results.json`, `summary.csv`, `soh_curves.png`). A canonical copy is stored in `sample_results/`.

---

## Member 5 (Dashboard Lead) Integration

Member 5 can import and use Member 4 either from the repository root or as an installed package.

### Endpoint: `simulate_bms_negotiation()`
The primary interactive endpoint for Streamlit sliders, interactive negotiation widgets, and real-time advisory:

```python
import sys
# If running from repo root:
sys.path.insert(0, ".")

import ev_member4

# 1. Single interactive negotiation turn (e.g. user requests 3.0C fast charge at 85% SOC, 42°C pack temp)
result = ev_member4.simulate_bms_negotiation(
    battery_state={
        "soc": 85.0,
        "temp_c": 42.0,
        "soh": 92.5  # Optional Member 2 SOH, or None if unavailable
    },
    requested_action={
        "action": "charge_fast",
        "charge_rate_c": 3.0,
        "target_soc": 100.0
    },
    member2_predictions={  # Optional Member 2 predictions
        "soh_pct": 92.5,
        "rul_cycles": 1450,
        "model_version": "v1.0-lgbm",
        "provenance": "Member 2 LightGBM model"
    },
    bms_limits_override={  # Optional dynamic custom BMS thresholds
        "max_cell_temp_c": 45.0,
        "critical_shutdown_temp_c": 50.0
    },
    use_llm=False  # True to use Gemini LLM, False for instant deterministic templates
)

# 2. Inspect negotiation dialogue and BMS decision
print("Driver Demand:", result["driver_proposal"]["message"])
print("BMS Round 1:", result["bms_evaluation"]["message"])
print("Driver Counter/Concession:", result["driver_counter"]["message"])
print("Final BMS Decision:", result["final_applied_action"]["decision"])
print("Applied C-Rate:", result["final_applied_action"]["charge_rate_c"])
print("Rules Triggered:", result["final_applied_action"]["rules_triggered"])
print("Provenance:", result["provenance"])
```

### Versioned Output Schema (`2.0.0`)
```json
{
  "schema_version": "2.0.0",
  "status": "success",
  "battery_state": { "soc": 85.0, "temp_c": 42.0, "soh": 92.5 },
  "member2_predictions": {
    "soh_pct": 92.5,
    "soh_status": "AVAILABLE",
    "rul_cycles": 1450,
    "rul_status": "AVAILABLE",
    "provenance": "Member 2 LightGBM model"
  },
  "driver_proposal": {
    "round": 1,
    "action": "charge_fast",
    "charge_rate_c": 3.0,
    "target_soc": 100.0,
    "message": "Need maximum charging power (3.0C) right now to reach 100% SOC ASAP."
  },
  "bms_evaluation": {
    "round": 1,
    "decision": "MODIFIED",
    "safe_charge_rate_c": 1.0,
    "safe_target_soc": 80.0,
    "rules_triggered": ["RULE_CELL_OVERHEAT_DERATE", "RULE_SOC_MAX_CEILING"],
    "message": "[BMS Advisory] Fast charge restricted: cell temperature is high (42.0°C). Rate derated."
  },
  "driver_counter": {
    "round": 2,
    "action": "charge_fast",
    "charge_rate_c": 1.5,
    "target_soc": 80.0,
    "message": "Accepted SOC cap of 80.0%, but requesting higher charging speed of 1.50C if safe."
  },
  "final_applied_action": {
    "round": 2,
    "action": "charge_fast",
    "charge_rate_c": 1.0,
    "target_soc": 80.0,
    "decision": "MODIFIED",
    "rules_triggered": ["RULE_CELL_OVERHEAT_DERATE", "RULE_SOC_MAX_CEILING"],
    "reason": "Final safety gate clamped action within verified thermal envelope."
  },
  "transcript": [
    { "round": 1, "speaker": "Driver", "text": "..." },
    { "round": 1, "speaker": "BMS", "text": "..." },
    { "round": 2, "speaker": "Driver", "text": "..." },
    { "round": 2, "speaker": "BMS", "text": "..." }
  ],
  "provenance": {
    "is_hypothetical_simulation": false,
    "soh_prediction_source": "Member 2 LightGBM model",
    "safety_rules_version": "2.0.0"
  }
}
```

---

## Configurable Safety Rules (`RULES_CATALOG`)

Deterministic safety rules enforced by `bms.py`:

| Rule ID | Parameter | Default Limit | Action |
|---|---|---|---|
| `RULE_CATASTROPHIC_SHUTDOWN` | `critical_shutdown_temp_c` | $\ge 50.0^\circ\text{C}$ | Cutoff charging immediately ($0.0\text{C}$). Critical thermal runaway mitigation. |
| `RULE_THERMAL_LOCKOUT` | `max_temp_for_fast_charge_c` | $> 35.0^\circ\text{C}$ | Lock out high C-rate fast charging. Fall back to standard rate $\le 1.0\text{C}$. |
| `RULE_CELL_OVERHEAT_DERATE` | `max_cell_temp_c` | $> 45.0^\circ\text{C}$ | Derate charging/discharge current to minimize internal resistive heating. |
| `RULE_SOC_MAX_CEILING` | `max_soc_pct` | $> 80.0\%$ | Cap high-voltage cutoff to mitigate cathode lattice stress and SEI growth. |
| `RULE_SOC_MIN_FLOOR` | `min_soc_pct` | $< 20.0\%$ | Protect against copper dissolution and deep-discharge under-voltage. |
| `RULE_CRATE_CEILING` | `max_charge_c_rate` | $> 1.5\text{C}$ | Cap charging C-rate to prevent lithium plating on anode surfaces. |

*Disclaimer: Thermal models use empirical lumped-parameter heat dissipation estimations and do not replace certified automotive UN ECE R100 or ISO 26262 ASIL-D functional safety hardware.*

---

## File Structure

```
ev_member4/
├── __init__.py               # Package exports (simulate_bms_negotiation, run_simulation, etc.)
├── config.py                 # RULES_CATALOG, BMS thresholds, climates, chemistries, vehicle profiles
├── state.py                  # SimState TypedDict with provenance and multi-round negotiation schema
├── bms.py                    # Deterministic BMS safety supervisor and rule evaluations
├── agents.py                 # Driver Persona & BMS Supervisor agents (LLM & deterministic templates)
├── graph.py                  # LangGraph 7-node multi-turn negotiation execution loop
├── placeholders.py           # Range engine connector & hypothetical comparative degradation models
├── run_simulation.py         # Batch runner, Member 5 interactive API, CLI & export tools
├── tests/
│   └── test_member4.py       # 15 comprehensive unit tests (100% passing)
├── vendor/member3/           # Vendored Member 3 range calculation engine & vehicle configs
├── results/                  # Generated simulation outputs (hot, temperate, cold)
└── sample_results/           # Verified canonical test and simulation benchmarks
```
