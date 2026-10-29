# Member 1 — Data Readiness, Preprocessing Validation & Research Formula Handoff

**Project:** Autonomous Multi-Agent EV Battery Life & Range Predictor
**Role:** Member 1 — data ingestion, data-quality validation, preprocessing readiness, provenance, cleaned-data contract
**Audit date:** 2026-10-29
**Canonical artifact audited:** `master_table.parquet` (SHA-256 `6fa7475d…73b2bb`, unchanged)
**Status:** Read-only audit complete; canonical data **not modified**; nothing staged or committed.

> Scope rule applied throughout: every prior finding was re-derived from the actual files. Anything that could not be reproduced from a verified local source is labelled **unverified** rather than trusted.
>
> **Follow-up (2026-10-29):** `AUDIT/MEMBER1_LFP_B1_B5_DATA_READINESS_REPORT.md` extends this handoff for LFP B1–B5. It confirms the findings below and adds that B1's single-cycle master coverage is an ingestion omission (the source holds 1,460 cycle sheets / 1,456 labels), that B4's blanket `NOT_CALCULATED` block is stricter than the raw data (complete 3C discharges are integrable), that B3 is the priority independent-validation candidate, and that B2 is suitable for cycle-level independent validation today. No conclusion below is retracted.

---

## 1. Executive summary

The canonical master table is structurally sound — 515,270 rows × 13 columns, exact schema, 0 duplicate composite keys, 0 time reversals, 0 out-of-bounds voltage/current/temperature — but it is **not ready to be used naively as a feature table**. Five issues materially affect Member 2:

1. **`soh_pct` is `capacity_ah / manufacturer_rated_Ah × 100`, not retention vs measured beginning-of-life.** Fresh cells therefore start at 79 % (Na-ion), 92–101 % (NASA), 95.6 % (B5). Treating it as "retention" or "fade from BOL" is wrong.
2. **B1 telemetry is labelled with the wrong cycle's capacity.** The builder collapses a `checkpoint → capacity` dict over a table where each checkpoint spans many cycles, so sampled telemetry (cycle 2) carries the last cycle's label (1.202223 Ah instead of 1.220155 Ah).
3. **B5 cycle 40 is an interrupted discharge.** Its partial trapezoid throughput (0.5428 Ah) was kept as `capacity_ah` while its `soh_pct` was correctly cleared — a partial label, not a missing cycle. Only this cycle is affected.
4. **Cycle-level labels are broadcast over every telemetry row** (median 333, max 4,789 repetitions), so row-wise training leaks the target.
5. **`time_sec` has inconsistent semantics.** It resets per cycle only for NASA; for B5, Na-ion and B1 it is cumulative source/workbook time. "Elapsed within cycle" in the README is inaccurate for three of four datasets.

Derivable features: voltage/current summaries, cycle duration, Ah/Wh throughput and C-rate are computable where current units are verified (NASA and Na-ion in amperes; LFP numerically amperes but header-underdetermined). NASA is the **only** source with measured temperature, so thermal features are NASA-only. SOC/DoD are blocked everywhere except Na-ion protocol constants; fast-charge detection is blocked. No research formula in the register is currently implementable as a validated ground-truth generator; only the EFC throughput adaptation (C-Eq10-12) and Member-3 scenario constants are defensible.

Deliverables produced (all under `AUDIT/`): this report, `MEMBER1_FORMULA_VALIDATION_MATRIX.csv` (16 formula rows), `MEMBER1_FEATURE_READINESS_MATRIX.csv` (30 feature rows), reproducible read-only `member1_audit.py`, `test_member1_audit.py` (18 passing tests), `make_matrices.py`, and generated outputs in `AUDIT/member1_audit_outputs/`.

---

## 2. Repository state (Phase 0)

| Check | Result |
|---|---|
| `git branch --show-current` | `main` |
| `git status --short` | clean (no pre-existing edits) |
| `git log -5 --oneline` | `e36eb73` Add Member 1 ingestion pipeline… ; `18fb971` first commit |
| `git diff --stat` / `git diff --cached --stat` | empty |
| HEAD | `e36eb7312aca7c25e0ed7b1e51faa6fd1755fea4` (2026-10-27) |
| Remote | `origin` = `https://github.com/ipseetajena414/Major-Project-EV-BATTERY.git` |
| Tracking | `main [origin/main]`, in sync with `origin/main` at `e36eb73` |

**Discrepancy vs the brief:** work is on `main`, not a dedicated "Member 1" branch. There is no other local or remote branch. No branch was created or switched. No fetch/reset/stash/clean was performed.

---

## 3. Actual data / source inventory (Phase 1)

Presence and parse status were verified directly; a dataset named in a script is not assumed present.

| Dataset | Chemistry | Raw source | Size (bytes) | Audit bundle | Master coverage | Verification status |
|---|---|---|---|---|---|---|
| NASA Ames PCoE | NMC | `DATA/archive (4).zip` — 34 `.mat` cells | 198,551,008 | inline (scipy) | **3 of 34 cells** (B0005, B0006, B0018); 468 cell-cycles; discharge-phase | Raw **verified & parsed** (partial cell coverage) |
| Zenodo / IonTech Na-ion | Na-ion | `DATA/NA-ion.zip` — 64 `.pkl` cells | 289,169,561 | inline (pickle) | **4 of 64 cells**, **25 of 91 cycles each** → 100 cell-cycles | Raw **verified & parsed** (partial); temperature unmeasured |
| LFP MultiStat B1 | LFP | `DATA/B1_Discharge0.5C_Charge0.5C(1).zip` | 482,913,032 | `AUDIT/B1_LFP_Audit_Output.zip` | **1 of 1,460** audited cycles (single sampled checkpoint, mixed-phase) | Raw **verified**; master coverage minimal + label bug (§5.1) |
| LFP MultiStat B2 | LFP | **missing** (no XLSX) | — | `DATA/B2_LFP_Member1_audit.zip` (derived exports) | not in master | Raw **not in repo**; derived `cycle_features.parquet` / `qv_curves.parquet` present |
| LFP MultiStat B3 | LFP | `DATA/B3_Discharge2C_Charge0.5C (1).zip` | 392,547,570 | `AUDIT/Existing_Audits_Available_B3_B4 (1).zip` | excluded | Raw **verified**, not ingested |
| LFP MultiStat B4 | LFP | `DATA/B4_Discharge3C_Charge0.5C (1).zip` | 290,588,332 | same B3/B4 bundle | excluded (blocked) | Raw **verified**; capacity/SOH **blocked** (`A/cm2` vs 3C conflict) |
| LFP MultiStat B5 | LFP | `DATA/B5_Discharge0.5C_Charge1C (1).zip` | 355,306,217 | `AUDIT/B5.zip` (`B5_Cycle_Summary.csv`, 1,499 cycles) | **250 of 1,499** audited cycles, discharge-only | Raw **verified & parsed** (partial); capacity **conditional** |
| LFP MultiStat B6 | LFP | **missing** | — | `AUDIT/LFP_B6_Audit_Package (1).zip` (external summaries) | not in master | Raw **not in repo**; audit-only |

Supporting docs present: `README.md`, `MEMBER1_STATUS_REPORT.md`, `MEMBER1_FINAL_VALIDATION_REPORT.md`, `FORMULA/paper_formula_register (1).csv`, `BLUEPRINT OF ENTIRE PROJ/EV_Battery_Major_Project_Blueprint (2).pdf`.

**Inventory gaps:** `FORMULA/paper_access_status.csv` is **absent**. The register exists only as `paper_formula_register (1).csv` (the exact name `paper_formula_register.csv` is absent). No reference PDFs or source spreadsheets for the papers are in the workspace. The blueprint PDF could not be text-extracted in this environment (no poppler/PDF library), so blueprint facts were taken from `README.md`; this is stated rather than assumed.

---

## 4. Verified schema and counts (Phase 2)

| Metric | Verified value |
|---|---|
| Shape | 515,270 × 13 |
| Schema | `chemistry, dataset_id, cell_id, protocol_id, cycle_id, time_sec, voltage_v, current_a, temp_c, capacity_ah, soh_pct, quality_flag, source_file` (exact) |
| Parquet SHA-256 | `6fa7475d1810007803a0f55431730fdedbb2eec7127cecddf3219a4d9073b2bb` (matches prior report) |
| CSV SHA-256 | `9e23792615b2f08692944680d041957f776428b7563a90b214b77ee74629895b` (matches prior report) |
| Duplicate composite keys | 0 |
| Negative / zero Δt, duplicate timestamps within cycle | 0 / 0 / 0 |
| Voltage range | 1.9991 – 4.2229 V |
| Current range | −5.8004 – +5.8005 A |
| Temperature (NASA only) | 22.35 – 42.01 °C |
| `temp_c` nulls | 379,834 (B1 4,789 + B5 333,344 + Na-ion 41,701) |
| `soh_pct` nulls | 625 (B5 cycle 40 only) |

Per-dataset verification:

| Dataset | Rows | Cells | `cycle_id` values | Cell-cycle pairs | Notes |
|---|---|---|---|---|---|
| LFP_MultiStat_B1 | 4,789 | 1 | {2} | 1 | single sampled cycle; mixed charge/discharge |
| LFP_MultiStat_B5 | 333,344 | 1 | blocks 1–100, 151–200, 401–500 (250) | 250 | missing 101–150 and 201–400; discharge-only |
| NASA_Ames_PCoE | 135,436 | 3 | 1–168 / 1–168 / 1–132 | 468 | temperature measured; discharge-phase |
| Zenodo_IonTech_NaIon | 41,701 | 4 | 1–25 each | 100 | temperature unmeasured; mixed charge/discharge |

Cycle-ID blocks and gaps were reproduced exactly: B5 = `[(1,100),(151,200),(401,500)]`, missing `101–150` and `201–400`. Quality-flag frequencies: `TEMP_UNMEASURED;CURRENT_UNIT_UNDERDETERMINED;CHARGE_BALANCE_DISCREPANCY` 333,301; `VALID` 135,436; `TEMP_UNMEASURED` 41,701; `TEMP_UNMEASURED;CURRENT_UNIT_UNDERDETERMINED;ok` 4,789; `…;VOLTAGE_OUT_OF_BOUNDS` 43.

Label constancy: capacity and SOH take exactly one value per `(cell, cycle)` in every dataset (verified `nunique == 1`). Implied nominal ratings are exact: 1.25 Ah (B1, B5), 2.0 Ah (NASA), 1.0 Ah (Na-ion); `soh_pct − capacity_ah/rated×100` residual < 2×10⁻⁴.

---

## 5. Data-quality issues and severity

### 5.1 HIGH — B1 telemetry is labelled with a different cycle's capacity
- Evidence: master B1 telemetry comes from `LFP_staging_sample_raw_rows.csv`, which contains only `cycle_checkpoint == 50` and `global_cycle_est == 2`. The builder's `dict(zip(cycle_level['cycle_checkpoint'], cycle_level['capacity_ah']))` collapses the 50-cycle checkpoint to its **last** row.
- Numeric proof: master `capacity_ah = 1.202223` = checkpoint-50 **last** cycle value; the telemetry cycle (est 2) is actually **1.220155 Ah (SOH 97.612 %)**.
- Impact: B1's single label is wrong-cycle and must not be used as a target. B1 telemetry also mixes charge and discharge (~49 % / 51 %).

### 5.2 HIGH — B5 cycle 40 is a partial (interrupted) discharge kept as capacity
- Source: `B5_Cycle_Summary.csv` marks cycle 40 `DISCHARGE_INTERRUPTED`, `capacity_use_status = EXCLUDE_FROM_SOH_INCOMPLETE_DISCHARGE`, and `discharge_capacity_ah_CALCULATED_CONDITIONAL_not_ground_truth = NaN`.
- Master: `capacity_ah = 0.542838` (the raw `dis_q_ah_trapz_CALCULATED`) while `soh_pct = NaN`. This is the only cycle where the raw trapezoid and usable-capacity columns differ; the builder used the raw column.
- Impact: it is a **partial label, not a missing cycle**. Exclude cycle 40 from any capacity/SOH target; neighbours are ~1.18 Ah.

### 5.3 HIGH — `soh_pct` means "fraction of rated nominal", not "retention vs BOL"
First-cycle values: Na-ion 79.0–90.7 %, NASA 92.8 % / 101.8 % / 92.8 %, B5 95.6 %, B1 96.2 %. A "fresh" cell can legitimately be < 100 % or > 100 %. Any fade/RUL work must either (a) state the nominal-rating definition, or (b) compute `capacity_k / capacity_first_observed` as a separate proxy (see feature F03).

### 5.4 HIGH — Cycle-level label leakage
`capacity_ah` and `soh_pct` are repeated on every row of a cycle (median 333, max 4,789 rows). Row-wise regression on telemetry leaks the target. **Collapse to one row per `(dataset, cell, cycle)` for targets**; keep telemetry-row aggregates as features only.

### 5.5 MEDIUM — `time_sec` semantics are inconsistent
| Dataset | Per-cycle start median | Resets each cycle? |
|---|---|---|
| NASA | 0 s | **Yes** |
| B5 | 301,729 s | No (cumulative within workbook; resets at workbook boundaries 41/51/151/401) |
| Na-ion | 17,340 s | No (cumulative experiment time) |
| B1 | 9,005 s | No (single absolute sample) |

The README's "elapsed time within cycle step" is only true for NASA. Always use `max − min` per cycle, and subtract the per-cycle minimum before any time-to-event feature.

### 5.6 MEDIUM — LFP current units underdetermined
LFP B1/B5 raw header is `I(A/cm2)` with surface area 0; the numeric magnitudes (~0.625 A) match the 0.5C-on-1.25 Ah protocol in amperes, so they are **numerically consistent with A but not unit-verified**. B4 is blocked for this reason. Treat LFP `current_a` as conditional; NASA and Na-ion currents are explicitly amperes.

### 5.7 MEDIUM — LFP/Na-ion temperature entirely unmeasured
379,834 null rows. No thermal model may be trained on LFP/Na-ion. Never impute temperature as if observed; keep null or label clearly as a model/scenario.

### 5.8 MEDIUM — B5 capacity is conditional, not ground truth
The audit marks all 250 ingested B5 cycles `capacity_ground_truth = NO` and `modelling_use_capacity_soh = CONDITIONAL_ONLY`; 18 are `USABLE_WITH_GAP_FLAG`; the charge channel is `UNSUPPORTED`. Within the ingested subset `Q_dis > Q_chg` in ~98 % of cycles (median ratio 1.013), consistent with the audit's full-campaign figure. B5 capacity/SOH may be used only as **conditional** targets.

### 5.9 MEDIUM — LFP downsampling and partial campaigns
Master holds **250 / 1,499** audited B5 cycles and **1 / 1,460** audited B1 cycles (representative telemetry samples), and only **4 / 64** Na-ion cells (**25 / 91** cycles) and **3 / 34** NASA cells. Cycle counts in the master are **not** full campaign lengths.

### 5.10 LOW — phase composition differs by source
B5 100 % discharge; NASA 97.7 % negative (the 2.3 % positive is ~14 mA offset noise); B1 and Na-ion ~50/50 charge/discharge. Always split statistics by phase.

### 5.11 LOW — quality-flag parsing
Flags are `;`-joined and include a mixed token `ok` (B1) alongside `VALID`. Parse as a set of tokens, not an exact-string equality.

Positive verifications: no impossible voltages/currents/temperatures; no duplicate timestamps; no time reversals; schema exact; hashes unchanged.

### 5.12 Proposed minimal source fixes (documented, NOT applied)

These target `master_table_build.py` and are left unapplied so the change stays reviewable; the canonical table was not rebuilt.

- **B5 (load_lfp_b5):** build the capacity map from `discharge_capacity_ah_CALCULATED_CONDITIONAL_not_ground_truth` instead of `dis_q_ah_trapz_CALCULATED`, i.e. `cap_map = dict(zip(df_cycle['global_cycle_id_nominal_checkpoint'], df_cycle['discharge_capacity_ah_CALCULATED_CONDITIONAL_not_ground_truth']))`. This makes interrupted cycles (cycle 40) NaN in **both** capacity and SOH (a fully missing label) instead of leaving a misleading partial capacity, and matches the audit's `modelling_use_capacity_soh == 'EXCLUDE'`.
- **B1 (load_lfp_b1):** join the label on `global_cycle_est` (unique per cycle) rather than `cycle_checkpoint` (repeated across a workbook), i.e. `cap_map = dict(zip(df_cycle['global_cycle_est'], df_cycle['capacity_ah']))`, and look up `row['global_cycle_est']`. Also filter the sampled telemetry to the discharge phase, as the NASA and B5 loaders already do, to avoid mixing charge and discharge.

Because these change canonical labels, any rebuild must go to a separate candidate path and be diffed against `master_table.parquet` (SHA `6fa7475d…73b2bb`) before adoption.

---

## 6. Provenance and unit limitations

- **NASA NMC:** capacity is measured discharge capacity from PCoE logs; SOH vs **2.0 Ah rated** (B0006 early cycles exceed 100 %). Temperature measured. Current in A, discharge negative.
- **Na-ion:** discharge capacity exported from `.pkl`; SOH vs **1.0 Ah rated** (cells are physically ~0.79–0.91 Ah, hence "SOH" < 91 %). Current in A, mixed charge/discharge, protocol 2.5C full 0–100 % SOC swing. `temperature_in_C = None`; `internal_resistance_in_ohm = None`; charge/discharge capacity arrays exist in the pkl but are not in the master.
- **LFP B1:** capacity from audit staging; SOH vs **1.25 Ah rated**; current unit conditional; temperature unmeasured; label-cycle mismatch (§5.1).
- **LFP B5:** capacity from trapezoidal integration of conditional current; SOH vs **1.25 Ah rated**; charge channel unsupported; temperature unmeasured; all labels conditional.

**Never** present a rated-nominal fraction, a model estimate, or a scenario value as measured telemetry.

---

## 7. Formula readiness summary (Phase 3)

Full detail: `AUDIT/MEMBER1_FORMULA_VALIDATION_MATRIX.csv` (16 rows). Citation verification: both DOIs/titles/authors/years were **confirmed via Crossref** on 2026-10-29 (`10.3390/wevj12010048`, `10.3390/app16157756`). Both MDPI full texts returned **HTTP 403**, so equations were **not re-fetched**, and the coefficient tables (Cristea Tables 4–5) were **not captured**. The originals each paper credits (refs 14–17; refs 24–30) were **not read**.

| Status | Count | Formula IDs |
|---|---|---|
| BLOCKED (missing coefficients/inputs/sign/chemistry) | 8 | P1-Eq1, P1-Eq3, C-Eq1, C-Eq2, C-Eq4, C-Eq5-9, C-Eq13-14, C-Eq16-17 |
| SCENARIO_ONLY (vehicle-level assumptions, Member 3) | 6 | P1-Eq5, P1-EoL, P1-HVAC, P1-Eff, P1-Cons, C-Eq3 |
| ADAPTABLE_NOT_EXACT (cell-level EFC only) | 1 | C-Eq10-12 |
| NOT_USABLE_AS_GROUND_TRUTH | 1 | C-Eq15 (model SoH 0–1 ≠ measured `soh_pct`) |

Key checks called out in the brief:
- **Activation-energy sign discrepancy** (P1-Eq1 `exp(−Ea/RT)` vs P1-Eq3 `exp(+Ea/RT)`): **unresolved** — cannot be verified without the full text or the credited original (Cordoba-Arenas 2015). Physically, degradation should increase with temperature, so the `+Ea` form is suspect; it is marked BLOCKED pending verification. No formula is implemented.
- **NCM-LMO parameters** are not applied to PCoE NMC/LFP/Na-ion.
- **Paper SoH vs measured SOH** kept separate (C-Eq15 marked not-usable-as-truth).
- **Vehicle → cell transfer** is treated as an explicit adaptation (EFC, EoL).
- **No formula is called "validated"** for our cells; none has verified units + coefficients + chemistry + inputs simultaneously.

---

## 8. Member 2 handoff — feature and label contract (Phases 4 & 6)

Full detail: `AUDIT/MEMBER1_FEATURE_READINESS_MATRIX.csv` (30 rows). A one-row-per-cycle candidate label table is generated at `AUDIT/member1_audit_outputs/candidate_cycle_labels.csv` (819 rows). It is a **candidate** artifact — it does not overwrite the master.

### 8.1 Label contract (targets)
- **Use one row per `(dataset_id, cell_id, cycle_id)`** — never telemetry rows as independent samples.
- `target_confidence`: NASA 468 **HIGH**, Na-ion 100 **HIGH**, B5 249 **CONDITIONAL** + 1 **INVALID** (cycle 40), B1 1 **INVALID** (label-cycle mismatch).
- Keep `soh_pct` semantics explicit (fraction of rated nominal). For fade analysis also compute `capacity_retention_vs_bol` as a proxy (F03) and say "proxy, not BOL".
- Do **not** forward/back-fill SOH or capacity. B5 cycle 40 stays missing/invalid; B5 missing blocks 101–150 and 201–400 are real gaps.

### 8.2 What is ready
- Voltage statistics and per-cycle curves (F04, F05) — use per-cycle-relative time.
- Current statistics split by phase (F06); cycle duration (F07).
- Ah/Wh throughput (F08–F11) where current units hold; C-rate (F23).
- NASA thermal features (F16–F20): mean/max T, time outside 20–30 °C, time > 35 °C, Kelvin conversion — all time-weighted, NASA only.

### 8.3 What is conditional
- `dQ/dV`, `dV/dQ`, IC peaks (F13–F15): technically feasible for NASA (A, discharge) and Na-ion (A, phase-split). For LFP only after accepting the `A/cm2`→A caveat; must segment charge/discharge, smooth, and handle dV→0. Report method and sensitivity. Not meaningful for B1 (single cycle).
- EFC as a **cell-level adaptation** (F25), not the paper's vehicle EFC.
- Cumulative discharge throughput (F26): do not forward-fill across B5 gaps.

### 8.4 What is blocked and why
- **SOC_min / SOC_max / DoD** (F21, F22): absent for NASA/LFP/B5; only Na-ion has protocol constants (`SOC_interval [0,1]`, `DoD = 1.0`). Do not infer SOC from voltage or cycle_id. Unblocked by: per-cycle SOC logs or a validated coulomb-counting setup with known initial SOC.
- **Fast-charge fraction** (F24): needs charge-phase telemetry with verified rate; B5 has none, B1 charge is 0.5C. Unblocked by: charge-phase telemetry.
- **Coulombic efficiency** (F12): charge channel unsupported for B5; NASA is noise-dominated. Only Na-ion/B1 (from pkl charge capacity) are defensible.
- **Calendar/cycle-aging formulas, paper SoH** (F27–F29): blocked (§7).
- **Internal resistance** (F30): Na-ion field exists but is `None`.

### 8.5 Leakage rules
- Never use `capacity_ah`/`soh_pct` of the **same cycle** as a feature for that cycle's target.
- If using within-cycle telemetry, aggregate to cycle level first and split by `cell_id` (grouped CV), not random row splits.
- Thermal and phase features are fine as features; they are not targets.

---

## 9. Member 3 integration note

- Distinguish **cell-level Ah** (this dataset) from **vehicle pack kWh** (not in this dataset). Do not derive pack energy from cell Ah without an explicit, stated pack configuration.
- Range requires **usable/nominal pack energy**, **consumption [Wh/km]** and explicit scenario inputs. The paper's consumption table, HVAC +15 % and charging-efficiency 95 % are **scenario assumptions, not measurements from this cell data** (P1-Cons, P1-HVAC, P1-Eff).
- "Range = usable energy / consumption" is an implied reading, not an equation printed in the paper — label it as such.
- An 80 % EoL applied to cell `soh_pct` is an **adaptation** from the paper's vehicle usable-energy definition, not an equivalence.

---

## 10. Member 4 integration note

- **Coverage:** NASA NMC 3 cells / 468 cycles with temperature; LFP B5 250 conditional discharge cycles (no temperature); Na-ion 4 cells / 25 cycles (no temperature); LFP B1 single invalid label; B3/B4/B6 not usable (excluded/blocked/missing).
- **Unknowns:** represent `temp_c` null as **unknown**, never 0. B5 cycle 40 is invalid; B5 blocks 101–150/201–400 are genuine gaps. `soh_pct` is vs rated nominal, not retention.
- **Versions:** canonical parquet SHA-256 `6fa7475d…73b2bb`, schema 13 columns as above; audit report at `AUDIT/member1_audit_outputs/audit_report.json`.
- Do **not** assert Member 2 model accuracy; refer to Member 2's own handoff and validation status.

---

## 11. Commands executed & verification results

```bash
git status --short; git branch --show-current; git log -5 --oneline; git diff --stat; git diff --cached --stat
git remote -v; git branch -vv; git rev-parse HEAD

# read-only audit (writes only AUDIT/member1_audit_outputs/)
python AUDIT/member1_audit.py          # 25/25 assertions passed, exit 0
python -m unittest AUDIT/test_member1_audit.py -v   # 18 tests, OK
python AUDIT/make_matrices.py          # 16 formula rows, 30 feature rows
```

Verification results: parquet and CSV hashes unchanged; 515,270 × 13; 0 duplicate keys; cycle blocks/gaps reproduced; B5 cycle-40 partial label reproduced; B1 label-cycle mismatch reproduced; temperature coverage reproduced; both matrices parse via pandas; all tests pass.

---

## 12. Unknowns and remaining blockers

- **Calendar time** is not in the master (only intra-cycle seconds); calendar-aging models cannot run.
- **SOC / DoD** absent except Na-ion protocol constants; SOC_min unavailable.
- **True BOL** capacities unknown (rated nominal used); retention is a proxy.
- **LFP current unit** (`A/cm2`→A) unverified; B4 blocked; B5 conditional.
- **Formula coefficients** for the Cristea models (Tables 4–5) not transcribed; full texts 403; credited originals not read; **P1 activation-energy sign unresolved**.
- **Coverage gaps:** B2 raw missing, B6 raw missing, B3/B4 excluded/blocked; LFP campaigns downsampled (250/1499 B5, 1/1460 B1); NASA 3/34 cells; Na-ion 4/64 cells, 25/91 cycles.
- **Blueprint PDF** not text-extracted (no poppler); blueprint facts taken from README.
- **`paper_access_status.csv`** absent.

---

## 13. Changed files and safety statement

New files only (all under `AUDIT/`; no canonical or user file edited):

| File | Purpose |
|---|---|
| `AUDIT/MEMBER1_DATA_READINESS_HANDOFF.md` | This report |
| `AUDIT/MEMBER1_FORMULA_VALIDATION_MATRIX.csv` | 16 formula rows |
| `AUDIT/MEMBER1_FEATURE_READINESS_MATRIX.csv` | 30 feature rows |
| `AUDIT/member1_audit.py` | Read-only audit → JSON + candidate labels |
| `AUDIT/test_member1_audit.py` | 18 explicit-expectation tests |
| `AUDIT/make_matrices.py` | Regenerates the two CSV matrices |
| `AUDIT/member1_audit_outputs/audit_report.json` | Generated audit evidence |
| `AUDIT/member1_audit_outputs/candidate_cycle_labels.csv` | Generated one-row-per-cycle candidate labels |

**No canonical data and no existing user change was overwritten.** `master_table.parquet`, `master_table.csv`, all `DATA/` archives, all existing `AUDIT/*.zip`, the existing reports, and `master_table_build.py` are byte-identical to before (parquet/CSV hashes re-verified). Nothing was staged, committed or pushed. The two B5/B1 defects are reported with a proposed minimal source fix (§5.1, §5.2) but **not** applied to the canonical builder, to keep the change reviewable.
