"""
Member 2 — Persistence-Anchored Delta Model (PADM)

This package implements the PADM architecture recommended by the Member 2
ML Architecture Review (November 2026) to replace the deprecated hybrid model
for NMC next-cycle SOH forecasting.

Architecture decision rationale: archived in
  AUDIT/MEMBER2_ML_ARCHITECTURE_REVIEW.md (see sections D, F, H, I)
  AUDIT/PADM_PRE_IMPLEMENTATION_REVIEW.md (full design specification)

Modules
-------
delta_features   — build delta-target pairs and lag features from master_table
padm_train       — fit Ridge/ElasticNet on training split; LOCO diagnostic
padm_evaluate    — evaluate PADM vs persistence and mean-fade on test split
padm_export      — write Member 3 aligned CSVs and PROVENANCE.json
"""

__all__: list[str] = []
