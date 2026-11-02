# Member 3 compatibility notes

Existing adapter: `member3/integration/member2_range_adapter.py`.

It requires `cycle_index`, `cell_id`, `chemistry` and `soh_pct`.
The active Member 1 datasets are snapshots and contain no cycle history,
so no `cycle_index` exists. This bundle therefore does not claim adapter
compatibility and does not fabricate a cycle index.

Bundle validation status: OK

Production readiness: false. Snapshot models are grouped-holdout
experimental models, and the range file is a formula estimate.
