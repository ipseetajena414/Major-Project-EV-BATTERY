"""
Member 1 - post-process sampled LFP telemetry with physical plausibility flags.

The raw archives contain a small number of transient current/voltage spikes.
They are kept (not deleted) and flagged so downstream filters are explicit.

Run:
    python AUDIT/member1_postprocess_telemetry.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

AUDIT_DIR = Path(__file__).resolve().parent
OUT = AUDIT_DIR / "candidates"
PROTOCOL_MAX_CURRENT = {"B1": 0.625, "B2": 1.25, "B3": 2.5, "B4": 3.75, "B5": 1.25, "B6": 2.4}


def append_flag(flags: pd.Series, token: str, mask: pd.Series) -> pd.Series:
    out = flags.fillna("").astype(str)
    out = out.where(~mask, out + ";" + token)
    return out.str.strip(";")


def main() -> int:
    report = {}
    for tag, max_a in PROTOCOL_MAX_CURRENT.items():
        path = OUT / f"telemetry_{tag}.parquet"
        df = pd.read_parquet(path)
        current_spike = df["current_a"].abs() > 1.25 * max_a
        voltage_spike = (df["voltage_v"] < 1.8) | (df["voltage_v"] > 4.0)
        df["quality_flag"] = append_flag(df["quality_flag"], "CURRENT_OUT_OF_PROTOCOL_BOUNDS", current_spike)
        df["quality_flag"] = append_flag(df["quality_flag"], "VOLTAGE_OUT_OF_PROTOCOL_BOUNDS", voltage_spike)
        df.to_parquet(path, index=False)
        summary_path = OUT / f"telemetry_{tag}_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["quality_flag_counts"] = df["quality_flag"].value_counts().head(20).to_dict()
        summary["current_spike_rows"] = int(current_spike.sum())
        summary["voltage_spike_rows"] = int(voltage_spike.sum())
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        report[tag] = {
            "rows": int(len(df)),
            "current_spike_rows": int(current_spike.sum()),
            "voltage_spike_rows": int(voltage_spike.sum()),
            "current_spike_cycles": int(df.loc[current_spike, "cycle_id"].nunique()),
        }
        print(tag, report[tag])
    (OUT / "telemetry_postprocess_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
