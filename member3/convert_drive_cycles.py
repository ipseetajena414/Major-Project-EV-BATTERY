import csv
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "drive_cycles" / "source"
OUTPUT = BASE / "drive_cycles"

CYCLES = {
    "UDDS.txt": "udds_city.csv",
    "HWFET.txt": "hwfet_highway.csv",
}

def convert(source_name, output_name):
    source_path = SOURCE / source_name
    output_path = OUTPUT / output_name
    records = []

    with source_path.open("r", encoding="utf-8-sig") as file:
        lines = file.readlines()

    for line_number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        # Skip the two descriptive header lines.
        if len(parts) != 2:
            continue

        try:
            time_s = float(parts[0])
            speed_mph = float(parts[1])
        except ValueError:
            continue

        if not math.isfinite(time_s) or not math.isfinite(speed_mph):
            raise ValueError(f"{source_name}:{line_number}: non-finite value")
        if time_s < 0 or speed_mph < 0:
            raise ValueError(f"{source_name}:{line_number}: negative time or speed")

        records.append((time_s, speed_mph * 0.44704, 0.0))

    if len(records) < 2:
        raise ValueError(f"{source_name}: fewer than two valid data rows")

    times = [row[0] for row in records]
    if any(current <= previous for previous, current in zip(times, times[1:])):
        raise ValueError(f"{source_name}: timestamps are not strictly increasing")

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s", "speed_mps", "grade_pct"])
        writer.writerows(
            (f"{t:g}", f"{v:.6f}", f"{grade:g}")
            for t, v, grade in records
        )

    distance_km = sum(
        (records[i][1] + records[i - 1][1]) / 2
        * (records[i][0] - records[i - 1][0]) / 1000
        for i in range(1, len(records))
    )

    print(f"Created: {output_path}")
    print(f"Data rows: {len(records)}")
    print(f"Duration: {records[-1][0] - records[0][0]:.1f} seconds")
    print(f"Distance from speed trace: {distance_km:.3f} km")
    print("Assumption: grade_pct = 0 (flat road; source contains no grade data)")
    print()

for source_name, output_name in CYCLES.items():
    convert(source_name, output_name)
