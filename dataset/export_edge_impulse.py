#!/usr/bin/env python3
"""
Reshapes rssi_samples.csv into per-bucket time-series CSVs for Edge
Impulse's plain "Upload data" flow, which expects one file = one
continuous recording = one label (not one CSV with multiple labels mixed
together, which is what caused the "found 850" parse error).

Writes near.csv, mid.csv, far.csv into dataset/edge_impulse_export/, each
with a timestamp column (ms, rebased to start at 0 per file) and rssi -
Edge Impulse treats each as a single labeled recording and lets its DSP
block do the windowing (comparable to our own window=8 rolling mean/std).

Usage:
    python export_edge_impulse.py

Then in Edge Impulse: Data acquisition -> Upload data -> select all three
files -> for each, set "Label" to match the filename (near/mid/far) -> it
should auto-detect the timestamp column. If it still complains, use the
"CSV Wizard" link Edge Impulse shows in the error message instead - it
lets you manually map which column is the timestamp and which is the
label/value.
"""
import csv
from pathlib import Path

IN_PATH = Path(__file__).parent / "rssi_samples.csv"
OUT_DIR = Path(__file__).parent / "edge_impulse_export"


def main():
    rows_by_bucket = {"near": [], "mid": [], "far": []}

    with open(IN_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bucket = row["distance_bucket"]
            if bucket in rows_by_bucket:
                rows_by_bucket[bucket].append(row)

    OUT_DIR.mkdir(exist_ok=True)

    for bucket, rows in rows_by_bucket.items():
        if not rows:
            continue
        rows.sort(key=lambda r: int(r["timestamp_ms"]))
        t0 = int(rows[0]["timestamp_ms"])

        out_path = OUT_DIR / f"{bucket}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "rssi"])
            for r in rows:
                rel_ts = int(r["timestamp_ms"]) - t0
                writer.writerow([rel_ts, r["rssi"]])
        print(f"Wrote {out_path} ({len(rows)} rows, label={bucket})")


if __name__ == "__main__":
    main()
