#!/usr/bin/env python3
"""
Capture RSSI samples from scanner_app over UART and append labeled rows to
dataset/rssi_samples.csv.

The DK prints lines like:
    CSV,<uptime_ms>,<mac_address>,<rssi>

This script keeps only lines whose MAC matches --mac (or --mac-prefix), tags
each surviving row with the distance bucket and session id you pass in, and
appends them to the dataset CSV.

Typical session:
    python log_to_csv.py --list-devices --port COM5 --seconds 10
    # ^ find out which MAC is your beacon

    python log_to_csv.py --port COM5 --mac AA:BB:CC:DD:EE:FF \\
        --bucket near --session 1 --seconds 45

Repeat with --bucket {near,mid,far} at each physical distance, several
sessions each (different room/orientation), per the project plan.
"""
import argparse
import csv
import datetime
import re
import sys
import time
from pathlib import Path

import serial

CSV_LINE_RE = re.compile(r"^CSV,(-?\d+),([0-9A-Fa-f:]{17}(?: \([a-z]+\))?),(-?\d+)\s*$")

DATASET_HEADER = [
    "timestamp_ms",
    "mac",
    "rssi",
    "distance_bucket",
    "session_id",
    "collected_at",
]

VALID_BUCKETS = ("near", "mid", "far")


def parse_line(line):
    """Return (uptime_ms, mac, rssi) or None if the line isn't a CSV row."""
    m = CSV_LINE_RE.match(line.strip())
    if not m:
        return None
    uptime_ms, mac_raw, rssi = m.groups()
    mac = mac_raw.split(" ")[0].upper()  # strip trailing " (random)"/"(public)"
    return int(uptime_ms), mac, int(rssi)


def list_devices(ser, seconds):
    print(f"Listening for {seconds}s to find nearby BLE MAC addresses...")
    seen = {}
    deadline = time.time() + seconds
    while time.time() < deadline:
        raw = ser.readline().decode(errors="replace")
        if not raw:
            continue
        parsed = parse_line(raw)
        if parsed is None:
            continue
        _, mac, rssi = parsed
        count, last_rssi = seen.get(mac, (0, rssi))
        seen[mac] = (count + 1, rssi)
    if not seen:
        print("No advertisements seen. Is the beacon device advertising nearby?")
        return
    print(f"{'MAC':<20}{'packets':<10}{'last RSSI'}")
    for mac, (count, rssi) in sorted(seen.items(), key=lambda kv: -kv[1][0]):
        print(f"{mac:<20}{count:<10}{rssi}")


def collect(ser, out_path, mac_filter, mac_prefix, bucket, session_id, seconds):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out_path.exists() or out_path.stat().st_size == 0

    with out_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(DATASET_HEADER)

        print(f"Logging bucket={bucket} session={session_id} -> {out_path}")
        print("Press Ctrl+C to stop early." if seconds is None else f"Collecting for {seconds}s.")

        deadline = None if seconds is None else time.time() + seconds
        n_written = 0
        try:
            while deadline is None or time.time() < deadline:
                raw = ser.readline().decode(errors="replace")
                if not raw:
                    continue
                parsed = parse_line(raw)
                if parsed is None:
                    continue
                uptime_ms, mac, rssi = parsed

                if mac_filter and mac != mac_filter:
                    continue
                if mac_prefix and not mac.startswith(mac_prefix):
                    continue

                collected_at = datetime.datetime.now().isoformat(timespec="seconds")
                writer.writerow([uptime_ms, mac, rssi, bucket, session_id, collected_at])
                n_written += 1
                if n_written % 20 == 0:
                    f.flush()
                    print(f"  {n_written} samples logged (last RSSI {rssi})")
        except KeyboardInterrupt:
            pass
        f.flush()
        print(f"Done. {n_written} samples appended.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="Serial port, e.g. COM5")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--mac", help="Exact beacon MAC to keep, e.g. AA:BB:CC:DD:EE:FF")
    ap.add_argument("--mac-prefix", help="Keep MACs starting with this prefix instead of an exact match")
    ap.add_argument("--bucket", choices=VALID_BUCKETS, help="Distance label for this session")
    ap.add_argument("--session", help="Session id/tag, e.g. room1_run2")
    ap.add_argument("--seconds", type=float, default=None, help="Auto-stop after N seconds (default: run until Ctrl+C)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "dataset" / "rssi_samples.csv"))
    ap.add_argument("--list-devices", action="store_true", help="Just print MACs seen, then exit (no logging)")
    args = ap.parse_args()

    if not args.list_devices:
        missing = [n for n in ("bucket", "session") if getattr(args, n) is None]
        if missing:
            ap.error(f"--{missing[0]} is required unless --list-devices is given")
        if not args.mac and not args.mac_prefix:
            ap.error("--mac or --mac-prefix is required unless --list-devices is given")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as e:
        print(f"Could not open {args.port}: {e}", file=sys.stderr)
        sys.exit(1)

    # Drain any stale buffered boot log so we don't misparse it.
    time.sleep(0.2)
    ser.reset_input_buffer()

    if args.list_devices:
        list_devices(ser, args.seconds or 10)
    else:
        collect(
            ser,
            Path(args.out),
            mac_filter=args.mac.upper() if args.mac else None,
            mac_prefix=args.mac_prefix.upper() if args.mac_prefix else None,
            bucket=args.bucket,
            session_id=args.session,
            seconds=args.seconds,
        )

    ser.close()


if __name__ == "__main__":
    main()
