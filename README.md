# ble-rssi-ml-classifier

On-device ML classifier that buckets a BLE device's distance (Near <1m,
Mid 1-3m, Far >3m) from RSSI, running entirely on an nRF52840 DK (Zephyr /
nRF Connect SDK). No extra hardware — just the DK's onboard radio plus a
phone or second BLE device as the moving beacon.

## Why

RSSI is a noisy proxy for distance (multipath, orientation, body
blocking). This project asks whether a small learned model handles that
noise better than hand-tuned filtering — the same problem I worked on with
hand-tuned RSSI filtering on Dialog's WiRa BLE positioning platform at
Glide, tackled here from scratch and end-to-end on my own hardware.

## Repo layout

```
data_collection/
  scanner_app/     Zephyr BLE observer firmware - streams RSSI+MAC+timestamp over UART
  log_to_csv.py     Host-side script: labels the UART stream, appends to dataset/
dataset/
  rssi_samples.csv  Collected + labeled data (near/mid/far)
model/
  training_notebook.ipynb  Feature engineering + (optional) local TFLite Micro training
firmware/
  inference_app/    Scanner + on-device classifier + LEDs + BLE result notify
results/
  accuracy_report.md  Measured accuracy / latency / flash+RAM footprint
```

## Status

Scaffolding and firmware/scripts are written; `dataset/rssi_samples.csv` is
still empty (header only). Next steps are physical, done by you:

1. **Flash `data_collection/scanner_app`** to the DK (`west build -b
   nrf52840dk/nrf52840 . -d build && west flash -d build` from that
   folder). Requires the nRF Connect SDK west workspace you already have at
   `D:\` — no separate toolchain setup needed.
2. **Collect data** — walk the beacon (phone, or a second DK/laptop
   Bluetooth) to 0.5/1/2/3/5m, hold ~30-60s each, multiple rooms/sessions.
   Use `data_collection/log_to_csv.py --list-devices` first to find the
   beacon's MAC, then `log_to_csv.py --mac <MAC> --bucket near --session
   <id>` etc. See `dataset/README.md` for the full checklist.
   - Note: phones typically rotate their BLE MAC address periodically for
     privacy, which can fragment a long session. If that's a problem in
     practice, use a second DK (or laptop) running a fixed-address
     advertiser as the beacon instead — more reliable for MAC-based
     filtering.
3. **Train** — upload `dataset/rssi_samples.csv` to Edge Impulse (primary
   path) or run `model/training_notebook.ipynb` locally (alternative,
   more control). See `model/README.md`.
4. **Integrate** — wire the exported model into
   `firmware/inference_app/src/inference.c` (TODO comment marks the exact
   spot), flash, and watch the LEDs / `RESULT,...` UART lines / BLE notify
   characteristic track the beacon's distance live.
5. **Measure** — fill in `results/accuracy_report.md` with real numbers
   (accuracy, inference latency, flash/RAM) and record `results/demo.gif`.

## Resume bullet (template — fill in X/Y from `results/accuracy_report.md`)

> Built an on-device ML classifier for BLE RSSI-based distance estimation
> on nRF52840 (Zephyr NCS), collecting and labeling a multi-environment
> dataset, training and INT8-quantizing a model via Edge Impulse, and
> achieving [X]% classification accuracy with sub-[Y]ms on-chip inference.

## Interview talking points

- At Glide, I worked on WiRa, a BLE RSSI-based indoor positioning system,
  using hand-tuned filtering to improve accuracy. This project explores
  whether a learned model handles the same noisy RSSI signal better than
  manual filters, built independently on my own nRF52840 DK with Edge
  Impulse / TFLite Micro.
- Why RSSI is noisy: multipath, device orientation, body blocking.
- Why bucket classification over regression: coarser but more robust
  target given single-antenna RSSI's poor distance resolution.
- Quantization tradeoffs: INT8 for flash/RAM/latency on a Cortex-M4,
  accuracy cost measured (not assumed) in `results/accuracy_report.md`.
- What I'd do with more compute: IMU fusion, trend-based movement
  detection (this repo's stretch-goal "Option B").

## License

MIT — see `LICENSE`.
