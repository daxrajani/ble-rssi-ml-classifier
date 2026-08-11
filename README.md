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

End-to-end pipeline works on real hardware as of 2026-08-11: flashed to a
real nRF52840 DK, 850 labeled samples collected from a phone beacon,
tuned via a feature/architecture sweep (`model/experiment.py`) to 96.2%
held-out accuracy (single-session split - see caveat below), beating a
hand-tuned-threshold baseline by +20.7 points, and running live on-device
with 61-91us inference latency. See `results/accuracy_report.md` for full
numbers and the sweep results.

**What's solid:** firmware builds and flashes cleanly (`scanner_app`:
83KB flash/22KB RAM, `inference_app`: 166KB flash/29KB RAM), data
collection pipeline works (including handling phone BLE MAC rotation),
training + INT8 weight extraction is scripted (`model/train.py` +
`model/extract_weights.py`), feature/architecture choices are backed by a
sweep rather than guesses (`model/experiment.py`), the trained model runs
on-device with a hand-rolled INT8 forward pass (no TFLite Micro
dependency - see `firmware/inference_app/src/inference.c`), and each
nearby BLE device is tracked in its own RSSI window
(`firmware/inference_app/src/scanner.c`) rather than one shared window
that used to blend samples across unrelated devices.

**What's still needed before quoting a real accuracy number:** all 850
samples came from one session in one room. Per the original data
collection checklist (`dataset/README.md`), collect several more sessions
across different rooms/orientations, then:
```bash
cd data_collection
python log_to_csv.py --list-devices --port COM3   # find current beacon MAC (it rotates)
python log_to_csv.py --port COM3 --mac <MAC> --bucket near --session room2_run1 --seconds 45
# ...repeat for mid/far, multiple rooms

cd ../model
python train.py             # now uses a proper session-held-out split automatically
python extract_weights.py   # regenerates firmware/inference_app/src/model_weights.c
```
Then rebuild+reflash `inference_app` and update
`results/accuracy_report.md` with the new numbers. Also still open:
record `results/demo.gif`.

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
