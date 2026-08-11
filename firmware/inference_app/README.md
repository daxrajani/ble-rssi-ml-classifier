# inference_app

Runs on the same nRF52840 DK as `scanner_app`, but adds:

- a windowed-average feature (`src/scanner.c`, `INFERENCE_WINDOW_LEN` samples)
- a classification hook (`src/inference.c`) — currently a **placeholder
  threshold heuristic**, not the trained model
- per-bucket LEDs: led0=NEAR, led1=MID, led2=FAR (`src/leds.c`)
- a BLE GATT service that notifies subscribers with the classification
  result (`src/ble_result_svc.c`)
- per-inference latency measurement, printed as `RESULT,<mac>,<rssi>,<bucket>,<us>`

## Before you flash this

1. Set `TARGET_BEACON_MAC` in `src/scanner.h` to your beacon's MAC (same
   one you used with `log_to_csv.py`), or leave it empty to classify
   whatever device is heard most recently.
2. Once you've trained a model (Edge Impulse export or
   `model/training_notebook.ipynb`), replace the body of
   `classify_distance()` in `src/inference.c` — see the TODO comment there
   for both integration paths — and wire the exported library into
   `CMakeLists.txt`.

## Build & flash

```bash
cd D:\ble-rssi-ml-classifier\firmware\inference_app
west build -b nrf52840dk/nrf52840 . -d build
west flash -d build
```

## Measuring the numbers for the resume bullet

With a serial terminal open on the DK's COM port:

- **Inference latency**: the `us` field on each `RESULT,...` line.
- **Flash/RAM footprint**: printed by `west build` at the end
  (`Memory region ... FLASH/RAM ... used`), or `west build -t rom_report` /
  `ram_report` for a per-symbol breakdown once the real model is linked in.
- **Accuracy**: compare `RESULT,...` bucket output against the ground-truth
  distance you're actually holding the beacon at, over a held-out session
  not used in training. Record in `../../results/accuracy_report.md`.
