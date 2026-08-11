# inference_app

Runs on the same nRF52840 DK as `scanner_app`, but adds:

- a windowed-average feature (`src/scanner.c`, `INFERENCE_WINDOW_LEN` samples)
- a real trained classifier (`src/inference.c` + `src/model_weights.c`) -
  a hand-rolled INT8 dense-layer forward pass over weights extracted from
  `model/model_int8.tflite`, not TFLite Micro (see the comment at the top
  of `inference.c` for why: this NCS workspace doesn't have a tflite-micro
  west module, and the model is small enough that porting the full C++
  runtime would be a much bigger, riskier change than ~80 lines of
  quantized matmul)
- per-bucket LEDs: led0=NEAR, led1=MID, led2=FAR (`src/leds.c`)
- a BLE GATT service that notifies subscribers with the classification
  result (`src/ble_result_svc.c`)
- per-inference latency measurement, printed as `RESULT,<mac>,<rssi>,<bucket>,<us>`
  (measured: 61-91us typical on nRF52840 - see `../../results/accuracy_report.md`)

## Before you flash this

1. Set `TARGET_BEACON_MAC` in `src/scanner.h` to your beacon's current MAC
   (same one you used with `log_to_csv.py` - remember it rotates, so
   re-check with `--list-devices` if it's stale), or leave it empty to
   classify whatever device is heard most recently.
2. After retraining on more data, regenerate `src/model_weights.c` - see
   `../../model/README.md` for the two-command retrain+extract flow -
   then rebuild.

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
