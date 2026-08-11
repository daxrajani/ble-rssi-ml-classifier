# Accuracy / latency / footprint report

First real measurement pass, from the initial data collection + training
run (2026-08-11). Numbers below are real, not estimated - re-run and update
this file after collecting multi-session/multi-room data (see the caveat
under Accuracy).

## Dataset

- Total labeled samples: 850
- Per-bucket counts: near=289, mid=287, far=274
- Number of collection sessions / environments: 1 (`room1_run1`) - single
  room, single orientation. **Not yet the multi-session/multi-room dataset
  the project plan calls for** - this is a first working pipeline, not the
  final accuracy claim.
- Raw RSSI ranges observed: near -56 to -78 dBm (mean -69.2), mid -70 to
  -78 dBm (mean -72.7), far -74 to -88 dBm (mean -79.6). Near/mid overlap
  significantly - expected RSSI noise, and exactly why a learned model
  (vs. a fixed threshold) is the point of this project.

## Model

- Architecture: Dense(3->8, ReLU) -> Dense(8->8, ReLU) -> Dense(8->3,
  softmax). Features per window: latest RSSI, rolling mean, rolling std
  (window=5).
- Quantization: INT8, full-integer (TFLite converter, INT8 in/out,
  per-channel weight quantization)
- `.tflite` file size: 3,160 bytes (`model/model_int8.tflite`)
- On-device integration: hand-rolled INT8 dense-layer forward pass
  (`firmware/inference_app/src/inference.c` + `model_weights.c`) generated
  from the trained model's weights - not TFLite Micro (see comment at top
  of `inference.c` for why).

## On-device measurements (nRF52840 DK, nrf52840dk/nrf52840, NCS v3.2.4)

- Inference latency (from `RESULT,...,<us>` in inference_app's UART
  output, sampled over ~350 live classifications): 61-91us typical,
  occasional outliers up to ~213us (likely scheduler jitter, not the math
  itself). Consistently sub-millisecond.
- Flash usage: 166,148 B / 1 MB (15.85%)
- RAM usage: 28,608 B / 256 KB (10.91%)
- (For comparison, `scanner_app` alone - no model, no BLE peripheral/GATT
  service - is 83,208 B flash / 22,192 B RAM, so the model + inference
  code + result-notify GATT service added roughly 83 KB flash / 6.4 KB
  RAM.)

## Accuracy

- Held-out test accuracy: **92.02%** (grouped confusion matrix below).
  Split method: **stratified random 75/25**, not a session-held-out
  split - there was only one `session_id` at training time, so a proper
  `GroupShuffleSplit` by session (the intended, less-optimistic method;
  see `model/train.py`) couldn't run. Treat 92% as an optimistic upper
  bound until retrained on multiple sessions/rooms and re-split by
  session.
- Confusion matrix (rows=true, cols=pred, order near/mid/far), n=213 test
  rows:
  ```
          near  mid  far
  near      59   11    2
  mid        3   68    1
  far        0    0   69
  ```
  far is perfectly separated; almost all errors are near<->mid confusion,
  consistent with their overlapping raw-RSSI ranges above.
- Live on-device sanity check (ambient BLE devices, unfiltered by MAC):
  strong/close signals (-57 to -63 dBm) consistently classified NEAR,
  weak/distant signals (-85 to -92 dBm) consistently classified FAR,
  mid-range signals (-70 to -80 dBm) split between buckets - matches the
  near/mid confusion seen in the held-out numbers above.

## Next steps to get a trustworthy accuracy number

1. Collect `mid`/`far` walk-arounds' worth of additional `session_id`s in
   different rooms/orientations (plan's original checklist).
2. Re-run `model/train.py` - once >=2 session_ids exist it automatically
   switches to the proper `GroupShuffleSplit`.
3. Re-run the weight-extraction step to regenerate
   `firmware/inference_app/src/model_weights.c`, rebuild, reflash.
