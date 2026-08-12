# Accuracy / latency / footprint report

Data collected 2026-08-11, model tuned via a feature/architecture sweep the
same day. Numbers below are real, not estimated. Still single-session data
(see caveat under Accuracy) - re-run after multi-room/session collection.

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
  softmax). Chosen via a sweep over (4,), (8,), (8,8), (16,16), (8,8,8) -
  (8,8) won on accuracy at a small size (see Feature/architecture sweep
  below).
- Features per window: latest RSSI, rolling mean, rolling std.
  **Window=8** (chosen via sweep - see below; window=5 was the original
  guess and is meaningfully worse).
- Quantization: INT8, full-integer (TFLite converter, INT8 in/out,
  per-channel weight quantization)
- `.tflite` file size: 3,160 bytes (`model/model_int8.tflite`)
- On-device integration: hand-rolled INT8 dense-layer forward pass
  (`firmware/inference_app/src/inference.c` + `model_weights.c`) generated
  from the trained model's weights - not TFLite Micro (see comment at top
  of `inference.c` for why).

## Feature/architecture sweep (`model/experiment.py`)

Compared on the same single-session dataset/split methodology, so
relative rankings are more trustworthy than absolute numbers.

Feature engineering (architecture fixed at (8,8)):
| feature_set | window | accuracy | tflite size |
|---|---|---|---|
| rssi_only | 5 | 88.73% | 3176 B |
| rssi_mean_std | 3 | 90.45% | 3221 B |
| rssi_mean_std | 5 | 92.64% | 3224 B |
| **rssi_mean_std** | **8** | **96.71%** | **3240 B** |
| rssi_ema_delta | 3 | 89.98% | 3248 B |
| rssi_ema_delta | 5 | 91.86% | 3248 B |
| rssi_ema_delta | 8 | 92.33% | 3248 B |

Architecture (feature config fixed at rssi_mean_std, window=8):
| hidden_units | accuracy | tflite size |
|---|---|---|
| (4,) | 94.05% | 2280 B |
| **(8, 8)** | **96.87%** | **3248 B** |
| (8,) | 95.77% | 2416 B |
| (16, 16) | 96.71% | 3936 B |
| (8, 8, 8) | 96.56% | 4088 B |

Takeaways: larger windows (more smoothing) consistently beat smaller ones
and beat the alternative EMA/delta feature at every window size tested -
plain rolling mean/std was already a good feature choice, it just needed
a bigger window than the original guess of 5. (8,8) beat both smaller and
larger architectures - the extra capacity of (16,16)/(8,8,8) didn't help
on this dataset size and cost 20-25% more flash for no accuracy gain.

## On-device measurements (nRF52840 DK, nrf52840dk/nrf52840, NCS v3.2.4)

- Inference latency (from `RESULT,...,<us>` in inference_app's UART
  output, sampled over ~350+ live classifications, both before and after
  the window=8 retrain): 61-91us typical, occasional outliers up to
  ~213us (likely scheduler jitter, not the math itself). Consistently
  sub-millisecond.
- Flash usage: 166,292 B / 1 MB (15.86%)
- RAM usage: 28,992 B / 256 KB (11.06%)
- (For comparison, `scanner_app` alone - no model, no BLE peripheral/GATT
  service, no multi-device tracking - is 83,208 B flash / 22,192 B RAM.)
- Firmware also now tracks each nearby BLE device in its own RSSI window
  (`scanner.c`, up to 8 devices, LRU-evicted) instead of sharing one
  window across all devices - fixes a real bug where classification could
  blend samples from unrelated physical devices.

## Accuracy

- Held-out test accuracy: **96.24%** (production config: window=8,
  hidden_units=(8,8)). Split method: **stratified random 75/25**, not a
  session-held-out split - there was only one `session_id` at training
  time, so a proper `GroupShuffleSplit` by session (the intended, less
  optimistic method; see `model/train.py`) couldn't run. **Treat 96% as
  an optimistic upper bound** until retrained on multiple sessions/rooms
  and re-split by session.
- **Baseline comparison** (nearest-mean RSSI threshold, i.e. a
  hand-tuned-filtering equivalent, fit on the same training data): 75.59%
  accuracy. **The trained model beats this baseline by +20.66 points** -
  the core claim of the project (learned model vs. hand-tuned filtering)
  is now backed by a real number, not just qualitative reasoning.
- Confusion matrix (rows=true, cols=pred, order near/mid/far), n=213 test
  rows:
  ```
          near  mid  far
  near      67    3    2
  mid        3   69    0
  far        0    0   69
  ```
- Per-class precision/recall/F1:
  - near: precision=0.957, recall=0.931, f1=0.944
  - mid: precision=0.958, recall=0.958, f1=0.958
  - far: precision=0.972, recall=1.000, f1=0.986
  - far is essentially perfectly separated; remaining errors are almost
    all near<->mid confusion, consistent with their overlapping raw-RSSI
    ranges above.
- Live on-device sanity check (ambient BLE devices, unfiltered by MAC,
  each now tracked in its own window): devices consistently close/strong
  (-57 to -63 dBm) classify NEAR every reading; devices consistently
  distant/weak (-80 to -90 dBm) classify FAR every reading - much more
  stable per-device than before the per-MAC window fix.

## Edge Impulse comparison (2026-08-11)

The project plan's primary recommended path (upload CSV to Edge Impulse's
web UI, let their DSP+NN pipeline auto-build a model) was run as a
comparison against the hand-rolled `train.py`/`extract_weights.py` path
actually used above. Same dataset (`dataset/rssi_samples.csv`, reshaped
into per-bucket recordings for their CSV Wizard - see
`dataset/export_edge_impulse.py`), same nRF52840 DK target device.

- **Data ingestion gotcha**: Edge Impulse's plain CSV upload silently
  discarded our real per-row timestamps and assumed a fixed ~250-333Hz
  sample rate unless routed through their **CSV Wizard** with the
  timestamp column explicitly mapped as "elapsed milliseconds" - worth
  knowing if you try this yourself, the error is silent (wrong recording
  length shown, not a rejected upload).
- **Impulse config**: window=2136ms / stride=500ms (≈8 raw samples/window,
  matching our own window=8), Flatten DSP block (avg/RMS/std-dev/skewness/
  kurtosis - close to but richer than our mean/std), Classification
  (Keras) learning block, EI's default architecture (Dense 20 → Dense 10,
  vs. our swept (8,8)).
- **Normalization mattered a lot**: first run, EI's default "don't
  normalize data" gave only **42.6%** accuracy. Switching to "Normalize
  using scikit-learn's StandardScaler" (equivalent to what `train.py`
  already does) jumped it to **95.3%** - a reminder that an unfair
  comparison (normalized vs. not) would have wrongly made this project's
  approach look far better than it is.
- **Result, apples-to-apples (both normalized, both INT8 quantized)**:

  | | This project (train.py) | Edge Impulse |
  |---|---|---|
  | Accuracy | 96.24% (window=8, (8,8)) | 95.3% (window≈8, Dense 20→10) |
  | far / mid / near F1 | 0.986 / 0.958 / 0.944 | 0.97 / 0.93 / 0.95 |
  | Model size | 3,160 B (.tflite) | 14.5 KB flash (EON Compiler estimate) |
  | Peak RAM | n/a (compiled into inference.c) | 1.4 KB |
  | Inference latency | 61-91us (measured live on DK) | 1 ms (EON Compiler estimate, not yet measured live) |

  Essentially tied on accuracy. Our hand-rolled path is meaningfully
  smaller and faster - expected, since Edge Impulse's default Dense
  20→10 network and 5-statistic Flatten block have more capacity than
  necessary for a 3-class problem on this little data, while our sweep
  (`model/experiment.py`) specifically searched for the smallest
  architecture that didn't lose accuracy.
- **Not done**: exporting/flashing the Edge Impulse model to actually
  measure its on-device latency/footprint on the DK (the 1ms/1.4KB/14.5KB
  numbers above are Edge Impulse's own EON Compiler estimates, not a live
  measurement like our 61-91us number is). Their deployment page has a
  direct "nRF Connect SDK" export if this comparison needs to go further.

## Next steps to get a trustworthy accuracy number

1. Collect `near`/`mid`/`far` walk-arounds' worth of additional
   `session_id`s in different rooms/orientations (plan's original
   checklist, `dataset/README.md`).
2. Re-run `model/train.py` - once >=2 session_ids exist it automatically
   switches to the proper `GroupShuffleSplit`.
3. Re-run `model/experiment.py` on the bigger dataset to check whether
   window=8/(8,8) is still the best config once there's more/noisier
   data to fit.
4. Re-run `model/extract_weights.py` to regenerate
   `firmware/inference_app/src/model_weights.c`, rebuild, reflash.
