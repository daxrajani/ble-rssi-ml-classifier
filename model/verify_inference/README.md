# verify_inference

Independent verification of `firmware/inference_app/src/inference.c`'s
hand-rolled INT8 forward pass against the official `tf.lite.Interpreter`
on all 850 real dataset windows.

## What this found

**A real bug, now fixed.** `scanner.c` was passing its circular RSSI
buffer straight to `classify_distance()` without reordering it
chronologically. `classify_distance()` reads index `[n-1]` as "the sample
just pushed" - correct only in the first 8 calls per device before the
buffer wraps, then correct only 1-in-8 calls afterward (right when
`window_pos` happens to wrap to 0). Mean/std features were unaffected
(order-independent), but the "latest RSSI" feature was stale up to 87.5%
of the time for any device tracked past its first 8 readings. Fixed by
reordering into a temporary chronological copy before calling
`classify_distance()` - see the `device_found()` change in `scanner.c`.

**Algorithm-level check: 98.71% agreement (839/850), not 100% - expected,
not a bug.** `verify_inference.py` transcribes `inference.c`'s algorithm
into Python (quantize -> 3x dense_layer -> argmax) and compares its
predictions against the official TFLite interpreter, run on identical
INT8 input tensors. The 11 disagreements are all boundary cases between
adjacent classes (near/mid), consistent with `inference.c`'s documented
design choice: it requantizes with plain `float` math (the nRF52840 has
an FPU) rather than TFLite's internal fixed-point/gemmlowp-style integer
requantization - a deliberate accuracy/simplicity tradeoff, not an
implementation error. C-port accuracy vs. true labels (96.12%) closely
matches the official model's (96.94%) and the production number reported
in `results/accuracy_report.md` (96.24%, measured with overlapping
windows rather than this script's non-overlapping ones).

## What wasn't possible here

Ideally you'd compile the *actual* `inference.c` + `model_weights.c` as a
native host executable (via `test_main.c`, included) and diff its output
directly - stronger evidence than a Python reimplementation, since it
tests the literal shipped code. Attempted with MSVC (`cl.exe`, present on
this machine via Visual Studio 2022), but the Windows SDK headers
(`stdio.h` etc.) aren't installed alongside it - only the VC compiler
component. Fixable with a large SDK download; not done here. If you have
a working C toolchain (MSVC with Windows SDK, or MinGW), run:
```bash
cl /O2 test_main.c ../../firmware/inference_app/src/inference.c ../../firmware/inference_app/src/model_weights.c /Fe:test_inference.exe
test_inference.exe ../../dataset/rssi_samples.csv > predictions.csv
```
and diff `predictions.csv`'s `pred_bucket` column against
`verify_inference.py`'s classifications for the same windows - should
match exactly, since both implement the identical algorithm.

## Running

```bash
cd model/verify_inference
python verify_inference.py
```
