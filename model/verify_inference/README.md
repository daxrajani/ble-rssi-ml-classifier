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

**A second real bug, found in review and now fixed: final-layer logit
precision.** The last dense layer used to requantize its logits at
scale 1.0 / zero point 0 and clamp to INT8. The model's real logits live
at roughly 0.1 per quantization step, so that rounded nearby logits to the
same integer (2.3 and 2.4 both became 2) and `argmax` could pick the wrong
class. The final layer now returns dequantized float logits
(`dense_layer_logits()` in `inference.c`) and takes `argmax` over those -
valid because softmax is monotonic. Effect on the check below: agreement
with the TFLite interpreter went from 98.71% (839/850, 11 disagreements)
to **99.65% (847/850, 3 disagreements)**.

**Algorithm-level check: 99.65% agreement (847/850), not 100%.**
`verify_inference.py` transcribes `inference.c`'s algorithm into Python
(quantize -> 2x dense_layer -> dense_layer_logits -> argmax) and compares
its predictions against the official TFLite interpreter, run on identical
INT8 input tensors. The 3 remaining disagreements are all near-boundary
windows between adjacent classes (near/mid, near/far); they were not
root-caused individually, but are consistent with `inference.c`'s
documented design choice to requantize hidden layers with plain `float`
math (the nRF52840 has an FPU) rather than TFLite's fixed-point
requantization, and with TFLite taking argmax over an INT8 softmax output
that can tie. C-port accuracy vs. true labels is 96.82% against the
official interpreter's 96.94%, close to the production figure in
`results/accuracy_report.md` (96.24%, measured with overlapping windows
rather than this script's per-bucket ones).

## Native check of the shipped C

`test_main.c` compiles the *actual* `inference.c` + `model_weights.c` as a
host executable (`inference.c` is pure C with no Zephyr dependency, by
design) and replays the dataset through `classify_distance()` with the
same windowing as `verify_inference.py`. With MinGW-w64 GCC
(`gcc 12.2`, C11, `-Wall -Wextra -Wpedantic -Wconversion`, no warnings in
`inference.c`), its predictions are **identical to the Python port's on
all 850 windows** (96.82% accuracy vs. true labels) - so the shipped C and
the Python port implement the same algorithm.

```bash
cd model/verify_inference
gcc -std=c11 -O2 -Wall -Wextra test_main.c \
    ../../firmware/inference_app/src/inference.c \
    ../../firmware/inference_app/src/model_weights.c -lm -o test_inference
./test_inference ../../dataset/rssi_samples.csv > predictions.csv
```

(MSVC works too if the Windows SDK headers are installed:
`cl /O2 test_main.c ...inference.c ...model_weights.c /Fe:test_inference.exe`.)
`predictions.csv`'s `pred_bucket` column can be diffed against
`verify_inference.py`'s per-window classifications.

## Running the Python check

```bash
cd model/verify_inference
python verify_inference.py
```
