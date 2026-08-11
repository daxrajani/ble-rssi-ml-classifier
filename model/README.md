# model/

## What actually happened (first pass, 2026-08-11)

`train.py` was used instead of the notebook for the first real training
run (same logic, easier to run non-interactively): loads
`../dataset/rssi_samples.csv`, engineers rolling-mean/std features, trains
the 3->8->8->3 Dense model, and exports `model_int8.tflite`.
`extract_weights.py` then pulls the INT8 weights/quantization params out
of that `.tflite` file and writes them as C source directly into
`../firmware/inference_app/src/model_weights.c` - see that folder's
README and `inference.c` for why this repo hand-rolls the INT8 forward
pass instead of linking TFLite Micro.

To retrain after collecting more data:
```bash
python train.py             # writes model_int8.tflite, normalization.txt
python extract_weights.py   # regenerates ../firmware/inference_app/src/model_weights.c
```
Then rebuild + reflash `inference_app`. Real measured accuracy/latency/
footprint numbers are in `../results/accuracy_report.md`.

## Primary path: Edge Impulse

1. Create a free Edge Impulse project.
2. Upload `dataset/rssi_samples.csv` (Data acquisition → Upload data), using
   `distance_bucket` as the label.
3. Impulse design: raw `rssi` (+ optionally a moving-average DSP block) →
   a small (2-3 layer) dense NN classifier, 3 classes (near/mid/far).
4. Enable INT8 quantization in the deployment step.
5. Export as a **Zephyr / nRF Connect SDK library** — this drops directly
   into `firmware/inference_app/` (see that folder's README for the
   integration point).

This is a manual, browser-driven process — do it yourself in the Edge
Impulse UI once you have real collected data in `dataset/rssi_samples.csv`.

## Alternative path: `training_notebook.ipynb`

For more control (and a stronger "I understand the internals" interview
story), `training_notebook.ipynb` trains a small Keras model locally,
INT8-quantizes it with the standard TFLite converter, and writes
`model_int8.tflite`. You'd then integrate that with TFLite Micro manually in
`firmware/inference_app/` instead of using the Edge Impulse export.

Pick one path, not both.
