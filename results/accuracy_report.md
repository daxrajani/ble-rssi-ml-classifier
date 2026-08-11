# Accuracy / latency / footprint report

Fill this in with real measured numbers once you have a trained model
flashed to the DK — don't estimate ahead of time (see project plan §5).

## Dataset

- Total labeled samples:
- Per-bucket counts (near / mid / far):
- Number of collection sessions / environments:

## Model

- Architecture (layers, params):
- Quantization: INT8 (Edge Impulse / manual TFLite Micro)
- `.tflite` file size:

## On-device measurements (nRF52840 DK)

- Inference latency (from `RESULT,...,<us>` in inference_app's UART output):
  - min / mean / max over N runs:
- Flash usage (`west build` memory report):
- RAM usage (`west build` memory report):

## Accuracy

- Held-out test accuracy (session-level split, not row-level):
- Confusion matrix (near/mid/far):
- Notes on failure modes (e.g. orientation-dependent misclassification):
