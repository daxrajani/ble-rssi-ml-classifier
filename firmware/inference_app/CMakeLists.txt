cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(rssi_inference)

target_sources(app PRIVATE
    src/main.c
    src/scanner.c
    src/inference.c
    src/model_weights.c
    src/leds.c
    src/ble_result_svc.c
)

# Model is a hand-rolled INT8 dense-layer forward pass over weights
# extracted from model/model_int8.tflite (see src/model_weights.c and the
# comment at the top of src/inference.c for why this isn't TFLite Micro).
# Regenerate src/model_weights.c after retraining - see model/README.md.
