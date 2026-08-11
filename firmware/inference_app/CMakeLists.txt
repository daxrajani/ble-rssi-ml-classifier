cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(rssi_inference)

target_sources(app PRIVATE
    src/main.c
    src/scanner.c
    src/inference.c
    src/leds.c
    src/ble_result_svc.c
)

# Once you export a model as a Zephyr/NCS library from Edge Impulse (or
# build model_int8.tflite yourself with TFLite Micro), add it here, e.g.:
#
#   add_subdirectory(edge_impulse_sdk)
#   target_link_libraries(app PRIVATE edge_impulse_sdk)
#
# See src/inference.c for the single function to wire it into.
