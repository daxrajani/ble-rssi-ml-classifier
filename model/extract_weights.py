#!/usr/bin/env python3
"""
Extracts INT8 weights/biases/quantization params + feature normalization
stats from model_int8.tflite (produced by train.py) and writes them as C
source into firmware/inference_app/src/model_weights.c.

Run this after every re-run of train.py, then rebuild+reflash
inference_app. See the top-of-file comment in
firmware/inference_app/src/inference.c for why this repo hand-rolls the
INT8 forward pass instead of linking TFLite Micro.

Assumes the exact 3->8->8->3 Dense/ReLU/Dense/ReLU/Dense/Softmax
architecture defined in train.py. If you change the architecture, you'll
need to update both this script's tensor-index assumptions and the
forward-pass code in inference.c to match.
"""
import numpy as np
import tensorflow as tf

TFLITE_PATH = "model_int8.tflite"
NORMALIZATION_PATH = "normalization.txt"
OUT_PATH = "../firmware/inference_app/src/model_weights.c"

# Tensor indices for this specific architecture, as produced by train.py's
# Sequential([Dense(8,relu), Dense(8,relu), Dense(3,softmax)]) once
# converted to INT8 TFLite. Verified against `interp.get_tensor_details()`
# output - if train.py's architecture changes, re-derive these.
IN = 0
L1_W, L1_B, L1_ACT = 6, 5, 7
L2_W, L2_B, L2_ACT = 4, 3, 8
L3_W, L3_B = 2, 1


def parse_normalization():
    ns = {}
    exec(open(NORMALIZATION_PATH).read(), {}, ns)
    return ns["mean"], ns["std"]


def c_i8_arr(name, arr):
    flat = arr.flatten().astype(int).tolist()
    return f"static const int8_t {name}[{len(flat)}] = {{" + ", ".join(str(v) for v in flat) + "};"


def c_i32_arr(name, arr):
    flat = arr.flatten().astype(int).tolist()
    return f"static const int32_t {name}[{len(flat)}] = {{" + ", ".join(str(v) for v in flat) + "};"


def c_f32_arr(name, arr):
    flat = [float(v) for v in np.array(arr).flatten()]
    return f"static const float {name}[{len(flat)}] = {{" + ", ".join(f"{v:.10g}f" for v in flat) + "};"


def main():
    mean, std = parse_normalization()

    interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interp.allocate_tensors()
    det = {d["index"]: d for d in interp.get_tensor_details()}

    def T(i):
        return interp.get_tensor(i)

    def QP(i):
        return det[i]["quantization_parameters"]

    in_scale, in_zp = float(QP(IN)["scales"][0]), int(QP(IN)["zero_points"][0])
    l1_w, l1_b = T(L1_W), T(L1_B)
    l2_w, l2_b = T(L2_W), T(L2_B)
    l3_w, l3_b = T(L3_W), T(L3_B)
    l1_w_scales = QP(L1_W)["scales"]
    l2_w_scales = QP(L2_W)["scales"]
    l3_w_scales = QP(L3_W)["scales"]
    l1_act_scale, l1_act_zp = float(QP(L1_ACT)["scales"][0]), int(QP(L1_ACT)["zero_points"][0])
    l2_act_scale, l2_act_zp = float(QP(L2_ACT)["scales"][0]), int(QP(L2_ACT)["zero_points"][0])

    lines = [
        "/* Auto-generated from model_int8.tflite by model/extract_weights.py.",
        " * Trained model weights/quantization params, embedded directly - no TFLite Micro",
        " * runtime dependency. See inference.c for the matching forward pass.",
        " * Regenerate whenever the model is retrained on new data. */",
        "",
        '#include "model_weights.h"',
        "",
        f"const float model_feature_mean[3] = {{ {mean[0]:.10g}f, {mean[1]:.10g}f, {mean[2]:.10g}f }};",
        f"const float model_feature_std[3]  = {{ {std[0]:.10g}f, {std[1]:.10g}f, {std[2]:.10g}f }};",
        "",
        f"const float model_input_scale = {in_scale:.10g}f;",
        f"const int32_t model_input_zero_point = {in_zp};",
        "",
        "/* Layer 1: 3 -> 8, ReLU */",
        c_i8_arr("l1_weights", l1_w),
        c_i32_arr("l1_bias", l1_b),
        c_f32_arr("l1_weight_scales", l1_w_scales),
        f"const float l1_act_scale = {l1_act_scale:.10g}f;",
        f"const int32_t l1_act_zero_point = {l1_act_zp};",
        "",
        "/* Layer 2: 8 -> 8, ReLU */",
        c_i8_arr("l2_weights", l2_w),
        c_i32_arr("l2_bias", l2_b),
        c_f32_arr("l2_weight_scales", l2_w_scales),
        f"const float l2_act_scale = {l2_act_scale:.10g}f;",
        f"const int32_t l2_act_zero_point = {l2_act_zp};",
        "",
        "/* Layer 3: 8 -> 3, no activation (argmax of logits == argmax of softmax) */",
        c_i8_arr("l3_weights", l3_w),
        c_i32_arr("l3_bias", l3_b),
        c_f32_arr("l3_weight_scales", l3_w_scales),
        "",
        "const struct model_weights g_model_weights = {",
        "\t.l1_w = l1_weights, .l1_b = l1_bias, .l1_w_scales = l1_weight_scales,",
        "\t.l1_act_scale = &l1_act_scale, .l1_act_zp = &l1_act_zero_point,",
        "\t.l2_w = l2_weights, .l2_b = l2_bias, .l2_w_scales = l2_weight_scales,",
        "\t.l2_act_scale = &l2_act_scale, .l2_act_zp = &l2_act_zero_point,",
        "\t.l3_w = l3_weights, .l3_b = l3_bias, .l3_w_scales = l3_weight_scales,",
        "};",
        "",
    ]

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
