#!/usr/bin/env python3
"""
Verifies firmware/inference_app/src/inference.c's INT8 forward-pass
algorithm against the official TFLite interpreter on real dataset
windows. This is an independent Python transcription of inference.c's
logic (quantize -> 3x dense_layer -> argmax), not a copy-paste, so
agreement between the two is real evidence the C algorithm is correct -
not just self-consistent.

(Note: a native compile-and-run of the actual inference.c was attempted
first - stronger evidence than a reimplementation - but this machine's
Visual Studio install has the compiler without the Windows SDK headers,
so that wasn't possible without a large SDK download. This script is the
fallback.)

Usage: python verify_inference.py
"""
import numpy as np
import pandas as pd
import tensorflow as tf

DATASET = "../../dataset/rssi_samples.csv"
TFLITE = "../model_int8.tflite"
WINDOW = 8


def quantize(real_value, scale, zero_point):
    q = int(round(real_value / scale)) + zero_point
    return max(-128, min(127, q))


def dense_layer(inp, in_zp, in_scale, w, bias, w_scales, out_scale, out_zp, relu):
    out_len = w.shape[0]
    in_len = w.shape[1]
    out = np.zeros(out_len, dtype=np.int32)
    for c in range(out_len):
        acc = 0
        for i in range(in_len):
            acc += (int(inp[i]) - in_zp) * int(w[c, i])
        real = acc * (in_scale * w_scales[c]) + bias[c] * (in_scale * w_scales[c])
        q = int(round(real / out_scale)) + out_zp
        if relu and q < out_zp:
            q = out_zp
        q = max(-128, min(127, q))
        out[c] = q
    return out


def classify_distance_c_port(rssi_window, weights):
    """Direct transcription of inference.c's classify_distance()."""
    n = len(rssi_window)
    mean = float(np.mean(rssi_window))
    std = float(np.std(rssi_window)) if n > 1 else 0.0  # population std, matches C's sqrtf(sum_sq/n)
    raw = np.array([rssi_window[-1], mean, std], dtype=np.float64)

    normalized = (raw - weights["feature_mean"]) / weights["feature_std"]
    q_in = np.array([quantize(v, weights["input_scale"], weights["input_zp"]) for v in normalized])

    a1 = dense_layer(q_in, weights["input_zp"], weights["input_scale"],
                      weights["l1_w"], weights["l1_b"], weights["l1_w_scales"],
                      weights["l1_act_scale"], weights["l1_act_zp"], relu=True)
    a2 = dense_layer(a1, weights["l1_act_zp"], weights["l1_act_scale"],
                      weights["l2_w"], weights["l2_b"], weights["l2_w_scales"],
                      weights["l2_act_scale"], weights["l2_act_zp"], relu=True)
    logits = dense_layer(a2, weights["l2_act_zp"], weights["l2_act_scale"],
                          weights["l3_w"], weights["l3_b"], weights["l3_w_scales"],
                          out_scale=1.0, out_zp=0, relu=False)
    return int(np.argmax(logits))


def load_weights():
    interp = tf.lite.Interpreter(model_path=TFLITE)
    interp.allocate_tensors()
    det = {d["index"]: d for d in interp.get_tensor_details()}

    def T(i):
        return interp.get_tensor(i)

    def QP(i):
        return det[i]["quantization_parameters"]

    with open("../normalization.txt") as f:
        ns = {}
        exec(f.read(), {}, ns)

    return {
        "input_scale": float(QP(0)["scales"][0]),
        "input_zp": int(QP(0)["zero_points"][0]),
        "l1_w": T(6), "l1_b": T(5), "l1_w_scales": QP(6)["scales"],
        "l1_act_scale": float(QP(7)["scales"][0]), "l1_act_zp": int(QP(7)["zero_points"][0]),
        "l2_w": T(4), "l2_b": T(3), "l2_w_scales": QP(4)["scales"],
        "l2_act_scale": float(QP(8)["scales"][0]), "l2_act_zp": int(QP(8)["zero_points"][0]),
        "l3_w": T(2), "l3_b": T(1), "l3_w_scales": QP(2)["scales"],
        "feature_mean": np.array(ns["mean"]), "feature_std": np.array(ns["std"]),
    }, interp


def official_tflite_predict(interp, rssi_window, weights):
    n = len(rssi_window)
    mean = float(np.mean(rssi_window))
    std = float(np.std(rssi_window)) if n > 1 else 0.0
    raw = np.array([rssi_window[-1], mean, std], dtype=np.float32)
    normalized = (raw - weights["feature_mean"]) / weights["feature_std"]
    q_in = np.array([quantize(v, weights["input_scale"], weights["input_zp"])
                      for v in normalized], dtype=np.int8).reshape(1, 3)

    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    interp.set_tensor(input_details[0]["index"], q_in)
    interp.invoke()
    out = interp.get_tensor(output_details[0]["index"])
    return int(np.argmax(out[0]))


def main():
    weights, interp = load_weights()
    df = pd.read_csv(DATASET)
    df = df.sort_values(["distance_bucket", "timestamp_ms"]).reset_index(drop=True)

    buckets = ["near", "mid", "far"]
    n_total, n_agree, n_correct_c, n_correct_tflite = 0, 0, 0, 0
    mismatches = []

    for bucket_name, group in df.groupby("distance_bucket"):
        rssi_vals = group["rssi"].values
        true_idx = buckets.index(bucket_name)
        for i in range(len(rssi_vals)):
            window = rssi_vals[max(0, i - WINDOW + 1):i + 1]
            c_pred = classify_distance_c_port(window, weights)
            tflite_pred = official_tflite_predict(interp, window, weights)

            n_total += 1
            if c_pred == tflite_pred:
                n_agree += 1
            else:
                mismatches.append((bucket_name, i, window.tolist(), c_pred, tflite_pred))
            if c_pred == true_idx:
                n_correct_c += 1
            if tflite_pred == true_idx:
                n_correct_tflite += 1

    print(f"Total windows checked: {n_total}")
    print(f"C-port vs official TFLite agreement: {n_agree}/{n_total} ({100*n_agree/n_total:.2f}%)")
    print(f"C-port accuracy vs true label:        {n_correct_c}/{n_total} ({100*n_correct_c/n_total:.2f}%)")
    print(f"Official TFLite accuracy vs true label: {n_correct_tflite}/{n_total} ({100*n_correct_tflite/n_total:.2f}%)")

    if mismatches:
        print(f"\n{len(mismatches)} mismatches (first 10):")
        for m in mismatches[:10]:
            print(f"  bucket={m[0]} idx={m[1]} window={m[2]} c_pred={buckets[m[3]]} tflite_pred={buckets[m[4]]}")
    else:
        print("\nNo mismatches - C-port algorithm exactly matches the official TFLite interpreter.")


if __name__ == "__main__":
    main()
