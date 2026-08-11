#!/usr/bin/env python3
"""
Sweeps feature-engineering and architecture variants using train.py's
train_and_evaluate(), and reports accuracy/baseline-gap/tflite-size for
each so a choice can be made with data instead of guessing. Does not
overwrite model_int8.tflite/model_weights.c - that's done by re-running
train.py + extract_weights.py with the chosen config once you've picked a
winner (see results printed at the end).

Caveat: like train.py, this is still evaluated on a single-session
dataset (stratified random split), so treat relative rankings between
configs as more trustworthy than the absolute accuracy numbers.
"""
import numpy as np
from train import train_and_evaluate, BUCKETS

FEATURE_SETS = ["rssi_only", "rssi_mean_std", "rssi_ema_delta"]
WINDOWS = [3, 5, 8]
ARCHITECTURES = [(4,), (8,), (8, 8), (16, 16), (8, 8, 8)]

N_REPEATS = 3  # average over a few seeds - single dense-layer training on ~850 rows is noisy


def run_config(feature_set, window, hidden_units):
    accs, baselines, sizes = [], [], []
    for seed in range(N_REPEATS):
        r = train_and_evaluate(window=window, feature_set=feature_set,
                                hidden_units=hidden_units, epochs=30, verbose=0)
        accs.append(r["test_acc"])
        baselines.append(r["baseline_acc"])
        sizes.append(r["tflite_size"])
    return {
        "feature_set": feature_set,
        "window": window,
        "hidden_units": hidden_units,
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "baseline_mean": float(np.mean(baselines)),
        "tflite_size": int(np.mean(sizes)),
    }


def main():
    print("=== Feature engineering sweep (architecture fixed at (8,8)) ===")
    feature_results = []
    for fs in FEATURE_SETS:
        for w in WINDOWS if fs != "rssi_only" else [5]:  # window doesn't affect rssi_only
            r = run_config(fs, w, (8, 8))
            feature_results.append(r)
            print(f"  feature_set={fs:16s} window={w}  "
                  f"acc={r['acc_mean']:.4f}+-{r['acc_std']:.4f}  "
                  f"baseline={r['baseline_mean']:.4f}  tflite={r['tflite_size']}B")

    best_feature = max(feature_results, key=lambda r: r["acc_mean"])
    print(f"\nBest feature config: feature_set={best_feature['feature_set']} "
          f"window={best_feature['window']} (acc={best_feature['acc_mean']:.4f})")

    print("\n=== Architecture sweep (feature config fixed at best-above) ===")
    arch_results = []
    for arch in ARCHITECTURES:
        r = run_config(best_feature["feature_set"], best_feature["window"], arch)
        arch_results.append(r)
        print(f"  hidden_units={str(arch):12s} "
              f"acc={r['acc_mean']:.4f}+-{r['acc_std']:.4f}  tflite={r['tflite_size']}B")

    best_arch = max(arch_results, key=lambda r: r["acc_mean"])
    print(f"\nBest architecture: {best_arch['hidden_units']} (acc={best_arch['acc_mean']:.4f}, "
          f"{best_arch['tflite_size']}B)")

    print("\n=== Summary ===")
    print(f"Current production config: feature_set=rssi_mean_std window=5 hidden_units=(8,8)")
    print(f"Best found: feature_set={best_feature['feature_set']} window={best_feature['window']} "
          f"hidden_units={best_arch['hidden_units']}")
    print("\nTo adopt the best config, edit the defaults in train.py's train_and_evaluate() "
          "call in main(), or call it directly with these kwargs, then re-run "
          "extract_weights.py and rebuild firmware. NOTE: changing hidden_units or the "
          "feature set changes the tensor-index assumptions in extract_weights.py AND the "
          "layer shapes in firmware/inference_app/src/inference.c - both need updating to match.")


if __name__ == "__main__":
    main()
