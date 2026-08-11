#!/usr/bin/env python3
"""
Training script for the RSSI distance-bucket classifier. Mirrors
training_notebook.ipynb but structured as reusable functions so
experiment.py (feature/architecture sweeps) can reuse the same
data-loading, splitting, and training logic without duplicating it.

Note: at the time this was first run, all collected rows shared a single
session_id ("room1_run1" - near/mid/far collected back-to-back in one
session/room). The session-grouped train/test split needs >=2 sessions to
do anything meaningful, so this falls back to a stratified random split
instead - which is more optimistic than a proper session-held-out split.
Re-run with multiple session_ids (different rooms/orientations) for a
trustworthy number before quoting accuracy on a resume.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, GroupShuffleSplit

DATASET = "../dataset/rssi_samples.csv"
BUCKETS = ["near", "mid", "far"]


def load_features(window=5, feature_set="rssi_mean_std", dataset_path=DATASET):
    """feature_set options:
       - "rssi_mean_std": [latest rssi, rolling mean, rolling std] (default/production)
       - "rssi_only": [latest rssi] - simplest possible, for comparison
       - "rssi_ema_delta": [latest rssi, EMA, rssi - EMA] - trend-sensitive alternative
    """
    df = pd.read_csv(dataset_path)
    df = df.sort_values(["distance_bucket", "timestamp_ms"]).reset_index(drop=True)
    grp = df.groupby("distance_bucket")["rssi"]

    if feature_set == "rssi_mean_std":
        df["f1"] = grp.transform(lambda s: s.rolling(window, min_periods=1).mean())
        df["f2"] = grp.transform(lambda s: s.rolling(window, min_periods=1).std().fillna(0))
        feature_cols = ["rssi", "f1", "f2"]
    elif feature_set == "rssi_only":
        feature_cols = ["rssi"]
    elif feature_set == "rssi_ema_delta":
        alpha = 2.0 / (window + 1)
        df["f1"] = grp.transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
        df["f2"] = df["rssi"] - df["f1"]
        feature_cols = ["rssi", "f1", "f2"]
    else:
        raise ValueError(f"unknown feature_set {feature_set}")

    return df, feature_cols


def split_data(df, feature_cols, test_size=0.25, seed=0):
    X = df[feature_cols].values.astype("float32")
    y = df["distance_bucket"].map({b: i for i, b in enumerate(BUCKETS)}).values
    n_sessions = df["session_id"].nunique()

    if n_sessions < 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y)
        split_method = "stratified_random (only 1 session_id present)"
    else:
        groups = df["session_id"].values
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(X, y, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        split_method = "session_held_out"

    return X_train, X_test, y_train, y_test, split_method


def baseline_threshold_accuracy(X_train, y_train, X_test, y_test):
    """Naive baseline: classify by nearest per-bucket mean RSSI (first
    feature column), thresholds fit from training data. Stands in for
    "hand-tuned RSSI filtering" (the WiRa-style approach this project is
    explicitly benchmarking a learned model against) rather than a fixed
    guess picked without looking at data.
    """
    bucket_means = [X_train[y_train == c, 0].mean() for c in range(len(BUCKETS))]
    order = np.argsort(bucket_means)  # near..far by ascending distance -> descending RSSI typically
    thresholds = [(bucket_means[order[i]] + bucket_means[order[i + 1]]) / 2
                  for i in range(len(order) - 1)]

    def classify(rssi):
        # thresholds are sorted by bucket_means[order], midpoints between
        # consecutive ones; assign to nearest mean.
        best, best_dist = None, None
        for c in range(len(BUCKETS)):
            d = abs(rssi - bucket_means[c])
            if best_dist is None or d < best_dist:
                best, best_dist = c, d
        return best

    y_pred = np.array([classify(x) for x in X_test[:, 0]])
    acc = (y_pred == y_test).mean()
    return acc, bucket_means


def per_class_metrics(y_test, y_pred):
    metrics = {}
    for c, name in enumerate(BUCKETS):
        tp = np.sum((y_pred == c) & (y_test == c))
        fp = np.sum((y_pred == c) & (y_test != c))
        fn = np.sum((y_pred != c) & (y_test == c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[name] = {"precision": precision, "recall": recall, "f1": f1,
                          "support": int(np.sum(y_test == c))}
    return metrics


def build_model(in_dim, hidden_units=(8, 8), num_classes=len(BUCKETS)):
    layers = [tf.keras.layers.Input(shape=(in_dim,))]
    for units in hidden_units:
        layers.append(tf.keras.layers.Dense(units, activation="relu"))
    layers.append(tf.keras.layers.Dense(num_classes, activation="softmax"))
    model = tf.keras.Sequential(layers)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def quantize_int8(model, X_train_n):
    def representative_dataset():
        for row in X_train_n[:100].astype("float32"):
            yield [row.reshape(1, -1)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def train_and_evaluate(window=8, feature_set="rssi_mean_std", hidden_units=(8, 8),
                        epochs=30, verbose=0, dataset_path=DATASET):
    """Returns a result dict; does not write any files (see main() for that)."""
    df, feature_cols = load_features(window, feature_set, dataset_path)
    X_train, X_test, y_train, y_test, split_method = split_data(df, feature_cols)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std

    model = build_model(len(feature_cols), hidden_units)
    model.fit(X_train_n, y_train, validation_data=(X_test_n, y_test),
              epochs=epochs, batch_size=16, verbose=verbose)

    test_loss, test_acc = model.evaluate(X_test_n, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test_n, verbose=0), axis=1)

    cm = np.zeros((len(BUCKETS), len(BUCKETS)), dtype=int)
    for t, p in zip(y_test, y_pred):
        cm[t, p] += 1

    baseline_acc, bucket_means = baseline_threshold_accuracy(X_train, y_train, X_test, y_test)
    pcm = per_class_metrics(y_test, y_pred)

    tflite_bytes = quantize_int8(model, X_train_n)

    return {
        "feature_cols": feature_cols,
        "hidden_units": hidden_units,
        "window": window,
        "feature_set": feature_set,
        "split_method": split_method,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_acc": float(test_acc),
        "baseline_acc": float(baseline_acc),
        "confusion_matrix": cm,
        "per_class_metrics": pcm,
        "tflite_size": len(tflite_bytes),
        "tflite_bytes": tflite_bytes,
        "model": model,
        "mean": mean,
        "std": std,
    }


def main():
    result = train_and_evaluate(verbose=2)

    print(f"\nSplit method: {result['split_method']} "
          f"(train={result['n_train']}, test={result['n_test']})")
    print(f"Held-out test accuracy: {result['test_acc']:.4f}")
    print(f"Baseline (nearest-mean RSSI threshold) accuracy: {result['baseline_acc']:.4f}")
    print(f"Model improvement over baseline: "
          f"{(result['test_acc'] - result['baseline_acc']) * 100:+.2f} points")

    print("\nConfusion matrix (rows=true, cols=pred), order near/mid/far:")
    print(result["confusion_matrix"])

    print("\nPer-class precision/recall/F1:")
    for name, m in result["per_class_metrics"].items():
        print(f"  {name:5s} precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"f1={m['f1']:.3f} support={m['support']}")

    out_path = "model_int8.tflite"
    with open(out_path, "wb") as f:
        f.write(result["tflite_bytes"])
    print(f"\nWrote {out_path} ({result['tflite_size']} bytes)")

    with open("normalization.txt", "w") as f:
        f.write(f"feature_cols = {result['feature_cols']}\n")
        f.write(f"mean = {result['mean'].tolist()}\n")
        f.write(f"std = {result['std'].tolist()}\n")

    return result


if __name__ == "__main__":
    main()
