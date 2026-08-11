#!/usr/bin/env python3
"""
Standalone training script mirroring training_notebook.ipynb, used for the
first real training pass on the collected dataset.

Note: at the time this was first run, all collected rows shared a single
session_id ("room1_run1" - near/mid/far collected back-to-back in one
session/room). The notebook's session-grouped train/test split needs >=2
sessions to do anything meaningful, so this script falls back to a
stratified random split instead - which is more optimistic than a proper
session-held-out split. Re-run with multiple session_ids (different
rooms/orientations) and switch back to GroupShuffleSplit for a trustworthy
number before quoting accuracy on a resume.
"""
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATASET = "../dataset/rssi_samples.csv"
WINDOW = 5
BUCKETS = ["near", "mid", "far"]


def main():
    df = pd.read_csv(DATASET)
    n_sessions = df["session_id"].nunique()
    print(f"Loaded {len(df)} rows, {n_sessions} session(s): {sorted(df['session_id'].unique())}")

    # Rows for each bucket were collected as one contiguous burst, so
    # grouping the rolling window by distance_bucket (rather than
    # session_id) doesn't smear across bucket boundaries here.
    df = df.sort_values(["distance_bucket", "timestamp_ms"]).reset_index(drop=True)
    grp = df.groupby("distance_bucket")["rssi"]
    df["rssi_roll_mean"] = grp.transform(lambda s: s.rolling(WINDOW, min_periods=1).mean())
    df["rssi_roll_std"] = grp.transform(lambda s: s.rolling(WINDOW, min_periods=1).std().fillna(0))

    feature_cols = ["rssi", "rssi_roll_mean", "rssi_roll_std"]
    X = df[feature_cols].values.astype("float32")
    y = df["distance_bucket"].map({b: i for i, b in enumerate(BUCKETS)}).values

    if n_sessions < 2:
        print("WARNING: only one session_id present - using a stratified random "
              "split instead of a session-held-out split. Treat this accuracy "
              "number as optimistic until multi-session data is collected.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0, stratify=y)
    else:
        from sklearn.model_selection import GroupShuffleSplit
        groups = df["session_id"].values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
        train_idx, test_idx = next(gss.split(X, y, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    print(f"train={len(X_train)} test={len(X_test)}")

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(feature_cols),)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(len(BUCKETS), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train_n, y_train, validation_data=(X_test_n, y_test),
              epochs=30, batch_size=16, verbose=2)

    test_loss, test_acc = model.evaluate(X_test_n, y_test, verbose=0)
    print(f"\nHeld-out test accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_test_n, verbose=0), axis=1)
    cm = np.zeros((len(BUCKETS), len(BUCKETS)), dtype=int)
    for t, p in zip(y_test, y_pred):
        cm[t, p] += 1
    print("Confusion matrix (rows=true, cols=pred), order near/mid/far:")
    print(cm)

    def representative_dataset():
        for row in X_train_n[:100].astype("float32"):
            yield [row.reshape(1, -1)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_int8 = converter.convert()

    out_path = "model_int8.tflite"
    with open(out_path, "wb") as f:
        f.write(tflite_int8)
    print(f"\nWrote {out_path} ({len(tflite_int8)} bytes)")

    # Save normalization stats too - needed to replicate preprocessing on-device.
    with open("normalization.txt", "w") as f:
        f.write(f"feature_cols = {feature_cols}\n")
        f.write(f"mean = {mean.tolist()}\n")
        f.write(f"std = {std.tolist()}\n")

    return test_acc, cm, len(tflite_int8)


if __name__ == "__main__":
    main()
