/*
 * Placeholder classifier.
 *
 * TODO (replace this whole function body once you have a trained model):
 *
 *   Edge Impulse export:
 *     #include <edge-impulse-sdk/classifier/ei_run_classifier.h>
 *     - build an ei::signal_t over rssi_window (and rssi_roll_mean/std if
 *       your DSP block used them - keep the feature set identical to what
 *       you trained on)
 *     - call run_classifier(&signal, &result, false)
 *     - map result.classification[i].label / .value back to a
 *       distance_bucket
 *
 *   Manual TFLite Micro path (model_int8.tflite from training_notebook.ipynb):
 *     - convert model_int8.tflite to a C array (xxd -i) and link it in
 *     - set up a tflite::MicroInterpreter with an AllOpsResolver (or a
 *       trimmed resolver for just Dense) and a tensor arena
 *     - quantize rssi_window into the input tensor using the model's
 *       input scale/zero_point, invoke, dequantize the output tensor
 *
 * Until then this is a simple hand-tuned threshold on the mean of the
 * window, just so the rest of the firmware (scanning, LEDs, BLE notify,
 * latency measurement) is testable end-to-end before the model exists.
 */

#include "inference.h"

enum distance_bucket classify_distance(const float *rssi_window, size_t n)
{
	if (n == 0) {
		return DIST_UNKNOWN;
	}

	float sum = 0.0f;

	for (size_t i = 0; i < n; i++) {
		sum += rssi_window[i];
	}
	float mean = sum / (float)n;

	/* Placeholder thresholds - replace with the real model. Tune/remove
	 * once you've looked at where near/mid/far actually cluster in
	 * dataset/rssi_samples.csv.
	 */
	if (mean >= -60.0f) {
		return DIST_NEAR;
	} else if (mean >= -75.0f) {
		return DIST_MID;
	} else {
		return DIST_FAR;
	}
}

const char *distance_bucket_str(enum distance_bucket b)
{
	switch (b) {
	case DIST_NEAR: return "NEAR";
	case DIST_MID:  return "MID";
	case DIST_FAR:  return "FAR";
	default:        return "UNKNOWN";
	}
}
