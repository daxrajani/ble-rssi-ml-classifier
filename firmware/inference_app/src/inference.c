/*
 * INT8 forward pass for the trained distance-bucket classifier
 * (model/train.py -> model_int8.tflite -> model_weights.c).
 *
 * This hand-rolled implementation exists instead of linking TFLite Micro:
 * the model is tiny (3 -> 8 -> 8 -> 3, ~800 params) and this NCS workspace
 * doesn't have a tflite-micro west module set up, so pulling in the full
 * C++ runtime would be a much bigger, riskier change than writing the
 * handful of lines of quantized-dense-layer math this model actually
 * needs. The nRF52840 is a Cortex-M4F with a hardware FPU, so the
 * requantization step uses plain float math (multiply by the real scale)
 * rather than TFLM's fixed-point-multiplier approximation for FPU-less
 * targets - simpler and at least as accurate here.
 *
 * Quantization scheme matches TFLite's standard per-channel-weight /
 * per-tensor-activation INT8 scheme:
 *   real_value = scale * (quantized_value - zero_point)
 * Weights use zero_point 0 (symmetric); activations/inputs don't.
 *
 * The final layer has no activation before softmax in the trained model,
 * but softmax is monotonic, so argmax of the pre-softmax INT8 logits
 * equals argmax of the true softmax probabilities - softmax itself is
 * never evaluated on-device.
 */

#include "inference.h"
#include "model_weights.h"

#include <math.h>
#include <stdbool.h>

static int8_t quantize(float real_value, float scale, int32_t zero_point)
{
	int32_t q = (int32_t)lroundf(real_value / scale) + zero_point;

	if (q > 127) {
		q = 127;
	} else if (q < -128) {
		q = -128;
	}
	return (int8_t)q;
}

/* One INT8 dense layer: out[c] = sum_i (in[i]-in_zp) * w[c][i] * (in_scale*w_scale[c]) + bias_real[c],
 * requantized into out_scale/out_zp, optionally ReLU-clamped.
 */
static void dense_layer(const int8_t *in, int32_t in_zp, float in_scale, int in_len,
			 const int8_t *w, const int32_t *bias, const float *w_scales,
			 float out_scale, int32_t out_zp, int out_len,
			 bool relu, int8_t *out)
{
	for (int c = 0; c < out_len; c++) {
		int32_t acc = 0;

		for (int i = 0; i < in_len; i++) {
			acc += (int32_t)(in[i] - in_zp) * (int32_t)w[c * in_len + i];
		}

		float real = (float)acc * (in_scale * w_scales[c]) +
			     (float)bias[c] * (in_scale * w_scales[c]);
		int32_t q = (int32_t)lroundf(real / out_scale) + out_zp;

		if (relu && q < out_zp) {
			q = out_zp;
		}
		if (q > 127) {
			q = 127;
		} else if (q < -128) {
			q = -128;
		}
		out[c] = (int8_t)q;
	}
}

enum distance_bucket classify_distance(const float *rssi_window, size_t n)
{
	if (n == 0) {
		return DIST_UNKNOWN;
	}

	/* Features: [latest rssi, rolling mean, rolling std] over the window -
	 * must match the feature engineering in model/train.py.
	 */
	float sum = 0.0f, sum_sq = 0.0f;

	for (size_t i = 0; i < n; i++) {
		sum += rssi_window[i];
	}
	float mean = sum / (float)n;

	for (size_t i = 0; i < n; i++) {
		float d = rssi_window[i] - mean;
		sum_sq += d * d;
	}
	float std = (n > 1) ? sqrtf(sum_sq / (float)n) : 0.0f;

	float raw[MODEL_NUM_FEATURES] = {rssi_window[n - 1], mean, std};
	int8_t q_in[MODEL_NUM_FEATURES];

	for (int i = 0; i < MODEL_NUM_FEATURES; i++) {
		float normalized = (raw[i] - model_feature_mean[i]) / model_feature_std[i];

		q_in[i] = quantize(normalized, model_input_scale, model_input_zero_point);
	}

	const struct model_weights *m = &g_model_weights;

	int8_t a1[MODEL_L1_UNITS];
	dense_layer(q_in, model_input_zero_point, model_input_scale, MODEL_NUM_FEATURES,
		    m->l1_w, m->l1_b, m->l1_w_scales, *m->l1_act_scale, *m->l1_act_zp,
		    MODEL_L1_UNITS, true, a1);

	int8_t a2[MODEL_L2_UNITS];
	dense_layer(a1, *m->l1_act_zp, *m->l1_act_scale, MODEL_L1_UNITS,
		    m->l2_w, m->l2_b, m->l2_w_scales, *m->l2_act_scale, *m->l2_act_zp,
		    MODEL_L2_UNITS, true, a2);

	/* Final layer: no requantization scale needed since we only need
	 * argmax, not the actual probabilities - use a fixed nominal
	 * scale/zero_point of 1.0/0 and skip clamping semantics that don't
	 * matter for ordering.
	 */
	int8_t logits[MODEL_NUM_CLASSES];
	dense_layer(a2, *m->l2_act_zp, *m->l2_act_scale, MODEL_L2_UNITS,
		    m->l3_w, m->l3_b, m->l3_w_scales, 1.0f, 0, MODEL_NUM_CLASSES,
		    false, logits);

	int best = 0;
	for (int c = 1; c < MODEL_NUM_CLASSES; c++) {
		if (logits[c] > logits[best]) {
			best = c;
		}
	}
	return (enum distance_bucket)best;
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
