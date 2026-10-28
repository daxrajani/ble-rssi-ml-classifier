/*
 * INT8 forward pass for the distance-bucket classifier.
 *
 * What this file does
 *   Runs a quantized 3 -> 8 -> 8 -> 3 fully-connected network on a Cortex-M4F
 *   (nRF52840). Input is a short window of RSSI samples; the three features
 *   (latest RSSI, window mean, window standard deviation) are standardized,
 *   quantized to INT8, pushed through two ReLU dense layers, and a final
 *   linear layer whose argmax selects NEAR / MID / FAR. Weights and
 *   quantization parameters live in model_weights.c, generated from
 *   model/model_int8.tflite by model/extract_weights.py.
 *
 * Why a dedicated implementation instead of the TFLite Micro runtime
 *   The model has 131 parameters (112 weights, 19 biases). At that size a
 *   purpose-built forward pass is smaller than the interpreter, arena
 *   allocator and kernel registry TFLM would pull in, has no C++ runtime
 *   dependency, and is short enough to verify line by line against the
 *   reference implementation.
 *
 * Quantization scheme (TFLite-compatible)
 *   real_value = scale * (quantized_value - zero_point)
 *   Weights are per-output-channel symmetric (zero_point 0). Activations and
 *   the network input are per-tensor asymmetric. Biases are INT32 with
 *   scale = input_scale * weight_scale[channel], so they fold directly into
 *   the INT32 accumulator.
 *
 * Why float requantization
 *   TFLM requantizes with fixed-point multiplier/shift pairs so it can run on
 *   cores without an FPU. The Cortex-M4F has a single-precision hardware FPU,
 *   so scaling the INT32 accumulator by the real-valued combined scale is
 *   both simpler and at least as accurate as the fixed-point approximation.
 *
 * Verification
 *   - model/verify_inference/verify_inference.py is an independent Python
 *     port of this algorithm, checked against the official TFLite
 *     interpreter on every window in the dataset.
 *   - model/verify_inference/test_main.c builds this exact file natively
 *     (it is pure C with no Zephyr dependency, deliberately) and replays the
 *     dataset through classify_distance().
 */

#include "inference.h"
#include "model_weights.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* The enum and the model must agree on the number of real classes:
 * DIST_UNKNOWN is the first value past the last class the network can emit.
 */
_Static_assert(DIST_UNKNOWN == MODEL_NUM_CLASSES,
	       "enum distance_bucket and MODEL_NUM_CLASSES have drifted apart");

/**
 * Clamp a 32-bit value into the signed 8-bit range.
 *
 * @param v  Value to saturate.
 * @return   v limited to [INT8_MIN, INT8_MAX].
 */
static int8_t saturate_int8(int32_t v)
{
	if (v > INT8_MAX) {
		return INT8_MAX;
	}
	if (v < INT8_MIN) {
		return INT8_MIN;
	}
	return (int8_t)v;
}

/**
 * Quantize a real value to INT8: q = round(real / scale) + zero_point.
 *
 * @param real_value  Value to quantize.
 * @param scale       Tensor scale; must be non-zero.
 * @param zero_point  Tensor zero point.
 * @return            Quantized value, saturated to INT8.
 */
static int8_t quantize(float real_value, float scale, int32_t zero_point)
{
	return saturate_int8((int32_t)lroundf(real_value / scale) + zero_point);
}

/**
 * Compute one output channel of a dense layer as a real (float) value:
 *   sum_i (in[i] - in_zp) * w[c][i] * (in_scale * w_scale[c])
 *     + bias[c] * (in_scale * w_scale[c])
 *
 * The combined scale is computed once per channel and applied to both the
 * accumulator and the bias.
 *
 * @param in        Input activations, in_len INT8 values.
 * @param in_zp     Input zero point.
 * @param in_scale  Input scale.
 * @param in_len    Number of inputs.
 * @param w         Weights, row-major [out_len][in_len], INT8, zero point 0.
 * @param bias      INT32 biases, one per output channel.
 * @param w_scales  Per-output-channel weight scales.
 * @param c         Output channel index to compute.
 * @return          Dequantized pre-activation value for channel c.
 */
static float dense_channel_real(const int8_t *in, int32_t in_zp, float in_scale, size_t in_len,
				const int8_t *w, const int32_t *bias, const float *w_scales,
				size_t c)
{
	const float combined_scale = in_scale * w_scales[c];
	int32_t acc = 0;

	for (size_t i = 0; i < in_len; i++) {
		acc += (int32_t)(in[i] - in_zp) * (int32_t)w[c * in_len + i];
	}

	return (float)acc * combined_scale + (float)bias[c] * combined_scale;
}

/**
 * One INT8 dense layer with INT8 output, for hidden layers.
 * Each channel is dequantized, optionally ReLU-clamped at the output zero
 * point (the quantized representation of 0.0), and requantized.
 *
 * @param in         Input activations, in_len INT8 values.
 * @param in_zp      Input zero point.
 * @param in_scale   Input scale.
 * @param in_len     Number of inputs.
 * @param w          Weights, row-major [out_len][in_len].
 * @param bias       INT32 biases, one per output channel.
 * @param w_scales   Per-output-channel weight scales.
 * @param out_scale  Output tensor scale; must be non-zero.
 * @param out_zp     Output tensor zero point.
 * @param out_len    Number of outputs.
 * @param relu       Clamp negative results to zero before saturating.
 * @param out        Destination, out_len INT8 values.
 */
static void dense_layer(const int8_t *in, int32_t in_zp, float in_scale, size_t in_len,
			const int8_t *w, const int32_t *bias, const float *w_scales,
			float out_scale, int32_t out_zp, size_t out_len,
			bool relu, int8_t *out)
{
	for (size_t c = 0; c < out_len; c++) {
		float real = dense_channel_real(in, in_zp, in_scale, in_len, w, bias, w_scales, c);
		int32_t q = (int32_t)lroundf(real / out_scale) + out_zp;

		if (relu && q < out_zp) {
			q = out_zp;
		}
		out[c] = saturate_int8(q);
	}
}

/**
 * Final dense layer, producing float logits instead of INT8 activations.
 *
 * The logits are only ever compared against each other, so they are kept as
 * dequantized floats rather than requantized to INT8. Requantizing at a
 * coarse scale would round nearby logits to the same integer (2.3 and 2.4
 * both becoming 2) and let argmax pick the wrong class.
 *
 * @param in        Input activations, in_len INT8 values.
 * @param in_zp     Input zero point.
 * @param in_scale  Input scale.
 * @param in_len    Number of inputs.
 * @param w         Weights, row-major [out_len][in_len].
 * @param bias      INT32 biases, one per output channel.
 * @param w_scales  Per-output-channel weight scales.
 * @param out_len   Number of logits.
 * @param logits    Destination, out_len floats.
 */
static void dense_layer_logits(const int8_t *in, int32_t in_zp, float in_scale, size_t in_len,
			       const int8_t *w, const int32_t *bias, const float *w_scales,
			       size_t out_len, float *logits)
{
	for (size_t c = 0; c < out_len; c++) {
		logits[c] = dense_channel_real(in, in_zp, in_scale, in_len, w, bias, w_scales, c);
	}
}

/**
 * Classify a window of RSSI samples into a distance bucket.
 *
 * Features are [latest RSSI, window mean, window standard deviation],
 * matching the feature engineering in model/train.py.
 *
 * Preconditions: samples are in chronological order (most recent last).
 * Callers keeping a circular buffer must linearize it first.
 *
 * @param rssi_window  RSSI samples in dBm, oldest first.
 * @param n            Number of samples in rssi_window.
 * @return             The predicted bucket, or DIST_UNKNOWN if rssi_window is
 *                     NULL, n is 0, or the model's feature normalization is
 *                     invalid (a zero standard deviation entry).
 */
enum distance_bucket classify_distance(const float *rssi_window, size_t n)
{
	if (rssi_window == NULL || n == 0) {
		return DIST_UNKNOWN;
	}

	float sum = 0.0f;
	float sum_sq = 0.0f;

	for (size_t i = 0; i < n; i++) {
		sum += rssi_window[i];
	}
	const float mean = sum / (float)n;

	for (size_t i = 0; i < n; i++) {
		const float d = rssi_window[i] - mean;

		sum_sq += d * d;
	}
	/* Population standard deviation (divide by n, not n - 1) to match
	 * numpy's default np.std used for the training features in model/train.py.
	 */
	const float std = (n > 1) ? sqrtf(sum_sq / (float)n) : 0.0f;

	const float raw[MODEL_NUM_FEATURES] = {rssi_window[n - 1], mean, std};
	int8_t q_in[MODEL_NUM_FEATURES];

	for (size_t i = 0; i < MODEL_NUM_FEATURES; i++) {
		if (model_feature_std[i] == 0.0f) {
			return DIST_UNKNOWN;
		}

		const float normalized = (raw[i] - model_feature_mean[i]) / model_feature_std[i];

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

	/* The trained model ends in softmax, but softmax is strictly monotonic
	 * across the class dimension, so the class with the largest pre-softmax
	 * logit is also the class with the largest probability. Taking argmax
	 * over the float logits therefore gives the same answer without
	 * evaluating exp() on-device.
	 */
	float logits[MODEL_NUM_CLASSES];
	dense_layer_logits(a2, *m->l2_act_zp, *m->l2_act_scale, MODEL_L2_UNITS,
			   m->l3_w, m->l3_b, m->l3_w_scales, MODEL_NUM_CLASSES, logits);

	size_t best = 0;

	for (size_t c = 1; c < MODEL_NUM_CLASSES; c++) {
		if (logits[c] > logits[best]) {
			best = c;
		}
	}
	return (enum distance_bucket)best;
}

/**
 * Human-readable name for a distance bucket.
 *
 * @param b  Bucket value.
 * @return   Static string: "NEAR", "MID", "FAR", or "UNKNOWN" for any other
 *           value. Never NULL.
 */
const char *distance_bucket_str(enum distance_bucket b)
{
	switch (b) {
	case DIST_NEAR: return "NEAR";
	case DIST_MID:  return "MID";
	case DIST_FAR:  return "FAR";
	default:        return "UNKNOWN";
	}
}
