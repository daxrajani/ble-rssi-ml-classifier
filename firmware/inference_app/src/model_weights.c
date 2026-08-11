/* Auto-generated from model_int8.tflite by model/extract_weights.py.
 * Trained model weights/quantization params, embedded directly - no TFLite Micro
 * runtime dependency. See inference.c for the matching forward pass.
 * Regenerate whenever the model is retrained on new data. */

#include "model_weights.h"

const float model_feature_mean[3] = { -73.75196075f, -73.75771332f, 1.217952847f };
const float model_feature_std[3]  = { 5.0278368f, 4.780543327f, 1.113464832f };

const float model_input_scale = 0.02214860357f;
const int32_t model_input_zero_point = -37;

/* Layer 1: 3 -> 8, ReLU */
static const int8_t l1_weights[24] = {66, 127, -111, -50, 127, 65, 100, 127, 8, -32, -88, 127, 115, 127, -57, 85, -22, -127, -69, -127, 5, 58, 127, 61};
static const int32_t l1_bias[8] = {4473, 2529, -3220, 1180, -150, 2029, 1014, -2536};
static const float l1_weight_scales[8] = {0.006268044934f, 0.008461474441f, 0.00749329431f, 0.006170054898f, 0.006724836305f, 0.006582790986f, 0.009417418391f, 0.005973126739f};
const float l1_act_scale = 0.02367488854f;
const int32_t l1_act_zero_point = -128;

/* Layer 2: 8 -> 8, ReLU */
static const int8_t l2_weights[64] = {-29, 10, 125, 24, 22, 7, 121, 127, 18, 117, 119, 75, 89, -16, -109, 127, 116, 74, -77, -41, 13, -127, 83, -98, 39, -124, 52, -81, -127, -47, 14, 122, 124, 127, 19, 62, 69, 88, -65, -65, -127, 19, -37, 35, 77, 16, 95, 35, -82, -127, -81, -24, -48, -27, 80, -73, -26, -116, 64, 63, 68, -21, 127, -14};
static const int32_t l2_bias[8] = {-22, -3055, 524, -295, 3269, 1366, -113, 753};
static const float l2_weight_scales[8] = {0.008557541296f, 0.00571972644f, 0.003897808725f, 0.004518433474f, 0.006930848584f, 0.005877620075f, 0.006229632068f, 0.005206585396f};
const float l2_act_scale = 0.04313766956f;
const int32_t l2_act_zero_point = -128;

/* Layer 3: 8 -> 3, no activation (argmax of logits == argmax of softmax) */
static const int8_t l3_weights[24] = {50, 87, -102, -81, 18, -18, -127, -49, -127, -76, 70, -11, 86, -112, -114, -11, 19, -35, -11, -38, -127, 64, 1, 88};
static const int32_t l3_bias[3] = {-1071, 1270, -507};
static const float l3_weight_scales[3] = {0.007924772799f, 0.008217817172f, 0.009911746718f};

const struct model_weights g_model_weights = {
	.l1_w = l1_weights, .l1_b = l1_bias, .l1_w_scales = l1_weight_scales,
	.l1_act_scale = &l1_act_scale, .l1_act_zp = &l1_act_zero_point,
	.l2_w = l2_weights, .l2_b = l2_bias, .l2_w_scales = l2_weight_scales,
	.l2_act_scale = &l2_act_scale, .l2_act_zp = &l2_act_zero_point,
	.l3_w = l3_weights, .l3_b = l3_bias, .l3_w_scales = l3_weight_scales,
};

