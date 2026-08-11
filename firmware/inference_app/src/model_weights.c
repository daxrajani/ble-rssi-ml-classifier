/* Auto-generated from model_int8.tflite by model/extract_weights.py.
 * Trained model weights/quantization params, embedded directly - no TFLite Micro
 * runtime dependency. See inference.c for the matching forward pass.
 * Regenerate whenever the model is retrained on new data. */

#include "model_weights.h"

const float model_feature_mean[3] = { -73.75196075f, -73.75395203f, 1.508615494f };
const float model_feature_std[3]  = { 5.0278368f, 4.66024065f, 1.242407441f };

const float model_input_scale = 0.03244039416f;
const int32_t model_input_zero_point = -71;

/* Layer 1: 3 -> 8, ReLU */
static const int8_t l1_weights[24] = {-127, 26, -81, -127, -60, 24, 21, -127, -124, -19, -127, 59, 127, 115, 47, 127, 45, -68, -127, 109, 61, -12, 127, 27};
static const int32_t l1_bias[8] = {2575, 610, 4359, 1243, -22, 884, -560, 2425};
static const float l1_weight_scales[8] = {0.007261885796f, 0.006498846225f, 0.004901304841f, 0.00774505781f, 0.008190944791f, 0.00839486625f, 0.004335574806f, 0.008916405961f};
const float l1_act_scale = 0.03368854895f;
const int32_t l1_act_zero_point = -128;

/* Layer 2: 8 -> 8, ReLU */
static const int8_t l2_weights[64] = {-73, 55, 114, 118, 15, -104, 127, 27, -127, -101, 124, 97, 7, -68, 86, -23, -127, -5, -95, 57, 57, 62, -5, 86, -30, -64, -19, 113, -70, -6, -127, 22, 48, -8, 87, -91, -30, 127, -5, 106, 7, -3, 5, -26, 91, -12, 52, 127, 120, 109, 8, 57, 33, -39, -64, -127, -112, -71, -50, 22, 32, -49, 127, -52};
static const int32_t l2_bias[8] = {153, -1051, -558, 184, 3368, 964, -422, -829};
static const float l2_weight_scales[8] = {0.005333176814f, 0.004104533698f, 0.01095979754f, 0.005632106215f, 0.007679581176f, 0.009427763522f, 0.00712406775f, 0.003964781761f};
const float l2_act_scale = 0.05367370695f;
const int32_t l2_act_zero_point = -128;

/* Layer 3: 8 -> 3, no activation (argmax of logits == argmax of softmax) */
static const int8_t l3_weights[24] = {78, -46, 127, -106, -24, 66, -115, -47, -96, 30, -82, 68, 59, 45, -127, -23, 101, 4, -79, 16, -84, -127, 98, 28};
static const int32_t l3_bias[3] = {-1039, 930, -332};
static const float l3_weight_scales[3] = {0.007908917032f, 0.009380440228f, 0.00803366024f};

const struct model_weights g_model_weights = {
	.l1_w = l1_weights, .l1_b = l1_bias, .l1_w_scales = l1_weight_scales,
	.l1_act_scale = &l1_act_scale, .l1_act_zp = &l1_act_zero_point,
	.l2_w = l2_weights, .l2_b = l2_bias, .l2_w_scales = l2_weight_scales,
	.l2_act_scale = &l2_act_scale, .l2_act_zp = &l2_act_zero_point,
	.l3_w = l3_weights, .l3_b = l3_bias, .l3_w_scales = l3_weight_scales,
};

