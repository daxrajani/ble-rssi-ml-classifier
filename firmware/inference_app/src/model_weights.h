#ifndef MODEL_WEIGHTS_H_
#define MODEL_WEIGHTS_H_

/*
 * Auto-generated weight/quantization data for the trained distance-bucket
 * classifier (3 -> 8 -> 8 -> 3, INT8, from model/model_int8.tflite).
 * See model_weights.c and inference.c.
 *
 * Regenerate by re-running model/train.py on a bigger/better dataset, then
 * re-running the weight-extraction step documented in model/README.md.
 */

#include <stdint.h>

#define MODEL_NUM_FEATURES 3
#define MODEL_L1_UNITS 8
#define MODEL_L2_UNITS 8
#define MODEL_NUM_CLASSES 3

extern const float model_feature_mean[MODEL_NUM_FEATURES];
extern const float model_feature_std[MODEL_NUM_FEATURES];
extern const float model_input_scale;
extern const int32_t model_input_zero_point;

struct model_weights {
	const int8_t *l1_w;    /* [MODEL_L1_UNITS][MODEL_NUM_FEATURES] */
	const int32_t *l1_b;   /* [MODEL_L1_UNITS] */
	const float *l1_w_scales; /* per-output-channel weight scale, [MODEL_L1_UNITS] */
	const float *l1_act_scale;
	const int32_t *l1_act_zp;

	const int8_t *l2_w;    /* [MODEL_L2_UNITS][MODEL_L1_UNITS] */
	const int32_t *l2_b;   /* [MODEL_L2_UNITS] */
	const float *l2_w_scales;
	const float *l2_act_scale;
	const int32_t *l2_act_zp;

	const int8_t *l3_w;    /* [MODEL_NUM_CLASSES][MODEL_L2_UNITS] */
	const int32_t *l3_b;   /* [MODEL_NUM_CLASSES] */
	const float *l3_w_scales;
};

extern const struct model_weights g_model_weights;

#endif /* MODEL_WEIGHTS_H_ */
