#ifndef INFERENCE_H_
#define INFERENCE_H_

#include <stddef.h>

enum distance_bucket {
	DIST_NEAR = 0,
	DIST_MID = 1,
	DIST_FAR = 2,
	DIST_UNKNOWN = 3,
};

/* Classify a short window of RSSI samples (most recent last) into a
 * distance bucket. `n` is between 1 and INFERENCE_WINDOW_LEN.
 */
enum distance_bucket classify_distance(const float *rssi_window, size_t n);

const char *distance_bucket_str(enum distance_bucket b);

#endif /* INFERENCE_H_ */
