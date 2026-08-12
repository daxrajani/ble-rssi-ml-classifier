/*
 * Native host-side test harness for firmware/inference_app/src/inference.c.
 *
 * Compiles the ACTUAL shipped inference.c + model_weights.c (no copy, no
 * reimplementation) as a standalone host executable - inference.c has zero
 * Zephyr dependencies (see its header comment), so this works unmodified.
 *
 * Reads dataset/rssi_samples.csv, replicates scanner.c's per-device
 * windowing (last up to INFERENCE_WINDOW_LEN samples, one bucket = one
 * continuous device in this offline test), and prints
 * "true_bucket,pred_bucket" per window so verify_inference.py can compute
 * agreement against the training-time expectation.
 *
 * Usage: test_inference.exe path/to/rssi_samples.csv
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../firmware/inference_app/src/inference.h"
#include "../../firmware/inference_app/src/scanner.h"

#define MAX_ROWS 8192

struct row {
	long timestamp_ms;
	float rssi;
	char bucket[8];
};

static int cmp_row(const void *a, const void *b)
{
	const struct row *ra = a, *rb = b;
	int c = strcmp(ra->bucket, rb->bucket);

	if (c != 0) {
		return c;
	}
	return (ra->timestamp_ms > rb->timestamp_ms) - (ra->timestamp_ms < rb->timestamp_ms);
}

int main(int argc, char **argv)
{
	if (argc < 2) {
		fprintf(stderr, "usage: %s rssi_samples.csv\n", argv[0]);
		return 1;
	}

	FILE *f = fopen(argv[1], "r");
	if (!f) {
		fprintf(stderr, "could not open %s\n", argv[1]);
		return 1;
	}

	static struct row rows[MAX_ROWS];
	int n = 0;
	char line[512];

	fgets(line, sizeof(line), f); /* header */
	while (fgets(line, sizeof(line), f) && n < MAX_ROWS) {
		/* timestamp_ms,mac,rssi,distance_bucket,session_id,collected_at */
		long ts;
		char mac[32], bucket[8];
		int rssi;

		int got = sscanf(line, "%ld,%31[^,],%d,%7[^,],", &ts, mac, &rssi, bucket);
		if (got != 4) {
			continue;
		}
		rows[n].timestamp_ms = ts;
		rows[n].rssi = (float)rssi;
		strncpy(rows[n].bucket, bucket, sizeof(rows[n].bucket) - 1);
		rows[n].bucket[sizeof(rows[n].bucket) - 1] = '\0';
		n++;
	}
	fclose(f);

	qsort(rows, n, sizeof(rows[0]), cmp_row);

	float window[INFERENCE_WINDOW_LEN];
	int wlen = 0, wpos = 0;
	char cur_bucket[8] = "";

	printf("true_bucket,pred_bucket\n");
	for (int i = 0; i < n; i++) {
		if (strcmp(rows[i].bucket, cur_bucket) != 0) {
			/* new bucket = new "device" in this offline replay - reset window */
			strncpy(cur_bucket, rows[i].bucket, sizeof(cur_bucket) - 1);
			cur_bucket[sizeof(cur_bucket) - 1] = '\0';
			wlen = 0;
			wpos = 0;
		}

		window[wpos] = rows[i].rssi;
		wpos = (wpos + 1) % INFERENCE_WINDOW_LEN;
		if (wlen < INFERENCE_WINDOW_LEN) {
			wlen++;
		}

		/* classify_distance expects the window ending at "now" - build a
		 * temporally-ordered copy since our circular buffer isn't.
		 */
		float ordered[INFERENCE_WINDOW_LEN];
		for (int k = 0; k < wlen; k++) {
			int idx = (wpos - wlen + k + INFERENCE_WINDOW_LEN) % INFERENCE_WINDOW_LEN;
			ordered[k] = window[idx];
		}

		enum distance_bucket pred = classify_distance(ordered, wlen);
		printf("%s,%s\n", rows[i].bucket, distance_bucket_str(pred));
	}

	return 0;
}
