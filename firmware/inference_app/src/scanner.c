#include "scanner.h"
#include "inference.h"
#include "leds.h"
#include "ble_result_svc.h"

#include <stdbool.h>
#include <string.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/hci.h>

/* Each nearby device gets its own RSSI window - mixing samples from
 * different physical devices into one shared window (the original,
 * simpler version of this file) corrupts the rolling mean/std features,
 * since consecutive "samples" could be from devices at completely
 * different distances. Small fixed-size table, LRU-evicted, since we're
 * only ever tracking a handful of nearby BLE devices at once.
 */
#define MAX_TRACKED_DEVICES 8

struct tracked_device {
	bt_addr_le_t addr;
	bool in_use;
	float window[INFERENCE_WINDOW_LEN];
	size_t window_len;
	size_t window_pos;
	int64_t last_seen_ms;
};

static struct tracked_device devices[MAX_TRACKED_DEVICES];

static struct tracked_device *find_or_alloc(const bt_addr_le_t *addr)
{
	struct tracked_device *oldest = NULL;

	for (int i = 0; i < MAX_TRACKED_DEVICES; i++) {
		if (devices[i].in_use && bt_addr_le_eq(&devices[i].addr, addr)) {
			return &devices[i];
		}
		if (!devices[i].in_use) {
			devices[i].in_use = true;
			devices[i].addr = *addr;
			devices[i].window_len = 0;
			devices[i].window_pos = 0;
			return &devices[i];
		}
		if (oldest == NULL || devices[i].last_seen_ms < oldest->last_seen_ms) {
			oldest = &devices[i];
		}
	}

	/* Table full: evict the least-recently-seen device. */
	oldest->addr = *addr;
	oldest->window_len = 0;
	oldest->window_pos = 0;
	return oldest;
}

static void push_sample(struct tracked_device *dev, float rssi)
{
	dev->window[dev->window_pos] = rssi;
	dev->window_pos = (dev->window_pos + 1) % INFERENCE_WINDOW_LEN;
	if (dev->window_len < INFERENCE_WINDOW_LEN) {
		dev->window_len++;
	}
}

static void device_found(const bt_addr_le_t *addr, int8_t rssi, uint8_t type,
			  struct net_buf_simple *ad)
{
	char addr_str[BT_ADDR_LE_STR_LEN];

	ARG_UNUSED(type);
	ARG_UNUSED(ad);

	bt_addr_le_to_str(addr, addr_str, sizeof(addr_str));

	if (strlen(TARGET_BEACON_MAC) > 0 &&
	    strncmp(addr_str, TARGET_BEACON_MAC, strlen(TARGET_BEACON_MAC)) != 0) {
		return;
	}

	struct tracked_device *dev = find_or_alloc(addr);

	dev->last_seen_ms = k_uptime_get();
	push_sample(dev, (float)rssi);

	uint32_t t0 = k_cycle_get_32();
	enum distance_bucket bucket = classify_distance(dev->window, dev->window_len);
	uint32_t t1 = k_cycle_get_32();
	uint32_t inference_us = k_cyc_to_us_floor32(t1 - t0);

	printk("RESULT,%s,%d,%s,%uus\n", addr_str, rssi, distance_bucket_str(bucket),
	       inference_us);

	/* LEDs/BLE-notify are a single shared indicator, so they can only
	 * ever reflect one device at a time - the one just classified
	 * (typically the most recently heard, or the sole TARGET_BEACON_MAC
	 * device if that's set).
	 */
	leds_show_bucket(bucket);
	ble_result_svc_notify(bucket);
}

int scanner_start(void)
{
	struct bt_le_scan_param scan_param = {
		.type     = BT_LE_SCAN_TYPE_ACTIVE,
		.options  = BT_LE_SCAN_OPT_NONE,
		.interval = BT_GAP_SCAN_FAST_INTERVAL_MIN,
		.window   = BT_GAP_SCAN_FAST_WINDOW,
	};
	int err = bt_le_scan_start(&scan_param, device_found);

	if (err) {
		printk("Start scanning failed (err %d)\n", err);
		return err;
	}
	printk("READY\n");
	return 0;
}
