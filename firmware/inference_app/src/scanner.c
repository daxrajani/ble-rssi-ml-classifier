#include "scanner.h"
#include "inference.h"
#include "leds.h"
#include "ble_result_svc.h"

#include <string.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/hci.h>

static float window[INFERENCE_WINDOW_LEN];
static size_t window_len;
static size_t window_pos;

static void push_sample(float rssi)
{
	window[window_pos] = rssi;
	window_pos = (window_pos + 1) % INFERENCE_WINDOW_LEN;
	if (window_len < INFERENCE_WINDOW_LEN) {
		window_len++;
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

	push_sample((float)rssi);

	uint32_t t0 = k_cycle_get_32();
	enum distance_bucket bucket = classify_distance(window, window_len);
	uint32_t t1 = k_cycle_get_32();
	uint32_t inference_us = k_cyc_to_us_floor32(t1 - t0);

	printk("RESULT,%s,%d,%s,%uus\n", addr_str, rssi, distance_bucket_str(bucket),
	       inference_us);

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
