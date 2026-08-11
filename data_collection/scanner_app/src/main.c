/*
 * BLE RSSI scanner for dataset collection.
 *
 * Continuously scans (no duplicate filtering, so we get one line per
 * advertisement received) and prints one CSV row per packet:
 *
 *   CSV,<uptime_ms>,<mac_address>,<rssi>
 *
 * over the DK's UART/USB-CDC console. log_to_csv.py on the host reads this
 * stream, filters to the beacon MAC being tracked in the current session,
 * and appends labeled rows to dataset/rssi_samples.csv.
 *
 * Based on the Zephyr `samples/bluetooth/observer` sample.
 */

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/hci.h>

static void device_found(const bt_addr_le_t *addr, int8_t rssi, uint8_t type,
			  struct net_buf_simple *ad)
{
	char addr_str[BT_ADDR_LE_STR_LEN];

	ARG_UNUSED(type);
	ARG_UNUSED(ad);

	bt_addr_le_to_str(addr, addr_str, sizeof(addr_str));
	printk("CSV,%lld,%s,%d\n", k_uptime_get(), addr_str, rssi);
}

static int scanner_start(void)
{
	/* Continuous active scanning, no duplicate filtering: we want a
	 * fresh sample every time an advertisement is heard, not just the
	 * first one per device.
	 */
	struct bt_le_scan_param scan_param = {
		.type     = BT_LE_SCAN_TYPE_ACTIVE,
		.options  = BT_LE_SCAN_OPT_NONE,
		.interval = BT_GAP_SCAN_FAST_INTERVAL_MIN,
		.window   = BT_GAP_SCAN_FAST_WINDOW,
	};
	int err;

	err = bt_le_scan_start(&scan_param, device_found);
	if (err) {
		printk("Start scanning failed (err %d)\n", err);
		return err;
	}

	printk("READY\n");
	return 0;
}

int main(void)
{
	int err;

	printk("Starting RSSI dataset scanner\n");

	err = bt_enable(NULL);
	if (err) {
		printk("Bluetooth init failed (err %d)\n", err);
		return 0;
	}

	(void)scanner_start();

	/* Everything happens in the bt scan callback; nothing to do here. */
	return 0;
}
