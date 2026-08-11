#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/bluetooth/bluetooth.h>

#include "scanner.h"
#include "leds.h"
#include "ble_result_svc.h"

int main(void)
{
	int err;

	printk("Starting RSSI distance classifier\n");

	err = leds_init();
	if (err) {
		printk("LED init failed (err %d)\n", err);
		return 0;
	}

	err = bt_enable(NULL);
	if (err) {
		printk("Bluetooth init failed (err %d)\n", err);
		return 0;
	}

	(void)ble_result_svc_start();
	(void)scanner_start();

	return 0;
}
