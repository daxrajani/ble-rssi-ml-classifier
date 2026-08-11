/*
 * Custom GATT service: one characteristic that notifies subscribers with
 * the latest distance_bucket classification (single byte: 0=near, 1=mid,
 * 2=far, 3=unknown). Vendor-specific 128-bit UUIDs (randomly generated for
 * this project - regenerate your own if you fork this for something else).
 */

#include "ble_result_svc.h"

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/gatt.h>
#include <zephyr/bluetooth/uuid.h>

/* 6e9b0001-... */
#define BT_UUID_RESULT_SVC_VAL \
	BT_UUID_128_ENCODE(0x6e9b0001, 0xb0f0, 0x4a6a, 0x9c9a, 0x000000000000)
#define BT_UUID_RESULT_CHRC_VAL \
	BT_UUID_128_ENCODE(0x6e9b0002, 0xb0f0, 0x4a6a, 0x9c9a, 0x000000000000)

static struct bt_uuid_128 result_svc_uuid = BT_UUID_INIT_128(BT_UUID_RESULT_SVC_VAL);
static struct bt_uuid_128 result_chrc_uuid = BT_UUID_INIT_128(BT_UUID_RESULT_CHRC_VAL);

static uint8_t last_bucket = DIST_UNKNOWN;
static bool notify_enabled;

static void result_ccc_cfg_changed(const struct bt_gatt_attr *attr, uint16_t value)
{
	ARG_UNUSED(attr);
	notify_enabled = (value == BT_GATT_CCC_NOTIFY);
}

static ssize_t read_result(struct bt_conn *conn, const struct bt_gatt_attr *attr,
			    void *buf, uint16_t len, uint16_t offset)
{
	ARG_UNUSED(conn);
	return bt_gatt_attr_read(conn, attr, buf, len, offset, &last_bucket, sizeof(last_bucket));
}

BT_GATT_SERVICE_DEFINE(result_svc,
	BT_GATT_PRIMARY_SERVICE(&result_svc_uuid),
	BT_GATT_CHARACTERISTIC(&result_chrc_uuid.uuid,
				BT_GATT_CHRC_READ | BT_GATT_CHRC_NOTIFY,
				BT_GATT_PERM_READ,
				read_result, NULL, &last_bucket),
	BT_GATT_CCC(result_ccc_cfg_changed, BT_GATT_PERM_READ | BT_GATT_PERM_WRITE),
);

static const struct bt_data ad[] = {
	BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
	BT_DATA(BT_DATA_NAME_COMPLETE, CONFIG_BT_DEVICE_NAME, sizeof(CONFIG_BT_DEVICE_NAME) - 1),
};

int ble_result_svc_start(void)
{
	int err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, ad, ARRAY_SIZE(ad), NULL, 0);

	if (err) {
		printk("Advertising failed to start (err %d)\n", err);
		return err;
	}
	printk("Advertising result service as \"%s\"\n", CONFIG_BT_DEVICE_NAME);
	return 0;
}

void ble_result_svc_notify(enum distance_bucket b)
{
	last_bucket = (uint8_t)b;

	if (!notify_enabled) {
		return;
	}
	bt_gatt_notify(NULL, &result_svc.attrs[1], &last_bucket, sizeof(last_bucket));
}
