#ifndef SCANNER_H_
#define SCANNER_H_

#define INFERENCE_WINDOW_LEN 5

/* Set this to the beacon's MAC address as printed by scanner_app /
 * log_to_csv.py during data collection, e.g. "AA:BB:CC:DD:EE:FF". Only
 * advertisements from this address feed the classifier - everything else
 * is ignored. Leave empty to classify whatever device is heard most
 * recently (fine for a quick demo, noisier with multiple BLE devices
 * around).
 */
#define TARGET_BEACON_MAC ""

int scanner_start(void);

#endif /* SCANNER_H_ */
