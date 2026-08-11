#ifndef BLE_RESULT_SVC_H_
#define BLE_RESULT_SVC_H_

#include "inference.h"

/* Starts advertising a custom GATT service with one notify characteristic
 * that broadcasts the latest distance_bucket classification.
 */
int ble_result_svc_start(void);

/* Call whenever a new classification is produced; notifies any subscribed
 * central. No-op if nobody is subscribed.
 */
void ble_result_svc_notify(enum distance_bucket b);

#endif /* BLE_RESULT_SVC_H_ */
