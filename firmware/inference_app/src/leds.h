#ifndef LEDS_H_
#define LEDS_H_

#include "inference.h"

/* Configures led0/1/2 (DK aliases). Returns 0 on success. */
int leds_init(void);

/* Lights the LED for `b` (led0=near, led1=mid, led2=far) and turns the
 * others off.
 */
void leds_show_bucket(enum distance_bucket b);

#endif /* LEDS_H_ */
