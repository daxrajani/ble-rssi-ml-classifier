#include "leds.h"

#include <zephyr/drivers/gpio.h>
#include <zephyr/sys/printk.h>

/* nRF52840 DK aliases: led0..led3 */
#define LED_NEAR_NODE DT_ALIAS(led0)
#define LED_MID_NODE  DT_ALIAS(led1)
#define LED_FAR_NODE  DT_ALIAS(led2)

static const struct gpio_dt_spec led_near = GPIO_DT_SPEC_GET(LED_NEAR_NODE, gpios);
static const struct gpio_dt_spec led_mid  = GPIO_DT_SPEC_GET(LED_MID_NODE, gpios);
static const struct gpio_dt_spec led_far  = GPIO_DT_SPEC_GET(LED_FAR_NODE, gpios);

int leds_init(void)
{
	const struct gpio_dt_spec *leds[] = {&led_near, &led_mid, &led_far};

	for (size_t i = 0; i < ARRAY_SIZE(leds); i++) {
		if (!gpio_is_ready_dt(leds[i])) {
			printk("LED %zu device not ready\n", i);
			return -ENODEV;
		}
		int ret = gpio_pin_configure_dt(leds[i], GPIO_OUTPUT_INACTIVE);
		if (ret < 0) {
			return ret;
		}
	}
	return 0;
}

void leds_show_bucket(enum distance_bucket b)
{
	gpio_pin_set_dt(&led_near, b == DIST_NEAR);
	gpio_pin_set_dt(&led_mid, b == DIST_MID);
	gpio_pin_set_dt(&led_far, b == DIST_FAR);
}
