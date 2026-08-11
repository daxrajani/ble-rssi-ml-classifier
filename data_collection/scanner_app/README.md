# scanner_app

Zephyr BLE observer firmware for the nRF52840 DK. Scans continuously (no
duplicate filtering) and prints one CSV row per advertisement received:

```
CSV,<uptime_ms>,<mac_address>,<rssi>
```

over the DK's console UART (the onboard USB-CDC "VCOM" port), plus assorted
boot log lines that don't start with `CSV,`. `log_to_csv.py` filters those
out.

## Build & flash

You already have an nRF Connect SDK west workspace at `D:\` (zephyr, nrf,
nrfxlib, modules as siblings under `D:\.west`). This app lives inside that
workspace tree, so `west` will find it automatically.

```bash
cd D:\ble-rssi-ml-classifier\data_collection\scanner_app
west build -b nrf52840dk/nrf52840 . -d build
west flash -d build
```

If you have more than one DK plugged in, add `--` `-r` `<snr>` to `west
flash`, or unplug the others.

## Verify it's working

Open a serial terminal on the DK's COM port (115200 8N1) — e.g. `nrf
Connect for Desktop`'s Serial Terminal, PuTTY, or:

```bash
python -m serial.tools.miniterm COM<N> 115200
```

You should see `Starting RSSI dataset scanner`, then `READY`, then a stream
of `CSV,...` lines as it picks up nearby BLE advertisers (your phone, other
DKs, laptop Bluetooth, etc).
