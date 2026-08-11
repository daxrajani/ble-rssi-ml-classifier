# dataset/

`rssi_samples.csv` is populated by `data_collection/log_to_csv.py` — it
starts as just a header. Columns:

| column           | meaning                                              |
|------------------|-------------------------------------------------------|
| timestamp_ms     | DK uptime (ms) when the advertisement was received     |
| mac              | beacon MAC address                                     |
| rssi             | raw RSSI (dBm) of that single advertisement            |
| distance_bucket  | `near` / `mid` / `far` — ground truth, set by you       |
| session_id       | your tag for this collection run (e.g. `room1_run2`)   |
| collected_at     | host wall-clock time the row was written                |

## Collection checklist (from the project plan)

- [ ] 0.5m, 1m, 2m, 3m, 5m physical distances, ~30-60s held at each
- [ ] 3-4 repeats per distance
- [ ] vary room/orientation between repeats — clean lab-only data overfits
      and is a weak interview answer
- [ ] 500-1000 labeled rows minimum across all buckets combined
- [ ] use a distinct `session_id` per run so the training notebook can
      split train/test by session (not by row) and avoid leakage
