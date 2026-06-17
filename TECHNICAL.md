# ChokePoint — Technical Reference

Two processes:

- `src/prepare.py` — batch job. Reads the CSV, cleans it, scores impact, trains
  the forecaster, writes everything to `cache/`. Run once after the data is in
  place, and again whenever `engine.py` or `prepare.py` changes.
- `src/server.py` — FastAPI app. Loads `cache/` at startup, serves a read-only
  JSON API and the single-file UI at `app/index.html`. No database.

`src/engine.py` holds the logic shared by both (cleaning, scoring, geohash,
naming, windows, holidays).

---

## 1. Dataset

Source CSV (anonymised) ships as `data/jan to may police violation_anonymized791b166.zip`.
Unzip and rename the CSV to `data/violations.csv` before running `prepare.py`.

- Span 2023-11-10 to 2024-04-08, 151 days.
- 298,450 raw rows, 243,405 events after cleaning, 54 police stations.
- Training cut `2024-03-01` (`engine.CUT`). The forecaster is trained only on
  panel rows with `date < 2024-03-01`. Dates on or after the cut are held out.
  `/api/init` returns only held-out dates that also have a next day in range as
  `holdout_dates`; the UI restricts selection to those.

Columns read from the CSV:

| Column | Use |
|---|---|
| `created_datetime` | parsed as UTC, converted to Asia/Kolkata → `date`, `hour`, `minute`, `dow` |
| `latitude`, `longitude` | numeric → geohash-7 (`gh7`) and geohash-6 (`gh6`) |
| `location` | free-text address → display name and road-class multiplier |
| `vehicle_type` | footprint weight and vehicle class |
| `offence_code` | severity weight and primary offence label |
| `police_station` | forecast/deployment unit (`ps`) |
| `device_id` | distinct devices → patrol-coverage proxy (`ncov`) |
| `created_by_id` | retained in events |
| `vehicle_number` | sweep-burst dedup key (dropped from modelling) |
| `validation_status` | rows with `rejected` / `duplicate` dropped for impact |

---

## 2. Cleaning (`engine.load_clean`)

In order:

1. Read all columns as strings, treat `""`, `NULL`, `null`, `NaN` as missing.
2. Parse `created_datetime` as UTC, drop unparseable rows, convert to IST. Derive
   `date` (midnight IST), `hour`, `minute`, `dow`.
3. Keep a raw copy (with `rejected` flag) for the records count; for everything
   else drop `validation_status in {rejected, duplicate}`.
4. Sweep-burst dedup: drop duplicates on `(vehicle_number, latitude, longitude,
   minute-floor)`.
5. Drop rows with non-numeric lat/lon. Encode `gh7 = geohash(lat, lon, 7)`,
   `gh6 = gh7[:6]`.
6. Compute `impact`, `poff`, `vclass` (below).

`prepare.py` additionally keeps only rows with a non-null `police_station`, sets
`ps = str(police_station)` and `w = window_of(hour)`.

---

## 3. Impact score (CIS)

```
impact = footprint(vehicle_type) * severity(offence_code) * road_multiplier(location)
```

Footprint (`engine.footprint`):

| Class | Match (substring, upper-cased) | Weight |
|---|---|---|
| Heavy | BUS, HGV, LORRY, TANKER, TRUCK | 5.0 |
| Car/Medium | LGV, TEMPO, GOODS, VAN, MAXI, JEEP, CAR | 3.0 |
| Two-wheeler/Auto | SCOOTER, MOTOR, MOPED, CYCLE, AUTO | 1.0 |
| (anything else) | — | 2.0 |

Severity (`engine.SEV`, by offence code; max over codes present, default 1):

| Code | Offence | Severity |
|---|---|---|
| 107 | On a main road | 4 |
| 109 | Double parking | 4 |
| 104 | Near road crossing | 3 |
| 108 | Opposite parked vehicle | 3 |
| 111 | Near bus-stop/school/hospital | 3 |
| 105 | On footpath | 2 |
| 112 | Wrong parking | 2 |
| 113 | No parking | 2 |
| 116 | Defective plate | 0 |

Road multiplier (`engine.roadmult`): `1.5` if `location` contains any of
`main road`, `ring road`, `flyover`, `underpass`, `junction`; otherwise `1.0`.

`poff` (primary offence, for display) is the code with the highest severity among
those present. `vclass` is the footprint bucket name.

---

## 4. Geohash and naming

Map detail uses geohash-7 cells. `engine.name_maps` assigns each `gh7` a display
name from the modal `location` string in that cell: `engine.pick_name` splits the
address on commas, prefers a segment matching a road/place keyword
(`ROAD_KW`), strips a leading house number, and truncates to 36 chars (plus a
secondary `area` of 28 chars). Station centroids are the mean lat/lon of a
station's events; nearest neighbours are the up-to-4 stations within 5 km.

---

## 5. Grain and windows

Forecast and deployment grain is **police-station × 2-hour window**. Windows
(`engine.window_of`, `engine.WINDOW_LABELS`), covering 06:00–20:00:

| w | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| label | 06-08 | 08-10 | 10-12 | 12-14 | 14-16 | 16-18 | 18-20 |

Hours outside 06:00–20:00 map to `w = -1` and are excluded from the panel and the
forecast.

---

## 6. Forecaster

Panel (`cache/stpanel.parquet`): the full cross-product of `station × date ×
window` over the data span, with summed `impact` per cell (0 where none), plus:

| Feature | Definition |
|---|---|
| `psid` | station category code |
| `w` | window index 0–6 |
| `dow` | day of week |
| `is_wknd` | dow ≥ 5 |
| `is_hol` | date in `engine.HOLIDAYS` |
| `lag1`, `lag2`, `lag7` | impact for the same station/window 1, 2, 7 days back |
| `roll7` | mean of the prior 7 days (same station/window) |
| `roll7max` | max of the prior 7 days |
| `zw_mean` | expanding mean to date (same station/window) |
| `dow_mean` | expanding mean for that station/window/day-of-week |
| `regime_mean` | expanding mean for that station/window/weekend-flag |

Targets: `impact` (next-day horizon) and `fwd3` = forward 3-day rolling mean
(outlook horizon). Relevance grading per `(date, w)` group: top-5 = 3, top-10 = 2,
top-15 = 1, else 0.

Model: `lightgbm.LGBMRanker(objective="lambdarank", n_estimators=500,
learning_rate=0.05, label_gain=[0,1,3,7], random_state=0)`, grouped by
`(date, w)`, trained on `date < 2024-03-01`. Two models are trained and pickled:
`ranker_day` (target `impact`) and `ranker_outlook` (target `fwd3`).

At query time the station ranking shown for a day is ordered by summed `roll7`
across windows; the ranker scores the per-window (shift) ordering. `/api/forecast`
also returns a held-out aggregate (`score_agg`): per-shift hit@10 for the model
vs a "tomorrow = today" persistence baseline, the day-level `where` hit@10, and
top-20 concentration, averaged over all held-out next-days.

---

## 7. Cache artifacts (`cache/`)

| File | Contents |
|---|---|
| `events.parquet` | cleaned events; columns `date, hour, minute, w, dow, gh7, gh6, lat, lon, impact, poff, vclass, device_id, created_by_id, location, ps` |
| `stpanel.parquet` | station×date×window panel with all features + `impact`, `roll7`, `fwd3` |
| `ranker.pkl` | `{ranker_day, ranker_outlook, features, psid_map, cut, wins}` |
| `names.json` | `{gh7: {name, area, lat, lon}, gh6: {gh: name}}` |
| `stations.json` | `{centroids: {ps: {lat, lon, n}}, neighbours: {ps: [[other, km], ...]}}` |
| `static.json` | `meta`, `stations[]`, `profile{ps:{w:mean}}`, `holiday_watch[]`, `holidays{}`, `reveal{naive,real,overlap,n_missed,n_over}`, `dates[]` |

---

## 8. API reference

All responses JSON. `station` defaults to `All` and scopes the result to one
police station when set. Dates are `YYYY-MM-DD`.

### GET `/api/init`
Startup metadata.
`dates[]`, `holdout_dates[]` (selectable), `stations[]`, `train_span`, `n_train`,
`n_holdout`, `cut`, `span`, `records`, `events`, `holidays{}`, `trend[{date,events}]`.

### GET `/api/day?date=&station=`
`date`, `station`, `hotspots[]`, `charts{hourly[24], vehicle[], offence[]}`,
`kpis{events, hotspots, top10_share, peak_hour, is_holiday, holiday_name}`,
`holiday_watch[]`.
Each hotspot: `gh, name, area, lat, lon, cis, cis_norm (0-100), tickets,
coverage, gap, ps, peak (hour), offmix[[label,%]], vehmix[[label,%]], hours[24]`.

### GET `/api/reveal?date=&station=`
Patrol-vs-reality for the day.
`naive[{name,lat,lon}]` (most-ticketed stations), `real[...]` (highest impact per
coverage), `overlap`, `n_missed`, `n_over`, `moves[]` (top 5), `covered_pct`,
`scope`, `date`. Each move: `frm, to, gain, pct, frm_lat, frm_lon, to_lat, to_lon`.

### GET `/api/forecast?date=&horizon=&station=`
`horizon` is `day` or `outlook`. Predicts the next day.
`next`, `has_next`, `is_holdout`, `hit`, `capture`, `windows[{w,label,items[]}]`,
`day_pred{hit, cap20, items[{station, pred(0-100), actual(0-100), predicted(bool),
correct(bool), lat, lon}], pred_list[]}`, `base_hit`,
`score_agg{n_days, model_shift, base_shift, edge, where_hit, cap20}`.

### GET `/api/schedule?date=&teams=&station=`
Team assignment per window for the next day. `teams` default 10.
`empty`, `blocks[{w, label, assign[{station, lat, lon, covers[]}]}]`, `teams`,
`next`, `is_holdout`, `hit`, `capture`.

### GET `/api/playbook?date=&station=`
Per-station shift grid for the next day.
`empty`, `next`, `is_holdout`, `windows[{w,label}]`, `rows[]` (active stations,
≤22), `cells{station:{w:{s, t, load}}}`, `pairs[]`.
`s` is one of `need` (gets a team), `hold` (busy, covers itself), `help` (quiet,
lends its team — `t` is the destination station), `routine`. `load` is 0-100.

### GET `/api/replay?date=&station=`
Day replay in 30-min steps (06:00–20:00) plus an end-of-day summary.
`buckets[]`, `total`, `next_day`, `next_pred[]`, `timeline[]`, `summary`.
Each bucket: `clock, events, new, pts[{name,lat,lon,v}], top_stations[],
reco, worst[{name,v,share}] (top 10), active_chokes, covered_pct`.
Each timeline beat: `clock, tone, tag, text, imp` (and `target`, `dist` for a
recommendation). `summary`: `events, locations, worst[], offences[[label,%]],
vehicles[[label,%]], peak_window`.

### POST `/api/ask`
Body `{q, date, station}`. Returns `{action{tab, spot?}, answer, follow[]}`.
A deterministic keyword router selects the answer and the screen to open; if
`OPENAI_API_KEY` is set, the answer text is rephrased by `gpt-4o-mini` (the
router still decides content and action). Intents: summarise, worst spot,
peak/when, what-to-do, same-day moves, predict/forecast, deploy/schedule,
holiday.

### GET `/`
Serves `app/index.html`.

---

## 9. Run and operate (local)

```
unzip "data/jan to may police violation_anonymized791b166.zip" -d data
mv "data/jan to may police violation_anonymized791b166.csv" data/violations.csv
pip install -r requirements.txt
python src/prepare.py
cd src && uvicorn server:app --port 8000
```

- CSV path override: set `CSV` before `prepare.py`.
- Assistant LLM: set `OPENAI_API_KEY` before `uvicorn` (optional).
- After editing `server.py`, restart uvicorn. After editing `engine.py` or
  `prepare.py`, re-run `prepare.py` then restart uvicorn. After editing
  `app/index.html`, hard-refresh.

Dependencies: `pandas, numpy, pygeohash, lightgbm, pyarrow, fastapi, uvicorn,
openai`.

---

## 10. Scope and limits

- Impact is derived from enforcement records, not measured traffic flow.
- No live ingestion; `prepare.py` runs on the historical CSV and Live mode is a
  replay of a stored day.
- Single process, in-memory artifacts, no auth, no rate limiting.
- Road class and display names are parsed from free-text `location`.
- The assistant answers a fixed set of intents; unmatched questions return a
  capability prompt.
