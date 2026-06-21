# ChokePoint — Technical Reference

Two processes:

- `src/prepare.py` — batch job. Reads the CSV, cleans it, scores impact, trains
  the forecaster, writes everything to `cache/`. Run once after the data is in
  place, and again whenever `engine.py` or `prepare.py` changes.
- `src/server.py` — FastAPI app. Loads `cache/` at startup, serves a read-only
  JSON API and the single-file UI at `app/index.html`. No database. On startup
  it pre-warms two held-out validation aggregates (geo + station) in a daemon
  thread so the forecast tabs never block on a cold compute.

`src/engine.py` holds the logic shared by both: cleaning, scoring, geohash,
naming, windows, holidays, the learning-to-rank model, and the **geo layer**
(zones and per-station corners).

---

## 1. Dataset

Source CSV (anonymised) ships as `data/jan to may police violation_anonymized791b166.zip`.
Unzip and rename the CSV to `data/violations.csv` before running `prepare.py`.

- Span 2023-11-10 to 2024-04-08, 151 days.
- 298,450 raw rows, 243,405 events after cleaning, 54 police stations.
- Training cut `2024-03-01` (`engine.CUT`, overridable via `CHOKEPOINT_CUT`).
  The forecaster trains only on panel rows with `date < cut`. Dates on or after
  the cut are held out. `/api/init` returns only held-out dates that also have a
  next day in range as `holdout_dates`; the UI restricts selection to those.
  This held-out window is **38 days** for the shipped data.

Columns read from the CSV:

| Column | Use |
|---|---|
| `created_datetime` | parsed as UTC → Asia/Kolkata → `date`, `hour`, `minute`, `dow` |
| `latitude`, `longitude` | numeric → geohash-7 (`gh7`), geohash-6 (`gh6`); `gh5 = gh7[:5]` |
| `location` | free-text address → display name and road-class multiplier |
| `vehicle_type` | footprint weight and vehicle class |
| `offence_code` | severity weight and primary offence label |
| `police_station` | the deployment unit (`ps`) — the WHO layer |
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
   `gh6 = gh7[:6]`. `gh5 = gh7[:5]` is derived where the geo layer needs it.
6. Compute `impact`, `poff`, `vclass` (below).

`prepare.py` (and `predict_upload`) additionally keep only rows with a non-null
`police_station`, set `ps = str(police_station)` and `w = window_of(hour)`.

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
`main road`, `ring road`, `flyover`, `underpass`, `junction`; else `1.0`.

`poff` (primary offence, for display) is the highest-severity code present.
`vclass` is the footprint bucket name.

---

## 4. Geohash and naming

- **gh7** (~150 m) — corner grain, used for the obstruction map and the
  station-scope forecast.
- **gh6** (~1.2 km) — intermediate.
- **gh5** (~5 km) — zone grain, the city WHERE layer.

`engine.name_maps` assigns each `gh7` a display name from the modal `location`
string in that cell. `engine._geo_label` names a zone/corner by the
highest-impact junction in it (strips BTP codes, relabels "Unnamed Road",
collapses whitespace). Station centroids are the mean lat/lon of a station's
events; nearest neighbours are the up-to-4 stations within 5 km
(`cache/stations.json`).

---

## 5. Grains and windows

ChokePoint works on three nested grains:

| Grain | Unit | Layer | Where used |
|---|---|---|---|
| **gh5 zone** (~5 km) | geohash-5 cell | 📍 WHERE (city) | Tomorrow's forecast (All) |
| **station** | police station (`ps`) | 👮 WHO | deployment, ranker, routing |
| **gh7 corner** (~150 m) | geohash-7 cell / junction | 🎯 EXACT | corners, station-scope forecast |

Deployment timing uses **2-hour windows** (`engine.window_of`,
`engine.WINDOW_LABELS`), covering 06:00–20:00:

| w | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| label | 06-08 | 08-10 | 10-12 | 12-14 | 14-16 | 16-18 | 18-20 |

Hours outside 06:00–20:00 map to `w = -1` and are excluded from the panel.

---

## 6. WHO layer — the learning-to-rank forecaster

Panel (`cache/stpanel.parquet`): the full cross-product of `station × date ×
window` over the span, summed `impact` per cell (0 where none), plus features:
`psid` (station code), `w`, `dow`, `is_wknd`, `is_hol`, `lag1`, `lag2`, `lag7`,
`roll7`, `roll7max`, `zw_mean`, `dow_mean`, `regime_mean`.

Targets: `impact` (next-day horizon) and `fwd3` = forward 3-day rolling mean
(outlook horizon). Relevance grading per `(date, w)`: top-5 = 3, top-10 = 2,
top-15 = 1, else 0.

Model: `lightgbm.LGBMRanker(objective="lambdarank", n_estimators=500,
learning_rate=0.05, label_gain=[0,1,3,7], random_state=0)`, grouped by
`(date, w)`, trained on `date < cut`. Two models are pickled: `ranker_day`
(target `impact`) and `ranker_outlook` (target `fwd3`).

At query time the day-level station ranking is ordered by summed `roll7` across
windows; the ranker scores the per-window (shift) ordering. Held-out station
performance (38 days): **top-10 hit ≈ 6.8 / 10**, top-10 capture ≈ 57.7%,
top-20 capture ≈ 75.9%.

---

## 7. WHERE / EXACT layers — the geo forecast

The geo layer is the headline. It is a **rank-normalized blend**, not the
LightGBM ranker — and a key honesty point is that the signal at this grain is so
stable that even a plain historical-total ranking lands near the same number.

`engine.geo_rank(hist, T, col="gh5")` returns a score per cell:

```
score = 0.35 * chronic        # all-history total, rank-normalized
      + 0.30 * recent_28d      # last 28 days
      + 0.20 * recent_7d       # last 7 days
      + 0.15 * day_of_week     # same weekday history
```

each term rank-normalized to (0, 1] over the cell index.

`engine.geo_forecast(df, date, K=25)` — the **WHERE** forecast (city):
- ranks gh5 zones with `geo_rank`, takes the top `K`;
- builds per-zone metadata (`geo_zone_meta`): centroid pin, plain label, and the
  covering station team(s) with share ≥ 12% (duplicate labels disambiguated);
- if the next day exists in `df`, scores it: `cap@K` (share of the next day's
  load inside the K zones), `oracle@K` (best possible), and a per-zone
  busy/quiet `hit`.
- Returns `{next, K, n_zones, cap, oracle, hit, has_actual, zones[], scope:"city"}`.

Held-out (38 unseen days): **cap@25 ≈ 95.8% on average**, oracle@25 ≈ 98.3%,
worst day ≈ 88.6%.

`engine.station_forecast(df, date, station, K=8)` — the **EXACT** forecast
(station drill). Same shape as `geo_forecast`, one grain finer: it filters to the
station and ranks **gh7 corners** inside it. Returns the same fields plus
`scope:"station", station`. Held-out per-station corner cap averages ~46%
(consistent with the city EXACT-corner tier).

`engine.corner_check(df, date, stations, topk=3)` — for the WHO worked example:
inside each flagged station, how much of its load its named top-`k` corners
captured. Returns `{avg_capk, per_station, top_detail, k}`.

**Held-out aggregates (cached, pre-warmed at startup):**

- `server.geo_agg(K=25)` — 38-day geo validation: `{n, avg_cap, avg_oracle,
  avg_hit, worst, series[]}`.
- `server.agg_validation()` — 38-day station validation: `{n, avg_hit,
  avg_cap10, avg_cap20, avg_oracle10, avg_corner, worst, series[]}`.
- `server.station_agg(station, K)` — per-station 38-day corner validation,
  computed lazily on first selection and cached: `{n, avg_cap, avg_oracle,
  avg_hit, worst, series[]}`.

In the UI, the "next-day check" tiles show the **selected day's** number with the
matching 38-day average beside it (the sparkline shows the full 38-day spread),
so the result is responsive to the date without cherry-picking.

---

## 8. Cache artifacts (`cache/`)

| File | Contents |
|---|---|
| `events.parquet` | cleaned events: `date, hour, minute, w, dow, gh7, gh6, lat, lon, impact, poff, vclass, device_id, created_by_id, location, ps` |
| `stpanel.parquet` | station×date×window panel with all features + `impact`, `roll7`, `fwd3` |
| `ranker.pkl` | `{ranker_day, ranker_outlook, features, psid_map, cut, wins}` |
| `names.json` | `{gh7: {name, area, lat, lon}, gh6: {gh: name}}` |
| `stations.json` | `{centroids: {ps: {lat, lon, n}}, neighbours: {ps: [[other, km], ...]}}` |
| `static.json` | `meta`, `stations[]`, `profile{ps:{w:mean}}`, `holiday_watch[]`, `holidays{}`, `reveal{...}`, `dates[]` |

The geo zones, station corners, and the held-out aggregates are computed at
runtime from `events.parquet` (and cached in-process), not pre-baked into
`cache/`.

---

## 9. API reference

All responses JSON. `station` defaults to `All` and scopes the result to one
police station when set. Dates are `YYYY-MM-DD`. Several endpoints return a
`scope` field (`"city"` / `"station"` / `"day"`) so the UI can adapt wording.

### GET `/api/init`
Startup metadata: `dates[]`, `holdout_dates[]` (selectable), `stations[]`,
`train_span`, `n_train`, `n_holdout`, `cut`, `span`, `records`, `events`,
`holidays{}`, `trend[{date,events}]`.

### GET `/api/day?date=&station=`
`date`, `station`, `hotspots[]`, `charts{...}`, `kpis{events, hotspots,
top10_share, peak_hour, is_holiday, holiday_name}`, `holiday_watch[]`. Each
hotspot: `gh, name, area, lat, lon, cis, cis_norm, tickets, coverage, gap, ps,
peak, offmix, vehmix, hours[24]`. **Honors `station`.**

### GET `/api/reveal?date=&station=`
Patrol-vs-reality for the day. **City (`station=All`):** cross-station
reallocation — `naive[]` (most-ticketed stations), `real[]` (highest impact per
coverage), `overlap`, `n_missed`, `n_over`, `moves[]`, `covered_pct`,
`scope:"day"`. **Single station:** within-station, corner-level — same shape with
spot-level `naive/real/moves` and `scope:"station"`. Each move: `frm, to, gain,
pct, frm_lat, frm_lon, to_lat, to_lon, to_spots[]`.

### GET `/api/geo?date=&k=&station=`
The **WHERE** forecast. **City:** top-`k` gh5 zones + `agg` = `geo_agg()`,
`scope:"city"`. **Single station:** top-8 gh7 corners inside the station + `agg`
= `station_agg(station)`, `scope:"station", station`. Both return
`{next, K, n_zones, cap, oracle, hit, has_actual, zones[], agg}`. Each zone:
`{zone, label, lat, lon, stations[{station,share}], score, actual, hit}`.

### GET `/api/forecast?date=&horizon=&station=`
`horizon` ∈ {`day`, `outlook`}. Predicts the next day (WHO layer).
`next`, `is_holdout`, `windows[{w,label,items[]}]`,
`day_pred{hit, cap10, cap20, oracle10, items[{station, pred, actual, predicted,
correct, lat, lon}], pred_list[], spots{...}, corner_check{...}}`,
`val_agg` (= `agg_validation()`), `geo_agg` (= `geo_agg()`). **Honors `station`**
for the per-window items.

### GET `/api/accuracy`
Returns `agg_validation()` directly (the 38-day station aggregate).

### GET `/api/schedule?date=&teams=&station=`
Team assignment per window for the next day (`teams` default 10).
`empty`, `blocks[{w, label, assign[{station, lat, lon, covers[]}]}]`, `teams`,
`next`, `is_holdout`. **Honors `station`.**

### GET `/api/playbook?date=&station=`
Per-station shift grid for the next day. `empty`, `next`, `is_holdout`, `focus`
(the selected station, or null), `windows[{w,label}]`, `rows[]`,
`cells{station:{w:{s, t, load}}}`, `pairs[{w, label, pairs[{need, helper, dist}]}]`,
`spots{station:[...]}`. `s` ∈ `need` / `hold` / `help` (`t` = destination) /
`routine`. **When a station is selected**, `rows` is that station **plus the
partner teams** that help it (or that it helps), and `focus` names it — so the
help relationships stay visible instead of collapsing to one row.

### GET `/api/replay?date=&station=`
Day replay in 30-min steps (06:00–20:00) + an end-of-day summary.
`buckets[]`, `total`, `next_day`, `next_pred[]`, `timeline[]`, `summary`.
**Honors `station`.**

### POST `/api/predict_upload`
Multipart: `file` (history CSV), optional `actual` (next-day CSV). Forecasts the
day after the uploaded history using the shipped model **and** the geo layer.
Returns `ok`, `next`, `n_days`, `span`, `stations`, `unknown[]`,
`enough_history`, `day_pred{items[], pred_list[], spots{}}` (WHO),
`windows[]` (per-shift), `geo{next, cap, oracle, has_actual, n_zones, zones[]}`
(the WHERE zones from your data — a pure prediction unless `actual` is supplied,
in which case the zones are scored), and `accuracy{...}` when `actual` is given.

### POST `/api/ask`
Body `{q, date, station}` → `{action{tab, spot?}, answer, follow[]}`. A
deterministic keyword router selects the answer and the screen; if
`OPENAI_API_KEY` is set, the answer is rephrased by `gpt-4o-mini` (the router
still decides content and action).

### GET `/`
Serves `app/index.html`.

---

## 10. Run and operate (local)

```
unzip "data/jan to may police violation_anonymized791b166.zip" -d data
mv "data/jan to may police violation_anonymized791b166.csv" data/violations.csv
pip install -r requirements.txt
python src/prepare.py
cd src && uvicorn server:app --port 8000
```

- CSV path override: set `CSV` before `prepare.py`.
- Train/forecast split: set `CHOKEPOINT_CUT` in the **same shell** for both
  `prepare.py` and `uvicorn` (see INTEGRATION.md).
- Assistant LLM: set `OPENAI_API_KEY` before `uvicorn` (optional).
- After editing `server.py`, restart uvicorn. After editing `engine.py` or
  `prepare.py`, re-run `prepare.py` then restart uvicorn. After editing
  `app/index.html`, hard-refresh.

Dependencies: `pandas, numpy, pygeohash, lightgbm, pyarrow, fastapi, uvicorn,
python-multipart, openai`.

---

## 11. Scope and limits

- Impact is derived from enforcement records, not measured traffic flow;
  "congestion" is an obstruction proxy.
- Every accuracy figure is measured on held-out dates (`date >= cut`) the model
  never trained on. The geo ~95.8% is **coverage of the enforcement-load proxy**
  in the 25 zones — never "congestion reduced".
- A zone routes to one or more stations (central zones fan out); this is shown
  openly rather than forced 1:1.
- No live ingestion; `prepare.py` runs on the historical CSV and Day replay is a
  replay of a stored day.
- Single process, in-memory artifacts, no auth, no rate limiting.
- The assistant answers a fixed set of intents; unmatched questions return a
  capability prompt.
