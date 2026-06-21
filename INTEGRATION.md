# ChokePoint — Data Integration Guide

How to point ChokePoint at your own enforcement records and get a next-day
forecast — the **WHERE** map zones, the **WHO** police teams, and the **EXACT**
corners — and (optionally) measure how accurate it was.

This guide assumes the local setup from README.md already runs.

---

## 1. What you provide, what you get

- **You provide:** a CSV of parking-violation / enforcement events (one row per
  ticket), with a timestamp, location, vehicle type, offence code and the
  recording police station.
- **You get:** for the day after your latest data —
  - **WHERE:** the ~5 km map zones that will hold most of the load (the headline
    layer);
  - **WHO:** the police stations and 2-hour windows most likely to choke, with a
    deployment playbook;
  - **EXACT:** the named corners inside each flagged area.
- **Optional:** drop in the *actual* next-day records later to score the
  forecast (zone coverage, station hit, corner capture).

There are two ways in: the **Predict-from-your-data tab** (upload a CSV in the
browser, no retrain) and a **full retrain** (best when your stations are not the
shipped Bengaluru set). Both are below.

---

## 2. Input schema

One row per enforcement event. Column names must match (case-sensitive).

| Column | Required | Meaning |
|---|---|---|
| `created_datetime` | yes | event time, ISO-8601 UTC (e.g. `2024-04-05T09:14:00Z`) |
| `latitude` | yes | decimal degrees |
| `longitude` | yes | decimal degrees |
| `police_station` | yes | station name — the WHO unit (`ps`) |
| `vehicle_type` | yes | e.g. CAR, BUS, AUTO, SCOOTER, LORRY (footprint) |
| `offence_code` | yes | numeric offence code (severity) |
| `location` | recommended | free-text address (road name + road-class multiplier); also drives zone/corner labels |
| `validation_status` | optional | rows equal to `rejected` / `duplicate` are dropped |
| `vehicle_number` | optional | used only for sweep-burst de-duplication |
| `device_id` | optional | distinct devices per area = patrol-coverage proxy |
| `created_by_id` | optional | retained, not modelled |

Offence codes map to severity in `src/engine.py` (`SEV`); vehicle keywords in
`engine.footprint`. Edit those if your coding scheme differs. `latitude`/
`longitude` feed the geohash grid that the WHERE zones and EXACT corners are
built on, so they matter even though there is no separate "zone" column.

Anonymise before sharing — no name/phone fields are used.

---

## 3. How much history is needed

The WHO forecaster builds, per station and 2-hour window: `lag1/2/7`, `roll7`,
`roll7max`, and expanding means by day-of-week and weekend flag. The WHERE/EXACT
geo blend uses all-history + 28-day + 7-day + same-weekday totals per cell. A row
is only fully usable once its 7-day window exists.

- **To predict one day (inference):** at least **7 days** of immediately
  preceding history; **2–4 weeks recommended** so the rolling/day-of-week terms
  have samples. With less than 7 days the station ranking thins and the zone
  blend leans on chronic totals only.
- **To (re)train on your data:** the shipped model trained on ~16 weeks
  (110 days). Use **≥ 8–12 weeks** for a usable retrain; more is better.

Rule of thumb: ship the model the last **3 months**; it trains on everything
before the cut and forecasts the days after it.

---

## 4. Predict from your data — the tab (no retrain)

The quickest path, and what the **🧪 Predict from your data** tab uses.

1. Open the tab, upload your **history CSV** (schema above). Optionally also
   upload the **actual next-day CSV** to score the forecast.
2. You get, for the day after your latest record: the **📍 WHERE** zones, the
   **👮 WHO** station list (with per-shift order and corners), and — if you
   supplied the actual day — a busy-vs-quiet check on the zones.

Programmatically, the same thing is `POST /api/predict_upload` (multipart:
`file`, optional `actual`). Relevant response fields:

```json
{
  "ok": true,
  "next": "2025-05-30",
  "geo": {                         // WHERE — the map zones from your data
    "next": "2025-05-30",
    "n_zones": 41,
    "has_actual": false,           // true if you uploaded the actual day
    "cap": null, "oracle": null,   // filled (scored) when has_actual
    "zones": [
      {"label": "KR Market Junction",
       "stations": [{"station": "City Market", "share": 64}],
       "hit": null}
      // ...25 zones
    ]
  },
  "day_pred": {                    // WHO — predicted stations
    "items": [{"station": "Malleshwaram", "pred": 100, "lat": 13.00, "lon": 77.57}],
    "pred_list": ["Malleshwaram", "..."],
    "spots": { "Malleshwaram": [/* EXACT corners */] }
  },
  "windows": [ {"w": 0, "label": "06-08", "items": ["..."]} ],
  "accuracy": { "hit": 8, "of": 10, "cap20": 79, "table": [/* ... */] }  // only if actual supplied
}
```

Note: the zone forecast needs a couple of weeks of history to be meaningful —
with only a few days uploaded the `enough_history` flag is false and the zones
are thin.

---

## 5. Predict on your data — full retrain (for a new city)

Best when your stations are not the shipped Bengaluru set, so the station feature
is learned from your data.

```
# 1. Format your data to the schema above and save it as the input CSV.
cp your_data.csv data/violations.csv            # PowerShell: Copy-Item ... data\violations.csv

# 2. Choose the train/forecast cut. Everything before CHOKEPOINT_CUT is training;
#    dates on/after it are held out and become forecastable/selectable.
#    Pick ~1-2 weeks before your latest data so there are days to score.
export CHOKEPOINT_CUT=2025-05-15                 # PowerShell: $env:CHOKEPOINT_CUT = "2025-05-15"

# 3. Build the cache + train the model on your data.
python src/prepare.py

# 4. Serve.
cd src && uvicorn server:app --port 8000
```

Open http://localhost:8000 — the date selector now lists your held-out dates.
`CHOKEPOINT_CUT` must be set in the **same shell** for both `prepare.py` and
`uvicorn`, so the app and the model agree on the split.

---

## 6. Get the forecast programmatically

Two endpoints cover the layers. `DATE` is the last day you have data for; the
model predicts `DATE + 1`.

**WHERE — the map zones:**
```
GET /api/geo?date=DATE&k=25                 # city: 25 gh5 zones
GET /api/geo?date=DATE&station=Malleshwaram # one station: its gh7 corners
```
Returns `{next, cap, oracle, has_actual, zones[{label, stations[], hit}], agg}`.

**WHO — the stations / shifts:**
```
GET /api/forecast?date=DATE&horizon=day
```
`day_pred.items` where `predicted=true` are the flagged stations (ordered by
`pred`, 0–100); `windows` carries the per-shift order; full deployment grid is at
`GET /api/playbook?date=DATE`.

Export tomorrow's station plan to CSV:

```python
import csv, requests

DATE = "2025-05-29"                      # your latest day with data
r = requests.get("http://localhost:8000/api/forecast",
                 params={"date": DATE, "horizon": "day"}).json()
rows = [x for x in r["day_pred"]["items"] if x["predicted"]]

with open("next_day_plan.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rank", "station", "predicted_load", "lat", "lon"])
    for i, x in enumerate(rows, 1):
        w.writerow([i, x["station"], x["pred"], x["lat"], x["lon"]])

print(f"Forecast for {r['next']}: {len(rows)} stations -> next_day_plan.csv")
```

---

## 7. Measure accuracy against the real next day (optional)

When the real next-day records arrive, append them and re-run `prepare.py` (same
`CHOKEPOINT_CUT`), so that day stays held out. Then read the **"Was the forecast
right?"** (forecast tab) / **"Did it work?"** (deployment tab) panels in the UI,
or pull the aggregates from the API:

```python
import requests
DATE = "2025-05-29"
f = requests.get("http://localhost:8000/api/forecast",
                 params={"date": DATE, "horizon": "day"}).json()
geo = f["geo_agg"]     # 38-day held-out geo aggregate
sta = f["val_agg"]     # 38-day held-out station aggregate
print("WHERE  zone coverage (avg cap@25):", geo["avg_cap"], "%   oracle:", geo["avg_oracle"], "%")
print("WHO    station hit@10 (avg):       ", sta["avg_hit"], "/ 10")
print("EXACT  corner capture (avg):       ", sta["avg_corner"], "%")
```

Per-day numbers (for the selected date specifically) are on the `/api/geo`
response (`cap`, `oracle`, `hit`) and `/api/forecast` `day_pred`
(`hit`, `cap10`, `corner_check.avg_capk`). The UI shows the per-day number with
the 38-day average beside it.

`avg_cap` is the share of next-day load that sat in the flagged zones; `avg_hit`
is how many of the flagged top-10 stations were genuinely among the worst;
`avg_corner` is the share of each flagged area's load captured by its named
corners.

---

## 8. Run it nightly

Schedule a refresh once your data lands each day.

Linux/macOS cron (02:00 daily):

```
0 2 * * * cd /path/chokepoint && CHOKEPOINT_CUT=2025-05-15 python src/prepare.py
```

Windows Task Scheduler: run a `.ps1` that sets `$env:CHOKEPOINT_CUT`, then
`python src\prepare.py`, then your export script from §6.

---

## 9. Notes and limits

- The shipped model knows the Bengaluru station set; for a different set, retrain
  (§5) so the station feature is learned from your data. The zone/corner layer
  is geometry-based and adapts to any lat/lon without retraining.
- Impact is a deterministic estimate (vehicle size × offence severity × road
  class), not a measured traffic feed — see TECHNICAL.md §3.
- If a forecast returns few stations/zones, you likely have under 7 days of
  history (§3) — add more.
- Offence-code and vehicle-type mappings are in `src/engine.py`; adjust them to
  your coding scheme before the first run.
