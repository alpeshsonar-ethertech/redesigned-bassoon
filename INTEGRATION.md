# ChokePoint — Data Integration Guide

How to point ChokePoint at your own enforcement records and get a next-day
deployment forecast, and (optionally) measure how accurate it was.

This guide assumes the local setup from README.md already runs.

---

## 1. What you provide, what you get

- **You provide:** a CSV of parking-violation / enforcement events (one row per
  ticket), with a timestamp, location, vehicle type, offence code and the
  recording police station.
- **You get:** for the day after your latest data — a ranked list of the
  stations and 2-hour windows most likely to choke, plus a deployment playbook.
- **Optional:** drop in the *actual* next-day records later to score the
  forecast (hit rate, concentration).

---

## 2. Input schema

One row per enforcement event. Column names must match (case-sensitive).

| Column | Required | Meaning |
|---|---|---|
| `created_datetime` | yes | event time, ISO-8601 UTC (e.g. `2024-04-05T09:14:00Z`) |
| `latitude` | yes | decimal degrees |
| `longitude` | yes | decimal degrees |
| `police_station` | yes | station name — the unit you deploy (`ps`) |
| `vehicle_type` | yes | e.g. CAR, BUS, AUTO, SCOOTER, LORRY (used for footprint) |
| `offence_code` | yes | numeric offence code (used for severity) |
| `location` | recommended | free-text address (road name + road-class multiplier) |
| `validation_status` | optional | rows equal to `rejected` / `duplicate` are dropped |
| `vehicle_number` | optional | used only for sweep-burst de-duplication |
| `device_id` | optional | distinct devices per area = patrol-coverage proxy |
| `created_by_id` | optional | retained, not modelled |

Offence codes map to severity in `src/engine.py` (`SEV`). If your codes differ,
edit that table. Vehicle keywords (heavy / car / two-wheeler) are in
`engine.footprint`.

Anonymise before sharing — no name/phone fields are used.

---

## 3. How much history is needed

The forecaster builds, per station and 2-hour window: `lag1`, `lag2`, `lag7`
(1/2/7 days back), `roll7`, `roll7max` (previous 7 days), and expanding means by
day-of-week and weekend flag. A row is only usable once its 7-day lag and
rolling window exist.

- **To predict one day (inference):** at least **7 days** of immediately
  preceding history per station; **2–4 weeks recommended** so most stations
  clear the 7-day window and the day-of-week means have samples. With less than
  7 days, many stations drop out and the ranking thins.
- **To (re)train on your data:** the shipped model was trained on ~16 weeks
  (110 days). Use **≥ 8–12 weeks** for a usable retrain; more is better.

Rule of thumb: ship the model the last **3 months**; it will train on everything
before the cut and forecast the days after it.

---

## 4. Predict on your data (retrain) — recommended

This trains the model on your history and forecasts the days after the cut. Best
when your stations are not the shipped Bengaluru set.

```
# 1. Format your data to the schema above and save it as the input CSV.
#    Windows PowerShell:
Copy-Item your_data.csv data\violations.csv
#    macOS/Linux:  cp your_data.csv data/violations.csv

# 2. Choose the train/forecast cut. Everything before CHOKEPOINT_CUT is training;
#    dates on/after it are held out and become forecastable/selectable.
#    Pick a date ~1-2 weeks before your latest data so there are days to score.
#    Windows PowerShell:
$env:CHOKEPOINT_CUT = "2025-05-15"
#    macOS/Linux:
export CHOKEPOINT_CUT=2025-05-15

# 3. Build the cache + train the model on your data.
python src/prepare.py

# 4. Serve.
cd src
uvicorn server:app --port 8000
```

Open http://localhost:8000 — the date selector now lists your held-out dates.
Pick the latest one to see the next-day forecast and deployment for your data.

`CHOKEPOINT_CUT` must be set in the **same shell** for both `prepare.py` and
`uvicorn`, so the app and the model agree on the split.

---

## 5. Get the prediction programmatically

The forecast for the day after `DATE` comes from one endpoint:

```
GET /api/forecast?date=DATE&horizon=day
```

`DATE` is the last day you have data for; the model predicts `DATE + 1`.

Response (relevant fields):

```json
{
  "next": "2025-05-30",
  "is_holdout": true,
  "day_pred": {
    "cap20": 78,
    "items": [
      {"station": "Malleshwaram", "pred": 100, "predicted": true,
       "correct": true, "lat": 13.00, "lon": 77.57},
      ...
    ]
  },
  "score_agg": {"where_hit": 6.8, "cap20": 79, "model_shift": 6.5, "base_shift": 5.8}
}
```

`items` where `predicted=true` are the stations the model flags for the next day,
ordered by `pred` (0–100). Export tomorrow's plan to CSV:

```python
import csv, requests

DATE = "2025-05-29"                      # your latest day with data
r = requests.get("http://localhost:8000/api/forecast",
                 params={"date": DATE, "horizon": "day"}).json()
dp = r["day_pred"]
rows = [x for x in dp["items"] if x["predicted"]]

with open("next_day_plan.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rank", "station", "predicted_load", "lat", "lon"])
    for i, x in enumerate(rows, 1):
        w.writerow([i, x["station"], x["pred"], x["lat"], x["lon"]])

print(f"Forecast for {r['next']}: {len(rows)} stations -> next_day_plan.csv")
```

Per-shift (which 2-hour window) detail is in the same response under `windows`,
and the full deployment grid is at `GET /api/playbook?date=DATE`.

---

## 6. Measure accuracy against the real next day (optional)

When the real next-day records arrive, append them to your CSV and re-run, so the
cut leaves that day held out:

```
# append the actual next-day events, then:
python src/prepare.py            # same CHOKEPOINT_CUT as before
```

Then either read it in the UI (the **"Did it work?"** panel on the Tomorrow's
deployment tab, and **"Was the forecast right?"** on the forecast tab), or pull
the numbers from the API:

```python
r = requests.get("http://localhost:8000/api/forecast",
                 params={"date": DATE, "horizon": "day"}).json()
agg = r["score_agg"]
print("where-hit @10:", agg["where_hit"], "/ 10")
print("top-20 concentration:", agg["cap20"], "%")
print("shift model vs baseline:", agg["model_shift"], "vs", agg["base_shift"])
```

`where_hit` is how many of the flagged top-10 stations were genuinely among the
worst; `cap20` is the share of next-day congestion sitting in the top-20.

---

## 7. Run it nightly

Schedule a refresh once your data lands each day.

Linux/macOS cron (02:00 daily):

```
0 2 * * * cd /path/chokepoint && CHOKEPOINT_CUT=2025-05-15 python src/prepare.py
```

Windows Task Scheduler: run a `.ps1` that sets `$env:CHOKEPOINT_CUT`, then
`python src\prepare.py`, then your export script from §5.

---

## 8. Notes and limits

- The shipped model knows the Bengaluru station set; for a different set, retrain
  (§4) so the station feature is learned from your data.
- Impact is a deterministic estimate (vehicle size × offence severity × road
  class), not a measured traffic feed — see TECHNICAL.md §3.
- If a forecast returns few stations, you likely have under 7 days of history for
  most stations (§3) — add more history.
- Offence-code and vehicle-type mappings are in `src/engine.py`; adjust them to
  your coding scheme before the first run.
