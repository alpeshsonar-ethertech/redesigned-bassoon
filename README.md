# ChokePoint — Parking Congestion Intelligence (Bengaluru Traffic Police)

Reads Bengaluru Traffic Police parking-violation records and produces, per day:
a geohash congestion map, a patrol-vs-reality gap with same-day team moves, a
next-day choke-point forecast (police-station x 2-hour-window), a deployment
playbook, and a simulated live replay. A right-side assistant answers questions
against the selected day's numbers.

## Dataset

The repository ships the anonymised source as a zip:

```
data/jan to may police violation_anonymized791b166.zip
```

Unzip it and rename the CSV inside to `violations.csv`:

```
data/violations.csv
```

- Span: 2023-11-10 to 2024-04-08 (151 days)
- Raw rows: 298,450 → 243,405 events after cleaning
- Police stations: 54
- Training cut: **2024-03-01**. The forecaster trains only on records *before*
  this date. Dates on or after it are held out, and are the **only dates the UI
  lets you select** (each has a next day available, so every prediction is
  scored against an unseen day).

## Prerequisites

- Python 3.10+
- `pip install -r requirements.txt`

## Setup and run (local)

```
# 1. data — unzip and rename the CSV to data/violations.csv
#    Windows PowerShell:
Expand-Archive "data\jan to may police violation_anonymized791b166.zip" -DestinationPath data
Rename-Item "data\jan to may police violation_anonymized791b166.csv" violations.csv
#    macOS / Linux:
#    unzip "data/jan to may police violation_anonymized791b166.zip" -d data
#    mv "data/jan to may police violation_anonymized791b166.csv" data/violations.csv

# 2. deps
pip install -r requirements.txt

# 3. build the cache (reads data/violations.csv, ~1 min, writes cache/)
python src/prepare.py

# 4. serve
cd src
uvicorn server:app --port 8000
```

Open http://localhost:8000 and hard-refresh (Ctrl+Shift+R).

The CSV path can be overridden: set `CSV` before `prepare.py`
(`$env:CSV="C:\path\file.csv"` on Windows PowerShell).

## Assistant key (optional)

The assistant works offline with a deterministic responder. To have answers
rephrased by an LLM, set a key before starting uvicorn:

```
$env:OPENAI_API_KEY = "sk-..."     # Windows PowerShell
```

Without a key the deterministic responder is used.

## Re-run rules

| You changed | Do this |
|---|---|
| `app/index.html` | hard-refresh the browser |
| `src/server.py` | restart uvicorn |
| `src/prepare.py` or `src/engine.py` | re-run `python src/prepare.py`, then restart uvicorn |

## Tabs

1. **Live mode** — simulated replay of the selected day (30-min steps), worst
   spots, key moments, end-of-day summary, next-day forecast on completion.
2. **Congestion map** — geohash hotspots for the day, with a blind-spot toggle
   (high impact, low patrol coverage).
3. **Coverage gaps (today)** — most-ticketed stations vs real choke points, and
   same-day team moves.
4. **Tomorrow's forecast** — predicted choke points for the next day and the
   predicted-vs-actual check.
5. **Tomorrow's deployment** — predicted top stations on a map, a per-station
   shift playbook, and a next-day "did it work" check.

The police-station selector in the top bar scopes every tab. The assistant rail
(desktop) becomes a button-triggered drawer on small screens; phones in portrait
get a rotate-to-landscape prompt.

## Layout

```
app/index.html      single-file UI (Bootstrap + Leaflet + Chart.js)
src/engine.py       cleaning, impact score, geohash, naming, windows, holidays
src/prepare.py      batch precompute -> cache/
src/server.py       FastAPI: read-only API + serves app/
cache/              generated artifacts (see TECHNICAL.md)
data/violations.csv input (you create this from the shipped zip)
DATA_REPORT.md      column triage and data analysis
TECHNICAL.md        data schema, scoring, model, cache, full API reference
```

## Scope

All figures come from enforcement records. Impact is an estimate (vehicle
footprint x offence severity x road class), not a live traffic feed. The
"did it work" checks compare night-before flags against the next day's actual
enforcement load on held-out dates. The base map is for display only.
