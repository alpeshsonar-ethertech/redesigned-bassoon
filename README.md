# ChokePoint — Parking Congestion Intelligence (Bengaluru Traffic Police)

**Hackathon theme — "Poor Visibility on Parking-Induced Congestion."** Today BTP
can't easily see *where* parking is choking traffic, *when* it builds, or whether
patrols are even aimed at the real problem — the signal is buried in raw
enforcement logs. ChokePoint turns those logs into visibility, then pushes one
step past it: a **next-day forecast** of where the obstruction will concentrate,
who should cover it, and the exact corners — every claim proven on days the model
never saw.

Reads Bengaluru Traffic Police parking-enforcement records and answers one
operational question: **where, and when, to send limited parking-enforcement
teams the next day** — and how well that call would have held up.

It does this on three grains, top-down by how reliable each is:

- **📍 WHERE** — the city split into fixed ~5 km map cells (geohash-5 "zones").
  A night-before forecast flags the **25 zones** that will carry most of the
  next day's enforcement load. Across 38 unseen days these 25 zones held
  **~95.8%** of the actual load on average (best any plan could do: ~98.3%).
- **👮 WHO** — each zone routes to the police **team(s)** that cover it. The
  per-station, per-shift ordering comes from a learning-to-rank model; the
  held-out top-10 station hit rate is **~6.8 / 10**.
- **🎯 EXACT** — inside each flagged area, the few **corners** where the load
  actually piles up. Naming those corners captures **~47%** of the area's
  load on average (the precise daily #1 rotates; the *set* is stable).

Pick a single station from the top bar and the whole app drills one grain
finer — the forecast becomes that station's **corners** (gh7, ~150 m), coverage
gaps becomes a within-station reallocation, and the deployment view shows that
station's plan plus the neighbouring teams that can cover it.

**Terminology.** ChokePoint reads parking-enforcement records (challans).
"Congestion" here means parking-induced obstruction *inferred* from those
records — a proxy, not a measured traffic feed. The forecast is honest about
this throughout; every accuracy claim is measured on **held-out, unseen days**.

A right-side assistant answers questions against the selected day's real,
computed numbers (an optional LLM only rephrases — it never invents facts).

## Approach, model & evaluation

**The core insight.** Predictable parking-obstruction signal lives at a *coarse*
spatial grain. *Which* exact corner is worst rotates day to day (noise), but the
~5 km area that carries the load barely moves. So the lever for accuracy was the
**spatial unit, not model complexity** — re-graining the same data from police
stations to gh5 zones lifted next-day coverage from ~76% to ~96%. Everything
hangs off that: zones (WHERE) → the teams that cover them (WHO) → the corners
inside (EXACT).

**Models.**
- *WHERE / EXACT (the geo layer)* — a **rank-normalized blend** per map cell:
  `0.35·chronic + 0.30·last-28d + 0.20·last-7d + 0.15·same-weekday`, each term
  ranked to (0, 1]. Deliberately simple and geometry-based (no per-city
  training). At zone grain the signal is so stable that a **chronic-total-only**
  ranking scores within ~a point of the full blend — strong evidence it is *not*
  overfit.
- *WHO (the station layer)* — **LightGBM `LGBMRanker` (LambdaRank)**, grouped by
  (date, 2-hour window), trained only on `date < 2024-03-01`, with lag / rolling
  / day-of-week / holiday features. Day-level station order is by `roll7`; the
  ranker scores the per-shift order.

**How accuracy was pushed (honestly).**
1. **Re-grain, don't over-model** — station → gh5 zone was the single biggest
   jump (~76% → ~96% coverage), on the same data and a *simpler* model.
2. **Rank-normalize the blend** — an early max-normalized version reported a
   falsely high 90.2%; ranking removed the bias and, tellingly, produced a
   *defensible* 95.8%.
3. **Blend horizons** — chronic + 28d + 7d + weekday smooths day-to-day rotation.
4. **Validate only on unseen days** — every number is measured on the **38
   held-out days** (`date ≥ cut`). The harness reproduces the independently-known
   station `cap@20 ≈ 75%`, so the geo numbers from the same harness are trusted.
5. **Show the ceiling, never cherry-pick** — each claim is paired with the
   *oracle* (best a perfect plan could do) and the **38-day average** beside the
   selected day's number; the sparkline shows the full spread.

**Evaluation matrix** — held-out, 38 unseen days (`date ≥ 2024-03-01`):

| Layer | Grain | Metric | Result | Oracle | Reference / baseline |
|---|---|---|---|---|---|
| 📍 WHERE | gh5 zone (~5 km) | coverage `cap@25` | **95.8%** (worst day 88.6%) | 98.3% | station `cap@20` = 75.9% |
| 👮 WHO | police station | top-10 hit | **6.8 / 10** | 66.3% (`oracle@10`) | `cap@10` 57.7%, `cap@20` 75.9% |
| 🎯 EXACT | gh7 corner (~150 m) | corner capture | **~47%** | — | the corner *set* is stable; daily #1 rotates |

`cap@K` = share of the next day's actual enforcement load that sat in the K
flagged units. All figures recompute live from `events.parquet`, so they track
whatever data/cut you ship.

## Dataset

The repository ships the anonymised source as a zip:

```
data/jan to may police violation_anonymized791b166.zip
```

Unzip it and rename the CSV inside to `data/violations.csv`.

- Span: 2023-11-10 to 2024-04-08 (151 days)
- Raw rows: 298,450 → 243,405 events after cleaning
- Police stations: 54
- Training cut: **2024-03-01**. The forecaster trains only on records *before*
  this date. Dates on or after it are held out, and are the **only dates the UI
  lets you select** — each has a next day available, so every prediction is
  scored against a day the model never saw. The held-out window is **38 days**.

## Prerequisites

- Python 3.10+
- `pip install -r requirements.txt`
  (`pandas, numpy, pygeohash, lightgbm, pyarrow, fastapi, uvicorn, python-multipart, openai`)

## Setup and run (local)

```
# 1. data — unzip and rename the CSV to data/violations.csv
#    macOS / Linux:
unzip "data/jan to may police violation_anonymized791b166.zip" -d data
mv "data/jan to may police violation_anonymized791b166.csv" data/violations.csv
#    Windows PowerShell:
#    Expand-Archive "data\jan to may police violation_anonymized791b166.zip" -DestinationPath data
#    Rename-Item "data\jan to may police violation_anonymized791b166.csv" violations.csv

# 2. deps
pip install -r requirements.txt

# 3. build the cache (reads data/violations.csv, ~1 min, writes cache/)
python src/prepare.py

# 4. serve  (run from src/ so `import engine` resolves)
cd src
uvicorn server:app --port 8000
```

Open http://localhost:8000 and hard-refresh (Ctrl+Shift+R). On first start the
server pre-warms the two 38-day validation aggregates (geo + station) in a
background thread, so the forecast tabs never block on a cold compute.

The CSV path can be overridden: set `CSV` before `prepare.py`. The train/forecast
split can be overridden with `CHOKEPOINT_CUT` (see INTEGRATION.md).

## Assistant key (optional)

The assistant works offline with a deterministic responder. To have its answers
rephrased by an LLM, set `OPENAI_API_KEY` before starting uvicorn. Without a key
the deterministic responder is used. The router always decides the content and
which screen to open; the LLM only rewords.

## Re-run rules

| You changed | Do this |
|---|---|
| `app/index.html` | hard-refresh the browser |
| `src/server.py` | restart uvicorn |
| `src/prepare.py` or `src/engine.py` | re-run `python src/prepare.py`, then restart uvicorn |

## Tabs

The police-station selector in the top bar scopes **every** tab. With **All
Bengaluru** selected you get the city view; pick a station and each tab drills
into it (one grain finer where relevant).

1. **📡 Day replay** — replays a real recorded day in 30-min steps: worst spots
   as they build, the moments that mattered, an end-of-day summary, and a
   next-day teaser when the day completes.
2. **🗺 Obstruction map** — geohash hotspots for the selected day (redder/bigger
   = higher estimated impact), with a toggle for spots police are under-covering.
3. **🎯 Coverage gaps (today)** — where patrols actually go vs where the real
   load is. City view = cross-station team moves; a single station = a
   within-station, corner-level reallocation.
4. **📅 Tomorrow's forecast** — the **WHERE** layer: the 25 flagged zones (or,
   for one station, its corners), each routed to its **WHO** police team(s);
   then **"Was the forecast right?"** — the selected day's result with the
   38-day average beside it, and a zone/corner-level busy-vs-quiet check.
5. **♟ Tomorrow's deployment** — the deployment plan, the **exact spots** to
   send each team, a **"Call in help — by time"** card (for a station: which
   neighbour team to pull in, and when), a per-station **shift playbook**
   (showing the station plus the teams that can cover it), and a
   **"Did it work?"** next-day check.
6. **🧪 Predict from your data** — upload your own enforcement CSV and get the
   same WHERE zones + WHO station list + per-shift order for the day after your
   latest record; add the actual next-day CSV to score it.

The assistant rail (desktop) becomes a button-triggered drawer on small screens.

## Layout

```
app/index.html      single-file UI (Bootstrap + Leaflet + Chart.js)
src/engine.py       cleaning, impact, geohash, naming, windows, holidays,
                    the ranker, and the GEO layer (zones + station corners)
src/prepare.py      batch precompute -> cache/
src/server.py       FastAPI: read-only API + serves app/
cache/              generated artifacts (see TECHNICAL.md)
data/violations.csv input (you create this from the shipped zip)
DATA_REPORT.md      column triage and data analysis
TECHNICAL.md        data schema, scoring, model, geo layer, cache, full API
INTEGRATION.md      point the pipeline at your own data + get predictions
```

## Where this fits (BTP systems)

BTP's ASTraM platform reads live CCTV/ANPR feeds for real-time flow, congestion
length and incident response. ChokePoint works from the enforcement records and
answers a complementary question: where to pre-position limited parking-
enforcement teams the *next* day, and which of *today's* patrols are aimed away
from the real choke points. It is positioned as a complement to ASTraM, not a
replacement.

Two integration points:
- ASTraM's measured congestion (length per segment) can validate or recalibrate
  the impact score used here, which is otherwise a proxy.
- The per-station shift playbook can feed ASTraM's geofenced e-attendance, so the
  planned station-to-corner assignment is the one officers clock in against.

## Scope and honesty

All figures come from enforcement records. Impact is an estimate (vehicle
footprint × offence severity × road class), not a live traffic feed. "Congestion"
is an obstruction proxy. Every accuracy number (the ~95.8% zone coverage, the
~6.8/10 station hit, the ~47% corner capture, and the per-day checks) is
measured on **held-out dates the model never trained on**. Recommendations are
labelled as recommendations and never auto-executed. The base map is for display
only.
