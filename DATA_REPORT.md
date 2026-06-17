# ChokePoint — Data Analysis Report
**Dataset:** Bengaluru Traffic Police parking-violation records
**Span:** 2023-11-10 → 2024-04-08 (151 days) · **Raw rows:** 298,450 · **After cleaning:** 243,405

---

## 1. Column triage (24 → 9 active)

**KEEP — the 9 columns that power everything (all 0–1% null):**
| Column | Role |
|---|---|
| latitude / longitude | violation location → geohash (map) |
| location | reverse-geocoded address → road-class parsing |
| vehicle_type | footprint weight in impact score |
| offence_code / violation_type | severity weight in impact score |
| created_datetime | true timestamp (UTC→IST) → hour, day-of-week, holiday |
| device_id | unique devices → patrol-coverage proxy |
| created_by_id | officer → sweep-burst dedup |
| police_station | jurisdiction (~2 km) → forecast & deployment unit |
| validation_status | drop `rejected`/`duplicate` only |

**DROP — no congestion signal:**
- 100% null: description, closed_datetime, action_taken_timestamp
- workflow audit (post-event paperwork; congestion identical with/without): modified_datetime, validation_timestamp, data_sent_to_scita(+timestamp)
- redundant/identifiers: id, vehicle_number*, center_code (= police_station), updated_vehicle_*
- junction_name: 50% "No Junction", only 168 usable → unusable; parse road-class from `location` instead

\*vehicle_number is dropped from modelling but useful for the repeat-offender view (§4).

---

## 2. The 5 real drivers of congestion

Congestion ≈ f(**where**, **what vehicle**, **how parked**, **when**, **road type**). Everything else is metadata.

1. **WHERE** — latitude/longitude (geohash) + police_station. Dominant factor.
2. **WHAT** — vehicle_type (tanker ≫ scooter footprint).
3. **HOW** — offence_code (main-road/crossing ≫ minor).
4. **WHEN** — hour, day-of-week, holiday.
5. **ROAD** — arterial vs minor (parsed from location).

**The one driver NOT in the data:** live traffic volume / road capacity. A parked car on an empty lane ≠ the same car on MG Road at 9 AM. This missing variable is a large part of the irreducible forecast ceiling (§6) — and the natural next data source.

---

## 3. Aggregation grain — geohash vs police-station (validated)

`police_station` was tested and confirmed to be a **geographic jurisdiction** (~2 km cluster of coordinates), **not** a reporting artifact. Tight spreads: Upparpet 0.6 km, Shivajinagar ~1 km; median station 80/10-pct spread ~2 km vs citywide ~38 km.

| | geohash-6 (785 cells) | police-station (54) |
|---|---|---|
| non-zero cells (zone×window×day) | 10% | **33%** |
| actual top-10 day-to-day overlap (ceiling) | 0.42 | **0.59** |
| persistence hit@10 | 0.33 | **0.52** |
| recurring stability (train→test) | 0.80 | 0.80–0.90 |

**Decision:** forecast & deploy at **police-station × 2-hour-window**; keep **geohash for the map** (pinpoint detail).

---

## 4. New analytical angles

**Repeat offenders:** 231,890 unique vehicles; 85% ticketed once, but **repeat plates (15%) = 34% of all tickets**; max 55 tickets on one vehicle; 3,489 vehicles with 5+. → chronic-offender angle available.

**Patrol coverage:** 2,998 devices / 2,619 officers; **every station has 34–235 devices (median 88), none thin.** → the enforcement "blind spot" is about *where officers ticket*, not lack of equipment.

**Road-class (text-mined from location):** junction 29%, main road 22%, cross 12%, ring road 5%, flyover/underpass 2%. **Arterial roads = 29% of tickets but 39% of total impact** → validates the road multiplier in CIS.

**Spatial concentration:** top-20 geohash-6 cells = **42% of impact**, top-50 = **61%**. Congestion sits in a few corridors → targeting is justified.

**Inter-station distance (routing):** median nearest-neighbour **1.9 km**; 78% of stations have a neighbour <3 km; **avg 8.4 stations within 5 km**; zero isolated. → **one team at a hub can cover 2–3 adjacent stations** — the nearest-station patrol logic is feasible.

---

## 5. Congestion regimes (weekday / weekend / holiday)

Volume is a **stable background on working days**; it swings on holidays.

| Regime | volume CV | ranking stability (same-type day-to-day) | early-vs-late recurring overlap |
|---|---|---|---|
| weekday | 21% | 0.591 | 8/10 |
| weekend | 20% | 0.571 | 9/10 |
| **holiday** | **34%** | 0.554 | **5/10** |

- **Top hotspots are constant across regimes** (Upparpet, Malleshwaram, Shivajinagar, HAL Old Airport, City Market).
- **Hotspots relocate on holidays** — weekday→holiday top-10 only 7/10 same. Stations that **over-index on holidays**: Whitefield (1.9×), Jalahalli (1.9×), Thalagattapura (1.6×), Byatarayanapura (1.6×).
- **Timing is enforcement-biased** — all regimes peak 10–11 AM (officers work mornings); the *station* matters far more than the *hour*.

**Implication:** predict weekday/weekend normally; treat **holidays as a distinct "watch-list" regime** (relocation is real but under-sampled at only 13 holidays in 5 months).

---

## 6. Forecast ceiling & model selection (exhaustively tested)

The **actual** top-10 only repeats ~0.59 day-to-day — this is reality's ceiling. Tested across 3 grains, 3 objectives (regression / classification / **learning-to-rank**), 2 horizons, ~8 feature schemes, 5 model families, per-window models, label variants, blends, walk-forward eval. **All converge:**

| | single-day | 3-day outlook |
|---|---|---|
| persistence (no ML) | 0.51 | 0.69 |
| **LambdaRank / boosted classifier** | **0.587** | **0.750** |
| single decision tree (interpretable) | 0.582 | — |

- **Capture rate:** predicted top-10 stations contain **~51% of next-day impact**, **~60% over 3 days**.
- **hit@10 = ~6 of 10** worst stations correct tomorrow; **~7–8 of 10** over a 3-day outlook.
- A **single decision tree (0.582)** is within 1 pt of the boosted model — fully interpretable if BTP trust matters.

**Honest framing:** "reaches the data's predictable ceiling," NOT "59% accurate" — the residual ~40% is genuine random churn (and the missing traffic-volume variable).

---

## 7. Final architecture

1. **CIS (impact)** = deterministic rules engine — vehicle × offence × road. No ML (police-auditable).
2. **Forecast** = LambdaRank (or single tree) classifier at **station × window**, features: station, window, day-of-week, is_holiday, is_weekend, lags, regime mean. Two horizons (tomorrow + 3-day outlook).
3. **Deployment** = nearest-station coverage optimizer (team covers its station + neighbours within ~2–5 km).
4. **Map** = geohash hotspot detail.
5. **Holiday mode** = surface the relocation watch-list (Whitefield, Jalahalli, …).

**Improvement path (for the deck):** longer data history (2–3 yrs → enough holiday samples to learn the relocation regime) and a live traffic-volume feed (the one missing driver).
