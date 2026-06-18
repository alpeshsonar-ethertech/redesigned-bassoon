"""Shared ChokePoint logic: cleaning, impact scoring, geohash, naming.
Used by prepare.py (precompute) and server.py (live queries)."""
import os
import pandas as pd, numpy as np, re
import pygeohash as pgh

CUT = pd.Timestamp(os.environ.get("CHOKEPOINT_CUT", "2024-03-01"))  # train < CUT, test >= CUT (held out)

SEV = {107: 4, 109: 4, 104: 3, 108: 3, 111: 3, 105: 2, 112: 2, 113: 2, 116: 0}
OFF = {104: "Near road crossing", 105: "On footpath", 107: "On a main road", 108: "Opposite parked vehicle",
       109: "Double parking", 111: "Near bus-stop/school/hospital", 112: "Wrong parking", 113: "No parking", 116: "Defective plate"}
ROAD_KW = re.compile(r'(road|circle|junction|cross|main|nagar|layout|market|colony|extension|flyover|ring|street|chowk|gate|temple|station|metro|stage|block|park|bridge|halli|pura|pet|town|garden)', re.I)

def footprint(v):
    v = (v or "").upper()
    if any(k in v for k in ["BUS", "HGV", "LORRY", "TANKER", "TRUCK"]): return 5.0
    if any(k in v for k in ["LGV", "TEMPO", "GOODS", "VAN", "MAXI", "JEEP", "CAR"]): return 3.0
    if any(k in v for k in ["SCOOTER", "MOTOR", "MOPED", "CYCLE", "AUTO"]): return 1.0
    return 2.0

def sev(s):
    c = [int(x) for x in re.findall(r'\d+', s or "")]
    return max((SEV.get(i, 1) for i in c), default=1)

def roadmult(loc):
    l = str(loc).lower()
    return 1.5 if any(k in l for k in ["main road", "ring road", "flyover", "underpass", "junction"]) else 1.0

def primary_off(s):
    c = [int(x) for x in re.findall(r'\d+', s or "")]
    c = [i for i in c if i in OFF] or c
    return max(c, key=lambda i: SEV.get(i, 1)) if c else 0

def vehclass(v):
    v = (v or "").upper()
    if any(k in v for k in ["BUS", "HGV", "LORRY", "TANKER", "TRUCK"]): return "Heavy"
    if any(k in v for k in ["LGV", "TEMPO", "GOODS", "VAN", "MAXI", "JEEP", "CAR"]): return "Car/Medium"
    return "Two-wheeler/Auto"

def _segs(s):
    return [p.strip() for p in re.split(r',', s) if p.strip()]

def pick_name(s):
    parts = _segs(s)
    if not parts: return "", ""
    cand = [p for p in parts if not re.fullmatch(r'[\d\s\-/.]+', p) and len(p) > 2]
    pref = [p for p in cand if ROAD_KW.search(p)]
    pick = pref[0] if pref else (cand[0] if cand else parts[0])
    pick = re.sub(r'^(no\.?\s*)?\d+\s*[,/-]\s*', '', pick, flags=re.I).strip()
    pick = re.sub(r'\s+', ' ', pick)
    name = pick[:36] or (cand[0][:36] if cand else parts[0][:36])
    area = cand[1][:28] if len(cand) > 1 else ""
    return name, area

def load_clean(csv):
    """Load CSV, convert to IST, dedup sweep bursts, drop rejected/duplicate for impact.
    Returns (events_df, raw_with_rejected_df)."""
    df = pd.read_csv(csv, dtype=str, keep_default_na=False, na_values=["", "NULL", "null", "NaN"])
    dt = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
    df = df[dt.notna()].copy(); dt = dt[dt.notna()]
    ist = dt.dt.tz_convert("Asia/Kolkata")
    df["date"] = ist.dt.normalize().dt.tz_localize(None)
    df["hour"] = ist.dt.hour
    df["minute"] = ist.dt.minute
    df["dow"] = ist.dt.dayofweek
    df["rejected"] = (df["validation_status"] == "rejected").astype(int)
    raw = df.copy()
    df = df[~df["validation_status"].isin(["rejected", "duplicate"])].copy()
    ist2 = ist[df.index]
    df["_min"] = ist2.dt.floor("min")
    df = df.drop_duplicates(subset=["vehicle_number", "latitude", "longitude", "_min"]).copy()
    lat = pd.to_numeric(df.latitude, errors="coerce"); lon = pd.to_numeric(df.longitude, errors="coerce")
    df = df[lat.notna() & lon.notna()].copy()
    df["lat"] = lat[df.index]; df["lon"] = lon[df.index]
    df["gh7"] = [pgh.encode(a, o, 7) for a, o in zip(df.lat, df.lon)]
    df["gh6"] = df.gh7.str[:6]
    df["fp"] = df["vehicle_type"].fillna("").map(footprint)
    df["sev"] = df["offence_code"].fillna("").map(sev)
    df["rm"] = df["location"].fillna("").map(roadmult)
    df["impact"] = df.fp * df.sev * df.rm
    df["poff"] = df["offence_code"].fillna("").map(primary_off)
    df["vclass"] = df["vehicle_type"].fillna("").map(vehclass)
    return df, raw

# Indian/Karnataka public holidays within the data span (calendar-derived)
HOLIDAYS = {
    "2023-11-12": "Diwali", "2023-11-13": "Diwali (Padwa)", "2023-11-14": "Bhai Dooj",
    "2023-11-27": "Guru Nanak Jayanti", "2023-12-25": "Christmas", "2024-01-01": "New Year",
    "2024-01-15": "Makar Sankranti", "2024-01-26": "Republic Day", "2024-03-08": "Maha Shivaratri",
    "2024-03-25": "Holi", "2024-03-29": "Good Friday", "2024-03-31": "Easter", "2024-04-07": "Ramzan",
    "2024-04-09": "Ugadi",
}
HOL = set(pd.Timestamp(d) for d in HOLIDAYS)

def is_holiday(d):
    return pd.Timestamp(d).normalize() in HOL

def window_of(hour):
    """2-hour enforcement window index 0..6 over 06:00-20:00; -1 outside."""
    h = int(hour)
    return (h - 6) // 2 if 6 <= h < 20 else -1

WINDOW_LABELS = ["06-08", "08-10", "10-12", "12-14", "14-16", "16-18", "18-20"]

def station_centroids(df):
    """police_station -> {lat, lon, n} centroid for routing/coverage."""
    out = {}
    for ps, g in df.groupby("police_station"):
        if pd.isna(ps): continue
        out[str(ps)] = {"lat": round(float(g.lat.mean()), 5), "lon": round(float(g.lon.mean()), 5), "n": int(len(g))}
    return out

def haversine(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, [la1, lo1, la2, lo2])
    d = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(d))

def name_maps(df):
    """Build gh7 -> {name, area, lat, lon} using the modal location string per cell."""
    m = {}
    for gh, g in df.groupby("gh7"):
        loc = g.location.dropna()
        if len(loc):
            nm, ar = pick_name(loc.mode().iloc[0])
        else:
            nm, ar = gh, ""
        m[gh] = {"name": nm or gh, "area": ar, "lat": round(float(g.lat.mean()), 5), "lon": round(float(g.lon.mean()), 5)}
    return m
