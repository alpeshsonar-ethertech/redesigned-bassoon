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


def build_panel(df, psid_map=None, extend_days=1, wins=7):
    """Station x date x window panel with the 13 ranking features.
    Mirrors prepare.py. `extend_days` appends future day(s) with impact=0 so the
    next day's lag/rolling features are computed from real history.
    `df` must already have an integer `w` column (= window_of(hour))."""
    dfw = df[df.w >= 0]
    stations = sorted(dfw.ps.dropna().astype(str).unique())
    g = dfw.groupby(["ps", "date", "w"])["impact"].sum().reset_index()
    last = g.date.max()
    dates = pd.date_range(g.date.min(), last + pd.Timedelta(days=extend_days), freq="D")
    idx = pd.MultiIndex.from_product([stations, dates, range(wins)], names=["ps", "date", "w"])
    p = g.set_index(["ps", "date", "w"]).reindex(idx, fill_value=0).reset_index().sort_values(["ps", "w", "date"])
    p["dow"] = p.date.dt.dayofweek
    p["is_wknd"] = (p.dow >= 5).astype(int)
    p["is_hol"] = p.date.map(is_holiday).astype(int)
    if psid_map is not None:
        p["psid"] = p.ps.map(psid_map).fillna(-1).astype(int)
    else:
        p["psid"] = p.ps.astype("category").cat.codes
    gw = p.groupby(["ps", "w"])
    p["lag1"] = gw.impact.shift(1)
    p["lag2"] = gw.impact.shift(2)
    p["lag7"] = gw.impact.shift(7)
    p["roll7"] = gw.impact.transform(lambda s: s.shift(1).rolling(7).mean())
    p["roll7max"] = gw.impact.transform(lambda s: s.shift(1).rolling(7).max())
    p["zw_mean"] = gw.impact.transform(lambda s: s.shift(1).expanding().mean())
    p["dow_mean"] = p.groupby(["ps", "w", "dow"]).impact.transform(lambda s: s.shift(1).expanding().mean())
    p["regime_mean"] = p.groupby(["ps", "w", "is_wknd"]).impact.transform(lambda s: s.shift(1).expanding().mean())
    return p


def junction_topk(df, date, min_dens=5.0, topk=10):
    """Per-station dense-junction ranking as of `date` (history strictly < date; no leakage).
    Junction layer of the hierarchical plan: station forecast says WHICH stations are hot;
    this says WHICH corners within them to post teams. Ranked by own-history blend
    (0.5*chronic + 0.3*roll7 + 0.2*dow_mean); only junctions averaging >= min_dens
    events/active-day in history are kept (sparse junctions are noise). Honest scope:
    named junctions exist only for the central stations (~half of enforcement)."""
    jcol = "junction_name" if "junction_name" in df.columns else "jn"
    scol = "police_station" if "police_station" in df.columns else "ps"
    j = df[df[jcol].notna() & (df[jcol] != "No Junction")].copy()
    if j.empty:
        return {}
    j["jn"] = j[jcol].astype(str); j["st"] = j[scol].astype(str)
    date = pd.Timestamp(date).normalize()
    hist = j[j.date < date]
    if hist.empty:
        return {}
    first = hist.date.min()
    elapsed = max((date - first).days, 1)
    dow = date.dayofweek
    rng = pd.date_range(first, date - pd.Timedelta(days=1), freq="D")
    same_dow_days = max(int((rng.dayofweek == dow).sum()), 1)
    last7 = hist[hist.date >= date - pd.Timedelta(days=7)]
    agg = hist.groupby(["st", "jn"]).agg(tot=("impact", "sum"), c=("impact", "size"), ad=("date", "nunique"))
    agg["dens"] = agg.c / agg.ad
    agg["chronic"] = agg.tot / elapsed
    agg["roll7"] = (last7.groupby(["st", "jn"]).impact.sum() / 7.0).reindex(agg.index).fillna(0)
    agg["dowm"] = (hist[hist.date.dt.dayofweek == dow].groupby(["st", "jn"]).impact.sum() / same_dow_days).reindex(agg.index).fillna(0)
    agg["score"] = 0.5 * agg.chronic + 0.3 * agg.roll7 + 0.2 * agg.dowm
    dense = agg[agg.dens >= min_dens].reset_index()
    out = {}
    for st, gg in dense.groupby("st"):
        gg = gg.sort_values("score", ascending=False).head(topk)
        out[st] = [{"jn": r.jn, "score": round(float(r.score), 1), "dens": round(float(r.dens), 1)} for r in gg.itertuples()]
    return out


def station_spots(df, date, topk=5, min_events=12, min_per_day=1.5):
    """Per-station top hot spots as of `date` (history strictly < date; no leakage).
    Spot = named junction where the ticket carried one, else the road (location).
    Gives EVERY station named spots (junctions in the centre, roads elsewhere),
    ranked by own-history blend (0.5*chronic + 0.3*roll7 + 0.2*dow_mean). Returns
    {station: [{name, is_junction, lat, lon, score, per_day}...]}. Honest: this is
    chronic concentration (where to look), not a daily single-spot prediction."""
    scol = "police_station" if "police_station" in df.columns else "ps"
    jcol = "junction_name" if "junction_name" in df.columns else ("jn" if "jn" in df.columns else None)
    cols = [scol, "location", "lat", "lon", "impact", "date"] + ([jcol] if jcol else [])
    if "w" in df.columns: cols.append("w")
    elif "hour" in df.columns: cols.append("hour")
    d = df[cols].copy()
    if "w" not in d.columns:
        d["w"] = d["hour"].map(window_of) if "hour" in d.columns else 0
    if jcol is None:                       # uploads may have no junction column -> road-level only
        d["__jn"] = np.nan; jcol = "__jn"
    d["st"] = d[scol].astype(str)
    isj = d[jcol].notna() & (d[jcol].astype(str) != "No Junction")
    d["spot"] = np.where(isj, d[jcol].astype(str), d["location"].astype(str))
    d["is_jn"] = isj.values
    d = d[d.spot.notna() & (d.spot.astype(str) != "nan")]
    date = pd.Timestamp(date).normalize()
    hist = d[d.date < date]
    if hist.empty:
        return {}
    first = hist.date.min(); elapsed = max((date - first).days, 1)
    dow = date.dayofweek
    rng = pd.date_range(first, date - pd.Timedelta(days=1), freq="D")
    sdd = max(int((rng.dayofweek == dow).sum()), 1)
    last7 = hist[hist.date >= date - pd.Timedelta(days=7)]
    g = hist.groupby(["st", "spot"])
    agg = g.agg(tot=("impact", "sum"), c=("impact", "size"),
                lat=("lat", "median"), lon=("lon", "median"), isj=("is_jn", "max"))
    agg["chronic"] = agg.tot / elapsed
    agg["roll7"] = (last7.groupby(["st", "spot"]).impact.sum() / 7.0).reindex(agg.index).fillna(0)
    agg["dowm"] = (hist[hist.date.dt.dayofweek == dow].groupby(["st", "spot"]).impact.sum() / sdd).reindex(agg.index).fillna(0)
    agg["score"] = 0.5 * agg.chronic + 0.3 * agg.roll7 + 0.2 * agg.dowm
    agg["per_day"] = agg.c / elapsed
    # drop noise: too few events overall AND below a daily floor (no "post a team where 0.1 tickets/day land")
    agg = agg[(agg.c >= min_events) & (agg.per_day >= min_per_day)].reset_index()
    st_tot = hist.groupby("st").impact.sum()                          # for share-of-station
    pws = hist[hist.w >= 0]                                            # peak window among real daytime windows
    if len(pws):
        pw = pws.groupby(["st", "spot", "w"]).impact.sum().reset_index()
        pw = pw.loc[pw.groupby(["st", "spot"]).impact.idxmax()].set_index(["st", "spot"])["w"]
    else:
        pw = pd.Series(dtype="int64")
    out = {}
    for st, gg in agg.groupby("st"):
        gg = gg.sort_values("score", ascending=False).head(topk)
        rows = []
        for r in gg.itertuples():
            nm = r.spot
            if nm.startswith("Unnamed Road"):           # relabel unnamed roads by nearest landmark
                seg = [s.strip() for s in nm.split(",")[1:] if s.strip()]
                nm = ("near " + seg[0]) if seg else "Unnamed road"
            pwi = int(pw.get((st, r.spot), 0))
            if pwi < 0 or pwi >= len(WINDOW_LABELS): pwi = 0
            rows.append({"name": nm, "is_junction": bool(r.isj),
                         "lat": round(float(r.lat), 5), "lon": round(float(r.lon), 5),
                         "score": round(float(r.score), 1), "per_day": round(float(r.per_day), 1),
                         "share": int(round(r.tot / float(st_tot.get(st, 1) or 1) * 100)),
                         "peak_w": pwi, "peak": WINDOW_LABELS[pwi] if pwi < len(WINDOW_LABELS) else ""})
        out[st] = rows
    return out


def corner_check(df, date, stations, topk=3):
    """Location-level validation: did each station's chronic top-k corners (from history < date)
    capture its ACTUAL obstruction on `date`? Returns {avg_capk, per_station, top_detail, k}.
    Honest: validates the corner SET (cap@k), not the exact daily #1 (which rotates)."""
    scol = "police_station" if "police_station" in df.columns else "ps"
    jcol = "junction_name" if "junction_name" in df.columns else ("jn" if "jn" in df.columns else None)
    d = df[[scol, "location", "lat", "lon", "impact", "date"] + ([jcol] if jcol else [])].copy()
    d["st"] = d[scol].astype(str)
    if jcol is None:
        d["__jn"] = np.nan; jcol = "__jn"
    isj = d[jcol].notna() & (d[jcol].astype(str) != "No Junction")
    d["spot"] = np.where(isj, d[jcol].astype(str), d["location"].astype(str)); d["is_jn"] = isj.values
    d = d[d.spot.notna() & (d.spot.astype(str) != "nan")]
    date = pd.Timestamp(date).normalize()
    hist = d[d.date < date]; act = d[d.date == date]
    if hist.empty or act.empty:
        return {"avg_capk": None, "k": topk, "per_station": {}, "top_detail": []}
    ch = hist.groupby(["st", "spot"]).impact.sum().reset_index()
    chosen = {st: list(g.sort_values("impact", ascending=False).head(topk).spot) for st, g in ch.groupby("st")}
    isj_map = dict(zip(d.spot, d.is_jn))
    actg = act.groupby(["st", "spot"]).impact.sum(); act_tot = act.groupby("st").impact.sum()
    per = {}
    for st in stations:
        if st not in chosen or st not in act_tot.index or act_tot[st] <= 0: continue
        per[st] = round(sum(actg.get((st, sp), 0) for sp in chosen[st]) / act_tot[st] * 100, 1)
    avg = round(float(np.mean(list(per.values()))), 1) if per else None
    top_detail = []
    cand = [s for s in stations if s in chosen and s in act_tot.index]
    if cand:
        ts = max(cand, key=lambda s: act_tot.get(s, 0))
        actsp = act[act.st == ts].groupby("spot").impact.sum().sort_values(ascending=False)
        act_top = set(actsp.head(max(topk, 3)).index)
        for sp in chosen[ts]:
            nm = sp
            if nm.startswith("Unnamed Road"):
                seg = [x.strip() for x in nm.split(",")[1:] if x.strip()]; nm = ("near " + seg[0]) if seg else "Unnamed road"
            elif nm.startswith("BTP") and " - " in nm:
                nm = nm.split(" - ", 1)[1]
            nm = ", ".join(nm.split(",")[:2]).strip()
            top_detail.append({"station": ts, "name": nm, "is_junction": bool(isj_map.get(sp, False)),
                               "hit": sp in act_top, "share": int(round(actsp.get(sp, 0) / act_tot[ts] * 100))})
    return {"avg_capk": avg, "k": topk, "per_station": per, "top_detail": top_detail}


# ================= GEO LAYER (gh5 "where", ~5km) =================
def _geo_prep(df):
    d = df.copy()
    if "gh5" not in d.columns:
        d["gh5"] = d["gh7"].astype(str).str[:5]
    return d[d.gh5.notna() & (d.gh5.astype(str) != "nan") & (d.gh5.astype(str).str.len() == 5)]

def geo_rank(hist, T, col="gh5"):
    """Held-out-verified blend: rank-normalized chronic + recent(28/7) + day-of-week. ~96% cap@25."""
    base = hist.groupby(col).impact.sum()
    idx = base.index
    def rn(s):
        s = s.reindex(idx).fillna(0); n = max(len(s), 1); return s.rank() / n
    c = rn(base)
    r28 = rn(hist[hist.date >= T - pd.Timedelta(days=28)].groupby(col).impact.sum())
    r7 = rn(hist[hist.date >= T - pd.Timedelta(days=7)].groupby(col).impact.sum())
    dw = rn(hist[hist.date.dt.dayofweek == T.dayofweek].groupby(col).impact.mean())
    return (0.35 * c + 0.30 * r28 + 0.20 * r7 + 0.15 * dw).sort_values(ascending=False)

def _geo_label(g):
    """Plain locality name for a zone: the highest-impact named junction, else commonest road token."""
    jn = g[g.junction_name.notna() & (g.junction_name.astype(str) != "No Junction")]
    if len(jn):
        top = jn.groupby("junction_name").impact.sum().idxmax()
        nm = str(top)
    else:
        loc = g.location.dropna().astype(str)
        nm = loc.mode().iloc[0] if len(loc) else "area"
    if nm.startswith("BTP") and " - " in nm:
        nm = nm.split(" - ", 1)[1]
    if nm.startswith("Unnamed Road"):
        seg = [s.strip() for s in nm.split(",")[1:] if s.strip()]; nm = ("near " + seg[0]) if seg else "Unnamed area"
    return re.sub(r"\s+", " ", ", ".join(nm.split(",")[:2])).strip()

def geo_zone_meta(df):
    """Per gh5 zone: centroid pin, plain label, locality (for disambiguation), and covering station team(s)."""
    d = _geo_prep(df); meta = {}
    for z, g in d.groupby("gh5"):
        st = g.groupby("ps").impact.sum().sort_values(ascending=False); tot = float(st.sum()) or 1
        stations = [{"station": s, "share": int(round(v / tot * 100))} for s, v in st.items() if v / tot >= 0.12][:3]
        if not stations and len(st):
            stations = [{"station": st.index[0], "share": int(round(st.iloc[0] / tot * 100))}]
        locs = g.location.dropna().astype(str)
        locality = ""
        if len(locs):
            BAD = ("bengaluru", "bangalore", "karnataka", "india", "pin-", "pin ")
            def hood(s):
                segs = [x.strip() for x in s.split(",")]
                cand = [x for x in segs[1:] if x and not any(b in x.lower() for b in BAD)]
                return cand[0] if cand else ""
            h = locs.map(hood); h = h[h.astype(bool)]
            locality = str(h.mode().iloc[0]) if len(h.mode()) else ""
        meta[z] = {"lat": round(float(g.lat.mean()), 5), "lon": round(float(g.lon.mean()), 5),
                   "label": _geo_label(g), "locality": locality, "stations": stations}
    return meta

def geo_forecast(df, date, K=25, light=False):
    """Geo 'where' forecast for the day AFTER `date`: ranked gh5 zones (labelled, pinned, routed to stations)
    + held-out cap@K / oracle@K / hit@K. The honest 'achieved coverage' layer. light=True skips zone meta."""
    d = _geo_prep(df); date = pd.Timestamp(date).normalize(); nxt = date + pd.Timedelta(days=1)
    hist = d[d.date < nxt]
    if hist.empty:
        return None
    rank = geo_rank(hist, nxt)
    pred = list(rank.head(K).index)
    act = d[d.date == nxt].groupby("gh5").impact.sum()
    tot = float(act.sum())
    has_actual = tot > 0
    cap = round(act.reindex(pred).fillna(0).sum() / tot * 100, 1) if has_actual else None
    oracle = round(act.nlargest(K).sum() / tot * 100, 1) if has_actual else None
    act_top = set(act.nlargest(K).index) if has_actual else set()
    hit = len(set(pred) & act_top) if has_actual else None
    if light:
        return {"next": str(nxt.date()), "K": K, "cap": cap, "oracle": oracle, "hit": hit, "has_actual": has_actual}
    meta = geo_zone_meta(hist)
    zones = []
    seen = {}
    for z in pred:
        m = meta.get(z, {})
        base = m.get("label", z); lab = base
        if base in seen:
            loc = (m.get("locality") or "").strip()
            lab = f"{base} · {loc}" if (loc and loc.lower() not in base.lower()) else f"{base} ({seen[base] + 1})"
            seen[base] += 1
        else:
            seen[base] = 1
        zones.append({"zone": z, "label": lab, "lat": m.get("lat"), "lon": m.get("lon"),
                      "stations": m.get("stations", []), "score": round(float(rank.get(z, 0)), 3),
                      "actual": round(float(act.get(z, 0)), 1) if has_actual else None,
                      "hit": (z in act_top) if has_actual else None})
    return {"next": str(nxt.date()), "K": K, "n_zones": int(d.gh5.nunique()),
            "cap": cap, "oracle": oracle, "hit": hit, "has_actual": has_actual, "zones": zones, "scope": "city"}


def station_forecast(df, date, station, K=8, light=False):
    """Station-scope 'where' forecast: same shape as geo_forecast, but one grain finer (gh7 ~150m corners)
    inside a single police station. For the drill-down view when a station is selected."""
    date = pd.Timestamp(date).normalize(); nxt = date + pd.Timedelta(days=1)
    d = df[df.ps == station].copy()
    if "gh7" not in d.columns or d.empty:
        return None
    d = d[d.gh7.notna() & (d.gh7.astype(str) != "nan")]
    hist = d[d.date < nxt]
    if hist.empty:
        return None
    rank = geo_rank(hist, nxt, col="gh7")
    K = int(min(K, len(rank)))
    pred = list(rank.head(K).index)
    act = d[d.date == nxt].groupby("gh7").impact.sum()
    tot = float(act.sum()); has_actual = tot > 0
    cap = round(act.reindex(pred).fillna(0).sum() / tot * 100, 1) if has_actual else None
    oracle = round(act.nlargest(K).sum() / tot * 100, 1) if has_actual else None
    act_top = set(act.nlargest(K).index) if has_actual else set()
    hit = len(set(pred) & act_top) if has_actual else None
    if light:
        return {"next": str(nxt.date()), "K": K, "cap": cap, "oracle": oracle, "hit": hit, "has_actual": has_actual}
    zones = []; seen = {}
    for z in pred:
        g = hist[hist.gh7 == z]
        if g.empty:
            g = d[d.gh7 == z]
        base = _geo_label(g); lab = base
        if base in seen:
            lab = f"{base} ({seen[base] + 1})"; seen[base] += 1
        else:
            seen[base] = 1
        zones.append({"zone": z, "label": lab, "lat": round(float(g.lat.mean()), 5), "lon": round(float(g.lon.mean()), 5),
                      "stations": [{"station": station, "share": 100}], "score": round(float(rank.get(z, 0)), 3),
                      "actual": round(float(act.get(z, 0)), 1) if has_actual else None,
                      "hit": (z in act_top) if has_actual else None})
    return {"next": str(nxt.date()), "K": K, "n_zones": int(d.gh7.nunique()),
            "cap": cap, "oracle": oracle, "hit": hit, "has_actual": has_actual, "zones": zones,
            "scope": "station", "station": station}
