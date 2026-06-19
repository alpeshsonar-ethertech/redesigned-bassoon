#!/usr/bin/env python3
"""ChokePoint engine: BTP parking log -> bias-corrected hotspots, enforcement-gap,
forecast, MCLP reallocation, enforcement-quality. Writes app/data.js for the dashboard."""
import pandas as pd, numpy as np, re, json, os, warnings
warnings.filterwarnings('ignore')
import pygeohash as pgh
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV = os.environ.get("CSV", os.path.join(ROOT, "data", "violations.csv"))
if not os.path.exists(CSV):
    CSV = "/home/claude/t1/jan to may police violation_anonymized791b166.csv"

print("loading", CSV)
df = pd.read_csv(CSV, dtype=str, keep_default_na=False, na_values=["", "NULL", "null", "NaN"])
N_RAW = len(df)
dt = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
df = df[dt.notna()].copy(); dt = dt[dt.notna()]; ist = dt.dt.tz_convert("Asia/Kolkata")
df["date"] = ist.dt.normalize().dt.tz_localize(None); df["hour"] = ist.dt.hour
df["dow"] = ist.dt.dayofweek; df["is_wknd"] = (df.dow >= 5).astype(int)
df["rejected"] = (df["validation_status"] == "rejected").astype(int)
RAW = df.copy()
df = df[~df["validation_status"].isin(["rejected", "duplicate"])].copy(); ist = ist[df.index]
df["_min"] = ist.dt.floor("min")
df = df.drop_duplicates(subset=["vehicle_number", "latitude", "longitude", "_min"]).copy()
N_EVENTS = len(df)

def footprint(v):
    v = (v or "").upper()
    if any(k in v for k in ["BUS", "HGV", "LORRY", "TANKER", "TRUCK"]): return 5.0
    if any(k in v for k in ["LGV", "TEMPO", "GOODS", "VAN", "MAXI", "JEEP", "CAR"]): return 3.0
    if any(k in v for k in ["SCOOTER", "MOTOR", "MOPED", "CYCLE", "AUTO"]): return 1.0
    return 2.0
SEV = {107: 4, 109: 4, 104: 3, 108: 3, 111: 3, 105: 2, 112: 2, 113: 2, 116: 0}
def sev(s):
    c = [int(x) for x in re.findall(r'\d+', s or "")]
    return max((SEV.get(i, 1) for i in c), default=1)
def roadmult(loc):
    l = str(loc).lower()
    return 1.5 if any(k in l for k in ["main road", "ring road", "flyover", "underpass", "junction"]) else 1.0
df["fp"] = df["vehicle_type"].fillna("").map(footprint)
df["sev"] = df["offence_code"].fillna("").map(sev)
df["rm"] = df["location"].fillna("").map(roadmult)
df["impact"] = df.fp * df.sev * df.rm
lat = pd.to_numeric(df.latitude, errors="coerce"); lon = pd.to_numeric(df.longitude, errors="coerce")
df["gh7"] = [pgh.encode(a, o, 7) for a, o in zip(lat, lon)]; df["gh6"] = df.gh7.str[:6]
df["lat"] = lat; df["lon"] = lon

OFF = {104: "Near road crossing", 105: "On footpath", 107: "On a main road", 108: "Opposite parked vehicle",
       109: "Double parking", 111: "Near bustop/school/hospital", 112: "Wrong parking", 113: "No parking", 116: "Defective plate"}
def primary_off(s):
    c = [int(x) for x in re.findall(r'\d+', s or "")]
    c = [i for i in c if i in OFF] or c
    return max(c, key=lambda i: SEV.get(i, 1)) if c else 0
df["poff"] = df["offence_code"].fillna("").map(primary_off)
def vehclass(v):
    v = (v or "").upper()
    if any(k in v for k in ["BUS", "HGV", "LORRY", "TANKER", "TRUCK"]): return "Heavy"
    if any(k in v for k in ["LGV", "TEMPO", "GOODS", "VAN", "MAXI", "JEEP", "CAR"]): return "Car/Medium"
    return "Two-wheeler/Auto"
df["vclass"] = df.vehicle_type.fillna("").map(vehclass)

# ---------- HOTSPOTS @ gh7 ----------
g = df.groupby("gh7")
H = g.agg(cis=("impact", "sum"), tickets=("impact", "size"), lat=("lat", "mean"), lon=("lon", "mean"),
         coverage=("device_id", "nunique"), days=("date", "nunique")).reset_index()
ev = df.groupby("gh7").apply(lambda d: d.drop_duplicates(["created_by_id", "_min"]).shape[0])
H["events"] = H.gh7.map(ev)
H["gap"] = H.cis / (H.coverage + 1)
H["chronic"] = (H.days >= 10)
ROAD_KW = re.compile(r'(road|circle|junction|cross|main|nagar|layout|market|colony|extension|flyover|ring|street|chowk|gate|temple|station|metro|stage|block|park|bridge|halli|pura|pet|town|garden|main rd)', re.I)
def _segs(s):
    return [p.strip() for p in re.split(r',', s) if p.strip()]
def _pick(parts):
    cand = [p for p in parts if not re.fullmatch(r'[\d\s\-/.]+', p) and len(p) > 2]
    pref = [p for p in cand if ROAD_KW.search(p)]
    pick = pref[0] if pref else (cand[0] if cand else (parts[0] if parts else ""))
    # strip a leading house number ONLY when followed by a separator (keep ordinals like "5th Main Road")
    pick = re.sub(r'^(no\.?\s*)?\d+\s*[,/-]\s*', '', pick, flags=re.I).strip()
    pick = re.sub(r'\s+', ' ', pick)
    return pick[:36], cand
def name_of(gh):
    locs = df[df.gh7 == gh].location.dropna()
    if len(locs) == 0: return gh
    pick, cand = _pick(_segs(locs.mode().iloc[0]))
    return pick or (cand[0][:36] if cand else gh)
def area_of(gh):
    locs = df[df.gh7 == gh].location.dropna()
    if len(locs) == 0: return ""
    _, cand = _pick(_segs(locs.mode().iloc[0]))
    return cand[1][:28] if len(cand) > 1 else ""
H = H.sort_values("cis", ascending=False).head(40).reset_index(drop=True)
H["name"] = H.gh7.map(name_of); H["area"] = H.gh7.map(area_of)
H["peak"] = H.gh7.map(lambda gh: int(df[df.gh7 == gh].hour.mode().iloc[0]))
mx = H.cis.max()
def mix(gh, col, top=4):
    d = df[df.gh7 == gh]
    vc = (d.groupby(col)["impact"].sum() if col == "poff" else d[col].value_counts())
    vc = vc.sort_values(ascending=False).head(top); tot = vc.sum() or 1
    return [[OFF.get(k, str(k)) if col == "poff" else k, int(round(v / tot * 100))] for k, v in vc.items()]
def hours(gh):
    d = df[df.gh7 == gh]; h = d.groupby("hour").size().reindex(range(24), fill_value=0)
    m = h.max() or 1
    return [int(round(x / m * 100)) for x in h]
hot = []
for _, r in H.iterrows():
    hot.append(dict(gh=r.gh7, name=(r["name"] or r.gh7), area=r["area"], lat=round(r.lat, 5), lon=round(r.lon, 5),
        cis=int(r.cis), cis_norm=int(round(r.cis / mx * 100)), tickets=int(r.tickets), events=int(r.events),
        coverage=int(r.coverage), days=int(r.days), chronic=bool(r.chronic), peak=int(r.peak),
        gap=round(float(r.gap), 1), offmix=mix(r.gh7, "poff"), vehmix=mix(r.gh7, "vclass"), hours=hours(r.gh7)))

# ---------- REVEAL: naive (tickets) vs corrected (gap) ----------
allz = g.agg(cis=("impact", "sum"), tickets=("impact", "size"), ncov=("device_id", "nunique")).reset_index()
allz["gap"] = allz.cis / (allz.ncov + 1)
naive_top = set(allz.sort_values("tickets", ascending=False).head(15).gh7)
gap_top = set(allz.sort_values("gap", ascending=False).head(15).gh7)
nm = {h["gh"]: h["name"] for h in hot}
missed = list(gap_top - naive_top); over = list(naive_top - gap_top)
reveal = dict(naive_top=list(naive_top), corrected_top=list(gap_top), n_missed=len(missed), n_over=len(over),
    naive_named=[name_of(g) for g in allz.sort_values("tickets", ascending=False).head(8).gh7],
    corrected_named=[name_of(g) for g in allz.sort_values("gap", ascending=False).head(8).gh7],
    missed_names=[name_of(x) for x in missed][:6],
    over_names=[name_of(x) for x in over][:6], overlap=len(naive_top & gap_top))

# ---------- FORECAST @ gh6 (Tweedie) ----------
daily = df.groupby(["gh6", "date"]).agg(cis=("impact", "sum")).reset_index()
zt = daily.groupby("gh6").size(); keep = zt[zt >= 20].index; daily = daily[daily.gh6.isin(keep)]
dates = pd.date_range(daily.date.min(), daily.date.max(), freq="D")
full = pd.MultiIndex.from_product([keep, dates], names=["gh6", "date"]).to_frame(index=False)
d = full.merge(daily, on=["gh6", "date"], how="left").fillna({"cis": 0}).sort_values(["gh6", "date"])
gg = d.groupby("gh6")
d["dow"] = d.date.dt.dayofweek; d["is_wknd"] = (d.dow >= 5).astype(int); d["month"] = d.date.dt.month; d["dom"] = d.date.dt.day
for L in [1, 2, 3, 7, 14]: d[f"lag{L}"] = gg.cis.shift(L)
d["roll7"] = gg.cis.shift(1).rolling(7).mean().reset_index(0, drop=True)
d["roll14"] = gg.cis.shift(1).rolling(14).mean().reset_index(0, drop=True)
d["roll7_max"] = gg.cis.shift(1).rolling(7).max().reset_index(0, drop=True)
d["trend"] = d.roll7 - d.roll14
d["zone_mean"] = gg.cis.apply(lambda s: s.shift(1).expanding().mean()).reset_index(0, drop=True)
d["dow_mean"] = gg.apply(lambda x: x.sort_values("date").assign(m=x.cis.shift(1).expanding().mean()).m).reset_index(0, drop=True)
d["active_ratio7"] = gg.cis.apply(lambda s: (s.shift(1) > 0).rolling(7).mean()).reset_index(0, drop=True)
d = d.dropna(subset=["lag14", "roll14", "zone_mean"]).copy()
F = ["dow", "is_wknd", "month", "dom", "lag1", "lag2", "lag3", "lag7", "lag14", "roll7", "roll14", "roll7_max", "trend", "zone_mean", "dow_mean", "active_ratio7"]
cut = pd.Timestamp("2024-03-01"); tr = d[d.date < cut]; te = d[d.date >= cut].copy()
m = lgb.LGBMRegressor(n_estimators=400, learning_rate=.05, objective="tweedie", tweedie_variance_power=1.3, random_state=0, verbose=-1)
m.fit(tr[F], tr.cis); te["pred"] = np.clip(m.predict(te[F]), 0, None)
def hitk(df_, k):
    hh = []
    for _, gp in df_.groupby("date"):
        if gp.cis.sum() == 0: continue
        a = set(gp.nlargest(k, "cis").gh6); p = set(gp.nlargest(k, "pred").gh6); hh.append(len(a & p) / min(k, len(a)))
    return float(np.mean(hh))
exp = te.groupby("gh6").agg(pred=("pred", "mean"), actual=("cis", "mean")).reset_index()
def period_hit(k):
    a = set(exp.nlargest(k, "actual").gh6); p = set(exp.nlargest(k, "pred").gh6); return len(a & p) / k
caps = []
for _, gp in te.groupby("date"):
    if gp.cis.sum() == 0: continue
    caps.append(gp.nlargest(10, "pred").cis.sum() / gp.cis.sum())
gh6name = {}
for gh in exp.gh6:
    sub = df[df.gh6 == gh].location.dropna()
    if len(sub) == 0:
        gh6name[gh] = gh; continue
    pk, cd = _pick(_segs(sub.mode().iloc[0]))
    gh6name[gh] = pk or (cd[0][:34] if cd else gh)
metrics = dict(single_day_hit10=round(hitk(te, 10), 3), period_hit10=round(period_hit(10), 2),
    period_hit20=round(period_hit(20), 2), capture_pct=round(float(np.mean(caps)) * 100, 1),
    train_span=f"{tr.date.min().date()} to {tr.date.max().date()}",
    test_span=f"{te.date.min().date()} to {te.date.max().date()}")
fr = exp.sort_values("pred", ascending=False).head(8)
forecast_zones = [dict(gh=r.gh6, name=gh6name.get(r.gh6, r.gh6), pred=round(float(r.pred), 1)) for _, r in fr.iterrows()]
ptop = list(exp.nlargest(10, "pred").gh6); atop = set(exp.nlargest(10, "actual").gh6)
validation = [dict(name=gh6name.get(z, z), correct=(z in atop)) for z in ptop]

# ---------- REALLOCATION (maximal covering) ----------
zc = g.agg(cis=("impact", "sum"), ncov=("device_id", "nunique")).reset_index()
UNITS = 10; total = zc.cis.sum()
cur = set(zc.sort_values("ncov", ascending=False).head(UNITS).gh7)
opt = set(zc.sort_values("cis", ascending=False).head(UNITS).gh7)
cov_before = zc[zc.gh7.isin(cur)].cis.sum() / total * 100
cov_after = zc[zc.gh7.isin(opt)].cis.sum() / total * 100
movefrom = list(cur - opt); moveto = list(opt - cur); moves = []
for f_, t_ in zip(movefrom, moveto):
    moves.append(dict(frm=name_of(f_), to=name_of(t_),
        gain=int(zc[zc.gh7 == t_].cis.iloc[0] - zc[zc.gh7 == f_].cis.iloc[0])))
realloc = dict(units=UNITS, cov_before=round(cov_before, 1), cov_after=round(cov_after, 1), n_moves=len(moveto), moves=moves[:5])

# ---------- QUALITY: rejection rate ----------
latr = pd.to_numeric(RAW.latitude, errors="coerce"); lonr = pd.to_numeric(RAW.longitude, errors="coerce")
R2 = RAW[latr.notna()].copy()
R2["gh7"] = [pgh.encode(a, o, 7) for a, o in zip(latr[latr.notna()], lonr[latr.notna()])]
q = R2.groupby("gh7").agg(n=("rejected", "size"), rej=("rejected", "sum")).reset_index()
q = q[q.n >= 80].copy(); q["rate"] = q.rej / q.n * 100
qworst = q.sort_values("rate", ascending=False).head(8)
quality = dict(overall_reject=round(RAW.rejected.mean() * 100, 1),
    worst=[dict(name=(name_of(r.gh7) if r.gh7 in set(df.gh7) else r.gh7), rate=round(float(r.rate), 1), n=int(r.n)) for _, r in qworst.iterrows()])

# ---------- PER-DAY data for date selection on the map ----------
topgh = [h["gh"] for h in hot]
dd = df[df.gh7.isin(topgh)].groupby(["date", "gh7"])["impact"].sum().reset_index()
pivd = dd.pivot(index="date", columns="gh7", values="impact").fillna(0)
day_data = {}
for dt_, row in pivd.iterrows():
    vals = {gh: int(v) for gh, v in row.items() if v > 0}
    if vals:
        day_data[dt_.strftime("%Y-%m-%d")] = vals
dates_list = sorted(day_data.keys())

# ---------- WRITE ----------
meta = dict(records=N_RAW, events=N_EVENTS, span=f"{RAW.date.min().date()} to {RAW.date.max().date()}",
    hotspots=int((allz.cis > 0).sum()), chronic=int((H.days >= 10).sum()),
    gap_zones=int(reveal["n_missed"]), total_cis=int(df.impact.sum()))
DATA = dict(meta=meta, hotspots=hot, reveal=reveal,
    forecast=dict(metrics=metrics, zones=forecast_zones, validation=validation),
    reallocation=realloc, quality=quality, day_data=day_data, dates=dates_list)
out = os.path.join(ROOT, "app", "data.js")
with open(out, "w") as f:
    f.write("window.CHOKEPOINT=" + json.dumps(DATA, indent=1) + ";")
print("wrote", out)
print(json.dumps(meta, indent=1))
print("forecast:", metrics)
print("reveal:", {k: reveal[k] for k in ['n_missed', 'n_over', 'overlap']})
print("realloc:", {k: realloc[k] for k in ['cov_before', 'cov_after', 'n_moves']})
print("quality overall reject %:", quality["overall_reject"])
