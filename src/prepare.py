#!/usr/bin/env python3
"""Precompute ChokePoint cache (station-centric rebuild).
Run once after placing the CSV:  python src/prepare.py
Produces cache/: events.parquet, stpanel.parquet, ranker.pkl, names.json, stations.json, static.json"""
import os, json, pickle, warnings
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import lightgbm as lgb
import engine as E

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "cache"); os.makedirs(CACHE, exist_ok=True)
CSV = os.environ.get("CSV", os.path.join(ROOT, "data", "violations.csv"))
if not os.path.exists(CSV):
    CSV = "/home/claude/t1/jan to may police violation_anonymized791b166.csv"

print("loading + cleaning", CSV)
df, raw = E.load_clean(CSV)
df = df[df.police_station.notna()].copy()
df["ps"] = df.police_station.astype(str)
df["w"] = df.hour.map(E.window_of)
print(f"  {len(raw)} rows -> {len(df)} events | stations={df.ps.nunique()}")

# ---- events cache (now WITH police_station + window) ----
keep = ["date", "hour", "minute", "w", "dow", "gh7", "gh6", "lat", "lon", "impact", "poff", "vclass", "device_id", "created_by_id", "location", "ps"]
df[keep].to_parquet(os.path.join(CACHE, "events.parquet"))
print("  wrote events.parquet (with police_station)")

# ---- geohash name maps (for the map) ----
names = E.name_maps(df)
gh6names = {}
for gh, g in df.groupby("gh6"):
    loc = g.location.dropna()
    nm, _ = E.pick_name(loc.mode().iloc[0]) if len(loc) else (gh, "")
    gh6names[gh] = nm or gh
json.dump({"gh7": names, "gh6": gh6names}, open(os.path.join(CACHE, "names.json"), "w"))
print("  wrote names.json")

# ---- station metadata: centroids + nearest neighbours (for routing/coverage) ----
cen = E.station_centroids(df)
stations = sorted(cen.keys())
# nearest neighbours within 5km for coverage
neigh = {}
for s in stations:
    ds = []
    for o in stations:
        if o == s: continue
        d = E.haversine(cen[s]["lat"], cen[s]["lon"], cen[o]["lat"], cen[o]["lon"])
        if d <= 5.0: ds.append((o, round(float(d), 2)))
    neigh[s] = sorted(ds, key=lambda x: x[1])[:4]
json.dump({"centroids": cen, "neighbours": neigh}, open(os.path.join(CACHE, "stations.json"), "w"))
print(f"  wrote stations.json ({len(stations)} stations)")

# ---- station x window panel + LambdaRank model ----
WINS = 7
df_w = df[df.w >= 0]
g = df_w.groupby(["ps", "date", "w"])["impact"].sum().reset_index()
dates = pd.date_range(g.date.min(), g.date.max(), freq="D")
idx = pd.MultiIndex.from_product([stations, dates, range(WINS)], names=["ps", "date", "w"])
p = g.set_index(["ps", "date", "w"]).reindex(idx, fill_value=0).reset_index().sort_values(["ps", "w", "date"])
p["dow"] = p.date.dt.dayofweek
p["is_wknd"] = (p.dow >= 5).astype(int)
p["is_hol"] = p.date.map(E.is_holiday).astype(int)
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
# next-3-day smoothed target (for the outlook horizon)
p["fwd3"] = gw.impact.transform(lambda s: s.shift(-1).rolling(3).mean())
F = ["psid", "w", "dow", "is_wknd", "is_hol", "lag1", "lag2", "lag7", "roll7", "roll7max", "zw_mean", "dow_mean", "regime_mean"]
psid_map = dict(zip(p.ps, p.psid))

def graded(d_, tgt):
    """relevance: top5=3, top10=2, top15=1 within each (date,w) group."""
    d_ = d_.copy(); d_["rel"] = 0
    for (_, _), gp in d_.groupby(["date", "w"]):
        d_.loc[gp.nlargest(15, tgt).index, "rel"] = 1
        d_.loc[gp.nlargest(10, tgt).index, "rel"] = 2
        d_.loc[gp.nlargest(5, tgt).index, "rel"] = 3
    return d_

P = p.dropna(subset=["lag7", "roll7", "zw_mean"]).copy()
tr = P[P.date < E.CUT].copy()

def train_ranker(target):
    t = graded(tr.dropna(subset=([target] if target == "fwd3" else [])), target).sort_values(["date", "w"])
    grp = t.groupby(["date", "w"]).size().values
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=500, learning_rate=.05,
                       random_state=0, verbose=-1, label_gain=[0, 1, 3, 7])
    m.fit(t[F], t["rel"], group=grp)
    return m

rk_day = train_ranker("impact")
rk_out = train_ranker("fwd3")
P.to_parquet(os.path.join(CACHE, "stpanel.parquet"))
pickle.dump({"ranker_day": rk_day, "ranker_outlook": rk_out, "features": F,
             "psid_map": psid_map, "cut": str(E.CUT.date()), "wins": WINS},
            open(os.path.join(CACHE, "ranker.pkl"), "wb"))
print(f"  trained LambdaRank (day+outlook) on {len(tr)} rows; wrote stpanel.parquet + ranker.pkl")

# ---- station recurring profile (typical hot windows) ----
prof = tr.groupby(["ps", "w"]).impact.mean().reset_index()
profile = {}
for ps_, gp in prof.groupby("ps"):
    profile[ps_] = {int(r.w): round(float(r.impact), 2) for _, r in gp.iterrows()}

# ---- holiday over-index watch-list ----
df_h = df.assign(hol=df.date.map(E.is_holiday))
base_share = df.groupby("ps").impact.sum() / df.impact.sum()
hol_share = df_h[df_h.hol].groupby("ps").impact.sum()
watch = []
if hol_share.sum() > 0:
    hs = hol_share / hol_share.sum()
    lift = (hs / base_share).replace([np.inf], np.nan).dropna().sort_values(ascending=False)
    watch = [{"station": k, "lift": round(float(v), 1)} for k, v in lift.head(6).items() if v > 1.2]

# ---- station-level period reveal (where patrols go vs real choke points) ----
gs = df.groupby("ps").agg(cis=("impact", "sum"), tickets=("impact", "size"), ncov=("device_id", "nunique")).reset_index()
gs["gap"] = gs.cis / (gs.ncov + 1)
N = 12
naive = gs.sort_values("tickets", ascending=False).head(N)
real = gs.sort_values("gap", ascending=False).head(N)
naive_set, real_set = set(naive.ps), set(real.ps)
over = list(naive_set - real_set); missed = list(real_set - naive_set)
cenpt = lambda s: {"name": s, "lat": cen[s]["lat"], "lon": cen[s]["lon"]}

static = {
    "meta": {"records": int(len(raw)), "events": int(len(df)),
             "span": f"{df.date.min().date()} to {df.date.max().date()}",
             "stations": len(stations), "cut": str(E.CUT.date())},
    "stations": stations,
    "profile": profile,
    "holiday_watch": watch,
    "holidays": E.HOLIDAYS,
    "reveal": {"naive": [cenpt(s) for s in naive.ps], "real": [cenpt(s) for s in real.ps],
               "overlap": len(naive_set & real_set), "n_missed": len(missed), "n_over": len(over)},
    "dates": sorted(df.date.dt.strftime("%Y-%m-%d").unique().tolist()),
}
json.dump(static, open(os.path.join(CACHE, "static.json"), "w"))
print("  wrote static.json")
print("DONE. cache ready ->", CACHE)
