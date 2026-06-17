#!/usr/bin/env python3
"""ChokePoint API server (station-centric rebuild).
Run: cd src && uvicorn server:app --port 8000
Forecast = LambdaRank at police-station x 2h-window. Map = geohash detail."""
import os, json, pickle, warnings
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import engine as E

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "cache"); APP = os.path.join(ROOT, "app")

EV = pd.read_parquet(os.path.join(CACHE, "events.parquet")); EV["date"] = pd.to_datetime(EV["date"])
PANEL = pd.read_parquet(os.path.join(CACHE, "stpanel.parquet")); PANEL["date"] = pd.to_datetime(PANEL["date"])
RK = pickle.load(open(os.path.join(CACHE, "ranker.pkl"), "rb"))
NAMES = json.load(open(os.path.join(CACHE, "names.json")))
GH7N, GH6N = NAMES["gh7"], NAMES["gh6"]
ST = json.load(open(os.path.join(CACHE, "stations.json")))
CEN, NEIGH = ST["centroids"], ST["neighbours"]
STATIC = json.load(open(os.path.join(CACHE, "static.json")))
CUT = pd.Timestamp(STATIC["meta"]["cut"]); F = RK["features"]; PSID = RK["psid_map"]; WINS = RK["wins"]
OFFLAB = {0: "Defective plate", 104: "Near crossing", 105: "On footpath", 107: "On main road",
          108: "Opposite parked", 109: "Double parking", 111: "Bus-stop/school", 112: "Wrong parking", 113: "No parking"}

app = FastAPI()

class Ask(BaseModel):
    q: str
    date: str | None = None
    station: str | None = None

# ---------- helpers ----------
def evday(date, station=None):
    d = EV[EV.date == pd.Timestamp(date)]
    if station and station != "All": d = d[d.ps == station]
    return d

def gh_hotspots(d, topn=40):
    """geohash hotspots from an events slice, deduped by road name."""
    if len(d) == 0: return []
    g = d.groupby("gh7").agg(cis=("impact", "sum"), tickets=("impact", "size"),
                             coverage=("device_id", "nunique")).reset_index()
    g["gap"] = g.cis / (g.coverage + 1)
    g = g.sort_values("cis", ascending=False).head(topn * 2)
    mx = g.cis.max() if len(g) else 1
    out, seen = [], set()
    for _, r in g.iterrows():
        m = GH7N.get(r.gh7, {"name": r.gh7, "area": "", "lat": float(d[d.gh7 == r.gh7].lat.mean()), "lon": float(d[d.gh7 == r.gh7].lon.mean())})
        nm = m["name"]
        if nm in seen: continue
        seen.add(nm)
        sub = d[d.gh7 == r.gh7]
        off = sub.groupby("poff")["impact"].sum().sort_values(ascending=False).head(4); offt = off.sum() or 1
        veh = sub.vclass.value_counts().head(3); veht = veh.sum() or 1
        hours = sub.groupby("hour").size().reindex(range(24), fill_value=0); hmax = hours.max() or 1
        out.append({"gh": r.gh7, "name": nm, "area": m.get("area", ""), "lat": m["lat"], "lon": m["lon"],
                    "cis": int(r.cis), "cis_norm": int(round(r.cis / mx * 100)), "tickets": int(r.tickets),
                    "coverage": int(r.coverage), "gap": round(float(r.gap), 1), "ps": sub.ps.mode().iloc[0] if len(sub) else "",
                    "peak": int(sub.hour.mode().iloc[0]) if len(sub) else 10,
                    "offmix": [[OFFLAB.get(k, str(k)), int(round(v / offt * 100))] for k, v in off.items()],
                    "vehmix": [[k, int(round(v / veht * 100))] for k, v in veh.items()],
                    "hours": [int(round(x / hmax * 100)) for x in hours]})
        if len(out) >= topn: break
    return out

def forecast_stations(date, horizon="day", station=None):
    """Rank police stations per window for the day AFTER `date`."""
    nxt = pd.Timestamp(date) + pd.Timedelta(days=1)
    rows = PANEL[PANEL.date == nxt]
    if len(rows) == 0: return None
    rk = RK["ranker_day"] if horizon == "day" else RK["ranker_outlook"]
    r = rows.copy(); r["score"] = rk.predict(r[F])
    # actual next-day impact per station-window (for accuracy view)
    out_windows = []
    hit_total = 0; cap_total = 0; nwin = 0
    for w in range(WINS):
        sub = r[r.w == w].copy()
        if len(sub) == 0: continue
        sub = sub.sort_values("score", ascending=False)
        actual_top = set(sub.nlargest(10, "impact").ps)
        pred_top = list(sub.head(10).ps)
        hits = sum(1 for s in pred_top if s in actual_top)
        tot = sub.impact.sum()
        cap = float(sub.head(10).impact.sum() / tot * 100) if tot else 0
        hit_total += hits; cap_total += cap; nwin += 1
        items = [{"station": x.ps, "score": round(float(x.score), 2), "pred_impact": round(float(x.impact), 1),
                  "correct": x.ps in actual_top, "lat": CEN[x.ps]["lat"], "lon": CEN[x.ps]["lon"]}
                 for x in sub.head(10).itertuples()]
        out_windows.append({"w": w, "label": E.WINDOW_LABELS[w], "items": items})
    if station and station != "All":
        for ow in out_windows:
            ow["items"] = [it for it in ow["items"] if it["station"] == station]
    # day-level rollup: rank stations by recent rolling pattern (roll7 summed over windows) = the
    # strong, honest "expected hotspots" baseline (~6.8/10); the window model handles shift-level timing.
    dayr = r.groupby("ps").agg(expected=("roll7", "sum"), actual=("impact", "sum")).reset_index()
    pred_set = set(dayr.nlargest(10, "expected").ps)
    act_set = set(dayr.nlargest(10, "actual").ps)
    act20 = set(dayr.nlargest(20, "expected").ps)
    day_hit = len(pred_set & act_set)
    tot_act = dayr.actual.sum() or 1
    cap10 = float(dayr[dayr.ps.isin(pred_set)].actual.sum() / tot_act * 100)
    cap20 = float(dayr[dayr.ps.isin(act20)].actual.sum() / tot_act * 100)
    emax = dayr.expected.max() or 1
    amax = dayr.actual.max() or 1
    day_items = [{"station": x.ps, "pred": int(round(x.expected / emax * 100)),
                  "actual": int(round(x.actual / amax * 100)), "raw_actual": round(float(x.actual), 1),
                  "predicted": x.ps in pred_set, "correct": x.ps in act_set,
                  "lat": CEN[x.ps]["lat"], "lon": CEN[x.ps]["lon"]}
                 for x in dayr.sort_values("expected", ascending=False).head(12).itertuples()]
    day_pred = {"hit": day_hit, "cap10": round(cap10, 1), "cap20": round(cap20, 1), "items": day_items,
                "pred_list": list(dayr.nlargest(10, "expected").ps), "act_list": list(dayr.nlargest(10, "actual").ps)}
    return {"date": str(date), "next": str(nxt.date()), "is_holdout": bool(nxt >= CUT),
            "hit": round(hit_total / nwin, 1) if nwin else 0, "capture": round(cap_total / nwin, 1) if nwin else 0,
            "windows": out_windows, "horizon": horizon, "day_pred": day_pred}

# ---------- endpoints ----------
@app.get("/api/init")
def init():
    dates = STATIC["dates"]; dset = set(dates)
    held = [d for d in dates if d >= STATIC["meta"]["cut"] and (pd.Timestamp(d) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") in dset]
    train = [d for d in dates if d < STATIC["meta"]["cut"]]
    de = EV.groupby(EV.date.dt.strftime("%Y-%m-%d")).size()
    trend = [{"date": d, "events": int(de.get(d, 0))} for d in held]
    return {"dates": dates, "holdout_dates": held, "stations": STATIC["stations"],
            "train_span": f"{train[0]} to {train[-1]}" if train else "", "n_train": len(train), "n_holdout": len(held),
            "cut": STATIC["meta"]["cut"], "span": STATIC["meta"]["span"], "records": STATIC["meta"]["records"],
            "events": STATIC["meta"]["events"], "holidays": STATIC["holidays"], "trend": trend}

@app.get("/api/day")
def day(date: str = Query(...), station: str = Query("All")):
    d = evday(date, station)
    hot = gh_hotspots(d)
    tot = float(d.impact.sum()) or 1
    top10 = sum(h["cis"] for h in hot[:10])
    dayh = d[(d.hour >= 7) & (d.hour <= 21)]
    peak = int(dayh.hour.mode().iloc[0]) if len(dayh) else (int(d.hour.mode().iloc[0]) if len(d) else 10)
    is_hol = E.is_holiday(date)
    hourly = [int(x) for x in d.groupby("hour").size().reindex(range(24), fill_value=0)]
    veh = d.vclass.value_counts().head(4)
    off = d.poff.map(lambda c: OFFLAB.get(c, str(c))).value_counts().head(5)
    charts = {"hourly": hourly,
              "vehicle": [[str(k), int(v)] for k, v in veh.items()],
              "offence": [[str(k), int(v)] for k, v in off.items()]}
    return {"date": date, "station": station, "hotspots": hot, "charts": charts,
            "kpis": {"events": int(len(d)), "hotspots": int(d.gh7.nunique()),
                     "top10_share": int(round(top10 / tot * 100)), "peak_hour": peak,
                     "is_holiday": bool(is_hol), "holiday_name": STATIC["holidays"].get(str(pd.Timestamp(date).date()), "")},
            "holiday_watch": STATIC["holiday_watch"] if is_hol else []}

@app.get("/api/reveal")
def reveal(date: str = Query(None), station: str = Query("All")):
    """Per-date station reveal + same-day move recommendation (react-now)."""
    if date:
        d = evday(date)
        gs = d.groupby("ps").agg(cis=("impact", "sum"), tickets=("impact", "size"), ncov=("device_id", "nunique")).reset_index()
    else:
        gs = None
    if gs is None or len(gs) < 4:
        R = STATIC["reveal"]
        return {**R, "scope": "period", "moves": []}
    gs["gap"] = gs.cis / (gs.ncov + 1)
    day_total = float(gs.cis.sum()) or 1
    N = min(10, len(gs))
    naive = gs.sort_values("tickets", ascending=False).head(N)
    real = gs.sort_values("gap", ascending=False).head(N)
    ns, rs = set(naive.ps), set(real.ps)
    over = list(ns - rs); missed = list(rs - ns)
    cism = dict(zip(gs.ps, gs.cis))
    srcs = sorted(over, key=lambda x: cism.get(x, 0))
    dsts = sorted(missed, key=lambda x: -cism.get(x, 0))
    moves = []
    for f_, t_ in zip(srcs, dsts):
        gain = int(cism.get(t_, 0) - cism.get(f_, 0))
        if gain > 0:
            moves.append({"frm": f_, "to": t_, "gain": gain, "pct": round(cism.get(t_, 0) / day_total * 100, 1),
                          "frm_lat": CEN[f_]["lat"], "frm_lon": CEN[f_]["lon"], "to_lat": CEN[t_]["lat"], "to_lon": CEN[t_]["lon"]})
    moves.sort(key=lambda m: -m["gain"])
    cenpt = lambda s: {"name": s, "lat": CEN[s]["lat"], "lon": CEN[s]["lon"]}
    covered_pct = round(sum(m["pct"] for m in moves[:5]), 1)
    return {"naive": [cenpt(s) for s in naive.ps], "real": [cenpt(s) for s in real.ps],
            "overlap": len(ns & rs), "n_missed": len(missed), "n_over": len(over),
            "scope": "day", "moves": moves[:5], "date": date, "covered_pct": covered_pct}

@app.get("/api/forecast")
def forecast(date: str = Query(...), horizon: str = Query("day"), station: str = Query("All")):
    f = forecast_stations(date, horizon, station)
    if f is None:
        return {"date": date, "next": str((pd.Timestamp(date) + pd.Timedelta(days=1)).date()), "has_next": False, "windows": []}
    f["has_next"] = True
    f["base_hit"] = _persistence_shift_hit(pd.Timestamp(date), pd.Timestamp(date) + pd.Timedelta(days=1))
    f["score_agg"] = score_agg()
    return f

def _persistence_shift_hit(date_ts, nxt_ts):
    """Baseline 'tomorrow = today': predict next-day per-shift top-10 = this date's per-shift top-10."""
    rN = PANEL[PANEL.date == date_ts]; rX = PANEL[PANEL.date == nxt_ts]
    if len(rX) == 0 or len(rN) == 0: return None
    tot = 0; nw = 0
    for w in range(WINS):
        sx = rX[rX.w == w]; sn = rN[rN.w == w]
        if len(sx) == 0 or len(sn) == 0: continue
        actual = set(sx.nlargest(10, "impact").ps)
        tot += len(set(sn.nlargest(10, "impact").ps) & actual); nw += 1
    return round(tot / nw, 1) if nw else None

_SCORE_AGG = None
def score_agg():
    """Average performance over every held-out next-day (computed once, cached).
    Shift-level: model vs 'tomorrow=today' baseline (where the model wins).
    Day-level 'where': recent-pattern accuracy (the model is NOT used here — it scores lower)."""
    global _SCORE_AGG
    if _SCORE_AGG is not None: return _SCORE_AGG
    import numpy as np
    P = PANEL.copy(); P["score"] = RK["ranker_day"].predict(P[F])
    dates = sorted(P.date.unique())
    M = []; B = []; WH = []; C20 = []
    for i, d in enumerate(dates[:-1]):
        nxt = dates[i + 1]
        if pd.Timestamp(nxt) < CUT: continue
        rN = P[P.date == d]; rX = P[P.date == nxt]
        if len(rX) == 0: continue
        mh = bh = 0; nw = 0
        for w in range(WINS):
            sx = rX[rX.w == w]; sn = rN[rN.w == w]
            if len(sx) == 0: continue
            actual = set(sx.nlargest(10, "impact").ps)
            mh += len(set(sx.nlargest(10, "score").ps) & actual)
            if len(sn): bh += len(set(sn.nlargest(10, "impact").ps) & actual)
            nw += 1
        if nw == 0: continue
        M.append(mh / nw); B.append(bh / nw)
        dn = rX.groupby("ps").agg(roll=("roll7", "sum"), act=("impact", "sum")).reset_index()
        a = set(dn.nlargest(10, "act").ps)
        WH.append(len(set(dn.nlargest(10, "roll").ps) & a))
        tot = dn.act.sum() or 1
        C20.append(dn[dn.ps.isin(set(dn.nlargest(20, "roll").ps))].act.sum() / tot * 100)
    _SCORE_AGG = {"n_days": len(M),
                  "model_shift": round(float(np.mean(M)), 1), "base_shift": round(float(np.mean(B)), 1),
                  "edge": round(float(np.mean(M) - np.mean(B)), 1),
                  "where_hit": round(float(np.mean(WH)), 1), "cap20": int(round(float(np.mean(C20))))}
    return _SCORE_AGG

@app.get("/api/schedule")
def schedule(date: str = Query(...), teams: int = Query(10), station: str = Query("All")):
    """Plan-ahead: assign teams to predicted top stations per window with nearest-station coverage."""
    f = forecast_stations(date, "day")
    if f is None: return {"empty": True, "blocks": []}
    blocks = []
    covered = set()
    for ow in f["windows"]:
        ranked = ow["items"]  # already top-10 by score
        assign = []
        used = set()
        for it in ranked:
            if len(assign) >= teams: break
            s = it["station"]
            if s in used: continue
            # this team also covers nearby stations within 5km
            nb = [n for n, dist in NEIGH.get(s, [])][:3]
            assign.append({"team": len(assign) + 1, "station": s, "score": it["score"],
                           "covers": nb, "lat": it["lat"], "lon": it["lon"]})
            used.add(s); used.update(nb); covered.update([s] + nb)
        blocks.append({"w": ow["w"], "label": ow["label"], "assign": assign})
    if station and station != "All":
        for b in blocks:
            b["assign"] = [a for a in b["assign"] if a["station"] == station or station in a["covers"]]
    return {"empty": False, "blocks": blocks, "teams": teams, "next": f["next"], "is_holdout": f["is_holdout"],
            "hit": f["hit"], "capture": f["capture"]}

@app.get("/api/playbook")
def playbook(date: str = Query(...), station: str = Query("All")):
    """Station-centric deployment grid for the day AFTER `date`.
    Each station has its own team. Per 2-hour window we classify it:
      need  = predicted top tier (a major choke point) -> gets a nearby team sent
      hold  = predicted hot, covers its own area
      help  = predicted quiet AND near a 'need' station -> its team is sent there
      routine = quiet and not near any need station
    Returns both the plan (status/target) and the predicted situation (load) per cell."""
    nxt = pd.Timestamp(date) + pd.Timedelta(days=1)
    rows = PANEL[PANEL.date == nxt]
    if len(rows) == 0:
        return {"empty": True, "windows": [], "rows": [], "cells": {}, "pairs": []}
    r = rows.copy(); r["score"] = RK["ranker_day"].predict(r[F])
    NEED, HOT = 5, 12
    cells = {}; pairs = []
    for w in range(WINS):
        sub = r[r.w == w].sort_values("score", ascending=False)
        if len(sub) == 0: continue
        smax = sub.score.max(); smin = sub.score.min()
        need = list(sub.head(NEED).ps)
        hot = list(sub.head(HOT).ps)
        quiet = set(sub.ps) - set(hot)
        used = set(); win_pairs = []
        for ns in need:                       # send nearest quiet neighbour within 5km
            for nb, dist in NEIGH.get(ns, []):
                if nb in quiet and nb not in used:
                    used.add(nb); win_pairs.append({"need": ns, "helper": nb, "dist": dist}); break
        helper_of = {p["helper"]: p["need"] for p in win_pairs}
        for x in sub.itertuples():
            load = int(round((x.score - smin) / (smax - smin) * 100)) if smax > smin else 50
            if x.ps in helper_of:
                stt, tgt = "help", helper_of[x.ps]
            elif x.ps in need:
                stt, tgt = "need", None
            elif x.ps in hot:
                stt, tgt = "hold", None
            else:
                stt, tgt = "routine", None
            cells.setdefault(x.ps, {})[w] = {"s": stt, "t": tgt, "load": load}
        pairs.append({"w": w, "label": E.WINDOW_LABELS[w], "pairs": win_pairs})
    # active = stations with a non-routine action in at least one window
    active = [s for s, wd in cells.items() if any(v["s"] != "routine" for v in wd.values())]
    active.sort(key=lambda s: -sum(cells[s].get(w, {}).get("load", 0) for w in range(WINS)))
    if station and station != "All":
        active = [s for s in active if s == station] or active[:0]
    active = active[:22]
    return {"empty": False, "next": str(nxt.date()), "is_holdout": bool(nxt >= CUT),
            "windows": [{"w": w, "label": E.WINDOW_LABELS[w]} for w in range(WINS)],
            "rows": active, "cells": {s: cells[s] for s in active}, "pairs": pairs}

@app.get("/api/replay")
def replay(date: str = Query(...), station: str = Query("All")):
    """30-minute snapshots of the day for the live replay: cumulative map,
    a move-a-team recommendation, and the standing top stations."""
    d = evday(date, station).copy()
    if len(d) == 0: return {"buckets": [], "total": 0, "next_pred": []}
    if "minute" not in d.columns: d["minute"] = 0
    d["mins"] = d.hour * 60 + d.minute
    buckets = []
    f = forecast_stations(date, "day")
    next_pred = (f["day_pred"]["pred_list"][:6] if f else [])
    next_day = f["next"] if f else ""
    first_seen = {}
    for slot in range(6 * 60, 20 * 60 + 1, 30):       # 06:00 .. 20:00 every 30 min
        sub = d[d.mins <= slot]
        win = d[(d.mins > slot - 30) & (d.mins <= slot)]   # last 30 min
        clock = f"{slot // 60:02d}:{slot % 60:02d}"
        g = sub.groupby("gh7").agg(cis=("impact", "sum")).reset_index().sort_values("cis", ascending=False).head(15)
        mx = g.cis.max() if len(g) else 1
        pts = [{"name": GH7N.get(r.gh7, {}).get("name", r.gh7), "lat": GH7N.get(r.gh7, {}).get("lat"),
                "lon": GH7N.get(r.gh7, {}).get("lon"), "v": int(round(r.cis / mx * 100))} for _, r in g.iterrows() if GH7N.get(r.gh7)]
        st = sub.groupby("ps").agg(cis=("impact", "sum"), ncov=("device_id", "nunique")).reset_index().sort_values("cis", ascending=False)
        total = float(st.cis.sum()) or 1
        smx = float(st.cis.iloc[0]) if len(st) else 1
        worst = [{"name": r.ps, "v": int(round(r.cis / smx * 100)), "share": round(r.cis / total * 100, 1)}
                 for r in st.head(10).itertuples()]
        active_chokes = int((st.cis / total * 100 >= 6).sum()) if len(st) else 0
        active_chokes = max(active_chokes, 1 if len(st) else 0)
        covered = int(round(st.head(10).cis.sum() / total * 100))
        top_st = list(st.head(3).ps)
        for s in top_st:
            first_seen.setdefault(s, clock)
        new_n = int(len(win))
        reco = ""
        if len(st):
            emerging = None
            if len(win):
                hs = win.groupby("ps").impact.sum().sort_values(ascending=False)
                if len(hs): emerging = hs.index[0]
            reco = (f"{emerging} is building — move a nearby team there now." if emerging and emerging not in top_st
                    else f"Hold positions; {top_st[0]} remains the priority ({new_n} new in last 30 min).")
        buckets.append({"clock": clock, "events": int(len(sub)), "new": new_n, "pts": pts,
                        "top_stations": top_st, "reco": reco, "worst": worst,
                        "active_chokes": active_chokes, "covered_pct": covered})
    # ---- curated "day so far" timeline (deterministic; real stations, helper, share) ----
    timeline = []
    if buckets:
        fin = buckets[-1]["worst"]                       # day's worst stations overall
        timeline.append({"clock": "06:30", "tone": "calm", "tag": "calm",
                         "text": "Day begins — light, even traffic across the city.", "imp": None})
        clamp = lambda c: c if c > "06:30" else "07:00"
        # 2nd & 3rd worst -> "building" beats at the time they first entered the top
        for rank, tone in [(1, "hot"), (2, "warn")]:
            if len(fin) > rank:
                stn = fin[rank]["name"]
                timeline.append({"clock": clamp(first_seen.get(stn, "07:30")), "tone": tone, "tag": "building",
                                 "text": f"<b>{stn}</b> climbing toward a choke point.", "imp": None})
        # #1 worst -> ONE clearly-labelled RECOMMENDATION (never phrased as an executed event)
        if fin:
            stn = fin[0]["name"]; share = int(round(fin[0]["share"]))
            ac = max(clamp(first_seen.get(stn, "08:30")), "08:00")
            filtered = bool(station and station != "All")
            if filtered:
                # single-station view: only this station's data exists, so no citywide share and no neighbour redirect
                timeline.append({"clock": ac, "tone": "act", "tag": "recommended",
                                 "text": f"<b>{stn}</b> hits its busiest stretch around now — <b>recommended:</b> concentrate enforcement here through the peak.", "imp": None})
            else:
                helper = None; dist = None; top_names = {w["name"] for w in fin}
                for nb, dd in NEIGH.get(stn, []):
                    if nb not in top_names: helper, dist = nb, dd; break
                imp = f"{stn} alone carries ~{share}% of the day's enforcement load" if 0 < share < 60 else None
                if helper:
                    timeline.append({"clock": ac, "tone": "act", "tag": "recommended action",
                                     "text": f"<b>{stn}</b> is the busiest choke point. <b>Recommended:</b> send {helper}'s unit ({dist} km away) to cover it — a quieter nearby station, so no extra staff. Hold the rest.",
                                     "imp": imp, "target": helper, "dist": dist})
                else:
                    timeline.append({"clock": ac, "tone": "act", "tag": "recommended action",
                                     "text": f"<b>{stn}</b> is the busiest choke point — <b>recommended:</b> concentrate patrols here.", "imp": imp})
        timeline.sort(key=lambda b: b["clock"])
    # ---- end-of-day summary (shown in Live once the replay completes) ----
    summary = None
    if buckets:
        n = len(d)
        offs = d.poff.map(lambda c: OFFLAB.get(c, str(c))).value_counts().head(3)
        vehs = d.vclass.value_counts()
        vmap = {"Two-wheeler/Auto": "Two-wheelers & autos", "Car/Medium": "Cars", "Heavy": "Heavy vehicles"}
        ww = d.assign(_w=(d.hour // 2 * 2)).groupby("_w").impact.sum()
        pkw = int(ww.idxmax()) if len(ww) else 8
        summary = {"events": int(n), "locations": int(d.gh7.nunique()),
                   "worst": [w["name"] for w in buckets[-1]["worst"][:3]],
                   "offences": [[str(k), int(round(v / n * 100))] for k, v in offs.items()],
                   "vehicles": [[vmap.get(k, str(k)), int(round(v / n * 100))] for k, v in vehs.items()],
                   "peak_window": f"{pkw:02d}:00–{pkw + 2:02d}:00"}
    return {"buckets": buckets, "total": int(len(d)), "next_day": next_day,
            "next_pred": next_pred, "timeline": timeline, "summary": summary}

# ---------- assistant (data-grounded + contextual suggestions) ----------
def day_summary(date, station):
    d = evday(date, station)
    if len(d) == 0: return f"No recorded activity on {date}" + (f" for {station}." if station != "All" else ".")
    hot = gh_hotspots(d, 5)
    scope = f"in {station}" if station and station != "All" else "across Bengaluru"
    top = ", ".join(h["name"] for h in hot[:3])
    n = len(d)
    off = d.poff.map(lambda c: OFFLAB.get(c, str(c))).value_counts().head(2)
    offtxt = " and ".join(f"{str(k).lower()} ({round(v / n * 100)}%)" for k, v in off.items())
    veh = d.vclass.value_counts()
    vname = {"Two-wheeler/Auto": "two-wheelers & autos", "Car/Medium": "cars", "Heavy": "heavy vehicles"}.get(veh.index[0], str(veh.index[0]).lower())
    vpct = round(veh.iloc[0] / n * 100)
    dh = d[(d.hour >= 7) & (d.hour <= 21)]
    pk = int((dh if len(dh) else d).groupby("hour").impact.sum().idxmax())
    hol = f" It's a holiday ({STATIC['holidays'].get(str(pd.Timestamp(date).date()), '')})." if E.is_holiday(date) else ""
    return (f"On {date} {scope} there were {n:,} enforcement events across {d.gh7.nunique()} spots, busiest around {pk:02d}:00. "
            f"Worst areas: {top}. The congestion is driven mainly by {offtxt}, and the vehicles involved are mostly {vname} ({vpct}%).{hol}")

def rich_answer(q, date, station):
    ql = q.lower(); station = station or "All"
    if any(k in ql for k in ["worst", "top spot", "biggest", "hotspot"]):
        hot = gh_hotspots(evday(date, station), 5)
        if hot:
            h = hot[0]
            return {"action": {"tab": "map", "spot": h["gh"]}, "answer": f"The worst spot on {date} is {h['name']} ({h['ps']} station), peaking ~{h['peak']:02d}:00.",
                    "follow": ["When does it peak?", "What should we do there?", "Predict tomorrow", "Which stations are missed?"]}
    if any(k in ql for k in ["peak", "when does", "what time", "busiest", "what hour", "when is it"]):
        hot = gh_hotspots(evday(date, station), 5)
        if hot:
            h = hot[0]; d = evday(date, station); dh = d[(d.hour >= 7) & (d.hour <= 21)]
            bz = int((dh if len(dh) else d).groupby("hour").impact.sum().idxmax()) if len(d) else 9
            return {"action": {"tab": "map", "spot": h["gh"]},
                    "answer": f"{h['name']} is busiest around {h['peak']:02d}:00. Citywide on {date}, congestion peaks around {bz:02d}:00 — best to have teams in place just before then.",
                    "follow": ["What should we do there?", "What's the worst spot?", "Predict tomorrow", "Summarise today"]}
    if any(k in ql for k in ["what should we do", "what to do", "do there", "what do we do", "recommend", "advice", "what's wrong", "what is wrong"]):
        hot = gh_hotspots(evday(date, station), 5)
        if hot:
            h = hot[0]
            off = h["offmix"][0][0].lower() if h.get("offmix") else "parking violations"
            veh = (h["vehmix"][0][0] if h.get("vehmix") else "vehicles").lower()
            return {"action": {"tab": "map", "spot": h["gh"]},
                    "answer": f"At {h['name']}, the main issue is {off} (mostly {veh}). Station a patrol there around its {h['peak']:02d}:00 peak and clear the obstructing vehicles — it's one of the day's biggest choke points, so it pays back fast.",
                    "follow": ["When does it peak?", "Where should we deploy?", "What's the worst spot?", "Summarise today"]}
    if any(k in ql for k in ["move", "react", "right place", "over-patrol", "bias", "reallocat", "deploy now", "shift team"]) or ("miss" in ql and "station" in ql):
        R = reveal(date, station)
        moves = R.get("moves", [])
        if moves:
            m = moves[0]
            return {"action": {"tab": "reveal"}, "answer": f"Today, move a team to {m['to']} from {m['frm']} (+{m['gain']} impact). {R['n_missed']} stations are under-served relative to their congestion.",
                    "follow": ["Show all moves", "What's the worst spot?", "Predict tomorrow", "Plan tomorrow's deployment"]}
        return {"action": {"tab": "reveal"}, "answer": f"Patrols broadly match the hotspots on {date}; few clear mismatches today.",
                "follow": ["Predict tomorrow", "What's the worst spot?", "Plan tomorrow's deployment", "Summarise today"]}
    if any(k in ql for k in ["summar", "overview", "what happen", "brief"]) or (ql.strip() in ("today", "summarise today", "summarize today")):
        return {"action": {"tab": "map"}, "answer": day_summary(date, station),
                "follow": ["What's the worst spot?", "Where should we deploy today?", "Predict tomorrow", "Is this a holiday pattern?"]}
    if any(k in ql for k in ["tomorrow", "predict", "forecast", "next day", "outlook"]):
        f = forecast_stations(date, "outlook" if "outlook" in ql or "3" in ql else "day", station)
        if f and f["windows"]:
            top = ", ".join(dict.fromkeys(it["station"] for ow in f["windows"] for it in ow["items"]))[:80]
            return {"action": {"tab": "fc"}, "answer": f"For {f['next']}, the model flags these stations: {top}. It gets ~{f['hit']}/10 right on this held-out test ({f['capture']}% of impact).",
                    "follow": ["Show the patrol schedule", "Did yesterday's prediction work?", "Which stations are missed?", "What's the 3-day outlook?"]}
    if any(k in ql for k in ["deploy", "schedule", "patrol plan", "teams", "where should", "shift", "plan"]):
        return {"action": {"tab": "plan"}, "answer": f"The deployment plan assigns teams to the predicted top stations each shift, each covering nearby stations within ~5 km. Opening the schedule.",
                "follow": ["Did the plan work next day?", "What's the worst spot?", "Predict the 3-day outlook", "Move teams today instead?"]}
    if "holiday" in ql or "festival" in ql:
        w = STATIC["holiday_watch"]
        if E.is_holiday(date) and w:
            return {"action": {"tab": "map"}, "answer": f"{date} is a holiday. On holidays congestion relocates — watch: {', '.join(x['station'] for x in w[:4])}.",
                    "follow": ["What's the worst spot?", "Predict tomorrow", "Where should we deploy?", "Summarise today"]}
        return {"action": {"tab": "map"}, "answer": f"{date} is a normal working/weekend day — standard hotspot pattern applies.",
                "follow": ["What's the worst spot?", "Predict tomorrow", "Where should we deploy?", "Summarise today"]}
    return {"action": {"tab": "map"}, "answer": "I can summarise the day, find the worst spot, recommend same-day moves, predict tomorrow, or plan deployment. Pick one.",
            "follow": ["Summarise today", "What's the worst spot?", "Where should we deploy?", "Predict tomorrow"]}

@app.post("/api/ask")
def ask(a: Ask):
    base = rich_answer(a.q, a.date, a.station)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return {**base, "via": "fallback"}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        sys = ("Rephrase this factual answer for a Bengaluru Traffic Police officer in ONE short sentence. "
               "Do NOT add, remove, or change any facts, numbers, or place names. Return only the sentence.")
        r = client.chat.completions.create(model="gpt-4o-mini", temperature=0.2,
              messages=[{"role": "system", "content": sys}, {"role": "user", "content": base["answer"]}])
        base["answer"] = r.choices[0].message.content.strip(); base["via"] = "openai"
        return base
    except Exception as e:
        return {**base, "via": "fallback", "error": str(e)[:120]}

@app.get("/")
def home():
    return RedirectResponse("/app/index.html")

app.mount("/app", StaticFiles(directory=APP, html=True), name="app")
