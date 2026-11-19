"""gta6-trophy-rarity-forecast-01 entry (Python 3.13, standard library only).

Forecasts every GTA VI trophy's Exophase earn rate at the case's target as
five quantiles, scored by pinball loss on the logit scale.

Model
-----
Per trophy, fit a least-squares line of logit(rate) against ln(1 + hours
since release) over the case's history and read it at the target, and
fit a saturating adoption curve r(t) = S*(1-exp(-k*t)) by grid search
over k.  The two are blended by in-sample fit quality: the saturating
curve alone when it clearly wins (trophy completion flattens -- a
log-time line overshoots badly when the early rise is steep), the
log-time line alone when rates are falling (dilution) and it clearly
wins, else their average in logit space.  Whatever the blend says, the move
from the last read is damped by half and capped at 1.0 logit: pushing
4.5 days past 2 days of data is genuinely uncertain, and both
saturation and the dilution headwind pull the truth back toward the
last rate.  The log-time curve also
captures the dilution that dominates early on -- every Exophase refresh
adds newly tracked players who hold fewer trophies, so rates fall at a
refresh and the fitted line nets growth against dilution.

The per-trophy slope is shrunk toward its tier's median slope, with the
weight growing in the number of distinct rates the trophy actually showed
(a trophy that never moved says little about its own slope).  The
extrapolated median move is capped at 2.2 logit units, and the median is
clamped to [0.05, 99.99].

Flat histories (the practice cases: every read identical, players_tracked
stuck at 638) carry no trend at all, yet the answer is the page after its
next refresh, when the tracked-player count jumps and almost every rate
falls.  For those, the median applies a refresh-dilution prior instead:
the next refresh multiplies players_tracked by up to ~6x (launch night's
first refresh went 638 -> 3,868), and a new player holds each trophy with
probability g (relative to the old rate) that falls with the trophy's
lateness -- early/easy trophies near 0.3, late chapter, grind and
completion trophies near 0.02.  The median is then
r * (1 + (m - 1) * g) / m.  This branch only fires when the history shows
at most two distinct snapshots, so it can never override real movement.

Spread: the per-trophy residual variance of the logit fit, plus a
random-walk term growing with the horizon (0.30 logit per sqrt day) and a
0.20 logit floor term; flat histories get a 0.70 logit floor for the
unseen refresh shock.  Quantiles are inv_logit(median_logit + z_tau * s)
with z = (-1.645, -0.674, 0, 0.674, 1.645), rounded to 4 decimals.

Deterministic: no randomness, no wall-clock reads, no network.  Same case
always writes the same bytes, in well under a second.

Run: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
Grader: python3 -P -s /bot/main.py, reading $MAKEFUN_INPUT_DIR else
/input, writing $MAKEFUN_OUTPUT_DIR else /output.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

CHAPTER_RE = re.compile(r"Complete Chapter (\d+)", re.IGNORECASE)
GRIND_RE = re.compile(
    r"250 Stunts|100% Completion|50 Challenges|100 locations|"
    r"Collectible Card sets|every spot|all of Leonida|every location|"
    r"all Shooting Range|20 different races|100 vehicles|"
    r"8 mission strands|Meditate|gold medal in all",
    re.IGNORECASE,
)
MISSION_RE = re.compile(r"Complete (the final mission|Breakfast Run|The Cartel Heist|"
                        r"Child's Play|Headliners|VCIA Takedown|Operation Savage Glory|"
                        r"Gone Fishing)", re.IGNORECASE)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(l):
    if l > 20:
        return 99.99
    if l < -20:
        return 0.01
    return 100.0 / (1.0 + math.exp(-l))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def hours_since_release(ts, release):
    return max(0.0, (ts - release).total_seconds() / 3600.0)


def lateness(trophy):
    """0 = earliest/easiest trophy, 1 = latest/grindiest, from its text."""
    name = trophy.get("name", "") or ""
    desc = trophy.get("description", "") or ""
    text = name + " " + desc
    tier = {"bronze": 0.20, "silver": 0.55, "gold": 0.80, "platinum": 1.00}.get(
        trophy.get("tier", "bronze"), 0.20)
    score = tier
    m = CHAPTER_RE.search(desc)
    if m:
        score = max(score, 0.10 + 0.15 * int(m.group(1)))
    if "final mission" in desc.lower():
        score = max(score, 0.70)
    if GRIND_RE.search(text):
        score = max(score, 0.85)
    if MISSION_RE.search(desc):
        score = max(score, 0.45)
    if trophy.get("secret"):
        score = min(1.0, score + 0.10)
    return min(1.0, max(0.0, score))


def new_player_hold_rate(trophy, rate):
    """g: probability a newly tracked player holds this trophy, relative
    to the rate among the players tracked so far.  1.0 for the trophy
    every tracked player holds; near 0.02 for the latest/grindiest.

    Scaled by 0.6: the launch-night refresh's published aggregate
    (players 638 -> 3,868, trophies earned 943 -> 4,276) constrains the
    scale to 0.36-1.0 depending on how universal the 100% trophy is
    among the new players; 0.6 is the central value."""
    if rate >= 99.99:
        return 1.0
    late = lateness(trophy)
    return 0.6 * (0.02 + 0.33 * (1.0 - late))


def ols(xs, ys):
    """Least-squares line; returns (intercept, slope)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def median_of(values):
    v = sorted(values)
    n = len(v)
    if n == 0:
        return 0.0
    if n % 2:
        return v[n // 2]
    return 0.5 * (v[n // 2 - 1] + v[n // 2])


def saturating_fit(ts_days, rates):
    """Fit r(t) = r0 + (S - r0)*(1 - exp(-k*t)) per trophy by grid search
    over k, S by least squares in rate space, scored in logit space.
    r0 is the first observed rate: trophy rates are already ~1% at the
    first read, so the curve must not be forced through zero.  Trophy
    completion is a saturating process -- the share of players who have
    done a thing by time t flattens -- so an unbounded log-time line
    overshoots it badly when the early rise is steep.  Returns
    (S, k, rmse_logit)."""
    r0 = rates[0]
    best = None
    grid = [0.0] + [round(0.05 * (1.25 ** i), 4) for i in range(32)]
    for k in grid:
        fs = [1.0 - math.exp(-k * t) for t in ts_days]
        denom = sum(f * f for f in fs)
        if denom <= 0:
            s_val = max(rates)
        else:
            s_val = r0 + sum((r - r0) * f for r, f in zip(rates, fs)) / denom
        s_val = min(99.99, max(max(rates), s_val))
        se = 0.0
        for t, r in zip(ts_days, rates):
            pred = r0 + (s_val - r0) * (1.0 - math.exp(-k * t))
            se += (logit(pred) - logit(r)) ** 2
        rmse = math.sqrt(se / len(rates))
        if best is None or rmse < best[2]:
            best = (s_val, k, rmse)
    return best


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    release = parse_time(case.get("game", {}).get("release_date", "2026-11-19T00:00:00Z")) \
        if case.get("game", {}).get("release_date") else RELEASE
    target = parse_time(case["target"])
    t_target = hours_since_release(target, release)
    x_target = math.log1p(t_target)
    horizon_days = max(0.0, case.get("horizon_minutes", 0) / 1440.0)

    # Per-read x values and the set of distinct snapshots (for flatness).
    reads = []
    for h in history:
        reads.append((math.log1p(hours_since_release(parse_time(h["at"]), release)), h))
    distinct = set()
    for _, h in reads:
        distinct.add((h.get("players_tracked"), tuple(h.get("rates") or [])))
    flat = len(distinct) <= 2

    # Last non-null rate per trophy.
    last_rate = []
    for i in range(len(trophies)):
        r = None
        for _, h in reversed(reads):
            rates = h.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                r = rates[i]
                break
        last_rate.append(r if r is not None else 1.0)

    # Per-trophy logit trend fits.
    fits = []  # (intercept, slope, resid_var, n_distinct)
    for i, trophy in enumerate(trophies):
        xs, ys = [], []
        for x, h in reads:
            rates = h.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                xs.append(x)
                ys.append(logit(rates[i]))
        n_distinct = len(set(ys))
        if len(xs) >= 2 and n_distinct >= 2:
            a, b = ols(xs, ys)
            resid = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / len(xs)
        elif xs:
            a, b = ys[-1], 0.0
            resid = 0.0
        else:
            a, b = logit(last_rate[i]), 0.0
            resid = 0.0
        fits.append((a, b, resid, n_distinct))

    # Tier median slopes, from trophies that actually moved.
    tier_slopes = {}
    for tier in ("bronze", "silver", "gold", "platinum"):
        moved = [fits[i][1] for i, t in enumerate(trophies)
                 if t.get("tier") == tier and fits[i][3] >= 3]
        if moved:
            tier_slopes[tier] = median_of(moved)
    global_slope = median_of([f[1] for f in fits if f[3] >= 3]) if any(
        f[3] >= 3 for f in fits) else 0.0

    # Flat-history refresh prior: expected players_tracked multiplier by
    # the target.  The first refresh after launch night multiplied the
    # tracked count ~6x; the multiplier grows with the unseen time since
    # the first read and is capped at 6.
    m_refresh = 1.0
    if flat and reads:
        t_first = hours_since_release(parse_time(reads[0][1]["at"]), release)
        m_refresh = min(6.0, (t_target / max(t_first, 0.5)) ** 1.5)

    rows = []
    for i, trophy in enumerate(trophies):
        a, b, resid, n_distinct = fits[i]
        tier_prior = tier_slopes.get(trophy.get("tier"), global_slope)
        w = n_distinct / (n_distinct + 4.0) if n_distinct else 0.0
        slope = w * b + (1.0 - w) * tier_prior
        y_last = logit(last_rate[i])
        if flat:
            g = new_player_hold_rate(trophy, last_rate[i])
            r_flat = last_rate[i] * (1.0 + (m_refresh - 1.0) * g) / m_refresh
            y_med = logit(r_flat)
            spread = 0.70
        else:
            y_a = a + slope * x_target
            y_a = min(y_last + 2.2, max(y_last - 2.2, y_a))
            # Saturating adoption curve r(t) = S*(1-exp(-k*t)): trophy
            # completion flattens, so an unbounded log-time line overshoots
            # when the early rise is steep.  Blend by in-sample fit: the
            # saturating fit alone when it clearly wins, the log-time line
            # alone when rates are falling (dilution) and it clearly wins,
            # else the average of the two in logit space.
            ts_days, rs = [], []
            for _, h in reads:
                rates = h.get("rates") or []
                if i < len(rates) and rates[i] is not None:
                    ts_days.append(max(0.0, (parse_time(h["at"]) - release).total_seconds() / 86400.0))
                    rs.append(rates[i])
            rmse_a = math.sqrt(resid)
            if len(rs) >= 3:
                s_sat, k_sat, rmse_b = saturating_fit(ts_days, rs)
                y_b = logit(rs[0] + (s_sat - rs[0]) * (1.0 - math.exp(-k_sat * t_target / 24.0)))
                y_b = min(y_last + 2.2, max(y_last - 2.2, y_b))
                if rmse_b > 1.3 * max(rmse_a, 0.05):
                    y_med, resid_use = y_a, resid
                elif rmse_a > 1.3 * max(rmse_b, 0.05):
                    y_med, resid_use = y_b, rmse_b ** 2
                else:
                    y_med = 0.5 * (y_a + y_b)
                    resid_use = 0.5 * (resid + rmse_b ** 2)
            else:
                y_med, resid_use = y_a, resid
            # Damp the extrapolated move from the last read: pushing 4.5
            # days past 2 days of data is genuinely uncertain, and both
            # saturation and the dilution headwind pull the truth back
            # toward the last rate.  Cap the damped move at 1.0 logit.
            y_med = y_last + 0.5 * (y_med - y_last)
            y_med = min(y_last + 1.0, max(y_last - 1.0, y_med))
            spread = math.sqrt(resid_use + 0.09 * horizon_days + 0.04)
            spread = min(1.8, max(0.30, spread))
        qs = []
        for z in Z:
            q = round4(clamp_pct(inv_logit(y_med + z * spread)))
            qs.append(q)
        # Enforce monotone non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def persistence_fallback(case):
    """Last-rate forecast with the house baseline's spread; only used if
    the model fails on a case, so the output is still valid."""
    trophies = case["trophies"]
    history = case.get("history") or []
    last = history[-1]["rates"] if history else [1.0] * len(trophies)
    spread = 0.5 * math.sqrt(max(0.0, case.get("horizon_minutes", 0) / 1440.0))
    rows = []
    for i, trophy in enumerate(trophies):
        r = last[i] if i < len(last) and last[i] is not None else 1.0
        lg = logit(r)
        qs = [round4(clamp_pct(inv_logit(lg + z * spread))) for z in Z]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = persistence_fallback(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
