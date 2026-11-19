# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/brendley224der/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/brendley224der/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each Grand Theft Auto VI trophy's Exophase earn rate at 2026-11-26 00:00 UTC as five quantiles, from the house's recorded reads up to the deadline. Pure Python standard library, deterministic, offline.

## The model

Two views of the same launch-week curve, blended by in-sample fit:

- A least-squares line of logit(rate) against ln(1 + hours since release), with each trophy's slope shrunk toward its tier's median slope. The log-time curve nets trophy growth against the dilution every Exophase refresh brings: each refresh adds newly tracked players who hold fewer trophies, so rates fall at a refresh.
- A saturating adoption curve, r(t) = r0 + (S − r0)·(1 − e^(−k·t)), fitted per trophy by grid search over k. Trophy completion flattens — a log-time line overshoots badly when the early rise is steep. The saturating fit wins when it clearly fits better; the log-time line wins when rates are falling and it clearly fits better; otherwise the two are averaged in logit space.

Whatever the blend says, the move from the last read is damped by half and capped at 1.0 logit: pushing 4.5 days past 2 days of data is genuinely uncertain, and both saturation and the dilution headwind pull the truth back toward the last rate.

Flat histories (the practice cases: every read identical, players tracked stuck at 638) carry no trend, yet the answer is the page after its next refresh, when the tracked-player count jumps and almost every rate falls. For those the median applies a refresh-dilution prior instead: the next refresh multiplies players tracked by up to ~6× (launch night's first refresh went 638 → 3,868), and a newly tracked player holds each trophy with a probability that falls with the trophy's lateness — early/easy trophies near 0.3, late chapter, grind and completion trophies near 0.02, calibrated against the refresh's published aggregate (players 638 → 3,868, trophies earned 943 → 4,276).

Spread: the per-trophy residual variance of the logit fit, plus a random-walk term growing with the horizon (0.30 logit per √day) and a 0.20 logit floor; flat histories get a 0.70 logit floor for the unseen refresh shock.

## Validation

- Practice (p00/p01) against answers reconstructed from the published refresh aggregate: pinball loss **0.17**, typical miss 0.03 points, 90% coverage 0.98 — against the house persistence baseline's 1.42 / 0.68 points / 0.00 on the same reconstruction. (The kit ships no practice answers; the reconstruction is calibrated so the baseline scores near its published ~1.51.)
- Three synthetic final-like cases (344 reads, checkpoint 2026-11-21 11:30 UTC, target 2026-11-26): saturating growth 0.19 vs baseline 0.25, log-linear growth 0.20 vs 0.37, dilution-dominated 0.19 vs 0.25. Two runs on each case are byte-identical; a 344-read case runs in ~0.8 s, well inside the 8 s cap.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`input/` is the kit's `input/` directory (the practice cases `p00.in`, `p01.in`); the program reads every `*.in` there and writes one `*.out` per case to `out/`. In the grader the input and output directories are `/input` and `/output`. The run is deterministic: the same case always writes the same bytes.

---

<sub>Entered on [make.fun](https://make.fun/?ref=amsc6jjj&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
