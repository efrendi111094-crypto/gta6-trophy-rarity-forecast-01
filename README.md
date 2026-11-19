# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/efrendi111094-crypto/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/efrendi111094-crypto/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

**55 trophies, 5 quantiles each: a counts-aware launch-week forecast of every GTA VI earn rate.**

An entry for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty. The program reads the house's recording of Exophase's GTA VI PS5 trophy page — every read of every trophy's earn rate up to the case's checkpoint — and forecasts each trophy's rate at the target (2026-11-26 00:00 UTC for the final) as five quantiles, for the lowest pinball loss on the logit scale.

## Methodology

The forecast is a counts-aware trend ensemble, fit per trophy on the case's own history:

1. **Trend.** A recency-weighted least-squares line of the trophy's logit earn rate against ln(1 + hours since launch day), fit on the last 24 hours of reads. Launch-week rates move fast and then settle, so log-time is the right axis — and only recent reads set the slope, because a launch-week curve bends as it saturates and an old slope over-extrapolates.

2. **Counts.** An earn rate is a ratio — earners over players tracked — and the two move for different reasons: players join the game (denominator growth) while the players already in it progress (numerator growth). So the model also fits log(earners) and log(players tracked) separately against log-time on the same recent window and divides the two extrapolations. When the player base is still swelling, this keeps a trophy's rate from being dragged up by a trend that belongs to the denominator.

3. **Damping + shrinkage.** Every extrapolation is damped — a fitted line is read part-way to the target (damping 0.8) rather than at it, because a straight line read at the full 4.5-day horizon overshoots a saturating curve. The two forecasts are averaged in logit space, then shrunk toward persistence (the last read's rate); the shrinkage is strong when the trend is weak and light when it is clear. The total move from the last rate is capped at 3.5 logit units.

4. **Spread.** The random-walk term (0.5 logit units per square-root day, the house baseline's) combined in quadrature with the trend fit's own prediction interval, which widens with the extrapolation distance. When the whole history is flat — every read identical, the page not yet refreshed — the spread widens to a full logit unit: a stale page says nothing about the next refresh.

## Validation

The model was backtested on synthetic launch weeks (players joining on a saturating curve, hourly cohorts earning trophies at difficulty-dependent paces, rates scored against the true week-one rates with the contest's pinball loss). It beat both house baselines — persistence and the logit-linear curve — in all 8 clean worlds (mean loss 0.249 vs 0.274 for the curve baseline and 0.349 for persistence) and in all 8 noisy worlds (read noise σ=0.08 logit). The damping constant and the 24-hour window were tuned on those worlds: damping 1.0 overshoots, 0.4 under-uses the trend; a 12-hour window wins on noiseless weeks but overfits read noise.

On the kit's practice cases the program is valid and byte-identical across runs (both histories are flat launch-night pages, so the forecast holds the last rates with the wide stale-page spread).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only — no install step, no network. The program reads every `*.in` case in the input directory and writes one `<id>.out` per case, fully deterministic: two runs on the same case write the same bytes.

---

<sub>Entered on [make.fun](https://make.fun/?ref=tzrh51cg&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
