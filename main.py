"""GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Reads a case file -- every read the house made of Exophase's GTA VI PS5 trophy
page before the case's checkpoint -- and forecasts each trophy's earn rate at
the case's target as five quantiles (5%, 25%, 50%, 75%, 95%).

The model is a counts-aware trend ensemble, fit per trophy on the case's own
history, in three parts:

1. TREND. A recency-weighted least-squares line of the trophy's logit earn
   rate against ln(1 + hours since launch day), fit on the last 24 hours of
   reads. Launch-week rates move fast and then settle, so log-time is the
   right axis -- and only recent reads set the slope, because a launch-week
   curve bends as it saturates and an old slope over-extrapolates.

2. COUNTS. Earn rates are a ratio -- earners over players tracked -- and the
   two move for different reasons: players join the game (denominator growth)
   while the players already in it progress (numerator growth). So the model
   also fits log(earners) and log(players tracked) separately against log-time
   on the same recent window and divides the two extrapolations. When the
   player base is still swelling, this keeps a trophy's rate from being
   dragged up by a trend that belongs to the denominator.

Every extrapolation is damped: a fitted line is read part-way to the target
rather than at it, because a straight line read at the full 4.5-day horizon
overshoots a saturating curve. The two forecasts are then averaged in logit
space and shrunk toward persistence (the last read's rate). The shrinkage is
strong when the trend is weak -- few reads, or a slope the noise swallows --
and light when the trend is clear. The total move from the last rate is
capped, so no trophy's forecast can run away on a 4.5-day horizon.

The spread is honest about two kinds of uncertainty, combined in quadrature:
a random-walk term that widens with the square root of the horizon (the house
baseline's 0.5 logit units per root day) and the trend fit's own prediction
interval, which widens with the extrapolation distance. When the whole history
is flat -- every read identical, the page not yet refreshed -- the spread
widens to a full logit unit: a stale page says nothing about the next refresh,
and the practice cases show a refresh can move every rate at once.

Everything is deterministic: pure functions of the case file, no network, no
clock, no randomness. Python 3.13, standard library only.

Run it outside the grader's container with:

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path; nothing beside this file is
# imported, but keep the directory first anyway so a future helper module
# shipped beside main.py would resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Launch day, the origin of the log-time axis (2026-11-19T00:00:00Z).
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# The trend fits use only the most recent reads: launch-week curves bend as
# they saturate, so an old slope over-extrapolates. (The house curve baseline
# fits the last 24 hours for the same reason. A 12-hour window wins on
# noiseless synthetic weeks but overfits read noise; 24 hours wins on noisy
# ones, so 24 it is.)
WINDOW_HOURS = 24.0
# Damping on every extrapolation: a fitted line is read not at the target but
# part-way there, because a launch-week curve flattens as it saturates and a
# straight line read at the full horizon overshoots it. (Tuned to 0.8 on
# synthetic launch-week backtests: 0.6-0.8 all beat the house baselines, 1.0
# overshoots and 0.4 under-uses the trend.)
DAMP = 0.8
# Random-walk spread, in logit units per square-root day (the house baseline's).
SIGMA_RW = 0.5
# Recency half-life for the trend fits, in hours.
HALF_LIFE_HOURS = 12.0
# The narrowest and widest spread allowed, in logit units.
SIGMA_MIN = 0.30
SIGMA_MAX = 2.50
# Spread used when the history is perfectly flat (a stale, unrefreshed page).
SIGMA_STALE = 1.0
# The furthest the median may move from the last read's rate, in logit units.
MAX_MOVE = 3.5
# Shrinkage toward persistence: from 0.6 (no trend signal) down to 0.2 (clear trend).
LAMBDA_MAX = 0.6
LAMBDA_MIN = 0.2


def clamp_pct(x: float) -> float:
    """A percent clamped to the range the logit can take (as the scorer clamps)."""
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def sigmoid(logit_value: float) -> float:
    """Inverse logit, as a percent."""
    if logit_value >= 0:
        return 100.0 / (1.0 + math.exp(-logit_value))
    e = math.exp(logit_value)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """x rounded to 4 decimals, half up, the way JavaScript rounds it.

    Whole numbers come back as ints, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    whole = math.floor(x * 10000 + 0.5)
    value = whole / 10000
    return int(value) if value.is_integer() else value


def normal_quantile(tau: float) -> float:
    """The standard normal's quantile at tau (Acklam's rational approximation).

    Only used when a case asks for levels other than the contest's five; the
    five standard levels use the exact Z table above.
    """
    # Coefficients for the central region and the tails.
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285984469534e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
         -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
         3.754408661907416e0]
    p_low = 0.02425
    if tau < p_low:
        q = math.sqrt(-2.0 * math.log(tau))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if tau <= 1.0 - p_low:
        q = tau - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - tau))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def parse_instant(text):
    """An ISO-8601 timestamp to a datetime, or None when it does not parse."""
    if not isinstance(text, str):
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def hours_since_launch(moment: datetime) -> float:
    return (moment - LAUNCH).total_seconds() / 3600.0


def log_time(moment: datetime) -> float:
    """The trend axis: ln(1 + hours since launch day)."""
    return math.log1p(max(0.0, hours_since_launch(moment)))


class TrendFit:
    """A recency-weighted least-squares line, with its prediction uncertainty."""

    __slots__ = ("n", "slope", "intercept", "resid_std", "x_mean", "sxx",
                 "n_eff", "x_lo", "x_hi")

    def __init__(self, xs, ys, weights):
        n = len(xs)
        self.n = n
        self.x_lo = min(xs)
        self.x_hi = max(xs)
        total_w = sum(weights)
        x_mean = sum(w * x for w, x in zip(weights, xs)) / total_w
        y_mean = sum(w * y for w, y in zip(weights, ys)) / total_w
        sxx = sum(w * (x - x_mean) ** 2 for w, x in zip(weights, xs))
        sxy = sum(w * (x - x_mean) * (y - y_mean)
                  for w, x, y in zip(weights, xs, ys))
        self.x_mean = x_mean
        self.sxx = sxx
        self.slope = sxy / sxx if sxx > 1e-12 else 0.0
        self.intercept = y_mean - self.slope * x_mean
        # Effective sample size of the weights, for the prediction interval.
        sum_w2 = sum(w * w for w in weights)
        self.n_eff = (total_w * total_w / sum_w2) if sum_w2 > 0 else float(n)
        # Weighted residual standard deviation, with a degrees-of-freedom
        # correction when there are enough points to estimate it.
        ss_res = sum(w * (y - (self.intercept + self.slope * x)) ** 2
                     for w, x, y in zip(weights, xs, ys))
        mean_sq = ss_res / total_w if total_w > 0 else 0.0
        if n > 2:
            mean_sq *= n / (n - 2)
        self.resid_std = math.sqrt(max(0.0, mean_sq))

    def predict(self, x: float) -> float:
        return self.intercept + self.slope * x

    def prediction_std(self, x: float) -> float:
        """The fit's own prediction standard deviation at x (residual scale)."""
        if self.n < 3 or self.sxx <= 1e-12:
            return self.resid_std
        leverage = 1.0 + 1.0 / max(1.0, self.n_eff) + \
            (x - self.x_mean) ** 2 / self.sxx
        return self.resid_std * math.sqrt(max(1.0, leverage))

    def quality(self) -> float:
        """How much the trend deserves to be trusted, 0 (none) to 1 (clear).

        A trend earns trust from many reads and from a slope that moves the
        line well clear of its own noise over the observed window.
        """
        if self.n < 3:
            return 0.0
        n_factor = min(1.0, self.n / 10.0)
        move = abs(self.slope) * (self.x_hi - self.x_lo)
        noise = max(self.resid_std, 0.05)
        signal = move / (move + 2.0 * noise)
        return n_factor * signal


def recency_weights(times, last_time):
    """Weights halving every HALF_LIFE_HOURS back from the last read."""
    return [0.5 ** ((last_time - t).total_seconds() / 3600.0 / HALF_LIFE_HOURS)
            for t in times]


def forecast_trophy(series, players_fit, x_target, days, stale_page):
    """Forecast one trophy's five quantiles.

    series: list of (moment, x, rate, players) for the reads that list this
        trophy, oldest first, x already on the log-time axis.
    players_fit: the TrendFit of log(players tracked) over the same reads.
    x_target: the target on the log-time axis. days: the horizon in days.
    stale_page: True when every read in the case is identical.
    """
    last_rate = series[-1][2]
    persist = logit(last_rate)
    rw = SIGMA_RW * math.sqrt(max(0.0, days))

    if len(series) < 3:
        # Too little history to fit anything: hold the last rate, wide band.
        sigma = min(SIGMA_MAX, max(SIGMA_MIN, math.hypot(rw, 0.75)))
        if stale_page:
            sigma = max(sigma, SIGMA_STALE)
        return persist, sigma

    # Fit on the recent window only: a launch-week curve bends as it
    # saturates, and the slope of the whole history over-extrapolates.
    # With too few recent reads, fall back to the whole series.
    last_moment = series[-1][0]
    cutoff = last_moment - timedelta(hours=WINDOW_HOURS)
    window = [p for p in series if p[0] >= cutoff]
    if len(window) < 3:
        window = list(series)
    moments = [p[0] for p in window]
    xs = [p[1] for p in window]
    ys = [logit(p[2]) for p in window]
    weights = recency_weights(moments, moments[-1])

    # The damped read point: part-way from the last read toward the target,
    # because a straight line read at the full horizon overshoots a
    # saturating curve.
    x_eff = xs[-1] + DAMP * (x_target - xs[-1])

    trend = TrendFit(xs, ys, weights)
    trend_logit = trend.predict(x_eff)
    quality = trend.quality()

    # The counts model: fit log(earners) on the same axis, divide by the
    # players fit's extrapolation. Earners are counts, so they are floored
    # at half a player (a 0.00% rate means fewer than one earner shows).
    counts_logit = None
    earners = []
    exs = []
    ews = []
    for (_moment, x, rate, players), w in zip(window, weights):
        if players is None or players <= 0:
            continue
        earned = max(0.5, rate / 100.0 * players)
        earners.append(math.log(earned))
        exs.append(x)
        ews.append(w)
    if len(exs) >= 3 and players_fit is not None:
        efit = TrendFit(exs, earners, ews)
        players_target = math.exp(players_fit.predict(x_eff))
        players_last = series[-1][3] or 0
        # Players do not shrink over a launch week, and the extrapolation
        # is not allowed to run wild either.
        players_target = min(max(players_target, players_last),
                             max(players_last, 1) * 20.0)
        earned_target = math.exp(efit.predict(x_eff))
        earned_last = max(0.5, last_rate / 100.0 * max(players_last, 1))
        earned_target = min(max(earned_target, earned_last),
                            0.98 * players_target)
        if players_target > 0:
            counts_logit = logit(100.0 * earned_target / players_target)

    # Ensemble the two forecasts, then shrink toward persistence.
    if counts_logit is not None:
        ensemble = 0.5 * trend_logit + 0.5 * counts_logit
    else:
        ensemble = trend_logit
    lam = LAMBDA_MAX - (LAMBDA_MAX - LAMBDA_MIN) * quality
    median = (1.0 - lam) * ensemble + lam * persist
    # Cap the move from the last rate: no trophy runs away on one forecast.
    move = median - persist
    if abs(move) > MAX_MOVE:
        median = persist + math.copysign(MAX_MOVE, move)

    # Spread: random walk and the trend fit's prediction interval, combined
    # in quadrature. (The interval widens with the extrapolation distance,
    # which is the honest uncertainty for a 4.5-day horizon.)
    sigma = math.sqrt(rw * rw + trend.prediction_std(x_target) ** 2)
    sigma = min(SIGMA_MAX, max(SIGMA_MIN, sigma))
    if stale_page:
        sigma = max(sigma, SIGMA_STALE)
    return median, sigma


def read_moments(history):
    """Each read's moment, oldest first, with a fallback for unparsed times.

    Prefers the read's `at` (when the house actually read the page), then its
    `slot` (when the read was scheduled). A read with neither gets the
    previous read's moment plus ten minutes, the in-cluster cadence.
    """
    moments = []
    last_good = None
    for index, read in enumerate(history):
        moment = parse_instant(read.get("at")) or parse_instant(read.get("slot"))
        if moment is None:
            if last_good is not None:
                moment = last_good + timedelta(minutes=10)
            else:
                moment = LAUNCH + timedelta(hours=index)
        moments.append(moment)
        last_good = moment
    return moments


def fallback_forecast(case):
    """A valid forecast with no modelling: last rates, wide bands.

    Used only when the full forecast fails on a case, so a case is never left
    without an output file. A trophy the last read does not list takes the
    cross-trophy median of the last read's rates.
    """
    history = case.get("history") or []
    last_rates = []
    if history:
        last_rates = [r for r in history[-1].get("rates") or [] if r is not None]
    median_rate = sorted(last_rates)[len(last_rates) // 2] if last_rates else 50.0
    last = history[-1].get("rates") if history else []
    rows = []
    for i, trophy in enumerate(case.get("trophies") or []):
        rate = last[i] if i < len(last) and last[i] is not None else median_rate
        med = logit(rate)
        rows.append({"id": trophy["id"],
                     "q": [round4(clamp_pct(sigmoid(med + z * 1.5))) for z in Z]})
    return {"trophies": rows}


def forecast_case(case, name):
    """Forecast every trophy of one case at its target."""
    levels = case.get("levels") or LEVELS
    if list(levels) == list(LEVELS):
        zs = Z
    else:
        zs = [normal_quantile(t) for t in levels]
    history = case.get("history") or []
    trophies = case.get("trophies") or []
    if not history or not trophies:
        return fallback_forecast(case)

    moments = read_moments(history)
    xs = [log_time(m) for m in moments]
    target_moment = parse_instant(case.get("target"))
    horizon = case.get("horizon_minutes")
    if target_moment is None:
        minutes = horizon if isinstance(horizon, (int, float)) else 0
        target_moment = moments[-1] + timedelta(minutes=minutes)
    if isinstance(horizon, (int, float)) and horizon > 0:
        days = horizon / 1440.0
    else:
        days = max(0.0, (target_moment - moments[-1]).total_seconds() / 86400.0)
    x_target = log_time(target_moment)

    # A stale page: every read identical, rates and players alike.
    first_rates = history[0].get("rates")
    first_players = history[0].get("players_tracked")
    stale_page = all(read.get("rates") == first_rates and
                     read.get("players_tracked") == first_players
                     for read in history)

    # The players fit: log(players tracked) against log-time, recency-weighted,
    # on the same recent window as the trophy fits (with the whole history as
    # a fallback when the window is too thin).
    players_fit = None
    p_xs, p_ys, p_ws = [], [], []
    weights_all = recency_weights(moments, moments[-1])
    cutoff = moments[-1] - timedelta(hours=WINDOW_HOURS)
    p_all = []
    for moment, x, read, w in zip(moments, xs, history, weights_all):
        players = read.get("players_tracked")
        if isinstance(players, (int, float)) and players > 0:
            p_all.append((moment, x, math.log(players), w))
    p_win = [p for p in p_all if p[0] >= cutoff]
    if len(p_win) < 3:
        p_win = p_all
    for _moment, x, y, w in p_win:
        p_xs.append(x)
        p_ys.append(y)
        p_ws.append(w)
    if len(p_xs) >= 3:
        players_fit = TrendFit(p_xs, p_ys, p_ws)

    # The cross-trophy median last rate, for a trophy no read lists.
    last_rates = [r for r in (history[-1].get("rates") or []) if r is not None]
    median_rate = sorted(last_rates)[len(last_rates) // 2] if last_rates else 50.0

    rows = []
    for i, trophy in enumerate(trophies):
        series = []
        for moment, x, read in zip(moments, xs, history):
            rates = read.get("rates") or []
            rate = rates[i] if i < len(rates) else None
            if rate is None:
                continue
            players = read.get("players_tracked")
            series.append((moment, x, float(rate),
                           players if isinstance(players, (int, float)) else None))
        if not series:
            med = logit(median_rate)
            sigma = 1.5
        else:
            med, sigma = forecast_trophy(series, players_fit, x_target,
                                         days, stale_page)
        rows.append({"id": trophy["id"],
                     "q": [round4(clamp_pct(sigmoid(med + z * sigma)))
                           for z in zs]})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            # Never leave a case without an output: fall back to last rates.
            forecast = fallback_forecast(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
