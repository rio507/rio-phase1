"""weather.py — what the sky is actually doing, from Google's Weather API.

RIO can see the sky. She cannot see three o'clock. The camera is a genuinely
good instrument for "those clouds ahead are getting dark" and a worthless one
for "it'll rain in about twenty minutes", and the failure mode when the two get
confused is specific and bad: a vision model asked what the weather will do
will ANSWER, fluently, from the look of the clouds, and a driver who is told
rain is coming changes their drive. A forecast inferred from a photograph is a
guess wearing a number.

So the split here is the same one places.py draws for businesses and the
navigation stack draws for roads:

    the camera   what is physically visible from the vehicle right now —
                 wet road, dark cloud, spray off the truck ahead, low sun
    this file    temperature, probability, when it starts, wind, visibility,
                 what happens next

Neither is allowed into the other's territory. The vision model may not forecast
and this file may not claim to see anything.

WHAT THE API ACTUALLY IS, having been asked rather than assumed
---------------------------------------------------------------
Verified against the live endpoint on the project's existing Maps key (see
LICENSING.md §3 for the full reading, and tools/weather_probe.py to re-run it):

  currentConditions:lookup   temperature, feels-like, condition text, humidity,
                             dew point, precipitation probability AND amount,
                             wind speed/gust/direction, visibility, pressure,
                             UV, cloud cover, thunderstorm probability
  forecast/hours:lookup      the same shape per hour, up to 240 hours, 24 to a
                             page
  forecast/days:lookup       up to 10 days, with sunrise/sunset and the day's
                             high and low

  THERE IS NO ALERTS ENDPOINT. Severe-weather alerts are not part of this API.
  `alerts` in the context below is therefore ALWAYS an empty list, and it is
  present only so that the shape does not change if a source is ever added. It
  is not a thing that is sometimes populated and happens to be empty now, and
  nothing downstream may treat an empty `alerts` as "no severe weather" — it
  means "not known", which is a different claim and the honest one.

  Units are a request parameter, not a response property. METRIC is the API
  default; this asks for what config.WEATHER_UNITS says and carries the unit
  NAMES out with every number, so nothing downstream has to infer whether 26
  was Celsius.

ONE REFRESH IS TWO OR THREE BILLED REQUESTS
-------------------------------------------
Every endpoint bills against one SKU ("Weather Usage"), so a refresh costs
exactly as many requests as it makes: current + hourly, plus daily when
config.WEATHER_INCLUDE_DAILY is on. They are the calls whose absence a driver
would notice — now, next few hours, and sunset — and nothing fans out beyond
them. Phase 2's route sampling would multiply this by the number of points
sampled along the polyline, which is precisely why it is phase 2.

THE HONESTY RULE, which is the whole of amendment C
---------------------------------------------------
Every context carries `fetched_at`, `fetched_for` and `age_s`, and `usable()`
refuses one that is too old or too far away. A confident forecast for somewhere
the car left fifteen minutes ago is worse than no forecast, because it is
indistinguishable from a good one at the moment it is spoken and only wrong
later, when the driver is in the rain they were told to expect elsewhere.

So staleness is not a warning attached to the data. It REMOVES the data:
`get_weather_context` returns `ok: False` with a reason, and the caller's
instruction in that case is to say she cannot pull the forecast — never to
reach for the sky instead.
"""
import math
import os
import threading
import time
from datetime import datetime, timezone

import httpx

import config

class NoCoverage(Exception):
    """Google has no weather for this place, and says so with a 404.

    Not a failure, and importantly not a transient one: Japan, South Korea and
    mainland China all answer "Information is not supported for this location"
    for every endpoint, while Australia, Germany, Brazil, Nigeria and India
    answer normally. It is a coverage map, not an outage, and retrying will not
    fix it.

    It gets its own type because the two cases need different logging and the
    same behaviour. The behaviour is identical on purpose -- RIO says she
    cannot pull the forecast, and forecasts nothing from the sky -- but a drive
    through a region with no coverage should not fill the log with what look
    like network errors, because somebody will eventually go looking for a
    network problem that does not exist.
    """


CURRENT_URL = "https://weather.googleapis.com/v1/currentConditions:lookup"
HOURS_URL = "https://weather.googleapis.com/v1/forecast/hours:lookup"
DAYS_URL = "https://weather.googleapis.com/v1/forecast/days:lookup"

# The `fields` mask is a payload reduction, NOT a billing one: unlike Places
# (New), where the mask selects the SKU, every Weather request bills the same
# regardless of what it returns. It is here because a smaller response is a
# faster one on a phone tether, and because naming the fields makes the set RIO
# can possibly say visible in one place rather than implied by the parser.
CURRENT_FIELDS = ",".join([
    "currentTime", "timeZone", "isDaytime",
    "weatherCondition.description.text", "weatherCondition.type",
    "temperature", "feelsLikeTemperature",
    "relativeHumidity", "cloudCover", "thunderstormProbability",
    "precipitation.probability", "precipitation.qpf",
    "wind.speed", "wind.direction.cardinal",
    "visibility",
])

HOURS_FIELDS = ",".join([
    "forecastHours.interval.startTime",
    "forecastHours.displayDateTime",
    "forecastHours.weatherCondition.description.text",
    "forecastHours.weatherCondition.type",
    "forecastHours.temperature",
    "forecastHours.precipitation.probability",
    "forecastHours.precipitation.qpf",
    "forecastHours.wind.speed",
    "forecastHours.thunderstormProbability",
    "timeZone",
])

DAYS_FIELDS = ",".join([
    "forecastDays.maxTemperature", "forecastDays.minTemperature",
    "forecastDays.daytimeForecast.weatherCondition.description.text",
    "forecastDays.daytimeForecast.precipitation.probability",
    "forecastDays.sunEvents",
    "timeZone",
])

# Conditions a driver is affected by, as opposed to conditions that are merely
# the weather. Used only to decide whether a PROACTIVE mention is even a
# candidate -- never to decide what is said. See weather_policy.py.
ROUGH_TYPES = frozenset({
    "RAIN", "HEAVY_RAIN", "LIGHT_RAIN", "RAIN_SHOWERS", "HEAVY_RAIN_SHOWERS",
    "LIGHT_RAIN_SHOWERS", "SCATTERED_SHOWERS", "THUNDERSTORM",
    "SNOW", "HEAVY_SNOW", "LIGHT_SNOW", "SNOW_SHOWERS", "BLOWING_SNOW",
    "RAIN_AND_SNOW", "HAIL", "HAIL_SHOWERS", "FREEZING_RAIN", "WINDY",
    "SLEET", "FOG", "HAZE", "DUST", "SMOKE",
})

_cache = {}                  # session_key -> context dict
_cache_lock = threading.Lock()
_inflight = set()            # session_keys currently fetching
_inflight_lock = threading.Lock()


def _api_key() -> str:
    """The same key navigation and place search use, read the same way.

    One key, one place it comes from (.env, gitignored), and it never leaves the
    server. Weather is answered here rather than in the panel for exactly that
    reason, even though the browser is the thing that knows where the car is.
    """
    key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is not set")
    return key


def haversine_m(lat1, lng1, lat2, lng2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Reading Google's shapes without inventing values for the ones that are absent
# ---------------------------------------------------------------------------
# Every reader below returns None rather than a plausible default. "No reading"
# and "zero" are different things to say about visibility, and only one of them
# is true when the field is missing.

def _num(d, *path):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    if isinstance(cur, bool) or not isinstance(cur, (int, float)):
        return None
    return float(cur)


def _round1(v):
    return None if v is None else (round(v, 1) if abs(v) < 100 else round(v))


def _unit(d, *path):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur if isinstance(cur, str) and cur else None


def _condition(block) -> str:
    wc = (block or {}).get("weatherCondition") or {}
    text = ((wc.get("description") or {}).get("text") or "").strip()
    return text


def _condition_type(block) -> str:
    wc = (block or {}).get("weatherCondition") or {}
    return (wc.get("type") or "").strip()


def _local_hhmm(hour: dict, tz_id: str) -> str:
    """The hour as a driver would hear it: "3 PM", in the car's own timezone.

    Google gives both an ISO instant and a pre-split local `displayDateTime`
    with a UTC offset. The second is preferred because it has already done the
    timezone arithmetic at the location being asked about, which is the one the
    driver is standing in -- and doing it again here from the instant would be
    a second, differently-wrong answer.
    """
    d = hour.get("displayDateTime") or {}
    h = d.get("hours")
    if isinstance(h, int):
        m = d.get("minutes") or 0
        suffix = "AM" if h < 12 else "PM"
        h12 = h % 12 or 12
        return f"{h12}:{m:02d} {suffix}" if m else f"{h12} {suffix}"
    iso = (hour.get("interval") or {}).get("startTime") or ""
    return iso[11:16] or ""


def _iso_local(iso: str, tz_id: str) -> str:
    """A UTC instant -> a spoken local clock time, when the zone is known."""
    if not iso:
        return ""
    try:
        t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return ""
    try:
        from zoneinfo import ZoneInfo

        t = t.astimezone(ZoneInfo(tz_id)) if tz_id else t
    except Exception:
        pass
    h12 = t.hour % 12 or 12
    suffix = "AM" if t.hour < 12 else "PM"
    return f"{h12}:{t.minute:02d} {suffix}"


def _shape_hour(hour: dict, tz_id: str) -> dict:
    return {
        "at": _local_hhmm(hour, tz_id),
        "starts_at": (hour.get("interval") or {}).get("startTime"),
        "condition": _condition(hour),
        "condition_type": _condition_type(hour),
        "temperature": _round1(_num(hour, "temperature", "degrees")),
        "precipitation_probability": _num(hour, "precipitation", "probability",
                                          "percent"),
        "precipitation_type": ((hour.get("precipitation") or {})
                               .get("probability") or {}).get("type"),
        "wind_speed": _round1(_num(hour, "wind", "speed", "value")),
    }


def _next_precipitation(hours: list, tz_id: str) -> dict:
    """The first hour whose chance of precipitation crosses the threshold.

    This is what makes "rain expected around 3:20" a fact rather than a
    flourish: it is the first forecast hour over config.WEATHER_PRECIP_PROB_
    THRESHOLD, named by ITS OWN start time and probability. Nothing is
    interpolated to a finer time than the API gives, because the API gives
    hours and a minute-precise claim from hourly data is invented precision --
    the sort a driver would reasonably plan around.

    None when no hour in the window crosses. That is "no rain expected in the
    next N hours", which the caller may say; it is not "it will stay dry",
    which it may not.
    """
    thresh = float(config.WEATHER_PRECIP_PROB_THRESHOLD)
    now = time.time()
    for h in hours:
        p = h.get("precipitation_probability")
        if p is None or p < thresh:
            continue
        starts = h.get("starts_at") or ""
        in_min = None
        try:
            t = datetime.fromisoformat(starts.replace("Z", "+00:00"))
            in_min = max(0, round((t.timestamp() - now) / 60.0))
        except ValueError:
            pass
        return {
            "at": h.get("at"),
            "in_minutes": in_min,
            "probability": p,
            "type": h.get("precipitation_type") or "RAIN",
            "condition": h.get("condition"),
        }
    return None


# ---------------------------------------------------------------------------
# The calls
# ---------------------------------------------------------------------------

def _get(url: str, lat: float, lng: float, fields: str, extra: dict = None) -> dict:
    params = {
        "key": _api_key(),
        "location.latitude": lat,
        "location.longitude": lng,
        "unitsSystem": config.WEATHER_UNITS,
        "fields": fields,
    }
    if extra:
        params.update(extra)
    r = httpx.get(url, params=params, timeout=config.WEATHER_TIMEOUT_S)
    if r.status_code == 404:
        raise NoCoverage(f"{lat:.4f},{lng:.4f}")
    r.raise_for_status()
    return r.json()


def _fetch(lat: float, lng: float) -> dict:
    """One refresh: current, hourly, and daily when it is switched on.

    Requests are counted into the context (`billed_requests`) rather than
    estimated later, because the cost of this feature is the number of times
    this function runs times that number, and both should be readable from a
    session log rather than reconstructed from a bill.
    """
    t0 = time.time()
    n = 0

    cur = _get(CURRENT_URL, lat, lng, CURRENT_FIELDS)
    n += 1
    tz_id = ((cur.get("timeZone") or {}).get("id") or "")

    hrs = _get(HOURS_URL, lat, lng, HOURS_FIELDS,
               {"hours": int(config.WEATHER_FORECAST_HOURS)})
    n += 1
    hours = [_shape_hour(h, tz_id) for h in (hrs.get("forecastHours") or [])]

    today = {}
    if config.WEATHER_INCLUDE_DAILY:
        day = _get(DAYS_URL, lat, lng, DAYS_FIELDS, {"days": 1})
        n += 1
        days = day.get("forecastDays") or []
        if days:
            d0 = days[0]
            sun = d0.get("sunEvents") or {}
            today = {
                "high": _round1(_num(d0, "maxTemperature", "degrees")),
                "low": _round1(_num(d0, "minTemperature", "degrees")),
                "condition": _condition(d0.get("daytimeForecast") or {}),
                "precipitation_probability": _num(
                    d0, "daytimeForecast", "precipitation", "probability",
                    "percent"),
                "sunrise": _iso_local(sun.get("sunriseTime") or "", tz_id),
                "sunset": _iso_local(sun.get("sunsetTime") or "", tz_id),
            }

    # Units are carried, never assumed. Everything downstream — the model, the
    # policy, the selftests — reads the number and the name of what it is in.
    units = {
        "temperature": _unit(cur, "temperature", "unit") or "",
        "wind_speed": _unit(cur, "wind", "speed", "unit") or "",
        "visibility": _unit(cur, "visibility", "unit") or "",
        "precipitation": _unit(cur, "precipitation", "qpf", "unit") or "",
    }

    now = time.time()
    ctx = {
        "ok": True,
        "location": {"latitude": round(lat, 5), "longitude": round(lng, 5)},
        "timezone": tz_id,
        "units": units,
        "current": {
            "temperature": _round1(_num(cur, "temperature", "degrees")),
            "feels_like": _round1(_num(cur, "feelsLikeTemperature", "degrees")),
            "condition": _condition(cur),
            "condition_type": _condition_type(cur),
            "precipitation": _num(cur, "precipitation", "qpf", "quantity"),
            "precipitation_probability": _num(cur, "precipitation",
                                              "probability", "percent"),
            "wind_speed": _round1(_num(cur, "wind", "speed", "value")),
            "wind_direction": _unit(cur, "wind", "direction", "cardinal") or "",
            "visibility": _round1(_num(cur, "visibility", "distance")),
            "humidity": _num(cur, "relativeHumidity"),
            "cloud_cover": _num(cur, "cloudCover"),
            "thunderstorm_probability": _num(cur, "thunderstormProbability"),
            "is_daytime": cur.get("isDaytime"),
        },
        "forecast": {
            "next_hour_precipitation_probability": (
                hours[0].get("precipitation_probability") if hours else None),
            "next_3_hours": hours[:3],
            "next_precipitation": _next_precipitation(hours, tz_id),
            "horizon_hours": len(hours),
            "today": today,
        },
        # ALWAYS EMPTY, ALWAYS. Google's Weather API has no alerts endpoint --
        # see the header. An empty list here means "not known", never "clear".
        "alerts": [],
        "alerts_source": None,
        "fetched_at": now,
        "fetched_for": {"latitude": round(lat, 5), "longitude": round(lng, 5)},
        "age_s": 0.0,
        "billed_requests": n,
        "took_ms": round((now - t0) * 1000, 1),
        "attribution": "Source: Includes weather data from Google",
    }
    return ctx


# ---------------------------------------------------------------------------
# The cache, which is a movement question before it is a time question
# ---------------------------------------------------------------------------

def _staleness(ctx: dict, lat: float, lng: float, now: float = None) -> tuple:
    """(age_s, moved_m) for a held context against where the car is NOW."""
    now = time.time() if now is None else now
    age = now - float(ctx.get("fetched_at") or 0.0)
    ff = ctx.get("fetched_for") or {}
    try:
        moved = haversine_m(float(ff["latitude"]), float(ff["longitude"]),
                            lat, lng)
    except (KeyError, TypeError, ValueError):
        moved = float("inf")
    return age, moved


def usable(ctx: dict, lat: float, lng: float, now: float = None) -> tuple:
    """May this context be SPOKEN for a car at (lat, lng)? -> (bool, reason)

    Amendment C, and the one function the rest of the system is expected to
    ask. It is deliberately stricter than the refresh policy: a context can be
    worth keeping in the cache and not worth saying out loud, and the gap
    between those two thresholds is where a refresh gets to happen without RIO
    going silent mid-sentence.

    A failure here is ABSENCE, not a caveat. There is no "the forecast is a bit
    old but" answer, because a driver hears the forecast and not the but.
    """
    if not isinstance(ctx, dict) or not ctx.get("ok"):
        return False, "no_context"
    age, moved = _staleness(ctx, lat, lng, now)
    if age > config.WEATHER_MAX_AGE_S:
        return False, "stale"
    if moved > config.WEATHER_MAX_DISTANCE_M:
        return False, "moved"
    return True, ""


def _needs_refresh(ctx: dict, lat: float, lng: float, now: float = None) -> bool:
    """Whether to spend requests, which is a softer question than `usable`.

    Three reasons, in the order they actually bite on a drive:
      age         config.WEATHER_REFRESH_S, the ordinary heartbeat
      movement    the car has left the area the reading describes
      volatility  conditions that are changing get a shorter clock than a
                  clear sky does, because "20% in the next hour" is a number
                  that moves and "sunny, 26 degrees" is not
    """
    if not isinstance(ctx, dict) or not ctx.get("ok"):
        return True
    age, moved = _staleness(ctx, lat, lng, now)
    if moved > config.WEATHER_REFRESH_DISTANCE_M:
        return True
    return age > _refresh_after(ctx)


def _refresh_after(ctx: dict) -> float:
    """The refresh clock for THIS context, in seconds.

    Volatility is read from the data rather than from the calendar: a forecast
    that has precipitation inside the window, or an active precipitation
    probability worth mentioning, is a forecast whose numbers are moving.
    """
    cur = ctx.get("current") or {}
    fc = ctx.get("forecast") or {}
    volatile = False
    if (cur.get("condition_type") or "") in ROUGH_TYPES:
        volatile = True
    p = fc.get("next_hour_precipitation_probability")
    if p is not None and p >= config.WEATHER_PRECIP_PROB_THRESHOLD:
        volatile = True
    if fc.get("next_precipitation"):
        volatile = True
    return (config.WEATHER_REFRESH_VOLATILE_S if volatile
            else config.WEATHER_REFRESH_S)


def cached(session_key: str = "default") -> dict:
    with _cache_lock:
        got = _cache.get(str(session_key or "default"))
        return dict(got) if got else {}


def forget(session_key: str = None) -> None:
    """A new drive starts knowing nothing about the sky. Tests use it too."""
    with _cache_lock:
        if session_key is None:
            _cache.clear()
        else:
            _cache.pop(str(session_key), None)


def _remember(session_key: str, ctx: dict) -> None:
    with _cache_lock:
        _cache[str(session_key or "default")] = ctx
        if len(_cache) > 64:
            oldest = sorted(_cache.items(),
                            key=lambda kv: kv[1].get("fetched_at") or 0.0)[:32]
            for k, _v in oldest:
                _cache.pop(k, None)


def _stamp(ctx: dict, now: float = None) -> dict:
    """A copy with `age_s` true as of now.

    The age is written at HAND-OUT time rather than at fetch time, because the
    number that matters is how old the data is when somebody is about to say
    it, and a field set once at fetch is a field that says zero forever.
    """
    now = time.time() if now is None else now
    out = dict(ctx)
    out["age_s"] = round(max(0.0, now - float(ctx.get("fetched_at") or now)), 1)
    return out


# ---------------------------------------------------------------------------
# The one entry point
# ---------------------------------------------------------------------------

RULES_OK = (
    "This is the local forecast for where the car is now, and it is the ONLY "
    "source for temperature, chance of rain, wind, visibility and anything "
    "about what happens later. Everything you say about those must be in this "
    "block.\n"
    "What the camera shows is a SEPARATE source and it is the authority on "
    "what is visible from the car right now — dark cloud, wet road, spray, "
    "low sun. Say what you can see in your own words, and take every number "
    "and every claim about LATER from here.\n"
    "WHEN THEY DISAGREE, say both and do not resolve it. Wet road with no "
    "active precipitation reported is 'the roads are wet, though the weather "
    "data isn't showing active rain here' — never 'it's raining'. You have not "
    "caught the data out and you have not caught the camera out; you have two "
    "readings and the driver gets both.\n"
    "Round the way a person speaks: 'about 70 percent', 'around 3:20', "
    "'low 70s'. Do not read out a decimal. Do not add a probability, a time or "
    "an amount that is not here, and do not sharpen one that is — "
    "`next_precipitation.at` is the start of an HOUR, so it is 'around 3' and "
    "not '3:00 exactly'.\n"
    "Temperatures are in the unit named in `units`. Say 'degrees', not the "
    "unit name.\n"
    "Two or three sentences. This is a passenger glancing at the sky and the "
    "forecast, not a weather report."
)

RULES_ABSENT = (
    "You do NOT have the local forecast. Say so plainly, in your own words — "
    "you can't pull the forecast right now.\n"
    "You may still say what you can SEE out of the window, because the camera "
    "is a different source and it is working: dark clouds ahead, wet road, "
    "bright sun. Describe that if it is worth describing.\n"
    "What you must not do is turn the picture into a forecast. Not a "
    "probability, not a time, not 'looks like it'll clear up', not 'that'll "
    "blow over in an hour'. The sky does not tell you when it will rain and "
    "neither does your training. 'Looks like rain ahead, but I can't pull the "
    "local forecast right now' is the whole of what you know."
)


def get_weather_context(latitude, longitude, session_key: str = "default",
                        force: bool = False) -> dict:
    """GPS -> Google Weather API -> the small context RIO reasons over.

    The conversational model never sees Google's response. It sees this: the
    dozen numbers a driver has a use for, each with its unit, plus the three
    fields that make it honest — when it was fetched, where it was fetched for,
    and how old it is now.

    Returns `ok: False` with a `note` rather than raising, for every reason
    including a bad fix, because every caller's recovery is the same sentence
    and a 500 is not something a voice session knows how to say.
    """
    t0 = time.time()
    key = str(session_key or "default")

    try:
        lat = float(latitude)
        lng = float(longitude)
    except (TypeError, ValueError):
        return {"ok": False, "note": "no_fix", "need_location": True,
                "rules": RULES_ABSENT}
    if not (math.isfinite(lat) and math.isfinite(lng)):
        return {"ok": False, "note": "no_fix", "need_location": True,
                "rules": RULES_ABSENT}
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
        return {"ok": False, "note": "no_fix", "need_location": True,
                "rules": RULES_ABSENT}

    if not config.WEATHER_ENABLED:
        return {"ok": False, "note": "weather is switched off",
                "rules": RULES_ABSENT}

    held = cached(key)
    if not force and held and not _needs_refresh(held, lat, lng):
        fresh = _stamp(held)
        good, why = usable(fresh, lat, lng)
        if good:
            fresh["cached"] = True
            fresh["billed_requests"] = 0
            fresh["rules"] = RULES_OK
            return fresh

    try:
        ctx = _fetch(lat, lng)
    except NoCoverage as e:
        # Deliberately NOT falling back to a held context here, unlike the
        # error path below. A 404 means this location is outside Google's
        # coverage; a reading held from before the car crossed into it is by
        # definition for somewhere else, and `usable` would let it through if
        # that somewhere else were close enough. Coverage edges are exactly
        # where that check is least trustworthy.
        print(f"[weather] no coverage at {e}", flush=True)
        return {
            "ok": False, "note": "no_coverage",
            "took_ms": round((time.time() - t0) * 1000, 1),
            "rules": RULES_ABSENT,
        }
    except Exception as e:
        print(f"[weather] fetch failed: {type(e).__name__}: {e}", flush=True)
        # A held context is still worth trying: the network failed, which says
        # nothing about whether the reading from four minutes ago is still true
        # of this piece of road. `usable` decides that, on the same terms it
        # decides everything else, and refuses it if it is old or elsewhere.
        if held:
            fresh = _stamp(held)
            good, _why = usable(fresh, lat, lng)
            if good:
                fresh["cached"] = True
                fresh["stale_after_error"] = True
                fresh["billed_requests"] = 0
                fresh["rules"] = RULES_OK
                return fresh
        return {
            "ok": False, "note": f"{type(e).__name__}",
            "took_ms": round((time.time() - t0) * 1000, 1),
            "rules": RULES_ABSENT,
        }

    _remember(key, ctx)
    out = _stamp(ctx)
    out["cached"] = False
    out["rules"] = RULES_OK
    return out


def status() -> dict:
    """What /health reports, without calling anything."""
    with _cache_lock:
        sessions = {
            k: {"age_s": round(time.time() - (v.get("fetched_at") or 0.0), 1),
                "fetched_for": v.get("fetched_for"),
                "condition": (v.get("current") or {}).get("condition")}
            for k, v in _cache.items()
        }
    return {
        "enabled": bool(config.WEATHER_ENABLED),
        "units": config.WEATHER_UNITS,
        "refresh_s": config.WEATHER_REFRESH_S,
        "refresh_volatile_s": config.WEATHER_REFRESH_VOLATILE_S,
        "max_age_s": config.WEATHER_MAX_AGE_S,
        "max_distance_m": config.WEATHER_MAX_DISTANCE_M,
        "forecast_hours": config.WEATHER_FORECAST_HOURS,
        "daily": bool(config.WEATHER_INCLUDE_DAILY),
        "requests_per_refresh": 3 if config.WEATHER_INCLUDE_DAILY else 2,
        "alerts_available": False,
        "sessions": sessions,
    }


def context_for_fix(where, session_key: str = "default") -> dict:
    """The browser's GPS fix -> a weather context, or a refusal with a reason.

    The panel attaches `where` to every tool call (see rio_realtime.js), so this
    is the shape weather actually arrives in during a drive. A fix older than
    config.WEATHER_MAX_FIX_AGE_S is refused rather than used, for the same
    reason places.py refuses one: the car has been moving, and weather fetched
    for a ten-minute-old position is a forecast for a different neighbourhood
    that looks exactly like a forecast for this one.

    Note which way this fails. There is no fallback to the last route origin, a
    city centroid, or the last place the fix was good. Weather that follows the
    vehicle is the entire requirement, and a silent 30 km error in it is
    invisible to the driver and wrong in a way they will act on.
    """
    if not isinstance(where, dict):
        return {"ok": False, "note": "no_fix", "need_location": True,
                "rules": RULES_ABSENT}
    try:
        lat = float(where.get("lat"))
        lng = float(where.get("lng"))
    except (TypeError, ValueError):
        return {"ok": False, "note": "no_fix", "need_location": True,
                "rules": RULES_ABSENT}
    age = where.get("age_s")
    if age is not None:
        try:
            if float(age) > config.WEATHER_MAX_FIX_AGE_S:
                return {"ok": False, "note": "stale_fix", "need_location": True,
                        "rules": RULES_ABSENT}
        except (TypeError, ValueError):
            pass
    return get_weather_context(lat, lng, session_key=session_key)
