import json
import logging
import time
import copy
import threading
from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from app.database import get_db

logger = logging.getLogger(__name__)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


WEATHER_CODES = {
    0: "Clear sky",

    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",

    45: "Fog",
    48: "Depositing rime fog",

    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",

    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",

    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",

    66: "Light freezing rain",
    67: "Heavy freezing rain",

    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",

    77: "Snow grains",

    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",

    85: "Slight snow showers",
    86: "Heavy snow showers",

    95: "Thunderstorm",

    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherRateLimitError(RuntimeError):
    """Raised when the weather provider returns HTTP 429."""


_RATE_LIMIT_UNTIL = 0.0
_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_COOLDOWN = 5 * 60  # 5 minutes
_MAX_RETRIES = 2


def _get_json(url: str):
    """Fetch JSON with short retries and a shared 429 cooldown."""
    global _RATE_LIMIT_UNTIL

    with _RATE_LIMIT_LOCK:
        if time.time() < _RATE_LIMIT_UNTIL:
            remaining = int(_RATE_LIMIT_UNTIL - time.time())
            raise WeatherRateLimitError(
                f"Weather provider is in rate-limit cooldown ({remaining}s remaining)."
            )

    request = Request(
        url,
        headers={"User-Agent": "SmartCrop-AI/1.0"},
    )

    for attempt in range(_MAX_RETRIES + 1):
        try:
            with urlopen(request, timeout=15) as response:
                return json.loads(response.read().decode("utf-8"))

        except HTTPError as exc:
            if exc.code == 429:
                cooldown = _RATE_LIMIT_COOLDOWN
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if retry_after:
                    try:
                        cooldown = max(cooldown, int(retry_after))
                    except ValueError:
                        try:
                            retry_date = parsedate_to_datetime(retry_after)
                            cooldown = max(
                                cooldown,
                                int(retry_date.timestamp() - time.time()),
                            )
                        except Exception:
                            pass
                with _RATE_LIMIT_LOCK:
                    _RATE_LIMIT_UNTIL = time.time() + cooldown
                raise WeatherRateLimitError(
                    f"Weather service rate limit (HTTP 429). Retry after about {cooldown}s."
                ) from exc

            # Retry only temporary server errors, not ordinary client errors.
            if exc.code >= 500 and attempt < _MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Weather service HTTP error: {exc.code}") from exc

        except URLError as exc:
            if attempt < _MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(
                f"Weather service unavailable: {exc.reason}"
            ) from exc

        except Exception as exc:
            raise RuntimeError(f"Weather service error: {exc}") from exc

    raise RuntimeError("Weather service request failed after retries.")

def geocode_location(location: str):
    import re

    location = location.strip()

    if not location:
        raise ValueError("Location cannot be empty.")

    # Split an address into useful search terms.
    parts = [
        part.strip()
        for part in location.split(",")
        if part.strip()
    ]

    # Try the full address first, then individual components.
    search_terms = [location]

    for part in parts:
        if part not in search_terms:
            search_terms.append(part)

    # Remove duplicate terms while preserving order.
    search_terms = list(dict.fromkeys(search_terms))

    results = []

    for term in search_terms:
        params = urlencode({
            "name": term,
            "count": 10,
            "language": "en",
            "format": "json",
        })

        url = f"{GEOCODING_URL}?{params}"

        data = _get_json(url)
        matches = data.get("results", [])

        if matches:
            results.extend(matches)

    if not results:
        raise ValueError(
            f"Could not locate '{location}'. "
            "Please check the village, town, district or PIN code."
        )

    # Prefer Indian results when the entered address
    # contains an Indian state or country reference.
    india_terms = [
        "india",
        "karnataka",
        "bangalore",
        "bengaluru",
        "mysore",
        "mysuru",
    ]

    expects_india = any(
        term in location.lower()
        for term in india_terms
    )

    if expects_india:
        indian_results = [
            item for item in results
            if item.get("country_code") == "IN"
        ]

        if indian_results:
            results = indian_results

    # Prefer a result whose name matches the first
    # address component (usually the village or town).
    first_place = parts[0].lower() if parts else ""

    results.sort(
        key=lambda item: (
            item.get("name", "").lower() != first_place,
            item.get("country_code") != "IN"
                if expects_india else False,
        )
    )

    result = results[0]

    return {
        "name": result.get("name"),
        "latitude": float(result["latitude"]),
        "longitude": float(result["longitude"]),
        "country": result.get("country"),
        "country_code": result.get("country_code"),
        "state": result.get("admin1"),
        "city": result.get("name"),
        "timezone": result.get("timezone"),
    }


def _mean(values):

    values = [
        float(v)
        for v in values
        if v is not None
    ]

    if not values:

        return None

    return sum(values) / len(values)


def _fetch_five_day_weather(latitude: float, longitude: float):

    params = urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,

            "forecast_days": 5,

            "timezone": "auto",

            "hourly": ",".join(
                [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "precipitation",
                    "soil_moisture_0_to_1cm",
                    "weather_code",
                ]
            ),

            "daily": ",".join(
                [
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "precipitation_probability_max",
                    "weather_code",
                ]
            ),
        }
    )

    url = f"{FORECAST_URL}?{params}"

    data = _get_json(url)

    daily = data.get("daily", {})
    hourly = data.get("hourly", {})

    dates = daily.get("time", [])

    daily_max = daily.get("temperature_2m_max", [])
    daily_min = daily.get("temperature_2m_min", [])
    daily_rain = daily.get("precipitation_sum", [])
    daily_rain_probability = daily.get(
        "precipitation_probability_max",
        []
    )
    daily_weather_codes = daily.get(
        "weather_code",
        []
    )

    hourly_times = hourly.get("time", [])
    hourly_temp = hourly.get(
        "temperature_2m",
        []
    )
    hourly_humidity = hourly.get(
        "relative_humidity_2m",
        []
    )
    hourly_rain = hourly.get(
        "precipitation",
        []
    )
    hourly_soil = hourly.get(
        "soil_moisture_0_to_1cm",
        []
    )
    hourly_weather = hourly.get(
        "weather_code",
        []
    )

    forecast = []

    for index, date in enumerate(dates):

        day_temperature = []

        day_humidity = []

        day_soil = []

        day_rain = []

        day_codes = []

        for i, timestamp in enumerate(hourly_times):

            if timestamp.startswith(date):

                if i < len(hourly_temp):

                    day_temperature.append(
                        hourly_temp[i]
                    )

                if i < len(hourly_humidity):

                    day_humidity.append(
                        hourly_humidity[i]
                    )

                if i < len(hourly_soil):

                    day_soil.append(
                        hourly_soil[i]
                    )

                if i < len(hourly_rain):

                    day_rain.append(
                        hourly_rain[i]
                    )

                if i < len(hourly_weather):

                    day_codes.append(
                        hourly_weather[i]
                    )

        code = None

        if index < len(daily_weather_codes):

            code = daily_weather_codes[index]

        if code is None and day_codes:

            code = day_codes[len(day_codes) // 2]

        weather_text = WEATHER_CODES.get(
            int(code) if code is not None else -1,
            "Unknown conditions"
        )

        humidity = _mean(day_humidity)

        soil_moisture = _mean(day_soil)

        # Open-Meteo soil moisture is volumetric water
        # content (m³/m³). Convert to a convenient
        # percentage representation for the existing
        # SmartCrop risk engine.
        if soil_moisture is not None:

            soil_moisture_percent = round(
                soil_moisture * 100,
                2
            )

        else:

            soil_moisture_percent = None

        rainfall = None

        if index < len(daily_rain):

            rainfall = daily_rain[index]

        if rainfall is None:

            rainfall = sum(
                float(v)
                for v in day_rain
                if v is not None
            )

        rain_probability = None

        if index < len(
            daily_rain_probability
        ):

            rain_probability = (
                daily_rain_probability[index]
            )

        forecast.append(
            {
                "date": date,

                "temperature_max": round(
                    float(daily_max[index]),
                    1
                ) if index < len(daily_max)
                else None,

                "temperature_min": round(
                    float(daily_min[index]),
                    1
                ) if index < len(daily_min)
                else None,

                "humidity": round(
                    humidity,
                    1
                ) if humidity is not None
                else None,

                "rainfall": round(
                    float(rainfall),
                    1
                ) if rainfall is not None
                else 0.0,

                "rain_probability": round(
                    float(rain_probability),
                    1
                ) if rain_probability is not None
                else None,

                "soil_moisture": soil_moisture_percent,

                "weather_code": code,

                "weather": weather_text,
            }
        )

    return {
        "latitude": latitude,

        "longitude": longitude,

        "timezone": data.get("timezone"),

        "generated_at": datetime.utcnow().isoformat(),

        "forecast": forecast,
    }
# ============================================================
# PERSISTENT WEATHER CACHE
# ============================================================

WEATHER_CACHE_TTL = 30 * 60          # 30 minutes
WEATHER_STALE_LIMIT = 6 * 60 * 60    # 6 hours

# Per-location locks prevent simultaneous requests for the same coordinates
# from creating a burst of calls to Open-Meteo.
_LOCATION_LOCKS = {}
_LOCATION_LOCKS_GUARD = threading.Lock()


def _location_key(latitude: float, longitude: float) -> str:
    return f"{round(float(latitude), 4)}:{round(float(longitude), 4)}"


def _get_location_lock(key: str):
    with _LOCATION_LOCKS_GUARD:
        if key not in _LOCATION_LOCKS:
            _LOCATION_LOCKS[key] = threading.Lock()
        return _LOCATION_LOCKS[key]


def _read_persistent_cache(key: str):
    """Return (forecast, saved_at); return None if DB/cache is unavailable."""
    try:
        with get_db() as db:
            row = db.execute(
                "SELECT forecast_json, saved_at FROM weather_cache WHERE location_key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None

        # Both SQLite Row and psycopg dict_row support keyed access.
        payload = row["forecast_json"]
        saved_at = float(row["saved_at"])
        forecast = json.loads(payload) if isinstance(payload, str) else payload
        if not isinstance(forecast, dict) or not isinstance(forecast.get("forecast"), list):
            return None
        return forecast, saved_at
    except Exception as exc:
        # A cache/database outage must not prevent a live weather fetch.
        logger.warning("Could not read persistent weather cache: %s", exc)
        return None


def _write_persistent_cache(key: str, forecast: dict):
    """Persist forecast using an upsert supported by SQLite and PostgreSQL."""
    try:
        with get_db() as db:
            db.execute(
                """
                INSERT INTO weather_cache (location_key, forecast_json, saved_at)
                VALUES (?, ?, ?)
                ON CONFLICT(location_key) DO UPDATE SET
                    forecast_json = excluded.forecast_json,
                    saved_at = excluded.saved_at
                """,
                (key, json.dumps(forecast), time.time()),
            )
    except Exception as exc:
        # Continue serving the live result even if persistence fails.
        logger.warning("Could not save persistent weather cache: %s", exc)


def _with_cache_status(forecast: dict, status: str):
    result = copy.deepcopy(forecast)
    result["cache_status"] = status
    return result


def get_five_day_weather(latitude: float, longitude: float):
    """Return five-day weather, preferring Neon cache and falling back safely."""
    key = _location_key(latitude, longitude)
    now = time.time()

    # Serialize requests for one location, while allowing other locations to run.
    with _get_location_lock(key):
        cached = _read_persistent_cache(key)

        if cached:
            cached_forecast, saved_at = cached
            age = max(0.0, now - saved_at)
            if age <= WEATHER_CACHE_TTL:
                return _with_cache_status(cached_forecast, "fresh_cache")

        try:
            forecast = _fetch_five_day_weather(latitude, longitude)
            _write_persistent_cache(key, forecast)
            return _with_cache_status(forecast, "live")

        except Exception as exc:
            # Re-read after a failed fetch so another worker/process may have
            # refreshed the shared Neon cache in the meantime.
            cached = _read_persistent_cache(key)
            if cached:
                cached_forecast, saved_at = cached
                age = max(0.0, time.time() - saved_at)
                if age <= WEATHER_STALE_LIMIT:
                    logger.warning(
                        "Using stale weather cache for %s after provider failure: %s",
                        key,
                        exc,
                    )
                    return _with_cache_status(cached_forecast, "stale_fallback")
            raise
