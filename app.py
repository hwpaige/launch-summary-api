import os
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal, Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field
import requests
import ast
import re
import redis
import json
import gzip
import math
import time
import threading
import zlib
import base64
import pytz
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env if present

GROK_MODEL = "grok-4-1-fast-reasoning"
GROK_API_URL = "https://api.x.ai/v1/chat/completions"
NOTIFY_COPY_TTL = 3600  # 1 hour; reschedules should not thrash xAI
NOTIFY_TITLE_MAX = 50
NOTIFY_BODY_MAX = 150
NOTIFY_COPY_CACHE_PREFIX = "notify_copy_v1"
NotifyEvent = Literal["t1h", "t24h", "scrub"]

app = FastAPI(
    title="SpaceX Launch Summary API",
    description=(
        "Witty SpaceX launch narratives, upcoming-launch data, weather, "
        "cached CelesTrak satellite GP/TLE for the dashboard globe, and "
        "Launch Buddy notification copy. Past-launch narratives and "
        "`/notify/copy` both use the xAI Grok model "
        f"`{GROK_MODEL}` via `XAI_API_KEY`."
    ),
    openapi_tags=[
        {
            "name": "Notifications",
            "description": (
                "Grok-powered push/local-notification title+body for Launch Buddy "
                "upcoming-launch alerts (`t24h`, `t1h`, `scrub`). "
                "Includes launch probability in the copy when provided."
            ),
        },
        {
            "name": "Satellites",
            "description": (
                "Cached CelesTrak GP/OMM element sets (Starlink, stations, and a "
                "small allowlist) plus deployed Starlink TLEs for the recent "
                "Starship V3 flight. SATCAT GP is preferred once cataloged; "
                "until then, SpaceX public MEME ephemerides fill the same "
                "TLE shape. Clients propagate with SGP4 (e.g. satellite.js); "
                "this is not live telemetry."
            ),
        },
    ],
)


# Redis Configuration for persistence across Heroku builds
def get_redis_client():
    # List of potential Redis environment variables used by various Heroku add-ons
    redis_env_vars = ["REDIS_URL", "REDISCLOUD_URL", "REDISTOGO_URL"]

    # Also check for HEROKU_REDIS_*_URL
    for key in os.environ:
        if key.startswith("HEROKU_REDIS_") and key.endswith("_URL"):
            redis_env_vars.append(key)

    for var in redis_env_vars:
        url = os.getenv(var)
        if url:
            try:
                # Heroku Redis often requires SSL with cert verification disabled for self-signed certs
                if url.startswith("rediss://"):
                    client = redis.from_url(url, decode_responses=True, ssl_cert_reqs=None)
                else:
                    client = redis.from_url(url, decode_responses=True)
                client.ping()
                print(f"Connected to Redis via {var}")
                return client
            except Exception as e:
                print(f"Failed to connect to Redis via {var}: {e}")

    print("No Redis instance found or connection failed. Using in-memory fallback (non-persistent).")
    return None


r = get_redis_client()


def set_cached_data(key, data, ttl=None):
    """Set data in Redis with compression and better error handling."""
    if not r:
        return False
    try:
        json_str = json.dumps(data)
        # Compress and base64 encode to store as string in the current Redis client
        compressed = zlib.compress(json_str.encode('utf-8'))
        encoded = base64.b64encode(compressed).decode('utf-8')
        if ttl:
            r.setex(key, ttl, encoded)
        else:
            r.set(key, encoded)
        return True
    except Exception as e:
        print(f"Redis write error for {key}: {e}")
        return False


def get_cached_data(key):
    """Get data from Redis with decompression support."""
    if not r:
        return None
    try:
        data = r.get(key)
        if not data:
            return None

        # Try to decompress (it will be base64 encoded string)
        try:
            decoded = base64.b64decode(data)
            decompressed = zlib.decompress(decoded)
            return json.loads(decompressed)
        except Exception:
            # Fallback for old uncompressed data
            try:
                return json.loads(data)
            except:
                return None
    except Exception as e:
        print(f"Redis read error for {key}: {e}")
        return None


# In-memory storage fallback (used only if Redis is unavailable)
_local_cache = {
    "launch_narratives": None,
    "last_updated": None
}
_local_notify_copy = {}
_local_metrics = {
    "total_requests": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "api_calls": 0
}
_local_metrics_history = []

CACHE_KEY = "launch_narratives_v2"
CACHE_TIME_KEY = "last_updated_v2"
METRICS_KEY = "app_metrics_v2"
METRICS_HISTORY_KEY = "app_metrics_history_v2"
LAUNCHES_CACHE_KEY = "launches_cache_v2"
RAW_LAUNCH_KEY_PREFIX = "launch_raw_v2:"
HOT_RAW_KEY_PREFIX = "launch_raw_hot_v1:"
CACHE_TTL = 900  # 15 minutes TTL in seconds (aligned with dashboard)
HOT_RAW_TTL = int(os.getenv("HOT_RAW_TTL", "20"))  # 20–30s stale-while-revalidate for ?hot=1
LAUNCHES_MEM_TTL = 45  # seconds; avoid repeatedly inflating the Redis blob
WEATHER_CACHE_TTL = 300
WEATHER_DEBOUNCE_SEC = 20.0
REFRESH_LOCK_TTL = 90
HEAVY_LAUNCH_FIELDS = ("all_data",)

# CelesTrak GP / OMM — 1h freshness, keep a longer stale copy so Pi clients
# are not sent to celestrak.org (CORS + rate limits) when upstream blips.
CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
CELESTRAK_HEADERS = {
    "User-Agent": (
        "launch-summary-api/satellites "
        "(https://github.com/hwpaige/launch-summary-api; SpaceX dashboard GP cache)"
    ),
    "Accept": "application/json",
}
SATELLITE_GROUPS = ("starlink", "stations", "visual", "oneweb", "gps-ops", "weather")
SATELLITES_CACHE_PREFIX = "satellites_gp_v1:"
# Pre-rendered JSON (fresh + stale) so a cold dyno can return bytes without
# rebuilding the 11k-record object graph on the request path.
SATELLITES_HTTP_CACHE_PREFIX = "satellites_gp_http_v1:"
SATELLITES_CACHE_TTL = 3600  # serve fresh for 1 hour
SATELLITES_STALE_TTL = 48 * 3600  # Redis retains stale GP for fallback
SATELLITES_FETCH_TIMEOUT = 45  # per-socket timeout; not a wall clock
# Wall-clock cap. requests' timeout resets on every socket read, so a
# trickle from CelesTrak can otherwise pin a worker well past Heroku's router.
SATELLITES_GP_DEADLINE_SEC = 40
# Cold request budget. Heroku's router returns 503 HTML at ~30s; stay under it.
SATELLITES_REQUEST_BUDGET_SEC = 18
SATELLITES_REFRESH_COOLDOWN_SEC = 60
OMM_PROP_FIELDS = (
    "OBJECT_NAME",
    "OBJECT_ID",
    "NORAD_CAT_ID",
    "EPOCH",
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
    "BSTAR",
    "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT",
    "EPHEMERIS_TYPE",
    "CLASSIFICATION_TYPE",
    "ELEMENT_SET_NO",
    "REV_AT_EPOCH",
)
SATELLITE_NOTE = (
    "Positions are SGP4 predictions from GP/TLE element sets, not live telemetry."
)
# SATCAT has no LAUNCH_DATE query. GROUP=starlink is filtered locally; only the
# slim recent-date index is cached (the raw catalog is ~4MB and is not stored).
CELESTRAK_SATCAT_URL = "https://celestrak.org/satcat/records.php"
CELESTRAK_SUP_GP_URL = "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php"
DEPLOYED_CACHE_PREFIX = "satellites_deployed_v2:"
SATCAT_RECENT_CACHE_KEY = "satellites_satcat_starlink_recent_v1"
SATCAT_RECENT_DAYS = 45
DEPLOYED_SAT_CAP = 128  # matches the globe Points-shell append cap
DEPLOYED_INTDES_CAP = 4
DEPLOYED_SOURCE = "celestrak-satcat"
# Pre-catalog bridge. SATCAT is still preferred once it lists the launch date.
DEPLOYED_MANIFEST_SOURCE = "spacex-manifest-ephemeris"
DEPLOYED_MANIFEST_TTL = 600  # OEM coast; shorter than the 1h SATCAT GP cache
DEPLOYED_EMPTY_TTL = 600  # do not pin a SATCAT miss for an hour
SPACEX_EPHEM_BASE = "https://api.starlink.com/public-files/ephemerides/"
SPACEX_MANIFEST_URL = SPACEX_EPHEM_BASE + "MANIFEST.txt"
# Falcon-era STARLINK ids in the public manifest top out below this
# (38451 on 2026-09-28). 40xxx is the post-Falcon series SpaceX publishes
# before CelesTrak assigns catalog numbers (Flight 14: 40075–40103).
SPACEX_POST_FALCON_STARLINK_ID_MIN = 40000
SPACEX_EPHEM_CONCURRENCY = 8
SPACEX_EPHEM_FETCH_TIMEOUT = 20
SPACEX_EPHEM_BUDGET_SEC = 22
# SpaceX republishes MEME files daily, so the window can start after the
# launch calendar day. Still treat the file as this flight for a few weeks;
# a much older launch date must not pick up today's post-Falcon set.
MEME_ROLL_FORWARD_DAYS = 21
SPACEX_EPHEM_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/plain,*/*",
}
# MEME rows are inertial km / km/s. UVW names the covariance block, not the state.
_EARTH_MU_KM3_S2 = 398600.4418
_WGS84_A_KM = 6378.137
_WGS84_E2 = (1.0 / 298.257223563) * (2.0 - 1.0 / 298.257223563)
_MANIFEST_NAME_RE = re.compile(
    r"(MEME_\d+_STARLINK-(\d+)_\d+_Operational_\d+_UNCLASSIFIED\.txt)"
)
_MEME_EPOCH_RE = re.compile(
    r"^(?P<epoch>\d{13}(?:\.\d+)?)\s+"
    r"(?P<x>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+"
    r"(?P<y>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+"
    r"(?P<z>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+"
    r"(?P<vx>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+"
    r"(?P<vy>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+"
    r"(?P<vz>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)
_MEME_STAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})")
_INTDES_RE = re.compile(r"^(\d{4}-\d{3})")
_TLE_ALPHA5 = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # I and O omitted (Space-Track alpha-5)


_UTC_NOW = object()


class _SingleFlight:
    """Coalesce concurrent callers so only one refresh function runs at a time."""

    def __init__(self):
        self._lock = threading.Lock()
        self._event = None
        self._result = None
        self._error = None

    def do(self, fn, wait_timeout=None):
        leader = False
        with self._lock:
            if self._event is None:
                self._event = threading.Event()
                self._result = None
                self._error = None
                leader = True
            event = self._event
        if not leader:
            # Callers on the HTTP path pass a short wait so a refresh that is
            # already running cannot hold them until Heroku's router times out.
            timeout = REFRESH_LOCK_TTL if wait_timeout is None else wait_timeout
            finished = event.wait(timeout=timeout)
            if self._error is not None and (finished or wait_timeout is None):
                raise self._error
            if not finished:
                return None
            return self._result
        try:
            self._result = fn()
            return self._result
        except Exception as exc:
            self._error = exc
            raise
        finally:
            event.set()
            with self._lock:
                if self._event is event:
                    self._event = None

    def reset(self):
        with self._lock:
            self._event = None
            self._result = None
            self._error = None


_weather_flight = _SingleFlight()
_launches_flight = _SingleFlight()
_narratives_flight = _SingleFlight()
_hot_raw_flights = {}
_hot_raw_flights_lock = threading.Lock()
_weather_last_refresh_at = 0.0
_weather_last_result = None
_local_weather_store = {}
_local_raw_launches = {}
_local_hot_raw = {}
_launches_mem = {"data": None, "at": 0.0}
_local_satellites = {}
_local_satellite_http = {}
_satellite_flights = {}
_satellite_flights_lock = threading.Lock()
_satellite_refresh_lock = threading.Lock()
_satellite_refresh_inflight = set()
_satellite_refresh_after = {}
_local_deployed = {}
_local_satcat_index = None
_deployed_flights = {}
_deployed_flights_lock = threading.Lock()
_satcat_flight = _SingleFlight()


def _reset_cache_coordination_for_tests():
    """Reset single-flight / debounce state between unit tests."""
    global _weather_last_refresh_at, _weather_last_result, _background_enabled, _TRAJECTORY_DATA_CACHE
    global _local_satcat_index
    _background_enabled = False
    _TRAJECTORY_DATA_CACHE = {}
    _weather_flight.reset()
    _launches_flight.reset()
    _narratives_flight.reset()
    with _hot_raw_flights_lock:
        for flight in _hot_raw_flights.values():
            flight.reset()
        _hot_raw_flights.clear()
    _weather_last_refresh_at = 0.0
    _weather_last_result = None
    _local_weather_store.clear()
    _local_raw_launches.clear()
    _local_hot_raw.clear()
    _local_notify_copy.clear()
    _launches_mem["data"] = None
    _launches_mem["at"] = 0.0
    _local_satellites.clear()
    _local_satellite_http.clear()
    with _satellite_refresh_lock:
        _satellite_refresh_inflight.clear()
        _satellite_refresh_after.clear()
    with _satellite_flights_lock:
        for flight in _satellite_flights.values():
            flight.reset()
        _satellite_flights.clear()
    _local_deployed.clear()
    _local_satcat_index = None
    _satcat_flight.reset()
    with _deployed_flights_lock:
        for flight in _deployed_flights.values():
            flight.reset()
        _deployed_flights.clear()


def _redis_single_flight(name, fn, wait_timeout=None):
    """Cross-process lock so multiple dyno workers don't stampede the same refresh."""
    if not r:
        return fn()
    lock_key = f"refresh_lock_v1_{name}"
    token = f"{time.time()}:{os.getpid()}:{threading.get_ident()}"
    try:
        acquired = r.set(lock_key, token, nx=True, ex=REFRESH_LOCK_TTL)
    except Exception as e:
        print(f"Redis lock error for {name}: {e}")
        return fn()
    if acquired:
        try:
            return fn()
        finally:
            try:
                if r.get(lock_key) == token:
                    r.delete(lock_key)
            except Exception:
                pass
    timeout = REFRESH_LOCK_TTL if wait_timeout is None else wait_timeout
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if not r.exists(lock_key):
                break
        except Exception:
            break
        time.sleep(0.15)
    return None


def _keep_list_trajectory(bucket, idx, upcoming_count):
    """Keep globe path on the next launch only (upcoming[0], or previous[0] if none)."""
    if bucket == "upcoming" and idx == 0:
        return True
    if bucket == "previous" and idx == 0 and upcoming_count == 0:
        return True
    return False


def _store_raw_launch(launch_id, raw):
    """Persist one launch's raw LL blob outside the slim list cache."""
    if not launch_id or raw is None:
        return
    if r:
        set_cached_data(f"{RAW_LAUNCH_KEY_PREFIX}{launch_id}", raw)
    else:
        _local_raw_launches[str(launch_id)] = raw


def _get_raw_launch(launch_id):
    if not launch_id:
        return None
    key = str(launch_id)
    if r:
        cached = get_cached_data(f"{RAW_LAUNCH_KEY_PREFIX}{key}")
        if cached is not None:
            return cached
    return _local_raw_launches.get(key)


def _hot_flight(launch_id):
    key = str(launch_id)
    with _hot_raw_flights_lock:
        flight = _hot_raw_flights.get(key)
        if flight is None:
            flight = _SingleFlight()
            _hot_raw_flights[key] = flight
        return flight


def _read_hot_raw(launch_id):
    """Return the short-TTL hot blob if it is still fresh."""
    key = str(launch_id)
    if r:
        return get_cached_data(f"{HOT_RAW_KEY_PREFIX}{key}")
    entry = _local_hot_raw.get(key)
    if entry and entry[1] > time.time():
        return entry[0]
    return None


def _write_hot_raw(launch_id, raw):
    """Store hot raw on a short TTL; also refresh the long-lived side store."""
    if not launch_id or raw is None:
        return
    key = str(launch_id)
    _local_hot_raw[key] = (raw, time.time() + HOT_RAW_TTL)
    if r:
        set_cached_data(f"{HOT_RAW_KEY_PREFIX}{key}", raw, ttl=HOT_RAW_TTL)
    _store_raw_launch(key, raw)


def _current_next_launch_ids(data=None):
    """IDs of the next upcoming launch, plus previous[0] if nothing is upcoming."""
    if data is None:
        data = _load_launch_payload(force=False)
    if not isinstance(data, dict):
        return set()
    ids = set()
    upcoming = [l for l in (data.get("upcoming") or []) if isinstance(l, dict)]
    previous = [l for l in (data.get("previous") or []) if isinstance(l, dict)]
    if upcoming and upcoming[0].get("id"):
        ids.add(str(upcoming[0]["id"]))
    next_l = get_next_launch_info(upcoming, pytz.UTC) if upcoming else None
    if next_l and next_l.get("id"):
        ids.add(str(next_l["id"]))
    if not upcoming and previous and previous[0].get("id"):
        ids.add(str(previous[0]["id"]))
    return ids


def _is_current_or_next_launch(launch_id):
    if not launch_id:
        return False
    return str(launch_id) in _current_next_launch_ids()


def _fetch_and_store_hot_raw(launch_id):
    details = fetch_launch_details(launch_id)
    if details:
        _write_hot_raw(launch_id, details)
        return details
    stale = _get_raw_launch(launch_id)
    if stale:
        # Failed LL2 GET: keep serving leftover and cooldown the hot key.
        _write_hot_raw(launch_id, stale)
        return stale
    return None


def _refresh_hot_launch_raw(launch_id):
    """Single-flight (in-process + Redis) so concurrent ?hot=1 polls share one LL2 GET."""
    def _do():
        cached = _read_hot_raw(launch_id)
        if cached is not None:
            return cached
        result = _redis_single_flight(f"hot_raw_{launch_id}", lambda: _fetch_and_store_hot_raw(launch_id))
        if result is None:
            return _read_hot_raw(launch_id) or _get_raw_launch(launch_id)
        return result

    return _hot_flight(launch_id).do(_do)


def _get_hot_launch_raw(launch_id):
    """Stale-while-revalidate: serve fresh ~20s cache, else one LL2 GET.

    Returns (payload, cache_hit).
    """
    cached = _read_hot_raw(launch_id)
    if cached is not None:
        return cached, True
    refreshed = _refresh_hot_launch_raw(launch_id)
    if refreshed is not None:
        return refreshed, False
    leftover = _get_raw_launch(launch_id)
    return leftover, leftover is not None


def _serve_cached_or_live_raw(launch_id):
    """Existing /launch_raw behavior: side store, leftover list blob, or live GET."""
    leftover = _get_raw_launch(launch_id)
    if leftover:
        return leftover, True
    cached = _load_launch_payload(force=False)
    for launch in (cached.get("upcoming", []) or []) + (cached.get("previous", []) or []):
        if launch.get("id") == launch_id:
            leftover = launch.get("all_data")
            if leftover:
                return leftover, True
            details = fetch_launch_details(launch_id)
            return details, False
    details = fetch_launch_details(launch_id)
    return details, False


def _strip_heavy_launch_fields(data, persist_key=None):
    """Drop leftover raw LL blobs / extra trajectories from a list payload."""
    if not isinstance(data, dict):
        return False
    changed = False
    upcoming_count = len(data.get("upcoming") or [])
    for bucket in ("upcoming", "previous"):
        launches = data.get(bucket) or []
        for idx, launch in enumerate(launches):
            if not isinstance(launch, dict):
                continue
            for field in HEAVY_LAUNCH_FIELDS:
                if field in launch:
                    launch.pop(field, None)
                    changed = True
            if launch.get("trajectory_data") and not _keep_list_trajectory(bucket, idx, upcoming_count):
                launch.pop("trajectory_data", None)
                changed = True
    if changed and persist_key:
        set_cached_data(persist_key, data)
    return changed


def _sanitize_launch_list_cache(data, persist=False):
    """Move inline all_data into the side store and keep the list cache slim."""
    if not isinstance(data, dict):
        return False
    changed = False
    for bucket in ("upcoming", "previous"):
        for launch in data.get(bucket) or []:
            if not isinstance(launch, dict):
                continue
            raw = launch.pop("all_data", None)
            if raw is not None:
                _store_raw_launch(launch.get("id"), raw)
                changed = True
    if _strip_heavy_launch_fields(data):
        changed = True
    if changed and persist:
        set_cached_data(LAUNCHES_CACHE_KEY, data)
    return changed


def _hydrate_launch_payload(data):
    """Copy a slim list and reattach all_data. Does not mutate the hot cache."""
    if not isinstance(data, dict):
        return {"upcoming": [], "previous": [], "last_updated": None}
    out = {
        "upcoming": [],
        "previous": [],
        "last_updated": data.get("last_updated"),
    }
    for bucket in ("upcoming", "previous"):
        for launch in data.get(bucket) or []:
            if not isinstance(launch, dict):
                out[bucket].append(launch)
                continue
            item = dict(launch)
            raw = item.get("all_data")
            if raw is None:
                raw = _get_raw_launch(item.get("id"))
            if raw is not None:
                item["all_data"] = raw
            out[bucket].append(item)
    return out


def _remember_launches(data):
    _launches_mem["data"] = data
    _launches_mem["at"] = time.time()
    return data


def _set_weather_loc(location, data, ttl=WEATHER_CACHE_TTL):
    _local_weather_store[location] = (data, time.time() + ttl)
    if r:
        try:
            r.setex(f"weather_cache_v2_{location}", ttl, json.dumps(data))
        except Exception:
            pass


def _read_weather_loc(location):
    if r:
        try:
            cached = r.get(f"weather_cache_v2_{location}")
            if cached:
                return _finalize_weather(json.loads(cached))
        except Exception:
            pass
    entry = _local_weather_store.get(location)
    if entry and entry[1] > time.time():
        return entry[0]
    return None


def _assemble_weather_all():
    weather_results = {}
    timestamps = []
    for loc in WEATHER_LOCATIONS:
        data = _read_weather_loc(loc)
        if not data:
            continue
        weather_results[loc] = data
        if data.get("last_updated"):
            timestamps.append(_utc_isoformat(data.get("last_updated")))
    if not weather_results:
        return None
    return {
        "weather": weather_results,
        "last_updated": min(timestamps) if timestamps else None,
    }


def _utc_isoformat(value=_UTC_NOW):
    """Return a UTC ISO-8601 timestamp without fractional seconds."""
    if value is _UTC_NOW:
        dt = datetime.now(timezone.utc)
    elif value is None:
        return None
    elif isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return value
    else:
        return value

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _coerce_weather_numbers(value):
    """Recursively convert weather numeric values to floats for Swift decoding."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        return {k: _coerce_weather_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_weather_numbers(v) for v in value]
    return value

# Canonical dashboard sites. Hardware/settings stay on-device; this API
# serves weather, launches, narratives, and derived dashboard snapshots.
DASHBOARD_LOCATIONS = {
    'Starbase': {
        'lat': 25.9975,
        'lon': -97.1566,
        'timezone': 'America/Chicago',
        'metar_stations': ['KBRO', 'KHRL', 'KMFE'],
        'radar_url': 'https://embed.windy.com/embed2.html?lat=25.7975&lon=-95.1566&zoom=8&level=surface&overlay=radar&menu=&message=true&marker=&calendar=&pressure=&type=map&location=coordinates&detail=&detailLat=25.9975&detailLon=-96.1566&metricWind=mph&metricTemp=%C2%B0F',
    },
    'Vandy': {
        'lat': 34.632,
        'lon': -120.611,
        'timezone': 'America/Los_Angeles',
        'metar_stations': ['KVBG', 'KLPC', 'KSMX'],
        'radar_url': 'https://embed.windy.com/embed2.html?lat=34.432&lon=-118.611&zoom=8&level=surface&overlay=radar&menu=&message=true&marker=&calendar=&pressure=&type=map&location=coordinates&detail=&detailLat=34.632&detailLon=-119.611&metricWind=mph&metricTemp=%C2%B0F',
    },
    'Cape': {
        'lat': 28.392,
        'lon': -80.605,
        'timezone': 'America/New_York',
        'metar_stations': ['KXMR', 'KTTS', 'KCOF', 'KMLB'],
        'radar_url': 'https://embed.windy.com/embed2.html?lat=28.192&lon=-78.605&zoom=8&level=surface&overlay=radar&menu=&message=true&marker=&calendar=&pressure=&type=map&location=coordinates&detail=&detailLat=28.392&detailLon=-79.605&metricWind=mph&metricTemp=%C2%B0F',
    },
    'Hawthorne': {
        'lat': 33.916,
        'lon': -118.352,
        'timezone': 'America/Los_Angeles',
        'metar_stations': ['KHHR', 'KLAX', 'KSMO'],
        'radar_url': 'https://embed.windy.com/embed2.html?lat=33.716&lon=-116.352&zoom=8&level=surface&overlay=radar&menu=&message=true&marker=&calendar=&pressure=&type=map&location=coordinates&detail=&detailLat=33.916&detailLon=-117.352&metricWind=mph&metricTemp=%C2%B0F',
    },
    'Bastrop': {
        'lat': 30.1105,
        'lon': -97.3151,
        'timezone': 'America/Chicago',
        'metar_stations': ['KAUS', 'KEDC', 'KHYI'],
        'radar_url': 'https://embed.windy.com/embed2.html?lat=29.9105&lon=-95.3151&zoom=8&level=surface&overlay=radar&menu=&message=true&marker=&calendar=&pressure=&type=map&location=coordinates&detail=&detailLat=30.1105&detailLon=-96.3151&metricWind=mph&metricTemp=%C2%B0F',
    },
}

METAR_STATIONS = {name: loc['metar_stations'] for name, loc in DASHBOARD_LOCATIONS.items()}
LOCATION_COORDS = {name: {'lat': loc['lat'], 'lon': loc['lon']} for name, loc in DASHBOARD_LOCATIONS.items()}
WEATHER_LOCATIONS = list(DASHBOARD_LOCATIONS.keys())
T_PLUS_ACTIVE_WINDOW_SECONDS = 45 * 60
HISTORY_LIMIT = 43200  # 30 days at 1 minute intervals
SEEDING_STATUS_KEY = "seeding_status_v2"
SEEDING_STOP_SIGNAL_KEY = "seeding_stop_signal_v2"
_last_snapshot_time = 0


# Trajectory Helpers and Cache
class Profiler:
    def mark(self, msg):
        print(f"[PROFILER] {msg}")


class Logger:
    def info(self, msg):
        print(f"[INFO] {msg}")

    def warning(self, msg):
        print(f"[WARNING] {msg}")


profiler = Profiler()
logger = Logger()

_TRAJECTORY_DATA_CACHE = None
TRAJECTORY_CACHE_FILE = "trajectory_cache.json"


def load_cache_from_file(filename):
    """Load cache from Redis instead of file if possible, or fallback to file."""
    if r:
        data = get_cached_data("trajectory_cache_v2")
        if data:
            return {"data": data}

    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                return {"data": json.load(f)}
        except Exception as e:
            print(f"Error loading cache file: {e}")
    return {"data": {}}


def save_cache_to_file(filename, data, updated_at=None):
    """Save cache to Redis and local file."""
    if r:
        set_cached_data("trajectory_cache_v2", data)

    try:
        with open(filename, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error saving cache file: {e}")


# ── Orbital-mechanics physical constants ──────────────────────────────────────
_EARTH_RADIUS_KM = 6371.0          # mean Earth radius, km
_GM_KM3_S2 = 3.986004418e5         # Earth gravitational parameter, km³/s²
_J2 = 1.08263e-3                   # Earth oblateness coefficient
# Use the sidereal year (365.25636 d) — the correct period for J2 RAAN precession
_N_SUN_RAD_S = 2 * math.pi / (365.25636 * 24 * 3600)  # Earth mean motion around Sun, rad/s
_EARTH_SIDEREAL_DAY_S = 86164.1    # sidereal rotation period, s
# Pre-computed orbital circumference at Earth's surface (km); used for landing-range fractions
_EARTH_CIRCUMFERENCE_KM = 2 * math.pi * _EARTH_RADIUS_KM


def _compute_sso_inclination_deg(altitude_km: float) -> float:
    """Return the Sun-synchronous inclination for a circular orbit at altitude_km.

    Derived from the J2-driven RAAN drift condition:
      dΩ/dt = -(3/2) * J2 * (R_E/a)² * n * cos(i) = n_sun
    Solving for i:
      cos(i) = -n_sun / [(3/2) * J2 * (R_E/a)² * n]

    Valid range is roughly 200–2000 km.  Outside this range the cosine saturates
    at ±1 and a warning is logged.
    """
    a_m = (_EARTH_RADIUS_KM + altitude_km) * 1e3          # semi-major axis, m
    R_E_m = _EARTH_RADIUS_KM * 1e3
    GM_m3 = _GM_KM3_S2 * 1e9                               # m³/s²
    n = math.sqrt(GM_m3 / a_m ** 3)                        # mean motion, rad/s
    cos_i_raw = -_N_SUN_RAD_S / (1.5 * _J2 * (R_E_m / a_m) ** 2 * n)
    if not (-1.0 <= cos_i_raw <= 1.0):
        logger.warning(
            f"SSO inclination: altitude {altitude_km} km is outside the valid SSO range "
            f"(cos i = {cos_i_raw:.4f} clamped to [-1, 1])"
        )
    cos_i = max(-1.0, min(1.0, cos_i_raw))
    return math.degrees(math.acos(cos_i))


def compute_orbital_period_min(orbit_label: str) -> float:
    """Return orbital period in minutes using Kepler's third law.

    Altitude assumptions per orbit family:
      Suborbital  – 20 min arc (not a closed orbit)
      LEO         – 400 km  (~92.6 min)
      ISS         – 420 km  (~92.7 min)
      SSO / Polar – 550 km  (~95.6 min)
      MEO (GPS)   – 20 200 km (~717.9 min)
      GTO         – a = (perigee_r + apogee_r)/2, perigee 185 km, apogee 35 786 km (~630.8 min)
      GEO         – 35 786 km (1 436 min, one sidereal day)
    """
    label = (orbit_label or '').lower()
    R_E = _EARTH_RADIUS_KM

    if 'suborbital' in label:
        return 20.0
    if (('geo' in label and 'stationary' in label)
            or 'geosynchronous' in label
            or label.strip() == 'geo'):
        alt_km = 35786.0
    elif 'gto' in label or 'geosynchronous transfer' in label:
        # Semi-major axis of GTO from the average of perigee and apogee radii
        alt_km = ((R_E + 185.0) + (R_E + 35786.0)) / 2.0 - R_E
    elif 'meo' in label or 'medium earth' in label:
        alt_km = 20200.0                   # GPS altitude
    elif 'sso' in label or 'sun-synchronous' in label:
        alt_km = 550.0
    elif 'polar' in label:
        alt_km = 600.0
    elif 'iss' in label or 'space station' in label:
        alt_km = 420.0
    else:
        alt_km = 400.0                     # generic LEO

    a_km = R_E + alt_km
    T_sec = 2.0 * math.pi * math.sqrt(a_km ** 3 / _GM_KM3_S2)
    return T_sec / 60.0


def compute_orbit_radius(orbit_label: str) -> float:
    """Return normalized orbital radius (Earth radii = 1.0 at surface).

    Values are physically derived from (R_E + altitude) / R_E.
    GEO and MEO are capped at visually meaningful display limits so they
    remain on-screen in the globe renderer.
    """
    label = (orbit_label or '').lower()
    R_E = _EARTH_RADIUS_KM

    if (('geo' in label and 'stationary' in label)
            or 'geosynchronous' in label
            or label.strip() == 'geo'):
        return min(2.0, (R_E + 35786.0) / R_E)    # physically ~6.62 → capped
    if 'gto' in label or 'geosynchronous transfer' in label:
        return 1.15                                 # representative apogee height
    if 'meo' in label or 'medium earth' in label:
        return 1.25                                 # GPS (physical 4.17 → scaled)
    if 'sso' in label or 'sun-synchronous' in label:
        return round((R_E + 550.0) / R_E, 4)       # ≈ 1.0863
    if 'polar' in label:
        return round((R_E + 600.0) / R_E, 4)       # ≈ 1.0942
    if 'iss' in label or 'space station' in label:
        return round((R_E + 420.0) / R_E, 4)       # ≈ 1.0659
    if 'suborbital' in label:
        return round((R_E + 100.0) / R_E, 4)       # Kármán line ≈ 1.0157
    # Generic LEO (~400 km, Starlink / Falcon 9 standard shell)
    return round((R_E + 400.0) / R_E, 4)           # ≈ 1.0628


def _ang_dist_deg(p1, p2):
    """Calculate angular distance between two points in degrees."""
    lat1, lon1 = math.radians(p1['lat']), math.radians(p1['lon'])
    lat2, lon2 = math.radians(p2['lat']), math.radians(p2['lon'])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1 - a)))
    return math.degrees(c)


def get_destination_point(lat, lon, distance_km, bearing_deg):
    """Calculate destination point given start point, distance, and bearing."""
    R = 6371.0
    brng = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(math.sin(lat1) * math.cos(distance_km / R) +
                     math.cos(lat1) * math.sin(distance_km / R) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(distance_km / R) * math.cos(lat1),
                             math.cos(distance_km / R) - math.sin(lat1) * math.sin(lat2))
    return {'lat': math.degrees(lat2), 'lon': (math.degrees(lon2) + 180) % 360 - 180}


def generate_ground_track(start_point, inclination_deg, num_points=2000, descending=False, duration_min=90):
    """Generate a ground track accounting for Earth's rotation."""
    lat0 = float(start_point['lat']);
    lon0 = float(start_point['lon'])
    eff_i_deg = max(0.1, min(179.9, abs(inclination_deg)))
    i_rad = math.radians(eff_i_deg)
    lat0_rad = math.radians(lat0);
    lon0_rad = math.radians(lon0)

    x0 = math.cos(lat0_rad) * math.cos(lon0_rad)
    y0 = math.cos(lat0_rad) * math.sin(lon0_rad)
    z0 = math.sin(lat0_rad)

    sin_i = math.sin(i_rad)
    u0 = math.asin(max(-1.0, min(1.0, z0 / (sin_i or 1e-6))))
    if descending:
        u0 = math.pi - u0

    Omega = math.atan2(y0, x0) - math.atan2(math.sin(u0) * math.cos(i_rad), math.cos(u0))

    cosO = math.cos(Omega);
    sinO = math.sin(Omega)
    cosi = math.cos(i_rad);
    sili = math.sin(i_rad)

    points = []
    omega_e = 2 * math.pi / _EARTH_SIDEREAL_DAY_S   # sidereal rotation rate, rad/s
    period_sec = duration_min * 60

    for k in range(num_points):
        t_frac = k / (num_points - 1)
        t_sec = t_frac * period_sec
        u = u0 + (2.0 * math.pi * t_frac)

        cu = math.cos(u);
        su = math.sin(u)
        xo = cu
        yo = su * cosi
        zo = su * sili

        x_i = cosO * xo - sinO * yo
        y_i = sinO * xo + cosO * yo
        z_i = zo

        lon_i = math.atan2(y_i, x_i)
        lon_fixed = lon_i - omega_e * t_sec

        lat = math.degrees(math.atan2(z_i, math.hypot(x_i, y_i)))
        lon = (math.degrees(lon_fixed) + 180) % 360 - 180

        points.append({'lat': lat, 'lon': lon})
    return points


# Surveyed Launch Library coordinates for Starbase OLP-2 (Boca Chica).
# Unknown pad names must not fall through to LC-39A.
STARBASE_LAUNCH_SITE = {'lat': 25.99677, 'lon': -97.15799, 'name': 'Starbase, TX'}
_LC39A_LAUNCH_SITE = {'lat': 28.6084, 'lon': -80.6043, 'name': 'Cape Canaveral, FL'}
_LC40_LAUNCH_SITE = {'lat': 28.5619, 'lon': -80.5773, 'name': 'Cape Canaveral, FL'}
_SLC4E_LAUNCH_SITE = {'lat': 34.6321, 'lon': -120.6107, 'name': 'Vandenberg, CA'}

# Case-insensitive substrings. Starbase aliases are first so OLP-2 / Boca Chica
# never match a later Cape key. Longer phrases sit beside the shorter prefixes.
_PAD_SITE_ALIASES = (
    (
        (
            'orbital launch pad 2',
            'orbital launch pad',
            'orbital launch mount',
            'olp-2',
            'olp 2',
            'olp-',
            'olp ',
            'olm-2',
            'olm 2',
            'olm-',
            'olm ',
            'boca chica',
            'starbase',
        ),
        STARBASE_LAUNCH_SITE,
    ),
    (
        ('launch complex 39a', 'lc-39a', 'pad 39a'),
        _LC39A_LAUNCH_SITE,
    ),
    (
        ('launch complex 40', 'slc-40', 'lc-40'),
        _LC40_LAUNCH_SITE,
    ),
    (
        ('space launch complex 4e', 'launch complex 4e', 'slc-4e'),
        _SLC4E_LAUNCH_SITE,
    ),
)

_SITE_MISMATCH_DEG = 0.2


def _finite_coord(value):
    """Parse a pad coordinate from LL (float or string). Invalid values are None."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _usable_pad_coords(lat, lon):
    if lat is None or lon is None:
        return False
    if abs(lat) > 90 or abs(lon) > 180:
        return False
    # Launch Library uses 0,0 when a pad has not been surveyed.
    if abs(lat) < 1e-6 and abs(lon) < 1e-6:
        return False
    return True


def _name_from_coords(lat, lon):
    """Canonical site name when coordinates fall on a known SpaceX complex."""
    if 25.8 <= lat <= 26.3 and -97.5 <= lon <= -96.8:
        return STARBASE_LAUNCH_SITE['name']
    if 28.3 <= lat <= 28.8 and -80.9 <= lon <= -80.4:
        return _LC39A_LAUNCH_SITE['name']
    if 34.4 <= lat <= 34.9 and -120.9 <= lon <= -120.4:
        return _SLC4E_LAUNCH_SITE['name']
    return None


def _match_known_pad(text):
    lowered = (text or '').lower()
    if not lowered.strip():
        return None
    for aliases, site in _PAD_SITE_ALIASES:
        if any(alias in lowered for alias in aliases):
            return site
    return None


def _site_cache_key(site):
    return f"{site['name']}:{float(site['lat']):.5f}:{float(site['lon']):.5f}"


def resolve_launch_site(pad, latitude=None, longitude=None, location_name=None):
    """Resolve a launch site from a pad name and optional upstream coordinates.

    Real pad latitude/longitude win over hardcoded defaults. OLP-2,
    "Orbital Launch Pad 2", and Boca Chica aliases map to Starbase, TX.
    An unknown pad is left unresolved instead of being treated as LC-39A.
    """
    pad_name = ''
    if isinstance(pad, dict):
        pad_name = pad.get('name') or ''
        if latitude is None:
            latitude = pad.get('latitude')
        if longitude is None:
            longitude = pad.get('longitude')
        loc = pad.get('location')
        if location_name is None and isinstance(loc, dict):
            location_name = loc.get('name')
    elif pad:
        pad_name = str(pad)

    known = _match_known_pad(f"{pad_name} {location_name or ''}")
    lat = _finite_coord(latitude)
    lon = _finite_coord(longitude)
    if _usable_pad_coords(lat, lon):
        prox_name = _name_from_coords(lat, lon)
        if known and (prox_name is None or prox_name == known['name']):
            name = known['name']
        elif prox_name:
            name = prox_name
        elif location_name:
            name = location_name
        elif known:
            name = known['name']
        else:
            name = 'Unknown'
        site = {'lat': lat, 'lon': lon, 'name': name}
        return site, _site_cache_key(site)

    if known:
        site = {'lat': known['lat'], 'lon': known['lon'], 'name': known['name']}
        return site, _site_cache_key(site)

    logger.info(f"Unknown pad {pad_name!r}; not defaulting to LC-39A")
    return None, None


def _pad_fields_from_launch(launch):
    """Pad name plus coordinates, preferring fields kept on the slim launch."""
    if not isinstance(launch, dict):
        return '', None, None, None
    pad = launch.get('pad') or ''
    if isinstance(pad, dict):
        pad_name = pad.get('name') or ''
        latitude = pad.get('latitude')
        longitude = pad.get('longitude')
        loc = pad.get('location')
        location_name = loc.get('name') if isinstance(loc, dict) else None
    else:
        pad_name = str(pad) if pad else ''
        latitude = launch.get('pad_latitude')
        longitude = launch.get('pad_longitude')
        location_name = launch.get('pad_location')
    raw = launch.get('all_data')
    raw_pad = raw.get('pad') if isinstance(raw, dict) else None
    if isinstance(raw_pad, dict):
        if not pad_name:
            pad_name = raw_pad.get('name') or ''
        if latitude is None:
            latitude = raw_pad.get('latitude')
        if longitude is None:
            longitude = raw_pad.get('longitude')
        if not location_name and isinstance(raw_pad.get('location'), dict):
            location_name = raw_pad['location'].get('name')
    return pad_name, latitude, longitude, location_name


def _coords_far(point, expected):
    if not isinstance(point, dict) or not isinstance(expected, dict):
        return False
    try:
        dlat = abs(float(point.get('lat')) - float(expected.get('lat')))
        dlon = abs(float(point.get('lon')) - float(expected.get('lon')))
    except (TypeError, ValueError):
        return False
    return dlat > _SITE_MISMATCH_DEG or dlon > _SITE_MISMATCH_DEG


def get_launch_trajectory_data(upcoming_launches, previous_launches=None):
    """
    Get trajectory data for the next upcoming launch or a specific launch.
    If upcoming_launches is a dict, treat it as a single launch object.
    If upcoming_launches is a list, use the first item (existing behavior).
    Standalone version of Backend.get_launch_trajectory.
    """
    profiler.mark("get_launch_trajectory_data Start")
    logger.info("get_launch_trajectory_data called")

    # Handle single launch object (dict) vs list of launches
    if isinstance(upcoming_launches, dict):
        # Single launch object
        display_launches = [upcoming_launches]
    else:
        # List of launches (existing behavior)
        display_launches = upcoming_launches
        if not display_launches:
            logger.info("No upcoming launches, trying recent launches")
            if previous_launches:
                recent_launches = previous_launches[:5]
                if recent_launches:
                    display_launches = [{
                        'mission': launch.get('mission', 'Unknown'),
                        'pad': launch.get('pad', 'Cape Canaveral'),
                        'orbit': launch.get('orbit', 'LEO'),
                        'net': launch.get('net', ''),
                        'landing_type': launch.get('landing_type'),
                        'landing_location': launch.get('landing_location')
                    } for launch in recent_launches]
                    logger.info(f"Using {len(display_launches)} recent launches for demo")

    if not display_launches:
        logger.info("No launches available at all")
        return None

    next_launch = display_launches[0]
    mission_name = next_launch.get('mission', 'Unknown')
    pad, pad_latitude, pad_longitude, pad_location = _pad_fields_from_launch(next_launch)
    orbit = next_launch.get('orbit', '')
    logger.info(f"Next launch: {mission_name} from {pad}")

    # Upstream LL pad coordinates win. OLP-2 / Boca Chica map to Starbase.
    # Unknown pads are not assigned LC-39A.
    launch_site, matched_site_key = resolve_launch_site(
        pad,
        latitude=pad_latitude,
        longitude=pad_longitude,
        location_name=pad_location,
    )
    if not launch_site:
        return None

    def _normalize_orbit(orbit_label: str, site_name: str) -> str:
        try:
            label = (orbit_label or '').lower()
            # GEO / GTO — evaluate GEO before the shorter 'geo' substring check
            if (('geo' in label and 'stationary' in label)
                    or 'geosynchronous' in label
                    or label.strip() == 'geo'):
                return 'GEO'
            if 'gto' in label or 'geosynchronous transfer' in label:
                return 'GTO'
            if 'suborbital' in label:
                return 'Suborbital'
            if 'meo' in label or 'medium earth' in label:
                return 'MEO'
            # SSO and polar before generic LEO
            if 'sso' in label or 'sun-synchronous' in label:
                return 'SSO'
            if 'polar' in label:
                return 'Polar'
            if 'iss' in label or 'space station' in label:
                return 'ISS'
            if 'leo' in label or 'low earth orbit' in label:
                # Vandenberg LEO launches are polar/SSO by convention
                if 'Vandenberg' in site_name:
                    return 'SSO'
                return 'LEO'
            # Vandenberg catch-all → SSO
            if 'Vandenberg' in site_name:
                return 'SSO'
            return 'LEO'
        except Exception:
            return 'LEO'

    normalized_orbit = _normalize_orbit(orbit, launch_site.get('name', ''))

    # Resolve an inclination assumption
    def _resolve_inclination_deg(norm_orbit: str, site_name: str, site_lat: float) -> float:
        """Return best-estimate orbital inclination in degrees.

        Priority order:
        1. Mission-name keyword overrides (ISS, Crew/Cargo Dragon, Starlink shell hints)
        2. Orbit-type formulas (SSO via J2, GPS/MEO standard, GTO ≈ site latitude)
        3. Generic LEO fallback clamped to a physically achievable range
        """
        try:
            label = (orbit or '').lower()

            # ── Mission-specific overrides ────────────────────────────────────
            if 'iss' in label or 'crew dragon' in label or 'cargo dragon' in label:
                return 51.6   # ISS inclination

            # Starlink: shell depends on launch site and payload name hints
            if 'starlink' in label:
                if 'Vandenberg' in site_name:
                    return _compute_sso_inclination_deg(550.0)  # ~97.6° polar shell
                if 'polar' in label or '70' in label:
                    return 70.0   # high-inclination Starlink shell (~70°)
                return 53.0       # most common Starlink shell from KSC

            # ── Orbit-type formulas ───────────────────────────────────────────
            if norm_orbit in ('SSO', 'Polar') or 'sso' in label or 'sun-synchronous' in label:
                # SSO inclination is altitude-dependent (J2 perturbation theory)
                # Use 550 km for SSO, 600 km for generic polar
                alt = 600.0 if norm_orbit == 'Polar' else 550.0
                return _compute_sso_inclination_deg(alt)

            if norm_orbit == 'MEO':
                return 55.0   # GPS / MEO standard inclination

            if norm_orbit == 'ISS':
                return 51.6

            if norm_orbit == 'GEO':
                # Direct-to-GEO inclination target is 0°; GTO transfers aim near 0°
                return 0.0

            if norm_orbit == 'GTO':
                # Minimum-energy GTO: launch due east → inclination ≈ site latitude.
                # Subtract a small correction (~0.5°) for typical azimuth biasing.
                return max(18.0, min(35.0, abs(site_lat) - 0.5))

            if norm_orbit == 'Suborbital':
                return max(10.0, min(45.0, abs(site_lat)))

            # ── Generic LEO ───────────────────────────────────────────────────
            # Minimum inclination = |site_lat|; add small gravity-turn offset
            base = abs(site_lat)
            return max(28.0, min(60.0, base + 0.5))

        except Exception:
            pass
        return 30.0

    assumed_incl = _resolve_inclination_deg(
        normalized_orbit,
        launch_site.get('name', ''),
        launch_site.get('lat', 0.0)
    )

    ORBIT_CACHE_VERSION = 'v261-pad-coords'
    landing_type = next_launch.get('landing_type')
    landing_loc = next_launch.get('landing_location')
    cache_key = f"{ORBIT_CACHE_VERSION}:{matched_site_key}:{normalized_orbit}:{round(assumed_incl, 1)}:{landing_type}:{landing_loc}"

    global _TRAJECTORY_DATA_CACHE
    if _TRAJECTORY_DATA_CACHE is None:
        logger.info("Loading trajectory cache from disk...")
        cache_loaded = load_cache_from_file(TRAJECTORY_CACHE_FILE)
        if cache_loaded and isinstance(cache_loaded.get('data'), dict):
            _TRAJECTORY_DATA_CACHE = cache_loaded['data']
        else:
            # Handle direct dictionary without 'data' wrapper if it exists
            _TRAJECTORY_DATA_CACHE = cache_loaded if isinstance(cache_loaded, dict) else {}

    if cache_key in _TRAJECTORY_DATA_CACHE:
        cached = _TRAJECTORY_DATA_CACHE[cache_key]
        logger.info(f"Trajectory in-memory cache hit for {cache_key}")
        return {
            'launch_site': cached.get('launch_site', launch_site),
            'trajectory': cached.get('trajectory', []),
            'booster_trajectory': cached.get('booster_trajectory', []),
            'sep_idx': cached.get('sep_idx'),
            'orbit_path': cached.get('orbit_path', []),
            'orbit': orbit or cached.get('orbit', normalized_orbit),
            'mission': mission_name,
            'pad': pad,
            'landing_type': landing_type,
            'landing_location': cached.get('landing_location', next_launch.get('landing_location'))
        }

    traj_cache = _TRAJECTORY_DATA_CACHE
    logger.info(f"Trajectory cache miss for {cache_key}; generating new trajectory")

    def generate_curved_trajectory(start_point, end_point, num_points, orbit_type='default', end_bearing_deg=None):
        points = []
        start_lat = start_point['lat']
        start_lon = start_point['lon']
        end_lat = end_point['lat']
        end_lon = end_point['lon']

        if end_bearing_deg is not None:
            lat1 = math.radians(start_lat);
            lon1 = math.radians(start_lon)
            lat2 = math.radians(end_lat);
            lon2 = math.radians(end_lon)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1 - a)))
            ang_deg = math.degrees(c)
            # Tighter control point distance (L) to avoid "flat" segments near insertion
            L = min(15.0, max(3.0, ang_deg / 4.0))
            br = math.radians(end_bearing_deg)
            cos_lat = max(1e-6, math.cos(math.radians(end_lat)))
            dlat_deg = L * math.cos(br)
            dlon_deg = (L * math.sin(br)) / cos_lat
            control_lat = end_lat - dlat_deg
            control_lon = (end_lon - dlon_deg + 180.0) % 360.0 - 180.0
        else:
            mid_lat = (start_lat + end_lat) / 2
            mid_lon = (start_lon + end_lon) / 2
            dist = _ang_dist_deg(start_point, end_point)

            # Scale control point offset based on distance
            offset = max(5, min(30, dist * 0.4))

            if orbit_type == 'polar':
                control_lat = max(-85.0, mid_lat - offset)
                control_lon = mid_lon - offset / 2
            elif orbit_type == 'equatorial':
                # Aim more towards equator
                target_equator_lat = 0
                control_lat = (mid_lat + target_equator_lat) / 2
                control_lon = mid_lon + offset
            elif orbit_type == 'gto':
                control_lat = mid_lat + offset
                control_lon = mid_lon + offset * 2
            elif orbit_type == 'suborbital':
                # Suborbital/Booster return needs a tighter arc
                control_lat = mid_lat + offset / 4
                control_lon = mid_lon + offset / 4
            else:
                control_lat = mid_lat + offset
                control_lon = mid_lon + offset * 1.5

        for i in range(num_points + 1):
            t = i / num_points
            lat = (1 - t) ** 2 * start_lat + 2 * (1 - t) * t * control_lat + t ** 2 * end_lat
            lon = (1 - t) ** 2 * start_lon + 2 * (1 - t) * t * control_lon + t ** 2 * end_lon
            lon = (lon + 180) % 360 - 180
            points.append({'lat': lat, 'lon': lon})
        return points

    # Main generation
    target_r = compute_orbit_radius(orbit)

    # Polar and SSO orbits from VAFB launch southward (descending pass); all
    # other sites launch eastward on the ascending pass.
    is_desc = normalized_orbit in ('SSO', 'Polar') and (
        'Vandenberg' in launch_site.get('name', '') or 'SLC-4E' in pad
    )

    # Orbital period drives both the ground-track duration and ascent fractions.
    orbital_period_min = compute_orbital_period_min(orbit or normalized_orbit)

    # Generate the Master Path (full one-orbit ground track).
    # Using the true sidereal orbital period gives accurate Earth-rotation westward
    # drift between successive ground-track passes.
    master_path = generate_ground_track(
        launch_site, assumed_incl,
        num_points=2000, descending=is_desc,
        duration_min=orbital_period_min
    )

    # ── Ascent trajectory fraction ────────────────────────────────────────────
    # Falcon 9 / Starship reach orbit insertion ~9 min after lift-off for LEO,
    # ~9 min for GTO (SECO), and slightly longer for SSO/polar missions.
    # Suborbital missions reach apogee in ~5 min.
    # Values are representative of historical Falcon 9 telemetry.
    ASCENT_TIME_MIN = {
        'GTO': 9.0, 'GEO': 9.0,
        'SSO': 9.5, 'Polar': 9.5,
        'LEO': 9.0, 'ISS': 9.0,
        'MEO': 10.0,
        'Suborbital': 5.0,
    }.get(normalized_orbit, 9.0)

    # Fraction of master_path that represents the ascent phase.
    # 20% upper cap prevents ascent from visually overlapping the orbit ring for
    # very short periods; 2.5% lower floor ensures a minimum visible ascent path
    # even for long-period orbits (GTO, GEO) where insertion is a tiny fraction.
    traj_frac = min(0.20, max(0.025, ASCENT_TIME_MIN / orbital_period_min))
    traj_len = max(50, int(len(master_path) * traj_frac))
    trajectory = [p.copy() for p in master_path[:traj_len]]

    # The orbit path is the REMAINING part of the Master Path to avoid overlap
    if normalized_orbit != 'Suborbital':
        orbit_path = [p.copy() for p in master_path[traj_len - 1:]]
    else:
        orbit_path = []

    # Set radii for the orbit path
    for p in orbit_path:
        p['r'] = target_r

    # Add radii to main trajectory (ascent altitude profile).
    # Exponent 0.4 approximates a real gravity-turn: very steep initial vertical
    # climb (first ~10 s) then a flattening pitch-over toward horizontal.
    # Lower values (< 0.5) give a sharper initial knee, higher values are more
    # gradual; 0.4 best matches Falcon 9 altitude-vs-time telemetry.
    for i, p in enumerate(trajectory):
        progress = i / max(1, len(trajectory) - 1)
        p['r'] = 1.0 + (target_r - 1.0) * (progress ** 0.4)

    # ── Booster Return Trajectory ─────────────────────────────────────────────
    # MECO (stage separation) for Falcon 9 occurs at T+~2:30 regardless of
    # mission type.  Compute the corresponding index into the ascent trajectory.
    MECO_TIME_MIN = 2.5
    sep_frac = MECO_TIME_MIN / max(0.1, ASCENT_TIME_MIN)
    booster_trajectory = []
    sep_idx = None
    if trajectory and len(trajectory) > 10:
        try:
            sep_idx = max(4, min(int(len(trajectory) * sep_frac), len(trajectory) - 10))

            sep_point = trajectory[sep_idx].copy()
            sep_radius = sep_point['r']

            l_type = (landing_type or '').upper()
            l_loc = (next_launch.get('landing_location') or '').upper()
            combined_landing_info = f"{l_type} {l_loc}"

            # Expanded detection for ASDS and RTLS
            asds_keywords = ['ASDS', 'DRONE', 'SHIP', 'OCISLY', 'JRTI', 'ASOG', 'GRAVITAS', 'INSTRUCTIONS',
                             'STILL LOVE YOU']
            rtls_keywords = ['RTLS', 'LAUNCH SITE', 'CATCH', 'TOWER', 'LZ', 'LANDING ZONE']

            if any(k in combined_landing_info for k in asds_keywords):
                # ASDS droneship: Falcon 9 lands ~650 km downrange for LEO,
                # ~690 km for GTO (longer coast after higher-energy MECO).
                # Express as fraction of the orbital circumference (≈ 40 030 km).
                asds_km = 690.0 if normalized_orbit == 'GTO' else 650.0
                dist_frac = asds_km / _EARTH_CIRCUMFERENCE_KM
                landing_idx = min(len(master_path) - 1, int(len(master_path) * dist_frac))
                landing_point = master_path[landing_idx]

                return_part = generate_curved_trajectory(sep_point, landing_point, 100, orbit_type='suborbital')

                # Altitude profile: parabolic arc peaking at ~80 km above sep altitude
                # (ASDS peak ≈ 80 km; RTLS peak ≈ 150 km)
                peak_dr = (80.0 / _EARTH_RADIUS_KM)
                for i, p in enumerate(return_part):
                    prog = i / max(1, len(return_part) - 1)
                    p['r'] = (sep_radius + (1.0 - sep_radius) * prog
                              + peak_dr * math.sin(prog * math.pi))

                booster_trajectory = return_part
                sep_idx = 0  # Indicates start of booster_trajectory in visual tools
                logger.info(f"Generated accurate ASDS booster trajectory (~{asds_km:.0f} km)")
            elif any(k in combined_landing_info for k in rtls_keywords):
                # RTLS / Mechazilla catch: booster returns to launch site with a
                # higher boostback arc (~150 km peak above sep altitude).
                return_part = generate_curved_trajectory(sep_point, launch_site, 100, orbit_type='suborbital')
                peak_dr = (150.0 / _EARTH_RADIUS_KM)
                for i, p in enumerate(return_part):
                    prog = i / max(1, len(return_part) - 1)
                    p['r'] = (sep_radius + (1.0 - sep_radius) * prog
                              + peak_dr * math.sin(prog * math.pi))

                booster_trajectory = return_part
                sep_idx = 0
                logger.info(f"Generated accurate RTLS booster trajectory")
            elif any(k in combined_landing_info for k in ['OCEAN', 'SPLASHDOWN']):
                # Expendable ocean splashdown: ~400 km downrange (between RTLS and ASDS)
                ocean_km = 400.0
                dist_frac = ocean_km / _EARTH_CIRCUMFERENCE_KM
                landing_idx = min(len(master_path) - 1, int(len(master_path) * dist_frac))
                landing_point = master_path[landing_idx]
                return_part = generate_curved_trajectory(sep_point, landing_point, 100, orbit_type='suborbital')
                peak_dr = (60.0 / _EARTH_RADIUS_KM)
                for i, p in enumerate(return_part):
                    prog = i / max(1, len(return_part) - 1)
                    p['r'] = (sep_radius + (1.0 - sep_radius) * prog
                              + peak_dr * math.sin(prog * math.pi))
                booster_trajectory = return_part
                sep_idx = 0
                logger.info(f"Generated Ocean splashdown booster trajectory")
            else:
                booster_trajectory = []
                sep_idx = None
                logger.info(f"Skipping booster trajectory for unknown/expendable type")
        except Exception as e:
            logger.warning(f"Booster trajectory generation failed: {e}")

    result = {
        'launch_site': launch_site,
        'trajectory': trajectory,
        'booster_trajectory': booster_trajectory,
        'sep_idx': sep_idx,
        'orbit_path': orbit_path,
        'orbit': orbit,
        'mission': mission_name,
        'pad': pad,
        'landing_type': landing_type,
        'landing_location': next_launch.get('landing_location')
    }

    # Persist to cache
    try:
        traj_cache[cache_key] = {
            'launch_site': launch_site,
            'trajectory': trajectory,
            'booster_trajectory': booster_trajectory,
            'sep_idx': sep_idx,
            'orbit_path': orbit_path,
            'orbit': normalized_orbit,
            'inclination_deg': assumed_incl,
            'landing_type': landing_type,
            'landing_location': next_launch.get('landing_location'),
            'model': 'v13-orbital-mechanics'
        }
        save_cache_to_file(TRAJECTORY_CACHE_FILE, traj_cache, datetime.now(pytz.utc))
    except Exception as e:
        logger.warning(f"Failed to save trajectory cache: {e}")

    return result


_local_seeding_status = {"is_running": False, "last_status": "Idle", "total_pulled": 0, "oldest_launch": None}
_stop_seeding_requested = False


def update_seeding_status(is_running, last_status, total_pulled=0, oldest_launch=None):
    """Update the seeding status in Redis and memory."""
    global _local_seeding_status
    status = {
        "is_running": is_running,
        "last_status": last_status,
        "total_pulled": total_pulled,
        "oldest_launch": oldest_launch,
        "updated_at": _utc_isoformat()
    }
    _local_seeding_status = status
    if r:
        try:
            r.set(SEEDING_STATUS_KEY, json.dumps(status))
        except Exception as e:
            print(f"Redis error in update_seeding_status: {e}")
    return status


def reset_stuck_seeding():
    """Reset seeding status if it's marked as running but the app just started."""
    if r:
        try:
            data = r.get(SEEDING_STATUS_KEY)
            if data:
                status = json.loads(data)
                if status.get("is_running"):
                    print("Detected stuck seeding status at startup. Resetting.")
                    update_seeding_status(False, "Interrupted (App Restart)", status.get("total_pulled", 0),
                                          status.get("oldest_launch"))
        except Exception as e:
            print(f"Error resetting stuck seeding: {e}")


def increment_metric(field):
    """Increment a metric in Redis or memory."""
    if r:
        try:
            r.hincrby(METRICS_KEY, field, 1)
            return
        except Exception as e:
            print(f"Redis error in increment_metric: {e}")

    # Fallback to in-memory
    if field in _local_metrics:
        _local_metrics[field] += 1


def record_snapshot():
    """Record a snapshot of current metrics for historical tracking."""
    global _last_snapshot_time
    now = datetime.now(timezone.utc).timestamp()
    if now - _last_snapshot_time < 55:  # Throttle to ~1m
        return
    _last_snapshot_time = now

    current_metrics = get_metrics(include_history=False)
    snapshot = {
        "timestamp": _utc_isoformat(),
        "data": current_metrics
    }

    if r:
        try:
            r.lpush(METRICS_HISTORY_KEY, json.dumps(snapshot))
            r.ltrim(METRICS_HISTORY_KEY, 0, HISTORY_LIMIT - 1)
        except Exception as e:
            print(f"Redis error in record_snapshot: {e}")

    _local_metrics_history.append(snapshot)
    if len(_local_metrics_history) > HISTORY_LIMIT:
        _local_metrics_history.pop(0)


@app.post("/reset_metrics")
def reset_app_metrics():
    """Endpoint to reset all metrics and history."""
    global _local_metrics, _local_metrics_history, _last_snapshot_time

    # Reset in-memory
    _local_metrics = {
        "total_requests": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "api_calls": 0
    }
    _local_metrics_history = []
    _last_snapshot_time = 0

    # Reset Redis
    if r:
        try:
            r.delete(METRICS_KEY)
            r.delete(METRICS_HISTORY_KEY)
        except Exception as e:
            print(f"Redis error in reset_app_metrics: {e}")
            return {"status": "Error", "message": str(e)}

    return {"status": "Success", "message": "All metrics and history have been reset."}


def get_metrics(include_history=True, range_type="1h"):
    """Retrieve metrics from Redis or memory."""
    current = {
        "total_requests": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "api_calls": 0
    }

    if r:
        try:
            data = r.hgetall(METRICS_KEY)
            if data:
                current = {k: int(v) for k, v in data.items()}
        except Exception as e:
            print(f"Redis error in get_metrics: {e}")
    else:
        current = _local_metrics.copy()

    if not include_history:
        return current

    history = []
    if r:
        try:
            history_data = r.lrange(METRICS_HISTORY_KEY, 0, -1)
            history = [json.loads(s) for s in history_data]
            history.reverse()  # Oldest first for charting
        except Exception as e:
            print(f"Redis error fetching history: {e}")
    else:
        history = _local_metrics_history.copy()

    # Filter by range
    now = datetime.now(timezone.utc)
    if range_type == "1h":
        start_time = now - timedelta(hours=1)
        history = [h for h in history if datetime.fromisoformat(h['timestamp'].replace('Z', '+00:00')) > start_time]
    elif range_type == "24h":
        start_time = now - timedelta(hours=24)
        history = [h for h in history if datetime.fromisoformat(h['timestamp'].replace('Z', '+00:00')) > start_time]
    elif range_type == "7d":
        start_time = now - timedelta(days=7)
        history = [h for h in history if datetime.fromisoformat(h['timestamp'].replace('Z', '+00:00')) > start_time]
    elif range_type == "30d":
        start_time = now - timedelta(days=30)
        history = [h for h in history if datetime.fromisoformat(h['timestamp'].replace('Z', '+00:00')) > start_time]

    # Calculate range-relative current stats if we have history
    range_stats = {
        "total_requests": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "api_calls": 0
    }
    if history:
        first = history[0]['data']
        last = history[-1]['data']
        range_stats = {
            "total_requests": last.get('total_requests', 0) - first.get('total_requests', 0),
            "cache_hits": last.get('cache_hits', 0) - first.get('cache_hits', 0),
            "cache_misses": last.get('cache_misses', 0) - first.get('cache_misses', 0),
            "api_calls": last.get('api_calls', 0) - first.get('api_calls', 0)
        }

    # Calculate live hits/day BEFORE downsampling for better accuracy
    hits_per_day = 0
    if len(history) > 1:
        first_h = history[0]
        last_h = history[-1]
        try:
            t1 = datetime.fromisoformat(first_h['timestamp'].replace('Z', '+00:00'))
            t2 = datetime.fromisoformat(last_h['timestamp'].replace('Z', '+00:00'))
            duration_hours = (t2 - t1).total_seconds() / 3600
            if duration_hours > 0.1:  # At least 6 mins of data
                hits_diff = last_h['data']['total_requests'] - first_h['data']['total_requests']
                hits_per_day = (hits_diff / duration_hours) * 24
        except:
            pass

    # Downsample history for charting
    if range_type == "24h" and len(history) > 100:
        history = history[::15]  # ~15m intervals
    elif range_type == "7d" and len(history) > 200:
        history = history[::60]  # ~1h intervals
    elif range_type == "30d" and len(history) > 300:
        history = history[::240]  # ~4h intervals

    return {
        "current": current,
        "range_stats": range_stats,
        "history": history,
        "hits_per_day": round(hits_per_day, 1)
    }


def generate_narratives(existing_narratives=None):
    """Fetch launches and generate narratives using Grok, appending new ones only."""
    increment_metric("api_calls")
    current_time = datetime.now(timezone.utc)
    three_months_ago = current_time - timedelta(days=90)
    # Use v2.3.0
    url = (
        f"https://ll.thespacedevs.com/2.3.0/launches/previous/"
        f"?lsp__name=SpaceX"
        f"&net__gte={three_months_ago.strftime('%Y-%m-%d')}"
        f"&net__lte={current_time.strftime('%Y-%m-%d')}"
        f"&limit=5"
        f"&ordering=-net"
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch launches: {response.status_code}")

    data = response.json().get('results', [])

    launches = []
    for launch in data:
        net_str = launch['net']
        # Handle cases where Z might be missing or other ISO formats
        try:
            net_dt = datetime.fromisoformat(net_str.replace('Z', '+00:00'))
        except ValueError:
            # Fallback if the format is slightly different
            net_dt = datetime.strptime(net_str, "%Y-%m-%dT%H:%M:%SZ")

        date_time = net_dt.strftime("%m/%d %H%M")

        mission = launch['name']
        pad = launch['pad']['name']
        rocket = launch['rocket']['configuration']['name']
        orbit = launch.get('mission', {}).get('orbit', {}).get('name', 'Unknown')
        status = launch['status']['name']

        launches.append({
            "date_time": date_time,
            "mission": mission,
            "pad": pad,
            "rocket": rocket,
            "orbit": orbit,
            "status": status
        })

    if not launches:
        return existing_narratives if existing_narratives else []

    # Identify new launches not in the current cache
    existing_keys = set()
    if existing_narratives:
        for narr in existing_narratives:
            # Extract "MM/DD HHMM" from the start of the narrative
            parts = narr.split(': ', 1)
            if parts:
                existing_keys.add(parts[0])

    new_launches = [l for l in launches if l['date_time'] not in existing_keys]

    if existing_narratives and not new_launches:
        print("No new launches found. Cache is up to date.")
        return existing_narratives

    # If we have existing narratives, only process the new ones to append
    # If no cache exists, process all fetched launches
    launches_to_process = new_launches if existing_narratives else launches

    launch_list = "\n".join([
        f"{l['date_time']}: {l['mission']} from {l['pad']}, {l['rocket']} to {l['orbit']}, status {l['status']}"
        for l in launches_to_process
    ])

    prompt = f"""Generate a list of short news like descriptions for these SpaceX launches:
{launch_list}

In the style of Cities Skylines notifications: kind of witty and dry. Factual, complete, somewhat technical - think Kerbal Space Program.

IMPORTANT: Return ONLY a Python list assignment and nothing else. Be extremely concise for each entry to avoid truncation. No conversational filler, no introductory text, no markdown formatting.

Examples:
- Falcon 9 hoists MTG-S1/Sentinel-4A to geosync from LC-39A; Ariane's loss is our nominal gain, booster recovered without drama.
- 500th Falcon 9 ignites with 27 Starlinks from SLC-40; B1067 clocks 29th flight, orbit insertion as predictable as gravity.
- Starship Flight 10 ignites from Starbase; hot-staging clean, ship splashes precisely in Indian Ocean, Super Heavy boosts back nominally.

Format each as: month/day HHMM: description

Output as a Python list assignment: launch_descriptions = [...]"""

    try:
        generated_text = call_grok(prompt, temperature=0.7, max_tokens=4000)
    except Exception as e:
        raise ValueError(f"Grok API call failed: {str(e)}")

    try:
        # Robust extraction: find all strings that match the pattern "month/day HHMM: description"
        new_descriptions = re.findall(r'["\'](\d{1,2}/\d{1,2} \d{4}: .*?)["\']', generated_text)

        if not new_descriptions:
            # Fallback for alternative formatting
            start_idx = generated_text.find('[')
            end_idx = generated_text.rfind(']') + 1
            if start_idx != -1 and end_idx > start_idx:
                try:
                    list_str = generated_text[start_idx:end_idx]
                    new_descriptions = ast.literal_eval(list_str)
                except:
                    pass

        if not new_descriptions:
            raise ValueError("No valid launch descriptions found in response")

        if not isinstance(new_descriptions, list) or not all(isinstance(d, str) for d in new_descriptions):
            raise ValueError("Parsed content is not a list of strings")

        print(f"Successfully generated {len(new_descriptions)} new narratives.")

        if existing_narratives:
            # Prepend new ones while preserving the full existing history.
            combined = new_descriptions + existing_narratives
            return combined

        return new_descriptions
    except Exception as e:
        raise ValueError(f"Failed to parse Grok response: {str(e)}")


def call_grok(prompt, temperature=0.7, max_tokens=4000):
    """Shared xAI Grok chat-completions path used by narratives and notify copy."""
    headers = {
        "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(GROK_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise ValueError(f"Grok API call failed: {response.status_code} - {response.text}")

    data = response.json()
    generated_text = data["choices"][0]["message"]["content"]
    print(f"DEBUG: Raw Grok response: {generated_text}")
    return generated_text


class NotifyCopyRequest(BaseModel):
    """Launch Buddy upcoming-launch alert input for Grok notification copy."""

    launch_id: str = Field(..., examples=["a7e1c2d4-1111-4b2a-9c33-0f1e2d3c4b5a"], description="Launch Library / Launch Buddy launch id")
    event: NotifyEvent = Field(
        ...,
        description="Alert type: T-24 hours, T-1 hour, or scrub/delay/hold",
        examples=["t1h"],
    )
    mission: Optional[str] = Field(default=None, examples=["Starlink Group 10-20"])
    net: Optional[str] = Field(default=None, examples=["2026-09-06T02:15:00Z"], description="Launch NET as ISO8601")
    status: Optional[str] = Field(default=None, examples=["Go"], description="Current launch status")
    pad: Optional[str] = Field(default=None, examples=["SLC-40"])
    rocket: Optional[str] = Field(default=None, examples=["Falcon 9"])
    orbit: Optional[str] = Field(default=None, examples=["LEO"])
    probability: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        examples=[70],
        description="Launch probability 0-100, or omit/null when unknown",
    )
    previous_status: Optional[str] = Field(
        default=None,
        examples=["Go"],
        description="Prior status, used for scrub/delay/hold context",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "launch_id": "a7e1c2d4-1111-4b2a-9c33-0f1e2d3c4b5a",
                    "event": "t1h",
                    "mission": "Starlink Group 10-20",
                    "net": "2026-09-06T02:15:00Z",
                    "status": "Go",
                    "pad": "SLC-40",
                    "rocket": "Falcon 9",
                    "orbit": "LEO",
                    "probability": 70,
                },
                {
                    "launch_id": "a7e1c2d4-1111-4b2a-9c33-0f1e2d3c4b5a",
                    "event": "t24h",
                    "mission": "Starlink Group 10-20",
                    "net": "2026-09-07T02:15:00Z",
                    "status": "Go",
                    "pad": "SLC-40",
                    "rocket": "Falcon 9",
                    "orbit": "LEO",
                    "probability": 80,
                },
                {
                    "launch_id": "a7e1c2d4-1111-4b2a-9c33-0f1e2d3c4b5a",
                    "event": "scrub",
                    "mission": "Starlink Group 10-20",
                    "net": "2026-09-06T02:15:00Z",
                    "status": "Hold",
                    "previous_status": "Go",
                    "pad": "SLC-40",
                    "rocket": "Falcon 9",
                    "orbit": "LEO",
                    "probability": 40,
                },
            ]
        }
    }


class NotifyCopyResponse(BaseModel):
    title: str = Field(..., description="Push/local notification title, ≤50 characters", max_length=NOTIFY_TITLE_MAX)
    body: str = Field(..., description="Push/local notification body, ≤150 characters", max_length=NOTIFY_BODY_MAX)
    launch_id: str
    event: NotifyEvent
    cached: bool = Field(..., description="True when served from the Redis/local notify-copy cache")
    model: str = Field(default=GROK_MODEL, examples=[GROK_MODEL])


def _normalize_probability(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    if number == int(number):
        return int(number)
    return number


def _probability_token(probability):
    normalized = _normalize_probability(probability)
    if normalized is None:
        return None
    return f"{normalized}%"


def _probability_phrase(probability):
    """Short phrase for copy: '70% go' or 'weather 40%'."""
    normalized = _normalize_probability(probability)
    if normalized is None:
        return None
    token = _probability_token(normalized)
    if normalized < 50:
        return f"weather {token}"
    return f"{token} go"


def _probability_mentioned(text, probability):
    token = _probability_token(probability)
    if not token:
        return True
    if not text:
        return False
    number = re.escape(token[:-1])
    return bool(re.search(rf"{number}\s*%|{number}\s*percent", text, flags=re.IGNORECASE))


def _short_mission(mission):
    if not mission or not str(mission).strip():
        return "Upcoming launch"
    name = str(mission).strip()
    if "|" in name:
        name = name.split("|")[-1].strip() or name
    return name


def _clip_notify_text(text, limit):
    cleaned = re.sub(r"[*_`#]+", "", str(text or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= limit:
        return cleaned
    clipped = cleaned[:limit].rsplit(" ", 1)[0].rstrip(".,;: ")
    return clipped or cleaned[:limit]


def _ensure_probability_in_copy(title, body, probability):
    if _probability_mentioned(f"{title} {body}", probability):
        return title, body
    phrase = _probability_phrase(probability)
    if not phrase:
        return title, body
    suffix = f" {phrase}."
    if len(body) + len(suffix) <= NOTIFY_BODY_MAX:
        return title, body + suffix
    trimmed = _clip_notify_text(body, NOTIFY_BODY_MAX - len(suffix))
    return title, trimmed + suffix


def notify_copy_cache_key(launch_id, event, net, status, probability):
    prob = _normalize_probability(probability)
    prob_part = "" if prob is None else str(prob)
    return (
        f"{NOTIFY_COPY_CACHE_PREFIX}:"
        f"{launch_id or ''}|{event or ''}|{net or ''}|{status or ''}|{prob_part}"
    )


def _get_notify_copy_cached(key):
    cached = get_cached_data(key)
    if isinstance(cached, dict) and cached.get("title") and cached.get("body"):
        return cached
    entry = _local_notify_copy.get(key)
    if not entry:
        return None
    data, expires_at = entry
    if time.time() >= expires_at:
        _local_notify_copy.pop(key, None)
        return None
    if isinstance(data, dict) and data.get("title") and data.get("body"):
        return data
    return None


def _set_notify_copy_cached(key, data):
    set_cached_data(key, data, ttl=NOTIFY_COPY_TTL)
    _local_notify_copy[key] = (data, time.time() + NOTIFY_COPY_TTL)


def fallback_notify_copy(payload: NotifyCopyRequest):
    """Template copy used when Grok is unavailable. Still includes probability."""
    mission = _short_mission(payload.mission)
    pad = (payload.pad or "").strip()
    status = (payload.status or "").strip()
    phrase = _probability_phrase(payload.probability)
    prob_bit = f" {phrase}." if phrase else ""
    pad_bit = f" from {pad}" if pad else ""

    if payload.event == "t1h":
        title = f"T-1h: {mission}"
        body = f"{mission} T-1 hour{pad_bit}.{prob_bit}"
    elif payload.event == "t24h":
        title = f"T-24h: {mission}"
        body = f"{mission} T-24 hours{pad_bit}.{prob_bit}"
    else:
        status_bit = status or "scrubbed / delayed"
        title = f"Hold: {mission}"
        body = f"{mission} {status_bit} — plans changed.{prob_bit}"

    title = _clip_notify_text(title, NOTIFY_TITLE_MAX)
    body = _clip_notify_text(body, NOTIFY_BODY_MAX)
    return _ensure_probability_in_copy(title, body, payload.probability)


def _event_prompt_instructions(event):
    if event == "t1h":
        return (
            "t1h: convey T-1 hour urgency without sounding like spam. "
            "Include the mission short name and a T-1h feel. Include the pad if it fits."
        )
    if event == "t24h":
        return (
            "t24h: this is a 24-hour heads-up, not last-call. "
            "Include the mission short name and a T-24h feel. Include the pad if it fits."
        )
    return (
        "scrub: make it clear plans changed (scrub / delay / hold). "
        "Include the new status if given."
    )


def build_notify_copy_prompt(payload: NotifyCopyRequest):
    prob = _normalize_probability(payload.probability)
    if prob is None:
        probability_line = "not provided — do not invent a percentage"
    else:
        probability_line = f"{prob} (ALWAYS mention this, e.g. '70% go' or 'weather 40%')"

    return f"""Write a short push notification for an upcoming SpaceX launch alert.

Event type: {payload.event}
Mission: {payload.mission or "Unknown mission"}
NET: {payload.net or "unknown"}
Status: {payload.status or "unknown"}
Previous status: {payload.previous_status or "n/a"}
Pad: {payload.pad or "unknown"}
Rocket: {payload.rocket or "unknown"}
Orbit: {payload.orbit or "unknown"}
Launch probability: {probability_line}

Tone: witty but clear, spaceflight-nerd friendly — same spirit as Cities Skylines / Kerbal Space Program notifications, but SHORT enough for a phone banner.

Rules:
- Title ≤ {NOTIFY_TITLE_MAX} characters. Body ≤ {NOTIFY_BODY_MAX} characters.
- No markdown, no hashtags, at most one emoji (zero is preferred).
- ALWAYS mention launch probability when it is provided (e.g. "70% go" or "weather 40%").
- {_event_prompt_instructions(payload.event)}
- Return ONLY a JSON object: {{"title":"...","body":"..."}}

Examples:
- t24h: {{"title":"T-24h: Starlink 10-20","body":"Falcon 9 is on the 24-hour clock at SLC-40. Weather 80% go."}}
- t1h: {{"title":"T-1h: Starlink 10-20","body":"T-1 hour at SLC-40. 70% go — last coffee before liftoff."}}
- scrub: {{"title":"Hold: Starlink 10-20","body":"Go became Hold at SLC-40. Plans changed; weather 40%."}}
"""


def _parse_notify_copy_response(generated_text):
    text = (generated_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    candidates = [text]
    match = re.search(r"\{[^{}]*\"title\"[^{}]*\"body\"[^{}]*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("title") and data.get("body"):
            return str(data["title"]), str(data["body"])
    raise ValueError("No valid notify copy JSON found in Grok response")


def generate_notify_copy(payload: NotifyCopyRequest, increment_metrics: bool = False):
    """Generate cached Grok (or fallback) notification title+body for an alert event."""
    cache_key = notify_copy_cache_key(
        payload.launch_id,
        payload.event,
        payload.net,
        payload.status,
        payload.probability,
    )
    cached = _get_notify_copy_cached(cache_key)
    if cached:
        if increment_metrics:
            increment_metric("cache_hits")
        return {
            "title": _clip_notify_text(cached["title"], NOTIFY_TITLE_MAX),
            "body": _clip_notify_text(cached["body"], NOTIFY_BODY_MAX),
            "launch_id": payload.launch_id,
            "event": payload.event,
            "cached": True,
            "model": GROK_MODEL,
        }

    if increment_metrics:
        increment_metric("cache_misses")

    try:
        if increment_metrics:
            increment_metric("api_calls")
        generated_text = call_grok(
            build_notify_copy_prompt(payload),
            temperature=0.7,
            max_tokens=250,
        )
        title, body = _parse_notify_copy_response(generated_text)
        title = _clip_notify_text(title, NOTIFY_TITLE_MAX)
        body = _clip_notify_text(body, NOTIFY_BODY_MAX)
        title, body = _ensure_probability_in_copy(title, body, payload.probability)
    except Exception as exc:
        print(f"Notify copy Grok fallback: {exc}")
        title, body = fallback_notify_copy(payload)

    result = {
        "title": title,
        "body": body,
        "launch_id": payload.launch_id,
        "event": payload.event,
        "cached": False,
        "model": GROK_MODEL,
    }
    _set_notify_copy_cached(cache_key, {"title": title, "body": body})
    return result


# --- Ported Fetch Functions from functions.py ---

LL_API_KEY = os.getenv("LL_API_KEY", "9b91363961799d7f79aabe547ed0f7be914664dd")


def _ll_request_headers():
    """Same Launch Library token used by the 10-min list fetch."""
    return {"Authorization": f"Token {LL_API_KEY}"} if LL_API_KEY else {}


def parse_launch_data(launch: dict, is_detailed: bool = False) -> dict:
    """Helper to parse raw API launch data into the dashboard's internal format."""
    launcher_stage = launch.get('rocket', {}).get('launcher_stage', [])
    landing_type = None
    landing_location = None
    if isinstance(launcher_stage, list) and len(launcher_stage) > 0:
        landing = launcher_stage[0].get('landing')
        if landing:
            landing_type = landing.get('type', {}).get('name')
            landing_location = landing.get('landing_location', {}).get('name')
            if not landing_location:
                landing_location = landing.get('location', {}).get('name')

    mission_data = launch.get('mission') or {}
    launch_name = launch.get('name', 'Unknown')
    normalized_net = _utc_isoformat(launch.get('net'))

    raw_pad = launch.get('pad')
    if isinstance(raw_pad, dict):
        pad_name = raw_pad.get('name') or 'Unknown'
        pad_latitude = _finite_coord(raw_pad.get('latitude'))
        pad_longitude = _finite_coord(raw_pad.get('longitude'))
        pad_location_obj = raw_pad.get('location')
        pad_location = pad_location_obj.get('name') if isinstance(pad_location_obj, dict) else None
    elif raw_pad:
        pad_name = str(raw_pad)
        pad_latitude = None
        pad_longitude = None
        pad_location = None
    else:
        pad_name = 'Unknown'
        pad_latitude = None
        pad_longitude = None
        pad_location = None

    raw_image = launch.get('image')
    image_url = ''
    if isinstance(raw_image, str):
        image_url = raw_image
    elif isinstance(raw_image, dict):
        image_url = (
            raw_image.get('image_url')
            or raw_image.get('url')
            or raw_image.get('thumbnail_url')
            or ''
        )

    # API v2.3.0 uses vid_urls, while v2.0.0 uses vidURLs
    raw_vid_urls = launch.get('vid_urls') or launch.get('vidURLs') or []
    vid_urls = raw_vid_urls if isinstance(raw_vid_urls, list) else []

    return {
        'id': launch.get('id'),
        'name': launch_name,
        'mission': launch_name,
        'date': normalized_net.split('T')[0] if normalized_net else 'TBD',
        'time': normalized_net.split('T')[1].split('Z')[0] if normalized_net and 'T' in normalized_net else 'TBD',
        'net': normalized_net,
        'status': launch.get('status', {}).get('name', 'Unknown'),
        'status_id': launch.get('status', {}).get('id'),
        'rocket': launch.get('rocket', {}).get('configuration', {}).get('name', 'Unknown'),
        'orbit': mission_data.get('orbit', {}).get('name', 'Unknown'),
        'pad': pad_name,
        'pad_latitude': pad_latitude,
        'pad_longitude': pad_longitude,
        'pad_location': pad_location,
        'video_url': vid_urls[0].get('url', '') if vid_urls else '',
        'x_video_url': next((v['url'] for v in vid_urls if
                             v.get('url') and ('x.com' in v['url'].lower() or 'twitter.com' in v['url'].lower())),
                            '') if vid_urls else '',
        'landing_type': landing_type,
        'landing_location': landing_location,
        'is_detailed': is_detailed,
        # New enriched fields for "ALL data" view
        'description': mission_data.get('description', ''),
        'image': image_url,
        'window_start': _utc_isoformat(launch.get('window_start')),
        'window_end': _utc_isoformat(launch.get('window_end')),
        'probability': launch.get('probability'),
        'holdreason': launch.get('holdreason'),
        'failreason': launch.get('failreason'),
        # Captured for GET /launches compatibility, then moved to a side store
        # so the Redis list cache / slim / dashboard hot path stays small.
        'all_data': {k: v for k, v in launch.items() if
                     k not in ['vidURLs', 'infoURLs', 'vid_urls', 'info_urls', 'infographic']},
    }


def fetch_launch_details(launch_id: str):
    """Fetch one launch from Launch Library using the same auth token as list fetch."""
    if not launch_id:
        return None
    increment_metric("api_calls")
    url = f"https://ll.thespacedevs.com/2.3.0/launches/{launch_id}/"
    headers = _ll_request_headers()
    print(f"Fetching details for launch {launch_id}")
    try:
        try:
            response = requests.get(url, headers=headers, timeout=10, verify=True)
        except Exception:
            # Fallback for SSL issues
            response = requests.get(url, headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Failed to fetch launch details for {launch_id}: {e}")
        return None


def fetch_launches(existing_previous=None, existing_upcoming=None):
    """Fetch SpaceX launch data (v2.3.0) using detailed mode to get all fields."""
    headers = _ll_request_headers()

    combined_prev = existing_previous or []
    combined_up = existing_upcoming or []

    # 1. Fetch Previous (Incremental)
    try:
        increment_metric("api_calls")
        prev_url = 'https://ll.thespacedevs.com/2.3.0/launches/previous/?lsp__name=SpaceX&limit=15&mode=detailed'
        prev_response = requests.get(prev_url, headers=headers, timeout=15)
        prev_response.raise_for_status()
        prev_data = prev_response.json().get('results', [])
        parsed_prev = [parse_launch_data(l, is_detailed=True) for l in prev_data]

        if existing_previous:
            # Incremental update: Replace existing ones with fresh data if present in latest fetch
            new_ids = {l['id'] for l in parsed_prev if 'id' in l}
            filtered_old = [l for l in existing_previous if l.get('id') not in new_ids]
            # Maintain newest-first order by sorting after merge
            combined_prev = sorted(parsed_prev + filtered_old, key=lambda x: x.get('net') or '', reverse=True)
            # Limit history to 2000 items for efficiency and to allow seeding
            combined_prev = combined_prev[:2000]
        else:
            combined_prev = sorted(parsed_prev, key=lambda x: x.get('net') or '', reverse=True)
    except Exception as e:
        print(f"Error fetching previous launches: {e}")

    # 2. Fetch Upcoming (Full Refresh)
    try:
        # Upcoming is always fully refreshed as statuses and dates shift frequently
        increment_metric("api_calls")
        up_url = 'https://ll.thespacedevs.com/2.3.0/launches/upcoming/?lsp__name=SpaceX&limit=15&mode=detailed'
        up_response = requests.get(up_url, headers=headers, timeout=15)
        up_response.raise_for_status()
        up_data = up_response.json().get('results', [])

        parsed_up = [parse_launch_data(l, is_detailed=True) for l in up_data]
        # Filter out finished launches from upcoming list to satisfy user request
        # Status IDs: 3=Success, 4=Failure, 7=Partial Failure
        # This keeps "active" launches like "In Flight" (ID 6) and "Go" (ID 1)
        combined_up = [l for l in parsed_up if l.get('status_id') not in [3, 4, 7]]
    except Exception as e:
        print(f"Error fetching upcoming launches: {e}")

    return {
        'previous': combined_prev,
        'upcoming': combined_up
    }


def seed_historical_launches():
    """Seed the historical launch cache by pulling increasingly older launches in batches of 5 until we hit the api limit."""
    print("Starting historical launch seeding...")
    cache_key = LAUNCHES_CACHE_KEY

    # Get initial state
    existing_previous = []
    cached_data = get_cached_data(cache_key)
    if cached_data:
        _sanitize_launch_list_cache(cached_data)
        existing_previous = cached_data.get('previous', [])

    # Sort to find the actual oldest launch
    if existing_previous:
        existing_previous.sort(key=lambda x: x.get('net') or '', reverse=True)
        oldest_so_far = existing_previous[-1].get('net')
    else:
        oldest_so_far = None

    total_so_far = len(existing_previous)
    update_seeding_status(True, "Starting batch fetch...", total_so_far, oldest_so_far)

    # We use a loop to keep fetching until we hit a limit or run out of data
    upcoming = []
    while True:
        # Check for stop signal
        global _stop_seeding_requested
        stop_signal = _stop_seeding_requested
        if r:
            try:
                if r.get(SEEDING_STOP_SIGNAL_KEY) == "true":
                    stop_signal = True
            except:
                pass

        if stop_signal:
            print("Seeding: Stop signal received. Terminating.")
            update_seeding_status(False, "Stopped by user.", total_so_far, oldest_so_far)
            _stop_seeding_requested = False
            if r:
                try:
                    r.delete(SEEDING_STOP_SIGNAL_KEY)
                except:
                    pass
            break

        # Get current state from cache to determine where we are
        # This allows us to pick up any new launches added by the regular refresh
        cached_data = get_cached_data(cache_key)
        if cached_data:
            _sanitize_launch_list_cache(cached_data)
            existing_previous = cached_data.get('previous', [])
            upcoming = cached_data.get('upcoming', [])

        if existing_previous:
            # Crucial: Always sort to ensure the last item is truly the oldest
            existing_previous.sort(key=lambda x: x.get('net') or '', reverse=True)
            oldest_so_far = existing_previous[-1].get('net')
        else:
            oldest_so_far = None

        total_so_far = len(existing_previous)

        if total_so_far >= 2000:
            print(f"Seeding: Already at history limit ({total_so_far}). Stopping.")
            update_seeding_status(False, f"Complete. Hit history limit ({total_so_far}).", total_so_far, oldest_so_far)
            break

        # Use date-based pagination for robustness against cache shifts
        status_msg = f"Fetching launches older than {oldest_so_far.split('T')[0] if oldest_so_far else 'now'}..."
        update_seeding_status(True, status_msg, total_so_far, oldest_so_far)
        print(f"Seeding: {status_msg}")

        try:
            headers = {'Authorization': f'Token {LL_API_KEY}'}
            params = {
                'lsp__name': 'SpaceX',
                'limit': 20,
                'mode': 'detailed',
                'ordering': '-net'
            }
            if oldest_so_far:
                params['net__lt'] = oldest_so_far

            # Pull increasingly older launches in batches
            increment_metric("api_calls")
            url = 'https://ll.thespacedevs.com/2.3.0/launches/previous/'
            response = requests.get(url, headers=headers, params=params, timeout=15)

            if response.status_code == 429:
                print("Hit API rate limit (429) during seeding. Stopping for now.")
                update_seeding_status(False, "Hit API rate limit (429).", total_so_far, oldest_so_far)
                break

            response.raise_for_status()
            data = response.json()
            results = data.get('results', [])

            if not results:
                print("No more historical launches found. Seeding complete.")
                update_seeding_status(False, "Seeding complete. No more data.", total_so_far, oldest_so_far)
                break

            parsed_new = [parse_launch_data(l, is_detailed=True) for l in results]

            # Filter out duplicates (possible if multiple launches have exact same timestamp)
            existing_ids = {l['id'] for l in existing_previous if 'id' in l}
            unique_new = [l for l in parsed_new if l.get('id') not in existing_ids]

            if not unique_new and results:
                print("Seeding: All fetched results are duplicates. Stopping.")
                update_seeding_status(False, "Stopped to avoid duplicate loop.", total_so_far, oldest_so_far)
                break

            # Append older launches and maintain sorted order
            combined_prev = sorted(existing_previous + unique_new, key=lambda x: x.get('net') or '', reverse=True)

            # Limit history to 2000 items for efficiency
            combined_prev = combined_prev[:2000]

            # Update cache in Redis
            result = {
                "upcoming": upcoming,
                "previous": combined_prev,
                "last_updated": _utc_isoformat()
            }
            _sanitize_launch_list_cache(result)
            set_cached_data(cache_key, result)
            _remember_launches(result)

            total_so_far = len(combined_prev)
            oldest_so_far = combined_prev[-1].get('net')
            print(f"Added {len(unique_new)} historical launches. Total: {total_so_far}")
            update_seeding_status(True, f"Added {len(unique_new)} launches.", total_so_far, oldest_so_far)

            # Brief sleep between batches
            time.sleep(1)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print("Hit API rate limit (429) during seeding. Stopping.")
                update_seeding_status(False, "Hit API rate limit (429).", total_so_far, oldest_so_far)
                break
            else:
                msg = f"HTTP error during seeding: {e}"
                print(msg)
                update_seeding_status(False, msg, total_so_far, oldest_so_far)
                break
        except Exception as e:
            msg = f"Unexpected error during seeding: {e}"
            print(msg)
            update_seeding_status(False, msg, total_so_far, oldest_so_far)
            break


def parse_metar(raw_metar: str):
    """Parse METAR string to extract weather data."""
    temperature_c = 25
    dewpoint_c = 15
    wind_speed_kts = 0
    wind_gust_kts = 0
    wind_direction = 0
    cloud_cover = 0
    visibility_sm = 10
    altimeter_inhg = 29.92

    try:
        # Extract temperature and dewpoint
        # Format: 18/14 or M01/M05
        temp_dew_match = re.search(r'(M?\d{2})/(M?\d{2})', raw_metar)
        if temp_dew_match:
            t_str = temp_dew_match.group(1)
            d_str = temp_dew_match.group(2)

            temperature_c = int(t_str.replace('M', '-'))
            dewpoint_c = int(d_str.replace('M', '-'))

        # Extract wind
        # Format: 16008KT or 16008G15KT or VRB05KT
        wind_match = re.search(r'(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?KT', raw_metar)
        if wind_match:
            dir_str = wind_match.group(1)
            wind_direction = int(dir_str) if dir_str != 'VRB' else 0
            wind_speed_kts = int(wind_match.group(2))
            if wind_match.group(3):
                wind_gust_kts = int(wind_match.group(3))

        # Extract visibility
        # Format: 10SM or 1/2SM
        vis_match = re.search(r'(\d+(?:\s\d/\d)?SM)', raw_metar)
        if vis_match:
            vis_str = vis_match.group(1).replace('SM', '')
            if ' ' in vis_str:
                parts = vis_str.split(' ')
                visibility_sm = float(parts[0]) + (eval(parts[1]) if '/' in parts[1] else 0)
            elif '/' in vis_str:
                visibility_sm = eval(vis_str)
            else:
                visibility_sm = float(vis_str)

        # Extract altimeter
        # Format: A3012
        alt_match = re.search(r'A(\d{4})', raw_metar)
        if alt_match:
            altimeter_inhg = int(alt_match.group(1)) / 100.0

        # Cloud cover estimation and ceiling
        ceiling_ft = 10000
        if 'SKC' in raw_metar or 'CLR' in raw_metar or 'NCD' in raw_metar:
            cloud_cover = 0
        elif 'FEW' in raw_metar:
            cloud_cover = 25
        elif 'SCT' in raw_metar:
            cloud_cover = 50
        elif 'BKN' in raw_metar:
            cloud_cover = 75
        elif 'OVC' in raw_metar:
            cloud_cover = 100
        else:
            cloud_cover = 50

        # Extract ceiling (lowest BKN or OVC layer)
        ceiling_match = re.search(r'(BKN|OVC)(\d{3})', raw_metar)
        if ceiling_match:
            ceiling_ft = int(ceiling_match.group(2)) * 100

        # Humidity calculation (August-Roche-Magnus)
        import math
        es = 6.112 * math.exp((17.67 * temperature_c) / (temperature_c + 243.5))
        e = 6.112 * math.exp((17.67 * dewpoint_c) / (dewpoint_c + 243.5))
        humidity = min(100, max(0, int(100 * (e / es))))

        # Flight category
        if visibility_sm > 5 and ceiling_ft > 3000:
            flight_category = "VFR"
        elif visibility_sm >= 3 and ceiling_ft >= 1000:
            flight_category = "MVFR"
        elif visibility_sm >= 1 and ceiling_ft >= 500:
            flight_category = "IFR"
        else:
            flight_category = "LIFR"

    except Exception as e:
        print(f"Error parsing METAR: {e}")
        humidity = 50
        flight_category = "VFR"

    return {
        'temperature_c': temperature_c,
        'temperature_f': round(temperature_c * 9 / 5 + 32, 1),
        'dewpoint_c': dewpoint_c,
        'dewpoint_f': round(dewpoint_c * 9 / 5 + 32, 1),
        'humidity': humidity,
        'wind_speed_kts': wind_speed_kts,
        'wind_gust_kts': wind_gust_kts,
        'wind_direction': wind_direction,
        'visibility_sm': visibility_sm,
        'altimeter_inhg': altimeter_inhg,
        'cloud_cover': cloud_cover,
        'flight_category': flight_category,
        'raw': raw_metar
    }


def fetch_forecast(location: str = None, lat: float = None, lon: float = None):
    """Fetch 7-day forecast (daily + hourly) from Open-Meteo."""
    increment_metric("api_calls")
    if lat is None or lon is None:
        coords = LOCATION_COORDS.get(location) or LOCATION_COORDS['Starbase']
        lat, lon = coords['lat'], coords['lon']

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,weathercode&hourly=temperature_2m,windspeed_10m,winddirection_10m&current_weather=true&timezone=auto"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching forecast for {location or (lat, lon)}: {e}")
        return None


def is_nws_fresh(timestamp_str):
    """Check if NWS timestamp is within the last 3 hours."""
    if not timestamp_str: return False
    try:
        # NWS timestamp is often like 2024-01-01T00:00:00+00:00
        ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return datetime.now(timezone.utc) - ts < timedelta(hours=3)
    except Exception:
        return False


def fetch_weather(location: str = None, station_id: str = None, lat: float = None, lon: float = None):
    """Fetch METAR weather data with multiple station fallbacks and Open-Meteo backup."""
    increment_metric("api_calls")

    stations_to_try = []
    if station_id:
        stations_to_try = [station_id]
    elif location and location in METAR_STATIONS:
        stations_to_try = METAR_STATIONS[location]
        if not lat or not lon:
            coords = LOCATION_COORDS.get(location)
            if coords:
                lat, lon = coords['lat'], coords['lon']
    elif lat is not None and lon is not None:
        try:
            # Search within 50nm
            search_url = f"https://aviationweather.gov/api/data/metar?radialDistance={lat},{lon};50&format=raw"
            res = requests.get(search_url, timeout=5)
            if res.status_code == 200 and res.text.strip():
                # Take the first METAR station ID
                first_metar = res.text.strip().split('\n')[0]
                found_sid = first_metar.split(' ')[0]
                stations_to_try = [found_sid]
        except Exception:
            pass

    if not stations_to_try:
        if location == 'Starbase' or not location:
            stations_to_try = ['KBRO']
            if not lat: lat, lon = LOCATION_COORDS['Starbase']['lat'], LOCATION_COORDS['Starbase']['lon']
        else:
            # If we have a location but it's not in METAR_STATIONS, we might still have coords
            if not lat and location in LOCATION_COORDS:
                lat, lon = LOCATION_COORDS[location]['lat'], LOCATION_COORDS[location]['lon']

    errors = []
    # Try each station until one succeeds
    for sid in stations_to_try:
        try:
            # 1. Try to fetch live wind/data from NWS API
            live_data = {}
            try:
                nws_headers = {'User-Agent': '(my-launch-app, contact@example.com)'}
                nws_url = f"https://api.weather.gov/stations/{sid}/observations/latest"
                nws_res = requests.get(nws_url, headers=nws_headers, timeout=5)
                if nws_res.status_code == 200:
                    props = nws_res.json().get('properties', {})
                    live_data = {
                        'temp': props.get('temperature', {}).get('value'),
                        'dew': props.get('dewpoint', {}).get('value'),
                        'wind_speed': props.get('windSpeed', {}).get('value'),
                        'wind_gust': props.get('windGust', {}).get('value'),
                        'wind_dir': props.get('windDirection', {}).get('value'),
                        'vis': props.get('visibility', {}).get('value'),
                        'timestamp': props.get('timestamp'),
                        'source': 'NWS Real-time'
                    }
            except Exception as e:
                print(f"NWS error for {sid}: {e}")

            # 2. Try to fetch METAR from Aviation Weather
            aw_url = f"https://aviationweather.gov/api/data/metar?ids={sid}&format=raw"
            aw_res = requests.get(aw_url, timeout=5)
            raw_metar = aw_res.text.strip() if aw_res.status_code == 200 else ""

            if raw_metar:
                parsed = parse_metar(raw_metar)
                # Merge with NWS data if NWS is fresh
                if live_data.get('timestamp') and is_nws_fresh(live_data['timestamp']):
                    if live_data['temp'] is not None:
                        parsed['temperature_c'] = live_data['temp']
                        parsed['temperature_f'] = round(live_data['temp'] * 9 / 5 + 32, 1)
                    if live_data['wind_speed'] is not None:
                        parsed['wind_speed_kts'] = round(live_data['wind_speed'] * 0.539957, 1)
                    if live_data['wind_gust'] is not None:
                        parsed['wind_gust_kts'] = round(live_data['wind_gust'] * 0.539957, 1)
                    if live_data['wind_dir'] is not None:
                        parsed['wind_direction'] = int(live_data['wind_dir'])
                    parsed['live_wind'] = live_data  # Backward compatibility
                return parsed

            # 3. If no METAR but NWS data is fresh, use NWS as primary
            if live_data.get('temp') is not None and is_nws_fresh(live_data['timestamp']):
                temp_c = live_data['temp']
                return {
                    'temperature_c': temp_c,
                    'temperature_f': round(temp_c * 9 / 5 + 32, 1),
                    'dewpoint_c': live_data.get('dew') or (temp_c - 5),
                    'dewpoint_f': round((live_data.get('dew') or (temp_c - 5)) * 9 / 5 + 32, 1),
                    'wind_speed_kts': round(live_data.get('wind_speed', 0) * 0.539957, 1) if live_data.get(
                        'wind_speed') else 0,
                    'wind_gust_kts': round(live_data.get('wind_gust', 0) * 0.539957, 1) if live_data.get(
                        'wind_gust') else 0,
                    'wind_direction': int(live_data.get('wind_dir') or 0),
                    'visibility_sm': round(live_data.get('vis', 16093) / 1609.34, 1) if live_data.get('vis') else 10.0,
                    'flight_category': 'VFR',
                    'timestamp': live_data['timestamp'],
                    'source': 'NWS Real-time (No METAR)'
                }
        except Exception as e:
            errors.append(f"{sid}: {str(e)}")

    # 4. If all station attempts failed, fall back to Open-Meteo (model-based current weather)
    if lat is not None and lon is not None:
        try:
            forecast = fetch_forecast(lat=lat, lon=lon)
            if forecast and 'current_weather' in forecast:
                cw = forecast['current_weather']
                temp_c = cw['temperature']
                return {
                    'temperature_c': temp_c,
                    'temperature_f': round(temp_c * 9 / 5 + 32, 1),
                    'wind_speed_kts': round(cw.get('windspeed', 0) * 0.539957, 1),
                    'wind_direction': cw.get('winddirection', 0),
                    'flight_category': 'VFR',
                    'source': 'Open-Meteo',
                    'note': 'Modeled data (all METAR/NWS stations failed)'
                }
        except Exception as e:
            errors.append(f"Open-Meteo: {str(e)}")

    # Final fallback if everything failed - at least it's marked as an error
    return {
        'temperature_c': 25, 'temperature_f': 77,
        'wind_speed_kts': 0, 'flight_category': 'VFR',
        'error': f"All weather sources failed: {'; '.join(errors)}"
    }


def fetch_external_narratives():
    """Fetch narratives from the external API as specified in functions.py."""
    url = "https://launch-narrative-api-dafccc521fb8.herokuapp.com/recent_launches_narratives"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and 'descriptions' in data:
            return data['descriptions']
        return data
    except Exception as e:
        print(f"Error fetching external narratives: {e}")
        return []


# --- Background Refresh Helpers ---

def refresh_narratives_internal():
    """Internal helper to refresh narratives cache."""
    def _do():
        print("Refreshing narratives cache...")
        cached_narratives = get_cached_data(CACHE_KEY)
        if not cached_narratives:
            cached_narratives = _local_cache["launch_narratives"]

        try:
            descriptions = generate_narratives(existing_narratives=cached_narratives)
            current_time_dt = datetime.now(timezone.utc)
            current_time = _utc_isoformat(current_time_dt)
            set_cached_data(CACHE_KEY, descriptions)
            set_cached_data(CACHE_TIME_KEY, current_time)

            _local_cache["launch_narratives"] = descriptions
            _local_cache["last_updated"] = current_time_dt
            return descriptions
        except Exception as e:
            print(f"Error in refresh_narratives_internal: {e}")
            return cached_narratives

    return _narratives_flight.do(_do)


def _refresh_launches_uncached():
    """Fetch launches and write a slim cache. Not concurrency-safe on its own."""
    print("Refreshing launches cache...")
    existing_previous = None
    existing_upcoming = None

    cached_data = get_cached_data(LAUNCHES_CACHE_KEY)
    if cached_data:
        _sanitize_launch_list_cache(cached_data)
        existing_previous = cached_data.get('previous')
        existing_upcoming = cached_data.get('upcoming')

    try:
        data = fetch_launches(existing_previous=existing_previous, existing_upcoming=existing_upcoming)

        # Add trajectory data to the first upcoming launch only
        upcoming = data.get("upcoming", [])
        previous = data.get("previous", [])
        if upcoming:
            traj = get_launch_trajectory_data(upcoming[0], previous)
            if traj:
                upcoming[0]['trajectory_data'] = traj
        elif previous:
            # Fallback to most recent previous launch if no upcoming
            traj = get_launch_trajectory_data(previous[0], previous)
            if traj:
                previous[0]['trajectory_data'] = traj

        last_updated = _utc_isoformat()
        result = {
            "upcoming": upcoming,
            "previous": previous,
            "last_updated": last_updated
        }
        _sanitize_launch_list_cache(result)
        set_cached_data(LAUNCHES_CACHE_KEY, result)
        return _remember_launches(result)
    except Exception as e:
        print(f"Error in refresh_launches_internal: {e}")
        return None


def refresh_launches_internal():
    """Internal helper to refresh launches cache (single-flight)."""
    def _do():
        result = _redis_single_flight("launches", _refresh_launches_uncached)
        if result is None:
            cached = get_cached_data(LAUNCHES_CACHE_KEY)
            if cached:
                _sanitize_launch_list_cache(cached, persist=True)
                return _remember_launches(cached)
        return result

    return _launches_flight.do(_do)


def _finalize_weather(data):
    """Normalize weather objects for dashboard clients."""
    if not isinstance(data, dict):
        return data
    normalized = _coerce_weather_numbers(data)
    def _normalize_timestamp_fields(obj):
        if isinstance(obj, dict):
            out = {}
            for key, value in obj.items():
                if key in ('last_updated', 'timestamp'):
                    out[key] = _utc_isoformat(value)
                else:
                    out[key] = _normalize_timestamp_fields(value)
            return out
        if isinstance(obj, list):
            return [_normalize_timestamp_fields(item) for item in obj]
        return obj

    normalized = _normalize_timestamp_fields(normalized)
    if 'wind_gusts_kts' not in normalized:
        gust = normalized.get('wind_gust_kts')
        speed = normalized.get('wind_speed_kts') or 0
        normalized['wind_gusts_kts'] = gust if gust not in (None, 0) else speed
    return normalized


def _fetch_all_weather():
    """Hit METAR / Open-Meteo for every dashboard site and write per-location cache."""
    print("Refreshing weather cache...")
    weather_results = {}
    timestamps = []

    for loc in WEATHER_LOCATIONS:
        data = fetch_weather(loc)
        forecast = fetch_forecast(loc)
        data['forecast'] = forecast
        last_updated = _utc_isoformat()
        data['last_updated'] = last_updated
        data = _finalize_weather(data)
        weather_results[loc] = data
        timestamps.append(last_updated)
        _set_weather_loc(loc, data)

    result = {
        "weather": weather_results,
        "last_updated": min(timestamps) if timestamps else _utc_isoformat()
    }
    global _weather_last_refresh_at, _weather_last_result
    _weather_last_refresh_at = time.time()
    _weather_last_result = result
    if r:
        try:
            r.setex("weather_refresh_done_at", 60, str(_weather_last_refresh_at))
        except Exception:
            pass
    return result


def _weather_is_fresh(force=False):
    if force:
        return False
    now = time.time()
    if _weather_last_result is not None and (now - _weather_last_refresh_at) < WEATHER_DEBOUNCE_SEC:
        return True
    if r:
        try:
            stamp = r.get("weather_refresh_done_at")
            if stamp and (now - float(stamp)) < WEATHER_DEBOUNCE_SEC:
                return True
        except Exception:
            pass
    return False


def refresh_weather_internal(force: bool = False):
    """Refresh all weather caches with debounce + single-flight coalescing."""
    def _do():
        if _weather_is_fresh(force=force):
            assembled = _weather_last_result or _assemble_weather_all()
            if assembled and assembled.get("weather"):
                return assembled
        result = _redis_single_flight("weather", _fetch_all_weather)
        if result is None:
            return _weather_last_result or _assemble_weather_all() or {
                "weather": {},
                "last_updated": None,
            }
        return result

    return _weather_flight.do(_do)


def _normalize_satellite_group(group: str) -> str:
    """Allowlisted CelesTrak GROUP values only (prevents unbounded upstream fetches)."""
    normalized = (group or "").strip().lower()
    if normalized not in SATELLITE_GROUPS:
        allowed = ", ".join(SATELLITE_GROUPS)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported satellite group '{group}'. Allowed: {allowed}",
        )
    return normalized


def _satellite_cache_key(group: str) -> str:
    return f"{SATELLITES_CACHE_PREFIX}{group}"


def _get_satellite_flight(group: str) -> _SingleFlight:
    with _satellite_flights_lock:
        flight = _satellite_flights.get(group)
        if flight is None:
            flight = _SingleFlight()
            _satellite_flights[group] = flight
        return flight


def _tle_checksum(line: str) -> str:
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return str(total % 10)


def _tle_norad5(norad_id) -> str:
    """5-char NORAD field; Space-Track alpha-5 for catalog numbers >= 100000."""
    try:
        n = int(norad_id)
    except (TypeError, ValueError):
        return "00000"
    if n < 0:
        n = 0
    if n < 100000:
        return f"{n:05d}"
    letter_idx = (n // 10000) - 10
    if 0 <= letter_idx < len(_TLE_ALPHA5):
        return f"{_TLE_ALPHA5[letter_idx]}{n % 10000:04d}"
    return f"{n:05d}"[-5:]


def _tle_intl_designator(object_id) -> str:
    if not object_id or "-" not in str(object_id):
        return " " * 8
    year, rest = str(object_id).split("-", 1)
    yy = year[-2:] if len(year) >= 2 and year[-2:].isdigit() else "  "
    return f"{yy}{rest:<6}"[:8]


def _tle_exp_field(value) -> str:
    """8-char TLE scientific notation with implied leading decimal (BSTAR / n_ddot)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    if number == 0.0:
        return " 00000+0"
    sign = "-" if number < 0 else " "
    abs_val = abs(number)
    exp = 0
    while abs_val >= 1.0:
        abs_val /= 10.0
        exp += 1
    while abs_val < 0.1:
        abs_val *= 10.0
        exp -= 1
    mantissa = int(round(abs_val * 1e5))
    if mantissa >= 100000:
        mantissa = 10000
        exp += 1
    exp_sign = "+" if exp >= 0 else "-"
    return f"{sign}{mantissa:05d}{exp_sign}{abs(exp)}"


def _tle_n_dot_field(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    sign = "-" if number < 0 else " "
    return f"{sign}{abs(number):.8f}"[0] + f"{abs(number):.8f}"[1:]


def _epoch_year_and_day(epoch) -> tuple:
    text = str(epoch).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    year2 = dt.year % 100
    seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
    epoch_day = dt.timetuple().tm_yday + seconds / 86400.0
    return year2, epoch_day


def omm_to_tle_lines(omm: dict):
    """Build TLE line 1/2 from a CelesTrak GP/OMM JSON object."""
    if not isinstance(omm, dict) or omm.get("NORAD_CAT_ID") is None or not omm.get("EPOCH"):
        return None, None
    try:
        norad5 = _tle_norad5(omm.get("NORAD_CAT_ID"))
        classification = str(omm.get("CLASSIFICATION_TYPE") or "U")[:1] or "U"
        intl = _tle_intl_designator(omm.get("OBJECT_ID"))
        year2, epoch_day = _epoch_year_and_day(omm.get("EPOCH"))
        n_dot = _tle_n_dot_field(omm.get("MEAN_MOTION_DOT") or 0)
        n_ddot = _tle_exp_field(omm.get("MEAN_MOTION_DDOT") or 0)
        bstar = _tle_exp_field(omm.get("BSTAR") or 0)
        eph_type = int(omm.get("EPHEMERIS_TYPE") or 0)
        elset = int(omm.get("ELEMENT_SET_NO") or 999) % 10000
        line1 = (
            f"1 {norad5}{classification} {intl} {year2:02d}{epoch_day:012.8f} "
            f"{n_dot} {n_ddot} {bstar} {eph_type} {elset:4d}"
        )
        if len(line1) < 68:
            line1 = line1.ljust(68)
        line1 = line1[:68] + _tle_checksum(line1)

        inclination = float(omm["INCLINATION"])
        raan = float(omm["RA_OF_ASC_NODE"])
        eccentricity = float(omm["ECCENTRICITY"])
        argp = float(omm["ARG_OF_PERICENTER"])
        mean_anomaly = float(omm["MEAN_ANOMALY"])
        mean_motion = float(omm["MEAN_MOTION"])
        rev = int(omm.get("REV_AT_EPOCH") or 0) % 100000
        ecc_digits = f"{int(round(abs(eccentricity) * 1e7)):07d}"[:7]
        line2 = (
            f"2 {norad5} {inclination:8.4f} {raan:8.4f} {ecc_digits} "
            f"{argp:8.4f} {mean_anomaly:8.4f} {mean_motion:11.8f}{rev:05d}"
        )
        if len(line2) < 68:
            line2 = line2.ljust(68)
        line2 = line2[:68] + _tle_checksum(line2)
        return line1, line2
    except (TypeError, ValueError, KeyError, OverflowError):
        return None, None


def slim_gp_record(omm: dict) -> dict:
    """Compact satellite.js record: name + norad + TLE, with slim OMM fallback."""
    name = omm.get("OBJECT_NAME") or ""
    norad = omm.get("NORAD_CAT_ID")
    record = {"name": name, "norad_id": norad}
    line1, line2 = omm_to_tle_lines(omm)
    if line1 and line2:
        record["tle_line1"] = line1
        record["tle_line2"] = line2
        return record
    for key in OMM_PROP_FIELDS:
        if key in omm:
            record[key] = omm[key]
    return record


def _satellite_fetched_dt(payload):
    raw = (payload or {}).get("fetched_at")
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def satellite_age_seconds(payload):
    dt = _satellite_fetched_dt(payload)
    if dt is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())


def satellite_cache_is_fresh(payload, ttl=SATELLITES_CACHE_TTL) -> bool:
    age = satellite_age_seconds(payload)
    return age is not None and age < ttl


def _read_satellite_cache(group: str):
    local = _local_satellites.get(group)
    if isinstance(local, dict) and local.get("satellites") is not None:
        return local
    cached = get_cached_data(_satellite_cache_key(group))
    if isinstance(cached, dict) and cached.get("satellites") is not None:
        _local_satellites[group] = cached
        return cached
    return None


def _satellite_http_cache_key(group: str) -> str:
    return f"{SATELLITES_HTTP_CACHE_PREFIX}{group}"


def _present_satellite_payload(payload: dict, *, stale: bool) -> dict:
    """Shallow copy with an honest stale flag. Satellites are not copied."""
    body = dict(payload)
    body["stale"] = bool(stale)
    body["ttl_seconds"] = SATELLITES_CACHE_TTL
    return body


def _render_satellite_http(payload: dict) -> dict:
    """JSON and gzip bytes for the fresh and stale variants of one catalog."""
    fresh = _present_satellite_payload(payload, stale=False)
    stale = _present_satellite_payload(payload, stale=True)
    json_fresh = json.dumps(fresh, separators=(",", ":")).encode("utf-8")
    json_stale = json.dumps(stale, separators=(",", ":")).encode("utf-8")
    count = payload.get("count")
    if count is None:
        count = len(payload.get("satellites") or [])
    return {
        "fetched_at": payload.get("fetched_at"),
        "count": count,
        "json_fresh": json_fresh,
        "json_stale": json_stale,
        "gzip_fresh": gzip.compress(json_fresh, compresslevel=6),
        "gzip_stale": gzip.compress(json_stale, compresslevel=6),
    }


def _persist_satellite_http(group: str, doc: dict):
    blob = {
        "fetched_at": doc.get("fetched_at"),
        "count": doc.get("count"),
        "gzip_fresh": base64.b64encode(doc["gzip_fresh"]).decode("ascii"),
        "gzip_stale": base64.b64encode(doc["gzip_stale"]).decode("ascii"),
    }
    set_cached_data(_satellite_http_cache_key(group), blob, ttl=SATELLITES_STALE_TTL)


def _remember_satellite_http(group: str, payload: dict, persist: bool = True) -> dict:
    """Keep pre-rendered response bytes so later requests do not re-serialize."""
    with _satellite_refresh_lock:
        existing = _local_satellite_http.get(group)
        if (
            existing
            and existing.get("gzip_fresh")
            and existing.get("fetched_at") == payload.get("fetched_at")
        ):
            return existing
        doc = _render_satellite_http(payload)
        _local_satellite_http[group] = doc
    if not persist:
        return doc
    if _background_enabled:
        threading.Thread(
            target=_persist_satellite_http,
            args=(group, doc),
            daemon=True,
            name=f"sat-http-{group}",
        ).start()
    else:
        _persist_satellite_http(group, doc)
    return doc


def _read_satellite_http(group: str):
    """Pre-rendered bytes only. Does not parse the satellite array."""
    doc = _local_satellite_http.get(group)
    if isinstance(doc, dict) and doc.get("gzip_fresh") and doc.get("gzip_stale"):
        return doc
    cached = get_cached_data(_satellite_http_cache_key(group))
    if not isinstance(cached, dict):
        return None
    try:
        gzip_fresh = base64.b64decode(cached["gzip_fresh"])
        gzip_stale = base64.b64decode(cached["gzip_stale"])
        doc = {
            "fetched_at": cached.get("fetched_at"),
            "count": cached.get("count"),
            "gzip_fresh": gzip_fresh,
            "gzip_stale": gzip_stale,
            "json_fresh": gzip.decompress(gzip_fresh),
            "json_stale": gzip.decompress(gzip_stale),
        }
    except Exception:
        return None
    _local_satellite_http[group] = doc
    return doc


def _client_accepts_gzip(request: Request) -> bool:
    if request is None:
        return False
    accept = request.headers.get("accept-encoding") or ""
    return "gzip" in accept.lower()


def _encoded_satellite_response(doc: dict, request: Request, *, stale: bool, cache_state: str):
    use_gzip = _client_accepts_gzip(request)
    content = doc["gzip_stale" if stale else "gzip_fresh"] if use_gzip else doc[
        "json_stale" if stale else "json_fresh"
    ]
    headers = {
        "Vary": "Accept-Encoding",
        "X-Satellite-Cache": cache_state,
    }
    if use_gzip:
        headers["Content-Encoding"] = "gzip"
    return Response(content=content, media_type="application/json", headers=headers)


def _schedule_satellite_refresh(group: str):
    """One background CelesTrak refresh per group. Never blocks the request."""
    if not _background_enabled:
        return
    now = time.monotonic()
    with _satellite_refresh_lock:
        if group in _satellite_refresh_inflight:
            return
        if now < _satellite_refresh_after.get(group, 0):
            return
        _satellite_refresh_inflight.add(group)
        _satellite_refresh_after[group] = now + SATELLITES_REFRESH_COOLDOWN_SEC

    def _run():
        try:
            print(f"Background satellite GP refresh starting for {group}")
            refresh_satellites_internal(group)
        except Exception as exc:
            print(f"Background satellite GP refresh failed for {group}: {exc}")
        finally:
            with _satellite_refresh_lock:
                _satellite_refresh_inflight.discard(group)

    threading.Thread(target=_run, daemon=True, name=f"sat-refresh-{group}").start()


def _write_satellite_cache(group: str, payload: dict):
    _local_satellites[group] = payload
    set_cached_data(_satellite_cache_key(group), payload, ttl=SATELLITES_STALE_TTL)
    _remember_satellite_http(group, payload, persist=True)


def _satellite_payload(group: str, satellites: list, stale: bool = False) -> dict:
    return {
        "group": group,
        "fetched_at": _utc_isoformat(),
        "ttl_seconds": SATELLITES_CACHE_TTL,
        "count": len(satellites),
        "stale": stale,
        "source": "celestrak",
        "note": SATELLITE_NOTE,
        "satellites": satellites,
    }


def _run_with_deadline(fn, seconds: float):
    """Run ``fn`` and abandon the wait when the wall clock expires.

    ``requests`` timeouts are per socket read. A slow trickle still completes
    them and can sit on a dyno until Heroku's router kills the HTTP request.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn)
    try:
        return future.result(timeout=seconds)
    except FuturesTimeout:
        raise TimeoutError(f"operation exceeded {seconds:.0f}s") from None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def fetch_celestrak_gp(group: str, deadline: float = None) -> list:
    """Fetch GP/OMM JSON for an allowlisted CelesTrak group and slim it."""
    seconds = SATELLITES_GP_DEADLINE_SEC if deadline is None else deadline
    increment_metric("api_calls")

    def _download():
        params = {"GROUP": group, "FORMAT": "JSON"}
        response = requests.get(
            CELESTRAK_GP_URL,
            params=params,
            headers=CELESTRAK_HEADERS,
            timeout=SATELLITES_FETCH_TIMEOUT,
        )
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError as exc:
            raise ValueError("CelesTrak returned non-JSON GP data") from exc
        if not isinstance(data, list):
            raise ValueError("CelesTrak GP JSON was not a list")
        satellites = []
        for item in data:
            if not isinstance(item, dict) or item.get("NORAD_CAT_ID") is None:
                continue
            satellites.append(slim_gp_record(item))
        if not satellites:
            raise ValueError(f"CelesTrak returned no GP records for group={group}")
        return satellites

    try:
        return _run_with_deadline(_download, seconds)
    except TimeoutError:
        print(f"CelesTrak GP deadline ({seconds:.0f}s) exceeded for group={group}")
        raise


def _refresh_satellites_uncached(group: str, force: bool = False, deadline: float = None):
    cached = _read_satellite_cache(group)
    if not force and cached and satellite_cache_is_fresh(cached):
        return _present_satellite_payload(cached, stale=False)

    try:
        satellites = fetch_celestrak_gp(group, deadline=deadline)
        payload = _satellite_payload(group, satellites, stale=False)
        _write_satellite_cache(group, payload)
        return payload
    except Exception as exc:
        print(f"Error fetching CelesTrak GP for {group}: {exc}")
        if cached and cached.get("satellites") is not None:
            return _present_satellite_payload(cached, stale=True)
        raise


def refresh_satellites_internal(
    group: str = "starlink",
    force: bool = False,
    wait_timeout: float = None,
    deadline: float = None,
):
    """Load cached GP or refresh from CelesTrak (single-flight per group)."""
    group = _normalize_satellite_group(group)

    def _do():
        result = _redis_single_flight(
            f"satellites_{group}",
            lambda: _refresh_satellites_uncached(group, force=force, deadline=deadline),
            wait_timeout=wait_timeout,
        )
        if result is None and wait_timeout is not None:
            cached = _read_satellite_cache(group)
            if cached and cached.get("satellites") is not None:
                return _present_satellite_payload(
                    cached, stale=not satellite_cache_is_fresh(cached)
                )
            raise TimeoutError(
                f"Satellite GP refresh for {group} exceeded the request budget"
            )
        return result

    return _get_satellite_flight(group).do(_do, wait_timeout=wait_timeout)


def _resolve_satellite_payload(group: str, force: bool = False):
    """Return ``(payload, cache_state)``.

    A stored catalog is returned immediately, fresh or stale. CelesTrak runs
    on the request only when the cache is empty (bounded) or ``force`` is set.
    """
    if not force:
        cached = _read_satellite_cache(group)
        if cached and cached.get("satellites") is not None:
            fresh = satellite_cache_is_fresh(cached)
            if not fresh:
                _schedule_satellite_refresh(group)
            state = "hit" if fresh else "stale"
            return _present_satellite_payload(cached, stale=not fresh), state
    payload = refresh_satellites_internal(
        group,
        force=force,
        wait_timeout=SATELLITES_REQUEST_BUDGET_SEC,
        deadline=SATELLITES_REQUEST_BUDGET_SEC,
    )
    if not isinstance(payload, dict) or payload.get("satellites") is None:
        raise TimeoutError(f"Satellite GP unavailable for group={group}")
    return payload, "miss"


def _satellite_meta_from_payload(group: str, payload):
    age = satellite_age_seconds(payload) if payload else None
    count = 0
    fetched_at = None
    stale = True
    if payload:
        fetched_at = payload.get("fetched_at")
        count = payload.get("count")
        if count is None:
            count = len(payload.get("satellites") or [])
        stale = not satellite_cache_is_fresh(payload)
    return {
        "group": group,
        "fetched_at": fetched_at,
        "count": count,
        "ttl_seconds": SATELLITES_CACHE_TTL,
        "age_seconds": None if age is None else int(age),
        "stale": stale,
        "source": "celestrak",
        "note": SATELLITE_NOTE,
        "allowed_groups": list(SATELLITE_GROUPS),
    }


@app.get("/launches")
def get_launches(
    force: bool = False,
    include_raw: bool = False,
    full: bool = False,
    slim: bool = False,
    internal: bool = False,
):
    """Launch list. Default payload is slim (no raw LL `all_data`).

    Matches what spacex-dashboard already does client-side (pop all_data).
    Convenience fields stay intact: mission, net, pad, video_url,
    trajectory_data on the next launch, etc.

    `?full=true` / `?include_raw=true` restore the legacy giant payload
    (LaunchBuddy fetchLaunchesFull). Raw blobs are hydrated from a side
    store for that response only — they are not kept in the list cache.
    `?slim=true` is an explicit alias for the default slim shape.
    """
    if not internal:
        increment_metric("total_requests")
    data = _load_launch_payload(force)
    if data and (data.get("upcoming") or data.get("previous")):
        increment_metric("cache_hits" if not force else "cache_misses")
    else:
        increment_metric("cache_misses")
    if full or include_raw:
        return _hydrate_launch_payload(data)
    return _slim_launch_payload(data, keep_next_trajectory=True)


@app.get("/launches_slim")
def get_launches_slim(force: bool = False, internal: bool = False):
    """Dashboard-optimized endpoint that strips 'all_data'."""
    if not internal:
        increment_metric("total_requests")
    data = _load_launch_payload(force)
    if data and (data.get("upcoming") or data.get("previous")):
        increment_metric("cache_hits" if not force else "cache_misses")
        return _slim_launch_payload(data, keep_next_trajectory=True)
    increment_metric("cache_misses")
    return {"upcoming": [], "previous": [], "last_updated": None}


@app.get("/launch_raw/{launch_id}")
def get_launch_raw(
    launch_id: str,
    hot: Annotated[
        bool,
        Query(
            description=(
                "If true, refresh the current/next launch from Launch Library on a "
                "~20s stale-while-revalidate TTL. Ignored for any other launch id."
            )
        ),
    ] = False,
    internal: bool = False,
):
    """Return the full Launch Library record for one launch.

    Default (no `hot`): leftover side-store / list blob, else one live LL GET.
    `?hot=1` on the current/next launch only: serve a ~20s hot cache, or one
    single-flight LL2 GET when that cache is stale. Other ids ignore `hot`.
    """
    if not internal:
        increment_metric("total_requests")
    if hot and _is_current_or_next_launch(launch_id):
        details, hit = _get_hot_launch_raw(launch_id)
    else:
        details, hit = _serve_cached_or_live_raw(launch_id)
    if hit:
        increment_metric("cache_hits")
    else:
        increment_metric("cache_misses")
    return details if details else {"error": "Launch not found"}


def _get_weather_cached(location: str, force: bool = False):
    """Internal helper to fetch weather with v2 caching metadata."""
    if not force:
        data = _read_weather_loc(location)
        if data:
            if not data.get('forecast'):
                data['forecast'] = fetch_forecast(location)
                data = _finalize_weather(data)
                _set_weather_loc(location, data)
            return data, True

    payload = refresh_weather_internal(force=force)
    loc_data = (payload or {}).get("weather", {}).get(location)
    if loc_data:
        return loc_data, False
    return {"error": f"Weather unavailable for {location}"}, False


@app.get("/weather/{location}")
def get_weather(location: str, force: bool = False, internal: bool = False):
    if not internal:
        increment_metric("total_requests")
    res, is_hit = _get_weather_cached(location, force)
    if is_hit:
        increment_metric("cache_hits")
    else:
        increment_metric("cache_misses")
    return res


@app.get("/user_weather")
def get_user_weather(lat: float, lon: float, station_id: str = None, internal: bool = False):
    """Retrieve weather (METAR + forecast) for a user-specified location."""
    if not internal:
        increment_metric("total_requests")

    # Always a cache miss as we don't cache user-specific locations by default
    increment_metric("cache_misses")

    # Fetch METAR
    weather_data = fetch_weather(station_id=station_id, lat=lat, lon=lon)

    # Fetch forecast
    forecast = fetch_forecast(lat=lat, lon=lon)
    weather_data['forecast'] = forecast
    weather_data['last_updated'] = _utc_isoformat()
    weather_data = _finalize_weather(weather_data)

    return weather_data


@app.get("/weather_all")
def get_all_weather(force: bool = False, internal: bool = False):
    if not internal:
        increment_metric("total_requests")
    if force:
        increment_metric("cache_misses")
        return refresh_weather_internal(force=True)

    weather_payload = _load_weather_all(force=False)
    locations = list(weather_payload.get("weather", {}).keys())
    hit_count = sum(1 for loc in locations if weather_payload["weather"].get(loc, {}).get("last_updated"))
    if locations and hit_count == len(locations):
        increment_metric("cache_hits")
    else:
        increment_metric("cache_misses")
    return weather_payload


@app.get(
    "/satellites/gp",
    tags=["Satellites"],
    summary="Cached CelesTrak GP / TLE for an allowlisted group",
)
def get_satellites_gp(
    group: str = Query("starlink", description="CelesTrak GROUP (allowlisted)"),
    force: bool = False,
    internal: bool = False,
    request: Request = None,
):
    """Return slim GP records for satellite.js / SGP4 (not live telemetry).

    Default `group=starlink`. Cached ~1 hour in Redis (in-memory fallback).
    A stale copy is returned immediately and refreshed in the background.
    CelesTrak is contacted on the request only when nothing is cached, and
    that fetch is capped so Heroku's router does not turn it into a 503.
    Clients that send ``Accept-Encoding: gzip`` receive a gzip body.
    """
    if not internal:
        increment_metric("total_requests")
    group = _normalize_satellite_group(group)
    if request is not None and not force:
        doc = _read_satellite_http(group)
        if doc is None:
            cached = _read_satellite_cache(group)
            if cached and cached.get("satellites") is not None:
                doc = _remember_satellite_http(group, cached)
        if doc is not None:
            fresh = satellite_cache_is_fresh(doc)
            if not fresh:
                _schedule_satellite_refresh(group)
            state = "hit" if fresh else "stale"
            increment_metric("cache_hits")
            return _encoded_satellite_response(
                doc, request, stale=not fresh, cache_state=state
            )
    try:
        payload, state = _resolve_satellite_payload(group, force=force)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Satellite GP unavailable for group '{group}': {exc}",
        )
    if state == "miss":
        increment_metric("cache_misses")
    else:
        increment_metric("cache_hits")
    if request is None:
        return payload
    doc = _local_satellite_http.get(group)
    if not isinstance(doc, dict) or not doc.get("gzip_fresh"):
        doc = _remember_satellite_http(group, payload)
    return _encoded_satellite_response(
        doc,
        request,
        stale=bool(payload.get("stale")),
        cache_state=state,
    )


@app.get(
    "/satellites/starlink",
    tags=["Satellites"],
    summary="Cached Starlink GP / TLE (alias for /satellites/gp?group=starlink)",
)
def get_satellites_starlink(
    force: bool = False,
    internal: bool = False,
    request: Request = None,
):
    """Convenience alias for the Starlink CelesTrak group."""
    return get_satellites_gp(
        group="starlink", force=force, internal=internal, request=request
    )


@app.get(
    "/satellites/meta",
    tags=["Satellites"],
    summary="Cache metadata for a satellite GP group",
)
def get_satellites_meta(
    group: str = Query("starlink", description="CelesTrak GROUP (allowlisted)"),
    internal: bool = False,
):
    """fetched_at / count / ttl without pulling the full GP list from CelesTrak."""
    if not internal:
        increment_metric("total_requests")
    group = _normalize_satellite_group(group)
    payload = _read_satellite_cache(group)
    if payload:
        increment_metric("cache_hits")
    else:
        increment_metric("cache_misses")
    return _satellite_meta_from_payload(group, payload)


def deployed_generation(text) -> Optional[str]:
    """`v3` when launch text contains `v3` or `group 31-` (Flight 14 / Group 31-1)."""
    blob = str(text or "").lower()
    if "v3" in blob or "group 31-" in blob:
        return "v3"
    return None


def _launch_blob(launch) -> str:
    if not isinstance(launch, dict):
        return ""
    return " ".join(
        str(launch.get(key) or "")
        for key in ("rocket", "mission", "name", "description")
    ).lower()


def _is_starship_launch_row(launch) -> bool:
    return "starship" in _launch_blob(launch)


def select_v3_starship(upcoming, previous=None, now=None):
    """Starship whose text marks V3 / Group 31, closest to ``now``.

    Upcoming and previous copies of the same flight are both eligible.
    A Starship that is not a V3 / Group 31 Starlink deployment is ignored
    so an older flight cannot select a Falcon launch date.
    """
    now_utc = now or datetime.now(timezone.utc)
    best = None
    best_delta = None
    for launch in list(upcoming or []) + list(previous or []):
        if not _is_starship_launch_row(launch):
            continue
        if deployed_generation(_launch_blob(launch)) != "v3":
            continue
        net = _parse_net_dt(launch.get("net"))
        if net is None:
            continue
        delta = abs((net - now_utc).total_seconds())
        if best is None or delta < best_delta:
            best = launch
            best_delta = delta
    return best


def _parse_launch_date_param(value) -> str:
    text = str(value or "").strip()
    if len(text) >= 10:
        text = text[:10]
    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="launch_date must be YYYY-MM-DD",
        )
    return text


def _norad_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _object_intdes(object_id):
    match = _INTDES_RE.match(str(object_id or "").strip())
    return match.group(1) if match else None


def _satcat_decayed(row) -> bool:
    return bool(str((row or {}).get("DECAY_DATE") or "").strip())


def slim_satcat_index(rows, *, now=None, extra_dates=()):
    """Keep non-decayed Starlink payloads, keyed by LAUNCH_DATE.

    Decayed objects, rocket bodies, and debris are dropped. Dates older than
    ``SATCAT_RECENT_DAYS`` are dropped unless listed in ``extra_dates`` (those
    keys are always present, even when empty, so a miss is not refetched).
    """
    if not isinstance(rows, list):
        raise ValueError("CelesTrak SATCAT JSON was not a list")
    now_utc = now or datetime.now(timezone.utc)
    cutoff = (now_utc.date() - timedelta(days=SATCAT_RECENT_DAYS)).isoformat()
    extras = {str(item)[:10] for item in extra_dates if item}
    dates = {extra: [] for extra in extras}
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        norad = _norad_int(row.get("NORAD_CAT_ID"))
        if norad is None:
            continue
        launch_date = str(row.get("LAUNCH_DATE") or "")[:10]
        if len(launch_date) != 10:
            continue
        if launch_date < cutoff and launch_date not in extras:
            continue
        if _satcat_decayed(row):
            continue
        obj_type = str(row.get("OBJECT_TYPE") or "").upper()
        if obj_type in ("R/B", "DEB"):
            continue
        name = str(row.get("OBJECT_NAME") or "").strip()
        if not name.upper().startswith("STARLINK"):
            continue
        key = (launch_date, norad)
        if key in seen:
            continue
        seen.add(key)
        dates.setdefault(launch_date, []).append({
            "name": name,
            "norad_id": norad,
            "object_id": str(row.get("OBJECT_ID") or ""),
        })
    for launch_date, items in dates.items():
        items.sort(key=lambda item: item.get("norad_id") or 0)
    return {"cutoff": cutoff, "dates": dates}


def _satcat_index_covers(index, launch_date) -> bool:
    if not isinstance(index, dict):
        return False
    cutoff = index.get("cutoff")
    dates = index.get("dates")
    if not cutoff or not isinstance(dates, dict):
        return False
    return launch_date >= cutoff or launch_date in dates


def _read_satcat_index():
    global _local_satcat_index
    local = _local_satcat_index
    if isinstance(local, dict) and isinstance(local.get("dates"), dict):
        return local
    cached = get_cached_data(SATCAT_RECENT_CACHE_KEY)
    if isinstance(cached, dict) and isinstance(cached.get("dates"), dict):
        _local_satcat_index = cached
        return cached
    return None


def _write_satcat_index(payload: dict):
    global _local_satcat_index
    _local_satcat_index = payload
    set_cached_data(SATCAT_RECENT_CACHE_KEY, payload, ttl=SATELLITES_STALE_TTL)


def fetch_starlink_satcat_rows() -> list:
    """Download CelesTrak SATCAT GROUP=starlink (JSON list)."""
    increment_metric("api_calls")
    response = requests.get(
        CELESTRAK_SATCAT_URL,
        params={"GROUP": "starlink", "FORMAT": "JSON"},
        headers=CELESTRAK_HEADERS,
        timeout=SATELLITES_FETCH_TIMEOUT,
    )
    response.raise_for_status()
    try:
        data = response.json()
    except ValueError as exc:
        raise ValueError("CelesTrak returned non-JSON SATCAT data") from exc
    if not isinstance(data, list):
        raise ValueError("CelesTrak SATCAT JSON was not a list")
    return data


def _load_or_fetch_satcat_index(wanted_date: str, force: bool = False):
    """Return ``(index, index_is_stale)``. Raise if SATCAT is unreachable and uncached."""
    cached = None if force else _read_satcat_index()
    if (
        cached
        and satellite_cache_is_fresh(cached)
        and _satcat_index_covers(cached, wanted_date)
    ):
        return cached, False
    try:
        rows = fetch_starlink_satcat_rows()
        slim = slim_satcat_index(rows, extra_dates=[wanted_date])
        payload = {
            "fetched_at": _utc_isoformat(),
            "cutoff": slim["cutoff"],
            "dates": slim["dates"],
        }
        _write_satcat_index(payload)
        return payload, False
    except Exception as exc:
        print(f"Error fetching CelesTrak SATCAT: {exc}")
        fallback = cached or _read_satcat_index()
        if fallback and _satcat_index_covers(fallback, wanted_date):
            return fallback, True
        raise


def _get_satcat_index(wanted_date: str, force: bool = False):
    if not force:
        cached = _read_satcat_index()
        if (
            cached
            and satellite_cache_is_fresh(cached)
            and _satcat_index_covers(cached, wanted_date)
        ):
            return cached, False

    def _do():
        result = _redis_single_flight(
            "satellites_satcat_starlink",
            lambda: _load_or_fetch_satcat_index(wanted_date, force=force),
        )
        if result is None:
            fallback = _read_satcat_index()
            if fallback and _satcat_index_covers(fallback, wanted_date):
                return fallback, True
            raise RuntimeError("CelesTrak SATCAT unavailable")
        return result

    return _satcat_flight.do(_do)


def _catalog_rows_for_date(index, launch_date) -> list:
    dates = (index or {}).get("dates") or {}
    rows = list(dates.get(launch_date) or [])
    rows.sort(key=lambda row: _norad_int(row.get("norad_id")) or 0)
    return rows


def _cached_starlink_tle_index() -> dict:
    """NORAD → compact TLE record from the existing Starlink GP cache, if any."""
    payload = _read_satellite_cache("starlink")
    index = {}
    for sat in (payload or {}).get("satellites") or []:
        if not isinstance(sat, dict):
            continue
        norad = _norad_int(sat.get("norad_id"))
        if norad is None or not sat.get("tle_line1") or not sat.get("tle_line2"):
            continue
        index[norad] = {
            "name": sat.get("name") or "",
            "norad_id": norad,
            "tle_line1": sat["tle_line1"],
            "tle_line2": sat["tle_line2"],
        }
    return index


def _fetch_gp_omm_list(url: str, params: dict) -> list:
    increment_metric("api_calls")
    response = requests.get(
        url,
        params=params,
        headers=CELESTRAK_HEADERS,
        timeout=SATELLITES_FETCH_TIMEOUT,
    )
    response.raise_for_status()
    try:
        data = response.json()
    except ValueError as exc:
        raise ValueError("CelesTrak returned non-JSON GP data") from exc
    if not isinstance(data, list):
        raise ValueError("CelesTrak GP JSON was not a list")
    return [item for item in data if isinstance(item, dict)]


def fetch_gp_by_intdes(intdes: str, needed_norads) -> dict:
    """Best OMM per NORAD for one launch designator.

    Main GP is enough when it already covers ``needed_norads``. Supplemental
    GP is only fetched for catalog numbers the main set does not have.
    """
    needed = {n for n in (_norad_int(item) for item in needed_norads) if n is not None}
    best = {}
    for url in (CELESTRAK_GP_URL, CELESTRAK_SUP_GP_URL):
        if needed and needed <= set(best):
            break
        try:
            rows = _fetch_gp_omm_list(url, {"INTDES": intdes, "FORMAT": "JSON"})
        except Exception as exc:
            print(f"CelesTrak GP INTDES {intdes} failed for {url}: {exc}")
            continue
        for omm in rows:
            norad = _norad_int(omm.get("NORAD_CAT_ID"))
            if norad is None:
                continue
            prev = best.get(norad)
            if prev is None or str(omm.get("EPOCH") or "") > str(prev.get("EPOCH") or ""):
                best[norad] = omm
    return best


def _join_deployed_tles(catalog_rows, generation) -> list:
    """Real GP TLEs for SATCAT rows only. Catalog misses stay unmatched."""
    index = _cached_starlink_tle_index()
    missing_by_intdes = {}
    for row in catalog_rows:
        norad = _norad_int(row.get("norad_id"))
        if norad is None:
            continue
        if norad in index:
            continue
        intdes = _object_intdes(row.get("object_id"))
        if not intdes:
            continue
        missing_by_intdes.setdefault(intdes, []).append(norad)

    ordered = sorted(missing_by_intdes.items(), key=lambda item: len(item[1]), reverse=True)
    for intdes, norads in ordered[:DEPLOYED_INTDES_CAP]:
        found = fetch_gp_by_intdes(intdes, norads)
        allowed = set(norads)
        for norad, omm in found.items():
            if norad not in allowed or norad in index:
                continue
            slim = slim_gp_record(omm)
            if slim.get("tle_line1") and slim.get("tle_line2"):
                index[norad] = slim

    satellites = []
    for row in catalog_rows:
        norad = _norad_int(row.get("norad_id"))
        sat = index.get(norad) if norad is not None else None
        if not sat or not sat.get("tle_line1") or not sat.get("tle_line2"):
            continue
        record = {
            "name": sat.get("name") or row.get("name") or "",
            "norad_id": norad,
            "tle_line1": sat["tle_line1"],
            "tle_line2": sat["tle_line2"],
        }
        if generation:
            record["generation"] = generation
        satellites.append(record)
        if len(satellites) >= DEPLOYED_SAT_CAP:
            break
    return satellites


def _gmst_rad(dt: datetime) -> float:
    """Greenwich mean sidereal time, radians. Matches the dashboard's Vallado form."""
    dt = dt.astimezone(timezone.utc)
    y, m = dt.year, dt.month
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    day = (
        dt.day
        + (dt.hour + dt.minute / 60.0 + dt.second / 3600.0 + dt.microsecond / 3.6e9) / 24.0
    )
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + day + b - 1524.5
    tut1 = (jd - 2451545.0) / 36525.0
    gmst_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * tut1
        + 0.093104 * tut1 * tut1
        - 6.2e-6 * tut1 * tut1 * tut1
    )
    return math.radians((gmst_sec % 86400.0) / 240.0)


def _teme_to_ecef(r, dt: datetime):
    """Rotate an inertial position into ECEF. z is the Earth axis, so it is unchanged."""
    theta = _gmst_rad(dt)
    c, s = math.cos(theta), math.sin(theta)
    x, y, z = r
    return (c * x + s * y, -s * x + c * y, z)


def _ecef_to_geodetic(x: float, y: float, z: float):
    """WGS84 latitude (deg), longitude (deg), altitude (km)."""
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - _WGS84_E2))
    alt = 0.0
    for _ in range(6):
        sin_lat = math.sin(lat)
        n = _WGS84_A_KM / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        cos_lat = math.cos(lat)
        if abs(cos_lat) < 1e-8:
            alt = abs(z) - _WGS84_A_KM * (1.0 - 1.0 / 298.257223563)
        else:
            alt = p / cos_lat - n
        lat = math.atan2(z, p * (1.0 - _WGS84_E2 * n / (n + alt)))
    return (
        math.degrees(lat),
        (math.degrees(lon) + 180.0) % 360.0 - 180.0,
        alt,
    )


def _inertial_to_kepler(r, v) -> Optional[dict]:
    """Osculating Keplerian elements from an inertial state (km, km/s)."""
    rx, ry, rz = r
    vx, vy, vz = v
    rmag = math.sqrt(rx * rx + ry * ry + rz * rz)
    vmag2 = vx * vx + vy * vy + vz * vz
    if rmag < 6000.0 or vmag2 <= 0.0:
        return None
    hx = ry * vz - rz * vy
    hy = rz * vx - rx * vz
    hz = rx * vy - ry * vx
    hmag = math.sqrt(hx * hx + hy * hy + hz * hz)
    if hmag < 1e-6:
        return None
    nx, ny = -hy, hx
    nmag = math.hypot(nx, ny)
    rdotv = rx * vx + ry * vy + rz * vz
    mu_over_r = _EARTH_MU_KM3_S2 / rmag
    ex = ((vmag2 - mu_over_r) * rx - rdotv * vx) / _EARTH_MU_KM3_S2
    ey = ((vmag2 - mu_over_r) * ry - rdotv * vy) / _EARTH_MU_KM3_S2
    ez = ((vmag2 - mu_over_r) * rz - rdotv * vz) / _EARTH_MU_KM3_S2
    ecc = math.sqrt(ex * ex + ey * ey + ez * ez)
    energy = vmag2 / 2.0 - _EARTH_MU_KM3_S2 / rmag
    if energy >= 0.0 or ecc >= 0.9:
        return None
    sma = -_EARTH_MU_KM3_S2 / (2.0 * energy)
    inc = math.acos(max(-1.0, min(1.0, hz / hmag)))
    if nmag < 1e-8:
        raan = 0.0
    else:
        raan = math.acos(max(-1.0, min(1.0, nx / nmag)))
        if ny < 0.0:
            raan = 2.0 * math.pi - raan
    if nmag < 1e-8 or ecc < 1e-8:
        argp = 0.0
    else:
        argp = math.acos(max(-1.0, min(1.0, (nx * ex + ny * ey) / (nmag * ecc))))
        if ez < 0.0:
            argp = 2.0 * math.pi - argp
    if ecc < 1e-8:
        nu = 0.0
    else:
        nu = math.acos(max(-1.0, min(1.0, (ex * rx + ey * ry + ez * rz) / (ecc * rmag))))
        if rdotv < 0.0:
            nu = 2.0 * math.pi - nu
    cos_e = (ecc + math.cos(nu)) / (1.0 + ecc * math.cos(nu))
    sin_e = (
        math.sqrt(max(0.0, 1.0 - ecc * ecc)) * math.sin(nu)
        / (1.0 + ecc * math.cos(nu))
    )
    ecc_anom = math.atan2(sin_e, cos_e)
    mean_anom = (ecc_anom - ecc * math.sin(ecc_anom)) % (2.0 * math.pi)
    mean_motion = math.sqrt(_EARTH_MU_KM3_S2 / sma ** 3) * 86400.0 / (2.0 * math.pi)
    return {
        "ecc": ecc,
        "inc_deg": math.degrees(inc) % 360.0,
        "raan_deg": math.degrees(raan) % 360.0,
        "argp_deg": math.degrees(argp) % 360.0,
        "mean_anom_deg": math.degrees(mean_anom) % 360.0,
        "mean_motion": mean_motion,
    }


def _parse_meme_stamp(text: str):
    match = _MEME_STAMP_RE.search(text or "")
    if not match:
        return None
    try:
        return datetime.strptime(
            f"{match.group(1)} {match.group(2)}", "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_meme_epoch(token: str):
    """Packed ``YYYYDDDHHMMSS.fff`` epoch from a MEME state row."""
    whole, _, frac = str(token).partition(".")
    if len(whole) < 13:
        return None
    try:
        year = int(whole[0:4])
        doy = int(whole[4:7])
        hh = int(whole[7:9])
        mm = int(whole[9:11])
        ss = int(whole[11:13])
        micro = int((frac + "000000")[:6]) if frac else 0
    except ValueError:
        return None
    if not (1 <= doy <= 366 and 0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 60):
        return None
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=doy - 1, hours=hh, minutes=mm, seconds=ss, microseconds=micro
    )


def manifest_starlink_filenames(text: str) -> list:
    """Post-Falcon STARLINK filenames in MANIFEST order, one per id.

    The manifest lists the whole constellation. Only ids at or above
    ``SPACEX_POST_FALCON_STARLINK_ID_MIN`` are candidates; the ephemeris
    window is checked after the file header is read.
    """
    found = {}
    for match in _MANIFEST_NAME_RE.finditer(text or ""):
        filename, sid_text = match.group(1), match.group(2)
        try:
            sid = int(sid_text)
        except ValueError:
            continue
        if sid < SPACEX_POST_FALCON_STARLINK_ID_MIN:
            continue
        found[sid] = filename
    return [(sid, found[sid]) for sid in sorted(found)]


def _meme_window_covers(header: dict, launch_date: str, when: datetime = None) -> bool:
    """True when this MEME file belongs to the launch.

    The published window must include the launch calendar day, or it must have
    rolled forward (start on or after the launch day) and still cover ``when``.
    Files that start before the launch day are not this deployment.
    """
    try:
        day = datetime.strptime(launch_date, "%Y-%m-%d").date()
    except ValueError:
        return False
    start, stop = header.get("start"), header.get("stop")
    if start is not None and stop is not None:
        if start.date() <= day <= stop.date():
            return True
        moment = None
        if when is not None:
            aware = _as_utc(when)
            if aware is not None:
                moment = aware.date()
        roll_end = day + timedelta(days=MEME_ROLL_FORWARD_DAYS)
        if (
            moment is not None
            and day <= start.date() <= roll_end
            and start.date() <= moment <= stop.date()
        ):
            return True
        return False
    created = header.get("created")
    return bool(created and created.date() == day)


def _lerp_meme_state(prev: dict, nxt: dict, when: datetime) -> dict:
    span = (nxt["t"] - prev["t"]).total_seconds()
    if span <= 0.0:
        return prev
    frac = max(0.0, min(1.0, (when - prev["t"]).total_seconds() / span))
    rv = tuple(prev["rv"][i] + (nxt["rv"][i] - prev["rv"][i]) * frac for i in range(6))
    return {"t": when, "rv": rv}


def _select_meme_state(prev, nxt, when: datetime):
    """Interpolate inside the table. One step outside uses the nearest row."""
    if prev is not None and nxt is not None:
        return _lerp_meme_state(prev, nxt, when)
    if prev is not None and 0.0 <= (when - prev["t"]).total_seconds() <= 90.0:
        return prev
    if nxt is not None and 0.0 <= (nxt["t"] - when).total_seconds() <= 90.0:
        return nxt
    return None


def meme_state_from_lines(lines, launch_date: str, when: datetime):
    """Inertial state at ``when`` from a MEME file, or None if it does not apply.

    State rows are ``YYYYDDDHHMMSS x y z vx vy vz`` in km and km/s, inertial.
    The following three lines are a 6×6 UVW covariance and are ignored.
    Reading stops once a sample is past ``when``.
    """
    header = {"created": None, "start": None, "stop": None}
    prev = None
    nxt = None
    for raw in lines:
        if raw is None:
            continue
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        line = str(raw).strip()
        if not line:
            continue
        if line.startswith("created:"):
            header["created"] = _parse_meme_stamp(line)
            continue
        if line.startswith("ephemeris_start:"):
            stamps = _MEME_STAMP_RE.findall(line)
            if stamps:
                header["start"] = _parse_meme_stamp(" ".join(stamps[0]))
            if len(stamps) > 1:
                header["stop"] = _parse_meme_stamp(" ".join(stamps[1]))
            # Reject before the state table so a streamed download can close.
            if header["start"] is not None and header["stop"] is not None:
                if not _meme_window_covers(header, launch_date, when):
                    return None
            continue
        match = _MEME_EPOCH_RE.match(line)
        if not match:
            continue
        stamp = _parse_meme_epoch(match.group("epoch"))
        if stamp is None:
            continue
        try:
            rv = tuple(float(match.group(key)) for key in ("x", "y", "z", "vx", "vy", "vz"))
        except ValueError:
            continue
        sample = {"t": stamp, "rv": rv}
        if stamp <= when:
            prev = sample
        else:
            nxt = sample
            break
    if not _meme_window_covers(header, launch_date, when):
        return None
    return _select_meme_state(prev, nxt, when)


def _lines_from_response(response):
    """Yield text lines. A test double may only set ``.text``."""
    iterator = getattr(response, "iter_lines", None)
    if callable(iterator):
        try:
            stream = iterator(decode_unicode=True)
            for line in stream:
                if isinstance(line, bytes):
                    yield line.decode("utf-8", "replace")
                elif isinstance(line, str):
                    yield line
                elif line is None:
                    continue
                else:
                    raise TypeError("non-text ephemeris line")
            return
        except TypeError:
            pass
    text = getattr(response, "text", None)
    if isinstance(text, str):
        yield from text.splitlines()


def _as_utc(value):
    """Aware UTC datetime, or None.

    Duck-typed so a test can replace the ``datetime`` name without dropping
    timestamps that were built from the real class.
    """
    try:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def _satellite_from_inertial_state(starlink_id: int, state: dict, generation):
    """Geodetic position plus an osculating TLE. No NORAD id is assigned."""
    rv = state.get("rv")
    when = _as_utc(state.get("t"))
    if not rv or len(rv) != 6 or when is None:
        return None
    r = rv[0:3]
    v = rv[3:6]
    elements = _inertial_to_kepler(r, v)
    if not elements:
        return None
    ecef = _teme_to_ecef(r, when)
    lat, lon, alt_km = _ecef_to_geodetic(*ecef)
    if not (100.0 <= alt_km <= 2500.0):
        return None
    # Catalog number 0 is only a TLE-format placeholder. It is not published
    # as norad_id — SpaceX has not been assigned one yet.
    omm = {
        "NORAD_CAT_ID": 0,
        "OBJECT_ID": "",
        "EPOCH": when.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "MEAN_MOTION": elements["mean_motion"],
        "ECCENTRICITY": elements["ecc"],
        "INCLINATION": elements["inc_deg"],
        "RA_OF_ASC_NODE": elements["raan_deg"],
        "ARG_OF_PERICENTER": elements["argp_deg"],
        "MEAN_ANOMALY": elements["mean_anom_deg"],
        "BSTAR": 0.0,
        "MEAN_MOTION_DOT": 0.0,
        "MEAN_MOTION_DDOT": 0.0,
        "EPHEMERIS_TYPE": 0,
        "CLASSIFICATION_TYPE": "U",
        "ELEMENT_SET_NO": 999,
        "REV_AT_EPOCH": 0,
    }
    line1, line2 = omm_to_tle_lines(omm)
    if not line1 or not line2:
        return None
    name = f"STARLINK-{int(starlink_id)}"
    record = {
        "name": name,
        "id": name,
        "lat": round(lat, 5),
        "lon": round(lon, 5),
        "alt_km": round(alt_km, 3),
        "epoch": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tle_line1": line1,
        "tle_line2": line2,
    }
    if generation:
        record["generation"] = generation
    return record


def _download_manifest_text() -> str:
    increment_metric("api_calls")
    response = requests.get(
        SPACEX_MANIFEST_URL,
        headers=SPACEX_EPHEM_HEADERS,
        timeout=SPACEX_EPHEM_FETCH_TIMEOUT,
    )
    response.raise_for_status()
    return response.text or ""


def _download_meme_satellite(filename, starlink_id, launch_date, when, generation):
    increment_metric("api_calls")
    response = requests.get(
        SPACEX_EPHEM_BASE + filename,
        headers=SPACEX_EPHEM_HEADERS,
        timeout=SPACEX_EPHEM_FETCH_TIMEOUT,
        stream=True,
    )
    try:
        response.raise_for_status()
        state = meme_state_from_lines(
            _lines_from_response(response), launch_date, when
        )
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()
    if not state:
        return None
    return _satellite_from_inertial_state(starlink_id, state, generation)


def _manifest_satellites_for_launch(launch_date: str, generation, when: datetime = None) -> list:
    """Download and parse post-Falcon MEME files whose window covers ``launch_date``."""
    when = when or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    candidates = manifest_starlink_filenames(_download_manifest_text())
    if not candidates:
        return []
    satellites = []
    workers = min(SPACEX_EPHEM_CONCURRENCY, len(candidates))
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = [
        pool.submit(
            _download_meme_satellite, filename, sid, launch_date, when, generation
        )
        for sid, filename in candidates
    ]
    try:
        for future in as_completed(futures, timeout=SPACEX_EPHEM_BUDGET_SEC):
            try:
                record = future.result()
            except Exception as exc:
                print(f"SpaceX ephemeris file failed: {exc}")
                continue
            if record:
                satellites.append(record)
    except TimeoutError:
        pending = [item for item in futures if not item.done()]
        print(
            f"SpaceX ephemeris budget exceeded with {len(pending)} file(s) still running"
        )
        if not satellites:
            raise
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    satellites.sort(key=lambda record: record.get("id") or "")
    return satellites


def _deployed_note_empty(launch_date: str) -> str:
    return (
        f"No Starlink SATCAT objects with LAUNCH_DATE {launch_date} yet. "
        + SATELLITE_NOTE
    )


def _deployed_note_no_tle(launch_date: str, catalog_count: int) -> str:
    return (
        f"SATCAT lists {catalog_count} Starlink objects for LAUNCH_DATE {launch_date} "
        "but no GP TLE is available yet. "
        + SATELLITE_NOTE
    )


def _deployed_note_ready(launch_date: str) -> str:
    return (
        f"Starlink SATCAT objects with LAUNCH_DATE {launch_date}, joined to CelesTrak GP TLEs. "
        + SATELLITE_NOTE
    )


def _deployed_note_unavailable() -> str:
    return "CelesTrak SATCAT unavailable. " + SATELLITE_NOTE


def _deployed_note_manifest(launch_date: str) -> str:
    return (
        f"No Starlink SATCAT objects with LAUNCH_DATE {launch_date} yet. "
        "Positions are interpolated from SpaceX public MEME ephemerides "
        "(api.starlink.com/public-files/ephemerides). "
        "tle_line1/tle_line2 are osculating elements fitted to that inertial "
        "state so the globe can coast them; they are not CelesTrak catalog numbers."
    )


def _deployed_ttl_seconds(payload: dict) -> int:
    source = (payload or {}).get("source")
    sats = (payload or {}).get("satellites") or []
    if source == DEPLOYED_MANIFEST_SOURCE and sats:
        return DEPLOYED_MANIFEST_TTL
    if not sats:
        return DEPLOYED_EMPTY_TTL
    return SATELLITES_CACHE_TTL


def _deployed_cache_is_fresh(payload) -> bool:
    """Freshness by source. An empty SATCAT cache from before the manifest
    check must not block the ephemeris fallback.
    """
    if not isinstance(payload, dict):
        return False
    age = satellite_age_seconds(payload)
    if age is None:
        return False
    sats = payload.get("satellites") or []
    if not sats and not payload.get("manifest_checked"):
        return False
    return age < _deployed_ttl_seconds(payload)


def _no_v3_launch_payload() -> dict:
    return {
        "launch_date": None,
        "generation": None,
        "mission": None,
        "fetched_at": _utc_isoformat(),
        "ttl_seconds": SATELLITES_CACHE_TTL,
        "count": 0,
        "catalog_count": 0,
        "stale": False,
        "empty": True,
        "source": DEPLOYED_SOURCE,
        "note": (
            "No Starship V3 or Group 31 launch is in the launch cache, "
            "so no SATCAT launch date was queried. "
            + SATELLITE_NOTE
        ),
        "satellites": [],
    }


def _deployed_payload(
    target, catalog_rows, satellites, *, stale: bool, note: str, source: str = None
) -> dict:
    sats = list(satellites or [])
    body = {
        "launch_date": target.get("launch_date"),
        "generation": target.get("generation"),
        "mission": target.get("mission"),
        "fetched_at": _utc_isoformat(),
        "ttl_seconds": SATELLITES_CACHE_TTL,
        "count": len(sats),
        "catalog_count": len(catalog_rows or []),
        "stale": bool(stale),
        "empty": len(sats) == 0,
        "source": source or DEPLOYED_SOURCE,
        "note": note,
        "satellites": sats,
    }
    body["ttl_seconds"] = _deployed_ttl_seconds(body)
    return body


def _mark_deployed_cached(payload: dict, *, stale: bool) -> dict:
    body = dict(payload)
    body["stale"] = stale
    body["empty"] = not body.get("satellites")
    body["count"] = len(body.get("satellites") or [])
    if body.get("catalog_count") is None:
        body["catalog_count"] = body["count"]
    body["source"] = body.get("source") or DEPLOYED_SOURCE
    body["ttl_seconds"] = _deployed_ttl_seconds(body)
    return body


def _deployed_cache_key(launch_date: str) -> str:
    return f"{DEPLOYED_CACHE_PREFIX}{launch_date}"


def _read_deployed_cache(launch_date: str):
    local = _local_deployed.get(launch_date)
    if isinstance(local, dict) and local.get("satellites") is not None:
        return local
    cached = get_cached_data(_deployed_cache_key(launch_date))
    if isinstance(cached, dict) and cached.get("satellites") is not None:
        _local_deployed[launch_date] = cached
        return cached
    return None


def _write_deployed_cache(launch_date: str, payload: dict):
    _local_deployed[launch_date] = payload
    set_cached_data(_deployed_cache_key(launch_date), payload, ttl=SATELLITES_STALE_TTL)


def _get_deployed_flight(launch_date: str) -> _SingleFlight:
    with _deployed_flights_lock:
        flight = _deployed_flights.get(launch_date)
        if flight is None:
            flight = _SingleFlight()
            _deployed_flights[launch_date] = flight
        return flight


def _resolve_deployed_target(launch_date_param):
    """Date, generation, and mission for this request. ``query`` is false when no V3 flight is known."""
    explicit = _parse_launch_date_param(launch_date_param) if launch_date_param else None
    data = _load_launch_payload(False) or {}
    upcoming = data.get("upcoming") or []
    previous = data.get("previous") or []
    if explicit:
        mission = None
        generation = None
        for launch in list(upcoming) + list(previous):
            if not isinstance(launch, dict) or not _is_starship_launch_row(launch):
                continue
            net = _parse_net_dt(launch.get("net"))
            if net is None or net.date().isoformat() != explicit:
                continue
            gen = deployed_generation(_launch_blob(launch))
            mission = launch.get("mission") or launch.get("name")
            generation = gen
            if gen == "v3":
                break
        return {
            "launch_date": explicit,
            "generation": generation,
            "mission": mission,
            "query": True,
        }
    chosen = select_v3_starship(upcoming, previous)
    if chosen is None:
        return {
            "launch_date": None,
            "generation": None,
            "mission": None,
            "query": False,
        }
    net = _parse_net_dt(chosen.get("net"))
    return {
        "launch_date": net.date().isoformat() if net else None,
        "generation": "v3",
        "mission": chosen.get("mission") or chosen.get("name"),
        "query": net is not None,
    }


def _refresh_deployed_uncached(target: dict, force: bool = False):
    launch_date = target["launch_date"]
    cached = None if force else _read_deployed_cache(launch_date)
    if cached and _deployed_cache_is_fresh(cached):
        return _mark_deployed_cached(cached, stale=False)

    satcat_failed = False
    index = None
    index_stale = False
    try:
        index, index_stale = _get_satcat_index(launch_date, force=force)
    except Exception as exc:
        print(f"Deployed SATCAT refresh failed for {launch_date}: {exc}")
        satcat_failed = True

    # A failed SATCAT refresh must not rebuild (and drop) TLEs we already stored.
    if (
        not satcat_failed
        and index_stale
        and cached
        and (cached.get("satellites") or [])
    ):
        return _mark_deployed_cached(cached, stale=True)

    catalog_rows = [] if satcat_failed else _catalog_rows_for_date(index, launch_date)
    satellites = []
    if catalog_rows:
        satellites = _join_deployed_tles(catalog_rows, target.get("generation"))
    if satellites:
        payload = _deployed_payload(
            target,
            catalog_rows,
            satellites,
            stale=index_stale,
            note=_deployed_note_ready(launch_date),
        )
        if index_stale:
            payload["fetched_at"] = index.get("fetched_at") or payload["fetched_at"]
            payload["stale"] = True
            _local_deployed[launch_date] = payload
            return payload
        _write_deployed_cache(launch_date, payload)
        return payload

    if satcat_failed and cached and (cached.get("satellites") or []):
        return _mark_deployed_cached(cached, stale=True)

    # SATCAT has nothing plottable yet. SpaceX public ephemerides cover the
    # gap until the catalog lists this launch date.
    manifest_failed = False
    manifest_sats = []
    try:
        manifest_sats = _manifest_satellites_for_launch(
            launch_date, target.get("generation")
        )
    except Exception as exc:
        print(f"Deployed SpaceX MANIFEST refresh failed for {launch_date}: {exc}")
        manifest_failed = True
    if manifest_sats:
        payload = _deployed_payload(
            target,
            manifest_sats,
            manifest_sats,
            stale=False,
            note=_deployed_note_manifest(launch_date),
            source=DEPLOYED_MANIFEST_SOURCE,
        )
        payload["manifest_checked"] = True
        _write_deployed_cache(launch_date, payload)
        return payload

    if satcat_failed:
        payload = _deployed_payload(
            target,
            [],
            [],
            stale=True,
            note=_deployed_note_unavailable(),
        )
        payload["manifest_checked"] = True
        _write_deployed_cache(launch_date, payload)
        return payload

    if not catalog_rows:
        note = _deployed_note_empty(launch_date)
    else:
        note = _deployed_note_no_tle(launch_date, len(catalog_rows))
    payload = _deployed_payload(
        target,
        catalog_rows,
        [],
        stale=bool(index_stale or manifest_failed),
        note=note,
    )
    payload["manifest_checked"] = True
    if index_stale:
        payload["fetched_at"] = (index or {}).get("fetched_at") or payload["fetched_at"]
        payload["stale"] = True
        _local_deployed[launch_date] = payload
        return payload
    _write_deployed_cache(launch_date, payload)
    return payload


def refresh_deployed_satellites_internal(launch_date: str = None, force: bool = False):
    """Load the deployed-sat feed for the V3 Starship date (single-flight)."""
    target = _resolve_deployed_target(launch_date)
    if not target.get("query") or not target.get("launch_date"):
        return _no_v3_launch_payload()
    date = target["launch_date"]

    def _do():
        result = _redis_single_flight(
            f"satellites_deployed_{date}",
            lambda: _refresh_deployed_uncached(target, force=force),
        )
        if result is None:
            cached = _read_deployed_cache(date)
            if cached:
                return _mark_deployed_cached(cached, stale=not satellite_cache_is_fresh(cached))
            return _deployed_payload(target, [], [], stale=True, note=_deployed_note_unavailable())
        return result

    return _get_deployed_flight(date).do(_do)


@app.get(
    "/satellites/deployed",
    tags=["Satellites"],
    summary="Starlink sats deployed by the recent Starship V3 flight",
)
def get_satellites_deployed(
    launch_date: Optional[str] = Query(
        None,
        description=(
            "UTC calendar date YYYY-MM-DD. Omit to use the Starship V3 / "
            "Group 31 NET date from the launch cache."
        ),
    ),
    force: bool = False,
    internal: bool = False,
):
    """Deployed Starlink TLEs for the globe's red V3 vertices.

    spacex-dashboard ``ingestDeployedPayload`` reads:

    - ``satellites[]``: ``name``, ``tle_line1``, ``tle_line2`` (satellite.js).
      Rows without both TLE lines are ignored. SATCAT rows also include
      ``norad_id``. Manifest rows use the SpaceX ``STARLINK-`` id and do not
      invent a NORAD catalog number; they add ``lat``, ``lon``, ``alt_km``.
    - ``catalog_count``: non-decayed Starlink SATCAT rows for ``launch_date``
      before the GP join, or the ephemeris count when SATCAT is empty.
      Greater than zero with an empty ``satellites`` list means the catalog
      matched and no TLE was available.
    - ``note``: contains ``unavailable`` only when CelesTrak SATCAT could not
      be fetched and no cached catalog exists.
    - ``source``: ``celestrak-satcat`` when SATCAT has TLEs, otherwise
      ``spacex-manifest-ephemeris`` when public MEME files cover the date.
    - ``empty`` / ``stale``: honest empty catalog vs. a stale cached copy.

    The default date is the UTC NET date of the Starship launch whose mission
    text contains ``v3`` or ``group 31-``. Same-day Starlink SATCAT rows are
    included. Decayed objects are not. Positions are never invented: if SATCAT
    and the SpaceX manifest are both empty, ``satellites`` is ``[]``.
    """
    if not internal:
        increment_metric("total_requests")
    # Direct calls (tests, the background worker) pass None. A FastAPI Query
    # default object is not a date string.
    if not isinstance(launch_date, str) or not launch_date.strip():
        launch_date = None
    cached_before = None
    try:
        target = _resolve_deployed_target(launch_date)
        if target.get("query") and target.get("launch_date") and not force:
            cached_before = _read_deployed_cache(target["launch_date"])
        payload = refresh_deployed_satellites_internal(launch_date, force=force)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Deployed satellite feed failed: {exc}",
        )
    serving_fresh = bool(
        cached_before and _deployed_cache_is_fresh(cached_before) and not force
    )
    if serving_fresh:
        increment_metric("cache_hits")
    else:
        increment_metric("cache_misses")
    if isinstance(payload, dict):
        payload = dict(payload)
        payload.pop("manifest_checked", None)
    return payload


def _parse_net_dt(net_str):
    """Parse a launch NET string into an aware UTC datetime."""
    if not net_str:
        return None
    try:
        text = str(net_str).strip()
        if text.endswith('Z'):
            text = text[:-1] + '+00:00'
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _resolve_tz(tz_name: str = None, location: str = None):
    """Resolve a pytz timezone from an IANA name or dashboard location."""
    if not tz_name and location and location in DASHBOARD_LOCATIONS:
        tz_name = DASHBOARD_LOCATIONS[location]['timezone']
    if tz_name:
        try:
            return pytz.timezone(tz_name)
        except Exception:
            pass
    return pytz.UTC


def is_launch_finished(status):
    """True when a launch status indicates success/failure/complete."""
    if not status:
        return False
    s = str(status).lower()
    return any(keyword in s for keyword in ('success', 'failure', 'successful', 'complete'))


def _slim_launch(launch, keep_trajectory=False):
    if not isinstance(launch, dict):
        return launch
    skip = {'all_data'}
    if not keep_trajectory:
        skip.add('trajectory_data')
    return {k: v for k, v in launch.items() if k not in skip}


def _repair_mismatched_trajectory(data):
    """Regenerate a cached trajectory whose origin does not match the pad.

    Flight 14 was stored with an LC-39A launch site while its pad is OLP-2.
    Slim responses serve that blob until the next launch refresh, so correct
    it on read and persist the Starbase track.
    """
    if not isinstance(data, dict):
        return False
    upcoming = data.get('upcoming') if isinstance(data.get('upcoming'), list) else []
    previous = data.get('previous') if isinstance(data.get('previous'), list) else []
    bucket = upcoming or previous
    if not bucket or not isinstance(bucket[0], dict):
        return False
    launch = bucket[0]
    traj = launch.get('trajectory_data')
    if not isinstance(traj, dict):
        return False
    pad, lat, lon, loc = _pad_fields_from_launch(launch)
    expected, _key = resolve_launch_site(
        pad, latitude=lat, longitude=lon, location_name=loc
    )
    if not expected:
        return False
    current = traj.get('launch_site') if isinstance(traj.get('launch_site'), dict) else None
    origin = None
    points = traj.get('trajectory')
    if isinstance(points, list) and points and isinstance(points[0], dict):
        origin = points[0]
    if not _coords_far(current, expected) and not _coords_far(origin, expected):
        return False
    new_traj = get_launch_trajectory_data(launch, previous)
    if not new_traj:
        return False
    launch['trajectory_data'] = new_traj
    logger.info(
        f"Regenerated trajectory for {launch.get('mission') or launch.get('id')} "
        f"at {new_traj.get('launch_site')}"
    )
    return True


def _load_launch_payload(force=False):
    now = time.time()
    if not force and _launches_mem["data"] is not None and (now - _launches_mem["at"]) < LAUNCHES_MEM_TTL:
        return _launches_mem["data"]
    if force:
        data = refresh_launches_internal()
    else:
        data = get_cached_data(LAUNCHES_CACHE_KEY)
        if data:
            _sanitize_launch_list_cache(data, persist=True)
    if not data:
        return {"upcoming": [], "previous": [], "last_updated": None}
    if _repair_mismatched_trajectory(data):
        set_cached_data(LAUNCHES_CACHE_KEY, data)
    return _remember_launches(data)


def _slim_launch_payload(data, keep_next_trajectory=True):
    upcoming = data.get("upcoming", []) or []
    previous = data.get("previous", []) or []
    return {
        "upcoming": [
            _slim_launch(launch, keep_trajectory=(keep_next_trajectory and i == 0))
            for i, launch in enumerate(upcoming)
        ],
        "previous": [
            _slim_launch(
                launch,
                keep_trajectory=(keep_next_trajectory and not upcoming and i == 0),
            )
            for i, launch in enumerate(previous)
        ],
        "last_updated": data.get("last_updated"),
    }


def _load_narratives(force=False):
    if force:
        descriptions = refresh_narratives_internal() or []
        return descriptions, _utc_isoformat()
    data = get_cached_data(CACHE_KEY)
    time_str = get_cached_data(CACHE_TIME_KEY)
    if data:
        return data, _utc_isoformat(time_str) if time_str else None
    cached_narratives = _local_cache.get("launch_narratives")
    last_updated = _local_cache.get("last_updated")
    if cached_narratives and last_updated:
        stamp = _utc_isoformat(last_updated)
        return cached_narratives, stamp
    return [], None


def _load_weather_all(force=False):
    if force:
        return refresh_weather_internal(force=True)
    assembled = _assemble_weather_all()
    if assembled and len(assembled.get("weather", {})) == len(WEATHER_LOCATIONS):
        return assembled
    return refresh_weather_internal(force=False)


def get_next_launch_info(upcoming_launches, tz_obj):
    """Find and format the next upcoming (or in-window T+) launch."""
    current_time = datetime.now(timezone.utc)
    future_launches = []
    active_launches = []
    for launch in upcoming_launches or []:
        if launch.get("time") == "TBD":
            continue
        lt_utc = _parse_net_dt(launch.get("net"))
        if not lt_utc:
            continue
        if lt_utc > current_time:
            future_launches.append(launch)
        elif not is_launch_finished(launch.get("status")):
            elapsed = (current_time - lt_utc).total_seconds()
            if elapsed <= T_PLUS_ACTIVE_WINDOW_SECONDS:
                active_launches.append(launch)

    valid_launches = future_launches if future_launches else active_launches
    if not valid_launches:
        return None

    next_l = min(
        valid_launches,
        key=lambda x: _parse_net_dt(x.get("net")) or datetime.max.replace(tzinfo=timezone.utc),
    )
    launch = _slim_launch(next_l, keep_trajectory=True)
    dt_utc = _parse_net_dt(next_l.get("net"))
    if dt_utc:
        local_dt = dt_utc.astimezone(tz_obj)
        launch["local_date"] = local_dt.strftime("%Y-%m-%d")
        launch["local_time"] = local_dt.strftime("%H:%M:%S")
        launch["timezone"] = getattr(tz_obj, "zone", str(tz_obj))
    return launch


def get_upcoming_launches_list(upcoming_launches, tz_obj, limit=10):
    """Sort and format upcoming launches for the dashboard list."""
    current_time = datetime.now(timezone.utc)
    valid_launches = []
    for launch in upcoming_launches or []:
        if launch.get("time") == "TBD":
            continue
        lt_utc = _parse_net_dt(launch.get("net"))
        if not lt_utc:
            continue
        if lt_utc > current_time or not is_launch_finished(launch.get("status")):
            valid_launches.append(launch)

    launches = []
    for launch in sorted(
        valid_launches,
        key=lambda x: _parse_net_dt(x.get("net")) or datetime.max.replace(tzinfo=timezone.utc),
    )[:limit]:
        item = _slim_launch(launch, keep_trajectory=False)
        dt_utc = _parse_net_dt(launch.get("net"))
        if dt_utc:
            local_dt = dt_utc.astimezone(tz_obj)
            item["local_date"] = local_dt.strftime("%Y-%m-%d")
            item["local_time"] = local_dt.strftime("%H:%M:%S")
        launches.append(item)
    return launches


def get_calendar_mapping(launch_data, tz_obj=None):
    """Map YYYY-MM-DD (in tz_obj) to slim launch objects for the calendar view."""
    mapping = {}
    if not launch_data:
        return mapping

    def _append(launch, launch_type):
        date_str = launch.get("date")
        time_str = launch.get("time")
        if tz_obj and launch.get("net"):
            dt_utc = _parse_net_dt(launch.get("net"))
            if dt_utc:
                local_dt = dt_utc.astimezone(tz_obj)
                date_str = local_dt.strftime("%Y-%m-%d")
                time_str = local_dt.strftime("%H:%M:%S")
        if not date_str:
            return
        typed = _slim_launch(launch, keep_trajectory=False)
        typed["type"] = launch_type
        typed["localDate"] = date_str
        typed["localTime"] = f"{date_str} {time_str}" if time_str else date_str
        mapping.setdefault(date_str, []).append(typed)

    for launch in launch_data.get("previous", []) or []:
        _append(launch, "past")
    for launch in launch_data.get("upcoming", []) or []:
        _append(launch, "upcoming")
    return mapping


def get_launch_trends_series(launches, chart_view_mode, current_year, current_month):
    """Bucket launches by rocket family for the dashboard trends chart."""
    rocket_types = ["Starship", "Falcon 9", "Falcon Heavy"]
    if chart_view_mode == "cumulative":
        all_months = [f"{current_year}-{m:02d}" for m in range(1, current_month + 1)]
    else:
        all_months = []
        for i in range(11, -1, -1):
            month = current_month - i
            year = current_year
            while month <= 0:
                month += 12
                year -= 1
            all_months.append(f"{year}-{month:02d}")

    counts = {month: {rocket: 0 for rocket in rocket_types} for month in all_months}
    for launch in launches or []:
        date_str = launch.get("date")
        if not date_str or date_str == "TBD":
            continue
        try:
            month_key = f"{int(date_str[:4]):04d}-{int(date_str[5:7]):02d}"
        except (ValueError, IndexError):
            continue
        if month_key not in counts:
            continue
        rocket = launch.get("rocket", "Unknown")
        matched = next((rt for rt in rocket_types if rt.lower() in str(rocket).lower()), None)
        if matched:
            counts[month_key][matched] += 1

    series = []
    for rocket in rocket_types:
        values = []
        cumulative = 0
        for month in all_months:
            val = counts[month][rocket]
            if chart_view_mode == "cumulative":
                cumulative += val
                values.append(cumulative)
            else:
                values.append(val)
        series.append({"label": rocket, "values": values})
    return all_months, series


def _parse_narratives(raw_list):
    parsed = []
    for item in raw_list or []:
        if isinstance(item, dict):
            narr = dict(item)
            narr.setdefault("source_date", narr.get("date", ""))
            narr.setdefault("source_text", narr.get("text", ""))
            narr.setdefault(
                "source_full",
                narr.get("full")
                or (
                    f"{narr.get('source_date', '')}: {narr.get('source_text', '')}".strip(": ")
                    if narr.get("source_date") or narr.get("source_text")
                    else ""
                ),
            )
            parsed.append(narr)
            continue
        if not isinstance(item, str):
            continue
        match = re.match(r"^(\d{1,2}/\d{1,2}\s+\d{4}):\s*(.*)", item)
        if match:
            parsed.append({
                "date": match.group(1),
                "text": match.group(2),
                "full": item,
                "source_date": match.group(1),
                "source_text": match.group(2),
                "source_full": item,
            })
        else:
            parsed.append({
                "date": "",
                "text": item,
                "full": item,
                "source_date": "",
                "source_text": item,
                "source_full": item,
            })
    return parsed


def prepare_narratives_for_display(narratives_list, launches=None, tz_obj=None):
    """Attach launch metadata and rewrite narrative dates for the requested timezone."""
    if not narratives_list:
        return []
    all_launches = []
    if launches:
        all_launches = (launches.get("upcoming", []) or []) + (launches.get("previous", []) or [])

    prepared = []
    for raw_narr in _parse_narratives(narratives_list):
        narr = dict(raw_narr)
        source_date = narr.get("source_date", "") or narr.get("date", "") or ""
        source_text = narr.get("source_text", "") or narr.get("text", "") or ""
        source_full = narr.get("source_full") or narr.get("full") or (f"{source_date}: {source_text}").strip(": ")
        narr["source_date"] = source_date
        narr["source_text"] = source_text
        narr["source_full"] = source_full
        narr["date"] = source_date
        narr["text"] = source_text
        narr["full"] = source_full
        if not all_launches or not source_date:
            prepared.append(narr)
            continue
        try:
            parts = source_date.split(" ")
            md = parts[0].split("/")
            month = int(md[0])
            day = int(md[1])
            hour = -1
            minute = -1
            if len(parts) > 1 and len(parts[1]) == 4 and parts[1].isdigit():
                hour = int(parts[1][:2])
                minute = int(parts[1][2:])
            best_match = None
            best_match_dt = None
            for launch in all_launches:
                l_dt = _parse_net_dt(launch.get("net"))
                if not l_dt:
                    continue
                if l_dt.month == month and l_dt.day == day:
                    if hour != -1:
                        if l_dt.hour == hour and abs(l_dt.minute - minute) <= 5:
                            best_match = launch
                            best_match_dt = l_dt
                            break
                    elif best_match is None:
                        best_match = launch
                        best_match_dt = l_dt
            if best_match:
                narr["status"] = best_match.get("status")
                narr["landing_location"] = best_match.get("landing_location")
                narr["landing_type"] = best_match.get("landing_type")
                narr["orbit"] = best_match.get("orbit")
                narr["rocket"] = best_match.get("rocket")
                narr["pad"] = best_match.get("pad")
                narr["mission"] = best_match.get("mission")
                display_dt = best_match_dt.astimezone(tz_obj) if tz_obj else best_match_dt
                display_date = f"{display_dt.month}/{display_dt.day} {display_dt.hour:02d}{display_dt.minute:02d}"
                narr["date"] = f"{display_date} {display_dt.strftime('%Z')}".strip()
                narr["day_of_week"] = display_dt.strftime("%a")
                narr["timezone_abbrev"] = display_dt.strftime("%Z")
                narr["full"] = f"{display_date}: {source_text}".strip() if source_text else display_date
        except Exception:
            pass
        prepared.append(narr)
    return prepared


def get_closest_x_video_url(launch_data):
    if not launch_data:
        return ""
    current_time = datetime.now(timezone.utc)
    closest_url = ""
    min_diff = float("inf")
    for launch in (launch_data.get("previous", []) or []) + (launch_data.get("upcoming", []) or []):
        x_url = launch.get("x_video_url")
        if not x_url:
            v_url = launch.get("video_url", "") or ""
            if "x.com" in v_url.lower() or "twitter.com" in v_url.lower():
                x_url = v_url
        if not x_url:
            continue
        launch_net = _parse_net_dt(launch.get("net"))
        if not launch_net:
            continue
        diff = abs((current_time - launch_net).total_seconds())
        if diff < min_diff:
            min_diff = diff
            closest_url = x_url
    return closest_url


def _find_launch_by_id(launch_data, launch_id):
    if not launch_data or not launch_id:
        return None
    for launch in (launch_data.get("upcoming", []) or []) + (launch_data.get("previous", []) or []):
        if launch.get("id") == launch_id:
            return launch
    return None


def _build_trends(launch_data, mode=None):
    now = datetime.now(timezone.utc)
    launches = (launch_data.get("previous", []) or []) + (launch_data.get("upcoming", []) or [])
    if mode:
        months, series = get_launch_trends_series(launches, mode, now.year, now.month)
        return {"year": now.year, "month": now.month, "mode": mode, "months": months, "series": series}
    cumulative_months, cumulative_series = get_launch_trends_series(
        launches, "cumulative", now.year, now.month
    )
    rolling_months, rolling_series = get_launch_trends_series(
        launches, "rolling", now.year, now.month
    )
    return {
        "year": now.year,
        "month": now.month,
        "cumulative": {"months": cumulative_months, "series": cumulative_series},
        "rolling": {"months": rolling_months, "series": rolling_series},
    }


@app.get("/locations")
def get_dashboard_locations(internal: bool = False):
    """Site metadata used by the kiosk (weather, Windy, timezone)."""
    if not internal:
        increment_metric("total_requests")
        increment_metric("cache_hits")
    return {"locations": DASHBOARD_LOCATIONS}


@app.get("/next_launch")
def get_next_launch(tz: str = None, location: str = None, force: bool = False, internal: bool = False):
    if not internal:
        increment_metric("total_requests")
    tz_obj = _resolve_tz(tz, location)
    data = _load_launch_payload(force)
    if data.get("upcoming") or data.get("previous"):
        increment_metric("cache_hits")
    else:
        increment_metric("cache_misses")
    return {
        "next_launch": get_next_launch_info(data.get("upcoming", []), tz_obj),
        "timezone": getattr(tz_obj, "zone", str(tz_obj)),
        "last_updated": data.get("last_updated"),
    }


@app.get("/upcoming_launches")
def get_upcoming_launches(
    tz: str = None,
    location: str = None,
    limit: int = 10,
    force: bool = False,
    internal: bool = False,
):
    if not internal:
        increment_metric("total_requests")
    tz_obj = _resolve_tz(tz, location)
    data = _load_launch_payload(force)
    if data.get("upcoming") or data.get("previous"):
        increment_metric("cache_hits")
    else:
        increment_metric("cache_misses")
    return {
        "upcoming": get_upcoming_launches_list(data.get("upcoming", []), tz_obj, limit=max(1, min(limit, 50))),
        "timezone": getattr(tz_obj, "zone", str(tz_obj)),
        "last_updated": data.get("last_updated"),
    }


@app.get("/calendar")
def get_calendar(tz: str = None, location: str = None, force: bool = False, internal: bool = False):
    if not internal:
        increment_metric("total_requests")
    tz_obj = _resolve_tz(tz, location)
    data = _load_launch_payload(force)
    if data.get("upcoming") or data.get("previous"):
        increment_metric("cache_hits")
    else:
        increment_metric("cache_misses")
    return {
        "calendar": get_calendar_mapping(data, tz_obj),
        "timezone": getattr(tz_obj, "zone", str(tz_obj)),
        "last_updated": data.get("last_updated"),
    }


@app.get("/launch_trends")
def get_launch_trends(mode: str = None, force: bool = False, internal: bool = False):
    if not internal:
        increment_metric("total_requests")
    data = _load_launch_payload(force)
    if data.get("upcoming") or data.get("previous"):
        increment_metric("cache_hits")
    else:
        increment_metric("cache_misses")
    normalized = None
    if mode:
        mode_key = mode.strip().lower()
        if mode_key in ("cumulative", "rolling", "monthly"):
            normalized = "cumulative" if mode_key == "cumulative" else "rolling"
    return {
        "trends": _build_trends(data, normalized),
        "last_updated": data.get("last_updated"),
    }


@app.get("/trajectory")
@app.get("/trajectory/{launch_id}")
def get_trajectory(launch_id: str = None, force: bool = False, internal: bool = False):
    if not internal:
        increment_metric("total_requests")
    data = _load_launch_payload(force)
    upcoming = data.get("upcoming", []) or []
    previous = data.get("previous", []) or []
    target = _find_launch_by_id(data, launch_id) if launch_id else (upcoming[0] if upcoming else (previous[0] if previous else None))
    if not target:
        increment_metric("cache_misses")
        return {"trajectory": None, "error": "Launch not found"}
    existing = target.get("trajectory_data")
    if existing and isinstance(existing, dict) and not launch_id:
        increment_metric("cache_hits")
        return {"trajectory": existing, "launch_id": target.get("id"), "mission": target.get("mission")}
    traj = get_launch_trajectory_data(target, previous)
    if traj:
        increment_metric("cache_hits")
    else:
        increment_metric("cache_misses")
    return {"trajectory": traj, "launch_id": target.get("id"), "mission": target.get("mission")}


@app.get("/dashboard")
def get_dashboard(
    tz: str = None,
    location: str = None,
    include_calendar: bool = True,
    force: bool = False,
    internal: bool = False,
):
    """Combined snapshot of all dashboard app data (not hardware/settings)."""
    if not internal:
        increment_metric("total_requests")
    tz_obj = _resolve_tz(tz, location)
    launch_data = _load_launch_payload(force)
    weather_payload = _load_weather_all(force)
    descriptions, narratives_updated = _load_narratives(force)
    slim = _slim_launch_payload(launch_data, keep_next_trajectory=True)
    next_launch = get_next_launch_info(launch_data.get("upcoming", []), tz_obj)
    upcoming_list = get_upcoming_launches_list(launch_data.get("upcoming", []), tz_obj, limit=10)
    if slim.get("upcoming") or slim.get("previous") or weather_payload.get("weather") or descriptions:
        increment_metric("cache_hits")
    else:
        increment_metric("cache_misses")

    payload = {
        "locations": DASHBOARD_LOCATIONS,
        "launches": slim,
        "weather": weather_payload.get("weather", {}),
        "narratives": {
            "descriptions": descriptions,
            "prepared": prepare_narratives_for_display(descriptions, launch_data, tz_obj),
            "last_updated": narratives_updated,
        },
        "next_launch": next_launch,
        "upcoming": upcoming_list,
        "trends": _build_trends(launch_data),
        "trajectory": (next_launch or {}).get("trajectory_data")
        or (slim.get("upcoming") or [{}])[0].get("trajectory_data"),
        "closest_x_video_url": get_closest_x_video_url(launch_data),
        "timezone": getattr(tz_obj, "zone", str(tz_obj)),
        "location": location if location in DASHBOARD_LOCATIONS else None,
        "last_updated": {
            "launches": slim.get("last_updated"),
            "weather": weather_payload.get("last_updated"),
            "narratives": narratives_updated,
        },
    }
    if include_calendar:
        payload["calendar"] = get_calendar_mapping(launch_data, tz_obj)
    return payload


@app.get("/launch_details/{launch_id}")
def get_launch_details(launch_id: str, internal: bool = False):
    if not internal:
        increment_metric("total_requests")
    increment_metric("cache_misses")  # Detailed fetch is always a direct API call/miss in this impl
    return fetch_launch_details(launch_id)


@app.get("/external_narratives")
def get_all_narratives(internal: bool = False):
    if not internal:
        increment_metric("total_requests")
    increment_metric("cache_misses")  # This endpoint always fetches from external source
    return {"descriptions": fetch_external_narratives()}


@app.get("/recent_launches_narratives")
def get_narratives(force: bool = False, internal: bool = False):
    """Serve from cache only (timer-based refresh)."""
    if not internal:
        increment_metric("total_requests")

    if force:
        increment_metric("cache_misses")
        descriptions = refresh_narratives_internal()
        return {"descriptions": descriptions, "last_updated": _utc_isoformat()}

    # Try to get from Redis
    data = get_cached_data(CACHE_KEY)
    time_str = get_cached_data(CACHE_TIME_KEY)

    if data and time_str:
        increment_metric("cache_hits")
        return {"descriptions": data, "last_updated": _utc_isoformat(time_str)}

    # Fallback to in-memory
    cached_narratives = _local_cache["launch_narratives"]
    last_updated = _local_cache["last_updated"]

    if cached_narratives and last_updated:
        increment_metric("cache_hits")
        return {"descriptions": cached_narratives, "last_updated": _utc_isoformat(last_updated)}

    increment_metric("cache_misses")
    return {"descriptions": [], "last_updated": None}


def _notify_copy_from_params(
    launch_id: str,
    event: NotifyEvent,
    mission: Optional[str] = None,
    net: Optional[str] = None,
    status: Optional[str] = None,
    pad: Optional[str] = None,
    rocket: Optional[str] = None,
    orbit: Optional[str] = None,
    probability: Optional[float] = None,
    previous_status: Optional[str] = None,
):
    increment_metric("total_requests")
    payload = NotifyCopyRequest(
        launch_id=launch_id,
        event=event,
        mission=mission,
        net=net,
        status=status,
        pad=pad,
        rocket=rocket,
        orbit=orbit,
        probability=probability,
        previous_status=previous_status,
    )
    return generate_notify_copy(payload, increment_metrics=True)


@app.post(
    "/notify/copy",
    response_model=NotifyCopyResponse,
    tags=["Notifications"],
    summary="Generate Grok notification copy",
    response_description="Short title+body for a Launch Buddy push/local notification",
)
def post_notify_copy(payload: NotifyCopyRequest):
    """Generate short push/local-notification title+body for an upcoming launch.

    Uses the same xAI Grok model as past-launch narratives
    (`grok-4-1-fast-reasoning` via `XAI_API_KEY`). Cached in Redis for ~1 hour
    keyed by `(launch_id, event, net, status, probability)`.

    Supported `event` values:
    - **t24h** — 24 hours before NET
    - **t1h** — one hour before NET
    - **scrub** — launch scrubbed / delayed / hold (status change)

    Always mentions `probability` when provided (e.g. `70% go` or `weather 40%`).
    Falls back to a template if Grok fails.
    """
    increment_metric("total_requests")
    return generate_notify_copy(payload, increment_metrics=True)


@app.get(
    "/notify/copy",
    response_model=NotifyCopyResponse,
    tags=["Notifications"],
    summary="Generate Grok notification copy (query params)",
    response_description="Short title+body for a Launch Buddy push/local notification",
)
def get_notify_copy(
    launch_id: str = Query(..., examples=["a7e1c2d4-1111-4b2a-9c33-0f1e2d3c4b5a"], description="Launch Library / Launch Buddy launch id"),
    event: NotifyEvent = Query(..., examples=["t24h"], description="t24h | t1h | scrub"),
    mission: Optional[str] = Query(default=None, examples=["Starlink Group 10-20"]),
    net: Optional[str] = Query(default=None, examples=["2026-09-07T02:15:00Z"], description="Launch NET as ISO8601"),
    status: Optional[str] = Query(default=None, examples=["Go"]),
    pad: Optional[str] = Query(default=None, examples=["SLC-40"]),
    rocket: Optional[str] = Query(default=None, examples=["Falcon 9"]),
    orbit: Optional[str] = Query(default=None, examples=["LEO"]),
    probability: Optional[float] = Query(default=None, ge=0, le=100, examples=[80], description="Launch probability 0-100"),
    previous_status: Optional[str] = Query(default=None, examples=["Go"], description="Prior status for scrub context"),
):
    """GET variant of `/notify/copy` for easy client testing. Same shape as POST."""
    return _notify_copy_from_params(
        launch_id=launch_id,
        event=event,
        mission=mission,
        net=net,
        status=status,
        pad=pad,
        rocket=rocket,
        orbit=orbit,
        probability=probability,
        previous_status=previous_status,
    )


@app.get("/metrics")
def get_app_metrics(range: str = "1h", internal: bool = False):
    """Endpoint to fetch application metrics."""
    if not internal:
        record_snapshot()
    return get_metrics(range_type=range)


@app.post("/reset_metrics")
def reset_app_metrics():
    """Endpoint to reset all metrics and history."""
    global _local_metrics, _local_metrics_history, _last_snapshot_time

    # Reset in-memory
    _local_metrics = {
        "total_requests": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "api_calls": 0
    }
    _local_metrics_history = []
    _last_snapshot_time = 0

    # Reset Redis
    if r:
        try:
            r.delete(METRICS_KEY)
            r.delete(METRICS_HISTORY_KEY)
        except Exception as e:
            print(f"Redis error in reset_app_metrics: {e}")
            return {"status": "Error", "message": str(e)}

    return {"status": "Success", "message": "All metrics and history have been reset."}


@app.get("/seed_status")
def get_seed_status():
    """Endpoint to check the status of historical seeding."""
    if r:
        try:
            data = r.get(SEEDING_STATUS_KEY)
            if data:
                return json.loads(data)
        except Exception as e:
            print(f"Redis error in get_seed_status: {e}")
    return _local_seeding_status


@app.post("/seed_history")
def trigger_seed_history():
    """Endpoint to manually trigger historical launch seeding."""
    status = get_seed_status()
    if status.get("is_running"):
        return {"status": "Seeding already in progress"}

    # Clear stop signal if it exists
    global _stop_seeding_requested
    _stop_seeding_requested = False
    if r:
        try:
            r.delete(SEEDING_STOP_SIGNAL_KEY)
        except:
            pass

    # Start seeding in a background thread
    seeding_thread = threading.Thread(target=seed_historical_launches, daemon=True)
    seeding_thread.start()
    return {"status": "Historical seeding started"}


@app.post("/stop_seeding")
def stop_seeding():
    """Endpoint to stop the historical launch seeding."""
    global _stop_seeding_requested
    _stop_seeding_requested = True
    if r:
        try:
            r.set(SEEDING_STOP_SIGNAL_KEY, "true")
        except:
            pass
    return {"status": "Stop signal sent"}


_background_enabled = True


def start_background_worker():
    def run():
        # Wait a bit for the app to start
        time.sleep(5)
        if not _background_enabled:
            return

        # Initial bootstrap (populate empty caches)
        print("Starting background worker bootstrap...")
        try:
            reset_stuck_seeding()
            refresh_narratives_internal()
            refresh_launches_internal()
            refresh_weather_internal()

        except Exception as e:
            print(f"Bootstrap error: {e}")

        try:
            refresh_satellites_internal("starlink")
            refresh_satellites_internal("stations")
        except Exception as e:
            print(f"Satellite GP bootstrap error: {e}")
        try:
            refresh_deployed_satellites_internal()
        except Exception as e:
            print(f"Deployed satellite bootstrap error: {e}")

        last_run = {
            "narratives": time.time(),
            "launches": time.time(),
            "weather": time.time(),
            "satellites": time.time(),
        }

        while _background_enabled:
            try:
                now = time.time()

                # Metrics (every 30s)
                record_snapshot()

                # Narratives (every 15m)
                if now - last_run["narratives"] >= 900:
                    refresh_narratives_internal()
                    last_run["narratives"] = now

                # Launches (every 10m)
                if now - last_run["launches"] >= 600:
                    refresh_launches_internal()
                    last_run["launches"] = now

                # Weather (every 2m for higher frequency wind updates)
                if now - last_run["weather"] >= 120:
                    refresh_weather_internal()
                    last_run["weather"] = now

                # Starlink / stations GP hourly (CelesTrak asks not to hammer).
                # Request handlers serve the cached catalog immediately; this
                # refresh stays off the Heroku request path. The deployed feed
                # is included; a manifest-backed copy expires sooner.
                if now - last_run["satellites"] >= SATELLITES_CACHE_TTL:
                    try:
                        refresh_satellites_internal("starlink")
                        refresh_satellites_internal("stations")
                    except Exception as e:
                        print(f"Satellite GP refresh error: {e}")
                    try:
                        refresh_deployed_satellites_internal()
                    except Exception as e:
                        print(f"Deployed satellite refresh error: {e}")
                    last_run["satellites"] = now

            except Exception as e:
                print(f"Background worker error: {e}")

            time.sleep(30)  # Loop interval

    thread = threading.Thread(target=run, daemon=True)
    thread.start()


start_background_worker()


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    """Serve the dashboard UI."""
    return r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SpaceX Launch Narratives Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; }
        .ticker-item { border-left: 4px solid #3b82f6; }
        .chart-container { height: 60px; width: 100%; margin-top: 1rem; }
        .tab-active { border-bottom: 2px solid #3b82f6; color: #3b82f6; }
        .range-active { background-color: #2563eb !important; color: white !important; }
        .hidden { display: none; }
        .launch-card:hover { transform: translateY(-1px); }
        pre::-webkit-scrollbar { width: 6px; height: 6px; }
        pre::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
        .forecast-day:hover { background-color: rgba(30, 41, 59, 0.5); }
    </style>
</head>
<body class="bg-slate-950 text-slate-200 min-h-screen">
    <nav class="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex items-center justify-between h-16">
                <div class="flex items-center gap-2">
                    <i data-lucide="rocket" class="text-blue-500 w-8 h-8"></i>
                    <span class="text-xl font-bold tracking-tight">SpaceX Narratives</span>
                </div>
                <div class="flex items-center gap-4">
                    <div class="flex items-center gap-2 px-3 py-1 bg-slate-950 border border-slate-800 rounded-lg">
                        <i data-lucide="globe" class="text-slate-500 w-4 h-4"></i>
                        <span id="utc-clock" class="text-xs font-mono font-bold text-slate-400">00:00:00 UTC</span>
                    </div>
                </div>
            </div>
        </div>
    </nav>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <!-- Metrics Header & Time Range -->
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
            <div>
                <h2 class="text-2xl font-bold tracking-tight">System Metrics</h2>
                <p class="text-slate-400 text-sm">Real-time performance and absolute API usage tracking.</p>
            </div>
            <div class="flex flex-col sm:flex-row gap-3">
                <div class="flex bg-slate-900 border border-slate-800 p-1 rounded-xl">
                    <button onclick="changeRange('1h')" id="range-1h" class="px-4 py-1.5 rounded-lg text-sm font-medium transition-all range-active">1h</button>
                    <button onclick="changeRange('24h')" id="range-24h" class="px-4 py-1.5 rounded-lg text-sm font-medium transition-all text-slate-400 hover:text-slate-200">24h</button>
                    <button onclick="changeRange('7d')" id="range-7d" class="px-4 py-1.5 rounded-lg text-sm font-medium transition-all text-slate-400 hover:text-slate-200">7d</button>
                    <button onclick="changeRange('30d')" id="range-30d" class="px-4 py-1.5 rounded-lg text-sm font-medium transition-all text-slate-400 hover:text-slate-200">30d</button>
                </div>
                <button onclick="resetMetrics()" class="flex items-center justify-center gap-2 px-4 py-1.5 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 rounded-xl text-red-400 text-sm font-bold transition-all" title="Reset all metrics and history">
                    <i data-lucide="trash-2" class="w-4 h-4"></i>
                    Reset
                </button>
            </div>
        </div>

        <!-- Key Stats Row -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
             <div class="bg-slate-900/50 border border-slate-800 p-4 rounded-2xl flex items-center gap-4">
                <div class="p-3 bg-blue-500/10 rounded-xl">
                    <i data-lucide="trending-up" class="text-blue-500 w-6 h-6"></i>
                </div>
                <div>
                    <p class="text-slate-500 text-[10px] font-bold uppercase tracking-wider">Live Hits / Day</p>
                    <p class="text-xl font-bold" id="stat-hits-day">0</p>
                </div>
             </div>
             <div class="bg-slate-900/50 border border-slate-800 p-4 rounded-2xl flex items-center gap-4">
                <div class="p-3 bg-emerald-500/10 rounded-xl">
                    <i data-lucide="clock" class="text-emerald-500 w-6 h-6"></i>
                </div>
                <div>
                    <p class="text-slate-500 text-[10px] font-bold uppercase tracking-wider">Uptime</p>
                    <p class="text-xl font-bold" id="stat-uptime">Nominal</p>
                </div>
             </div>
             <div class="bg-slate-900/50 border border-slate-800 p-4 rounded-2xl flex items-center gap-4">
                <div class="p-3 bg-purple-500/10 rounded-xl">
                    <i data-lucide="server" class="text-purple-500 w-6 h-6"></i>
                </div>
                <div>
                    <p class="text-slate-500 text-[10px] font-bold uppercase tracking-wider">Storage</p>
                    <p class="text-xl font-bold">Redis Cloud</p>
                </div>
             </div>
        </div>

        <!-- Refresh Schedule Row -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
             <div class="bg-slate-900/30 border border-slate-800/50 p-3 rounded-xl flex items-center justify-between">
                <div class="flex flex-col">
                    <div class="flex items-center gap-2">
                        <i data-lucide="list" class="text-blue-500 w-4 h-4"></i>
                        <span class="text-[10px] font-bold uppercase tracking-wider text-slate-500">Narrative Refresh</span>
                    </div>
                    <span class="text-[9px] text-slate-600 mt-1 font-medium">Last: <span id="last-ref-narratives" class="text-slate-400">--:--:--</span></span>
                </div>
                <span id="timer-narratives" class="text-sm font-mono font-bold text-blue-400">00:00</span>
             </div>
             <div class="bg-slate-900/30 border border-slate-800/50 p-3 rounded-xl flex items-center justify-between">
                <div class="flex flex-col">
                    <div class="flex items-center gap-2">
                        <i data-lucide="rocket" class="text-emerald-500 w-4 h-4"></i>
                        <span class="text-[10px] font-bold uppercase tracking-wider text-slate-500">Launch Refresh</span>
                    </div>
                    <span class="text-[9px] text-slate-600 mt-1 font-medium">Last: <span id="last-ref-launches" class="text-slate-400">--:--:--</span></span>
                </div>
                <span id="timer-launches" class="text-sm font-mono font-bold text-emerald-400">00:00</span>
             </div>
             <div class="bg-slate-900/30 border border-slate-800/50 p-3 rounded-xl flex items-center justify-between">
                <div class="flex flex-col">
                    <div class="flex items-center gap-2">
                        <i data-lucide="cloud-sun" class="text-blue-400 w-4 h-4"></i>
                        <span class="text-[10px] font-bold uppercase tracking-wider text-slate-500">Weather Refresh</span>
                    </div>
                    <span class="text-[9px] text-slate-600 mt-1 font-medium">Last: <span id="last-ref-weather" class="text-slate-400">--:--:--</span></span>
                </div>
                <span id="timer-weather" class="text-sm font-mono font-bold text-blue-400">00:00</span>
             </div>
        </div>

        <!-- Metrics Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
            <!-- Total Requests Card -->
            <div class="bg-slate-900 border border-slate-800 p-6 rounded-2xl flex flex-col justify-between">
                <div>
                    <div class="flex items-center justify-between mb-4">
                        <span class="text-slate-400 text-sm font-medium">Total Requests</span>
                        <i data-lucide="activity" class="text-blue-400 w-5 h-5"></i>
                    </div>
                    <div class="text-3xl font-bold" id="metric-total-requests">0</div>
                </div>
                <div class="chart-container">
                    <canvas id="chart-requests"></canvas>
                </div>
            </div>

            <!-- Cache Hits Card -->
            <div class="bg-slate-900 border border-slate-800 p-6 rounded-2xl flex flex-col justify-between">
                <div>
                    <div class="flex items-center justify-between mb-4">
                        <span class="text-slate-400 text-sm font-medium">Cache Hits</span>
                        <i data-lucide="database" class="text-emerald-400 w-5 h-5"></i>
                    </div>
                    <div class="text-3xl font-bold text-emerald-400" id="metric-cache-hits">0</div>
                </div>
                <div class="chart-container">
                    <canvas id="chart-hits"></canvas>
                </div>
            </div>

            <!-- API Calls Card -->
            <div class="bg-slate-900 border border-slate-800 p-6 rounded-2xl flex flex-col justify-between">
                <div>
                    <div class="flex items-center justify-between mb-4">
                        <span class="text-slate-400 text-sm font-medium">Grok API Calls</span>
                        <i data-lucide="brain-circuit" class="text-purple-400 w-5 h-5"></i>
                    </div>
                    <div class="text-3xl font-bold text-purple-400" id="metric-api-calls">0</div>
                </div>
                <div class="chart-container">
                    <canvas id="chart-api"></canvas>
                </div>
            </div>

            <!-- Efficiency Card -->
            <div class="bg-slate-900 border border-slate-800 p-6 rounded-2xl flex flex-col justify-between">
                <div>
                    <div class="flex items-center justify-between mb-4">
                        <span class="text-slate-400 text-sm font-medium">Cache Efficiency</span>
                        <i data-lucide="zap" class="text-yellow-400 w-5 h-5"></i>
                    </div>
                    <div class="text-3xl font-bold text-yellow-400" id="metric-efficiency">0%</div>
                </div>
                <div class="chart-container">
                    <canvas id="chart-efficiency"></canvas>
                </div>
            </div>
        </div>

        <!-- Tabs -->
        <div class="flex gap-8 mb-6 border-b border-slate-800">
            <button onclick="showTab('narratives')" id="tab-narratives" class="pb-2 font-semibold transition-colors tab-active">Narratives</button>
            <button onclick="showTab('launches')" id="tab-launches" class="pb-2 font-semibold text-slate-400 hover:text-slate-200 transition-colors">Launches</button>
            <button onclick="showTab('weather')" id="tab-weather" class="pb-2 font-semibold text-slate-400 hover:text-slate-200 transition-colors">Weather</button>
        </div>

        <!-- Narratives Section -->
        <div id="content-narratives" class="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
            <div class="px-6 py-4 border-b border-slate-800 bg-slate-800/30 flex items-center justify-between">
                <h2 class="text-lg font-semibold flex items-center gap-2">
                    <i data-lucide="list" class="w-5 h-5 text-blue-500"></i>
                    Current Narratives
                </h2>
                <div class="flex items-center gap-4">
                    <span class="text-xs text-slate-500 uppercase tracking-widest font-bold" id="last-updated">Updating...</span>
                    <button onclick="refreshTab('narratives')" class="p-1.5 hover:bg-slate-700/50 rounded-lg transition-colors text-slate-400 hover:text-blue-400" title="Force Refresh Narratives">
                        <i data-lucide="refresh-cw" class="w-4 h-4" id="refresh-icon-narratives"></i>
                    </button>
                </div>
            </div>
            <div class="p-6">
                <div id="narratives-list" class="space-y-4">
                    <!-- Loaded dynamically -->
                    <div class="animate-pulse flex space-x-4">
                        <div class="flex-1 space-y-4 py-1">
                            <div class="h-4 bg-slate-800 rounded w-3/4"></div>
                            <div class="h-4 bg-slate-800 rounded"></div>
                            <div class="h-4 bg-slate-800 rounded w-5/6"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Launches Section -->
        <div id="content-launches" class="hidden space-y-8">
            <!-- Historical Seeding Control -->
            <div class="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden p-6 mb-8">
                <div class="flex flex-col md:flex-row items-center justify-between gap-6">
                    <div class="flex items-start gap-4">
                        <div class="p-3 bg-purple-500/10 rounded-xl">
                            <i data-lucide="history" class="w-6 h-6 text-purple-400"></i>
                        </div>
                        <div>
                            <h3 class="text-lg font-semibold flex items-center gap-2 mb-1">
                                Historical Data Seeding
                            </h3>
                            <p class="text-sm text-slate-400 max-w-md">Pull increasingly older launches from SpaceX history to populate the cache. This operation respects API rate limits and runs in the background.</p>
                        </div>
                    </div>
                    <div class="flex flex-col sm:flex-row items-center gap-6 w-full md:w-auto">
                        <div id="seeding-stats" class="bg-slate-800/50 px-4 py-2 rounded-xl border border-slate-700/50 min-w-[200px]">
                            <div class="flex items-center justify-between mb-1">
                                <span class="text-[10px] font-bold uppercase tracking-wider text-slate-500">Seeding Stats</span>
                                <span id="seed-status-tag" class="text-[10px] font-bold uppercase px-1.5 py-0.5 rounded bg-slate-700 text-slate-400">Idle</span>
                            </div>
                            <div class="text-sm font-mono flex flex-col">
                                <span class="flex justify-between gap-4">Pulled: <span id="seed-count" class="text-white font-bold">0</span></span>
                                <span class="flex justify-between gap-4">Oldest: <span id="seed-oldest" class="text-white font-bold text-xs">--</span></span>
                            </div>
                        </div>
                        <div class="flex flex-col sm:flex-row gap-3 w-full sm:w-auto">
                            <button id="btn-seed-history" onclick="triggerSeeding()" class="w-full sm:w-auto px-6 py-3 bg-purple-600 hover:bg-purple-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-xl font-bold transition-all shadow-lg shadow-purple-900/20 flex items-center justify-center gap-2 whitespace-nowrap">
                                <i data-lucide="database-zap" class="w-5 h-5"></i>
                                Seed History
                            </button>
                            <button id="btn-stop-seeding" onclick="stopSeeding()" class="hidden w-full sm:w-auto px-6 py-3 bg-red-600 hover:bg-red-500 text-white rounded-xl font-bold transition-all shadow-lg shadow-red-900/20 flex items-center justify-center gap-2 whitespace-nowrap">
                                <i data-lucide="square" class="w-5 h-5"></i>
                                Stop Seeding
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <div class="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
                <div class="px-6 py-4 border-b border-slate-800 bg-slate-800/30 flex items-center justify-between">
                    <h2 class="text-lg font-semibold flex items-center gap-2">
                        <i data-lucide="rocket" class="w-5 h-5 text-emerald-500"></i>
                        Upcoming Launches
                    </h2>
                    <button onclick="refreshTab('launches')" class="p-1.5 hover:bg-slate-700/50 rounded-lg transition-colors text-slate-400 hover:text-emerald-400" title="Force Refresh Launches">
                        <i data-lucide="refresh-cw" class="w-4 h-4" id="refresh-icon-launches"></i>
                    </button>
                </div>
                <div class="p-6" id="upcoming-launches-list">
                    <div class="animate-pulse space-y-4">
                        <div class="h-12 bg-slate-800 rounded w-full"></div>
                        <div class="h-12 bg-slate-800 rounded w-full"></div>
                    </div>
                </div>
            </div>

            <div class="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
                <div class="px-6 py-4 border-b border-slate-800 bg-slate-800/30">
                    <h2 class="text-lg font-semibold flex items-center gap-2">
                        <i data-lucide="history" class="w-5 h-5 text-blue-500"></i>
                        Previous Launches
                    </h2>
                </div>
                <div class="p-6" id="previous-launches-list">
                    <div class="animate-pulse space-y-4">
                        <div class="h-12 bg-slate-800 rounded w-full"></div>
                        <div class="h-12 bg-slate-800 rounded w-full"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Weather Section -->
        <div id="content-weather" class="hidden space-y-6">
            <div class="flex items-center justify-between">
                <h2 class="text-2xl font-bold tracking-tight">Weather Conditions</h2>
                <button onclick="refreshTab('weather')" class="flex items-center gap-2 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-800 transition-colors rounded-lg font-semibold text-xs text-slate-300">
                    <i data-lucide="refresh-cw" class="w-3.5 h-3.5" id="refresh-icon-weather"></i>
                    Refresh Weather
                </button>
            </div>
            <div id="weather-grid" class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- Weather cards will be loaded here -->
            </div>
        </div>
    </main>

    <!-- Weather Hourly Modal -->
    <div id="weather-modal" class="fixed inset-0 z-[60] hidden bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
        <div class="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-2xl max-h-[80vh] flex flex-col shadow-2xl">
            <div class="p-6 border-b border-slate-800 flex items-center justify-between">
                <div>
                    <h3 id="modal-title" class="text-xl font-bold text-white">Hourly Forecast</h3>
                    <p id="modal-subtitle" class="text-sm text-slate-400">Loading...</p>
                </div>
                <button onclick="closeWeatherModal()" class="p-2 hover:bg-slate-800 rounded-lg transition-colors">
                    <i data-lucide="x" class="w-6 h-6 text-slate-400"></i>
                </button>
            </div>
            <div id="modal-content" class="p-6 overflow-y-auto grid grid-cols-4 sm:grid-cols-6 gap-4">
                <!-- Hourly data will be injected here -->
            </div>
        </div>
    </div>

    <script>
        lucide.createIcons();

        // Chart instances
        const charts = {};
        let currentRange = '1h';

        // Launch data management for lazy loading and pagination
        let allPreviousLaunches = [];
        let displayedPreviousCount = 20;

        async function toggleLaunch(id) {
            const detailsEl = document.getElementById('details-' + id);
            if (!detailsEl) return;

            const isExpanding = detailsEl.classList.contains('hidden');

            // Toggle visibility
            detailsEl.classList.toggle('hidden');

            if (isExpanding) {
                const contentEl = document.getElementById('details-content-' + id);
                if (contentEl && contentEl.getAttribute('data-loaded') !== 'true') {
                    await fetchLaunchDetailsOnDemand(id);
                }
                lucide.createIcons();
            }
        }

        async function fetchLaunchDetailsOnDemand(id) {
            const contentEl = document.getElementById('details-content-' + id);
            const rawEl = document.getElementById('raw-content-' + id);

            if (!contentEl) return;

            try {
                const response = await fetch('/launch_raw/' + id + '?internal=true');
                const rawData = await response.json();

                if (rawData.error) throw new Error(rawData.error);

                contentEl.innerHTML = renderDataCards(rawData);
                contentEl.setAttribute('data-loaded', 'true');
                if (rawEl) rawEl.textContent = JSON.stringify(rawData, null, 2);
            } catch (error) {
                console.error('Error fetching launch raw data:', error);
                contentEl.innerHTML = `<div class="p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-xs flex items-center gap-2">
                    <i data-lucide="alert-circle" class="w-4 h-4"></i>
                    Failed to load detailed mission data: ${error.message}
                </div>`;
                lucide.createIcons();
            }
        }

        function initChart(id, color) {
            const ctx = document.getElementById(id).getContext('2d');
            return new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        data: [],
                        borderColor: color,
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.4,
                        fill: true,
                        backgroundColor: color + '20'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false }, tooltip: { enabled: true } },
                    scales: {
                        x: { display: false },
                        y: { display: false, beginAtZero: true }
                    }
                }
            });
        }

        charts.requests = initChart('chart-requests', '#3b82f6');
        charts.hits = initChart('chart-hits', '#10b981');
        charts.api = initChart('chart-api', '#a855f7');
        charts.efficiency = initChart('chart-efficiency', '#eab308');

        // Timer management
        const refreshIntervals = {
            metrics: 30,
            narratives: 900, // 15m
            launches: 600,   // 10m
            weather: 300     // 5m
        };

        // Initialize next refresh targets
        const nextRefresh = {
            metrics: Date.now() + refreshIntervals.metrics * 1000,
            narratives: Date.now() + refreshIntervals.narratives * 1000,
            launches: Date.now() + refreshIntervals.launches * 1000,
            weather: Date.now() + refreshIntervals.weather * 1000
        };

        function syncTimer(category, lastUpdatedIso) {
            if (!lastUpdatedIso) return;
            const lastUpdatedDate = new Date(lastUpdatedIso);
            const lastUpdated = lastUpdatedDate.getTime();
            const interval = refreshIntervals[category] * 1000;

            // Update the next refresh target based on when the backend last updated
            // Robustness: Ensure we don't set a target in the past, which causes infinite refresh loops
            const target = lastUpdated + interval;
            const now = Date.now();

            if (target <= now + 5000) { // If expired or expiring in next 5s
                // Backend is lagging. Set next refresh to 30s from now to avoid spamming
                nextRefresh[category] = now + 30000;
            } else {
                nextRefresh[category] = target;
            }

            // Update the "Last Refreshed" display in the timer cards (in UTC to match clock)
            const lastRefEl = document.getElementById(`last-ref-${category}`);
            if (lastRefEl) {
                const utcStr = lastUpdatedDate.getUTCHours().toString().padStart(2, '0') + ':' + 
                               lastUpdatedDate.getUTCMinutes().toString().padStart(2, '0') + ':' + 
                               lastUpdatedDate.getUTCSeconds().toString().padStart(2, '0');
                lastRefEl.textContent = utcStr + ' UTC';
            }
        }

        function updateTimers() {
            const now = Date.now();

            // Update UTC Clock
            const nowDate = new Date();
            const utcString = nowDate.getUTCHours().toString().padStart(2, '0') + ':' + 
                             nowDate.getUTCMinutes().toString().padStart(2, '0') + ':' + 
                             nowDate.getUTCSeconds().toString().padStart(2, '0');
            const utcClockEl = document.getElementById('utc-clock');
            if (utcClockEl) {
                utcClockEl.textContent = utcString + ' UTC';
            }

            const formatTime = (ms) => {
                const totalSeconds = Math.max(0, Math.floor(ms / 1000));
                const minutes = Math.floor(totalSeconds / 60);
                const seconds = totalSeconds % 60;
                return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            };

            if (document.getElementById('timer-narratives')) {
                document.getElementById('timer-narratives').textContent = formatTime(nextRefresh.narratives - now);
            }
            if (document.getElementById('timer-launches')) {
                document.getElementById('timer-launches').textContent = formatTime(nextRefresh.launches - now);
            }
            if (document.getElementById('timer-weather')) {
                document.getElementById('timer-weather').textContent = formatTime(nextRefresh.weather - now);
            }

            // Check if any timer expired
            if (now >= nextRefresh.metrics) {
                fetchMetrics();
                nextRefresh.metrics = now + refreshIntervals.metrics * 1000;
            }
            if (now >= nextRefresh.narratives) {
                fetchNarratives();
                nextRefresh.narratives = now + refreshIntervals.narratives * 1000;
            }
            if (now >= nextRefresh.launches) {
                fetchLaunches();
                nextRefresh.launches = now + refreshIntervals.launches * 1000;
            }
            if (now >= nextRefresh.weather) {
                fetchWeatherAll();
                nextRefresh.weather = now + refreshIntervals.weather * 1000;
            }
        }

        function changeRange(range) {
            currentRange = range;
            ['1h', '24h', '7d', '30d'].forEach(r => {
                const btn = document.getElementById(`range-${r}`);
                if (btn) {
                    if (r === range) {
                        btn.classList.add('range-active');
                        btn.classList.remove('text-slate-400', 'hover:text-slate-200');
                        btn.classList.add('text-white');
                    } else {
                        btn.classList.remove('range-active', 'text-white');
                        btn.classList.add('text-slate-400', 'hover:text-slate-200');
                    }
                }
            });
            fetchMetrics();
            // Reset metrics timer
            nextRefresh.metrics = Date.now() + refreshIntervals.metrics * 1000;
        }

        async function resetMetrics() {
            if (!confirm('Are you sure you want to reset all metrics and history? This cannot be undone.')) {
                return;
            }

            try {
                const response = await fetch('/reset_metrics', { method: 'POST' });
                const result = await response.json();
                if (result.status === 'Success') {
                    // Force refresh metrics immediately
                    fetchMetrics();
                    alert('Metrics have been reset successfully.');
                } else {
                    alert('Error: ' + result.message);
                }
            } catch (error) {
                console.error('Error resetting metrics:', error);
                alert('Failed to reset metrics. See console for details.');
            }
        }

        function showTab(tab) {
            ['narratives', 'launches', 'weather'].forEach(t => {
                document.getElementById(`tab-${t}`).classList.remove('tab-active');
                document.getElementById(`tab-${t}`).classList.add('text-slate-400');
                document.getElementById(`content-${t}`).classList.add('hidden');
            });

            document.getElementById(`tab-${tab}`).classList.add('tab-active');
            document.getElementById(`tab-${tab}`).classList.remove('text-slate-400');
            document.getElementById(`content-${tab}`).classList.remove('hidden');

            // Note: Automatic fetches removed from here to make it strictly timer-based
            // and prevent unnecessary API calls/loading states when switching tabs.
        }

        async function fetchMetrics() {
            try {
                const response = await fetch(`/metrics?range=${currentRange}&internal=true`);
                const data = await response.json();

                const current = data.current || {};
                const rangeStats = data.range_stats || {};
                const history = data.history || [];

                // Use live absolute totals for the main metrics
                const total = current.total_requests || 0;
                const hits = current.cache_hits || 0;
                const apiCalls = current.api_calls || 0;

                document.getElementById('metric-total-requests').textContent = total.toLocaleString();
                document.getElementById('metric-cache-hits').textContent = hits.toLocaleString();
                document.getElementById('metric-api-calls').textContent = apiCalls.toLocaleString();

                const efficiency = total > 0 ? Math.round((hits / total) * 100) : 0;
                document.getElementById('metric-efficiency').textContent = efficiency + '%';

                document.getElementById('stat-hits-day').textContent = (data.hits_per_day || 0).toLocaleString();

                // Update charts
                if (history.length > 0) {
                    const updateChart = (chart, key, isEfficiency = false) => {
                        const values = [];
                        for (let i = 0; i < history.length; i++) {
                            if (isEfficiency) {
                                const t = history[i].data.total_requests || 0;
                                values.push(t > 0 ? (history[i].data.cache_hits / t) * 100 : 0);
                            } else {
                                if (i === 0) {
                                    values.push(0);
                                } else {
                                    const diff = (history[i].data[key] || 0) - (history[i-1].data[key] || 0);
                                    values.push(Math.max(0, diff));
                                }
                            }
                        }

                        chart.data.labels = history.map(h => '');
                        chart.data.datasets[0].data = values;
                        chart.update('none');
                    };

                    updateChart(charts.requests, 'total_requests');
                    updateChart(charts.hits, 'cache_hits');
                    updateChart(charts.api, 'api_calls');
                    updateChart(charts.efficiency, '', true);
                }
            } catch (error) {
                console.error('Error fetching metrics:', error);
            }
        }

        function formatValue(val) {
            if (val === null || val === undefined) return '<span class="text-slate-600 italic text-[10px]">null</span>';
            if (val === '') return '<span class="text-slate-600 italic text-[10px]">empty</span>';
            if (typeof val === 'boolean') return `<span class="${val ? 'text-emerald-400' : 'text-red-400'} font-bold text-[10px]">${val}</span>`;
            if (typeof val === 'string' && (val.startsWith('http://') || val.startsWith('https://'))) {
                const isImage = /\.(jpg|jpeg|png|gif|webp|bmp|svg)($|\?)/i.test(val);
                if (isImage) {
                    return `
                        <div class="flex flex-col gap-2">
                            <a href="${val}" target="_blank" class="text-blue-400 hover:underline truncate block max-w-full text-[10px]">${val}</a>
                            <img src="${val}" class="max-w-full h-auto rounded-lg border border-slate-800 shadow-sm mt-1" onerror="this.style.display='none'">
                        </div>
                    `;
                }
                return `<a href="${val}" target="_blank" class="text-blue-400 hover:underline truncate block max-w-full text-[10px]">${val}</a>`;
            }
            return `<span class="text-slate-300 break-words text-[10px]">${val}</span>`;
        }

        function renderSubData(val, depth = 0) {
            if (val === null || val === undefined || val === '') return formatValue(val);
            if (depth > 3) {
                const str = JSON.stringify(val);
                return `<span class="text-[9px] text-slate-500 italic font-mono" title='${str.replace(/'/g, "&apos;")}'>${str.length > 40 ? str.substring(0, 40) + '...' : str}</span>`;
            }

            if (Array.isArray(val)) {
                if (val.length === 0) return formatValue('');
                return val.map(v => `
                    <div class="pl-2 border-l border-slate-800/30 my-1">
                        ${(typeof v === 'object' && v !== null) ? renderSubData(v, depth + 1) : formatValue(v)}
                    </div>
                `).join('');
            }

            if (typeof val === 'object') {
                return Object.entries(val).map(([k, v]) => {
                    return `
                        <div class="flex flex-col mt-1">
                            <p class="text-[9px] text-slate-600 uppercase font-medium tracking-tight">${k.replace(/_/g, ' ')}</p>
                            ${(typeof v === 'object' && v !== null) ? renderSubData(v, depth + 1) : formatValue(v)}
                        </div>
                    `;
                }).join('');
            }

            return formatValue(val);
        }

        function renderDataCards(data) {
            if (!data || typeof data !== 'object') return '';

            let html = '<div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">';

            // Prioritize these fields to appear first
            const priority = ['status', 'net', 'window_start', 'window_end', 'probability', 'holdreason', 'failreason', 'rocket', 'mission', 'pad'];

            const keys = Object.keys(data).sort((a, b) => {
                const aPrio = priority.indexOf(a);
                const bPrio = priority.indexOf(b);
                if (aPrio !== -1 && bPrio !== -1) return aPrio - bPrio;
                if (aPrio !== -1) return -1;
                if (bPrio !== -1) return 1;

                const aIsObj = typeof data[a] === 'object' && data[a] !== null;
                const bIsObj = typeof data[b] === 'object' && data[b] !== null;
                return aIsObj - bIsObj;
            });

            for (const key of keys) {
                const value = data[key];

                html += `
                    <div class="bg-slate-950/40 p-3 rounded-xl border border-slate-800/50 space-y-1.5 flex flex-col">
                        <p class="text-[10px] text-slate-500 uppercase font-bold tracking-wider border-b border-slate-800/50 pb-1">${key.replace(/_/g, ' ')}</p>
                        <div class="flex-1">
                            ${(typeof value === 'object' && value !== null) ? renderSubData(value) : formatValue(value)}
                        </div>
                    </div>
                `;
            }

            html += '</div>';
            return html;
        }

        async function fetchNarratives(force = false) {
            try {
                const url = force ? '/recent_launches_narratives?force=true&internal=true' : '/recent_launches_narratives?internal=true';
                const response = await fetch(url);
                const data = await response.json();
                const list = document.getElementById('narratives-list');
                list.innerHTML = '';

                if (data.descriptions && data.descriptions.length > 0) {
                    data.descriptions.forEach(desc => {
                        const parts = desc.split(': ', 2);
                        const date = parts[0];
                        const text = parts[1] || '';

                        const div = document.createElement('div');
                        div.className = 'ticker-item bg-slate-800/40 p-4 rounded-r-lg border-l-4 border-blue-500 hover:bg-slate-800/60 transition-colors';
                        div.innerHTML = `
                            <div class="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4">
                                <span class="text-blue-400 font-mono font-bold whitespace-nowrap">${date}</span>
                                <p class="text-slate-200 leading-relaxed">${text}</p>
                            </div>
                        `;
                        list.appendChild(div);
                    });
                } else {
                    list.innerHTML = '<p class="text-slate-500 text-center py-8">No narratives available.</p>';
                }

                if (data.last_updated) {
                    syncTimer('narratives', data.last_updated);
                    const luDate = new Date(data.last_updated);
                    const luUtc = luDate.getUTCHours().toString().padStart(2, '0') + ':' + 
                                 luDate.getUTCMinutes().toString().padStart(2, '0') + ':' + 
                                 luDate.getUTCSeconds().toString().padStart(2, '0') + ' UTC';
                    document.getElementById('last-updated').textContent = 'Last updated: ' + luUtc;
                }
            } catch (error) {
                console.error('Error fetching narratives:', error);
            }
        }

        async function fetchLaunches(force = false) {
            const upList = document.getElementById('upcoming-launches-list');

            try {
                const url = force ? '/launches_slim?force=true&internal=true' : '/launches_slim?internal=true';
                const response = await fetch(url);
                const data = await response.json();

                allPreviousLaunches = data.previous || [];

                renderList(upList, data.upcoming);
                renderPreviousList(false);

                if (data.last_updated) {
                    syncTimer('launches', data.last_updated);
                }
            } catch (error) {
                console.error('Error fetching launches:', error);
                const upList = document.getElementById('upcoming-launches-list');
                const prevList = document.getElementById('previous-launches-list');
                if (upList) upList.innerHTML = '<p class="text-red-400 text-center py-4">Error loading upcoming launches.</p>';
                if (prevList) prevList.innerHTML = '<p class="text-red-400 text-center py-4">Error loading previous launches.</p>';
            }
        }

        function renderList(el, launches, isHistorical = false) {
            if (!isHistorical) el.innerHTML = '';
            if (!launches || launches.length === 0) {
                if (!isHistorical) el.innerHTML = '<p class="text-slate-500 text-center py-4">No launches found.</p>';
                return;
            }
            launches.forEach(l => {
                const div = document.createElement('div');
                div.className = 'launch-card border-b border-slate-800 last:border-0 hover:bg-slate-800/10 transition-all';

                const id = 'details-' + l.id;
                const rawId = 'raw-' + l.id;

                div.innerHTML = `
                    <div class="p-4 cursor-pointer" onclick="toggleLaunch('${l.id}')">
                        <div class="flex items-center justify-between">
                            <div class="flex flex-col">
                                <span class="font-bold text-slate-100">${l.mission}</span>
                                <span class="text-xs text-slate-400">${l.rocket} • ${l.pad}</span>
                            </div>
                            <div class="flex flex-col items-end text-right">
                                <div class="flex items-center gap-2 mb-1">
                                    <span class="text-sm font-mono text-blue-400">${l.date} ${l.time}</span>
                                    <i data-lucide="chevron-down" class="w-4 h-4 text-slate-500"></i>
                                </div>
                                <span class="text-[10px] px-2 py-0.5 rounded bg-slate-800 uppercase tracking-tighter ${l.status === 'Success' ? 'text-emerald-400' : 'text-yellow-400'}">${l.status}</span>
                            </div>
                        </div>
                    </div>

                    <div id="${id}" class="hidden px-4 pb-6 space-y-4 border-t border-slate-800/50 pt-4 bg-slate-900/30">
                        ${(l.image && typeof l.image === 'string') ? `<img src="${l.image}" class="w-full h-48 object-cover rounded-xl border border-slate-700 shadow-lg" onerror="this.style.display='none'">` : ''}

                        ${l.description ? `<p class="text-sm text-slate-300 leading-relaxed bg-slate-950/50 p-4 rounded-xl border border-slate-800">${l.description}</p>` : ''}

                        <div class="space-y-4">
                            <h3 class="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-2 border-b border-slate-800 pb-1">Detailed Mission Data</h3>
                            <div id="details-content-${l.id}" class="min-h-[50px] flex flex-col justify-center">
                                <div class="animate-pulse flex space-x-2">
                                    <div class="h-4 bg-slate-800 rounded w-full"></div>
                                </div>
                                <span class="text-[10px] text-slate-600 italic mt-2 text-center">Loading detailed mission data...</span>
                            </div>
                        </div>

                        <div class="flex flex-wrap gap-3">
                            ${(l.video_url && typeof l.video_url === 'string') ? `
                            <a href="${l.video_url}" target="_blank" class="flex items-center gap-2 px-3 py-1.5 bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded-lg transition-colors border border-red-500/20 text-xs font-semibold">
                                <i data-lucide="play-circle" class="w-4 h-4"></i> Webcast
                            </a>` : ''}
                            ${(l.x_video_url && typeof l.x_video_url === 'string') ? `
                            <a href="${l.x_video_url}" target="_blank" class="flex items-center gap-2 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg transition-colors border border-slate-700 text-xs font-semibold">
                                <i data-lucide="twitter" class="w-4 h-4"></i> X Update
                            </a>` : ''}
                            <button onclick="document.getElementById('${rawId}').classList.toggle('hidden')" class="flex items-center gap-2 px-3 py-1.5 bg-blue-500/10 hover:bg-blue-500/20 text-blue-400 rounded-lg transition-colors border border-blue-500/20 text-xs font-semibold ml-auto">
                                <i data-lucide="code" class="w-4 h-4"></i> View Raw Data
                            </button>
                        </div>

                        <div id="${rawId}" class="hidden mt-4">
                            <div class="flex items-center justify-between mb-2">
                                <span class="text-[10px] text-slate-500 uppercase font-bold">API Response Object</span>
                                <button onclick="navigator.clipboard.writeText(this.parentElement.nextElementSibling.textContent)" class="text-[10px] text-blue-400 hover:text-blue-300 uppercase font-bold">Copy JSON</button>
                            </div>
                            <pre id="raw-content-${l.id}" class="bg-slate-950 p-4 rounded-xl border border-slate-800 text-[10px] text-slate-400 overflow-x-auto font-mono max-h-60">Loading JSON...</pre>
                        </div>
                    </div>
                `;
                el.appendChild(div);
            });
            lucide.createIcons();
        }

        function renderPreviousList(append = false) {
            const el = document.getElementById('previous-launches-list');
            if (!append) {
                el.innerHTML = '';
                displayedPreviousCount = 0;
            }

            const existingBtn = document.getElementById('btn-load-more');
            if (existingBtn) existingBtn.remove();

            const start = displayedPreviousCount;
            const end = Math.min(start + (append ? 50 : 20), allPreviousLaunches.length);
            const launches = allPreviousLaunches.slice(start, end);

            renderList(el, launches, true);
            displayedPreviousCount = end;

            if (displayedPreviousCount < allPreviousLaunches.length) {
                const btn = document.createElement('button');
                btn.id = 'btn-load-more';
                btn.className = 'w-full py-8 text-slate-400 hover:text-white font-bold transition-all border-t border-slate-800 bg-slate-900/10 hover:bg-slate-900/30 flex items-center justify-center gap-2';
                btn.innerHTML = `<i data-lucide="plus-circle" class="w-5 h-5"></i> Load More (${allPreviousLaunches.length - displayedPreviousCount} remaining)`;
                btn.onclick = () => {
                    renderPreviousList(true);
                };
                el.appendChild(btn);
                lucide.createIcons();
            }
        }

        function renderWeatherCard(card, loc, data) {
            const getFlightCategoryColor = (cat) => {
                switch(cat) {
                    case 'VFR': return 'text-emerald-400';
                    case 'MVFR': return 'text-blue-400';
                    case 'IFR': return 'text-yellow-400';
                    case 'LIFR': return 'text-red-400';
                    default: return 'text-slate-400';
                }
            };

            const getWeatherIcon = (code) => {
                // WMO Weather interpretation codes (WW)
                if (code === 0) return 'sun';
                if (code <= 3) return 'cloud-sun';
                if (code <= 48) return 'cloud';
                if (code <= 67) return 'cloud-rain';
                if (code <= 77) return 'cloud-snow';
                if (code <= 82) return 'cloud-rain';
                if (code <= 86) return 'cloud-snow';
                if (code <= 99) return 'cloud-lightning';
                return 'cloud-sun';
            };

            let forecastHtml = '';
            if (data.forecast && data.forecast.daily) {
                forecastHtml = `
                    <div class="mt-4 pt-4 border-t border-slate-800">
                        <p class="text-[10px] font-bold uppercase tracking-wider text-slate-500 mb-3">7-Day Forecast</p>
                        <div class="grid grid-cols-7 gap-1">
                            ${data.forecast.daily.time.map((time, i) => {
                                const date = new Date(time + 'T00:00:00');
                                const dayName = date.toLocaleDateString('en-US', { weekday: 'short' });
                                const maxTemp = Math.round(data.forecast.daily.temperature_2m_max[i]);
                                const minTemp = Math.round(data.forecast.daily.temperature_2m_min[i]);
                                const code = data.forecast.daily.weathercode[i];
                                return `
                                    <div class="flex flex-col items-center p-1.5 rounded-lg forecast-day cursor-pointer transition-colors" 
                                         onclick="showHourly('${loc}', '${time}', ${i})">
                                        <span class="text-[9px] text-slate-500 font-bold uppercase">${dayName}</span>
                                        <i data-lucide="${getWeatherIcon(code)}" class="w-4 h-4 my-1 text-blue-400"></i>
                                        <span class="text-[10px] font-bold">${maxTemp}°</span>
                                        <span class="text-[9px] text-slate-500">${minTemp}°</span>
                                    </div>
                                `;
                            }).join('')}
                        </div>
                    </div>
                `;
            }

            card.innerHTML = `
                <div class="flex items-center justify-between mb-4">
                    <div class="flex flex-col">
                        <h3 class="text-xl font-bold">${loc}</h3>
                        <span class="text-[10px] font-bold uppercase tracking-widest ${getFlightCategoryColor(data.flight_category)}">${data.flight_category || 'Unknown'}</span>
                    </div>
                    <i data-lucide="cloud-sun" class="text-blue-400 w-6 h-6"></i>
                </div>
                <div class="grid grid-cols-3 gap-y-4 gap-x-4">
                    <div>
                        <p class="text-[9px] text-slate-500 uppercase font-bold tracking-wider mb-1">Temp</p>
                        <p class="text-lg font-semibold">${Math.round(data.temperature_f)}°F <span class="text-[10px] text-slate-400 font-normal">/ ${Math.round(data.temperature_c)}°C</span></p>
                    </div>
                    <div>
                        <p class="text-[9px] text-slate-500 uppercase font-bold tracking-wider mb-1">Dewpoint</p>
                        <p class="text-lg font-semibold">${Math.round(data.dewpoint_f)}°F <span class="text-[10px] text-slate-400 font-normal">/ ${Math.round(data.dewpoint_c)}°C</span></p>
                    </div>
                    <div>
                        <p class="text-[9px] text-slate-500 uppercase font-bold tracking-wider mb-1">Humidity</p>
                        <p class="text-lg font-semibold">${data.humidity}%</p>
                    </div>
                    <div>
                        <p class="text-[9px] text-slate-500 uppercase font-bold tracking-wider mb-1">Wind</p>
                        <p class="text-lg font-semibold">${Math.round(data.live_wind && data.live_wind.speed_kts !== undefined ? data.live_wind.speed_kts : data.wind_speed_kts)}${ (data.live_wind && data.live_wind.gust_kts) ? `<span class="text-red-400 text-sm ml-1">G${Math.round(data.live_wind.gust_kts)}</span>` : (data.wind_gust_kts ? `<span class="text-red-400 text-sm ml-1">G${data.wind_gust_kts}</span>` : '')} <span class="text-[10px] text-slate-400 font-normal">kts</span></p>
                        ${data.live_wind ? `<p class="text-[8px] text-blue-500 font-bold mt-0.5"><i data-lucide="zap" class="w-2 h-2 inline-block"></i> LIVE</p>` : ''}
                    </div>
                    <div>
                        <p class="text-[9px] text-slate-500 uppercase font-bold tracking-wider mb-1">Direction</p>
                        <p class="text-lg font-semibold">${data.live_wind && data.live_wind.direction !== undefined ? data.live_wind.direction : data.wind_direction}°</p>
                    </div>
                    <div>
                        <p class="text-[9px] text-slate-500 uppercase font-bold tracking-wider mb-1">Visibility</p>
                        <p class="text-lg font-semibold">${data.visibility_sm} <span class="text-[10px] text-slate-400 font-normal">sm</span></p>
                    </div>
                    <div>
                        <p class="text-[9px] text-slate-500 uppercase font-bold tracking-wider mb-1">Pressure</p>
                        <p class="text-lg font-semibold">${data.altimeter_inhg ? data.altimeter_inhg.toFixed(2) : '29.92'} <span class="text-[10px] text-slate-400 font-normal">inHg</span></p>
                    </div>
                    <div>
                        <p class="text-[9px] text-slate-500 uppercase font-bold tracking-wider mb-1">Clouds</p>
                        <p class="text-lg font-semibold">${data.cloud_cover}%</p>
                    </div>
                </div>
                ${forecastHtml}
                <div class="mt-4 pt-3 border-t border-slate-800">
                    <p class="text-[9px] font-mono text-slate-600 truncate uppercase" title="${data.raw || ''}">${data.raw || 'No raw METAR data'}</p>
                </div>
            `;
        }

        let lastWeatherData = null;

        function showHourly(loc, dateStr, dayIndex) {
            if (!lastWeatherData || !lastWeatherData[loc] || !lastWeatherData[loc].forecast) return;
            const forecast = lastWeatherData[loc].forecast;
            const modal = document.getElementById('weather-modal');
            const content = document.getElementById('modal-content');
            const title = document.getElementById('modal-title');
            const subtitle = document.getElementById('modal-subtitle');

            const date = new Date(dateStr + 'T00:00:00');
            title.textContent = `${loc} - ${date.toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' })}`;
            subtitle.textContent = "Hourly Temperature (°C)";

            content.innerHTML = '';

            // Hourly data starts from index (dayIndex * 24)
            const startIndex = dayIndex * 24;
            for (let i = 0; i < 24; i++) {
                const idx = startIndex + i;
                if (idx >= forecast.hourly.time.length) break;

                const timeStr = forecast.hourly.time[idx];
                const hour = new Date(timeStr).getHours();
                const temp = Math.round(forecast.hourly.temperature_2m[idx]);
                const windSpeed = Math.round(forecast.hourly.windspeed_10m[idx]);
                const windDir = forecast.hourly.winddirection_10m[idx];

                const item = document.createElement('div');
                item.className = 'flex flex-col items-center p-3 bg-slate-800/50 rounded-xl border border-slate-700/50';
                item.innerHTML = `
                    <span class="text-[10px] font-bold text-slate-400 mb-1">${hour}:00</span>
                    <span class="text-sm font-bold text-white">${temp}°C</span>
                    <span class="text-[9px] text-slate-500 mt-1">${Math.round(temp * 9/5 + 32)}°F</span>
                    <div class="mt-2 pt-2 border-t border-slate-700/50 w-full flex flex-col items-center">
                         <span class="text-[10px] font-bold text-blue-400">${windSpeed} km/h</span>
                         <span class="text-[8px] text-slate-500 uppercase">${windDir}°</span>
                    </div>
                `;
                content.appendChild(item);
            }

            modal.classList.remove('hidden');
            document.body.style.overflow = 'hidden';
            lucide.createIcons();
        }

        function closeWeatherModal() {
            const modal = document.getElementById('weather-modal');
            modal.classList.add('hidden');
            document.body.style.overflow = 'auto';
        }

        async function fetchWeatherAll(force = false) {
            const container = document.getElementById('weather-grid');

            // Show loaders
            container.innerHTML = '';
            ['Starbase', 'Vandy', 'Cape', 'Hawthorne', 'Bastrop'].forEach(loc => {
                const card = document.createElement('div');
                card.className = 'bg-slate-900 border border-slate-800 p-6 rounded-2xl flex flex-col gap-4';
                card.innerHTML = `<div class="animate-pulse h-32 bg-slate-800 rounded"></div>`;
                container.appendChild(card);
            });

            try {
                const url = force ? `/weather_all?force=true&internal=true` : `/weather_all?internal=true`;
                const response = await fetch(url);
                const data = await response.json();
                lastWeatherData = data.weather;

                container.innerHTML = '';
                if (data.weather) {
                    Object.entries(data.weather).forEach(([loc, weatherData]) => {
                        const card = document.createElement('div');
                        card.className = 'bg-slate-900 border border-slate-800 p-6 rounded-2xl flex flex-col gap-4';
                        renderWeatherCard(card, loc, weatherData);
                        container.appendChild(card);
                    });
                    lucide.createIcons();
                }

                if (data.last_updated) {
                    syncTimer('weather', data.last_updated);
                }
            } catch (error) {
                container.innerHTML = '<p class="text-red-400 text-center py-4">Error loading weather.</p>';
            }
        }

        async function refreshTab(tab) {
            const icon = document.getElementById(`refresh-icon-${tab}`);
            if (icon) icon.classList.add('animate-spin');

            try {
                if (tab === 'narratives') await fetchNarratives(true);
                if (tab === 'launches') await fetchLaunches(true);
                if (tab === 'weather') await fetchWeatherAll(true);

                // Reset timer for this specific tab after manual refresh
                nextRefresh[tab] = Date.now() + refreshIntervals[tab] * 1000;

                // Also update metrics as a force refresh counts as API calls
                fetchMetrics();
                nextRefresh.metrics = Date.now() + refreshIntervals.metrics * 1000;
            } catch (error) {
                console.error(`Error refreshing ${tab}:`, error);
            } finally {
                if (icon) {
                    setTimeout(() => icon.classList.remove('animate-spin'), 500);
                }
            }
        }

        // --- Seeding Logic ---
        let seedingPollInterval = null;

        async function triggerSeeding() {
            const btn = document.getElementById('btn-seed-history');
            btn.disabled = true;
            btn.innerHTML = '<i data-lucide="loader-2" class="w-5 h-5 animate-spin"></i> Triggering...';
            lucide.createIcons();

            try {
                const response = await fetch('/seed_history', { method: 'POST' });
                const data = await response.json();
                console.log('Seeding response:', data);

                // Small delay to let the background thread start and update status
                setTimeout(startSeedingPoll, 1000);
            } catch (error) {
                console.error('Error triggering seeding:', error);
                btn.disabled = false;
                btn.innerHTML = '<i data-lucide="database-zap" class="w-5 h-5"></i> Seed History';
                lucide.createIcons();
            }
        }

        async function stopSeeding() {
            const btn = document.getElementById('btn-stop-seeding');
            btn.disabled = true;
            btn.innerHTML = '<i data-lucide="loader-2" class="w-5 h-5 animate-spin"></i> Stopping...';
            lucide.createIcons();

            try {
                const response = await fetch('/stop_seeding', { method: 'POST' });
                const data = await response.json();
                console.log('Stop response:', data);
            } catch (error) {
                console.error('Error stopping seeding:', error);
                btn.disabled = false;
                btn.innerHTML = '<i data-lucide="square" class="w-5 h-5"></i> Stop Seeding';
                lucide.createIcons();
            }
        }

        function startSeedingPoll() {
            if (seedingPollInterval) clearInterval(seedingPollInterval);
            pollSeedingStatus();
            seedingPollInterval = setInterval(pollSeedingStatus, 3000);
        }

        async function pollSeedingStatus() {
            try {
                const response = await fetch('/seed_status');
                const data = await response.json();

                const btn = document.getElementById('btn-seed-history');
                const stopBtn = document.getElementById('btn-stop-seeding');
                const tag = document.getElementById('seed-status-tag');
                const count = document.getElementById('seed-count');
                const oldest = document.getElementById('seed-oldest');

                count.textContent = (data.total_pulled || 0).toLocaleString();
                oldest.textContent = data.oldest_launch ? data.oldest_launch.split('T')[0] : '--';

                if (data.is_running) {
                    btn.disabled = true;
                    btn.innerHTML = '<i data-lucide="loader-2" class="w-5 h-5 animate-spin"></i> Seeding...';
                    stopBtn.classList.remove('hidden');
                    // Only reset stop button text if not already in "Stopping..." state
                    if (!stopBtn.disabled) {
                        stopBtn.innerHTML = '<i data-lucide="square" class="w-5 h-5"></i> Stop Seeding';
                    }
                    tag.textContent = 'Active';
                    tag.className = 'text-[10px] font-bold uppercase px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-400 animate-pulse';
                } else {
                    btn.disabled = false;
                    btn.innerHTML = '<i data-lucide="database-zap" class="w-5 h-5"></i> Seed History';
                    stopBtn.classList.add('hidden');
                    tag.textContent = data.last_status || 'Idle';
                    tag.className = 'text-[10px] font-bold uppercase px-1.5 py-0.5 rounded bg-slate-700 text-slate-400';

                    // If it stopped and we are polling, we can stop if it's completed or hit limit
                    if (!data.is_running && seedingPollInterval && data.last_status !== 'Starting batch fetch...') {
                        // We keep it polling just in case, or stop it to save resources?
                        // Let's keep it but at a much slower rate? No, let's just stop if it's Idle.
                    }
                }
                lucide.createIcons();
            } catch (error) {
                console.error('Error polling seeding status:', error);
            }
        }

        // Initial load - Fetch all data once to bootstrap the UI
        fetchMetrics();
        fetchNarratives();
        fetchLaunches();
        fetchWeatherAll();
        startSeedingPoll();

        // Start the master timer loop (updates UI countdowns and triggers refreshes)
        setInterval(updateTimers, 1000);
    </script>
</body>
</html>
    """


@app.post("/refresh")
def refresh_cache():
    """Force refresh the cache (call this via scheduler)."""
    print("Manual cache refresh triggered via POST /refresh.")
    descriptions = refresh_narratives_internal()
    count = len(descriptions) if descriptions else 0
    return {
        "status": "Cache refreshed",
        "count": count,
        "timestamp": _utc_isoformat()
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))