"""Launch-specific charts for one Launch Library 2 launch.

``GET /launch/{ll2_launch_id}/insights`` reads the launch record this
process already cached (hot raw or the ``launch_raw_v2`` side store) and
fills booster flight history from one Launch Library query per serial.
It does not refresh ``/launches`` and it does not build an ascent curve:
the globe path is a ground track, so ``ascent_profile`` is never returned.

Event times are Launch Library timeline rows only. A miss, HTTP error, or
429 omits the section that call would have filled and does not retry.
Insights calls stay inside the free-tier surplus left after the main list
refresh (``LL_HOURLY_BUDGET`` minus ``LL_RESERVED_FOR_MAIN_FEED``). Booster
history is skipped when that surplus is down to its last call.

The background worker warms the next 12 upcoming launches from the stored
list. ``INSIGHTS_WARM=0`` turns that pass off. It uses the same surplus and
does not refresh ``/launches``.
"""

from __future__ import annotations

import math
import os
import threading
import time
from datetime import datetime, timezone

import requests
from fastapi import FastAPI

import live_activity_push

LL_LAUNCHES = "https://ll.thespacedevs.com/2.3.0/launches/"
LAUNCH_TTL = 6 * 3600
BOOSTER_TTL = 24 * 3600
LAUNCH_PREFIX = "launch_insights_v1:"
BOOSTER_PREFIX = "launch_insights_booster_v1:"
# Free-tier cap is about 15/hour. The list worker spends two calls every
# 10 minutes, so insights only spends the surplus. Override when the token's
# real cap is higher.
LL_HOURLY_BUDGET = int(os.getenv("LL_HOURLY_BUDGET", "15"))
LL_RESERVED_FOR_MAIN_FEED = int(os.getenv("LL_RESERVED_FOR_MAIN_FEED", "12"))
# Keep the last surplus call for a launch-detail miss. Booster history waits.
BOOSTER_MIN_REMAINING = 2
LAUNCH_TIMEOUT = (3.05, 8)
BOOSTER_TIMEOUT = (3.05, 15)
BUDGET_TTL = 3700
WARM_LIMIT = 12
WARM_LOCK_KEY = "launch_insights_warm_lock_v1"
WARM_LOCK_TTL = 600

_SKIP_STAGE = (
    "second",
    "upper",
    "fairing",
    "stage 2",
    "stage-2",
    "ship",
    "dragon",
)
_SKIP_SERIAL = {"unknown", "tbd", "n/a", "na", "none"}
_BOOSTER_FIELDS = (
    "serial",
    "flight_number_this_launch",
    "total_flights",
    "landing_success_count",
    "landing_attempt_count",
    "first_flight_date",
)

_mem_lock = threading.Lock()
_mem = {}
_locks_guard = threading.Lock()
_serial_locks = {}
_launch_locks = {}
_session_lock = threading.Lock()
_session = None
_warm_lock = threading.Lock()


class _Budget:
    def __init__(self):
        self.lock = threading.Lock()
        self.hour = None
        self.used = 0
        self.blocked = False

    def reset(self):
        with self.lock:
            self.hour = None
            self.used = 0
            self.blocked = False

    def try_reserve(self, kind: str) -> bool:
        """Count one insights call, or refuse before it is sent."""
        with self.lock:
            self._roll()
            if self._blocked():
                return False
            allowance = _allowance()
            used = self._used_count()
            remaining = allowance - used
            need = BOOSTER_MIN_REMAINING if kind == "booster" else 1
            if remaining < need:
                return False
            new_used = self._commit(used)
            if new_used is None:
                return False
            # A concurrent dyno may have taken the last surplus call between
            # the read and the increment. Give that call back.
            if kind == "booster" and allowance - (new_used - 1) < BOOSTER_MIN_REMAINING:
                self._rollback()
                return False
            return True

    def remaining(self) -> int:
        """Surplus calls left this hour. Zero after a 429."""
        with self.lock:
            self._roll()
            if self._blocked():
                return 0
            return max(0, _allowance() - self._used_count())

    def block(self):
        """A 429 stops further insights calls until the hour changes."""
        with self.lock:
            self._roll()
            self.blocked = True
            client = _redis()
            if client is None:
                return
            try:
                client.setex(_blocked_key(self.hour), BUDGET_TTL, "1")
            except Exception as exc:
                print(f"LL2 insights budget block failed: {exc.__class__.__name__}")

    def _roll(self):
        hour = _hour_bucket()
        if hour != self.hour:
            self.hour = hour
            self.used = 0
            self.blocked = False

    def _blocked(self) -> bool:
        if self.blocked:
            return True
        client = _redis()
        if client is None:
            return False
        try:
            flagged = client.get(_blocked_key(self.hour))
        except Exception:
            return False
        if flagged:
            self.blocked = True
            return True
        return False

    def _used_count(self) -> int:
        client = _redis()
        if client is None:
            return self.used
        try:
            raw = client.get(_used_key(self.hour))
        except Exception:
            return self.used
        if raw in (None, ""):
            return 0
        try:
            return int(raw)
        except (TypeError, ValueError):
            return self.used

    def _commit(self, used: int):
        client = _redis()
        if client is None:
            self.used = used + 1
            return self.used
        try:
            new_used = int(client.incr(_used_key(self.hour)))
            client.expire(_used_key(self.hour), BUDGET_TTL)
        except Exception:
            self.used = used + 1
            return self.used
        allowance = _allowance()
        if new_used > allowance:
            _safe_decr(client, self.hour)
            self.used = max(0, new_used - 1)
            return None
        self.used = new_used
        return new_used

    def _rollback(self):
        client = _redis()
        if client is not None:
            _safe_decr(client, self.hour)
            try:
                raw = client.get(_used_key(self.hour))
                if raw not in (None, ""):
                    self.used = int(raw)
                    return
            except Exception:
                pass
        self.used = max(0, self.used - 1)


_budget = _Budget()


def reset_state():
    """Drop in-process insights cache and budget. Tests only."""
    _budget.reset()
    with _mem_lock:
        _mem.clear()
    with _locks_guard:
        _serial_locks.clear()
        _launch_locks.clear()


def install(fastapi_app: FastAPI) -> None:
    @fastapi_app.get(
        "/launch/{ll2_launch_id}/insights",
        tags=["Launches"],
        summary="Booster history, timeline, orbit, and landing for one launch",
        description=(
            "Launch-specific chart data. Sections with no Launch Library data "
            "are omitted. There is no ascent profile. Cached for 6 hours per "
            "launch and 24 hours per booster serial. A Launch Library error "
            "or rate limit returns 200 without the missing section and is "
            "not retried."
        ),
    )
    def get_launch_insights(ll2_launch_id: str):
        import app as app_module

        app_module.increment_metric("total_requests")
        body, cache_hit = build_insights(ll2_launch_id)
        app_module.increment_metric("cache_hits" if cache_hit else "cache_misses")
        return body


def insights_warm_enabled() -> bool:
    """``INSIGHTS_WARM=0`` disables the background warmer. Anything else is on."""
    flag = os.getenv("INSIGHTS_WARM", "1").strip().lower()
    return flag not in ("0", "false", "no", "off")


def warm_upcoming_insights():
    """Fill insights for the next 12 stored upcoming launches.

    Reads the list cache only. Soonest net first. A launch that is already
    fresh, including booster history for every core on that document, is
    skipped. One Redis lock (or the in-process lock when Redis is down)
    keeps a second dyno or a restart from running a second pass. The lock
    expires so a dead worker does not hold it.
    """
    if not insights_warm_enabled():
        return {"status": "disabled", "considered": 0, "warmed": 0, "skipped": 0}
    if not _warm_lock.acquire(blocking=False):
        return {"status": "busy", "considered": 0, "warmed": 0, "skipped": 0}
    client = _redis()
    token = None
    try:
        if client is not None:
            token = f"{time.time()}:{os.getpid()}:{threading.get_ident()}"
            try:
                acquired = client.set(WARM_LOCK_KEY, token, nx=True, ex=WARM_LOCK_TTL)
            except Exception as exc:
                print(f"Insights warm lock failed: {exc.__class__.__name__}")
                return {"status": "lock-error", "considered": 0, "warmed": 0, "skipped": 0}
            if not acquired:
                return {"status": "busy", "considered": 0, "warmed": 0, "skipped": 0}
        try:
            return _warm_pass(client, token)
        finally:
            _release_warm_lock(client, token)
    finally:
        _warm_lock.release()


def _warm_pass(client, token) -> dict:
    ids = _stored_upcoming_ids(WARM_LIMIT)
    warmed = 0
    skipped = 0
    for launch_id in ids:
        _touch_warm_lock(client, token)
        if _insights_fresh(launch_id):
            skipped += 1
            continue
        doc_fresh = _launch_doc_fresh(launch_id)
        remaining = _budget.remaining()
        if doc_fresh and remaining < BOOSTER_MIN_REMAINING:
            skipped += 1
            continue
        if not doc_fresh and remaining < 1:
            skipped += 1
            continue
        try:
            build_insights(launch_id)
        except Exception as exc:
            print(f"Insights warm failed for {launch_id}: {exc.__class__.__name__}")
        warmed += 1
    return {"status": "ok", "considered": len(ids), "warmed": warmed, "skipped": skipped}


def _stored_upcoming_ids(limit: int) -> list:
    """Soonest upcoming ids from the list cache. Does not call Launch Library."""
    import app as app_module

    data = app_module._launches_mem.get("data")
    if not isinstance(data, dict):
        try:
            data = app_module.get_cached_data(app_module.LAUNCHES_CACHE_KEY)
        except Exception:
            data = None
    upcoming = []
    if isinstance(data, dict):
        for launch in data.get("upcoming") or []:
            if isinstance(launch, dict) and launch.get("id"):
                upcoming.append(launch)
    upcoming.sort(key=_upcoming_sort_key)
    ids = []
    seen = set()
    for launch in upcoming:
        launch_id = str(launch.get("id"))
        if not launch_id or launch_id in seen:
            continue
        seen.add(launch_id)
        ids.append(launch_id)
        if len(ids) >= limit:
            break
    return ids


def _upcoming_sort_key(launch: dict):
    parsed = _parse_dt(launch.get("net")) if isinstance(launch.get("net"), str) else None
    if parsed is None:
        return (1, 0.0)
    return (0, parsed.timestamp())


def _launch_doc_fresh(launch_id: str) -> bool:
    cached = _cache_get(_launch_key(launch_id), LAUNCH_TTL)
    return isinstance(cached, dict) and bool(cached.get("_detailed"))


def _booster_cache_fresh(serial: str) -> bool:
    cached = _cache_get(_booster_key(serial), BOOSTER_TTL)
    return isinstance(cached, dict) and isinstance(cached.get("flights"), list)


def _insights_fresh(launch_id: str) -> bool:
    """Launch document and every core's flight list are still inside their TTL."""
    cached = _cache_get(_launch_key(launch_id), LAUNCH_TTL)
    if not isinstance(cached, dict) or not cached.get("_detailed"):
        return False
    for stub in cached.get("booster") or []:
        if not isinstance(stub, dict):
            continue
        serial = stub.get("serial")
        if isinstance(serial, str) and serial and not _booster_cache_fresh(serial):
            return False
    return True


def _touch_warm_lock(client, token):
    if client is None or not token:
        return
    try:
        if client.get(WARM_LOCK_KEY) == token:
            client.expire(WARM_LOCK_KEY, WARM_LOCK_TTL)
    except Exception:
        pass


def _release_warm_lock(client, token):
    if client is None or not token:
        return
    try:
        if client.get(WARM_LOCK_KEY) == token:
            client.delete(WARM_LOCK_KEY)
    except Exception:
        pass


def build_insights(ll2_launch_id: str):
    """Return ``(body, cache_hit)``. ``cache_hit`` means no LL2 call was sent."""
    launch_id = str(ll2_launch_id or "").strip()
    used = [False]
    doc = _launch_document(launch_id, used)
    body = _public_body(doc, launch_id, used)
    return body, not used[0]


def _launch_document(launch_id: str, used: list) -> dict:
    cached = _cache_get(_launch_key(launch_id), LAUNCH_TTL)
    if isinstance(cached, dict) and cached.get("_detailed"):
        return cached

    lock = _lock_for(_launch_locks, launch_id)
    with lock:
        cached = _cache_get(_launch_key(launch_id), LAUNCH_TTL)
        if isinstance(cached, dict) and cached.get("_detailed"):
            return cached

        record = _cached_launch_record(launch_id)
        if _has_launch_detail(record):
            return _store_launch(launch_id, record)

        if not _budget.try_reserve("launch"):
            return _undetailed(launch_id, record)

        used[0] = True
        status, body = _ll_get(
            f"{LL_LAUNCHES}{launch_id}/",
            {"format": "json", "mode": "detailed"},
            LAUNCH_TIMEOUT,
        )
        if status == 429:
            _budget.block()
            return _undetailed(launch_id, record)
        if status == 200 and isinstance(body, dict):
            returned = body.get("id")
            if not returned or str(returned) == launch_id:
                return _store_launch(launch_id, body)
        return _undetailed(launch_id, record)


def _store_launch(launch_id: str, record: dict) -> dict:
    doc = _sections_from_record(record, launch_id)
    doc["_detailed"] = True
    return _cache_set(_launch_key(launch_id), doc, LAUNCH_TTL)


def _undetailed(launch_id: str, record) -> dict:
    if isinstance(record, dict):
        return _sections_from_record(record, launch_id)
    return {"ll2_launch_id": launch_id}


def _public_body(doc: dict, launch_id: str, used: list) -> dict:
    body = {"ll2_launch_id": launch_id}
    name = doc.get("name")
    if isinstance(name, str) and name.strip():
        body["name"] = name.strip()

    current_rows = doc.get("_current_flights") if isinstance(doc.get("_current_flights"), dict) else {}
    boosters = []
    seen = set()
    for stub in doc.get("booster") or []:
        if not isinstance(stub, dict):
            continue
        serial = stub.get("serial")
        if not isinstance(serial, str) or serial in seen:
            continue
        seen.add(serial)
        item = {key: stub[key] for key in _BOOSTER_FIELDS if key in stub}
        flights = _flights_for(serial, current_rows.get(serial), launch_id, used)
        if flights:
            item["flights"] = flights
        boosters.append(item)
    if boosters:
        body["booster"] = boosters

    for section in ("events", "orbit", "landing"):
        value = doc.get(section)
        if value:
            body[section] = value
    body["cached"] = not used[0]
    return body


def _flights_for(serial: str, current_row, launch_id: str, used: list):
    cached = _cache_get(_booster_key(serial), BOOSTER_TTL)
    rows = cached.get("flights") if isinstance(cached, dict) else None
    if not isinstance(rows, list):
        rows = _fetch_booster_flights(serial, used)
    if rows is None:
        return None
    merged = [dict(row) for row in rows if isinstance(row, dict)]
    if isinstance(current_row, dict) and current_row.get("ll2_launch_id"):
        if not any(str(row.get("ll2_launch_id")) == str(current_row.get("ll2_launch_id")) for row in merged):
            merged.append(dict(current_row))
    return _finalize_flights(merged, launch_id)


def _fetch_booster_flights(serial: str, used: list):
    lock = _lock_for(_serial_locks, serial)
    with lock:
        cached = _cache_get(_booster_key(serial), BOOSTER_TTL)
        if isinstance(cached, dict) and isinstance(cached.get("flights"), list):
            return cached["flights"]
        if not _budget.try_reserve("booster"):
            return None
        used[0] = True
        status, body = _ll_get(
            LL_LAUNCHES,
            {
                "format": "json",
                "mode": "detailed",
                "limit": 100,
                "ordering": "net",
                "serial_number": serial,
            },
            BOOSTER_TIMEOUT,
        )
        if status == 429:
            _budget.block()
            return None
        if status != 200 or not isinstance(body, dict):
            return None
        rows = _rows_for_serial(body.get("results"), serial)
        if rows is None:
            return None
        stored = _cache_set(_booster_key(serial), {"flights": rows}, BOOSTER_TTL)
        return stored.get("flights") or []


def _rows_for_serial(results, serial: str):
    """Flight rows for this serial, or None when the payload cannot be checked.

    An empty result list is a real empty history. A page of launches that do
    not carry this serial is not cached: the filter may have been ignored.
    """
    if not isinstance(results, list):
        return None
    if not results:
        return []
    rows = []
    seen = set()
    for launch in results:
        if not isinstance(launch, dict):
            continue
        stage = _stage_for_serial(launch, serial)
        if stage is None:
            continue
        row = _flight_row(launch, stage)
        launch_key = str(row.get("ll2_launch_id") or "")
        if not launch_key or launch_key in seen:
            continue
        seen.add(launch_key)
        rows.append(row)
    if not rows:
        return None
    return rows


def _sections_from_record(record: dict, launch_id: str) -> dict:
    doc = {"ll2_launch_id": launch_id}
    name = _launch_name(record)
    if name:
        doc["name"] = name

    if _has_launch_detail(record):
        stages = _first_stages(record)
        boosters = []
        current = {}
        for stage in stages:
            stub = _booster_stub(stage)
            if stub is None:
                continue
            boosters.append(stub)
            row = _flight_row(record, stage)
            if row.get("ll2_launch_id"):
                current[stub["serial"]] = row
        if boosters:
            doc["booster"] = boosters
            doc["_current_flights"] = current
        events = _events(record)
        if events:
            doc["events"] = events
        landing = _landing_section(stages)
        if landing:
            doc["landing"] = landing
    else:
        landing = _landing_from_slim(record)
        if landing:
            doc["landing"] = landing

    orbit = _orbit_section(record)
    if orbit:
        doc["orbit"] = orbit
    return doc


def _events(record: dict):
    """Non-negative Launch Library timeline rows. No modeled times."""
    if not isinstance(record.get("timeline"), list):
        return None
    events = []
    for label, seconds in live_activity_push.timeline_events(record):
        if seconds is None or seconds < 0 or not label:
            continue
        events.append({
            "name": label,
            "t_plus_s": _json_number(seconds),
            "source": "ll2",
        })
    events.sort(key=lambda event: event["t_plus_s"] if event["t_plus_s"] is not None else 0)
    return events or None


def _orbit_section(record: dict):
    mission = record.get("mission")
    orbit = mission.get("orbit") if isinstance(mission, dict) else None
    if isinstance(orbit, dict):
        out = {}
        name = orbit.get("name") or orbit.get("abbrev")
        if isinstance(name, str) and name.strip():
            out["name"] = name.strip()
        for source, dest in (
            ("apogee_km", "apogee_km"),
            ("apogee", "apogee_km"),
            ("perigee_km", "perigee_km"),
            ("perigee", "perigee_km"),
            ("inclination_deg", "inclination_deg"),
            ("inclination", "inclination_deg"),
        ):
            if dest in out or source not in orbit:
                continue
            number = _json_number(orbit.get(source))
            if number is not None:
                out[dest] = number
        return out or None
    label = record.get("orbit")
    if isinstance(label, str) and label.strip() and label.strip().lower() != "unknown":
        return {"name": label.strip()}
    return None


def _landing_section(stages: list):
    chosen = _choose_landing(stages)
    if not isinstance(chosen, dict):
        return None
    out = {}
    kind = _landing_type(chosen.get("type"))
    if kind:
        out["type"] = kind
    loc = _landing_location(chosen)
    if isinstance(loc, dict):
        if isinstance(loc.get("name"), str) and loc["name"].strip():
            out["location_name"] = loc["name"].strip()
        if isinstance(loc.get("abbrev"), str) and loc["abbrev"].strip():
            out["location_abbrev"] = loc["abbrev"].strip()
        successes = _nonnegative(loc.get("successful_landings"))
        attempts = _nonnegative(loc.get("attempted_landings"))
        if successes is not None:
            out["location_successful_landings"] = successes
        if attempts is not None:
            out["location_attempted_landings"] = attempts
    down = _json_number(chosen.get("downrange_distance"))
    if _publish_downrange(down):
        out["downrange_km"] = down
    return out or None


def _landing_from_slim(record: dict):
    out = {}
    kind = _landing_type(record.get("landing_type"))
    if kind:
        out["type"] = kind
    location = record.get("landing_location")
    if isinstance(location, str) and location.strip():
        out["location_name"] = location.strip()
    down = _json_number(record.get("landing_downrange_km"))
    if _publish_downrange(down):
        out["downrange_km"] = down
    return out or None


def _choose_landing(stages: list):
    """Same priority as the globe: coordinates, then a real downrange, then any recovery."""
    import app as app_module

    parsed = []
    for stage in stages:
        landing = stage.get("landing") if isinstance(stage, dict) else None
        if isinstance(landing, dict):
            parsed.append(landing)
    if not parsed:
        return None

    def recoverable(landing):
        kind = _landing_type(landing.get("type"))
        if kind and "expend" in kind.lower():
            return False
        if kind:
            return True
        return bool(landing.get("attempt"))

    def has_coords(landing):
        loc = _landing_location(landing)
        if not isinstance(loc, dict):
            return False
        lat = app_module._finite_coord(loc.get("latitude"))
        lon = app_module._finite_coord(loc.get("longitude"))
        return app_module._usable_pad_coords(lat, lon)

    for landing in parsed:
        if recoverable(landing) and has_coords(landing):
            return landing
    for landing in parsed:
        down = _json_number(landing.get("downrange_distance"))
        if recoverable(landing) and _publish_downrange(down):
            return landing
    for landing in parsed:
        if recoverable(landing):
            return landing
    return parsed[0]


def _publish_downrange(down) -> bool:
    """Sub-30 km figures are the pad offset, not the booster's flight range."""
    import app as app_module

    return isinstance(down, (int, float)) and not isinstance(down, bool) and down >= app_module._ASDS_DOWNRANGE_MIN_KM


def _landing_type(type_obj):
    abbrev = ""
    name = ""
    if isinstance(type_obj, dict):
        abbrev = str(type_obj.get("abbrev") or "").strip()
        name = str(type_obj.get("name") or "").strip()
    elif isinstance(type_obj, str):
        name = type_obj.strip()
    blob = f"{abbrev} {name}".lower()
    if not blob.strip():
        return None
    if "asds" in blob or "drone" in blob:
        return "ASDS"
    if "rtls" in blob or "return to launch" in blob:
        return "RTLS"
    if "expend" in blob:
        return "expendable"
    return abbrev or name


def _landing_location(landing: dict):
    loc = landing.get("landing_location")
    if isinstance(loc, dict):
        return loc
    loc = landing.get("location")
    if isinstance(loc, dict):
        return loc
    return None


def _first_stages(record: dict) -> list:
    rocket = record.get("rocket") if isinstance(record.get("rocket"), dict) else {}
    stages = rocket.get("launcher_stage")
    if not isinstance(stages, list):
        return []
    return [stage for stage in stages if _is_first_stage(stage)]


def _is_first_stage(stage) -> bool:
    if not isinstance(stage, dict):
        return False
    kind = stage.get("type")
    if not isinstance(kind, str) or not kind.strip():
        return True
    text = kind.lower()
    return not any(token in text for token in _SKIP_STAGE)


def _booster_stub(stage: dict):
    launcher = stage.get("launcher") if isinstance(stage.get("launcher"), dict) else {}
    if launcher.get("is_placeholder") is True:
        return None
    serial = launcher.get("serial_number")
    if not isinstance(serial, str):
        return None
    serial = serial.strip()
    if not serial or serial.lower() in _SKIP_SERIAL:
        return None
    stub = {"serial": serial}
    flight_no = _nonnegative(stage.get("launcher_flight_number"))
    if flight_no is not None and flight_no > 0:
        stub["flight_number_this_launch"] = flight_no
    total = _nonnegative(launcher.get("flights"))
    if total is not None:
        stub["total_flights"] = total
    successes = _nonnegative(launcher.get("successful_landings"))
    if successes is not None:
        stub["landing_success_count"] = successes
    attempts = _nonnegative(launcher.get("attempted_landings"))
    if attempts is not None:
        stub["landing_attempt_count"] = attempts
    first = launcher.get("first_launch_date")
    if isinstance(first, str) and first.strip():
        stub["first_flight_date"] = first.strip()
    return stub


def _flight_row(launch: dict, stage: dict) -> dict:
    landing = stage.get("landing") if isinstance(stage.get("landing"), dict) else {}
    loc = _landing_location(landing) if landing else None
    abbrev = None
    if isinstance(loc, dict) and isinstance(loc.get("abbrev"), str) and loc["abbrev"].strip():
        abbrev = loc["abbrev"].strip()
    success = landing.get("success") if isinstance(landing, dict) and "success" in landing else None
    if not isinstance(success, bool):
        success = None
    mission = launch.get("mission") if isinstance(launch.get("mission"), dict) else {}
    mission_name = mission.get("name") if isinstance(mission.get("name"), str) and mission["name"].strip() else None
    if mission_name is None and isinstance(launch.get("name"), str) and launch["name"].strip():
        mission_name = launch["name"].strip()
    net = launch.get("net") if isinstance(launch.get("net"), str) else None
    row = {
        "date": net,
        "mission_name": mission_name,
        "ll2_launch_id": launch.get("id"),
        "landing_location_abbrev": abbrev,
        "landing_success": success,
    }
    hint = _turnaround_hint(stage, _parse_dt(net))
    if hint is not None:
        row["turnaround_hint"] = hint
    return row


def _turnaround_hint(stage: dict, flight_dt):
    prev = _parse_dt(stage.get("previous_flight_date")) if isinstance(stage, dict) else None
    if prev is not None and flight_dt is not None:
        delta = (flight_dt - prev).total_seconds()
        if delta >= 0:
            return int(delta // 86400)
    if not isinstance(stage, dict):
        return None
    seconds = live_activity_push._parse_relative(stage.get("turn_around_time"))
    if seconds is None or seconds < 0:
        return None
    return int(seconds // 86400)


def _finalize_flights(rows: list, current_id: str) -> list:
    ordered = sorted(rows, key=lambda row: row.get("date") or "9999")
    prev = None
    public = []
    for row in ordered:
        dt = _parse_dt(row.get("date"))
        days = None
        if prev is not None and dt is not None:
            delta = (dt - prev).total_seconds()
            if delta >= 0:
                days = int(delta // 86400)
        hint = row.get("turnaround_hint")
        if days is None and type(hint) is int and hint >= 0:
            days = hint
        if dt is not None:
            prev = dt
        success = row.get("landing_success")
        public.append({
            "date": row.get("date"),
            "mission_name": row.get("mission_name"),
            "ll2_launch_id": row.get("ll2_launch_id"),
            "landing_location_abbrev": row.get("landing_location_abbrev"),
            "landing_success": success if isinstance(success, bool) else None,
            "turnaround_days": days,
            "is_current": str(row.get("ll2_launch_id") or "") == str(current_id),
        })
    return public


def _stage_for_serial(launch: dict, serial: str):
    wanted = serial.strip().lower()
    for stage in _first_stages(launch):
        launcher = stage.get("launcher") if isinstance(stage.get("launcher"), dict) else {}
        found = launcher.get("serial_number")
        if isinstance(found, str) and found.strip().lower() == wanted:
            return stage
    return None


def _has_launch_detail(record) -> bool:
    if not isinstance(record, dict):
        return False
    if isinstance(record.get("rocket"), dict):
        return True
    if isinstance(record.get("mission"), dict):
        return True
    return isinstance(record.get("timeline"), list)


def _launch_name(record: dict):
    for key in ("name", "mission"):
        value = record.get(key)
        if isinstance(value, str) and value.strip() and value.strip().lower() != "unknown":
            return value.strip()
    mission = record.get("mission")
    if isinstance(mission, dict):
        value = mission.get("name")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _cached_launch_record(launch_id: str):
    """Side store and list cache only. Does not call Launch Library."""
    import app as app_module

    hot = app_module._read_hot_raw(launch_id)
    if isinstance(hot, dict):
        return hot
    raw = app_module._get_raw_launch(launch_id)
    if isinstance(raw, dict):
        return raw
    data = app_module._launches_mem.get("data")
    if not isinstance(data, dict):
        try:
            data = app_module.get_cached_data(app_module.LAUNCHES_CACHE_KEY)
        except Exception:
            data = None
    if not isinstance(data, dict):
        return None
    for bucket in ("upcoming", "previous"):
        for launch in data.get(bucket) or []:
            if isinstance(launch, dict) and str(launch.get("id") or "") == launch_id:
                raw = launch.get("all_data")
                if isinstance(raw, dict):
                    return raw
                return launch
    return None


def _ll_get(url: str, params: dict, timeout):
    """One Launch Library GET. No retry. 429 is reported to the caller."""
    import app as app_module

    app_module.increment_metric("api_calls")
    headers = dict(app_module._ll_request_headers())
    headers["Accept"] = "application/json"
    try:
        response = _http().get(url, params=params, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        print(f"LL2 insights request failed: {exc.__class__.__name__}")
        return 0, None
    if response.status_code != 200:
        if response.status_code != 429:
            print(f"LL2 insights HTTP {response.status_code}")
        return response.status_code, None
    try:
        body = response.json()
    except ValueError:
        return response.status_code, None
    return 200, body


def _http():
    global _session
    with _session_lock:
        if _session is None:
            _session = requests.Session()
            _session.headers.update({
                "Accept": "application/json",
                "User-Agent": (
                    "launch-narrative-api/insights "
                    "(https://github.com/hwpaige/launch-summary-api; Launch Buddy charts)"
                ),
            })
        return _session


def _cache_get(key: str, ttl: int):
    now = time.time()
    with _mem_lock:
        entry = _mem.get(key)
        if entry and entry[0] > now:
            return entry[1]
        if entry:
            _mem.pop(key, None)
    try:
        import app as app_module
        data = app_module.get_cached_data(key)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    stamp = data.get("_cached_at")
    if not isinstance(stamp, (int, float)) or now - float(stamp) >= ttl:
        return None
    with _mem_lock:
        _mem[key] = (float(stamp) + ttl, data)
    return data


def _cache_set(key: str, data: dict, ttl: int) -> dict:
    stored = dict(data)
    stamp = time.time()
    stored["_cached_at"] = stamp
    with _mem_lock:
        _mem[key] = (stamp + ttl, stored)
    try:
        import app as app_module
        app_module.set_cached_data(key, stored, ttl=ttl)
    except Exception as exc:
        print(f"LL2 insights cache write failed: {exc.__class__.__name__}")
    return stored


def _lock_for(table: dict, key: str):
    with _locks_guard:
        lock = table.get(key)
        if lock is None:
            lock = threading.Lock()
            table[key] = lock
        return lock


def _allowance() -> int:
    return max(0, int(LL_HOURLY_BUDGET) - int(LL_RESERVED_FOR_MAIN_FEED))


def _hour_bucket() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H")


def _used_key(hour: str) -> str:
    return f"ll_insights_budget_v1:{hour}"


def _blocked_key(hour: str) -> str:
    return f"ll_insights_blocked_v1:{hour}"


def _redis():
    try:
        import app as app_module
        return app_module.r
    except Exception:
        return None


def _safe_decr(client, hour: str):
    try:
        client.decr(_used_key(hour))
    except Exception:
        pass


def _parse_dt(value):
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _json_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if value.is_integer():
            return int(value)
        return value
    return int(value)


def _nonnegative(value):
    number = _json_number(value)
    if isinstance(number, (int, float)) and number >= 0:
        return int(number) if isinstance(number, float) and number.is_integer() else number
    return None


def _launch_key(launch_id: str) -> str:
    return f"{LAUNCH_PREFIX}{launch_id}"


def _booster_key(serial: str) -> str:
    return f"{BOOSTER_PREFIX}{serial.strip().upper()}"
