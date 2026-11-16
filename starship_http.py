"""Cached Starship tracker and closure reads for Launch Buddy.

One background poller ingests the public feeds. ``GET /starship/tracker``
and ``GET /starship/closures`` return the last document. They do not call
upstream and they do not invent a vehicle position.

A vehicle ``lat`` / ``lon`` is a feed fix (or, only for ``?fixture=flight_test``,
a marked fixture). Predicted points stay in ``predicted`` with
``kind: "predicted"``. The modeled pad path stays on ``GET /trajectory``
and is described here as ``planned``, not as a vehicle.
"""
from __future__ import annotations

import gzip
import json
import os
import random
import threading
import time
from datetime import datetime, timezone

from fastapi import FastAPI, Query, Request
from fastapi.responses import Response

import closure_areas
import starship_flight_test as flight_test
from starship_flight_test import FlightTestRun, env_mode
from starship_tracker import (
    ACTIVE_POLL_S,
    BEYOND_STAGE_ZERO_URL,
    FALLBACK_TRACKER_URL,
    IDLE_POLL_S,
    PRIMARY_TRACKER_URL,
    STARDASH_FEED_URL,
    TrackerHostState,
    TrackerState,
    fetch_flight_history,
    fetch_with_failover,
    http_backoff_s,
    poll_interval_s,
    ship_trail_is_partial,
)

TRACKER_CACHE_CONTROL = "public, max-age=5"
CLOSURE_CACHE_CONTROL = "public, max-age=60"
FIXTURE_CACHE_CONTROL = "no-store"
STATE_REDIS_KEY = "starship_tracker_state_v1"
TRACKER_REDIS_KEY = "starship_tracker_public_v1"
CLOSURE_REDIS_KEY = "starship_closures_public_v1"
LOCK_REDIS_KEY = "starship_poller_lock"
LOCK_MARGIN_S = 30
# Flight 15 rings are the last resort. Live FAA/NGA refresh on this cadence.
CLOSURE_REFRESH_S = closure_areas.REFRESH_S

_service = None
_service_lock = threading.Lock()
_poller_started = False


def _now():
    return datetime.now(timezone.utc)


def _iso(moment):
    return moment.astimezone(timezone.utc).isoformat()


def _feed_kind(url):
    text = str(url or "")
    if not text:
        return "none"
    if text in (PRIMARY_TRACKER_URL, FALLBACK_TRACKER_URL) or "spacex.com" in text or "azureedge" in text:
        return "official"
    if text in (BEYOND_STAGE_ZERO_URL, STARDASH_FEED_URL) or "beyondstagezero" in text or "stardash" in text:
        return "hobby"
    return "other"


def _number_or_none(value, scale):
    if value is None:
        return None
    try:
        return float(value) * scale
    except (TypeError, ValueError):
        return None


def annotate_tracker_payload(payload, *, source, done_by_key=None, feed_url=None, generated_at=None):
    """Add iOS aliases and mark predicted / planned data. Does not invent a fix.

    ``source`` is ``feed`` (a real tracker document was ingested), ``fixture``
    (FlightTestRun), or ``none`` (no fix yet). ``live_positions`` is true only
    for feed fixes.
    """
    body = dict(payload or {})
    done_by_key = done_by_key or {}
    vehicles = []
    for vehicle in body.get("vehicles") or []:
        if not isinstance(vehicle, dict):
            continue
        item = dict(vehicle)
        item["id"] = item.get("key") or item.get("id")
        item["alt_km"] = _number_or_none(item.get("alt_m"), 0.001)
        item["speed_kmh"] = _number_or_none(item.get("speed_mps"), 3.6)
        item["gps_time"] = item.get("gps_s")
        item["down"] = bool(done_by_key.get(item.get("key"), item.get("missing")))
        item["position_kind"] = "fixture" if source == "fixture" else "feed"
        item["labels"] = {
            "altitude": item.get("altitude_text") or "",
            "speed": item.get("speed_text") or "",
            "mission": item.get("mission_text") or "",
            "apsis": item.get("apsis_label") or "",
        }
        item["trail"] = [
            dict(point, kind="flown")
            for point in (item.get("trail") or [])
            if isinstance(point, dict)
        ]
        item["predicted"] = [
            dict(point, kind="predicted")
            for point in (item.get("predicted") or [])
            if isinstance(point, dict)
        ]
        vehicles.append(item)
    body["vehicles"] = vehicles
    have_fix = source == "feed" and bool(vehicles)
    if source == "fixture":
        position_source = "fixture"
    elif have_fix:
        position_source = "feed"
    else:
        position_source = "none"
    body["position_source"] = position_source
    body["live_positions"] = position_source == "feed"
    body["generated_at"] = generated_at or _iso(_now())
    if source == "fixture":
        body["feed"] = {"kind": "fixture", "url": None}
        body["test"] = True
    else:
        body["feed"] = {
            "kind": _feed_kind(feed_url) if have_fix else "none",
            "url": feed_url if have_fix else None,
        }
    hidden = bool(body.get("hide_planned"))
    body["planned"] = {
        "kind": "planned",
        "hidden": hidden,
        "endpoint": "/trajectory",
        "note": (
            "Modeled pad ground track from GET /trajectory. "
            "Not a vehicle position. Hidden from T-0 until the mission ends."
        ),
    }
    telemetry = dict(body.get("telemetry") or {})
    telemetry["position_source"] = position_source
    if source == "fixture":
        telemetry["test"] = True
    body["telemetry"] = telemetry
    body["orbit"] = {
        "kind": "derived",
        "in_orbit": bool(telemetry.get("in_orbit")),
        "orbit_label": telemetry.get("orbit_label") or "",
        "orbit_number": telemetry.get("orbit_number") or 0,
        "completed_laps": telemetry.get("completed_laps") or 0,
        "apsis_label": telemetry.get("apsis_label") or "",
    }
    return body


def _empty_tracker():
    return annotate_tracker_payload(
        {
            "show": False,
            "stale": True,
            "hold": False,
            "hide_planned": False,
            "launch_active": False,
            "vehicles": [],
            "telemetry": {"show": False, "stale": True, "in_orbit": False},
        },
        source="none",
    )


def _render(payload):
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return {"json": raw, "gzip": gzip.compress(raw, compresslevel=6)}


class StarshipService:
    """Owns the tracker state. The request path only reads rendered bytes."""

    def __init__(self, launch_loader=None, path=None):
        self.path = path or os.path.join(
            os.environ.get("STARSHIP_CACHE_DIR") or "/tmp/launch-narrative-starship",
            "starship_tracker_log.json",
        )
        self._launch_loader = launch_loader
        self.state = TrackerState.load(self.path)
        self.host = TrackerHostState()
        self._lock = threading.RLock()
        self._rng = random.Random()
        self._backoff = 0
        self._have_feed = bool(self.state.vehicles)
        self._feed_url = None
        self._backfill_started = False
        self._leader_token = f"starship-{os.getpid()}-{time.time_ns()}"
        self._leader = False
        self._closure_areas = None
        self._closure_source = "fallback"
        self._closure_mono = 0.0
        self._fixture_run = None
        self._fixture_mode = ""
        self._fixture_step_at = 0.0
        self._fixture_doc = None
        areas, source = closure_areas.load_local()
        self._closure_areas = areas
        self._closure_source = source
        self._restore_state_from_redis()
        # Init must not read the launch cache. The poller fills closures.
        self._tracker_doc = _render(self._current_tracker_payload(_now()))
        self._closure_doc = _render(
            self._current_closure_payload(
                _now(), launches={"upcoming": [], "previous": []}
            )
        )

    def _redis(self):
        import app as app_module

        return getattr(app_module, "r", None)

    def _launch_data(self):
        if self._launch_loader is not None:
            data = self._launch_loader()
            return data if isinstance(data, dict) else {"upcoming": [], "previous": []}
        import app as app_module

        # Cached list only. force=False does not refresh Launch Library.
        data = app_module._load_launch_payload(False)
        return data if isinstance(data, dict) else {"upcoming": [], "previous": []}

    def _restore_state_from_redis(self):
        if self.state.vehicles or self.state.liftoff_utc:
            return
        client = self._redis()
        if not client:
            return
        import app as app_module

        try:
            cached = app_module.get_cached_data(STATE_REDIS_KEY)
        except Exception:
            return
        if not isinstance(cached, dict) or not cached.get("vehicles"):
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(cached, handle)
            self.state = TrackerState.load(self.path)
            self._have_feed = bool(self.state.vehicles)
        except OSError:
            return

    def _persist_state(self):
        self.state.save()
        client = self._redis()
        if not client or not os.path.exists(self.path):
            return
        import app as app_module

        try:
            with open(self.path, encoding="utf-8") as handle:
                blob = json.load(handle)
            app_module.set_cached_data(STATE_REDIS_KEY, blob, ttl=48 * 3600)
        except (OSError, json.JSONDecodeError, Exception):
            return

    def _publish(self, key, payload):
        client = self._redis()
        if not client:
            return
        import app as app_module

        try:
            app_module.set_cached_data(key, payload, ttl=48 * 3600)
        except Exception:
            return

    def _pull_published(self):
        client = self._redis()
        if not client:
            return
        import app as app_module

        try:
            tracker = app_module.get_cached_data(TRACKER_REDIS_KEY)
            closures = app_module.get_cached_data(CLOSURE_REDIS_KEY)
        except Exception:
            return
        with self._lock:
            if isinstance(tracker, dict) and tracker.get("position_source") != "fixture":
                self._tracker_doc = _render(tracker)
            if isinstance(closures, dict):
                self._closure_doc = _render(closures)

    def _ensure_leader(self, seconds):
        client = self._redis()
        ttl = max(int(LOCK_MARGIN_S), int(seconds) + LOCK_MARGIN_S)
        if not client:
            self._leader = True
            return True
        try:
            if self._leader:
                current = client.get(LOCK_REDIS_KEY)
                if current == self._leader_token:
                    client.expire(LOCK_REDIS_KEY, ttl)
                    return True
                self._leader = False
            ok = client.set(LOCK_REDIS_KEY, self._leader_token, nx=True, ex=ttl)
            self._leader = bool(ok)
            return self._leader
        except Exception:
            # A Redis blip must not stop the only dyno from polling.
            self._leader = True
            return True

    def _done_by_key(self):
        found = {}
        for key, slot in (self.state.vehicles or {}).items():
            if isinstance(slot, dict):
                found[key] = bool(slot.get("done"))
        return found

    def _current_tracker_payload(self, now):
        raw = self.state.globe_payload(now, in_window=self.state.wants_fast_poll(now))
        source = "feed" if self._have_feed else "none"
        return annotate_tracker_payload(
            raw,
            source=source,
            done_by_key=self._done_by_key(),
            feed_url=self._feed_url,
            generated_at=_iso(now),
        )

    def _current_closure_payload(self, now, launches=None):
        if launches is None:
            try:
                launches = self._launch_data()
            except Exception:
                launches = {"upcoming": [], "previous": []}
        payload = closure_areas.globe_payload(
            launches,
            self.state,
            now,
            areas=self._closure_areas,
            source=self._closure_source,
        )
        payload = dict(payload)
        payload["generated_at"] = _iso(now)
        payload["kind"] = "closure"
        # Fallback rings are the Flight 15 notices, not a vehicle position.
        payload["fallback"] = payload.get("source") == "fallback"
        return payload

    def _maybe_backfill(self, now, get=None):
        """One history read after a mid-flight start. Not on the request path."""
        if self._backfill_started:
            return
        liftoff = self.state.liftoff_utc
        if liftoff is None or not self.state.net_passed:
            return
        if not ship_trail_is_partial(self.state, liftoff, now):
            self._backfill_started = True
            return
        self._backfill_started = True

        def _work():
            # name="starship-backfill" — off the tracker poll thread.
            try:
                grouped = fetch_flight_history(now, get=get)
            except Exception:
                return
            if not grouped:
                return
            with self._lock:
                self.state.apply_backfill(grouped, now)
                self._persist_state()
                payload = self._current_tracker_payload(_now())
                self._tracker_doc = _render(payload)
            self._publish(TRACKER_REDIS_KEY, payload)

        threading.Thread(target=_work, name="starship-backfill", daemon=True).start()

    def refresh_closures(self, get=None, now=None):
        """FAA + NGA. Failures keep the Flight 15 fallback rings."""
        clock = now or _now()
        try:
            areas, source = closure_areas.refresh(get=get, now=clock)
        except Exception:
            return
        with self._lock:
            self._closure_areas = areas
            self._closure_source = source
            self._closure_mono = time.monotonic()
            payload = self._current_closure_payload(clock)
            self._closure_doc = _render(payload)
        self._publish(CLOSURE_REDIS_KEY, payload)

    def poll_once(self, now=None, get=None, refresh_closures=True):
        """One upstream cycle. Returns the sleep until the next poll.

        Ingests a document only when a feed returned one. An empty or failed
        poll leaves the last feed fix in place and never fills in a position.
        """
        clock = now or _now()
        launches = self._launch_data()
        with self._lock:
            self.state.update_mission(launches, clock)
            allow_alternate = bool(self.state.net_passed)
            in_window = self.state.wants_fast_poll(clock)
        result = fetch_with_failover(
            self.host,
            now=clock,
            get=get,
            allow_alternate=allow_alternate,
        )
        result = result if isinstance(result, dict) else {}
        status = result.get("status")
        if status in (403, 429, 503):
            waited = http_backoff_s(
                status,
                result.get("retry_after"),
                previous=self._backoff,
                now=clock,
            )
            self._backoff = waited or 0
        else:
            self._backoff = 0
        body = result.get("body")
        with self._lock:
            if body:
                # Only a document the feed actually returned. No filled-in fix.
                self.state.ingest(body, clock)
                self._have_feed = True
                self._feed_url = result.get("url") or self.host.source
            if self.state.liftoff_utc or self._have_feed:
                self._maybe_backfill(clock, get=get)
            if body or self.state.vehicles or self.state.liftoff_utc:
                self._persist_state()
            payload = self._current_tracker_payload(clock)
            self._tracker_doc = _render(payload)
            closure_payload = self._current_closure_payload(clock, launches=launches)
            self._closure_doc = _render(closure_payload)
        self._publish(TRACKER_REDIS_KEY, payload)
        self._publish(CLOSURE_REDIS_KEY, closure_payload)
        if refresh_closures and (time.monotonic() - self._closure_mono) >= CLOSURE_REFRESH_S:
            self._closure_mono = time.monotonic()
            threading.Thread(
                target=self.refresh_closures,
                kwargs={"now": clock},
                name="starship-closures",
                daemon=True,
            ).start()
        feed_hot = self.state.live_layer_active(clock) or self.state.wants_fast_poll(clock)
        # Hobby mirrors stay on their own 30s gate inside fetch_with_failover.
        # The outer cadence stays 10s so SpaceX is tried again on the next poll.
        return poll_interval_s(
            in_window=in_window or feed_hot,
            feed_hot=feed_hot,
            backoff_s=self._backoff,
            rng=self._rng,
        )

    def tracker_payload(self):
        with self._lock:
            raw = self._tracker_doc["json"]
        return json.loads(raw)

    def closure_payload(self):
        with self._lock:
            raw = self._closure_doc["json"]
        return json.loads(raw)

    def _fixture_mode_now(self):
        mode = env_mode()
        return mode or "loop"

    def fixture_payload(self, mode=None):
        """Advance FlightTestRun at most once per wall-clock tick. Not a live fix."""
        selected = mode or self._fixture_mode_now()
        laps = flight_test.LONG_LAPS if selected == "laps" else 1
        with self._lock:
            run = self._fixture_run
            restart = (
                run is None
                or self._fixture_mode != selected
                or (run.finished and selected in ("loop", "laps"))
            )
            if restart:
                run = FlightTestRun(laps=laps)
                self._fixture_run = run
                self._fixture_mode = selected
                self._fixture_step_at = 0.0
            now_mono = time.monotonic()
            if self._fixture_doc is None or (not run.finished and now_mono - self._fixture_step_at >= flight_test.TICK_S):
                globe = run.step()
                self._fixture_step_at = now_mono
                done = {}
                for key, slot in run.state.vehicles.items():
                    done[key] = bool(slot.get("done"))
                annotated = annotate_tracker_payload(
                    globe,
                    source="fixture",
                    done_by_key=done,
                    generated_at=_iso(run.virtual_now()),
                )
                annotated["clock_rate"] = globe.get("clock_rate")
                self._fixture_doc = _render(annotated)
            raw = self._fixture_doc["json"]
        return json.loads(raw)

    def rendered_tracker(self, fixture=False):
        if fixture:
            self.fixture_payload()
            with self._lock:
                return dict(self._fixture_doc)
        with self._lock:
            return dict(self._tracker_doc)

    def rendered_closures(self):
        with self._lock:
            return dict(self._closure_doc)


def get_service():
    global _service
    with _service_lock:
        if _service is None:
            _service = StarshipService()
        return _service


def reset_service_for_tests(service=None):
    global _service
    with _service_lock:
        _service = service


def _accepts_gzip(request: Request) -> bool:
    if request is None:
        return False
    return "gzip" in (request.headers.get("accept-encoding") or "").lower()


def _json_response(rendered, request: Request, cache_control: str, source: str):
    use_gzip = _accepts_gzip(request)
    content = rendered["gzip"] if use_gzip else rendered["json"]
    headers = {
        "Cache-Control": cache_control,
        "Vary": "Accept-Encoding",
        "X-Starship-Source": source,
    }
    if use_gzip:
        headers["Content-Encoding"] = "gzip"
    return Response(content=content, media_type="application/json", headers=headers)


def _fixture_requested(fixture: str) -> bool:
    text = str(fixture or "").strip().lower()
    return text in ("flight_test", "flight-test", "1", "true", "yes")


def install(fastapi_app: FastAPI):
    """Register the two read routes. Safe to call once at import."""

    @fastapi_app.get(
        "/starship/tracker",
        tags=["Starship"],
        summary="Cached Starship ship and booster positions",
    )
    def get_starship_tracker(
        request: Request,
        fixture: str = Query(
            default=None,
            description=(
                "Dev-only. `flight_test` serves FlightTestRun through this "
                "same shape. The body is marked position_source=fixture and "
                "live_positions=false. Omit it for the polled public feeds."
            ),
        ),
    ):
        import app as app_module

        app_module.increment_metric("total_requests")
        app_module.increment_metric("cache_hits")
        service = get_service()
        use_fixture = _fixture_requested(fixture) or bool(env_mode())
        # Env STARSHIP_TRACKER_TEST replaces the live body on a dev dyno.
        # A query fixture does too, and is never stored as the live document.
        if env_mode() and not _fixture_requested(fixture):
            use_fixture = True
        rendered = service.rendered_tracker(fixture=use_fixture)
        cache = FIXTURE_CACHE_CONTROL if use_fixture else TRACKER_CACHE_CONTROL
        source = "fixture" if use_fixture else "cache"
        return _json_response(rendered, request, cache, source)

    @fastapi_app.get(
        "/starship/closures",
        tags=["Starship"],
        summary="Cached FAA and maritime closure polygons",
    )
    def get_starship_closures(request: Request):
        import app as app_module

        app_module.increment_metric("total_requests")
        app_module.increment_metric("cache_hits")
        service = get_service()
        return _json_response(
            service.rendered_closures(),
            request,
            CLOSURE_CACHE_CONTROL,
            "cache",
        )


def _poller_enabled():
    flag = os.getenv("STARSHIP_POLLER", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    import app as app_module

    return bool(getattr(app_module, "_background_enabled", False))


def start_starship_poller():
    """Daemon poller. One leader when Redis is up; otherwise this dyno."""
    global _poller_started
    if _poller_started or not _poller_enabled():
        return
    _poller_started = True
    service = get_service()

    def _loop():
        # Hold the lock across the first sleep so a second dyno does not
        # also poll. Idle is 5 min; fast is 10s plus jitter.
        if not service._ensure_leader(ACTIVE_POLL_S):
            service._pull_published()
        while _poller_enabled():
            try:
                if service._ensure_leader(ACTIVE_POLL_S):
                    interval = service.poll_once()
                else:
                    service._pull_published()
                    interval = 5.0
            except Exception as exc:
                print(f"Starship poller error: {exc}")
                interval = float(IDLE_POLL_S)
            service._ensure_leader(interval)
            time.sleep(interval)

    threading.Thread(target=_loop, name="starship-tracker", daemon=True).start()
