"""Public Starship tracker: parse the real sample, drop pad garbage, keep the trail."""

import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from starship_tracker import (
    ACTIVE_POLL_S,
    EARTH_ROTATION_RAD_S,
    FALLBACK_TRACKER_URL,
    GROUNDED_ALT_M,
    GROUNDED_SPEED_MPS,
    GROUNDED_STABLE_S,
    GPS_LEAP_SECONDS,
    IDLE_POLL_S,
    MAX_TRAIL_POINTS,
    MIN_POLL_S,
    ORBIT_COAST_SPEED_MPS,
    BEYOND_STAGE_ZERO_URL,
    HOBBY_POLL_S,
    HOBBY_STALE_S,
    PRIMARY_TRACKER_URL,
    STARDASH_FEED_URL,
    TrackerHostState,
    TrackerState,
    _downsample_trail,
    classify_vehicle,
    fetch_public_tracker,
    fetch_with_failover,
    format_altitude_km,
    geodetic_to_ecef,
    format_speed_kmh,
    format_webcast_clock,
    gps_to_utc,
    http_backoff_s,
    format_apsis_label,
    inertial_revolutions,
    inertial_speed_mps,
    osculating_apsides_m,
    ship_apsis_label,
    launch_window_active,
    parse_feed,
    parse_retry_after,
    poll_interval_s,
    pre_launch_window,
    accept_hobby_document,
    estimated_orbit_number,
    fill_inertial_speeds,
    _is_grounded,
    orbital_period_s,
    predicted_is_sane,
    replay_feeds,
    BSZ_SHIP_TRAIL_URL,
    fetch_flight_history,
    parse_stardash_history,
    rows_to_trail,
    sample_in_coast,
    ship_completed_laps,
    ship_orbit_status,
    ship_trail_is_partial,
    stardash_feed_to_tracker,
    stardash_history_url,
    synthetic_inflight_feeds,
    utc_to_gps,
    tracker_user_agent,
    webcast_identity,
)


class _ZeroJitter:
    def uniform(self, _low, _high):
        return 0.0

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "tests" / "fixtures" / "starship_tracker_sample.json"


def _sample():
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


def test_gps_epoch_subtracts_leap_seconds():
    gps = 1_478_542_958.295974
    utc = gps_to_utc(gps)
    from starship_tracker import GPS_EPOCH
    assert utc == GPS_EPOCH + timedelta(seconds=gps - GPS_LEAP_SECONDS)
    assert utc.tzinfo is not None
    assert utc.year == 2026


def test_sample_classifies_ship_and_rejects_pad_trajectory():
    body = _sample()
    assert "ship42" in body
    assert classify_vehicle("ship42") == "ship"
    assert classify_vehicle("booster19") == "booster"
    assert classify_vehicle("metadata") == "other"
    parsed = parse_feed(body, gps_to_utc(body["metadata"]["generation_time"]))
    assert parsed["stale"] is False
    assert len(parsed["vehicles"]) == 1
    ship = parsed["vehicles"][0]
    assert ship["key"] == "ship42"
    assert ship["name"] == "Ship 42"
    assert ship["speed_mps"] == 0
    assert ship["predicted_sane"] is False
    assert ship["predicted"] == []
    assert predicted_is_sane([]) is False


def test_sample_is_stale_a_minute_later_and_hidden_on_the_pad():
    body = _sample()
    generated = gps_to_utc(body["metadata"]["generation_time"])
    stale = parse_feed(body, generated + timedelta(seconds=90))
    assert stale["stale"] is True
    state = TrackerState(path="/tmp/starship-tracker-test-unused.json")
    state.save = lambda path=None: None
    state.ingest(body, generated)
    payload = state.globe_payload(generated, in_window=False)
    assert payload["show"] is False
    assert payload["vehicles"] == []


def _quiet_state(path="/tmp/starship-tracker-test-unused.json"):
    state = TrackerState(path=path)
    state.save = lambda path=None: None
    return state


def _launch(net, rocket="Starship"):
    return {"upcoming": [{"net": net.isoformat(), "rocket": rocket}], "previous": []}


def _flying(alt, speed, **extra):
    slot = {
        "in_flight": True,
        "done": False,
        "done_reason": None,
        "missing": False,
        "missing_since": None,
        "stable_since": None,
        "moving": False,
        "current": {"alt_m": alt, "speed_mps": speed, "lat": 5.0, "lon": 80.0, "mission_time_s": 4000},
        "samples": [
            {"lat": 26.0, "lon": -97.0, "alt_m": 8000, "speed_mps": 500, "mission_time_s": 60, "fresh": True},
            {"lat": 10.0, "lon": 40.0, "alt_m": 20000, "speed_mps": 1500, "mission_time_s": 400, "fresh": True},
        ],
    }
    slot.update(extra)
    return slot


def _cadence(state, launch_data, now, backoff=0):
    state.update_mission(launch_data, now)
    zero = _ZeroJitter()
    return poll_interval_s(
        in_window=state.wants_fast_poll(now),
        feed_hot=False,
        backoff_s=backoff,
        rng=zero,
    )


def test_launch_window_and_poll_cadence():
    now = gps_to_utc(1_478_542_958)
    early = _launch(now + timedelta(minutes=31))
    edge = _launch(now + timedelta(minutes=30))
    later = _launch(now + timedelta(hours=2))
    assert pre_launch_window(early, now) is False
    assert launch_window_active(early, now) is False
    assert pre_launch_window(edge, now) is True
    assert launch_window_active(later, now) is False
    zero = _ZeroJitter()
    assert poll_interval_s(in_window=True, feed_hot=False, rng=zero) == ACTIVE_POLL_S
    assert poll_interval_s(in_window=False, feed_hot=False, rng=zero) == IDLE_POLL_S
    # Retry-After / backoff still replaces the 10s rate.
    assert poll_interval_s(in_window=True, feed_hot=False, backoff_s=45, rng=zero) == 45
    assert poll_interval_s(in_window=False, feed_hot=False, backoff_s=1000, rng=zero) == IDLE_POLL_S


def test_mission_stays_active_until_every_vehicle_is_down():
    """T-30 through mission end. A stale orbit does not end the flight."""
    now = gps_to_utc(1_478_542_958)
    none = {"upcoming": [], "previous": []}

    before = _quiet_state()
    assert _cadence(before, _launch(now + timedelta(minutes=31)), now) == IDLE_POLL_S
    at_window = _quiet_state()
    assert _cadence(at_window, _launch(now + timedelta(minutes=30)), now) == ACTIVE_POLL_S

    orbit = _quiet_state()
    orbit.liftoff_utc = now - timedelta(hours=8)
    orbit.liftoff_from_motion = True
    orbit.feed_stale = True
    orbit.vehicles["ship42"] = _flying(220_000, 7600, last_seen=(now - timedelta(hours=3)).isoformat())
    assert _cadence(orbit, none, now) == ACTIVE_POLL_S
    assert orbit.vehicles["ship42"]["done"] is False

    mixed = _quiet_state()
    mixed.liftoff_utc = now - timedelta(hours=2)
    mixed.liftoff_from_motion = True
    mixed.vehicles["booster1"] = _flying(
        120, 0, done=True, done_reason="landed", stable_since=(now - timedelta(minutes=20)).isoformat()
    )
    mixed.vehicles["ship42"] = _flying(180_000, 7500)
    assert _cadence(mixed, none, now) == ACTIVE_POLL_S

    down = _quiet_state()
    down.liftoff_utc = now - timedelta(hours=12)
    down.liftoff_from_motion = True
    down.vehicles["booster1"] = _flying(80, 0, done=True, done_reason="landed")
    down.vehicles["ship42"] = _flying(20, 0, done=True, done_reason="landed")
    assert _cadence(down, none, now) == ACTIVE_POLL_S
    assert down.hold_started is True
    after_hold = now + timedelta(hours=1, minutes=1)
    assert _cadence(down, none, after_hold) == IDLE_POLL_S

    vanished = _quiet_state()
    vanished.liftoff_utc = now - timedelta(hours=3)
    vanished.liftoff_from_motion = True
    vanished.vehicles["ship42"] = _flying(
        200_000,
        7500,
        missing=True,
        missing_since=(now - timedelta(minutes=31)).isoformat(),
    )
    assert _cadence(vanished, none, now) == ACTIVE_POLL_S
    assert vanished.vehicles["ship42"]["done_reason"] == "vanished"

    brief_gap = _quiet_state()
    brief_gap.liftoff_utc = now - timedelta(hours=3)
    brief_gap.liftoff_from_motion = True
    brief_gap.vehicles["ship42"] = _flying(
        200_000,
        7500,
        missing=True,
        missing_since=(now - timedelta(minutes=10)).isoformat(),
    )
    assert _cadence(brief_gap, none, now) == ACTIVE_POLL_S
    assert brief_gap.vehicles["ship42"].get("done") is not True

    settling = _quiet_state()
    settling.liftoff_utc = now - timedelta(hours=1)
    settling.liftoff_from_motion = True
    settling.vehicles["booster1"] = _flying(
        100, 0, stable_since=(now - timedelta(minutes=9)).isoformat()
    )
    settling.vehicles["ship42"] = _flying(
        15, 0, stable_since=(now - timedelta(minutes=11)).isoformat()
    )
    assert _cadence(settling, none, now) == ACTIVE_POLL_S
    assert settling.vehicles["ship42"]["done"] is True
    assert settling.vehicles["booster1"].get("done") is not True
    settling.vehicles["booster1"]["stable_since"] = (now - timedelta(minutes=10)).isoformat()
    assert _cadence(settling, none, now) == ACTIVE_POLL_S
    assert settling.hold_started is True

    capped = _quiet_state()
    capped.liftoff_utc = now - timedelta(hours=48, seconds=5)
    capped.liftoff_from_motion = True
    capped.vehicles["ship42"] = _flying(200_000, 7500)
    assert _cadence(capped, none, now) == IDLE_POLL_S
    assert capped.cap_logged is True


def test_reboot_resumes_fast_poll_and_a_slip_moves_the_window(tmp_path):
    now = gps_to_utc(1_478_542_958)
    path = tmp_path / "starship_tracker_log.json"
    flying = TrackerState(path=str(path))
    flying.liftoff_utc = now - timedelta(hours=6)
    flying.liftoff_from_motion = True
    flying.vehicles["ship42"] = _flying(150_000, 7400)
    flying.save()
    resumed = TrackerState.load(str(path))
    resumed.save = lambda path=None: None
    assert _cadence(resumed, {"upcoming": [], "previous": []}, now) == ACTIVE_POLL_S
    assert resumed.liftoff_utc == flying.liftoff_utc

    slipped = _quiet_state()
    assert _cadence(slipped, _launch(now + timedelta(hours=2)), now) == IDLE_POLL_S
    assert _cadence(slipped, _launch(now + timedelta(minutes=10)), now) == ACTIVE_POLL_S

    scrubbed = _quiet_state()
    assert _cadence(scrubbed, _launch(now - timedelta(minutes=5)), now) == ACTIVE_POLL_S
    assert scrubbed.liftoff_utc is not None
    assert _cadence(scrubbed, _launch(now + timedelta(hours=4)), now) == IDLE_POLL_S
    assert scrubbed.liftoff_utc is None

    falcon = _quiet_state()
    assert _cadence(falcon, _launch(now + timedelta(minutes=10), rocket="Falcon 9"), now) == IDLE_POLL_S


def test_overnight_gap_keeps_the_flown_trail():
    state = _quiet_state()
    start_gps = 1_478_600_000

    def body(offset, lat):
        gps = start_gps + offset
        return {
            "ship42": {
                "current": {
                    "gps_time": gps,
                    "mission_time": 100 + offset,
                    "altitude": 200_000,
                    "speed": 7500,
                    "latitude": lat,
                    "longitude": -90,
                },
                "trajectory": [],
            },
            "metadata": {"generation_time": gps},
        }

    state.ingest(body(0, 26), gps_to_utc(start_gps))
    state.ingest(body(10 * 3600, 10), gps_to_utc(start_gps + 10 * 3600))
    samples = state.vehicles["ship42"]["samples"]
    assert [point["lat"] for point in samples] == [26, 10]


def test_long_trail_downsamples_old_points_instead_of_dropping_them():
    samples = [{"lat": float(i), "lon": 0.0, "alt_m": float(i)} for i in range(2500)]
    thinned = _downsample_trail(samples, MAX_TRAIL_POINTS)
    assert len(thinned) <= MAX_TRAIL_POINTS
    assert thinned[0]["lat"] == 0
    assert thinned[-1]["lat"] == 2499
    assert any(0 < point["lat"] < 400 for point in thinned)


def test_synthetic_flight_keeps_both_trails_through_catch():
    feeds = synthetic_inflight_feeds()
    _state, inflight, catch = replay_feeds(feeds)
    assert inflight["show"] is True
    by_kind = {item["kind"]: item for item in inflight["vehicles"]}
    assert set(by_kind) == {"ship", "booster"}
    for kind in ("ship", "booster"):
        assert len(by_kind[kind]["trail"]) >= 10
        assert by_kind[kind]["predicted"]
    # Ship moves east, booster stays near Starbase and comes home.
    ship = by_kind["ship"]["trail"]
    booster = by_kind["booster"]["trail"]
    assert ship[-1]["lon"] > ship[0]["lon"] + 50
    assert abs(booster[-1]["lon"] - booster[0]["lon"]) < 1.5
    assert catch["show"] is True
    catch_kinds = {item["kind"]: item for item in catch["vehicles"]}
    assert len(catch_kinds["ship"]["trail"]) >= 10
    assert len(catch_kinds["booster"]["trail"]) >= 10
    # Catch is immediate; the 1h hold starts only after every vehicle is down.
    assert inflight["hide_planned"] is True
    assert catch["hide_planned"] is True


def test_retry_after_parses_seconds_and_http_date():
    assert parse_retry_after(None) is None
    assert parse_retry_after("  ") is None
    assert parse_retry_after("nope") is None
    assert parse_retry_after("-5") is None
    assert parse_retry_after("120") == 120
    assert parse_retry_after("12.4") == 12.4
    now = datetime(2015, 10, 21, 7, 28, 0, tzinfo=timezone.utc)
    later = "Wed, 21 Oct 2015 07:28:30 GMT"
    assert abs(parse_retry_after(later, now=now) - 30) < 0.01
    earlier = "Wed, 21 Oct 2015 07:27:00 GMT"
    assert parse_retry_after(earlier, now=now) == 0


def test_429_and_503_honor_retry_after_else_exponential():
    now = datetime(2026, 11, 12, 16, 0, tzinfo=timezone.utc)
    assert http_backoff_s(429, "45", previous=10, now=now) == 45
    assert http_backoff_s(503, "Wed, 12 Nov 2026 16:02:00 GMT", previous=10, now=now) == 120
    # A huge Retry-After is capped at the 5 minute idle interval.
    assert http_backoff_s(429, "9999", previous=10, now=now) == IDLE_POLL_S
    assert http_backoff_s(429, None, previous=0) == 20
    assert http_backoff_s(503, "nope", previous=20) == 40
    assert http_backoff_s(429, None, previous=200) == IDLE_POLL_S
    assert http_backoff_s(403, "10") == IDLE_POLL_S
    assert http_backoff_s(500, "10") is None


def test_jitter_stays_inside_the_active_and_idle_windows():
    rng = random.Random(0)
    for _ in range(200):
        active = poll_interval_s(in_window=True, feed_hot=False, rng=rng)
        idle = poll_interval_s(in_window=False, feed_hot=False, rng=rng)
        assert ACTIVE_POLL_S - 2 <= active <= ACTIVE_POLL_S + 2
        assert active >= MIN_POLL_S
        assert IDLE_POLL_S - 30 <= idle <= IDLE_POLL_S + 30
        assert idle >= MIN_POLL_S
    assert poll_interval_s(in_window=True, feed_hot=False, backoff_s=1, rng=rng) == MIN_POLL_S
    # A Retry-After wait is not smeared by the cadence jitter.
    assert poll_interval_s(in_window=True, feed_hot=True, backoff_s=45, rng=rng) == 45


def _response(status, body=None, headers=None):
    response = MagicMock()
    response.status_code = status
    response.headers = headers or {}
    response.json.return_value = body or {"ok": True}
    response.raise_for_status.side_effect = None if status < 400 else RuntimeError(f"HTTP {status}")
    if status >= 400 and status not in (403, 429, 503):
        response.raise_for_status.side_effect = RuntimeError(f"HTTP {status}")
    return response


def _host_fix(marker=None):
    """A document with one plottable fix, so failover treats the host as up."""
    current = {
        "gps_time": 1_478_600_000,
        "mission_time": 120,
        "altitude": 250_000,
        "speed": 7600,
        "latitude": -20.0,
        "longitude": 150.0,
    }
    if marker is not None:
        current["marker"] = marker
    return {
        "ship42": {"current": current},
        "metadata": {"generation_time": 1_478_600_000},
    }


def test_tracker_failover_sticks_then_returns_to_primary():
    now = datetime(2026, 11, 13, 18, 0, tzinfo=timezone.utc)
    state = TrackerHostState()
    calls = []

    def get(url, headers=None, timeout=None):
        calls.append((url, dict(headers or {})))
        if url == PRIMARY_TRACKER_URL:
            raise TimeoutError("timed out")
        return _response(200, _host_fix(), {"ETag": "edge", "Last-Modified": "edge-time"})

    result = fetch_with_failover(state, now=now, get=get)
    assert result["body"] == _host_fix()
    assert result["url"] == FALLBACK_TRACKER_URL
    assert [url for url, _headers in calls] == [PRIMARY_TRACKER_URL, FALLBACK_TRACKER_URL]
    assert state.preferred == FALLBACK_TRACKER_URL

    calls.clear()
    again = fetch_with_failover(state, now=now + timedelta(minutes=1), get=get)
    assert again["url"] == FALLBACK_TRACKER_URL
    assert [url for url, _headers in calls] == [FALLBACK_TRACKER_URL]
    assert calls[0][1]["If-None-Match"] == "edge"

    def recovered(url, headers=None, timeout=None):
        calls.append((url, dict(headers or {})))
        if url == PRIMARY_TRACKER_URL:
            return _response(200, _host_fix("back"), {"ETag": "primary"})
        return _response(200, _host_fix(), {"ETag": "edge"})

    calls.clear()
    back = fetch_with_failover(state, now=now + timedelta(minutes=11), get=recovered)
    assert back["url"] == PRIMARY_TRACKER_URL
    assert state.preferred == PRIMARY_TRACKER_URL
    assert calls[0][1].get("If-None-Match") != "edge"

    calls.clear()

    def not_modified(url, headers=None, timeout=None):
        calls.append(url)
        return _response(304)

    stayed = fetch_with_failover(state, now=now + timedelta(minutes=12), get=not_modified)
    assert stayed["not_modified"] is True
    assert calls == [PRIMARY_TRACKER_URL]


def test_fetch_sends_conditional_headers_and_user_agent():
    response = MagicMock()
    response.status_code = 429
    response.headers = {"Retry-After": "15", "ETag": "abc"}
    with patch("starship_tracker.requests.get", return_value=response) as get:
        result = fetch_public_tracker(etag='"v1"', modified="Wed, 12 Nov 2026 00:00:00 GMT")
    assert result["status"] == 429
    assert result["retry_after"] == "15"
    kwargs = get.call_args.kwargs
    headers = kwargs["headers"]
    assert headers["If-None-Match"] == '"v1"'
    assert headers["If-Modified-Since"] == "Wed, 12 Nov 2026 00:00:00 GMT"
    agent = headers["User-Agent"]
    assert agent.startswith("spacex-dashboard/")
    assert "(+https://github.com/hwpaige/spacex-dashboard)" in agent
    assert tracker_user_agent() == agent


def _tracker_body(gps, lat, lon, alt, speed, predicted=None, key="ship42", mission_time=30):
    return {
        key: {
            "current": {
                "gps_time": gps,
                "mission_time": mission_time,
                "altitude": alt,
                "speed": speed,
                "latitude": lat,
                "longitude": lon,
            },
            "trajectory": predicted or [],
        },
        "metadata": {"generation_time": gps},
    }


def _sane_prediction(gps, lat, lon, alt):
    return [
        {"time": gps + 60, "latitude": lat + 0.1, "longitude": lon + 0.2, "altitude": alt + 200},
        {"time": gps + 120, "latitude": lat + 0.2, "longitude": lon + 0.4, "altitude": alt + 400},
    ]


def test_planned_trajectory_hides_at_t0_before_two_fresh_samples():
    """Planned path drops at T-0. The live layer still waits for two fresh points."""
    gps = 1_478_600_000
    now = gps_to_utc(gps)
    none = {"upcoming": [], "previous": []}

    parked = _quiet_state()
    parked.ingest(_tracker_body(gps, 25.99, -97.15, 120, 0), now)
    parked.update_mission(_launch(now + timedelta(minutes=20)), now)
    parked_payload = parked.globe_payload(now, in_window=True)
    assert parked.wants_fast_poll(now) is True
    assert parked_payload["show"] is False
    assert parked_payload["hide_planned"] is False

    moving = _quiet_state()
    moving.ingest(_tracker_body(gps, 26.2, -97.4, 8000, 400), now)
    early = moving.globe_payload(now, in_window=False)
    # One airborne sample, and no launch window, is not a liftoff.
    assert moving.liftoff_utc is None
    assert early["show"] is False
    assert early["hide_planned"] is False

    waiting = _quiet_state()
    waiting.update_mission(_launch(now - timedelta(minutes=5)), now)
    assert waiting.liftoff_utc is not None
    at_net = waiting.globe_payload(now, in_window=True)
    assert at_net["hide_planned"] is True
    assert at_net["launch_active"] is False
    assert at_net["show"] is False
    waiting.ingest(
        _tracker_body(gps, 26.0, -97.0, 5000, 300, _sane_prediction(gps, 26.0, -97.0, 5000)),
        now,
    )
    one = waiting.globe_payload(now, in_window=True)
    assert one["hide_planned"] is True
    assert one["launch_active"] is False

    second_gps = gps + 30
    second = gps_to_utc(second_gps)
    waiting.ingest(
        _tracker_body(
            second_gps, 26.4, -96.2, 20000, 900, _sane_prediction(second_gps, 26.4, -96.2, 20000)
        ),
        second,
    )
    waiting.update_mission(_launch(now - timedelta(minutes=5)), second)
    handed = waiting.globe_payload(second, in_window=waiting.wants_fast_poll(second))
    assert handed["hide_planned"] is True
    assert handed["show"] is True
    assert len(handed["vehicles"][0]["trail"]) >= 2
    assert len(handed["vehicles"][0]["predicted"]) == 2

    insane = _quiet_state()
    bad_pred = [
        {"time": gps + 60, "latitude": 26.0, "longitude": -97.0, "altitude": -4_700_000},
        {"time": gps + 120, "latitude": -2.0, "longitude": 110.0, "altitude": -4_600_000},
    ]
    insane.ingest(_tracker_body(gps, 26.0, -97.0, 5000, 300, bad_pred), now)
    insane.ingest(_tracker_body(second_gps, 26.5, -96.0, 22000, 900, bad_pred), second)
    insane.update_mission(_launch(now - timedelta(minutes=5)), second)
    dropped = insane.globe_payload(second, in_window=True)
    assert dropped["hide_planned"] is True
    assert dropped["vehicles"][0]["predicted"] == []

    stale_now = gps_to_utc(gps) + timedelta(seconds=180)
    stale = _quiet_state()
    stale.ingest(_tracker_body(gps, 26.0, -97.0, 5000, 300), stale_now)
    stale.ingest(_tracker_body(second_gps, 27.0, -95.0, 30000, 900), stale_now)
    stale.update_mission(_launch(stale_now - timedelta(minutes=10)), stale_now)
    assert stale.feed_stale is True
    assert stale.fresh_sample_count(stale.vehicles["ship42"]) == 0
    # NET has passed, so the path is hidden even though the samples are too
    # stale to count as liftoff. The live layer stays off.
    stale_payload = stale.globe_payload(stale_now, in_window=True)
    assert stale_payload["hide_planned"] is True
    assert stale_payload["launch_active"] is False

    later = second + timedelta(seconds=180)
    waiting.ingest(_tracker_body(second_gps, 26.8, -95.5, 40000, 1200), later)
    waiting.update_mission(none, later)
    assert waiting.feed_stale is True
    assert waiting.replaces_planned_trajectory(later) is True
    assert waiting.globe_payload(later, in_window=True)["hide_planned"] is True

    scrubbed = _quiet_state()
    scrubbed.update_mission(_launch(now - timedelta(minutes=2)), now)
    assert scrubbed.liftoff_utc is not None
    assert scrubbed.globe_payload(now, in_window=True)["hide_planned"] is True
    # A hold that moves NET later brings the planned path back.
    scrubbed.update_mission(_launch(now + timedelta(hours=3)), now)
    assert scrubbed.liftoff_utc is None
    assert scrubbed.globe_payload(now, in_window=False)["hide_planned"] is False


def test_webcast_strip_converts_meters_per_second_and_hides_on_the_pad():
    """Feed speed is m/s. The strip shows km/h and km, and only after liftoff."""
    assert format_speed_kmh(1000) == "3,600"
    assert format_speed_kmh(1500) == "5,400"
    assert format_speed_kmh(1500)[1] == ","
    assert format_speed_kmh(None) == "—"
    assert "." not in format_speed_kmh(1500)
    assert format_altitude_km(12540) == "12.5"
    assert format_altitude_km(1_234_500) == "1,234.5"
    assert format_altitude_km(200000) == "200.0"
    assert format_webcast_clock(6 * 60 + 35) == "T+00:06:35"
    assert format_webcast_clock(-15) == "T-00:00:15"
    assert webcast_identity("booster19", "booster") == ("SUPER HEAVY", "B19")
    assert webcast_identity("ship42", "ship") == ("STARSHIP", "S42")

    gps = 1_478_600_000
    now = gps_to_utc(gps)
    parked = _quiet_state()
    parked.ingest(_tracker_body(gps, 25.997, -97.156, 120, 0, mission_time=0), now)
    parked.update_mission(_launch(now + timedelta(hours=30)), now)
    hidden = parked.globe_payload(now)
    assert hidden["telemetry"]["show"] is False
    assert hidden["telemetry"]["booster"]["speed_kmh"] == "—"
    assert hidden["telemetry"]["booster"]["alt_km"] == "—"

    flying = _quiet_state()
    first = _tracker_body(gps, 26.2, -96.4, 12000, 1000, key="booster19", mission_time=40)
    first.update(_tracker_body(gps, 25.6, -94.0, 20000, 1800, key="ship42", mission_time=40))
    flying.ingest(first, now)
    later = gps_to_utc(gps + 40)
    second = _tracker_body(gps + 40, 26.4, -96.0, 25000, 1500, key="booster19", mission_time=80)
    second.update(_tracker_body(gps + 40, 25.2, -88.0, 80000, 2500, key="ship42", mission_time=80))
    flying.ingest(second, later)
    flying.update_mission(_launch(now - timedelta(minutes=5)), later)
    shown = flying.globe_payload(later)
    assert shown["telemetry"]["show"] is True
    assert shown["telemetry"]["clock"] == "T+00:01:20"
    assert shown["telemetry"]["booster"]["title"] == "SUPER HEAVY"
    assert shown["telemetry"]["booster"]["code"] == "B19"
    assert shown["telemetry"]["booster"]["speed_kmh"] == "5,400"
    assert shown["telemetry"]["booster"]["alt_km"] == "25.0"
    assert shown["telemetry"]["ship"]["title"] == "STARSHIP"
    assert shown["telemetry"]["ship"]["code"] == "S42"
    assert shown["telemetry"]["ship"]["speed_kmh"] == "9,000"
    assert shown["telemetry"]["ship"]["alt_km"] == "80.0"
    assert shown["telemetry"]["ship"]["color"] == "#515ADB"
    assert shown["telemetry"]["booster"]["color"] == "#FF9F1C"

    flying.feed_stale = True
    stale = flying.globe_payload(later)
    assert stale["telemetry"]["ship"]["stale"] is True
    assert stale["telemetry"]["ship"]["speed_kmh"] == "9,000"



def test_on_pad_feed_a_day_before_net_keeps_the_planned_path():
    """Ship 42 sitting on the pad must not hide the planned trajectory or go live."""
    gps = 1_478_600_000
    now = gps_to_utc(gps)
    state = _quiet_state()
    # GPS jitter and a running mission clock, still at pad altitude and speed.
    points = [
        (25.997, -97.156, 120, 0, 0),
        (26.002, -97.151, 140, 3, 0),
        (25.995, -97.160, 110, 8, 38897),
        (25.999, -97.154, 130, 1, 38910),
    ]
    for index, (lat, lon, alt, speed, mission) in enumerate(points):
        sample_gps = gps + index * 10
        state.ingest(
            _tracker_body(sample_gps, lat, lon, alt, speed, mission_time=mission),
            gps_to_utc(sample_gps),
        )
    state.update_mission(_launch(now + timedelta(hours=30)), now)
    payload = state.globe_payload(now, in_window=state.wants_fast_poll(now))
    assert state.wants_fast_poll(now) is False
    assert state.liftoff_from_motion is False
    assert payload["show"] is False
    assert payload["hide_planned"] is False
    assert payload["vehicles"] == []


def test_globe_page_draws_tracker_from_backend_payload():
    """Conditional fetch and a daemon poller. The kiosk HTML is not in this API."""
    fetch = (ROOT / "starship_tracker.py").read_text(encoding="utf-8")
    poller = (ROOT / "starship_http.py").read_text(encoding="utf-8")
    assert "If-None-Match" in fetch
    assert "azureedge" in fetch
    assert 'name="starship-tracker"' in poller
    assert "daemon=True" in poller


def _orbit_projection(speed_kmh=26_800, alt_m=190_000, points=16, start_lon=-60.0, lat=0.0):
    """Points 60s apart along the equator at the given ground speed."""
    km_per_min = speed_kmh / 60.0
    deg_per_min = km_per_min / 111.195
    gps = 1_478_600_000
    return [
        {"lat": lat, "lon": start_lon + i * deg_per_min, "alt_m": alt_m, "time": gps + i * 60}
        for i in range(points)
    ]


def test_orbital_coast_projection_passes_the_sanity_filter():
    projection = _orbit_projection()
    assert predicted_is_sane(projection) is True
    assert predicted_is_sane(projection, 26_800 / 3.6) is True
    # ~28,000 km/h still passes on the floor alone.
    assert predicted_is_sane(_orbit_projection(speed_kmh=28_000)) is True
    # A hemisphere-sized hop between samples is still garbage.
    assert predicted_is_sane(_orbit_projection(speed_kmh=200_000), 26_800 / 3.6) is False
    underground = _orbit_projection()
    underground[3] = dict(underground[3], alt_m=-5_000)
    assert predicted_is_sane(underground, 26_800 / 3.6) is False


def test_orbital_projection_survives_parse_feed():
    gps = 1_478_600_000
    projection = [
        {"time": p["time"], "latitude": p["lat"], "longitude": p["lon"], "altitude": p["alt_m"], "r_ecef": [0, 0, 0]}
        for p in _orbit_projection()
    ]
    body = {
        "ship42": {
            "current": {
                "gps_time": gps, "mission_time": 1800, "altitude": 190_000, "speed": 7444,
                "latitude": 0.0, "longitude": -60.0, "r_ecef": [0, 0, 0],
            },
            "trajectory": projection,
        },
        "metadata": {"generation_time": gps, "trajectory_version": 1, "coordinate_frame": "ECEF", "sample_period": 60},
    }
    ship = parse_feed(body, gps_to_utc(gps))["vehicles"][0]
    assert ship["predicted_sane"] is True
    assert len(ship["predicted"]) == 16
    # The real on-pad sample is still rejected.
    pad = parse_feed(_sample(), gps_to_utc(_sample()["metadata"]["generation_time"]))["vehicles"][0]
    assert pad["predicted_sane"] is False


FLIGHT15_NET = datetime(2026, 11, 13, 22, 45, tzinfo=timezone.utc)


def _stale_log(path, liftoff, *, airborne=True, hold_until=None):
    """A tracker log as older builds left it on disk (no hold_started)."""
    pad = {"lat": 25.997, "lon": -97.156, "alt_m": 120, "speed_mps": 0,
           "mission_time_s": 38897, "fresh": True, "utc": liftoff.isoformat()}
    samples = [pad]
    if airborne:
        samples += [
            {"lat": 26.1, "lon": -96.0, "alt_m": 40_000, "speed_mps": 1200,
             "mission_time_s": 120, "fresh": True, "utc": liftoff.isoformat()},
            {"lat": 25.0, "lon": -80.0, "alt_m": 150_000, "speed_mps": 7000,
             "mission_time_s": 500, "fresh": True, "utc": liftoff.isoformat()},
        ]
    slot = {"kind": "ship", "name": "Ship 42", "max_mission_time": 38897,
            "samples": samples, "current": pad, "last_seen": liftoff.isoformat(),
            "predicted": [], "in_flight": True, "done": False, "done_reason": None,
            "missing": False, "missing_since": None, "moving": False}
    path.write_text(json.dumps({
        "hold_until": hold_until.isoformat() if hold_until else None,
        "vehicles": {"ship42": slot},
        "liftoff_utc": liftoff.isoformat(),
        "liftoff_from_motion": True,
    }), encoding="utf-8")


def _check_planned_path_visible(state, now):
    state.update_mission(_launch(FLIGHT15_NET), now)
    payload = state.globe_payload(now, in_window=state.wants_fast_poll(now))
    assert payload["hide_planned"] is False
    assert payload["launch_active"] is False
    assert payload["show"] is False
    assert state.wants_fast_poll(now) is False
    assert state.liftoff_utc is None


def test_stale_motion_latch_does_not_hide_the_planned_path_a_day_out(tmp_path):
    """Regression: T-26h before Flight 15 the Pi hid the planned trajectory.

    An old on-disk log with a motion latch and airborne-looking samples kept
    the live layer on outside every NET window; the latch held its own
    window open. update_mission now drops a latch no NET explains.
    """
    now = FLIGHT15_NET - timedelta(hours=26)
    log = tmp_path / "starship_tracker_log.json"
    _stale_log(log, datetime(2026, 9, 28, 23, 30, tzinfo=timezone.utc))
    state = TrackerState.load(str(log))
    state.persist = False
    _check_planned_path_visible(state, now)
    # Still clean on the next poll, and the trail shrinks to the pad fix.
    _check_planned_path_visible(state, now + timedelta(seconds=600))
    assert len(state.vehicles["ship42"]["samples"]) == 1


def test_pad_jitter_latch_from_yesterday_with_vehicles_on_the_pad(tmp_path):
    """Ship on the pad, mission clock running, latch saved 2h ago by pad jitter."""
    now = FLIGHT15_NET - timedelta(hours=26)
    log = tmp_path / "starship_tracker_log.json"
    _stale_log(log, now - timedelta(hours=2), airborne=True,
               hold_until=now - timedelta(minutes=30))
    state = TrackerState.load(str(log))
    state.persist = False
    # Older logs never saved hold_started; a saved hold_until means it began.
    assert state.hold_started is True
    gps = 1_478_600_000
    base = gps_to_utc(gps)
    shift = now - base
    for index, (lat, lon, alt, speed) in enumerate([
        (25.997, -97.156, 120, 0), (26.002, -97.151, 140, 3), (25.995, -97.160, 110, 8),
    ]):
        sample_gps = gps + index * 10
        body = _tracker_body(sample_gps, lat, lon, alt, speed, mission_time=38_900 + index * 10)
        state.ingest(body, gps_to_utc(sample_gps) + shift)
    _check_planned_path_visible(state, now)


def test_saved_latch_inside_the_net_window_survives_a_restart(tmp_path):
    """A real mid-flight reboot keeps the flight on screen."""
    log = tmp_path / "starship_tracker_log.json"
    _stale_log(log, FLIGHT15_NET + timedelta(minutes=2))
    state = TrackerState.load(str(log))
    state.persist = False
    now = FLIGHT15_NET + timedelta(minutes=20)
    state.update_mission(_launch(FLIGHT15_NET), now)
    assert state.liftoff_utc is not None
    assert state.live_layer_active(now) is True
    assert state.globe_payload(now, in_window=True)["hide_planned"] is True


def test_stale_latch_then_net_passed_confirms_ship_in_orbit(tmp_path):
    """Dropping the latch must not leave the tracker idle once NET has passed.

    The next launch list has the real NET, the vehicle is already in orbit,
    and two position fixes confirm the flight. The orbit counter uses the
    recovered liftoff, not the short trail gathered after the restart.
    """
    now = FLIGHT15_NET + timedelta(hours=20, minutes=30)
    log = tmp_path / "starship_tracker_log.json"
    _stale_log(log, datetime(2026, 9, 28, 23, 30, tzinfo=timezone.utc))
    state = TrackerState.load(str(log))
    state.persist = False
    state.update_mission(_launch(FLIGHT15_NET + timedelta(days=40)), now)
    assert state.liftoff_utc is None
    assert state.wants_fast_poll(now) is False
    assert state._flight_confirmed() is False

    state.update_mission(_launch(FLIGHT15_NET), now)
    assert state.net_passed is True
    assert state.wants_fast_poll(now) is True
    assert state.liftoff_utc is not None
    assert abs((state.liftoff_utc - FLIGHT15_NET).total_seconds()) < 1
    assert state.live_layer_active(now) is False

    for step in (0, 1):
        moment = now + timedelta(seconds=30 * step)
        state.ingest(
            _tracker_body(
                utc_to_gps(moment), 0.0, 20.0 + 5.0 * step, 250_000, 7600,
                mission_time=70_000 + 30 * step,
            ),
            moment,
        )
    later = now + timedelta(seconds=30)
    state.update_mission(_launch(FLIGHT15_NET), later)
    assert state._flight_confirmed() is True
    assert state.live_layer_active(later) is True
    assert state.wants_fast_poll(later) is True
    payload = state.globe_payload(later, in_window=True)
    assert payload["show"] is True
    assert payload["telemetry"]["in_orbit"] is True
    assert payload["telemetry"]["orbit_number"] >= 13
    assert payload["telemetry"]["orbit_number"] == estimated_orbit_number(
        payload["vehicles"][0], FLIGHT15_NET, later
    )


def test_not_modified_after_net_still_reads_the_hobby_position():
    """A stuck SpaceX 304 must not skip the feeds that actually have the ship."""
    now = datetime(2026, 11, 14, 19, 0, tzinfo=timezone.utc)
    state = TrackerHostState()
    calls = []
    coast = _coast_document(now, age_s=10)

    def get(url, headers=None, timeout=None):
        calls.append(url)
        if url in (PRIMARY_TRACKER_URL, FALLBACK_TRACKER_URL):
            return _response(304, None)
        if url == BEYOND_STAGE_ZERO_URL:
            return _response(200, coast)
        raise AssertionError(url)

    result = fetch_with_failover(state, now=now, get=get, allow_alternate=True)
    assert result.get("body")
    assert BEYOND_STAGE_ZERO_URL in calls
    ship = parse_feed(result["body"], now)["vehicles"][0]
    assert ship["alt_m"] >= 150_000

    calls.clear()
    idle = fetch_with_failover(state, now=now, get=get, allow_alternate=False)
    assert idle.get("not_modified") is True
    assert BEYOND_STAGE_ZERO_URL not in calls


def test_launch_refresh_kicks_the_vehicle_poll():
    """The API poller, not the kiosk, owns the vehicle fetch."""
    poller = (ROOT / "starship_http.py").read_text(encoding="utf-8")
    assert "def start_starship_poller" in poller
    assert 'name="starship-tracker"' in poller
    assert "daemon=True" in poller
    assert "fetch_with_failover" in poller
    assert "allow_alternate" in poller
    assert "net_passed" in poller


def test_inertial_speed_adds_earth_rotation_to_the_ecef_difference():
    """ECEF difference is Earth-relative. The strip wants inertial speed."""
    earlier = {"gps": 1000.0, "ex": 7_000_000.0, "ey": 0.0, "ez": 0.0}
    later = {"gps": 1001.0, "ex": 7_000_000.0, "ey": 3_000.0, "ez": 0.0}
    ecef = 3_000.0
    rotation = EARTH_ROTATION_RAD_S * 7_000_000.0
    speed = inertial_speed_mps(earlier, later)
    # ω × r is not purely parallel to Δr, so allow a fraction of a metre.
    assert abs(speed - (ecef + rotation)) < 0.5
    assert speed > ecef
    # A one-minute gap is still a sample; a three-minute gap is not.
    assert inertial_speed_mps(earlier, dict(later, gps=1060.0)) is not None
    assert inertial_speed_mps(earlier, dict(later, gps=1121.0)) is None


def _coast_document(now, *, age_s=20.0, alt=254_000.0, met=52_000.0, lon=-97.0):
    """SpaceX-shaped hobby document: speed 0, ECEF current and a 60 s step."""
    gps = utc_to_gps(now) - age_s
    radius = 6_371_000.0 + alt
    theta = 60.0 * (7750.0 / radius)
    r0 = [radius, 0.0, 0.0]
    r1 = [radius * math.cos(theta), radius * math.sin(theta), 0.0]
    return {
        "metadata": {
            "generation_time": gps,
            "trajectory_version": 1,
            "coordinate_frame": "ECEF",
            "sample_period": 60,
        },
        "ship42": {
            "current": {
                "gps_time": gps,
                "mission_time": met,
                "altitude": alt,
                "speed": 0,
                "latitude": 0.0,
                "longitude": lon,
                "r_ecef": r0,
            },
            "trajectory": [
                {
                    "time": gps,
                    "latitude": 0.0,
                    "longitude": lon,
                    "altitude": alt,
                    "r_ecef": r0,
                },
                {
                    "time": gps + 60.0,
                    "latitude": 0.0,
                    "longitude": lon + 4.0,
                    "altitude": alt - 400.0,
                    "r_ecef": r1,
                },
            ],
        },
    }


def _stardash_feed(now, *, age_s=20.0, lon=-97.0, met=52_000.0):
    doc = _coast_document(now, age_s=age_s, lon=lon, met=met)
    current = doc["ship42"]["current"]
    traj = doc["ship42"]["trajectory"]
    return {
        "samples": [{
            "veh": "ship42",
            "wall": now.timestamp() - age_s,
            "gps": current["gps_time"],
            "met": met,
            "alt": current["altitude"],
            "spd": 0,
            "lat": current["latitude"],
            "lon": current["longitude"],
            "ecef": geodetic_to_ecef(
                current["latitude"], current["longitude"], current["altitude"]
            ),
            "traj": [
                [item["latitude"], item["longitude"], item["altitude"], item["time"]]
                for item in traj
            ],
        }],
        "status": {
            "generation_time": doc["metadata"]["generation_time"],
            "trajectory_version": 1,
            "connected": True,
        },
    }


def test_zero_speed_tracker_document_gets_inertial_speed_from_ecef():
    """Beyond Stage Zero is the SpaceX shape with speed left at 0."""
    now = datetime(2026, 11, 14, 13, 0, tzinfo=timezone.utc)
    raw = _coast_document(now)
    assert raw["ship42"]["current"]["speed"] == 0
    body = accept_hobby_document(raw, now)
    assert body is not None
    assert body["ship42"]["current"]["speed"] > ORBIT_COAST_SPEED_MPS
    # A speed the real SpaceX feed already filled in is not replaced.
    kept = _coast_document(now)
    kept["ship42"]["current"]["speed"] = 7444
    fill_inertial_speeds(kept)
    assert kept["ship42"]["current"]["speed"] == 7444
    parsed = parse_feed(body, now)
    assert parsed["stale"] is False
    ship = parsed["vehicles"][0]
    assert ship["name"] == "Ship 42"
    assert ship["alt_m"] == 254_000.0
    assert ship["mission_time_s"] == 52_000.0
    assert ship["speed_mps"] > ORBIT_COAST_SPEED_MPS
    assert sample_in_coast(ship)
    assert ship["predicted_sane"] is True
    assert len(ship["predicted"]) == 2


def _landed_hobby_document(now, *, age_s=20.0):
    """Flight 15 after catch: pad fix, speed 0, predicted path into the Earth.

    The live Beyond Stage Zero / StarDash document of 2026-11-14 21:36 UTC
    looked like this. fill_inertial_speeds used to prefer the 60 s underground
    pair and report 3,529 km/h, so the ship never counted as grounded.
    """
    gps = utc_to_gps(now) - age_s
    r0 = [-714814.6875, -5691862.0, 2778792.25]
    # ~60 s predicted step from the real post-landing feed (~17 km underground).
    r1 = [-712793.523991, -5676137.060606, 2771060.500323]
    r2 = [-706524.116631, -5628811.519152, 2747780.223978]
    return {
        "metadata": {
            "generation_time": gps,
            "trajectory_version": 6400,
            "coordinate_frame": "ECEF",
            "sample_period": 60,
        },
        "ship42": {
            "current": {
                "gps_time": gps,
                "mission_time": 82115.623946,
                "altitude": 105.614236,
                "speed": 0,
                "latitude": 25.996827,
                "longitude": -97.158038,
                "r_ecef": r0,
            },
            "trajectory": [
                {
                    "time": gps,
                    "r_ecef": r0,
                    "latitude": 25.996827,
                    "longitude": -97.158038,
                    "altitude": 105.407638,
                },
                {
                    "time": gps + 60.0,
                    "r_ecef": r1,
                    "latitude": 25.996825,
                    "longitude": -97.157578,
                    "altitude": -17533.451893,
                },
                {
                    "time": gps + 120.0,
                    "r_ecef": r2,
                    "latitude": 25.996823,
                    "longitude": -97.154307,
                    "altitude": -70645.388832,
                },
            ],
        },
    }


def test_landed_hobby_document_does_not_inherit_predicted_speed():
    """A ship sitting at Starbase must not show ~3,500 km/h from the pred path."""
    now = datetime(2026, 11, 14, 21, 36, tzinfo=timezone.utc)
    raw = _landed_hobby_document(now)
    assert raw["ship42"]["current"]["speed"] == 0
    body = accept_hobby_document(raw, now)
    assert body is not None
    speed = body["ship42"]["current"]["speed"]
    assert speed <= GROUNDED_SPEED_MPS
    parsed = parse_feed(body, now)
    ship = parsed["vehicles"][0]
    assert ship["alt_m"] < GROUNDED_ALT_M
    assert ship["speed_mps"] <= GROUNDED_SPEED_MPS
    assert _is_grounded(ship) is True
    # The underground prediction is not a display path.
    assert ship["predicted_sane"] is False
    assert ship["predicted"] == []


def test_landed_ship_is_marked_done_after_it_stays_on_the_pad():
    now = datetime(2026, 11, 14, 21, 20, tzinfo=timezone.utc)
    net = datetime(2026, 11, 13, 22, 47, 36, tzinfo=timezone.utc)
    launches = _launch(net)
    state = TrackerState(persist=False)
    # Confirm the flight with two coast samples, then sit on the pad.
    coast = accept_hobby_document(_coast_document(now - timedelta(hours=1), met=80_000.0), now - timedelta(hours=1))
    state.ingest(coast, now - timedelta(hours=1))
    later = now - timedelta(hours=1) + timedelta(seconds=60)
    coast2 = accept_hobby_document(_coast_document(later, age_s=20.0, met=80_060.0, lon=-93.0), later)
    state.ingest(coast2, later)
    state.update_mission(launches, later)
    assert state.liftoff_utc is not None
    assert state._flight_confirmed() is True
    landed = accept_hobby_document(_landed_hobby_document(now), now)
    state.ingest(landed, now)
    state.update_mission(launches, now)
    ship = state.vehicles["ship42"]
    assert _is_grounded(ship["current"]) is True
    assert ship.get("done") is not True
    settled = now + timedelta(seconds=GROUNDED_STABLE_S)
    # Re-ingest so last_seen stays fresh while the ship stays put.
    landed2 = accept_hobby_document(_landed_hobby_document(settled), settled)
    state.ingest(landed2, settled)
    state.update_mission(launches, settled)
    ship = state.vehicles["ship42"]
    assert ship.get("done") is True
    assert ship.get("done_reason") == "landed"
    assert state.hold_started is True
    assert ship["current"]["speed_mps"] <= GROUNDED_SPEED_MPS


def _flight15_launches(*, upcoming=False, status="Launch Successful"):
    row = {
        "id": "0f867f0c-5422-4d44-93b8-d7faaae7019b",
        "mission": "Starship | Starlink Group 31-2 (Starship Flight 15)",
        "name": "Starship | Starlink Group 31-2 (Starship Flight 15)",
        "net": "2026-11-13T22:47:36Z",
        "status": status,
        "rocket": "Starship",
        "pad": "Orbital Launch Pad 2",
    }
    if upcoming:
        return {"upcoming": [row], "previous": []}
    return {"upcoming": [], "previous": [row]}


def _post_catch_state(now, *, stable_since, hold_started=False, hold_until=None, done=False):
    """Persisted Flight 15 state after the catch: ship on the pad, trail still airborne."""
    net = datetime(2026, 11, 13, 22, 47, 36, tzinfo=timezone.utc)
    state = TrackerState(persist=False)
    pad = {
        "lat": 25.996826,
        "lon": -97.158036,
        "alt_m": 94.898987,
        "speed_mps": 0.07928783652200268,
        "mission_time_s": 82455.630823,
        "fresh": True,
        "utc": stable_since.isoformat(),
    }
    airborne = [
        {
            "lat": 26.1, "lon": -96.0, "alt_m": 272_596.0, "speed_mps": 7600.0,
            "mission_time_s": 17_548.0, "fresh": True,
            "utc": (net + timedelta(hours=5)).isoformat(),
        },
        {
            "lat": -29.5, "lon": 61.4, "alt_m": 251_203.0, "speed_mps": 7758.0,
            "mission_time_s": 77_582.0, "fresh": True,
            "utc": (net + timedelta(hours=21, minutes=33)).isoformat(),
        },
        pad,
    ]
    state.liftoff_utc = net
    state.liftoff_from_motion = True
    state.hold_started = hold_started
    state.hold_until = hold_until
    state.vehicles["ship42"] = {
        "kind": "ship",
        "name": "Ship 42",
        "in_flight": True,
        "done": done,
        "done_reason": "landed" if done else None,
        "missing": False,
        "missing_since": None,
        "moving": False,
        "stable_since": stable_since.isoformat(),
        "current": pad,
        "samples": airborne,
        "predicted": [],
    }
    return state


def test_replay_of_flight_15_the_morning_after_leaves_starship_mode():
    """The live 2026-11-15 morning state: S42 parked at Starbase, T+40h.

    Official SpaceX tracker is empty. Beyond Stage Zero / StarDash still
    publish a fresh pad fix (speed 0, alt ~96 m) with a predicted path
    into the Earth, and the launch list has moved Flight 15 to previous
    with status 'Launch Successful'. The on-disk log still has the flown
    trail and a grounded latch from the previous afternoon, but no hold
    had been saved. Starship mode must not stay on for another hour.
    """
    now = datetime(2026, 11, 15, 15, 5, tzinfo=timezone.utc)
    stable = datetime(2026, 11, 14, 21, 41, 54, tzinfo=timezone.utc)
    state = _post_catch_state(now, stable_since=stable)
    assert state._flight_confirmed() is True
    landed = accept_hobby_document(_landed_hobby_document(now), now)
    assert landed is not None
    state.ingest(landed, now)
    state.update_mission(_flight15_launches(), now)
    payload = state.globe_payload(now, in_window=state.wants_fast_poll(now))
    ship = state.vehicles["ship42"]
    assert _is_grounded(ship["current"]) is True
    assert ship.get("done") is True
    assert ship.get("done_reason") == "landed"
    assert state.hold_started is True
    assert state.hold_until is not None
    assert now >= state.hold_until
    assert state.live_layer_active(now) is False
    assert payload["launch_active"] is False
    assert payload["telemetry"]["show"] is False
    assert state.wants_fast_poll(now) is False


def test_restart_after_hours_on_the_pad_does_not_start_a_fresh_hold():
    """hold_until is anchored to landing time, not the process start."""
    now = datetime(2026, 11, 15, 15, 5, tzinfo=timezone.utc)
    stable = datetime(2026, 11, 14, 21, 41, 54, tzinfo=timezone.utc)
    state = _post_catch_state(now, stable_since=stable, done=False, hold_started=False)
    state.update_mission(_flight15_launches(), now)
    assert state.vehicles["ship42"]["done"] is True
    assert state.hold_started is True
    assert now >= state.hold_until
    assert state.live_layer_active(now) is False
    assert state.globe_payload(now)["launch_active"] is False


def test_launch_successful_does_not_end_an_in_orbit_ship():
    """'Launch Successful' often means the ascent, not that the ship is down."""
    now = datetime(2026, 11, 14, 16, 0, tzinfo=timezone.utc)
    net = datetime(2026, 11, 13, 22, 47, 36, tzinfo=timezone.utc)
    state = TrackerState(persist=False)
    state.liftoff_utc = net
    state.liftoff_from_motion = True
    state.vehicles["ship42"] = _flying(251_000, 7750)
    state.update_mission(_flight15_launches(status="Launch Successful"), now)
    assert state.vehicles["ship42"].get("done") is not True
    assert state.hold_started is False
    assert state.live_layer_active(now) is True
    assert state.globe_payload(now, in_window=True)["launch_active"] is True


def test_landed_latch_survives_a_pad_speed_glitch():
    """A few m/s of leftover predicted speed must not reopen the flight."""
    now = datetime(2026, 11, 14, 22, 0, tzinfo=timezone.utc)
    stable = now - timedelta(minutes=20)
    state = _post_catch_state(now, stable_since=stable, done=True, hold_started=True,
                              hold_until=now + timedelta(minutes=30))
    state.vehicles["ship42"]["current"] = {
        "lat": 25.9968, "lon": -97.1580, "alt_m": 96.0, "speed_mps": 5.0,
        "mission_time_s": 83_000, "fresh": True, "utc": now.isoformat(),
    }
    state.update_mission(_flight15_launches(), now)
    ship = state.vehicles["ship42"]
    assert ship["done"] is True
    assert ship["done_reason"] == "landed"
    assert state.hold_started is True
    assert state.live_layer_active(now) is True
    # After the original hold, mode still ends.
    later = now + timedelta(minutes=31)
    state.update_mission(_flight15_launches(), later)
    assert state.live_layer_active(later) is False


def test_hold_is_not_cancelled_unless_the_ship_is_clearly_airborne():
    now = datetime(2026, 11, 14, 22, 0, tzinfo=timezone.utc)
    stable = now - timedelta(minutes=20)
    state = _post_catch_state(now, stable_since=stable, done=True, hold_started=True,
                              hold_until=now + timedelta(minutes=30))
    # Un-grounded but still on the pad (speed above the 2 m/s latch, alt low).
    state.vehicles["ship42"]["done"] = False
    state.vehicles["ship42"]["done_reason"] = None
    state.vehicles["ship42"]["current"]["speed_mps"] = 5.0
    state.update_mission(_flight15_launches(), now)
    assert state.hold_started is True
    assert state.live_layer_active(now) is True
    # A real new flight does cancel it.
    state.vehicles["ship42"]["current"] = {
        "lat": 26.2, "lon": -96.5, "alt_m": 40_000, "speed_mps": 1200,
        "mission_time_s": 120, "fresh": True, "utc": now.isoformat(),
    }
    state.update_mission(_flight15_launches(), now)
    assert state.hold_started is False
    assert state.live_layer_active(now) is True


def test_stardash_feed_maps_onto_the_same_parser():
    now = datetime(2026, 11, 14, 13, 0, tzinfo=timezone.utc)
    body = accept_hobby_document(stardash_feed_to_tracker(_stardash_feed(now), now), now)
    assert body is not None
    assert body["metadata"]["source"] == "stardash"
    ship = parse_feed(body, now)["vehicles"][0]
    assert ship["key"] == "ship42"
    assert ship["alt_m"] == 254_000.0
    assert ship["speed_mps"] > ORBIT_COAST_SPEED_MPS
    assert ship["predicted_sane"] is True


def test_hobby_document_rejects_a_stale_or_empty_feed():
    now = datetime(2026, 11, 14, 13, 0, tzinfo=timezone.utc)
    assert accept_hobby_document(_coast_document(now, age_s=HOBBY_STALE_S + 30), now) is None
    assert stardash_feed_to_tracker({"samples": []}, now) is None
    assert accept_hobby_document({"metadata": {"generation_time": utc_to_gps(now)}}, now) is None


def _history_response(payload, status=200, headers=None):
    response = _response(status, payload, headers)
    response.json.return_value = payload
    return response


def test_empty_spacex_uses_beyond_stage_zero_before_stardash_and_switches_back():
    now = datetime(2026, 11, 14, 13, 0, tzinfo=timezone.utc)
    state = TrackerHostState()
    calls = []
    bsz = _coast_document(now)

    def get(url, headers=None, timeout=None):
        calls.append(url)
        assert headers["User-Agent"].startswith("spacex-dashboard/")
        if url in (BEYOND_STAGE_ZERO_URL, STARDASH_FEED_URL):
            assert timeout == 8
        if url == BEYOND_STAGE_ZERO_URL:
            return _history_response(bsz)
        if url == STARDASH_FEED_URL:
            raise AssertionError("StarDash was called while Beyond Stage Zero had a fix")
        if url == PRIMARY_TRACKER_URL:
            return _response(200, {})
        raise RuntimeError("HTTP 404")

    result = fetch_with_failover(state, now=now, get=get, allow_alternate=True)
    assert result["url"] == BEYOND_STAGE_ZERO_URL
    ship = parse_feed(result["body"], now)["vehicles"][0]
    assert ship["key"] == "ship42"
    assert ship["speed_mps"] > ORBIT_COAST_SPEED_MPS
    assert calls == [PRIMARY_TRACKER_URL, FALLBACK_TRACKER_URL, BEYOND_STAGE_ZERO_URL]
    # SpaceX stays preferred so the next poll can switch back.
    assert state.preferred == PRIMARY_TRACKER_URL

    calls.clear()
    held = fetch_with_failover(state, now=now + timedelta(seconds=10), get=get, allow_alternate=True)
    assert held["not_modified"] is True
    assert calls == [PRIMARY_TRACKER_URL, FALLBACK_TRACKER_URL]
    assert HOBBY_POLL_S == 30

    def recovered(url, headers=None, timeout=None):
        calls.append(url)
        if url == PRIMARY_TRACKER_URL:
            return _response(200, _host_fix("back"), {"ETag": "primary"})
        raise AssertionError(url)

    calls.clear()
    back = fetch_with_failover(
        state, now=now + timedelta(seconds=40), get=recovered, allow_alternate=True
    )
    assert back["url"] == PRIMARY_TRACKER_URL
    assert state.preferred == PRIMARY_TRACKER_URL
    assert calls == [PRIMARY_TRACKER_URL]


def test_stardash_feed_is_fourth_after_beyond_stage_zero_fails():
    now = datetime(2026, 11, 14, 13, 0, tzinfo=timezone.utc)
    state = TrackerHostState()
    calls = []

    def get(url, headers=None, timeout=None):
        calls.append(url)
        if url == BEYOND_STAGE_ZERO_URL:
            return _response(503, {}, {"Retry-After": "10"})
        if url == STARDASH_FEED_URL:
            return _history_response(_stardash_feed(now))
        if url == PRIMARY_TRACKER_URL:
            return _response(200, {})
        raise RuntimeError("HTTP 404")

    result = fetch_with_failover(state, now=now, get=get, allow_alternate=True)
    assert result["url"] == STARDASH_FEED_URL
    assert parse_feed(result["body"], now)["vehicles"][0]["speed_mps"] > ORBIT_COAST_SPEED_MPS
    assert calls == [
        PRIMARY_TRACKER_URL,
        FALLBACK_TRACKER_URL,
        BEYOND_STAGE_ZERO_URL,
        STARDASH_FEED_URL,
    ]


def test_hobby_mirrors_stay_off_outside_the_mission_window_and_back_off():
    now = datetime(2026, 11, 14, 13, 0, tzinfo=timezone.utc)
    state = TrackerHostState()
    calls = []

    def get(url, headers=None, timeout=None):
        calls.append(url)
        if url in (BEYOND_STAGE_ZERO_URL, STARDASH_FEED_URL):
            raise TimeoutError("hobby down")
        return _response(200, {})

    blocked = fetch_with_failover(state, now=now, get=get, allow_alternate=False)
    assert blocked.get("not_modified") is True
    assert BEYOND_STAGE_ZERO_URL not in calls
    assert STARDASH_FEED_URL not in calls

    calls.clear()
    failed = fetch_with_failover(state, now=now, get=get, allow_alternate=True)
    assert failed["error"]
    assert BEYOND_STAGE_ZERO_URL in calls and STARDASH_FEED_URL in calls
    calls.clear()
    waiting = fetch_with_failover(
        state, now=now + timedelta(seconds=5), get=get, allow_alternate=True
    )
    assert waiting.get("not_modified") is True or waiting.get("error")
    assert BEYOND_STAGE_ZERO_URL not in calls
    assert STARDASH_FEED_URL not in calls


def test_hobby_fix_drives_the_strip_inside_the_window(tmp_path):
    now = datetime(2026, 11, 14, 13, 6, tzinfo=timezone.utc)
    state = TrackerState(path=str(tmp_path / "tracker.json"), persist=False)
    launches = _launch(datetime(2026, 11, 13, 22, 47, tzinfo=timezone.utc))
    for step in (0, 1):
        moment = now + timedelta(seconds=30 * step)
        body = accept_hobby_document(
            _coast_document(moment, age_s=15, met=52_000.0 + 30 * step, lon=-97.0 + 5.0 * step),
            moment,
        )
        state.ingest(body, moment)
    state.update_mission(launches, now + timedelta(seconds=30))
    payload = state.globe_payload(now + timedelta(seconds=30), in_window=True)
    assert payload["show"] is True
    assert payload["launch_active"] is True
    ship = payload["vehicles"][0]
    assert ship["name"] == "Ship 42"
    assert ship["alt_m"] == 254_000.0
    assert ship["speed_mps"] > ORBIT_COAST_SPEED_MPS
    assert len(ship["trail"]) >= 2
    telemetry = payload["telemetry"]
    assert telemetry["show"] is True
    assert telemetry["ship"]["present"] is True
    assert telemetry["ship"]["alt_km"] != "—"
    assert telemetry["in_orbit"] is True
    # Two local fixes, hours after NET. Longitude would say ORBIT 1.
    assert telemetry["orbit_number"] > 1
    assert telemetry["orbit_number"] == estimated_orbit_number(
        ship,
        datetime(2026, 11, 13, 22, 47, tzinfo=timezone.utc),
        now + timedelta(seconds=30),
    )



def _history_row(gps, lat, lon, alt, ex, ey, ez, met):
    return {
        "gps": gps, "veh": "ship42", "lat": lat, "lon": lon, "alt": alt,
        "ex": ex, "ey": ey, "ez": ez, "met": met, "spd": 0,
    }


def test_history_rows_keep_the_pad_start_and_drop_underground_jumps():
    radius = 6_625_000.0
    theta = 60.0 * 7750.0 / radius
    rows = [
        _history_row(1_000_000.0, 0.0, 0.0, 94.0, radius, 0.0, 0.0, 0.0),
        _history_row(1_000_010.0, 0.0, 0.0, 94.2, radius, 0.0, 0.0, 10.0),
        _history_row(1_000_020.0, 0.0, 0.0, -5_000.0, radius, 0.0, 0.0, 20.0),
        _history_row(
            1_000_060.0, 0.0, 4.0, 250_000.0,
            radius * math.cos(theta), radius * math.sin(theta), 0.0, 60.0,
        ),
    ]
    trail = rows_to_trail(rows)
    assert trail[0]["alt_m"] == 94.0
    assert all(sample["alt_m"] > -1000 for sample in trail)
    assert any(sample["alt_m"] == 250_000.0 and sample["speed_mps"] > ORBIT_COAST_SPEED_MPS for sample in trail)
    # The second pad sample did not move, so it is not a second point.
    assert sum(1 for sample in trail if sample["alt_m"] < 1000) == 1


def test_backfill_merges_partial_local_trail_and_counts_laps(tmp_path):
    liftoff = datetime(2026, 11, 13, 22, 47, 36, tzinfo=timezone.utc)
    now = liftoff + timedelta(hours=6)
    state = TrackerState(path=str(tmp_path / "tracker.json"))
    # A reboot an hour ago only stored the latest coast fix.
    state.vehicles["ship42"] = {
        "kind": "ship",
        "name": "Ship 42",
        "samples": [{
            "lat": 0.0, "lon": 0.0, "alt_m": 250_000.0, "speed_mps": 7700.0,
            "mission_time_s": 20_000.0, "utc": (now - timedelta(minutes=5)).isoformat(),
            "fresh": True,
        }],
    }
    assert ship_trail_is_partial(state, liftoff, now) is True
    # Three ground-track laps, 10° steps, starting at liftoff.
    start_gps = utc_to_gps(liftoff)
    incoming = []
    for index, step in enumerate(range(0, 360 * 3 + 1, 10)):
        lon = ((step + 180) % 360) - 180
        gps = start_gps + index * 30
        incoming.append({
            "lat": 0.0,
            "lon": float(lon),
            "alt_m": 250_000.0,
            "speed_mps": 7700.0,
            "mission_time_s": 600.0 + step * 15,
            "utc": gps_to_utc(gps).isoformat(),
            "gps": gps,
            "fresh": True,
        })
    assert state.apply_backfill({"ship42": incoming}, now) is True
    assert ship_trail_is_partial(state, liftoff, now) is False
    assert ship_completed_laps(state.vehicles["ship42"]["samples"]) == 3
    state.liftoff_utc = liftoff
    # Three fast longitude laps, then the local fix five hours later.
    # Ground track still says 3 laps. The inertial counter fills that hole
    # with the Kepler period, so the ship is on orbit 7.
    covered = ship_orbit_status(
        state.vehicles["ship42"],
        liftoff_utc=liftoff,
        now_utc=gps_to_utc(incoming[-1]["gps"]),
    )
    assert ship_completed_laps(state.vehicles["ship42"]["samples"]) == 3
    assert covered["orbit_label"] == "ORBIT 7"
    # Saved, so the next process can skip the network.
    reloaded = TrackerState.load(str(tmp_path / "tracker.json"))
    assert ship_completed_laps(reloaded.vehicles["ship42"]["samples"]) == 3
    # A second archive that is not earlier does not grow the file again.
    assert state.apply_backfill({"ship42": incoming[-1:]}, now) is False


def test_strip_clock_and_readouts_advance_between_samples():
    """T+ is elapsed since liftoff. Alt and speed have a step to lerp across."""
    liftoff = datetime(2026, 11, 13, 22, 47, 36, tzinfo=timezone.utc)
    now = liftoff + timedelta(hours=6, seconds=12)
    state = TrackerState(persist=False)
    earlier = now - timedelta(seconds=30)
    state.ingest(
        _tracker_body(utc_to_gps(earlier), 0.0, 8.0, 249_000, 7600, mission_time=50),
        earlier,
    )
    gps = utc_to_gps(now)
    state.ingest(
        _tracker_body(
            gps, 0.0, 10.0, 250_000, 7700,
            predicted=_sane_prediction(gps, 0.0, 10.0, 250_000),
            mission_time=80,
        ),
        now,
    )
    state.update_mission(_launch(liftoff), now)
    payload = state.globe_payload(now, in_window=True)
    tel = payload["telemetry"]
    assert abs(tel["clock_base_s"] - (now - liftoff).total_seconds()) < 1.0
    assert tel["clock"] == "T+00:01:20"
    ship = tel["ship"]
    assert ship["motion_span_s"] == 60.0
    assert ship["speed_mps"] == 7700
    assert ship["alt_m_next"] == 250_200
    assert payload["vehicles"][0]["gps_s"] == gps
    assert payload["vehicles"][0]["predicted"][0]["t"] == gps + 60
    assert payload["telemetry"]["flight_label"] == ""


def test_flight_label_uses_the_mission_name_and_the_fix_carries_its_age():
    liftoff = datetime(2026, 11, 13, 22, 47, 36, tzinfo=timezone.utc)
    now = liftoff + timedelta(minutes=40)
    state = TrackerState(persist=False)
    gps = utc_to_gps(now - timedelta(seconds=16))
    state.ingest(_tracker_body(gps - 30, 0.0, 8.0, 249_000, 7600, mission_time=50), now - timedelta(seconds=30))
    state.ingest(
        _tracker_body(gps, 0.0, 10.0, 250_000, 7700, predicted=_sane_prediction(gps, 0.0, 10.0, 250_000), mission_time=80),
        now,
    )
    launches = {
        "upcoming": [{
            "net": liftoff.isoformat(),
            "rocket": "Starship",
            "mission": "Starlink Group 31-2 (Starship Flight 15)",
        }],
        "previous": [],
    }
    state.update_mission(launches, now)
    payload = state.globe_payload(now, in_window=True)
    assert payload["telemetry"]["flight_label"] == "Flight 15"
    assert abs(payload["vehicles"][0]["sample_age_s"] - 16.0) < 0.05


def test_osculating_apsides_are_altitudes_above_the_mean_sphere():
    """A circular 265 km orbit reads 265 / 265. Suborbital keeps apogee only."""
    radius = 6_371_000.0 + 265_000.0
    speed = math.sqrt(3.986004418e14 / radius)
    apo, peri = osculating_apsides_m((radius, 0.0, 0.0), (0.0, speed, 0.0))
    assert abs(apo - 265_000.0) < 2_000.0
    assert abs(peri - 265_000.0) < 2_000.0
    assert format_apsis_label(apo, peri) == "265 km / 265 km"
    assert format_apsis_label(180_000.0, -20_000.0) == "180 km / —"
    assert format_apsis_label(40_000.0, 10_000.0) == ""
    assert format_apsis_label(None, 200_000.0) == ""
    # Same circle through the slot API. Positions are ECEF on the 6371 km
    # sphere; a geodetic 265 km altitude sits on the WGS84 equator instead.
    motion = speed / radius
    start = datetime(2026, 11, 13, 23, 0, tzinfo=timezone.utc)

    def at(second):
        theta = motion * second
        earth = EARTH_ROTATION_RAD_S * second
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        c, s = math.cos(earth), math.sin(earth)
        return {
            "lat": 0.0,
            "lon": 0.0,
            "alt_m": 265_000.0,
            "speed_mps": speed,
            "ecef": (c * x + s * y, -s * x + c * y, 0.0),
            "gps_s": utc_to_gps(start) + second,
        }

    samples = [at(second) for second in range(0, 301, 60)]
    current = samples[-1]
    # A straight predicted chord used to lean inward and invent eccentricity.
    # The fit uses the flown samples only.
    slot = {
        "current": current,
        "samples": samples,
        "predicted": [{
            "lat": 0.0,
            "lon": 0.0,
            "alt_m": 200_000.0,
            "time": current["gps_s"] + 60,
            "ecef": (radius, 0.0, 0.0),
        }],
    }
    label = ship_apsis_label(slot)
    apo_km, peri_km = (int(part) for part in label.replace(" km", "").split(" / "))
    assert abs(apo_km - 265) <= 3
    assert abs(peri_km - 265) <= 3


def test_apsis_label_does_not_jump_on_a_bad_sample():
    """One wild fix may move the label by at most 8 km."""
    radius = 6_371_000.0 + 265_000.0
    speed = math.sqrt(3.986004418e14 / radius)
    motion = speed / radius
    start = datetime(2026, 11, 13, 23, 0, tzinfo=timezone.utc)

    def at(second, radial=0.0):
        theta = motion * second
        earth = EARTH_ROTATION_RAD_S * second
        r = radius + radial
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        c, s = math.cos(earth), math.sin(earth)
        return {
            "lat": 0.0, "lon": 0.0, "alt_m": 265_000.0 + radial,
            "speed_mps": speed,
            "ecef": (c * x + s * y, -s * x + c * y, 0.0),
            "gps_s": utc_to_gps(start) + second,
        }

    samples = [at(second) for second in range(0, 301, 60)]
    slot = {"current": samples[-1], "samples": samples}
    first = ship_apsis_label(slot)
    assert first == "265 km / 265 km"
    # Same call, same fix: the cache must not step the label twice.
    assert ship_apsis_label(slot) == first
    wild = [at(second, radial=40_000.0 if second >= 240 else 0.0) for second in range(0, 301, 60)]
    slot["current"] = wild[-1]
    slot["samples"] = wild
    jumped = ship_apsis_label(slot)
    apo_km = int(jumped.split(" km")[0])
    assert abs(apo_km - 265) <= 8


def test_restart_mid_flight_without_a_mission_clock_is_airborne():
    """Hobby fixes omit mission_time. Orbital altitude still confirms the flight.

    A kiosk that woke at T+17h45m used to log that it was waiting for the
    ship to leave the pad, then that the vehicles were down, and the
    partial trail read ORBIT 1.
    """
    liftoff = datetime(2026, 11, 13, 22, 47, 36, tzinfo=timezone.utc)
    now = liftoff + timedelta(hours=17, minutes=45)
    state = TrackerState(persist=False)
    samples = []
    for index, age in enumerate((120, 60, 0)):
        moment = now - timedelta(seconds=age)
        samples.append({
            "lat": 0.0,
            "lon": float(index),
            "alt_m": 265_000.0,
            "speed_mps": 7700.0,
            "mission_time_s": None,
            "fresh": True,
            "utc": moment.isoformat(),
            "gps": utc_to_gps(moment),
        })
    state.vehicles["ship42"] = {
        "kind": "ship",
        "name": "Ship 42",
        "in_flight": False,
        "done": False,
        "missing": False,
        "samples": samples,
        "current": samples[-1],
    }
    # A booster saved as already down must not start the post-catch hold
    # while the ship is still in orbit and the flight was never confirmed.
    state.vehicles["booster22"] = {
        "kind": "booster",
        "in_flight": True,
        "done": True,
        "done_reason": "landed",
        "missing": True,
        "missing_since": (now - timedelta(hours=2)).isoformat(),
        "samples": [],
        "current": {"alt_m": 0, "speed_mps": 0, "mission_time_s": 400},
    }
    state.update_mission(_launch(liftoff), now)
    assert state.liftoff_from_motion is True
    assert state.hold_started is False
    assert state.live_layer_active(now) is True
    status = ship_orbit_status(
        state.vehicles["ship42"], liftoff_utc=state.liftoff_utc, now_utc=now
    )
    assert status["orbit_number"] >= 11


def test_inertial_rev_beats_one_ground_track_lap():
    """One inertial period at the equator is not yet 360° of longitude."""
    period = 89.6 * 60.0
    radius = 6_378_137.0 + 265_000.0
    motion = 2.0 * math.pi / period
    start = datetime(2026, 11, 13, 23, 0, tzinfo=timezone.utc)
    samples = []
    for second in range(0, int(period) + 1, 60):
        theta = motion * second
        earth = EARTH_ROTATION_RAD_S * second
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        c, s = math.cos(earth), math.sin(earth)
        xe, ye = c * x + s * y, -s * x + c * y
        lon = math.degrees(math.atan2(ye, xe))
        samples.append({
            "lat": 0.0,
            "lon": lon,
            "alt_m": 265_000.0,
            "speed_mps": 7600.0,
            "utc": (start + timedelta(seconds=second)).isoformat(),
        })
    revs = inertial_revolutions(samples)
    assert abs(revs - 1.0) < 0.02
    assert ship_completed_laps(samples) == 0
    done = ship_orbit_status(
        {"missing": False, "current": samples[-1], "samples": samples},
        now_utc=start + timedelta(seconds=period),
    )
    assert done["orbit_number"] == 2
    assert done["orbit_label"] == "ORBIT 2"
    inserted = ship_orbit_status(
        {"missing": False, "current": samples[0], "samples": samples[:1]},
        now_utc=start,
    )
    assert inserted["orbit_label"] == "ORBIT 1"


def test_partial_trail_estimates_the_orbit_instead_of_reporting_one():
    """A mid-flight join with no history must not sit on ORBIT 1."""
    liftoff = datetime(2026, 11, 13, 22, 47, 36, tzinfo=timezone.utc)
    now = liftoff + timedelta(hours=6)
    radius = math.sqrt(sum(c * c for c in geodetic_to_ecef(0.0, 0.0, 250_000.0)))
    mu = 3.986004418e14
    speed = 7700.0
    sma = 1.0 / (2.0 / radius - speed * speed / mu)
    period = 2.0 * math.pi * math.sqrt(sma ** 3 / mu)
    current = {
        "lat": 0.0, "lon": 20.0, "alt_m": 250_000.0, "speed_mps": speed,
        "utc": now.isoformat(), "mission_time_s": 21_000.0,
    }
    assert abs(orbital_period_s(current) - period) < 1e-6
    samples = [
        {
            "lat": 0.0, "lon": 10.0, "alt_m": 250_000.0, "speed_mps": speed,
            "utc": (now - timedelta(minutes=2)).isoformat(), "mission_time_s": 20_880.0,
        },
        current,
    ]
    assert ship_completed_laps(samples) == 0
    status = ship_orbit_status(
        {"missing": False, "current": current, "samples": samples},
        liftoff_utc=liftoff,
        now_utc=now,
    )
    expect = int(math.floor((now - liftoff).total_seconds() / period)) + 1
    assert expect > 1
    assert status["orbit_number"] == expect
    assert status["orbit_label"] == f"ORBIT {expect}"
    # Same state vector via ECEF, and a trail that reaches liftoff stays on longitude.
    x, y, z = geodetic_to_ecef(0.0, 0.0, 250_000.0)
    assert abs(orbital_period_s({**current, "ecef": [x, y, z]}) - period) < 1e-6
    inserted = {
        "missing": False,
        "current": {
            "lat": 0.0, "lon": 1.0, "alt_m": 200_000.0, "speed_mps": 7600.0,
            "utc": (liftoff + timedelta(minutes=15)).isoformat(),
        },
        "samples": [{
            "lat": 0.0, "lon": 0.0, "alt_m": 200_000.0, "speed_mps": 7600.0,
            "utc": (liftoff + timedelta(minutes=12)).isoformat(),
        }],
    }
    assert ship_orbit_status(
        inserted, liftoff_utc=liftoff, now_utc=liftoff + timedelta(minutes=15)
    )["orbit_label"] == "ORBIT 1"


def test_pad_sample_at_liftoff_does_not_count_a_short_coast_trail():
    """ORBIT 2 on a device that restarted mid-flight.

    The saved log still has the pad fix from T+0, so the trail "reaches
    liftoff", but the only coast fixes are the last hour. The count has to
    use the liftoff time and the current period, not that local arc.
    """
    liftoff = datetime(2026, 11, 13, 22, 47, 36, tzinfo=timezone.utc)
    now = liftoff + timedelta(hours=20, minutes=30)
    pad = {
        "lat": 25.99, "lon": -97.15, "alt_m": 120.0, "speed_mps": 0.0,
        "utc": liftoff.isoformat(),
    }
    samples = [pad]
    start = now - timedelta(minutes=80)
    for minute in range(0, 81, 5):
        moment = start + timedelta(minutes=minute)
        samples.append({
            "lat": 0.0,
            "lon": ((minute * 4.5 + 180) % 360) - 180,
            "alt_m": 250_000.0,
            "speed_mps": 7600.0,
            "utc": moment.isoformat(),
        })
    current = samples[-1]
    local = ship_orbit_status(
        {"missing": False, "current": current, "samples": samples[1:]},
        now_utc=now,
    )
    assert local["orbit_number"] < 4
    status = ship_orbit_status(
        {"missing": False, "current": current, "samples": samples},
        liftoff_utc=liftoff,
        now_utc=now,
    )
    expect = estimated_orbit_number(current, liftoff, now)
    assert expect >= 13
    assert status["orbit_number"] == expect
    assert status["orbit_label"] == f"ORBIT {expect}"


def test_midflight_join_uses_the_passed_net_as_liftoff():
    liftoff = datetime(2026, 11, 13, 22, 47, 36, tzinfo=timezone.utc)
    now = liftoff + timedelta(hours=6)
    state = TrackerState(persist=False)
    gps = utc_to_gps(now)
    state.ingest(_tracker_body(gps, 0.0, 10.0, 250_000, 7700, mission_time=20_000), now)
    later = now + timedelta(seconds=30)
    state.ingest(
        _tracker_body(gps + 30, 0.0, 12.0, 250_000, 7700, mission_time=20_030),
        later,
    )
    state.update_mission(_launch(liftoff), later)
    assert state.liftoff_utc == liftoff
    payload = state.globe_payload(later, in_window=True)
    assert payload["telemetry"]["orbit_number"] > 1
    assert payload["telemetry"]["orbit_number"] == estimated_orbit_number(
        payload["vehicles"][0], liftoff, later
    )


def test_flight_history_prefers_a_real_ship_trail_then_stardash():
    now = datetime(2026, 11, 14, 13, 0, tzinfo=timezone.utc)
    calls = []
    radius = 6_625_000.0
    theta = 60.0 * 7750.0 / radius
    bsz_points = [{
        "veh": "ship42", "gps": utc_to_gps(now) - 120, "lat": 0.0, "lon": 0.0,
        "alt": 250_000.0, "ex": radius, "ey": 0.0, "ez": 0.0, "met": 1000,
    }, {
        "veh": "ship42", "gps": utc_to_gps(now) - 60, "lat": 0.0, "lon": 4.0,
        "alt": 250_000.0,
        "ex": radius * math.cos(theta), "ey": radius * math.sin(theta), "ez": 0.0,
        "met": 1060,
    }]

    def get(url, headers=None, timeout=None):
        calls.append(url)
        assert headers["User-Agent"].startswith("spacex-dashboard/")
        assert timeout == 15
        if url == BSZ_SHIP_TRAIL_URL:
            return _history_response({"points": bsz_points})
        raise AssertionError(url)

    grouped = fetch_flight_history(now, get=get)
    assert calls == [BSZ_SHIP_TRAIL_URL]
    assert "ship42" in grouped
    assert len(grouped["ship42"]) >= 2

    def fallback(url, headers=None, timeout=None):
        calls.append(url)
        if url == BSZ_SHIP_TRAIL_URL:
            return _response(404, {"error": "Ship trail returned 404"})
        assert url == stardash_history_url(now)
        return _history_response({"history_ok": True, "samples": bsz_points})

    calls.clear()
    grouped = fetch_flight_history(now, get=fallback)
    assert calls[0] == BSZ_SHIP_TRAIL_URL
    assert "stardash.live/api/history?w=" in calls[1]
    assert parse_stardash_history({"history_ok": False, "error": "bad chunk request"}) is None
    assert grouped["ship42"][0]["alt_m"] == 250_000.0


def test_backfill_runs_once_off_the_tracker_poll_thread():
    poller = (ROOT / "starship_http.py").read_text(encoding="utf-8")
    assert 'name="starship-backfill"' in poller
    assert "daemon=True" in poller
    assert "fetch_flight_history" in poller
    assert "_tracker_busy" not in poller
    assert "_maybe_backfill" in poller
    assert "allow_alternate" in poller
    assert "net_passed" in poller
    # The poller must not invent a vehicle from the scripted test flight.
    assert "synthetic_inflight_feeds(" not in poller
