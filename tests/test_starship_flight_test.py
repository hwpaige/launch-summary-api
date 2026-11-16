"""Flight test mode: simulated Starship flight through the real tracker."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import starship_flight_test as sim
import starship_tracker
from starship_tracker import haversine_km, parse_feed

START = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parent.parent


def _run_all():
    run = sim.FlightTestRun(START)
    frames = []
    while not run.finished:
        mission = run.mission_t
        frames.append((mission, run.step()))
    return run, frames


def test_samples_are_plausible_for_booster_and_ship():
    booster = [sim.booster_sample(t) for t in range(0, 421, 5)]
    assert max(alt for _, _, alt, _ in booster) <= 75_000
    assert max(alt for _, _, alt, _ in booster) >= 65_000
    top_kmh = max(speed for *_, speed in booster) * 3.6
    assert 4_800 <= top_kmh <= 5_600
    lat, lon, alt, speed = sim.booster_sample(420)
    assert haversine_km(lat, lon, sim.STARBASE_LAT, sim.STARBASE_LON) < 1
    assert alt < 2000 and speed == 0

    ship = [sim.ship_sample(t) for t in range(0, int(sim.SHIP_CATCH_S) + 1, 20)]
    peak_alt = max(alt for _, _, alt, _ in ship)
    assert 150_000 <= peak_alt <= 200_000
    peak_kmh = max(speed for *_, speed in ship) * 3.6
    assert 25_000 <= peak_kmh <= 27_500
    lat, lon, _, _ = sim.ship_sample(400)
    assert -95 < lon < -80 and 20 < lat < 26.5
    lat, lon, alt, speed = sim.ship_sample(sim.SHIP_CATCH_S)
    assert haversine_km(lat, lon, sim.STARBASE_LAT, sim.STARBASE_LON) < 1
    assert speed == 0


def test_simulated_feed_matches_public_tracker_shape():
    doc = sim.simulated_feed(300, START)
    assert set(doc) == {sim.BOOSTER_KEY, sim.SHIP_KEY, "metadata"}
    current = doc[sim.SHIP_KEY]["current"]
    for key in ("gps_time", "mission_time", "altitude", "speed", "latitude", "longitude", "r_ecef"):
        assert key in current
    liftoff = sim.FlightTestRun(START).liftoff_utc
    doc = sim.simulated_feed(300, liftoff)
    parsed = parse_feed(doc, liftoff + timedelta(seconds=300))
    assert parsed["stale"] is False
    kinds = sorted(v["kind"] for v in parsed["vehicles"])
    assert kinds == ["booster", "ship"]


def test_full_run_drives_the_live_tracker_pipeline_and_ends():
    run, frames = _run_all()
    wall = run.ticks * sim.TICK_S
    assert 180 <= wall <= 300
    pre = [p for t, p in frames if t < 0]
    assert pre and all(p["show"] is False and p["hide_planned"] is False for p in pre)
    at_t0 = [p for t, p in frames if t == 0]
    assert at_t0 and all(p["hide_planned"] and p["launch_active"] is False for p in at_t0)
    assert all(p["telemetry"]["test"] is True for _, p in frames)
    flying = [p for t, p in frames if 60 <= t <= 300]
    assert flying and all(p["show"] and p["hide_planned"] and p["telemetry"]["show"] for p in flying)
    names = {v["name"] for v in flying[-1]["vehicles"]}
    assert all("TEST" in name for name in names)
    boost = [p["telemetry"]["booster"] for t, p in frames if 0 < t < 420]
    assert any(int(b["speed_kmh"].replace(",", "")) >= 4_500 for b in boost if b["speed_kmh"] != "—")
    assert run.state.vehicles[sim.BOOSTER_KEY]["done"] is True
    assert run.state.vehicles[sim.SHIP_KEY]["done"] is True
    ship_speeds = [p["telemetry"]["ship"]["speed_kmh"] for t, p in frames if 1000 < t < 4000]
    assert all(int(s.replace(",", "")) > 25_000 for s in ship_speeds)
    assert frames[-1][1]["hold"] is True
    trail = [v for v in frames[-1][1]["vehicles"] if v["kind"] == "ship"][0]["trail"]
    assert len(trail) > 100


def test_flight_test_never_touches_network_or_real_cache(tmp_path):
    real_log = tmp_path / "starship_tracker_log.json"
    with patch.object(starship_tracker, "LOG_FILE", str(real_log)), \
         patch("requests.get", side_effect=AssertionError("network")), \
         patch("requests.Session.request", side_effect=AssertionError("network")), \
         patch.object(starship_tracker, "fetch_public_tracker", side_effect=AssertionError("fetch")), \
         patch("os.replace", side_effect=AssertionError("disk write")):
        run, _ = _run_all()
    assert run.finished
    assert not real_log.exists()
    assert run.state.persist is False
    assert run.state.path != starship_tracker.LOG_FILE


def test_test_mode_is_off_by_default():
    assert sim.env_mode({}) == ""
    assert sim.env_mode({"STARSHIP_TRACKER_TEST": "0"}) == ""
    assert sim.env_mode({"STARSHIP_TRACKER_TEST": "1"}) == "once"
    assert sim.env_mode({"STARSHIP_TRACKER_TEST": "loop"}) == "loop"
    poller = (ROOT / "starship_http.py").read_text(encoding="utf-8")
    assert "fixture=flight_test" in poller or "flight_test" in poller
    assert "live_positions" in poller
    assert 'position_source") != "fixture"' in poller or "fixture" in poller
