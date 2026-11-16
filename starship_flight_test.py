"""Simulated Starship flight for previewing the live tracker before launch.

The simulation builds documents in the same shape as SpaceX's public tracker
JSON and pushes them through the real ``TrackerState`` pipeline (ingest,
liftoff latch, handoff from the planned path, webcast strip, catch, hold),
on a virtual clock. The Settings preview flies several orbits (about four
minutes). A one-lap profile is still available for the short unit path.

It never touches the network and never writes the real tracker log: the
state is in-memory only (``persist=False``). Nothing here runs unless a
caller asks for ``?fixture=flight_test`` or sets ``STARSHIP_TRACKER_TEST``.
Positions from this module are fixture data, never a live vehicle fix.
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone

from starship_tracker import GPS_EPOCH, GPS_LEAP_SECONDS, TrackerState

STARBASE_LAT = 25.997
STARBASE_LON = -97.157
# Ground track: a ~26.5 deg inclined lap east from Starbase that comes home.
INCLINATION_DEG = 26.5
SHIP_KEY = "ship42"
BOOSTER_KEY = "booster19"

# Wall-clock tick. Each tick is one simulated poll.
TICK_S = 1.0
PRE_ROLL_S = -60.0
# Mission seconds per tick: 10x through booster catch (one tick is one real
# 10s poll), then 40x for the ship's coast, reentry and catch.
BOOST_WARP_S = 10.0
COAST_WARP_S = 40.0
WARP_SWITCH_S = 480.0
# Ticks of post-catch hold to show before handing back to the real tracker.
HOLD_SHOW_TICKS = 8
MAX_TICKS = 420
# Default preview and the explicit multi-orbit option. The ship coasts
# LONG_LAPS ground-track laps before the deorbit burn, with a faster coast
# warp so the whole flight still plays in about four minutes.
LONG_LAPS = 3
LONG_COAST_WARP_S = 90.0
LONG_MAX_TICKS = 480
COAST_SPEED_KM_S = 7.45

ENV_VAR = "STARSHIP_TRACKER_TEST"

# (mission s, altitude km, speed m/s, downrange km)
_ASCENT = [
    (0, 0.10, 0, 0.0),
    (10, 0.6, 60, 0.0),
    (20, 1.6, 120, 0.1),
    (30, 2.5, 180, 0.3),
    (60, 10.0, 450, 3.0),
    (90, 22.0, 800, 12.0),
    (120, 38.0, 1100, 30.0),
    (150, 56.0, 1380, 62.0),
    (162, 64.0, 1450, 80.0),
]
BOOSTER_KEYFRAMES = _ASCENT + [
    # Boostback, apogee near 72 km, then home to the tower.
    (175, 67.5, 1250, 90.0),
    (200, 70.5, 700, 98.0),
    (225, 72.0, 450, 100.0),
    (255, 68.0, 700, 92.0),
    (290, 55.0, 1050, 72.0),
    (320, 38.0, 1150, 50.0),
    (350, 18.0, 700, 25.0),
    (375, 6.0, 300, 8.0),
    (400, 1.2, 90, 1.5),
    (420, 0.12, 0, 0.0),
]
SHIP_ASCENT = _ASCENT + [
    (200, 85.0, 1900, 140.0),
    (260, 112.0, 2700, 260.0),
    (330, 138.0, 3800, 460.0),
    (400, 160.0, 5200, 760.0),
    (460, 178.0, 6500, 1110.0),
    (510, 190.0, 7450, 1450.0),
]
SECO_S = 510.0
DEORBIT_S = 4700.0
# (mission s, altitude km, speed m/s) after the deorbit burn.
SHIP_REENTRY = [
    (4700, 190.0, 7450),
    (4900, 150.0, 7480),
    (5000, 120.0, 7500),
    (5150, 80.0, 7200),
    (5300, 60.0, 5600),
    (5420, 45.0, 3600),
    (5520, 30.0, 2000),
    (5600, 18.0, 900),
    (5680, 8.0, 320),
    (5740, 2.5, 250),
    (5760, 1.0, 120),
    (5780, 0.15, 0),
]
SHIP_CATCH_S = 5780.0
BOOSTER_CATCH_S = 420.0
_TABLE_STEP_S = 10.0


def _lerp_frames(frames, t, cols):
    if t <= frames[0][0]:
        return [frames[0][c] for c in cols]
    for left, right in zip(frames, frames[1:]):
        if t <= right[0]:
            span = right[0] - left[0]
            u = (t - left[0]) / span if span else 1.0
            # Smoothstep keeps speed and altitude from kinking at keyframes.
            u = u * u * (3 - 2 * u)
            return [left[c] + (right[c] - left[c]) * u for c in cols]
    return [frames[-1][c] for c in cols]


def _ship_alt_speed(t):
    if t <= SECO_S:
        alt, speed = _lerp_frames(SHIP_ASCENT, t, (1, 2))
        return alt, speed
    if t < DEORBIT_S:
        # Slightly elliptical coast so the altitude readout breathes.
        phase = (t - SECO_S) / 5400.0 * 2 * math.pi
        return 190.0 + 4.0 * math.sin(phase), 7450.0 + 6.0 * math.cos(phase)
    alt, speed = _lerp_frames(SHIP_REENTRY, min(t, SHIP_CATCH_S), (1, 2))
    return alt, speed


def _build_ship_downrange():
    """Downrange km every 10s: ascent from keyframes, then integrated speed."""
    table = []
    t = 0.0
    dr = 0.0
    while t <= SHIP_CATCH_S + 1e-6:
        if t <= SECO_S:
            dr = _lerp_frames(SHIP_ASCENT, t, (3,))[0]
        else:
            _, speed = _ship_alt_speed(t - _TABLE_STEP_S / 2)
            dr += speed * _TABLE_STEP_S / 1000.0
        table.append(dr)
        t += _TABLE_STEP_S
    return table


_SHIP_DR = _build_ship_downrange()
LAP_KM = _SHIP_DR[-1]
_PHASE = math.acos(
    max(-1.0, min(1.0, math.sin(math.radians(STARBASE_LAT)) / math.sin(math.radians(INCLINATION_DEG))))
)


def _ship_downrange(t):
    if t <= 0:
        return 0.0
    if t >= SHIP_CATCH_S:
        return LAP_KM
    index = t / _TABLE_STEP_S
    lo = int(index)
    hi = min(lo + 1, len(_SHIP_DR) - 1)
    frac = index - lo
    return _SHIP_DR[lo] + (_SHIP_DR[hi] - _SHIP_DR[lo]) * frac


def ground_point(downrange_km):
    """Lat/lon along the lap. Every multiple of LAP_KM is Starbase."""
    f = max(0.0, downrange_km / LAP_KM)
    sin_i = math.sin(math.radians(INCLINATION_DEG))
    lat = math.degrees(math.asin(sin_i * math.cos(2 * math.pi * f + _PHASE)))
    lon = STARBASE_LON + 360.0 * f
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lat, lon


# Peak westward offset of the extra laps (real tracks step ~22.5 deg/lap).
EXTRA_LAP_DRIFT_DEG = 24.0


def extra_coast_s(laps=1):
    """Coast time added for each lap past the first (exactly one LAP_KM)."""
    return max(0, int(laps) - 1) * LAP_KM / COAST_SPEED_KM_S


def ship_catch_s(laps=1):
    return SHIP_CATCH_S + extra_coast_s(laps)


def ship_sample(t, laps=1):
    """Ship (lat, lon, alt_m, speed_mps) at mission time ``t`` seconds.

    ``laps`` > 1 inserts whole extra coast laps before the deorbit burn, so
    the ground track wraps the antimeridian and crosses its own earlier laps.
    """
    extra = extra_coast_s(laps)
    if t <= 0:
        return STARBASE_LAT, STARBASE_LON, 150.0, 0.0
    if t >= SHIP_CATCH_S + extra:
        return STARBASE_LAT, STARBASE_LON, 150.0, 0.0
    if extra and t >= DEORBIT_S:
        if t < DEORBIT_S + extra:
            phase = (t - SECO_S) / 5400.0 * 2 * math.pi
            alt_km = 190.0 + 4.0 * math.sin(phase)
            speed = 7450.0 + 6.0 * math.cos(phase)
            dr = _ship_downrange(DEORBIT_S) + (t - DEORBIT_S) * COAST_SPEED_KM_S
            lat, lon = ground_point(dr)
            # Earth turns under the orbit, so real laps step west. Bow the
            # extra laps west (0 at both ends) so they sit beside lap 1 and
            # the deorbit still lands at Starbase.
            u = (t - DEORBIT_S) / extra
            lon = ((lon - EXTRA_LAP_DRIFT_DEG * math.sin(math.pi * u) + 180.0) % 360.0) - 180.0
            return lat, lon, alt_km * 1000.0, speed
        base = t - extra
        alt_km, speed = _ship_alt_speed(base)
        lat, lon = ground_point(_ship_downrange(base) + (int(laps) - 1) * LAP_KM)
        return lat, lon, alt_km * 1000.0, max(0.0, speed)
    alt_km, speed = _ship_alt_speed(t)
    lat, lon = ground_point(_ship_downrange(t))
    return lat, lon, alt_km * 1000.0, max(0.0, speed)


def booster_sample(t):
    if t <= 0 or t >= BOOSTER_CATCH_S:
        return STARBASE_LAT, STARBASE_LON, 120.0, 0.0
    alt_km, speed, dr = _lerp_frames(BOOSTER_KEYFRAMES, t, (1, 2, 3))
    lat, lon = ground_point(dr)
    return lat, lon, alt_km * 1000.0, max(0.0, speed)


def utc_to_gps(moment):
    return (moment - GPS_EPOCH).total_seconds() + GPS_LEAP_SECONDS


def _vehicle_doc(sampler, mission_t, gps, flying_until):
    lat, lon, alt, speed = sampler(mission_t)
    trajectory = []
    if 0 < mission_t < flying_until:
        for step in range(1, 5):
            ahead = min(mission_t + step * 60, flying_until)
            p_lat, p_lon, p_alt, _ = sampler(ahead)
            trajectory.append({
                "time": gps + step * 60,
                "latitude": p_lat,
                "longitude": p_lon,
                "altitude": p_alt,
                "r_ecef": [0, 0, 0],
            })
    return {
        "current": {
            "gps_time": gps,
            "mission_time": mission_t,
            "altitude": alt,
            "speed": speed,
            "latitude": lat,
            "longitude": lon,
            "r_ecef": [0, 0, 0],
        },
        "trajectory": trajectory,
    }


def simulated_feed(mission_t, liftoff_utc, laps=1):
    """One tracker document, same shape as starship_tracker_public.json."""
    moment = liftoff_utc + timedelta(seconds=mission_t)
    gps = utc_to_gps(moment)

    def ship(t):
        return ship_sample(t, laps)

    return {
        BOOSTER_KEY: _vehicle_doc(booster_sample, mission_t, gps, BOOSTER_CATCH_S),
        SHIP_KEY: _vehicle_doc(ship, mission_t, gps, ship_catch_s(laps)),
        "metadata": {
            "generation_time": gps,
            "trajectory_version": 1,
            "coordinate_frame": "ECEF",
            "sample_period": 60,
        },
    }


def warp_step_s(mission_t, laps=1):
    if mission_t < WARP_SWITCH_S:
        return BOOST_WARP_S
    if laps > 1:
        return LONG_COAST_WARP_S
    return COAST_WARP_S


def env_mode(environ=None):
    """'' (off), 'once', 'loop' or 'laps' from STARSHIP_TRACKER_TEST."""
    value = str((environ if environ is not None else os.environ).get(ENV_VAR, "")).strip().lower()
    if value in ("loop", "repeat"):
        return "loop"
    if value in ("laps", "orbits", "long"):
        return "laps"
    if value in ("1", "true", "yes", "on", "once"):
        return "once"
    return ""


class FlightTestRun:
    """Drives an in-memory TrackerState with simulated feed documents."""

    def __init__(self, start_utc=None, laps=1):
        self.laps = max(1, int(laps or 1))
        self.catch_s = ship_catch_s(self.laps)
        self.max_ticks = LONG_MAX_TICKS if self.laps > 1 else MAX_TICKS
        start = start_utc or datetime.now(timezone.utc)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        # Virtual liftoff one pre-roll after the start, on its own clock.
        self.liftoff_utc = start - timedelta(seconds=PRE_ROLL_S)
        self.mission_t = PRE_ROLL_S
        self.state = TrackerState(path=os.devnull, persist=False)
        self.launch_data = {
            "upcoming": [{
                "net": self.liftoff_utc.isoformat(),
                "rocket": "Starship",
                "mission": "Starship flight test (simulated)",
            }],
            "previous": [],
        }
        self.ticks = 0
        self.hold_ticks = 0
        self.finished = False
        self.last_payload = {"show": False, "vehicles": [], "test": True}

    def virtual_now(self):
        return self.liftoff_utc + timedelta(seconds=self.mission_t)

    def step(self):
        """Advance one tick and return the globe payload for it."""
        if self.finished:
            return self.last_payload
        now = self.virtual_now()
        self.state.ingest(simulated_feed(self.mission_t, self.liftoff_utc, self.laps), now)
        self.state.update_mission(self.launch_data, now)
        payload = self.state.globe_payload(now, in_window=True)
        payload = mark_test_payload(payload)
        warp = warp_step_s(self.mission_t, self.laps)
        # Mission seconds per wall second, so the globe can glide between
        # ticks at the same virtual speed (real feed: 1.0).
        payload["clock_rate"] = warp / TICK_S
        if self.mission_t >= self.catch_s and self.state.hold_started:
            self.hold_ticks += 1
        self.ticks += 1
        if self.hold_ticks > HOLD_SHOW_TICKS or self.ticks >= self.max_ticks:
            self.finished = True
        self.mission_t += warp
        self.last_payload = payload
        return payload


def mark_test_payload(payload):
    payload = dict(payload or {})
    payload["test"] = True
    telemetry = dict(payload.get("telemetry") or {})
    telemetry["test"] = True
    payload["telemetry"] = telemetry
    vehicles = []
    for vehicle in payload.get("vehicles") or []:
        item = dict(vehicle)
        item["name"] = f"{item.get('name') or item.get('key')}  TEST"
        vehicles.append(item)
    payload["vehicles"] = vehicles
    return payload


def expected_wall_seconds():
    """Rough length of one run, for the Settings subtitle and tests."""
    run = FlightTestRun(datetime(2026, 1, 1, tzinfo=timezone.utc))
    while not run.finished:
        run.step()
    return run.ticks * TICK_S
