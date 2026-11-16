"""Live Starship ship + booster positions from SpaceX's public tracker.

The feed has no CORS headers, so the dashboard fetches it here and hands a
small payload to the globe. Parsing, the sanity filter, and the flown-path
log are Qt-free so tests can run them without a window.

SpaceX's two hosts stay first. Beyond Stage Zero, then StarDash's live
feed, are used only after both of those return no vehicle fix, and only
once a Starship NET has passed.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import subprocess
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests

from dashboard_paths import CACHE_DIR
from launch_dates import _get_parsed_dt

logger = logging.getLogger(__name__)

# spacex.com/vehicle-tracker polls this host. It sends ACAO: *.
PRIMARY_TRACKER_URL = (
    "https://content.spacex.com/cms-assets/starship_tracker_public.json"
)
# Used only after the primary errors, times out, or returns a non-200.
FALLBACK_TRACKER_URL = (
    "https://sxcontent9668.azureedge.us/cms-assets/starship_tracker_public.json"
)
# Beyond Stage Zero's archived trail. Their proxy currently 404s; tried first
# anyway so a working response is preferred over StarDash.
BSZ_SHIP_TRAIL_URL = "https://starship.beyondstagezero.com/api/ship-trail"
# One StarDash history read. ``w`` is the current one-minute cache bucket and
# returns the flight from the pad to now (thinned to a few hundred points).
# Hourly ``h=<unix hour>`` chunks answer ``bad chunk request`` and are not used.
STARDASH_HISTORY_URL = "https://stardash.live/api/history"
BACKFILL_TIMEOUT_S = 15
# A trail whose first point is still this far after liftoff is only partial.
BACKFILL_GAP_S = 20 * 60
BACKFILL_MAX_DT_S = 600.0
TRAIL_DEDUPE_S = 20.0
# Hobby mirrors, used only after both SpaceX hosts have no fix.
# Beyond Stage Zero returns the raw tracker document (speed is 0; ECEF is real).
BEYOND_STAGE_ZERO_URL = "https://starship.beyondstagezero.com/api/tracker"
# StarDash's live sample plus a 16-point predicted path. /api/history is the
# past chunk; the flown trail is accumulated from current fixes, same as SpaceX,
# so the poll does not also pull history.
STARDASH_FEED_URL = "https://stardash.live/api/feed"
TRACKER_URL = PRIMARY_TRACKER_URL
# Both hobby feeds are cached about once a second and move on a ~30–60 s sample.
# Do not poll them on the 10 s SpaceX cadence.
HOBBY_POLL_S = 30
HOBBY_TIMEOUT_S = 8
# A newest sample older than this is not a live fix.
HOBBY_STALE_S = 180
# Mean sidereal Earth rotation. v_eci = v_ecef + ω × r.
EARTH_ROTATION_RAD_S = 7.2921151467e-5
# While the fallback is sticky, try the primary again on this interval.
PRIMARY_RETRY_S = 10 * 60
# Nothing in the poller, including Retry-After, waits less than this.
MIN_POLL_S = 5
LOG_FILE = os.path.join(CACHE_DIR, "starship_tracker_log.json")

# GPS time is seconds since 1980-01-06 and does not count leap seconds.
# 18s is the GPS-UTC offset in force since 2017-01-01.
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_LEAP_SECONDS = 18

ACTIVE_POLL_S = 10
IDLE_POLL_S = 300
ACTIVE_JITTER_S = 2
IDLE_JITTER_S = 30
_UA_REPO = "https://github.com/hwpaige/spacex-dashboard"
FETCH_TIMEOUT_S = 4
STALE_AFTER_S = 60
# Fast poll starts at T-30min. After liftoff it stays on until the mission
# ends (not a fixed T+hours). NET is re-read every cycle.
WINDOW_BEFORE_S = 30 * 60
# Landed or caught: barely moving and near the ground, held for 10 minutes.
GROUNDED_ALT_M = 2000.0
GROUNDED_SPEED_MPS = 2.0
GROUNDED_STABLE_S = 10 * 60
# Hobby feeds leave speed at 0, so we reconstruct it from ECEF. Inertial
# speed includes Earth rotation (~418 m/s at Starbase). A vehicle that is
# barely moving relative to the ground should report ~0, not 1,500 km/h.
PAD_ECEF_SPEED_MPS = 20.0
# A key that leaves the feed. A key that stays but stops updating is not this.
KEY_GONE_S = 30 * 60
# Safety stop so a stuck "in flight" flag cannot poll forever.
MISSION_CAP_S = 48 * 3600
# Keep the flown line up after every vehicle is down, then go idle.
POST_CATCH_HOLD_S = 60 * 60
# A grounded vehicle whose last fresh fix is this old is treated as down
# even if the feed never sends another sample.
STALE_GROUNDED_S = 15 * 60
# Markers and the flown trail wait for a real liftoff, not a ship on the pad.
# Two samples must agree: the mission clock is running and the vehicle is
# clearly off the pad (above ~1 km or faster than ~100 m/s). The planned
# path is separate: it hides at T-0, before those two samples.
FRESH_SAMPLES_FOR_HANDOVER = 2
AIRBORNE_ALT_M = 1000.0
AIRBORNE_SPEED_MPS = 100.0
# Coast band for the orbit counter. Ascent is slower than this; reentry
# drops through the altitude floor while still fast. A lap is 360° of
# unwrapped ground-track longitude after the first sample in this band,
# but only when the trail reaches back to liftoff. A partial trail is
# timed from liftoff with the Kepler period of the current state.
ORBIT_COAST_SPEED_MPS = 7000.0
ORBIT_COAST_ALT_M = 150_000.0
_MU_M3_S2 = 3.986004418e14
PREDICTED_MIN_ALT_M = -1000.0
# Orbital coast is ~447 km/min (26,800 km/h), so the floor clears ~28,000 km/h
# (467 km/min) with margin. A faster vehicle gets speed * 60 s * 1.5, capped
# well below the pad garbage (thousands of km per minute).
PREDICTED_MAX_KM_PER_MIN = 600.0
PREDICTED_SPEED_MARGIN = 1.5
PREDICTED_MAX_KM_PER_MIN_CAP = 900.0
MOVING_SPEED_MPS = 1.0
ALTITUDE_CHANGE_M = 200.0
TRAIL_MIN_STEP_KM = 0.4
# Multi-orbit flights: 2400 stored points is ~6.7 h of 10 s samples before
# the oldest laps are decimated; the globe gets up to 1800 (~5 h at full
# rate; past that the head is thinned and the latest third stays dense).
MAX_TRAIL_POINTS = 2400
DISPLAY_TRAIL_CAP = 1800

_EARTH_KM = 6371.0


def gps_to_utc(gps_seconds):
    """Convert GPS-epoch seconds to an aware UTC datetime."""
    return GPS_EPOCH + timedelta(seconds=float(gps_seconds) - GPS_LEAP_SECONDS)


def classify_vehicle(key):
    """ship* -> ship, booster* -> booster. Anything else is still tracked."""
    name = str(key or "").strip().lower()
    if name.startswith("ship"):
        return "ship"
    if name.startswith("booster"):
        return "booster"
    return "other"


def display_name(key, kind):
    digits = "".join(ch for ch in str(key) if ch.isdigit())
    if kind == "ship" and digits:
        return f"Ship {digits}"
    if kind == "booster" and digits:
        return f"Booster {digits}"
    return str(key)


def format_mission_time(seconds):
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return ""
    sign = "-" if value < 0 else "+"
    whole = int(abs(value))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"T{sign}{hours}:{minutes:02d}:{secs:02d}"
    return f"T{sign}{minutes}:{secs:02d}"


def format_altitude(meters):
    """Globe label altitude. Kilometres only."""
    try:
        meters = float(meters)
    except (TypeError, ValueError):
        return ""
    return f"{meters / 1000.0:.1f} km"


def format_speed(mps):
    """Globe label speed: km/h like the webcast strip, e.g. '26,816 km/h'."""
    try:
        kmh = float(mps) * 3.6
    except (TypeError, ValueError):
        return ""
    return f"{_with_commas(f'{kmh:.0f}')} km/h"


# The public tracker reports speed in m/s and altitude in meters.
# The webcast strip shows km/h and km.
SHIP_COLOR = "#515ADB"
BOOSTER_COLOR = "#FF9F1C"
_SPEED_GAUGE_KMH = 28000.0
_ALT_GAUGE_M = 200000.0


def _with_commas(number_text):
    """Group digits with an ASCII comma. Independent of LC_NUMERIC."""
    sign = ""
    if number_text.startswith("-"):
        sign = "-"
        number_text = number_text[1:]
    whole, dot, frac = number_text.partition(".")
    groups = []
    while whole:
        groups.append(whole[-3:])
        whole = whole[:-3]
    grouped = ",".join(reversed(groups)) if groups else "0"
    if dot:
        return f"{sign}{grouped}.{frac}"
    return f"{sign}{grouped}"


def format_speed_kmh(mps):
    """Convert feed speed (m/s) to a webcast km/h readout."""
    try:
        kmh = float(mps) * 3.6
    except (TypeError, ValueError):
        return "—"
    return _with_commas(f"{kmh:.0f}")


def format_altitude_km(meters):
    try:
        km = float(meters) / 1000.0
    except (TypeError, ValueError):
        return "—"
    if abs(km) >= 100:
        return _with_commas(f"{km:.1f}")
    return f"{km:.1f}"


def format_webcast_clock(seconds):
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "T+00:00:00"
    sign = "-" if value < 0 else "+"
    whole = int(abs(value))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    return f"T{sign}{hours:02d}:{minutes:02d}:{secs:02d}"


def webcast_identity(key, kind):
    digits = "".join(ch for ch in str(key or "") if ch.isdigit())
    if kind == "booster":
        return "SUPER HEAVY", (f"B{digits}" if digits else "—")
    if kind == "ship":
        return "STARSHIP", (f"S{digits}" if digits else "—")
    label = str(key or "VEHICLE").upper()
    return label, (digits or "—")


def _gauge(value, full):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if full <= 0:
        return 0.0
    return max(0.0, min(1.0, number / full))


def readout_motion(slot, now_utc=None):
    """Alt/speed at the latest fix and at the next predicted point.

    The strip lerps between them. ``motion_age_s`` is how old the fix
    already is, so a 30 s sample does not restart the lerp at zero.
    """
    current = (slot or {}).get("current") or {}
    alt = _num(current.get("alt_m"))
    speed = _num(current.get("speed_mps")) or 0.0
    gps = _num(current.get("gps_s"))
    age = 0.0
    if gps is not None and now_utc is not None:
        age = max(0.0, (_aware(now_utc) - gps_to_utc(gps)).total_seconds())
    alt_next = alt
    speed_next = speed
    span = None
    future = []
    if gps is not None:
        for point in (slot or {}).get("predicted") or []:
            when = _num(point.get("time"))
            if when is not None and when > gps + 1.0:
                future.append(point)
    if future and alt is not None:
        nxt = future[0]
        span = nxt["time"] - gps
        alt_next = _num(nxt.get("alt_m"))
        if alt_next is None:
            alt_next = alt
        x0, y0, z0 = geodetic_to_ecef(current.get("lat") or 0.0, current.get("lon") or 0.0, alt)
        x1, y1, z1 = geodetic_to_ecef(nxt.get("lat") or 0.0, nxt.get("lon") or 0.0, alt_next)
        derived = inertial_speed_mps(
            {"gps": gps, "ex": x0, "ey": y0, "ez": z0},
            {"gps": nxt["time"], "ex": x1, "ey": y1, "ez": z1},
        )
        if derived is not None:
            speed_next = derived
    return {
        "alt_m": alt,
        "speed_mps": speed,
        "alt_m_next": alt_next,
        "speed_mps_next": speed_next,
        "motion_span_s": span,
        "motion_age_s": age,
    }


def vehicle_readout(key, slot, kind, now_utc=None):
    """One side of the strip. Missing vehicles are dashes, stale ones stay grey."""
    title, code = webcast_identity(key, kind)
    color = SHIP_COLOR if kind == "ship" else BOOSTER_COLOR
    motion = readout_motion(slot, now_utc)
    blank = {
        "title": title,
        "code": code,
        "speed_kmh": "—",
        "alt_km": "—",
        "stale": False,
        "present": False,
        "speed_frac": 0.0,
        "alt_frac": 0.0,
        "color": color,
        "alt_m": None,
        "speed_mps": 0.0,
        "alt_m_next": None,
        "speed_mps_next": 0.0,
        "motion_span_s": None,
        "motion_age_s": 0.0,
    }
    if not slot or slot.get("missing") or not slot.get("current"):
        return blank
    current = slot.get("current") or {}
    speed = current.get("speed_mps") or 0
    alt = current.get("alt_m")
    blank.update({
        "speed_kmh": format_speed_kmh(speed),
        "alt_km": format_altitude_km(alt) if alt is not None else "—",
        "present": True,
        "speed_frac": _gauge(float(speed) * 3.6, _SPEED_GAUGE_KMH),
        "alt_frac": _gauge(alt if alt is not None else 0, _ALT_GAUGE_M),
    })
    blank.update(motion)
    return blank


def unwrap_lon_delta(prev_lon, lon):
    """Shortest east-positive step, in degrees. Antimeridian is not a lap.

    Same fold as the globe's multi-lap trail (``trailLapIndex``): the
    per-step delta is wrapped into (-180, 180], with an exact -180° step
    kept eastbound so a boundary sample does not cancel a lap.
    """
    try:
        delta = float(lon) - float(prev_lon)
    except (TypeError, ValueError):
        return 0.0
    if delta != delta:  # NaN
        return 0.0
    delta = (delta + 540.0) % 360.0 - 180.0
    if delta == -180.0:
        return 180.0
    return delta


def sample_in_coast(sample):
    """True when this fix is in the orbital coast, not ascent or reentry."""
    if not sample:
        return False
    try:
        speed = float(sample.get("speed_mps") or 0)
        alt = float(sample.get("alt_m") or 0)
    except (TypeError, ValueError):
        return False
    return speed >= ORBIT_COAST_SPEED_MPS and alt >= ORBIT_COAST_ALT_M


def ship_completed_laps(samples):
    """Completed 360° laps since the first coast sample. None before insertion.

    ``samples`` is chronological. Longitude is unwrapped with
    ``unwrap_lon_delta``, matching the globe trail. The insertion fix
    itself is lap 0.
    """
    start = None
    for index, sample in enumerate(samples or []):
        if sample_in_coast(sample) and sample.get("lon") is not None:
            start = index
            break
    if start is None:
        return None
    total = 0.0
    prev = samples[start].get("lon")
    for sample in samples[start + 1:]:
        lon = sample.get("lon")
        if lon is None:
            continue
        total += unwrap_lon_delta(prev, lon)
        prev = lon
    return math.floor(abs(total) / 360.0 + 1e-9)


def state_radius_m(sample):
    """Geocentric radius. An ECEF fix wins; otherwise lat/lon/alt."""
    if not isinstance(sample, dict):
        return None
    raw = sample.get("ecef") or sample.get("r_ecef")
    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        x, y, z = _num(raw[0]), _num(raw[1]), _num(raw[2])
        if None not in (x, y, z):
            radius = math.sqrt(x * x + y * y + z * z)
            if radius > 6_000_000.0:
                return radius
    alt = _num(sample.get("alt_m") if "alt_m" in sample else sample.get("alt"))
    if alt is None:
        return None
    lat = _num(sample.get("lat") if "lat" in sample else sample.get("latitude")) or 0.0
    lon = _num(sample.get("lon") if "lon" in sample else sample.get("longitude")) or 0.0
    x, y, z = geodetic_to_ecef(lat, lon, alt)
    return math.sqrt(x * x + y * y + z * z)


def orbital_period_s(sample):
    """Osculating Kepler period from inertial speed and geocentric radius.

    ``speed_mps`` is the inertial speed the tracker already stores (ECEF
    delta plus Earth rotation). Escape and non-LEO results are ignored.
    """
    radius = state_radius_m(sample)
    speed = _num((sample or {}).get("speed_mps") if isinstance(sample, dict) else None)
    if radius is None or speed is None or speed < 1000.0 or radius < 6_300_000.0:
        return None
    specific = 2.0 / radius - (speed * speed) / _MU_M3_S2
    if specific <= 0.0:
        return None
    sma = 1.0 / specific
    period = 2.0 * math.pi * math.sqrt(sma ** 3 / _MU_M3_S2)
    if not (50.0 * 60.0 <= period <= 200.0 * 60.0):
        return None
    return period


def sample_epoch(sample):
    """UTC time of a trail sample, from GPS seconds or an ISO stamp."""
    if not isinstance(sample, dict):
        return None
    gps = _num(sample.get("gps") if "gps" in sample else sample.get("gps_time"))
    if gps is not None:
        return gps_to_utc(gps)
    return _stamp(sample.get("utc"))


def trail_reaches_liftoff(samples, liftoff_utc, gap_s=BACKFILL_GAP_S):
    """True when some sample is within ``gap_s`` of liftoff (pad or insertion)."""
    lifted = _aware(liftoff_utc)
    if lifted is None:
        return False
    epochs = [sample_epoch(sample) for sample in (samples or [])]
    epochs = [item for item in epochs if item is not None]
    if not epochs:
        return False
    return min(epochs) <= lifted + timedelta(seconds=gap_s)


def coast_trail_reaches_liftoff(samples, liftoff_utc, gap_s=BACKFILL_GAP_S):
    """True when a coast fix, not merely a pad point, is within ``gap_s`` of liftoff.

    A saved pad sample from T+0 plus an hour of local coast fixes still
    "reaches liftoff" if every sample counts. The orbit sum only sees the
    coast, so that trail would report ORBIT 2 twenty hours into the flight.
    """
    coast = [sample for sample in (samples or []) if sample_in_coast(sample)]
    return trail_reaches_liftoff(coast, liftoff_utc, gap_s)


def _ecef_vector(sample):
    """ECEF metres. A stored vector wins; otherwise lat/lon/alt."""
    if not isinstance(sample, dict):
        return None
    raw = sample.get("ecef") or sample.get("r_ecef")
    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        x, y, z = _num(raw[0]), _num(raw[1]), _num(raw[2])
        if None not in (x, y, z):
            radius = math.sqrt(x * x + y * y + z * z)
            if radius > 6_000_000.0:
                return (x, y, z)
    alt = _num(sample.get("alt_m") if "alt_m" in sample else sample.get("alt"))
    lat = _num(sample.get("lat") if "lat" in sample else sample.get("latitude"))
    lon = _num(sample.get("lon") if "lon" in sample else sample.get("longitude"))
    if None in (lat, lon, alt):
        return None
    return tuple(geodetic_to_ecef(lat, lon, alt))


def _rotate_z(vector, ang):
    c, s = math.cos(ang), math.sin(ang)
    x, y, z = vector
    return (x * c - y * s, x * s + y * c, z)


def inertial_step_rad(earlier, later, dt_s):
    """Angle swept from one ECEF fix to the next, in ECI.

    A fixed inertial point drifts west in ECEF, so
    ``r_ecef = Rz(-ω t) r_eci`` and ``r_eci = Rz(ω t) r_ecef``. Comparing
    the two fixes in the frame of the first is ``earlier`` against
    ``Rz(ω dt) later``. The result is the smaller angle, so the step has
    to be well under half a rev.
    """
    if dt_s and dt_s > 0.0:
        later = _rotate_z(later, EARTH_ROTATION_RAD_S * dt_s)
    n1 = math.sqrt(sum(c * c for c in earlier))
    n2 = math.sqrt(sum(c * c for c in later))
    if n1 < 1.0e6 or n2 < 1.0e6:
        return 0.0
    dot = (earlier[0] * later[0] + earlier[1] * later[1] + earlier[2] * later[2]) / (n1 * n2)
    dot = max(-1.0, min(1.0, dot))
    return math.acos(dot)


def inertial_revolutions(samples):
    """Inertial revolutions since the first on-orbit sample.

    Insertion is the first coast fix: speed at least 7 km/s and altitude
    at least 150 km. That is after ascent and before reentry, which is
    the SECO stand-in the coast band already uses. An ascending node would
    start the count at whatever node the ground track happens to cross.
    A gap longer than about half a rev is filled with the Kepler period
    instead of the chord, which cannot see past 180°.
    """
    coast = []
    for sample in samples or []:
        if not sample_in_coast(sample):
            continue
        if _ecef_vector(sample) is None:
            continue
        coast.append(sample)
    if not coast:
        return None
    total = 0.0
    for left, right in zip(coast, coast[1:]):
        t1, t2 = sample_epoch(left), sample_epoch(right)
        dt = (t2 - t1).total_seconds() if t1 is not None and t2 is not None else 0.0
        period = orbital_period_s(right) or orbital_period_s(left)
        if dt > 0.0 and period and dt > 0.45 * period:
            total += dt / period
            continue
        step = inertial_step_rad(_ecef_vector(left), _ecef_vector(right), dt)
        total += step / (2.0 * math.pi)
    return total


def estimated_orbit_number(sample, liftoff_utc, now_utc):
    """Current orbit from elapsed time since liftoff and the Kepler period.

    Used when the trail does not reach insertion. Orbit 1 still includes
    ascent. Counting longitude from the first local coast sample would
    stay on ORBIT 1 for the whole flight.
    """
    period = orbital_period_s(sample) or (90.0 * 60.0)
    lifted = _aware(liftoff_utc)
    now = _aware(now_utc)
    if lifted is None or now is None:
        return None
    elapsed = (now - lifted).total_seconds()
    if elapsed < 0.0:
        return None
    return int(math.floor(elapsed / period)) + 1


def ship_orbit_status(slot, liftoff_utc=None, now_utc=None):
    """Orbit in progress, or None when the ship is not in coast.

    ``ORBIT N`` is completed inertial revolutions since insertion, plus
    one. Right after insertion that is ``ORBIT 1``. A trail that reaches
    liftoff sums the ECI angle from the first coast sample and then runs
    the Kepler period forward to ``now``. A trail that starts mid-flight
    uses the time since liftoff instead. Hidden on the pad, in ascent,
    in reentry, and when the ship has left the feed.
    """
    if not slot or slot.get("missing"):
        return None
    current = slot.get("current") or {}
    if not sample_in_coast(current):
        return None
    samples = list(slot.get("samples") or [])
    last = samples[-1] if samples else None
    same_fix = (
        last is not None
        and last.get("lon") == current.get("lon")
        and last.get("mission_time_s") == current.get("mission_time_s")
    )
    if not same_fix:
        samples.append(current)
    number = None
    revs = None
    estimate = None
    if liftoff_utc is not None and now_utc is not None:
        estimate = estimated_orbit_number(current, liftoff_utc, now_utc)
    # A pad point near liftoff does not make the coast trail complete.
    # Count from liftoff plus the current period until a coast fix itself
    # reaches back to insertion.
    reaches = liftoff_utc is not None and coast_trail_reaches_liftoff(samples, liftoff_utc)
    if estimate is not None and not reaches:
        number = estimate
    if number is None:
        revs = inertial_revolutions(samples)
        if revs is None:
            return None
        coast = [sample for sample in samples if sample_in_coast(sample)]
        epoch = sample_epoch(coast[-1]) if coast else None
        period = orbital_period_s(current)
        now = _aware(now_utc)
        if epoch is not None and now is not None and period:
            extra = (now - epoch).total_seconds()
            if extra > 0.0:
                revs += extra / period
        number = int(math.floor(revs + 1e-9)) + 1
        # A hole the chord cannot see (no speed on the older fix) leaves the
        # local sum on ORBIT 1 or 2. The liftoff period is the flight clock.
        if estimate is not None and number + 1 < estimate:
            number = estimate
    completed = max(0, number - 1)
    return {
        "in_orbit": True,
        "completed_laps": completed,
        "orbit_number": number,
        "orbit_label": f"ORBIT {number}",
    }


def haversine_km(lat1, lon1, lat2, lon2):
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    )
    return 2 * _EARTH_KM * math.asin(min(1.0, math.sqrt(a)))


def _num(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _point(raw):
    if not isinstance(raw, dict):
        return None
    lat = _num(raw.get("latitude"))
    lon = _num(raw.get("longitude"))
    alt = _num(raw.get("altitude"))
    if lat is None or lon is None or alt is None:
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return {
        "lat": lat,
        "lon": lon,
        "alt_m": alt,
        "time": _num(raw.get("time") if "time" in raw else raw.get("gps_time")),
    }


def predicted_jump_limit_km_per_min(speed_mps=None):
    """Largest believable gap between projected points, in km per minute."""
    limit = PREDICTED_MAX_KM_PER_MIN
    speed = _num(speed_mps)
    if speed is not None and speed > 0:
        limit = max(limit, speed * 60.0 / 1000.0 * PREDICTED_SPEED_MARGIN)
    return min(limit, PREDICTED_MAX_KM_PER_MIN_CAP)


def predicted_is_sane(points, speed_mps=None):
    """Drop pad garbage: underground paths and hemisphere-sized jumps."""
    if not points or len(points) < 2:
        return False
    max_km_per_min = predicted_jump_limit_km_per_min(speed_mps)
    for point in points:
        if point["alt_m"] < PREDICTED_MIN_ALT_M:
            return False
    for left, right in zip(points, points[1:]):
        dt = right["time"] - left["time"] if right["time"] is not None and left["time"] is not None else 60.0
        minutes = max(abs(dt), 1.0) / 60.0
        dist = haversine_km(left["lat"], left["lon"], right["lat"], right["lon"])
        if dist / minutes > max_km_per_min:
            return False
    return True


def utc_to_gps(moment):
    """Inverse of ``gps_to_utc`` for an aware UTC datetime."""
    moment = _aware(moment)
    return (moment - GPS_EPOCH).total_seconds() + GPS_LEAP_SECONDS


def inertial_speed_mps(earlier, later, max_dt=120.0):
    """Inertial speed (m/s) from two ECEF fixes.

    Subtracting the positions is Earth-relative (~7.3 km/s in this orbit).
    The tracker ``speed`` field is what the webcast strip shows, and that
    gauge is scaled to circular-orbit inertial speed (28,000 km/h). Adding
    Earth rotation, v_eci = v_ecef + ω × r, is that speed: about 7.6–7.8 km/s
    for Ship 42 in coast. ``earlier`` and ``later`` carry gps, ex, ey, ez.
    ``max_dt`` rejects a gap too wide for a straight chord. Live samples use
    120 s. History backfill allows several minutes because the archive is
    thinned to a few minutes between points.
    """
    gps0 = _num(earlier.get("gps") if isinstance(earlier, dict) else None)
    gps1 = _num(later.get("gps") if isinstance(later, dict) else None)
    if gps0 is None or gps1 is None:
        return None
    dt = gps1 - gps0
    if dt < 1.0 or dt > max_dt:
        return None
    x0, y0, z0 = (_num(earlier.get("ex")), _num(earlier.get("ey")), _num(earlier.get("ez")))
    x1, y1, z1 = (_num(later.get("ex")), _num(later.get("ey")), _num(later.get("ez")))
    if None in (x0, y0, z0, x1, y1, z1):
        return None
    vx = (x1 - x0) / dt
    vy = (y1 - y0) / dt
    vz = (z1 - z0) / dt
    # ω is along ECEF +Z. ω × r = (−ω y, ω x, 0).
    wx = -EARTH_ROTATION_RAD_S * y1
    wy = EARTH_ROTATION_RAD_S * x1
    speed = math.sqrt((vx + wx) ** 2 + (vy + wy) ** 2 + vz ** 2)
    if speed > 12000.0:
        return None
    return speed


# WGS84. Used when a hobby trajectory has lat/lon/alt but no r_ecef.
_WGS84_A = 6378137.0
_WGS84_E2 = 6.69437999014e-3


def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    """Geodetic degrees and metres to ECEF metres."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    slat = math.sin(lat)
    clat = math.cos(lat)
    normal = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * slat * slat)
    return [
        (normal + alt_m) * clat * math.cos(lon),
        (normal + alt_m) * clat * math.sin(lon),
        (normal * (1.0 - _WGS84_E2) + alt_m) * slat,
    ]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(vector):
    return math.sqrt(_dot(vector, vector))


def inertial_velocity_mps(r_from, r_to, dt_s, at_end=False):
    """Inertial velocity from an ECEF chord. ``v_eci = v_ecef + ω × r``.

    ``ω × r`` is taken at the current end of the chord: the start for a
    forward step, the end for a step that arrives at the current fix.
    """
    if r_from is None or r_to is None or dt_s < 1.0 or dt_s > 180.0:
        return None
    vx = (r_to[0] - r_from[0]) / dt_s
    vy = (r_to[1] - r_from[1]) / dt_s
    vz = (r_to[2] - r_from[2]) / dt_s
    origin = r_to if at_end else r_from
    return (
        vx - EARTH_ROTATION_RAD_S * origin[1],
        vy + EARTH_ROTATION_RAD_S * origin[0],
        vz,
    )


def osculating_apsides_m(position, velocity):
    """Apogee and perigee altitude in metres above the 6371 km sphere.

    ``position`` is ECEF metres and ``velocity`` is inertial
    (``v_ecef + ω × r``). The altitudes are geocentric radius at apoapsis
    and periapsis minus Earth's mean radius, 6371 km — the same sphere
    this module uses for ground distance. WGS84 geodetic height follows
    the ellipsoid and would move several kilometres with latitude, so it
    is not the apogee number. A hyperbola, or a state that is not a
    closed ellipse, returns ``(None, None)``.
    """
    if position is None or velocity is None:
        return None, None
    rmag = _norm(position)
    if rmag < 6_300_000.0:
        return None, None
    energy = _dot(velocity, velocity) / 2.0 - _MU_M3_S2 / rmag
    if energy >= 0.0:
        return None, None
    moment = _cross(position, velocity)
    if _dot(moment, moment) <= 0.0:
        return None, None
    ecc_vec = _cross(velocity, moment)
    ecc_vec = tuple(component / _MU_M3_S2 - position[index] / rmag for index, component in enumerate(ecc_vec))
    ecc = _norm(ecc_vec)
    if ecc >= 1.0:
        return None, None
    sma = -_MU_M3_S2 / (2.0 * energy)
    earth = _EARTH_KM * 1000.0
    return sma * (1.0 + ecc) - earth, sma * (1.0 - ecc) - earth


def format_apsis_label(apo_m, peri_m):
    """``256 km / 180 km``, or ``256 km / —`` when only apogee is real.

    Whole kilometres. Hidden when apogee is missing or below 50 km.
    Perigee below 80 km is a suborbital arc: the apogee still shows.
    """
    if apo_m is None:
        return ""
    try:
        apo_km = int(round(float(apo_m) / 1000.0))
    except (TypeError, ValueError):
        return ""
    if apo_km < 50:
        return ""
    peri_km = None
    if peri_m is not None:
        try:
            peri_km = int(round(float(peri_m) / 1000.0))
        except (TypeError, ValueError):
            peri_km = None
    if peri_km is None or peri_km < 80:
        return f"{apo_km} km / —"
    return f"{apo_km} km / {peri_km} km"


def _point_ecef_time(point):
    """ECEF and GPS seconds for a trail sample or a predicted point."""
    if not isinstance(point, dict):
        return None, None
    when = _num(point.get("gps_s") if point.get("gps_s") is not None else point.get("gps"))
    if when is None:
        when = _num(point.get("time") if point.get("time") is not None else point.get("t"))
    vector = _ecef_vector(point)
    if vector is None:
        vector = _ecef_tuple(point)
        if vector is not None and _norm(vector) <= 6_000_000.0:
            vector = None
    return vector, when


def _scale_velocity(velocity, speed):
    """Keep the chord's direction. Use the stored inertial speed for the magnitude.

    A one-sided position chord is a secant: over 60 s it leans inward by
    about 2°, which is ~270 m/s of fake radial velocity and moves apogee
    by a couple of hundred kilometres. A central difference is nearly
    tangent. Either way the stored inertial speed is the magnitude the
    strip already trusts.
    """
    if velocity is None:
        return None
    mag = _norm(velocity)
    if mag < 1.0:
        return None
    if speed is None or speed < 1000.0 or speed > 12000.0:
        return velocity
    scale = speed / mag
    return tuple(component * scale for component in velocity)


# How far back the apsis fit looks. Each step is a short inertial chord
# carried forward to the current fix. A longer single chord leans inward.
_APSIS_FIT_S = 900.0
# Displayed apogee/perigee may move this far per new fix. A bad chord
# used to jump the label by ~100 km between devices on the same pass.
_APSIS_STEP_M = 8_000.0


def _rotate_about(axis, vector, ang):
    """Rodrigues rotation of ``vector`` about ``axis`` by ``ang`` radians."""
    length = _norm(axis)
    if length < 1.0:
        return vector
    k = tuple(component / length for component in axis)
    c, s = math.cos(ang), math.sin(ang)
    crossed = _cross(k, vector)
    along = _dot(k, vector)
    return tuple(
        vector[i] * c + crossed[i] * s + k[i] * along * (1.0 - c)
        for i in range(3)
    )


def _velocity_at_fix(points):
    """Inertial velocity at t = 0 from short ECI chords.

    The chord between two fixes is the velocity at the midpoint of a
    circular arc, not at the latest fix. Each chord is spun forward by
    the angle the ship travelled after that midpoint, then the chords
    are averaged. One ECEF secant is what turned a 265 km circle into
    a 366/171 km ellipse.
    """
    if len(points) < 2:
        return None
    found = []
    for (t1, r1), (t2, r2) in zip(points, points[1:]):
        dt = t2 - t1
        if dt < 20.0 or dt > _APSIS_FIT_S:
            continue
        n1, n2 = _norm(r1), _norm(r2)
        if n1 < 1.0 or n2 < 1.0:
            continue
        cosang = max(-1.0, min(1.0, _dot(r1, r2) / (n1 * n2)))
        swept = math.acos(cosang)
        if swept < 1.0e-4:
            continue
        velocity = tuple((r2[i] - r1[i]) / dt for i in range(3))
        half = swept / 2.0
        stretch = half / math.sin(half)
        velocity = tuple(component * stretch for component in velocity)
        mid_r = tuple((r1[i] + r2[i]) / 2.0 for i in range(3))
        rate = swept / dt
        spun = _rotate_about(_cross(mid_r, velocity), velocity, rate * (0.0 - (t1 + t2) / 2.0))
        found.append(spun)
    if not found:
        return None
    count = float(len(found))
    return tuple(sum(item[axis] for item in found) / count for axis in range(3))


def fitted_inertial_velocity(slot, position, t0):
    """Inertial velocity at the current fix from the flown trail.

    Positions are rotated into the inertial frame of ``t0``
    (``r_eci = Rz(ω Δt) r_ecef``). The predicted path is left out: it
    is a straight ECEF chord, and devices with different trail gaps
    were turning that lean into different apogees.
    """
    cloud = []
    for sample in (slot or {}).get("samples") or []:
        point, when = _point_ecef_time(sample)
        if point is None or when is None:
            continue
        alt = _num(sample.get("alt_m") if "alt_m" in sample else sample.get("alt"))
        if alt is None or alt < ORBIT_COAST_ALT_M:
            continue
        dt = when - t0
        if dt < -_APSIS_FIT_S or dt > 20.0:
            continue
        cloud.append((dt, _rotate_z(point, EARTH_ROTATION_RAD_S * dt)))
    cloud.append((0.0, position))
    cloud.sort(key=lambda item: item[0])
    deduped = []
    for item in cloud:
        if deduped and abs(item[0] - deduped[-1][0]) < 1.0:
            deduped[-1] = item
            continue
        deduped.append(item)
    return _velocity_at_fix(deduped)


def _smooth_apsis(slot, t0, apo_m, peri_m):
    """Hold the label to an 8 km step per new fix. Same fix returns the cache."""
    if slot.get("_apsis_stamp") == t0 and slot.get("_apsis_label"):
        return slot["_apsis_label"]
    apo = apo_m
    peri = peri_m
    prev_a = _num(slot.get("_apsis_apo_m"))
    prev_p = _num(slot.get("_apsis_peri_m"))
    if prev_a is not None and apo is not None:
        apo = prev_a + max(-_APSIS_STEP_M, min(_APSIS_STEP_M, apo - prev_a))
    if prev_p is not None and peri is not None:
        peri = prev_p + max(-_APSIS_STEP_M, min(_APSIS_STEP_M, peri - prev_p))
    label = format_apsis_label(apo, peri)
    slot["_apsis_stamp"] = t0
    slot["_apsis_apo_m"] = apo
    slot["_apsis_peri_m"] = peri
    slot["_apsis_label"] = label
    return label


def ship_apsis_label(slot):
    """Stable osculating apogee / perigee for the ship's current state."""
    if not isinstance(slot, dict):
        return ""
    current = slot.get("current") or {}
    position, t0 = _point_ecef_time(current)
    if position is None or t0 is None:
        return ""
    if slot.get("_apsis_stamp") == t0 and slot.get("_apsis_label"):
        return slot["_apsis_label"]
    velocity = fitted_inertial_velocity(slot, position, t0)
    if velocity is None:
        return slot.get("_apsis_label") or ""
    speed = _norm(velocity)
    alt = _num(current.get("alt_m")) or 0.0
    if speed < 1500.0 and alt < 40_000.0:
        return ""
    # A fit that wandered off orbital speed is scaled back to the fix.
    # The direction stays the tangent from the window.
    reported = _num(current.get("speed_mps"))
    if reported and 7_000.0 <= reported <= 8_500.0 and not (7_000.0 <= speed <= 8_500.0):
        velocity = _scale_velocity(velocity, reported)
    apo, peri = osculating_apsides_m(position, velocity)
    if apo is None:
        return slot.get("_apsis_label") or ""
    return _smooth_apsis(slot, t0, apo, peri)


def _ecef_tuple(raw):
    if not isinstance(raw, dict):
        return None
    coords = raw.get("r_ecef")
    if isinstance(coords, (list, tuple)) and len(coords) >= 3:
        x, y, z = _num(coords[0]), _num(coords[1]), _num(coords[2])
        if None not in (x, y, z):
            return x, y, z
    lat = _num(raw.get("latitude") if "latitude" in raw else raw.get("lat"))
    lon = _num(raw.get("longitude") if "longitude" in raw else raw.get("lon"))
    alt = _num(raw.get("altitude") if "altitude" in raw else raw.get("alt"))
    if None in (lat, lon, alt):
        return None
    x, y, z = geodetic_to_ecef(lat, lon, alt)
    return x, y, z


def _fix_time(raw):
    if not isinstance(raw, dict):
        return None
    if "time" in raw:
        return _num(raw.get("time"))
    return _num(raw.get("gps_time") if "gps_time" in raw else raw.get("gps"))


def _ecef_speed_mps(earlier, later, max_dt=120.0):
    """Ground-relative speed (m/s) from two ECEF fixes. No Earth rotation."""
    gps0 = _num(earlier.get("gps") if isinstance(earlier, dict) else None)
    gps1 = _num(later.get("gps") if isinstance(later, dict) else None)
    if gps0 is None or gps1 is None:
        return None
    dt = gps1 - gps0
    if dt < 1.0 or dt > max_dt:
        return None
    x0, y0, z0 = (_num(earlier.get("ex")), _num(earlier.get("ey")), _num(earlier.get("ez")))
    x1, y1, z1 = (_num(later.get("ex")), _num(later.get("ey")), _num(later.get("ez")))
    if None in (x0, y0, z0, x1, y1, z1):
        return None
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) / dt


def _speed_sample(t, r):
    return {"gps": t, "ex": r[0], "ey": r[1], "ez": r[2]}


def _usable_speed_pair(t0, r0, t1, r1, alt0=None, alt1=None):
    """Inertial speed from two fixes, or ground-relative speed when parked.

    After Flight 15 landed, the hobby predicted path dove underground and
    the 60 s pair on that path reconstructed 3,529 km/h. A parked ship also
    has inertial speed equal to Earth rotation (~1,500 km/h) when ECEF is
    still. Skip underground points. If the vehicle is barely moving relative
    to the ground, report that Earth-relative speed (≈0), not ω × r.
    """
    if alt0 is not None and alt0 < PREDICTED_MIN_ALT_M:
        return None
    if alt1 is not None and alt1 < PREDICTED_MIN_ALT_M:
        return None
    earlier = _speed_sample(t0, r0)
    later = _speed_sample(t1, r1)
    ecef_speed = _ecef_speed_mps(earlier, later)
    if ecef_speed is not None and ecef_speed <= PAD_ECEF_SPEED_MPS:
        return ecef_speed
    return inertial_speed_mps(earlier, later)


def _fix_altitude(raw):
    if not isinstance(raw, dict):
        return None
    return _num(raw.get("altitude") if "altitude" in raw else raw.get("alt"))


def fill_inertial_speeds(body):
    """Set current.speed from ECEF deltas when the feed left speed at 0.

    Beyond Stage Zero and StarDash both republish the tracker with speed 0.
    The next predicted point is about 60 s ahead and carries r_ecef (or
    lat/lon/alt, which we convert). A speed the feed already filled in is kept.
    Predicted points that go underground are ignored, and a parked vehicle
    does not inherit Earth rotation as airspeed.
    """
    if not isinstance(body, dict):
        return body
    for key, raw in body.items():
        if key == "metadata" or not isinstance(raw, dict):
            continue
        current = raw.get("current")
        if not isinstance(current, dict):
            continue
        reported = _num(current.get("speed"))
        if reported is not None and reported > 1.0:
            continue
        fixes = []
        current_ecef = _ecef_tuple(current)
        current_time = _fix_time(current)
        if current_ecef is not None and current_time is not None:
            fixes.append((current_time, current_ecef, _fix_altitude(current)))
        for item in raw.get("trajectory") or []:
            ecef = _ecef_tuple(item)
            when = _fix_time(item)
            if ecef is None or when is None:
                continue
            alt = _fix_altitude(item)
            if alt is not None and alt < PREDICTED_MIN_ALT_M:
                continue
            fixes.append((when, ecef, alt))
        fixes.sort(key=lambda item: item[0])
        # The current fix is repeated as trajectory[0] a fraction of a second
        # off. Prefer the sample-period step (~60 s) over that near-duplicate.
        ranked = []
        for (t0, r0, alt0), (t1, r1, alt1) in zip(fixes, fixes[1:]):
            dt = t1 - t0
            speed = _usable_speed_pair(t0, r0, t1, r1, alt0, alt1)
            if speed is None:
                continue
            ranked.append((abs(dt - 60.0), speed))
        if ranked:
            ranked.sort(key=lambda item: item[0])
            current["speed"] = ranked[0][1]
    return body


def generation_age_s(body, now):
    """Age of metadata.generation_time. None when the document has no clock."""
    if not isinstance(body, dict):
        return None
    meta = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    gps = _num(meta.get("generation_time"))
    if gps is None:
        return None
    return (_aware(now) - gps_to_utc(gps)).total_seconds()


def accept_hobby_document(body, now):
    """Copy of a SpaceX-shaped hobby document, or None if it is not live.

    Speed is filled from ECEF when the feed left it at 0. A sample several
    minutes old is dropped. A sample newer than that is stamped as fresh so
    the 60 s parser stale check does not hide a 90 s old but still-valid fix.
    The vehicle keeps its own GPS time.
    """
    if not isinstance(body, dict) or not body:
        return None
    import copy
    cloned = copy.deepcopy(body)
    age = generation_age_s(cloned, now)
    if age is None or age > HOBBY_STALE_S or age < -120:
        return None
    fill_inertial_speeds(cloned)
    if not feed_has_vehicle(cloned):
        return None
    meta = cloned.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
        cloned["metadata"] = meta
    meta["generation_time"] = utc_to_gps(_aware(now))
    return cloned


def stardash_feed_to_tracker(payload, now):
    """StarDash /api/feed → a SpaceX-shaped document, before the stale check.

    ``traj`` entries are ``[lat, lon, alt, time]``. None when there is no sample.
    """
    if not isinstance(payload, dict):
        return None
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        return None
    status = payload.get("status") if isinstance(payload.get("status"), dict) else {}
    body = {
        "metadata": {
            "coordinate_frame": "ECEF",
            "source": "stardash",
            "trajectory_version": status.get("trajectory_version"),
            "sample_period": 60,
        }
    }
    newest_gps = None
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        veh = str(sample.get("veh") or "").strip()
        if not veh or veh == "metadata":
            continue
        gps = _num(sample.get("gps"))
        lat = _num(sample.get("lat"))
        lon = _num(sample.get("lon"))
        alt = _num(sample.get("alt"))
        if None in (gps, lat, lon, alt):
            continue
        current = {
            "gps_time": gps,
            "mission_time": _num(sample.get("met")),
            "altitude": alt,
            "speed": _num(sample.get("spd")) or 0.0,
            "latitude": lat,
            "longitude": lon,
            "r_ecef": None,
        }
        ecef = sample.get("ecef")
        if isinstance(ecef, (list, tuple)) and len(ecef) >= 3:
            current["r_ecef"] = [ecef[0], ecef[1], ecef[2]]
        else:
            current["r_ecef"] = geodetic_to_ecef(lat, lon, alt)
        trajectory = []
        for item in sample.get("traj") or []:
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                continue
            tlat, tlon, talt, ttime = (
                _num(item[0]), _num(item[1]), _num(item[2]), _num(item[3])
            )
            if None in (tlat, tlon, talt, ttime):
                continue
            trajectory.append({
                "latitude": tlat,
                "longitude": tlon,
                "altitude": talt,
                "time": ttime,
                "r_ecef": geodetic_to_ecef(tlat, tlon, talt),
            })
        body[veh] = {"current": current, "trajectory": trajectory}
        if newest_gps is None or gps > newest_gps:
            newest_gps = gps
    if newest_gps is None:
        return None
    gen = _num(status.get("generation_time")) or newest_gps
    body["metadata"]["generation_time"] = gen
    return body


def stardash_history_url(now):
    """Current minute bucket. That response is the pad-to-now archive."""
    moment = _aware(now) or datetime.now(timezone.utc)
    bucket = int(moment.timestamp() * 1000) // 60_000
    return f"{STARDASH_HISTORY_URL}?w={bucket}"


def _row_lat_lon_alt(row):
    lat = _num(row.get("lat") if "lat" in row else row.get("latitude"))
    lon = _num(row.get("lon") if "lon" in row else row.get("longitude"))
    alt = _num(row.get("alt") if "alt" in row else row.get("altitude"))
    return lat, lon, alt


def _row_ecef(row, lat, lon, alt):
    ex, ey, ez = _num(row.get("ex")), _num(row.get("ey")), _num(row.get("ez"))
    if None not in (ex, ey, ez):
        return ex, ey, ez
    coords = row.get("r_ecef") or row.get("ecef")
    if isinstance(coords, (list, tuple)) and len(coords) >= 3:
        ex, ey, ez = _num(coords[0]), _num(coords[1]), _num(coords[2])
        if None not in (ex, ey, ez):
            return ex, ey, ez
    if None in (lat, lon, alt):
        return None
    return tuple(geodetic_to_ecef(lat, lon, alt))


def rows_to_trail(rows, max_dt=BACKFILL_MAX_DT_S):
    """History rows → trail samples.

    Drops underground and off-world points, derives inertial speed, and keeps
    a point only when it is the pad start or the vehicle has actually moved.
    A jump faster than the predicted-path cap is left out.
    """
    cleaned = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        gps = _num(row.get("gps") if "gps" in row else row.get("gps_time") or row.get("time"))
        lat, lon, alt = _row_lat_lon_alt(row)
        if None in (gps, lat, lon, alt):
            continue
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue
        if alt < PREDICTED_MIN_ALT_M:
            continue
        ecef = _row_ecef(row, lat, lon, alt)
        if ecef is None:
            continue
        cleaned.append({
            "gps": gps,
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "ecef": ecef,
            "met": _num(row.get("met") if "met" in row else row.get("mission_time")),
        })
    cleaned.sort(key=lambda item: item["gps"])
    deduped = []
    for item in cleaned:
        if deduped and item["gps"] - deduped[-1]["gps"] < 1.0:
            continue
        deduped.append(item)
    # Speed and the jump cap use the previous archive point, including pad
    # samples that are not themselves drawn. Anchoring those checks on the
    # last kept point made the whole coast look like one teleport off the pad.
    speeds = [0.0]
    usable = [True]
    for previous, item in zip(deduped, deduped[1:]):
        dt = item["gps"] - previous["gps"]
        dist_km = haversine_km(previous["lat"], previous["lon"], item["lat"], item["lon"])
        minutes = max(dt, 1.0) / 60.0
        if dt <= 0 or dt > max_dt or dist_km / minutes > PREDICTED_MAX_KM_PER_MIN_CAP:
            speeds.append(0.0)
            usable.append(False)
            continue
        derived = inertial_speed_mps(
            {"gps": previous["gps"], "ex": previous["ecef"][0], "ey": previous["ecef"][1], "ez": previous["ecef"][2]},
            {"gps": item["gps"], "ex": item["ecef"][0], "ey": item["ecef"][1], "ez": item["ecef"][2]},
            max_dt=max_dt,
        )
        speeds.append(derived if derived is not None else 0.0)
        usable.append(True)
    trail = []
    for index, item in enumerate(deduped):
        if not usable[index]:
            continue
        speed = speeds[index]
        previous = deduped[index - 1] if index else None
        record = {
            "lat": item["lat"],
            "lon": item["lon"],
            "alt_m": item["alt"],
            "speed_mps": speed,
            "mission_time_s": item["met"],
            "utc": gps_to_utc(item["gps"]).isoformat(),
            "gps": item["gps"],
            "fresh": True,
        }
        if previous is None:
            trail.append(record)
            continue
        moved = (
            speed > MOVING_SPEED_MPS
            or abs(item["alt"] - previous["alt"]) >= ALTITUDE_CHANGE_M
            or haversine_km(previous["lat"], previous["lon"], item["lat"], item["lon"]) >= 1.0
        )
        stepped = (
            haversine_km(previous["lat"], previous["lon"], item["lat"], item["lon"]) >= TRAIL_MIN_STEP_KM
            or abs(item["alt"] - previous["alt"]) >= 100.0
        )
        if stepped and moved:
            trail.append(record)
    return trail


def _history_groups(rows, default_key="ship42"):
    grouped = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        veh = str(row.get("veh") or row.get("key") or default_key).strip()
        if not veh or veh == "metadata":
            continue
        grouped.setdefault(veh, []).append(row)
    out = {}
    for veh, items in grouped.items():
        trail = rows_to_trail(items)
        if len(trail) >= 2:
            out[veh] = trail
    return out or None


def parse_stardash_history(payload):
    """StarDash ``/api/history`` samples, or None when the chunk is unusable."""
    if not isinstance(payload, dict) or payload.get("history_ok") is False:
        return None
    if payload.get("error"):
        return None
    samples = payload.get("samples")
    if not isinstance(samples, list):
        return None
    return _history_groups(samples)


def parse_bsz_ship_trail(payload):
    """Beyond Stage Zero ship trail, if the proxy actually returns points."""
    rows = None
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and not payload.get("error"):
        for key in ("points", "trail", "samples", "positions"):
            if isinstance(payload.get(key), list):
                rows = payload[key]
                break
    if not rows:
        return None
    return _history_groups(rows)


def fetch_flight_history(now=None, get=None, timeout=BACKFILL_TIMEOUT_S):
    """Past positions for the current flight. None when both sources fail.

    Beyond Stage Zero's trail is tried first. A 404 or an empty body falls
    through to StarDash's history bucket. Network errors stay in the caller.
    """
    getter = get or requests.get
    moment = _aware(now) or datetime.now(timezone.utc)
    headers = {"User-Agent": tracker_user_agent(), "Accept": "application/json"}
    for url, parser in (
        (BSZ_SHIP_TRAIL_URL, parse_bsz_ship_trail),
        (stardash_history_url(moment), parse_stardash_history),
    ):
        try:
            response = getter(url, headers=headers, timeout=timeout)
        except Exception as exc:
            logger.debug("Trail backfill %s failed: %s", url, exc)
            continue
        status = getattr(response, "status_code", 200)
        if status != 200:
            continue
        try:
            payload = response.json()
        except Exception as exc:
            logger.debug("Trail backfill %s was not JSON: %s", url, exc)
            continue
        grouped = parser(payload)
        if grouped:
            return grouped
    return None


def trail_epoch(sample):
    """Unix seconds for a trail sample, from GPS time or the stored UTC."""
    if not isinstance(sample, dict):
        return None
    gps = _num(sample.get("gps"))
    if gps is not None and gps > 1_000_000:
        return gps_to_utc(gps).timestamp()
    return None if _stamp(sample.get("utc")) is None else _stamp(sample.get("utc")).timestamp()


def ship_trail_is_partial(state, liftoff, now, gap_s=BACKFILL_GAP_S):
    """True when the ship trail does not reach back near liftoff."""
    if liftoff is None or state is None:
        return False
    slot = None
    for key, item in (getattr(state, "vehicles", None) or {}).items():
        if not isinstance(item, dict):
            continue
        if item.get("kind") == "ship" or str(key).lower().startswith("ship"):
            slot = item
            break
    samples = (slot or {}).get("samples") or []
    if not samples:
        return True
    epochs = [trail_epoch(sample) for sample in samples]
    epochs = [item for item in epochs if item is not None]
    if not epochs:
        return True
    return min(epochs) > _aware(liftoff).timestamp() + gap_s


def merge_trail_samples(local, incoming):
    """Union of two trails, dropping a point within ``TRAIL_DEDUPE_S`` of one kept.

    A local sample wins when it lands on the same instant as a backfilled one.
    The result is thinned with the same cap the live ingest uses.
    """
    tagged = [("local", sample) for sample in (local or []) if isinstance(sample, dict)]
    tagged += [("backfill", sample) for sample in (incoming or []) if isinstance(sample, dict)]
    tagged = [(source, sample) for source, sample in tagged if trail_epoch(sample) is not None]
    tagged.sort(key=lambda item: (trail_epoch(item[1]), 0 if item[0] == "local" else 1))
    kept = []
    for source, sample in tagged:
        if kept and abs(trail_epoch(sample) - trail_epoch(kept[-1])) < TRAIL_DEDUPE_S:
            if source == "local":
                kept[-1] = sample
            continue
        kept.append(sample)
    return _downsample_trail(kept, MAX_TRAIL_POINTS)


def feed_has_vehicle(body):
    """True when a tracker document contains at least one plottable fix."""
    if not isinstance(body, dict) or not body:
        return False
    return bool(parse_feed(body).get("vehicles"))


def parse_feed(body, now_utc=None):
    """Turn one tracker document into vehicles plus a staleness flag."""
    now = now_utc or datetime.now(timezone.utc)
    if not isinstance(body, dict):
        return {"stale": True, "generation_utc": None, "vehicles": []}
    meta = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    generated = _num(meta.get("generation_time"))
    generation_utc = gps_to_utc(generated) if generated is not None else None
    stale = True
    if generation_utc is not None:
        stale = (now - generation_utc).total_seconds() > STALE_AFTER_S
    vehicles = []
    for key, raw in body.items():
        if key == "metadata" or not isinstance(raw, dict):
            continue
        current = raw.get("current") if isinstance(raw.get("current"), dict) else {}
        point = _point(current)
        if point is None:
            continue
        kind = classify_vehicle(key)
        predicted = []
        for item in raw.get("trajectory") or []:
            parsed = _point(item)
            if parsed is not None:
                predicted.append(parsed)
        mission = _num(current.get("mission_time"))
        speed = _num(current.get("speed")) or 0.0
        sane = predicted_is_sane(predicted, speed)
        vehicles.append({
            "key": str(key),
            "kind": kind,
            "name": display_name(key, kind),
            "lat": point["lat"],
            "lon": point["lon"],
            "alt_m": point["alt_m"],
            "speed_mps": speed,
            "mission_time_s": mission,
            "gps_s": point["time"],
            "gps_utc": gps_to_utc(point["time"]) if point["time"] is not None else None,
            "predicted": predicted if sane else [],
            "predicted_sane": sane,
        })
    return {
        "stale": stale,
        "generation_utc": generation_utc,
        "vehicles": vehicles,
    }


def _is_moving(sample, previous):
    if sample["speed_mps"] > MOVING_SPEED_MPS:
        return True
    if previous is None:
        return False
    if abs(sample["alt_m"] - previous["alt_m"]) >= ALTITUDE_CHANGE_M:
        return True
    return haversine_km(
        previous["lat"], previous["lon"], sample["lat"], sample["lon"]
    ) >= 1.0


def _aware(moment):
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment


def _counts_for_tracker(launch):
    """Starship flights, plus sparse test records that only carry a NET."""
    if not isinstance(launch, dict):
        return False
    if launch.get("rocket") or launch.get("mission") or launch.get("name"):
        from live_window import is_starship_launch

        return is_starship_launch(launch)
    return True


def _relevant_nets(launch_data):
    if not launch_data:
        return []
    nets = []
    launches = list(launch_data.get("upcoming") or []) + list(launch_data.get("previous") or [])
    for launch in launches:
        if not _counts_for_tracker(launch):
            continue
        net = _get_parsed_dt(launch.get("net"))
        if net:
            nets.append(_aware(net))
    return nets


def pre_launch_window(launch_data, now_utc):
    """True from T-30min until NET. The NET list is whatever was just passed in."""
    now = _aware(now_utc)
    for net in _relevant_nets(launch_data):
        delta = (now - net).total_seconds()
        if -WINDOW_BEFORE_S <= delta < 0:
            return True
    return False


def _sample_age_s(gps_s, now_utc):
    """Seconds the fix is already old. Capped so a dead feed cannot run away."""
    gps = _num(gps_s)
    now = _aware(now_utc)
    if gps is None or now is None:
        return 0.0
    age = (now - gps_to_utc(gps)).total_seconds()
    if age < 0.0:
        return 0.0
    if age > 180.0:
        return 180.0
    return age


def starship_flight_label(launch_data, now_utc):
    """``Flight 15`` for the Starship that owns the tracker window, or ''."""
    now = _aware(now_utc)
    if now is None or not isinstance(launch_data, dict):
        return ""
    target = liftoff_net(launch_data, now)
    chosen = None
    chosen_net = None
    launches = list(launch_data.get("upcoming") or []) + list(launch_data.get("previous") or [])
    for launch in launches:
        if not isinstance(launch, dict) or not _counts_for_tracker(launch):
            continue
        net = _aware(_get_parsed_dt(launch.get("net")))
        if net is None:
            continue
        if target is not None:
            if abs((net - target).total_seconds()) < 1.0:
                chosen = launch
                break
            continue
        delta = (now - net).total_seconds()
        if -WINDOW_BEFORE_S <= delta < 0 and (chosen_net is None or net < chosen_net):
            chosen = launch
            chosen_net = net
    if chosen is None:
        return ""
    from nsf_stream import launch_flight_number

    number = launch_flight_number(chosen)
    if not number:
        return ""
    return f"Flight {number}"


def liftoff_net(launch_data, now_utc):
    """Most recent relevant NET that has passed, within the 48h safety cap."""
    now = _aware(now_utc)
    best = None
    for net in _relevant_nets(launch_data):
        delta = (now - net).total_seconds()
        if 0 <= delta <= MISSION_CAP_S and (best is None or net > best):
            best = net
    return best


def launch_window_active(launch_data, now_utc):
    """Backward-compatible name: the pre-launch T-30min window only."""
    return pre_launch_window(launch_data, now_utc)


def tracker_user_agent():
    """Identify the kiosk without a new request on every poll."""
    return f"spacex-dashboard/{_tracker_version()} (+{_UA_REPO})"


def _tracker_version():
    cached = getattr(_tracker_version, "value", None)
    if cached:
        return cached
    version = "dev"
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=1,
            cwd=os.path.dirname(__file__),
            check=False,
        )
        short = (res.stdout or "").strip()
        if res.returncode == 0 and short:
            version = short
    except (OSError, subprocess.SubprocessError):
        version = "dev"
    _tracker_version.value = version
    return version


def parse_retry_after(value, now=None):
    """Retry-After as delta-seconds or an HTTP-date. None if it cannot be read."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        seconds = None
    if seconds is not None:
        if seconds < 0:
            return None
        return seconds
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)
    return max(0.0, (when - clock).total_seconds())


def http_backoff_s(status, retry_after=None, previous=0, now=None):
    """Wait after 429/503 (Retry-After, else exponential) and after 403.

    Other status codes return None so the caller keeps its generic backoff.
    The wait is capped at the idle interval (5 minutes).
    """
    try:
        code = int(status)
    except (TypeError, ValueError):
        return None
    if code == 403:
        return IDLE_POLL_S
    if code not in (429, 503):
        return None
    parsed = parse_retry_after(retry_after, now=now)
    if parsed is not None:
        return int(min(IDLE_POLL_S, max(MIN_POLL_S, math.ceil(parsed))))
    base = previous or ACTIVE_POLL_S
    return int(min(IDLE_POLL_S, max(20, base * 2)))


def jitter_interval(base_s, rng=None):
    """Spread the 10s and 5min cadences so devices do not poll together."""
    roll = rng or random
    if base_s == ACTIVE_POLL_S:
        delta = roll.uniform(-ACTIVE_JITTER_S, ACTIVE_JITTER_S)
    elif base_s == IDLE_POLL_S:
        delta = roll.uniform(-IDLE_JITTER_S, IDLE_JITTER_S)
    else:
        return base_s
    return max(float(MIN_POLL_S), base_s + delta)


def poll_interval_s(*, in_window, feed_hot, backoff_s=0, rng=None):
    if backoff_s and backoff_s > 0:
        base = min(IDLE_POLL_S, int(backoff_s))
    elif in_window or feed_hot:
        base = ACTIVE_POLL_S
    else:
        base = IDLE_POLL_S
    if base in (ACTIVE_POLL_S, IDLE_POLL_S):
        interval = jitter_interval(base, rng)
    else:
        interval = base
    return max(float(MIN_POLL_S), interval)


class TrackerHostState:
    """Sticky host plus a separate ETag and Last-Modified for each one."""

    def __init__(self):
        self.preferred = PRIMARY_TRACKER_URL
        self.prefer_fallback_until = None
        self.validators = {}
        self.source = None
        # Hobby mirrors are not preferred hosts. One gate covers both.
        self.hobby_not_before = None
        self.hobby_backoff_s = 0
        self.hobby_have_fix = False
        self.hobby_url = None

    def headers_for(self, url):
        saved = self.validators.get(url) or {}
        return saved.get("etag"), saved.get("modified")

    def remember(self, url, etag, modified):
        self.validators[url] = {"etag": etag, "modified": modified}

    def order(self, now):
        until = self.prefer_fallback_until
        if (
            self.preferred == FALLBACK_TRACKER_URL
            and until is not None
            and now < until
        ):
            return [FALLBACK_TRACKER_URL]
        if self.preferred == FALLBACK_TRACKER_URL:
            return [PRIMARY_TRACKER_URL, FALLBACK_TRACKER_URL]
        return [PRIMARY_TRACKER_URL, FALLBACK_TRACKER_URL]

    def note_success(self, url, now):
        self.preferred = url
        if url == PRIMARY_TRACKER_URL:
            self.prefer_fallback_until = None
        else:
            self.prefer_fallback_until = now + timedelta(seconds=PRIMARY_RETRY_S)

    def note_primary_failed(self, now):
        """Stay on the fallback and wait before probing the primary again."""
        if self.preferred == FALLBACK_TRACKER_URL:
            self.prefer_fallback_until = now + timedelta(seconds=PRIMARY_RETRY_S)

    def note_source(self, url):
        if url and url != self.source:
            logger.info("Starship tracker source %s", url)
        self.source = url

    def hobby_waiting(self, now):
        until = self.hobby_not_before
        return until is not None and now < until

    def note_hobby_success(self, now, url):
        self.hobby_backoff_s = 0
        self.hobby_have_fix = True
        self.hobby_url = url
        self.hobby_not_before = now + timedelta(seconds=HOBBY_POLL_S)

    def note_hobby_failure(self, now, retry_after=None):
        parsed = parse_retry_after(retry_after, now=now) if retry_after else None
        if parsed is not None:
            wait = min(float(IDLE_POLL_S), max(float(HOBBY_POLL_S), parsed))
        elif self.hobby_backoff_s:
            wait = min(float(IDLE_POLL_S), max(float(HOBBY_POLL_S), self.hobby_backoff_s * 2))
        else:
            wait = float(HOBBY_POLL_S)
        self.hobby_backoff_s = wait
        self.hobby_not_before = now + timedelta(seconds=wait)


def fetch_public_tracker(etag=None, modified=None, timeout=FETCH_TIMEOUT_S, url=TRACKER_URL, get=None):
    """GET the public JSON. 304 comes back as not_modified.

    403, 429, and 503 do not raise. They return ``backoff_s`` so the poller
    can wait. Conditional headers are sent whenever we already have them.
    """
    headers = {"User-Agent": tracker_user_agent(), "Accept": "application/json"}
    if etag:
        headers["If-None-Match"] = etag
    if modified:
        headers["If-Modified-Since"] = modified
    getter = get or requests.get
    response = getter(url, headers=headers, timeout=timeout)
    if response.status_code == 304:
        return {"not_modified": True, "etag": etag, "modified": modified}
    if response.status_code in (403, 429, 503):
        return {
            "error": f"HTTP {response.status_code}",
            "status": response.status_code,
            "retry_after": response.headers.get("Retry-After"),
            "etag": etag,
            "modified": modified,
        }
    response.raise_for_status()
    return {
        "not_modified": False,
        "body": response.json(),
        "etag": response.headers.get("ETag") or etag,
        "modified": response.headers.get("Last-Modified") or modified,
        "url": url,
    }


def _hobby_document(url, payload, now):
    if url == BEYOND_STAGE_ZERO_URL:
        return accept_hobby_document(payload, now)
    if url == STARDASH_FEED_URL:
        return accept_hobby_document(stardash_feed_to_tracker(payload, now), now)
    return None


def _fetch_hobby(holder, now, get, timeout):
    """Beyond Stage Zero, then StarDash /api/feed. At most one try per 30 s.

    None when the gate is closed and there is no fix to keep. A failure of
    both backs off. Once a fix has been shown, later failures return
    not_modified so the SpaceX poll is not slowed down.
    """
    if holder.hobby_waiting(now):
        if holder.hobby_have_fix:
            return {"not_modified": True, "url": holder.hobby_url or BEYOND_STAGE_ZERO_URL}
        return None
    headers = {"User-Agent": tracker_user_agent(), "Accept": "application/json"}
    last = None
    saw_rate_limit = False
    for url in (BEYOND_STAGE_ZERO_URL, STARDASH_FEED_URL):
        try:
            response = get(url, headers=headers, timeout=timeout)
        except Exception as exc:
            logger.debug("Hobby tracker %s failed: %s", url, exc)
            last = {"error": str(exc), "url": url}
            continue
        status = getattr(response, "status_code", 200)
        if status in (403, 429, 503):
            retry_after = response.headers.get("Retry-After") if getattr(response, "headers", None) else None
            last = {
                "error": f"HTTP {status}",
                "status": status,
                "retry_after": retry_after,
                "url": url,
            }
            saw_rate_limit = True
            continue
        if status != 200:
            last = {"error": f"HTTP {status}", "status": status, "url": url}
            continue
        try:
            payload = response.json()
        except Exception as exc:
            logger.debug("Hobby tracker %s was not JSON: %s", url, exc)
            last = {"error": "hobby json", "url": url}
            continue
        body = _hobby_document(url, payload, now)
        if body is None:
            last = {"error": "hobby stale or empty", "url": url}
            continue
        holder.note_hobby_success(now, url)
        holder.note_source(url)
        return {"not_modified": False, "body": body, "url": url}
    retry_after = last.get("retry_after") if saw_rate_limit and last else None
    holder.note_hobby_failure(now, retry_after=retry_after)
    if holder.hobby_have_fix:
        return {"not_modified": True, "url": holder.hobby_url or BEYOND_STAGE_ZERO_URL}
    return last or {"error": "hobby tracker failed", "url": BEYOND_STAGE_ZERO_URL}


def fetch_with_failover(state, now=None, timeout=FETCH_TIMEOUT_S, get=None, allow_alternate=False):
    """SpaceX, then the Azure mirror, then hobby mirrors if a mission is up.

    A 304 stays on that SpaceX host. An empty document is not a success, so
    ``{}`` falls through. The mirror sticks for about 10 minutes, then the
    primary is tried again. Hobby sources are never the preferred host: the
    next poll tries SpaceX first and switches back as soon as it has a fix.
    Order after that is Beyond Stage Zero, then StarDash's live feed.
    ``allow_alternate`` is the active mission window (NET already passed).
    """
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)
    holder = state or TrackerHostState()
    getter = get or requests.get
    last = None
    unchanged = None
    saw_primary_failure = False
    for url in holder.order(clock):
        etag, modified = holder.headers_for(url)
        try:
            result = fetch_public_tracker(
                etag=etag, modified=modified, timeout=timeout, url=url, get=getter
            )
        except Exception as exc:
            last = {"error": str(exc), "url": url}
            if url == PRIMARY_TRACKER_URL:
                saw_primary_failure = True
            continue
        if result.get("not_modified"):
            holder.remember(url, result.get("etag"), result.get("modified"))
            holder.note_success(url, clock)
            holder.note_source(url)
            result["url"] = url
            # Outside a mission a 304 is the answer. After NET, a stuck
            # SpaceX document must not hide the hobby position feeds: a
            # device that dropped its latch never confirms the flight.
            if not allow_alternate:
                return result
            if unchanged is None:
                unchanged = result
            continue
        body = result.get("body")
        if body is not None and feed_has_vehicle(body):
            holder.remember(url, result.get("etag"), result.get("modified"))
            holder.note_success(url, clock)
            holder.note_source(url)
            result["url"] = url
            return result
        if result.get("status") in (403, 429, 503):
            last = dict(result)
            last["url"] = url
        else:
            last = {"error": "tracker empty", "url": url, "empty": True}
        if url == PRIMARY_TRACKER_URL:
            saw_primary_failure = True
    if saw_primary_failure:
        holder.note_primary_failed(clock)
    if allow_alternate:
        hobby = _fetch_hobby(holder, clock, getter, HOBBY_TIMEOUT_S)
        if hobby is not None and hobby.get("body"):
            return hobby
        if unchanged is not None:
            return unchanged
        if hobby is not None:
            return hobby
    if last and last.get("status") in (403, 429, 503):
        return last
    if last and last.get("error") and not last.get("empty"):
        return last
    # Nothing plottable. Do not ingest an empty document (that would mark
    # every vehicle missing) and do not back off the mission poll.
    return {"not_modified": True, "empty": True, "url": (last or {}).get("url") or PRIMARY_TRACKER_URL}


def _downsample_trail(samples, cap):
    """Thin older points so a long flight keeps its shape instead of losing the start."""
    if cap < 2 or len(samples) <= cap:
        return list(samples)
    tail_n = max(2, cap // 3)
    tail = list(samples[-tail_n:])
    head = list(samples[:-tail_n])
    budget = cap - len(tail)
    if len(head) <= budget:
        return head + tail
    if budget <= 1:
        return [head[0]] + tail
    last_index = len(head) - 1
    picked = []
    for step in range(budget):
        idx = round(step * last_index / (budget - 1))
        picked.append(head[idx])
    return picked + tail


def _stamp(value):
    if not value or not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    return _aware(moment)


def _is_grounded(current):
    try:
        speed = float(current.get("speed_mps") or 0)
        alt = float(current.get("alt_m"))
    except (TypeError, ValueError):
        return False
    return speed <= GROUNDED_SPEED_MPS and alt <= GROUNDED_ALT_M


def _is_clearly_airborne(current):
    """Off the pad in a way a GPS glitch or a leftover predicted speed is not."""
    if not current:
        return False
    try:
        alt = float(current.get("alt_m") or 0)
        speed = float(current.get("speed_mps") or 0)
    except (TypeError, ValueError):
        return False
    return alt > AIRBORNE_ALT_M or speed > AIRBORNE_SPEED_MPS


def _launch_status_text(launch):
    if not isinstance(launch, dict):
        return ""
    status = launch.get("status")
    if isinstance(status, dict):
        parts = [status.get("name"), status.get("abbrev"), status.get("description")]
        return " ".join(str(part) for part in parts if part)
    return str(status or "")


def _is_terminal_launch_status(text):
    """True for a completed attempt. 'Launch Successful' is not 'still flying'."""
    blob = " ".join(str(text or "").lower().split())
    if not blob:
        return False
    tokens = (
        "launch successful",
        "successful",
        "success",
        "landed",
        "caught",
        "catch",
        "failure",
        "failed",
        "partial failure",
        "abort",
        "aborted",
        "expended",
    )
    return any(token in blob for token in tokens)


def window_starship_launch(launch_data, now_utc):
    """The Starship row that owns the current tracker window, or None."""
    now = _aware(now_utc)
    if now is None or not isinstance(launch_data, dict):
        return None
    target = liftoff_net(launch_data, now)
    chosen = None
    chosen_net = None
    launches = list(launch_data.get("upcoming") or []) + list(launch_data.get("previous") or [])
    for launch in launches:
        if not isinstance(launch, dict) or not _counts_for_tracker(launch):
            continue
        net = _aware(_get_parsed_dt(launch.get("net")))
        if net is None:
            continue
        if target is not None:
            if abs((net - target).total_seconds()) < 1.0:
                return launch
            continue
        delta = (now - net).total_seconds()
        if -WINDOW_BEFORE_S <= delta < 0 and (chosen_net is None or net < chosen_net):
            chosen = launch
            chosen_net = net
    return chosen


def _sample_record(vehicle, seen_utc, fresh=True):
    return {
        "lat": vehicle["lat"],
        "lon": vehicle["lon"],
        "alt_m": vehicle["alt_m"],
        "speed_mps": vehicle["speed_mps"],
        "mission_time_s": vehicle["mission_time_s"],
        "gps_s": vehicle.get("gps_s"),
        "utc": seen_utc.isoformat(),
        "fresh": bool(fresh),
    }


class TrackerState:
    """In-memory trail plus the on-disk copy that survives a mid-flight restart."""

    def __init__(self, path=None, persist=True):
        self.path = path or LOG_FILE
        # Flight test mode runs with persist=False so nothing reaches disk.
        self.persist = bool(persist)
        self.vehicles = {}
        self.hold_until = None
        self.generation_utc = None
        self.feed_stale = True
        self.window_baseline = None
        self.generation_advanced = False
        self.liftoff_utc = None
        self.liftoff_from_motion = False
        self.hold_started = False
        self.cap_logged = False
        self.pre_window = False
        self.net_passed = False
        self.flight_label = ""

    @classmethod
    def load(cls, path=LOG_FILE):
        state = cls()
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return state
        if not isinstance(payload, dict):
            return state
        hold = payload.get("hold_until")
        if isinstance(hold, str):
            try:
                state.hold_until = datetime.fromisoformat(hold)
            except ValueError:
                state.hold_until = None
        vehicles = payload.get("vehicles")
        if isinstance(vehicles, dict):
            state.vehicles = vehicles
        baseline = payload.get("window_baseline")
        if isinstance(baseline, dict):
            state.window_baseline = baseline
        state.generation_advanced = bool(payload.get("generation_advanced"))
        generated = payload.get("generation_utc")
        if isinstance(generated, str):
            try:
                state.generation_utc = datetime.fromisoformat(generated)
            except ValueError:
                state.generation_utc = None
        lifted = payload.get("liftoff_utc")
        if isinstance(lifted, str):
            try:
                state.liftoff_utc = datetime.fromisoformat(lifted)
            except ValueError:
                state.liftoff_utc = None
        state.liftoff_from_motion = bool(payload.get("liftoff_from_motion"))
        # Older logs did not save hold_started. A saved hold_until means the
        # hold already began; re-arming it on every restart kept a finished
        # (or bogus) flight on screen for another hour after each update.
        state.hold_started = bool(payload.get("hold_started")) or state.hold_until is not None
        state.cap_logged = bool(payload.get("cap_logged"))
        return state

    def save(self, path=None):
        if not getattr(self, "persist", True):
            return
        path = path or self.path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "hold_until": self.hold_until.isoformat() if self.hold_until else None,
            "vehicles": self.vehicles,
            "window_baseline": self.window_baseline,
            "generation_advanced": self.generation_advanced,
            "generation_utc": self.generation_utc.isoformat() if self.generation_utc else None,
            "liftoff_utc": self.liftoff_utc.isoformat() if self.liftoff_utc else None,
            "liftoff_from_motion": self.liftoff_from_motion,
            "hold_started": self.hold_started,
            "cap_logged": self.cap_logged,
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp, path)

    def clear(self):
        self.vehicles = {}
        self.hold_until = None
        self.hold_started = False
        self.window_baseline = None
        self.generation_advanced = False
        self.liftoff_utc = None
        self.liftoff_from_motion = False
        self.cap_logged = False

    def _new_flight(self, parsed):
        for vehicle in parsed["vehicles"]:
            stored = self.vehicles.get(vehicle["key"])
            if not stored:
                continue
            previous = stored.get("max_mission_time")
            current = vehicle["mission_time_s"]
            if previous is not None and current is not None and current + 120 < previous:
                return True
        return False

    def ingest(self, body, now_utc):
        parsed = parse_feed(body, now_utc)
        parsed["now"] = now_utc
        if self._new_flight(parsed):
            self.clear()
        self.feed_stale = parsed["stale"]
        seen_keys = set()
        for vehicle in parsed["vehicles"]:
            seen_keys.add(vehicle["key"])
            slot = self.vehicles.setdefault(vehicle["key"], {
                "kind": vehicle["kind"],
                "name": vehicle["name"],
                "max_mission_time": None,
                "samples": [],
                "last_seen": None,
                "predicted": [],
            })
            slot["kind"] = vehicle["kind"]
            slot["name"] = vehicle["name"]
            slot["predicted"] = vehicle["predicted"]
            slot["last_seen"] = now_utc.isoformat()
            slot["current"] = _sample_record(vehicle, now_utc)
            slot["missing"] = False
            mission = vehicle["mission_time_s"]
            if mission is not None:
                previous_max = slot.get("max_mission_time")
                slot["max_mission_time"] = mission if previous_max is None else max(previous_max, mission)
            samples = slot["samples"]
            previous = samples[-1] if samples else None
            moving = _is_moving(vehicle, previous)
            slot["moving"] = moving
            if moving or (vehicle["speed_mps"] or 0) > 0:
                slot["last_moved"] = now_utc.isoformat()
            stepped = previous is None or haversine_km(
                previous["lat"], previous["lon"], vehicle["lat"], vehicle["lon"]
            ) >= TRAIL_MIN_STEP_KM or abs(vehicle["alt_m"] - previous["alt_m"]) >= 100
            # The pad point is the start of the flown line. Later points are
            # recorded only while the vehicle is actually moving.
            if stepped and (moving or previous is None):
                samples.append(_sample_record(vehicle, now_utc, fresh=not self.feed_stale))
                slot["samples"] = _downsample_trail(samples, MAX_TRAIL_POINTS)
            if self.liftoff_utc:
                slot["in_flight"] = True
        for key, slot in self.vehicles.items():
            if key not in seen_keys:
                slot["missing"] = True
                if not slot.get("missing_since"):
                    slot["missing_since"] = now_utc.isoformat()
            else:
                slot["missing_since"] = None
                if slot.get("done_reason") == "vanished":
                    slot["done"] = False
                    slot["done_reason"] = None
        self._refresh_done(now_utc)
        self._note_generation(parsed["generation_utc"])
        self.save()
        return parsed

    def apply_backfill(self, grouped, now_utc):
        """Merge a one-shot history into the trail and persist it.

        Local points at the same time stay. Nothing is written when the
        archive does not reach earlier than the trail already on disk.
        """
        if not isinstance(grouped, dict) or not grouped:
            return False
        changed = False
        for key, incoming in grouped.items():
            if not incoming:
                continue
            kind = classify_vehicle(key)
            slot = self.vehicles.setdefault(key, {
                "kind": kind,
                "name": display_name(key, kind),
                "max_mission_time": None,
                "samples": [],
                "last_seen": None,
                "predicted": [],
            })
            slot["kind"] = kind
            slot["name"] = display_name(key, kind)
            previous = list(slot.get("samples") or [])
            merged = merge_trail_samples(previous, incoming)
            if not merged:
                continue
            old_epoch = min(
                (item for item in (trail_epoch(sample) for sample in previous) if item is not None),
                default=None,
            )
            new_epoch = trail_epoch(merged[0])
            if old_epoch is not None and (new_epoch is None or new_epoch >= old_epoch - 1) and len(merged) <= len(previous):
                continue
            slot["samples"] = merged
            missions = [
                sample.get("mission_time_s") for sample in merged
                if sample.get("mission_time_s") is not None
            ]
            if missions:
                slot["max_mission_time"] = max(
                    [slot.get("max_mission_time") or missions[0]] + missions
                )
            if not slot.get("current"):
                slot["current"] = dict(merged[-1])
            slot["missing"] = False
            changed = True
        if changed:
            self.save()
        return changed

    def update_mission(self, launch_data, now_utc):
        """Latch liftoff from the latest NET or from motion, then see who is down.

        Called on every poll so a slip or a scrub moves the window immediately.
        A ship that stays in the feed but stops sending new points is still up.
        """
        now = _aware(now_utc)
        self.pre_window = pre_launch_window(launch_data, now)
        passed = liftoff_net(launch_data, now)
        self.net_passed = passed is not None
        self.flight_label = starship_flight_label(launch_data, now)
        in_window = self.pre_window or self.net_passed
        if self._stale_latch(launch_data, now) or (not in_window and self.liftoff_utc is None):
            # Outside T-30min..NET+48h with no believable liftoff, nothing
            # from the log may count as flight: a latch saved before the
            # gating fix (pad jitter), or old airborne samples, used to keep
            # the live layer on at T-26h and hide the planned path.
            self._drop_flight_history()
        confirmed = self._flight_confirmed()
        # A ship on the pad, even with a running mission clock or GPS jitter,
        # is not liftoff. Require airborne samples and T-30min onward.
        if confirmed and in_window:
            if self.liftoff_utc is None:
                # NET is the liftoff a kiosk that woke up mid-flight did not
                # see. Stamping `now` would start T+ and ORBIT 1 at the poll.
                if passed is not None:
                    self.liftoff_utc = passed
                    logger.info("Starship liftoff set from the passed NET")
                else:
                    self.liftoff_utc = now
                    logger.info("Starship liftoff latched from vehicle flight")
            self.liftoff_from_motion = True
        elif passed is not None and self.liftoff_utc is None:
            self.liftoff_utc = passed
            self.liftoff_from_motion = False
            logger.info("Starship NET passed; waiting for the vehicle to leave the pad")
        elif self.liftoff_utc is not None and not confirmed and passed is None:
            logger.info("Clearing liftoff; the vehicle has not flown")
            self.liftoff_utc = None
            self.liftoff_from_motion = False
            self.hold_started = False
            self.hold_until = None
        if self.liftoff_utc:
            for slot in self.vehicles.values():
                slot["in_flight"] = True
        self._refresh_done(now, launch_data=launch_data)
        if self._past_safety_cap(now):
            if not self.cap_logged:
                logger.warning(
                    "Starship tracker reached 48h after liftoff; falling back to idle poll"
                )
                self.cap_logged = True
            self.save()
            return
        self._arm_post_catch_hold(now)
        self.save()

    def _stale_latch(self, launch_data, now):
        """A saved liftoff that no known Starship NET window can explain.

        Valid when liftoff falls in [NET - 30 min, NET + 48 h] for some
        relevant NET. With no NETs at all (no launch data yet) the latch is
        kept, so a restart mid-flight while offline still shows the flight.
        """
        if self.liftoff_utc is None:
            return False
        nets = _relevant_nets(launch_data)
        if not nets:
            return False
        lifted = _aware(self.liftoff_utc)
        for net in nets:
            delta = (lifted - net).total_seconds()
            if -WINDOW_BEFORE_S <= delta <= MISSION_CAP_S:
                return False
        return True

    def _drop_flight_history(self):
        """Forget liftoff, hold and trails; keep each vehicle's latest fix."""
        if self.liftoff_utc is not None:
            logger.info("Dropping a stale Starship liftoff latch from the tracker log")
        self.liftoff_utc = None
        self.liftoff_from_motion = False
        self.hold_started = False
        self.hold_until = None
        self.cap_logged = False
        for slot in self.vehicles.values():
            current = slot.get("current")
            slot["samples"] = [current] if current else []
            slot["in_flight"] = False
            slot["done"] = False
            slot["done_reason"] = None
            slot["stable_since"] = None

    def _sample_airborne(self, sample):
        """Off the pad. Orbital altitude counts even with no mission clock.

        Hobby feeds often omit ``mission_time``. A restart at T+17h then
        never confirmed the flight, logged "waiting for the vehicle to
        leave the pad", and treated the ship as down.
        """
        if not sample or sample.get("fresh") is False:
            return False
        try:
            alt = float(sample.get("alt_m") or 0)
            speed = float(sample.get("speed_mps") or 0)
        except (TypeError, ValueError):
            return False
        if alt >= ORBIT_COAST_ALT_M or speed >= ORBIT_COAST_SPEED_MPS:
            return True
        try:
            mission = float(sample.get("mission_time_s"))
        except (TypeError, ValueError):
            return False
        if mission <= 0:
            return False
        return alt > AIRBORNE_ALT_M or speed > AIRBORNE_SPEED_MPS

    def _airborne_count(self, slot):
        return sum(1 for sample in (slot or {}).get("samples") or [] if self._sample_airborne(sample))

    def _flight_confirmed(self):
        """True once one vehicle has repeated airborne samples, not a pad jitter."""
        return any(
            self._airborne_count(slot) >= FRESH_SAMPLES_FOR_HANDOVER
            for slot in self.vehicles.values()
        )

    def _in_launch_window(self):
        """T-30min onward: the pre-launch window, or a NET that has passed.

        A motion latch counts too, but update_mission drops any latch that
        no NET window explains, so it can no longer hold itself open.
        """
        return bool(self.pre_window or self.net_passed or (
            self.liftoff_utc is not None and self.liftoff_from_motion
        ))

    def _past_safety_cap(self, now):
        if not self.liftoff_utc:
            return False
        return (now - _aware(self.liftoff_utc)).total_seconds() >= MISSION_CAP_S

    def _all_done(self):
        slots = [slot for slot in self.vehicles.values() if slot.get("in_flight")]
        return bool(slots) and all(slot.get("done") for slot in slots)

    def _any_clearly_airborne(self):
        for slot in self.vehicles.values():
            if not slot.get("in_flight") or slot.get("missing"):
                continue
            if _is_clearly_airborne(slot.get("current") or {}):
                return True
        return False

    def _terminal_window_launch(self, launch_data, now):
        launch = window_starship_launch(launch_data, now)
        return launch if launch and _is_terminal_launch_status(_launch_status_text(launch)) else None

    def _vehicle_finished_at(self, slot, now):
        """When this vehicle first counted as down, or *now* if that is unknown."""
        reason = slot.get("done_reason")
        now = _aware(now)
        if reason == "landed":
            since = _stamp(slot.get("stable_since"))
            if since is not None:
                finished = since + timedelta(seconds=GROUNDED_STABLE_S)
                # Already-down slots may have stable_since filled in on this
                # poll; the finish time cannot be in the future.
                return finished if finished <= now else now
            return now
        if reason == "vanished":
            since = _stamp(slot.get("missing_since"))
            if since is not None:
                finished = since + timedelta(seconds=KEY_GONE_S)
                return finished if finished <= now else now
            return now
        return now

    def _hold_deadline(self, now):
        """End of the post-catch hold, anchored to landing time, not process start.

        A restart (or a first update after the speed fix) used to set
        hold_until = now + 1h even when the ship had already been on the
        pad for hours, so Starship mode could come back after every boot
        until the 48h cap.
        """
        finished = [
            self._vehicle_finished_at(slot, now)
            for slot in self.vehicles.values()
            if slot.get("in_flight") and slot.get("done")
        ]
        if not finished:
            return _aware(now) + timedelta(seconds=POST_CATCH_HOLD_S)
        return max(finished) + timedelta(seconds=POST_CATCH_HOLD_S)

    def _arm_post_catch_hold(self, now):
        if not self.liftoff_utc or not self._flight_confirmed():
            return
        if not self._all_done():
            # A brief un-ground (predicted-path speed, pad GPS jitter) must
            # not cancel a hold already in progress. Only a clear takeoff
            # starts the episode again.
            if self.hold_started and self._any_clearly_airborne():
                self.hold_started = False
                self.hold_until = None
            return
        deadline = self._hold_deadline(now)
        already = _aware(self.hold_until) if self.hold_until else None
        # Keep the earlier of a stored deadline and the one implied by
        # landing time, so a morning restart cannot push the hold out.
        if already is not None and already < deadline:
            deadline = already
        if not self.hold_started or self.hold_until is None or _aware(self.hold_until) != deadline:
            if not self.hold_started:
                if now >= deadline:
                    logger.info(
                        "Starship vehicles have been down past the 1h hold; leaving Starship mode"
                    )
                else:
                    logger.info("Starship vehicles are down; keeping the flown trail for 1h")
            self.hold_started = True
            self.hold_until = deadline

    def _refresh_done(self, now, launch_data=None):
        if not self.liftoff_utc:
            return
        now = _aware(now)
        terminal = bool(self._terminal_window_launch(launch_data, now))
        stale_s = None
        if self.feed_stale and self.generation_utc is not None:
            stale_s = (now - _aware(self.generation_utc)).total_seconds()
        for slot in self.vehicles.values():
            if not slot.get("in_flight"):
                continue
            if slot.get("missing"):
                since = _stamp(slot.get("missing_since"))
                if since is None:
                    slot["missing_since"] = now.isoformat()
                elif (now - since).total_seconds() >= KEY_GONE_S:
                    slot["done"] = True
                    if not slot.get("done_reason"):
                        slot["done_reason"] = "vanished"
                continue
            current = slot.get("current") or {}
            if _is_clearly_airborne(current):
                slot["stable_since"] = None
                if slot.get("done_reason") in ("landed", "stale"):
                    slot["done"] = False
                    slot["done_reason"] = None
                continue
            if _is_grounded(current):
                since = _stamp(slot.get("stable_since"))
                if since is None:
                    slot["stable_since"] = now.isoformat()
                    since = now
                needed = GROUNDED_STABLE_S
                # The official clock often says "Launch Successful" while
                # the ship is still in orbit. Only cut the settle short
                # once the vehicle is actually on the ground.
                if terminal:
                    needed = min(needed, 60)
                if (now - since).total_seconds() >= needed:
                    slot["done"] = True
                    slot["done_reason"] = "landed"
                continue
            # Not grounded, but also not clearly flying: keep a landed latch
            # and do not restart the settle timer on a 3 m/s glitch.
            if slot.get("done_reason") == "landed":
                continue
            if stale_s is not None and stale_s >= STALE_GROUNDED_S:
                last = current
                if last and (
                    _is_grounded(last)
                    or (last.get("alt_m") is not None and float(last.get("alt_m") or 0) <= GROUNDED_ALT_M)
                ):
                    slot["done"] = True
                    slot["done_reason"] = "stale"
                    continue
            slot["stable_since"] = None

    def fresh_sample_count(self, slot):
        total = 0
        for sample in (slot or {}).get("samples") or []:
            if sample.get("fresh", True):
                total += 1
        return total

    def live_layer_active(self, now_utc=None):
        """Markers, labels, and the flown trail.

        They stay off until the vehicle has actually left the pad and the
        clock is inside T-30min onward. A ship sitting on the pad does not
        qualify, no matter how many samples the feed has published. The
        planned path hides earlier, at T-0 (see ``planned_path_hidden``).
        """
        now = _aware(now_utc) if now_utc is not None else None
        if now is not None and self._past_safety_cap(now):
            return False
        if not self._flight_confirmed() or not self._in_launch_window():
            return False
        if (
            now is not None
            and self._all_done()
            and self.hold_started
            and self.hold_until
            and now >= _aware(self.hold_until)
        ):
            return False
        return True

    def planned_path_hidden(self, now_utc):
        """Hide the planned trajectory from T-0 until the mission window ends.

        T-0 is the motion-latched liftoff when that exists, otherwise the
        NET once it has passed. Two airborne samples are not required, so
        the path is gone at liftoff instead of waiting for the tracker to
        confirm the vehicle is flying. A scrub or hold that moves NET later
        clears an unconfirmed latch and the path comes back. Before T-0 the
        path stays visible. After the post-catch hold, or past the 48h cap,
        it can show again for the next launch.
        """
        now = _aware(now_utc)
        if now is None or self.liftoff_utc is None:
            return False
        if self._past_safety_cap(now):
            return False
        if now < _aware(self.liftoff_utc):
            return False
        if (
            self._flight_confirmed()
            and self._all_done()
            and self.hold_started
            and self.hold_until
            and now >= _aware(self.hold_until)
        ):
            return False
        # NET passed (or a motion latch) keeps the window open. A latch that
        # update_mission already dropped does not hide the path.
        if not self._in_launch_window() and not self.liftoff_from_motion:
            return False
        return True

    def replaces_planned_trajectory(self, now_utc=None):
        """The planned path yields at T-0, before the two-sample liftoff latch."""
        if now_utc is None:
            return False
        return self.planned_path_hidden(now_utc)

    def wants_fast_poll(self, now_utc):
        """10s from T-30min until the mission ends, the hold expires, or 48h passes."""
        now = _aware(now_utc)
        if self._past_safety_cap(now):
            return False
        # T-30min onward follows the NET. A vehicle on the pad does not.
        if self.pre_window or self.net_passed:
            if self._all_done() and self.hold_started and self.hold_until and now >= _aware(self.hold_until):
                return False
            return True
        if self.live_layer_active(now):
            return True
        if self.hold_started and self.hold_until and now < _aware(self.hold_until):
            return True
        return False

    def _note_generation(self, generation_utc):
        if generation_utc is None:
            return
        self.generation_utc = generation_utc

    def globe_payload(self, now_utc, in_window=False):
        """What the globe draws. The flown trail stays bright through catch."""
        holding = bool(self.hold_until and now_utc < self.hold_until)
        handover = self.live_layer_active(now_utc)
        vehicles_out = []
        any_show = False
        for key, slot in self.vehicles.items():
            current = slot.get("current") or {}
            samples = slot.get("samples") or []
            moving = bool(slot.get("moving"))
            # No marker, label, or trail until the vehicle has left the pad
            # inside the launch window. The planned path is a separate group.
            show_vehicle = handover or (holding and self._flight_confirmed())
            if not show_vehicle:
                continue
            any_show = True
            trail = [
                {"lat": item["lat"], "lon": item["lon"], "alt_m": item["alt_m"]}
                for item in _downsample_trail(samples, DISPLAY_TRAIL_CAP)
            ]
            predicted = slot.get("predicted") or []
            vehicles_out.append({
                "key": key,
                "kind": slot.get("kind") or "other",
                "name": slot.get("name") or key,
                "moving": moving,
                "missing": bool(slot.get("missing")),
                "stale": bool(self.feed_stale or slot.get("missing")),
                "lat": current.get("lat"),
                "lon": current.get("lon"),
                "alt_m": current.get("alt_m"),
                "speed_mps": current.get("speed_mps") or 0,
                "altitude_text": format_altitude(current.get("alt_m")),
                "speed_text": format_speed(current.get("speed_mps")),
                "mission_text": format_mission_time(current.get("mission_time_s")),
                # Numeric clock for the globe's dead-reckoning between polls.
                "mission_time_s": current.get("mission_time_s"),
                # GPS seconds of this fix. The globe walks the predicted
                # path in ECEF from here; mission_time_s is not that clock.
                "gps_s": current.get("gps_s"),
                # How old the fix already is. The globe samples the predicted
                # path at gps + this, not at the path's first point.
                "sample_age_s": _sample_age_s(current.get("gps_s"), now_utc),
                "apsis_label": ship_apsis_label(slot) if slot.get("kind") == "ship" else "",
                "trail": trail,
                "predicted": [
                    {
                        "lat": item["lat"],
                        "lon": item["lon"],
                        "alt_m": item["alt_m"],
                        "t": item.get("time"),
                    }
                    for item in predicted
                ],
            })
        return {
            "show": any_show,
            "stale": bool(self.feed_stale),
            "hold": holding,
            # T-0 (NET, or the motion liftoff if that is earlier). Not the
            # two-sample latch: the path is gone when the clock hits zero.
            "hide_planned": self.planned_path_hidden(now_utc),
            # Real liftoff (two airborne samples, T-30min onward) until the
            # mission ends (post-catch hold expired). The globe shows only
            # the Starlink V3 shell (/satellites/deployed) while this is on.
            "launch_active": handover,
            "vehicles": vehicles_out,
            "telemetry": self.telemetry_strip(handover, now_utc),
        }

    def telemetry_strip(self, show, now_utc=None):
        """Webcast numbers for the overlay. Built once per poll, not per frame."""
        ship_key = booster_key = None
        ship = booster = None
        for key, slot in self.vehicles.items():
            kind = slot.get("kind")
            if kind == "ship" and ship is None:
                ship_key, ship = key, slot
            elif kind == "booster" and booster is None:
                booster_key, booster = key, slot
        clock = None
        for slot in (ship, booster):
            current = (slot or {}).get("current") or {}
            if slot and not slot.get("missing") and current.get("mission_time_s") is not None:
                clock = current.get("mission_time_s")
                if slot is ship:
                    break
        feed_stale = bool(self.feed_stale)
        booster_out = vehicle_readout(booster_key, booster, "booster", now_utc)
        ship_out = vehicle_readout(ship_key, ship, "ship", now_utc)
        if feed_stale:
            for side in (booster_out, ship_out):
                if side["present"]:
                    side["stale"] = True
        orbit = ship_orbit_status(ship, liftoff_utc=self.liftoff_utc, now_utc=now_utc) if show else None
        clock_base_s = None
        if self.liftoff_utc is not None and now_utc is not None:
            # Local elapsed since liftoff. The sample's mission_time_s only
            # changes when a new fix arrives, so the strip must not display it.
            clock_base_s = (_aware(now_utc) - _aware(self.liftoff_utc)).total_seconds()
        return {
            "show": bool(show),
            "clock": format_webcast_clock(clock) if clock is not None else "T+00:00:00",
            "clock_base_s": clock_base_s,
            "flight_label": getattr(self, "flight_label", "") or "",
            # Apogee / perigee above the clock. Empty on the pad and when
            # the state vector is not a closed ellipse.
            "apsis_label": ship_apsis_label(ship) if show else "",
            "stale": feed_stale,
            "booster": booster_out,
            "ship": ship_out,
            # Above the T clock once the ship is in coast. Empty otherwise.
            "in_orbit": bool(orbit),
            "orbit_label": orbit["orbit_label"] if orbit else "",
            "orbit_number": orbit["orbit_number"] if orbit else 0,
            "completed_laps": orbit["completed_laps"] if orbit else 0,
        }


def synthetic_inflight_feeds():
    """Booster up-and-back to Starbase, ship arcing east toward the Indian Ocean.

    Returns a list of tracker documents, one per ~30s of flight, plus a few
    post-catch samples with speed 0 so the trail-hold path can be checked.
    """
    start_gps = 1_478_600_000.0
    feeds = []
    steps = 36
    for i in range(steps):
        gps = start_gps + i * 30
        if i <= 16:
            b_alt = 150 + i * 4500
            b_lat = 25.997 + i * 0.35
            b_lon = -97.15 - i * 0.2
        else:
            back = i - 16
            remain = max(0, 16 - back)
            b_alt = 150 + remain * 4500
            b_lat = 25.997 + remain * 0.35
            b_lon = -97.15 - remain * 0.2
        b_speed = 0 if i in (0, steps - 1) else 280 + i * 8
        frac = i / (steps - 1)
        s_lon = -97.15 + frac * 150
        s_lat = 25.99 - frac * 18
        s_alt = 200 + math.sin(frac * math.pi) * 130000
        s_speed = 0 if i == 0 else 1800
        def _veh(lat, lon, alt, speed, mission):
            predicted = []
            for step in range(1, 5):
                predicted.append({
                    "time": gps + step * 60,
                    "latitude": lat + step * 0.05,
                    "longitude": lon + step * 0.4,
                    "altitude": max(0, alt + step * 200),
                    "r_ecef": [0, 0, 0],
                })
            return {
                "current": {
                    "gps_time": gps,
                    "mission_time": mission,
                    "altitude": alt,
                    "speed": speed,
                    "latitude": lat,
                    "longitude": lon,
                    "r_ecef": [0, 0, 0],
                },
                "trajectory": predicted,
            }
        feeds.append({
            "booster19": _veh(b_lat, b_lon, b_alt, b_speed, 30 * i),
            "ship42": _veh(s_lat, s_lon, s_alt, s_speed, 30 * i),
            "metadata": {
                "generation_time": gps,
                "trajectory_version": 1,
                "coordinate_frame": "ECEF",
                "sample_period": 30,
            },
        })
    # Catch: both vehicles stopped at the tower / ship still downrange but slow.
    catch_gps = start_gps + steps * 30
    feeds.append({
        "booster19": {
            "current": {
                "gps_time": catch_gps,
                "mission_time": 30 * steps,
                "altitude": 120,
                "speed": 0,
                "latitude": 25.997,
                "longitude": -97.15,
                "r_ecef": [0, 0, 0],
            },
            "trajectory": [],
        },
        "ship42": {
            "current": {
                "gps_time": catch_gps,
                "mission_time": 30 * steps,
                "altitude": 8000,
                "speed": 40,
                "latitude": 8.0,
                "longitude": 52.0,
                "r_ecef": [0, 0, 0],
            },
            "trajectory": [],
        },
        "metadata": {
            "generation_time": catch_gps,
            "trajectory_version": 1,
            "coordinate_frame": "ECEF",
            "sample_period": 30,
        },
    })
    return feeds


def _launch_at(moment):
    return {"upcoming": [{"net": moment.isoformat(), "rocket": "Starship"}], "previous": []}


def replay_feeds(feeds, hold_clock=True):
    """Ingest a scripted flight. Returns (state, inflight_payload, catch_payload)."""
    state = TrackerState(path=os.path.join(os.devnull, "unused"))
    state.save = lambda path=None: None  # type: ignore[method-assign]
    inflight = None
    start = gps_to_utc(feeds[0]["metadata"]["generation_time"])
    for index, feed in enumerate(feeds):
        now = gps_to_utc(feed["metadata"]["generation_time"])
        state.ingest(feed, now)
        if index == len(feeds) - 2:
            state.update_mission(_launch_at(start), now)
            inflight = state.globe_payload(now, in_window=True)
    catch_now = gps_to_utc(feeds[-1]["metadata"]["generation_time"])
    if not hold_clock:
        catch_now = start
    # The ship is still flying here, so the mission (and both trails) stays up.
    state.update_mission(_launch_at(start), catch_now)
    catch = state.globe_payload(catch_now, in_window=True)
    return state, inflight, catch
