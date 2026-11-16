"""FAA and maritime closure polygons for the Starship flight on the globe.

Runtime fetch prefers the public FAA TFR list and NGA broadcast warnings.
When a source fails, the hardcoded notices captured for Starship Flight 15
(Starlink Group 31-2, NET 13 Nov 2026) are used instead. The planned path
is only checked against these areas; it is never rewritten.
"""
from __future__ import annotations

import html
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone

import requests

from dashboard_paths import CACHE_DIR
from launch_dates import _get_parsed_dt

logger = logging.getLogger(__name__)

FAA_LIST_URL = "https://tfr.faa.gov/tfrapi/exportTfrList"
FAA_TEXT_URL = "https://tfr.faa.gov/tfrapi/getWebText?notamId={notam_id}"
NGA_BROADCAST_URL = "https://msi.nga.mil/api/publications/broadcast-warn"
CACHE_FILE = os.path.join(CACHE_DIR, "closure_areas.json")
REFRESH_S = 6 * 3600
SHOW_BEFORE_S = 24 * 3600
SHOW_AFTER_CAP_S = 48 * 3600
FETCH_TIMEOUT_S = 8
_MAX_TFR_DETAILS = 6

# Gulf / Yucatan box for rocket warnings that do not name Starship.
_GULF_BOX = (18.0, 31.0, -98.5, -80.0)

_lock = threading.Lock()
_memory = {"areas": None, "source": "fallback", "fetched_at": None}
_warned_paths = set()

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}

# Hydrographic pairs: 25-58.00N 097-13.00W and 25-58-00N 097-13-00W.
_HYDRO_PAIR = re.compile(
    r"(\d{1,3})-(\d{2})(?:-(\d{2}))?(?:\.(\d+))?([NS])\s+"
    r"(\d{1,3})-(\d{2})(?:-(\d{2}))?(?:\.(\d+))?([EW])",
    re.IGNORECASE,
)
_DMS_PAIR = re.compile(
    r"(\d{1,3})\s*[°º]\s*(\d{1,2})\s*'\s*(\d{1,2}(?:\.\d+)?)\s*\"?\s*([NS])\s+"
    r"(\d{1,3})\s*[°º]\s*(\d{1,2})\s*'\s*(\d{1,2}(?:\.\d+)?)\s*\"?\s*([EW])",
    re.IGNORECASE,
)
_FAA_WHEN = re.compile(
    r"(Beginning|Ending) Date and Time\s*:\s*"
    r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})\s+at\s+(\d{3,4})\s+UTC",
    re.IGNORECASE,
)


def _dms_to_deg(deg, minutes, seconds, fraction, hemi):
    minutes_value = int(minutes)
    if fraction:
        minutes_value += int(fraction) / (10 ** len(fraction))
    if seconds:
        minutes_value += int(seconds) / 60.0
    value = int(deg) + minutes_value / 60.0
    if hemi.upper() in ("S", "W"):
        value = -value
    return value


def _hydro_match(match):
    lat = _dms_to_deg(match.group(1), match.group(2), match.group(3), match.group(4), match.group(5))
    lon = _dms_to_deg(match.group(6), match.group(7), match.group(8), match.group(9), match.group(10))
    return lat, lon


def _dms_match(match):
    lat = _dms_to_deg(match.group(1), match.group(2), match.group(3), None, match.group(4))
    lon = _dms_to_deg(match.group(5), match.group(6), match.group(7), None, match.group(8))
    return lat, lon


def _normalize_notice(text):
    plain = html.unescape(text or "")
    plain = plain.replace("&#xBA;", "°").replace("º", "°")
    plain = re.sub(r"<[^>]+>", " ", plain)
    return plain


def _rings_from_matches(text, pattern, converter):
    matches = list(pattern.finditer(text))
    rings = []
    current = []
    last_end = None
    for match in matches:
        if last_end is not None:
            gap = text[last_end:match.start()]
            gap = re.sub(r"\b(to|from|and)\b", " ", gap, flags=re.IGNORECASE)
            # FAA rows include a fix/radial/distance such as BRO078011.3.
            gap = re.sub(r"\b[A-Za-z]{2,}\d[\d.]*", " ", gap)
            if re.search(r"[A-Za-z]", gap):
                if len(current) >= 3:
                    rings.append(current)
                current = []
        current.append({"lat": converter(match)[0], "lon": converter(match)[1]})
        last_end = match.end()
    if len(current) >= 3:
        rings.append(current)
    cleaned = []
    for ring in rings:
        if ring[0] == ring[-1] and len(ring) > 3:
            ring = ring[:-1]
        cleaned.append(ring)
    return cleaned


def rings_from_notice(text):
    """Pull one or more closed rings out of a NOTAM or broadcast warning."""
    plain = _normalize_notice(text)
    hydro = _rings_from_matches(plain, _HYDRO_PAIR, _hydro_match)
    if hydro:
        return hydro
    return _rings_from_matches(plain, _DMS_PAIR, _dms_match)


def _parse_faa_when(plain):
    found = {}
    for match in _FAA_WHEN.finditer(plain):
        month = _MONTHS.get(match.group(2).lower())
        if not month:
            continue
        hour_min = match.group(5).zfill(4)
        found[match.group(1).lower()] = datetime(
            int(match.group(4)), month, int(match.group(3)),
            int(hour_min[:2]), int(hour_min[2:]), tzinfo=timezone.utc,
        )
    return found.get("beginning"), found.get("ending")


def _in_gulf(ring):
    lat0, lat1, lon0, lon1 = _GULF_BOX
    for point in ring:
        if lat0 <= point["lat"] <= lat1 and lon0 <= point["lon"] <= lon1:
            return True
    return False


def _keyword_hit(text):
    blob = (text or "").lower()
    return any(word in blob for word in ("starship", "starbase", "spacex", "boca chica"))


def _known_flight_id(text):
    blob = (text or "").upper()
    return any(token in blob for token in ("1080/26", "1081/26", "3256/26", "6/7129"))


# Notices captured 12 Nov 2026 for Starship Flight 15 / Starlink Group 31-2.
# FAA FDC 6/7129: STARBASE space operations, 13 Nov 2026 2230Z–14 Nov 0040Z.
# The same ring is also FDC 6/7130 (14 Nov 2230Z–15 Nov 0040Z),
# FDC 6/7131 (14 Nov 2053Z–15 Nov 0014Z), and FDC 6/7132 (15 Nov 2053Z–16 Nov 0014Z).
# NAVAREA IV 1080/26 (issued 090753Z NOV 26): rocket launching,
# 13 Nov 2245Z–14 Nov 0040Z, alternate 2245Z–0040Z daily 14–19 Nov,
# cancel 200140Z NOV 26. Area A is the downrange corridor. Area B is the
# high-risk zone (booster return / near-shore). USCG Heartland District
# LNM 45/2026 (LNM08G-61C4A6, notice 58067) repeats these two rings.
# NAVAREA IV 1081/26 (issued 090809Z NOV 26): space debris off Texas,
# 14 Nov 2108Z–15 Nov 0014Z, alternate 2108Z–0014Z daily 15–20 Nov,
# cancel 210114Z NOV 26.
# HYDROPAC 3256/26 (issued 090840Z NOV 26): Pacific space debris,
# 2217Z–0144Z daily 14–20 Nov, cancel 210244Z NOV 26.
_FALLBACK_TEXTS = (
    {
        "notice_id": "FDC 6/7129",
        "kind": "airspace",
        "role": "airspace",
        "name": "Starbase airspace",
        "effective_start": "2026-11-13T22:30:00+00:00",
        "effective_end": "2026-11-14T00:40:00+00:00",
        "also": ["FDC 6/7130", "FDC 6/7131", "FDC 6/7132"],
        "text": (
            "FDC 6/7129 STARBASE, Texas space operations. "
            "From: 25°56'00\"N 97°10'00\"W To: 26°01'00\"N 97°12'00\"W "
            "To: 26°04'00\"N 97°07'00\"W To: 26°06'00\"N 96°55'00\"W "
            "To: 26°00'00\"N 96°55'00\"W To: 26°00'00\"N 97°00'00\"W "
            "To: 25°56'00\"N 97°10'00\"W"
        ),
    },
    {
        "notice_id": "NAVAREA IV 1080/26",
        "kind": "maritime",
        "role": "corridor",
        "name": "Downrange hazard",
        "effective_start": "2026-11-13T22:45:00+00:00",
        "effective_end": "2026-11-20T01:40:00+00:00",
        "text": (
            "NAVAREA IV 1080/26. CARIBBEAN SEA. GULF OF AMERICA. YUCATAN CHANNEL. TEXAS. "
            "HAZARDOUS OPERATIONS, ROCKET LAUNCHING "
            "132245Z TO 140040Z NOV, ALTERNATE 2245Z TO 0040Z DAILY 14 THRU 19 NOV "
            "IN AREAS BOUND BY: "
            "A. 25-58.00N 097-13.00W, 26-02.00N 097-13.00W, 26-04.00N 097-10.00W, "
            "26-06.00N 097-04.00W, 26-06.00N 096-49.00W, 26-01.00N 095-12.00W, "
            "25-41.00N 093-29.00W, 25-11.00N 091-59.00W, 23-32.00N 088-52.00W, "
            "23-21.00N 088-29.00W, 22-31.00N 086-29.00W, 21-41.00N 085-07.00W, "
            "21-38.00N 084-59.00W, 21-33.00N 084-32.00W, 21-26.00N 083-49.00W, "
            "20-59.00N 082-38.00W, 20-43.00N 081-42.00W, 20-31.00N 081-03.00W, "
            "20-12.00N 081-13.00W, 20-36.00N 083-14.00W, 21-06.00N 084-54.00W, "
            "21-43.00N 086-42.00W, 21-59.00N 087-26.00W, 22-16.00N 089-19.00W, "
            "22-16.00N 089-44.00W, 22-16.00N 091-09.00W, 23-16.00N 093-29.00W, "
            "24-36.00N 095-19.00W, 25-38.00N 096-49.00W, 25-48.00N 097-05.00W. "
            "B. 25-12.00N 092-58.00W, 25-31.00N 093-52.00W, 25-42.00N 095-24.00W, "
            "26-02.00N 096-32.00W, 26-04.00N 096-49.00W, 26-04.00N 097-04.00W, "
            "26-04.00N 097-10.00W, 26-02.00N 097-13.00W, 25-58.00N 097-13.00W, "
            "25-48.00N 097-05.00W, 25-41.00N 096-47.00W, 25-02.00N 095-05.00W, "
            "24-29.00N 094-08.00W, 24-20.00N 093-32.00W, 24-25.00N 092-52.00W, "
            "24-45.00N 092-43.00W. "
            "2. CANCEL THIS MSG 200140Z NOV 26."
        ),
    },
    {
        "notice_id": "NAVAREA IV 1081/26",
        "kind": "maritime",
        "role": "debris",
        "name": "Texas debris",
        "effective_start": "2026-11-14T21:08:00+00:00",
        "effective_end": "2026-11-21T01:14:00+00:00",
        "text": (
            "NAVAREA IV 1081/26. GULF OF AMERICA. TEXAS. "
            "HAZARDOUS OPERATIONS, SPACE DEBRIS "
            "142108Z TO 150014Z NOV IN AREA BOUND BY "
            "25-56.00N 097-09.00W, 25-56.00N 097-05.00W, 25-57.00N 096-28.00W, "
            "25-59.00N 096-25.00W, 26-10.00N 096-22.00W, 26-20.00N 096-52.00W, "
            "26-46.00N 097-28.00W, 26-36.00N 097-27.00W, 26-22.00N 097-13.00W, "
            "26-14.00N 097-09.00W, 26-06.00N 097-08.00W, 26-04.00N 097-09.00W, "
            "26-03.00N 097-11.00W, 26-01.00N 097-13.00W, 25-58.00N 097-12.00W."
        ),
    },
    {
        "notice_id": "HYDROPAC 3256/26",
        "kind": "maritime",
        "role": "debris",
        "name": "Pacific debris",
        "effective_start": "2026-11-14T22:17:00+00:00",
        "effective_end": "2026-11-21T02:44:00+00:00",
        "text": (
            "HYDROPAC 3256/26. NORTH PACIFIC. HAWAII. "
            "HAZARDOUS OPERATIONS, SPACE DEBRIS "
            "2217Z TO 0144Z DAILY 14 THR 20 NOV IN AREA BOUND BY "
            "09-09.00N 153-46.00E, 10-48.00N 156-18.00E, 14-04.00N 162-04.00E, "
            "16-07.00N 165-35.00E, 20-38.00N 174-31.00E, 22-43.00N 179-15.00E, "
            "24-51.00N 176-13.00W, 25-53.00N 173-20.00W, 26-04.00N 171-02.00W, "
            "28-31.00N 161-14.00W, 30-33.00N 150-16.00W, 29-23.00N 150-15.00W, "
            "26-01.00N 153-49.00W, 24-46.00N 154-54.00W, 25-24.00N 157-46.00W, "
            "24-54.00N 161-42.00W, 23-11.00N 168-32.00W, 22-08.00N 173-08.00W, "
            "20-12.00N 179-07.00W, 19-21.00N 178-21.00E, 18-13.00N 174-58.00E, "
            "17-02.00N 170-17.00E, 15-11.00N 165-56.00E, 11-55.00N 159-26.00E, "
            "08-48.00N 154-06.00E."
        ),
    },
)

_ROLE_FOR_LABEL = {"A": "corridor", "B": "high_risk"}


def areas_from_notice(spec, rings=None):
    """Turn one notice record into globe areas. Role B is the high-risk ring."""
    found = rings if rings is not None else rings_from_notice(spec.get("text") or "")
    areas = []
    for index, ring in enumerate(found):
        role = spec.get("role") or "corridor"
        name = spec.get("name") or spec.get("notice_id")
        if len(found) > 1:
            label = chr(ord("A") + index)
            role = _ROLE_FOR_LABEL.get(label, role)
            name = f"{name} {label}"
        areas.append({
            "notice_id": spec.get("notice_id"),
            "also": list(spec.get("also") or []),
            "kind": spec.get("kind") or "maritime",
            "role": role,
            "name": name,
            "effective_start": spec.get("effective_start"),
            "effective_end": spec.get("effective_end"),
            "points": ring,
        })
    return areas


def fallback_areas():
    areas = []
    for spec in _FALLBACK_TEXTS:
        areas.extend(areas_from_notice(spec))
    return areas


def _starship_tfr(item):
    if not isinstance(item, dict):
        return False
    blob = " ".join(str(item.get(key) or "") for key in ("type", "description", "facility", "state")).lower()
    if "starship" in blob or "starbase" in blob:
        return "space" in blob or "starbase" in blob
    return False


def _get(url, get, timeout):
    from starship_tracker import tracker_user_agent

    getter = get or requests.get
    response = getter(
        url,
        headers={"User-Agent": tracker_user_agent(), "Accept": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response


def fetch_faa_areas(get=None, timeout=FETCH_TIMEOUT_S):
    """Starbase space-operations TFRs. Identical rings collapse to one polygon."""
    response = _get(FAA_LIST_URL, get, timeout)
    payload = response.json()
    items = [item for item in payload if _starship_tfr(item)]
    items.sort(key=lambda item: item.get("notam_id") or "")
    areas = []
    seen = set()
    siblings = []
    for item in items[:_MAX_TFR_DETAILS]:
        notam_id = item.get("notam_id")
        if not notam_id:
            continue
        detail = _get(FAA_TEXT_URL.format(notam_id=notam_id), get, timeout)
        body = detail.json()
        text = ""
        if isinstance(body, list) and body:
            text = body[0].get("text") or ""
        elif isinstance(body, dict):
            text = body.get("text") or ""
        plain = _normalize_notice(text)
        start, end = _parse_faa_when(plain)
        rings = rings_from_notice(plain)
        if not rings:
            continue
        key = tuple((round(point["lat"], 4), round(point["lon"], 4)) for point in rings[0])
        label = f"FDC {notam_id}"
        if key in seen:
            siblings.append(label)
            continue
        seen.add(key)
        spec = {
            "notice_id": label,
            "kind": "airspace",
            "role": "airspace",
            "name": "Starbase airspace",
            "effective_start": start.isoformat() if start else None,
            "effective_end": end.isoformat() if end else None,
            "also": [],
        }
        built = areas_from_notice(spec, rings)
        areas.extend(built)
    if areas and siblings:
        areas[0]["also"] = siblings
    return areas


def _nga_texts(payload):
    texts = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key.lower() in ("text", "message", "body", "warning", "msgtext", "navtext") and isinstance(value, str):
                    texts.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str) and ("NAVAREA" in node.upper() or "HYDROPAC" in node.upper() or "NAVTEX" in node.upper()):
            texts.append(node)

    walk(payload)
    return texts


def _maritime_wanted(text, rings):
    if _keyword_hit(text) or _known_flight_id(text):
        return True
    blob = text.upper()
    hazard = any(token in blob for token in ("ROCKET", "SPACE DEBRIS", "SPACE OPERATIONS", "STARSHIP"))
    return hazard and any(_in_gulf(ring) for ring in rings)


def fetch_nga_areas(get=None, timeout=FETCH_TIMEOUT_S):
    response = _get(NGA_BROADCAST_URL, get, timeout)
    texts = _nga_texts(response.json())
    areas = []
    for text in texts:
        rings = rings_from_notice(text)
        if not rings or not _maritime_wanted(text, rings):
            continue
        notice_id = "NGA broadcast"
        match = re.search(r"((?:NAVAREA\s+[IVX]+|HYDROPAC|NAVTEX)\s+\d+/\d+)", text, re.IGNORECASE)
        if match:
            notice_id = re.sub(r"\s+", " ", match.group(1).upper())
        role = "debris" if "DEBRIS" in text.upper() else "corridor"
        areas.extend(areas_from_notice({
            "notice_id": notice_id,
            "kind": "maritime",
            "role": role,
            "name": notice_id,
            "text": text,
        }, rings))
    return areas


def _save(path, areas, source, when):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump({
            "fetched_at": when.isoformat(),
            "source": source,
            "areas": areas,
        }, handle)
    os.replace(tmp, path)


def _load(path):
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not payload.get("areas"):
        return None
    return payload


def refresh(path=CACHE_FILE, get=None, now=None):
    """Fetch FAA and NGA. A failed side keeps that side's fallback polygons."""
    clock = now or datetime.now(timezone.utc)
    air_source = "fallback"
    sea_source = "fallback"
    try:
        air = fetch_faa_areas(get=get)
        if air:
            air_source = "live"
    except Exception as exc:
        logger.info("FAA closure fetch failed: %s", exc)
        air = []
    try:
        sea = fetch_nga_areas(get=get)
        if sea:
            sea_source = "live"
    except Exception as exc:
        logger.info("NGA closure fetch failed: %s", exc)
        sea = []
    fallback = fallback_areas()
    if not air:
        air = [area for area in fallback if area["kind"] == "airspace"]
    if not sea:
        sea = [area for area in fallback if area["kind"] == "maritime"]
    areas = air + sea
    if air_source == "live" and sea_source == "live":
        source = "live"
    elif air_source == "fallback" and sea_source == "fallback":
        source = "fallback"
    else:
        source = "mixed"
    if path:
        try:
            _save(path, areas, source, clock)
        except OSError as exc:
            logger.info("Could not cache closure areas: %s", exc)
    with _lock:
        _memory["areas"] = areas
        _memory["source"] = source
        _memory["fetched_at"] = clock
    return areas, source


def load_local(path=CACHE_FILE):
    """Disk cache, or the hardcoded Flight 15 notices. No network."""
    with _lock:
        if _memory["areas"]:
            return _memory["areas"], _memory["source"]
    cached = _load(path) if path else None
    if cached:
        with _lock:
            _memory["areas"] = cached["areas"]
            _memory["source"] = cached.get("source") or "cache"
            return _memory["areas"], _memory["source"]
    areas = fallback_areas()
    with _lock:
        _memory["areas"] = areas
        _memory["source"] = "fallback"
    return areas, "fallback"


def _aware(moment):
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment


def _relevant_launch(launch):
    if not isinstance(launch, dict):
        return False
    if launch.get("rocket") or launch.get("mission") or launch.get("name"):
        from live_window import is_starship_launch
        return is_starship_launch(launch)
    return True


def focus_net(launch_data, now):
    """Starship NET closest to now, ignoring flights older than the 48h cap."""
    if not launch_data:
        return None
    clock = _aware(now)
    best = None
    best_gap = None
    launches = list(launch_data.get("upcoming") or []) + list(launch_data.get("previous") or [])
    for launch in launches:
        if not _relevant_launch(launch):
            continue
        net = _get_parsed_dt(launch.get("net"))
        net = _aware(net)
        if net is None:
            continue
        delta = (clock - net).total_seconds()
        if delta > SHOW_AFTER_CAP_S or delta < -7 * 24 * 3600:
            continue
        gap = abs(delta)
        if best_gap is None or gap < best_gap:
            best = net
            best_gap = gap
    return best


def _mission_over(tracker, now):
    if tracker is None or not getattr(tracker, "liftoff_utc", None):
        return False
    if getattr(tracker, "_past_safety_cap", None) and tracker._past_safety_cap(now):
        return True
    if getattr(tracker, "_all_done", None) and tracker._all_done() and getattr(tracker, "hold_started", False):
        hold = getattr(tracker, "hold_until", None)
        if hold is not None and now >= _aware(hold):
            return True
    return False


def closures_visible(launch_data, tracker, now):
    """True from about T-24h until the tracker says the mission is over."""
    clock = _aware(now)
    if tracker is not None and getattr(tracker, "liftoff_utc", None) and not _mission_over(tracker, clock):
        return True
    if _mission_over(tracker, clock):
        return False
    net = focus_net(launch_data, clock)
    if net is None:
        return False
    delta = (clock - net).total_seconds()
    return -SHOW_BEFORE_S <= delta <= SHOW_AFTER_CAP_S


def globe_payload(launch_data, tracker, now, areas=None, source=None):
    if areas is None:
        areas, source = load_local()
    show = closures_visible(launch_data, tracker, now)
    return {
        "show": bool(show),
        "source": source or "fallback",
        "areas": areas if show else [],
    }


def _unwrap(points):
    lons = [point["lon"] for point in points]
    if not lons or max(lons) - min(lons) <= 180:
        return [(point["lat"], point["lon"]) for point in points]
    return [(point["lat"], point["lon"] if point["lon"] >= 0 else point["lon"] + 360) for point in points]


def _point_in_ring(lat, lon, ring):
    flat = _unwrap(ring)
    if flat and max(point[1] for point in flat) > 180 and lon < 0:
        lon += 360
    inside = False
    previous = len(flat) - 1
    for index, (lat_i, lon_i) in enumerate(flat):
        lat_j, lon_j = flat[previous]
        if ((lat_i > lat) != (lat_j > lat)) and (
            lon < (lon_j - lon_i) * (lat - lat_i) / ((lat_j - lat_i) or 1e-12) + lon_i
        ):
            inside = not inside
        previous = index
    return inside


def point_in_areas(lat, lon, areas):
    for area in areas:
        if _point_in_ring(lat, lon, area.get("points") or []):
            return True
    return False


def _km_from(origin, point):
    dlat = (point["lat"] - origin["lat"]) * 111.0
    dlon = (point["lon"] - origin["lon"]) * 111.0
    return (dlat * dlat + dlon * dlon) ** 0.5


def path_misses_corridor(points, areas):
    """True when the downrange path never enters the launch corridor.

    The pad itself sits in the airspace, so only points well downrange count.
    """
    corridor = [
        area for area in areas
        if area.get("role") in ("airspace", "corridor", "high_risk") and area.get("points")
    ]
    usable = [point for point in points or [] if isinstance(point, dict) and "lat" in point and "lon" in point]
    if len(usable) < 2 or not corridor:
        return False
    downrange = [point for point in usable[1:] if _km_from(usable[0], point) >= 150]
    if not downrange:
        downrange = usable[len(usable) // 2:]
    step = max(1, len(downrange) // 40)
    for point in downrange[::step]:
        if point_in_areas(point["lat"], point["lon"], corridor):
            return False
    return True


def warn_if_path_misses_corridor(points, areas=None):
    """Log once per path shape. Does not change the trajectory."""
    if areas is None:
        areas, _source = load_local()
    if not path_misses_corridor(points, areas):
        return False
    usable = [point for point in points if isinstance(point, dict)]
    key = (
        round(usable[0]["lat"], 2), round(usable[0]["lon"], 2),
        round(usable[-1]["lat"], 2), round(usable[-1]["lon"], 2),
        len(usable),
    )
    if key in _warned_paths:
        return True
    _warned_paths.add(key)
    logger.warning(
        "Planned Starship trajectory does not enter the published closure corridor"
    )
    return True
