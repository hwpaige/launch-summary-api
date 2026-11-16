"""Closure polygons parsed from the Flight 15 notices, and when they show."""

from datetime import datetime, timedelta, timezone

from closure_areas import (
    SHOW_BEFORE_S,
    closures_visible,
    fallback_areas,
    path_misses_corridor,
    point_in_areas,
    rings_from_notice,
)

NOW = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)
NET = datetime(2026, 11, 13, 22, 45, tzinfo=timezone.utc)


def _launch(net):
    return {"upcoming": [{"net": net.isoformat(), "rocket": "Starship"}], "previous": []}


def test_real_notice_text_parses_the_flight_15_rings():
    areas = fallback_areas()
    by_id = {}
    for area in areas:
        by_id.setdefault(area["notice_id"], []).append(area)

    air = by_id["FDC 6/7129"]
    assert len(air) == 1
    assert air[0]["kind"] == "airspace"
    assert len(air[0]["points"]) == 6
    assert air[0]["effective_start"].startswith("2026-11-13T22:30")
    assert "FDC 6/7130" in air[0]["also"]

    corridor = by_id["NAVAREA IV 1080/26"]
    assert [area["role"] for area in corridor] == ["corridor", "high_risk"]
    assert len(corridor[0]["points"]) == 30
    assert len(corridor[1]["points"]) == 16

    debris = by_id["NAVAREA IV 1081/26"][0]
    assert debris["kind"] == "maritime"
    assert len(debris["points"]) == 15

    pacific = by_id["HYDROPAC 3256/26"][0]
    assert pacific["points"][0]["lon"] > 0
    assert any(point["lon"] < 0 for point in pacific["points"])

    html = (
        "From: 25&#xBA;56'00\"N 97&#xBA;10'00\"W To: 26&#xBA;01'00\"N 97&#xBA;12'00\"W "
        "To: 26&#xBA;04'00\"N 97&#xBA;07'00\"W To: 26&#xBA;06'00\"N 96&#xBA;55'00\"W "
        "To: 26&#xBA;00'00\"N 96&#xBA;55'00\"W To: 26&#xBA;00'00\"N 97&#xBA;00'00\"W"
    )
    parsed = rings_from_notice(html)
    assert len(parsed) == 1
    assert abs(parsed[0][0]["lat"] - air[0]["points"][0]["lat"]) < 1e-6


def test_starbase_is_inside_the_launch_airspace():
    areas = fallback_areas()
    air = [area for area in areas if area["kind"] == "airspace"]
    assert point_in_areas(25.997, -97.15, air)
    assert not point_in_areas(30.0, -97.15, air)


def test_closures_show_from_t_minus_24h_until_the_mission_ends():
    early = NET - timedelta(seconds=SHOW_BEFORE_S + 60)
    assert closures_visible(_launch(NET), None, early) is False
    inside = NET - timedelta(hours=12)
    assert closures_visible(_launch(NET), None, inside) is True

    class Mission:
        liftoff_utc = NET
        hold_started = False
        hold_until = None

        def _past_safety_cap(self, now):
            return False

        def _all_done(self):
            return False

    flying = Mission()
    assert closures_visible(_launch(NET - timedelta(days=2)), flying, NOW) is True

    class Finished(Mission):
        hold_started = True
        hold_until = NOW - timedelta(minutes=1)

        def _all_done(self):
            return True

    assert closures_visible(_launch(NET), Finished(), NOW) is False

    scrubbed = NET + timedelta(days=3)
    assert closures_visible(_launch(scrubbed), None, NOW) is False


def test_planned_path_is_checked_against_the_corridor_and_not_rewritten():
    areas = fallback_areas()
    east = [
        {"lat": 25.99, "lon": -97.16},
        {"lat": 25.9, "lon": -90.0},
        {"lat": 22.0, "lon": -86.0},
    ]
    north = [
        {"lat": 25.99, "lon": -97.16},
        {"lat": 40.0, "lon": -97.16},
        {"lat": 48.0, "lon": -90.0},
    ]
    assert path_misses_corridor(east, areas) is False
    assert path_misses_corridor(north, areas) is True
    assert north[0]["lat"] == 25.99
