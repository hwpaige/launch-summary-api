"""Shape and omission tests for GET /launch/{id}/insights. LL2 is mocked."""

import os
import unittest
from unittest.mock import patch

import app
import launch_insights

app._background_enabled = False

LAUNCH_ID = "11111111-1111-1111-1111-111111111111"
PAST_ID = "22222222-2222-2222-2222-222222222222"
OTHER_ID = "33333333-3333-3333-3333-333333333333"

FLIGHT_KEYS = {
    "date",
    "mission_name",
    "ll2_launch_id",
    "landing_location_abbrev",
    "landing_success",
    "turnaround_days",
    "is_current",
}
EVENT_KEYS = {"name", "t_plus_s", "source"}


def _stage(serial, flight_no, flights, successes, attempts, first, landing, previous=None, turnaround=None):
    stage = {
        "type": "Core",
        "launcher_flight_number": flight_no,
        "launcher": {
            "serial_number": serial,
            "flights": flights,
            "successful_landings": successes,
            "attempted_landings": attempts,
            "first_launch_date": first,
            "is_placeholder": False,
        },
        "landing": landing,
    }
    if previous:
        stage["previous_flight_date"] = previous
    if turnaround:
        stage["turn_around_time"] = turnaround
    return stage


def _asds(success, downrange=574):
    return {
        "attempt": True,
        "success": success,
        "downrange_distance": downrange,
        "landing_location": {
            "name": "Of Course I Still Love You",
            "abbrev": "OCISLY",
            "successful_landings": 230,
            "attempted_landings": 238,
        },
        "type": {"name": "Autonomous Spaceport Drone Ship", "abbrev": "ASDS"},
    }


def _launch(launch_id, name, mission, net, stages, timeline=None, orbit=None):
    record = {
        "id": launch_id,
        "name": name,
        "net": net,
        "mission": {
            "name": mission,
            "orbit": orbit if orbit is not None else {"name": "Low Earth Orbit", "abbrev": "LEO"},
        },
        "rocket": {
            "configuration": {"name": "Falcon 9", "apogee": 200.0},
            "launcher_stage": stages,
        },
    }
    if timeline is not None:
        record["timeline"] = timeline
    return record


def _timeline():
    return [
        {"type": {"abbrev": "Prop Load"}, "relative_time": "-PT35M"},
        {"type": {"abbrev": "Liftoff"}, "relative_time": "P0D"},
        {"type": {"abbrev": "Max-Q"}, "relative_time": "PT1M7S"},
        {"type": {"abbrev": "MECO"}, "relative_time": "PT2M25S"},
        {"type": {"abbrev": "SECO-1"}, "relative_time": "PT8M39S"},
    ]


def _current():
    return _launch(
        LAUNCH_ID,
        "Falcon 9 Block 5 | Starlink Group 15-20",
        "Starlink Group 15-20",
        "2024-02-12T00:00:00Z",
        [_stage("B1100", 3, 3, 2, 2, "2024-01-01T00:00:00Z", _asds(None), previous="2024-01-20T00:00:00Z")],
        timeline=_timeline(),
    )


def _past():
    return _launch(
        PAST_ID,
        "Falcon 9 Block 5 | Starlink Group 15-10",
        "Starlink Group 15-10",
        "2024-01-01T00:00:00Z",
        [_stage("B1100", 1, 1, 1, 1, "2024-01-01T00:00:00Z", _asds(True, 8))],
        timeline=[],
    )


def _middle():
    return _launch(
        OTHER_ID,
        "Falcon 9 Block 5 | Starlink Group 15-15",
        "Starlink Group 15-15",
        "2024-01-22T00:00:00Z",
        [_stage(
            "B1100", 2, 2, 2, 2, "2024-01-01T00:00:00Z", _asds(True),
            previous="2024-01-20T00:00:00Z",
        )],
    )


class LaunchInsightsTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()
        self._redis = app.r
        app.r = None
        launch_insights.LL_HOURLY_BUDGET = 100
        launch_insights.LL_RESERVED_FOR_MAIN_FEED = 0
        launch_insights.reset_state()

    def tearDown(self):
        app.r = self._redis
        launch_insights.LL_HOURLY_BUDGET = int(os.getenv("LL_HOURLY_BUDGET", "15"))
        launch_insights.LL_RESERVED_FOR_MAIN_FEED = int(os.getenv("LL_RESERVED_FOR_MAIN_FEED", "12"))
        launch_insights.reset_state()

    def _router(self, history=None):
        history = history if history is not None else [_past(), _middle(), _current()]

        def fake(url, params, timeout):
            if params.get("serial_number"):
                return 200, {"count": len(history), "results": history}
            return 200, _current()

        return fake

    def test_full_shape_uses_ll2_fields_and_omits_the_ascent_curve(self):
        with patch.object(launch_insights, "_ll_get", side_effect=self._router()), \
             patch.object(app, "get_launch_trajectory_data", side_effect=AssertionError("trajectory")):
            body, hit = launch_insights.build_insights(LAUNCH_ID)

        self.assertFalse(hit)
        self.assertEqual(set(body), {
            "ll2_launch_id", "name", "booster", "events", "orbit", "landing", "cached",
        })
        self.assertNotIn("ascent_profile", body)
        self.assertNotIn("ascent_profile_meta", body)
        self.assertEqual(body["ll2_launch_id"], LAUNCH_ID)
        self.assertFalse(body["cached"])

        booster = body["booster"][0]
        self.assertEqual(booster["serial"], "B1100")
        self.assertEqual(booster["flight_number_this_launch"], 3)
        self.assertEqual(booster["total_flights"], 3)
        self.assertEqual(booster["landing_success_count"], 2)
        self.assertEqual(booster["landing_attempt_count"], 2)
        self.assertEqual(booster["first_flight_date"], "2024-01-01T00:00:00Z")
        self.assertEqual(len(booster["flights"]), 3)
        self.assertEqual(set(booster["flights"][0]), FLIGHT_KEYS)
        self.assertEqual(
            [flight["ll2_launch_id"] for flight in booster["flights"]],
            [PAST_ID, OTHER_ID, LAUNCH_ID],
        )
        self.assertIsNone(booster["flights"][0]["turnaround_days"])
        self.assertEqual(booster["flights"][1]["turnaround_days"], 21)
        self.assertEqual(booster["flights"][2]["turnaround_days"], 21)
        self.assertEqual(
            [flight["is_current"] for flight in booster["flights"]],
            [False, False, True],
        )
        self.assertIsNone(booster["flights"][2]["landing_success"])
        self.assertTrue(booster["flights"][1]["landing_success"])
        self.assertEqual(booster["flights"][0]["landing_location_abbrev"], "OCISLY")
        self.assertEqual(booster["flights"][1]["landing_location_abbrev"], "OCISLY")

        self.assertEqual(
            [(event["name"], event["t_plus_s"], event["source"]) for event in body["events"]],
            [("Liftoff", 0, "ll2"), ("Max-Q", 67, "ll2"), ("MECO", 145, "ll2"), ("SECO-1", 519, "ll2")],
        )
        self.assertTrue(all(set(event) == EVENT_KEYS for event in body["events"]))
        self.assertEqual(body["orbit"], {"name": "Low Earth Orbit"})
        self.assertNotIn("apogee_km", body["orbit"])
        self.assertEqual(body["landing"], {
            "type": "ASDS",
            "location_name": "Of Course I Still Love You",
            "location_abbrev": "OCISLY",
            "location_successful_landings": 230,
            "location_attempted_landings": 238,
            "downrange_km": 574,
        })

    def test_date_gap_wins_over_a_different_turnaround_hint(self):
        middle = _middle()
        middle["rocket"]["launcher_stage"][0]["previous_flight_date"] = "2024-01-20T00:00:00Z"
        with patch.object(launch_insights, "_ll_get", side_effect=self._router([_past(), middle, _current()])):
            body, _hit = launch_insights.build_insights(LAUNCH_ID)
        self.assertEqual(body["booster"][0]["flights"][1]["turnaround_days"], 21)

    def test_omits_sections_that_have_no_real_data(self):
        bare = {
            "id": LAUNCH_ID,
            "name": "Falcon 9 Block 5 | Bare",
            "net": "2024-02-12T00:00:00Z",
            "mission": {},
            "rocket": {
                "configuration": {"name": "Falcon 9", "apogee": 200.0},
                "launcher_stage": [
                    {"type": "Second Stage", "launcher": {"serial_number": "B9999"}},
                    {"type": "Core", "launcher": {"serial_number": "Unknown", "is_placeholder": True}},
                ],
            },
        }

        def fake(url, params, timeout):
            return 200, bare

        with patch.object(launch_insights, "_ll_get", side_effect=fake):
            body, _hit = launch_insights.build_insights(LAUNCH_ID)

        self.assertEqual(set(body), {"ll2_launch_id", "name", "cached"})
        for section in ("booster", "events", "orbit", "landing", "ascent_profile", "ascent_profile_meta"):
            self.assertNotIn(section, body)

    def test_orbit_numbers_are_copied_only_from_the_mission_orbit(self):
        record = _current()
        record["mission"]["orbit"] = {
            "name": "Geostationary Transfer Orbit",
            "abbrev": "GTO",
            "perigee": 185,
            "apogee": 35786,
            "inclination": 27.0,
        }

        def fake(url, params, timeout):
            if params.get("serial_number"):
                return 500, None
            return 200, record

        with patch.object(launch_insights, "_ll_get", side_effect=fake):
            body, _hit = launch_insights.build_insights(LAUNCH_ID)
        self.assertEqual(body["orbit"], {
            "name": "Geostationary Transfer Orbit",
            "apogee_km": 35786,
            "perigee_km": 185,
            "inclination_deg": 27,
        })

    def test_expendable_landing_keeps_only_the_known_type(self):
        record = _current()
        record["rocket"]["launcher_stage"][0]["landing"] = {
            "attempt": False,
            "success": None,
            "type": {"name": "Expended", "abbrev": "EX"},
        }
        record["timeline"] = []

        def fake(url, params, timeout):
            if params.get("serial_number"):
                return 200, {"results": []}
            return 200, record

        with patch.object(launch_insights, "_ll_get", side_effect=fake):
            body, _hit = launch_insights.build_insights(LAUNCH_ID)
        self.assertEqual(body["landing"], {"type": "expendable"})
        self.assertNotIn("events", body)
        self.assertEqual(len(body["booster"][0]["flights"]), 1)
        self.assertTrue(body["booster"][0]["flights"][0]["is_current"])
        self.assertEqual(body["booster"][0]["flights"][0]["turnaround_days"], 23)

    def test_side_store_does_not_call_ll2_when_booster_budget_is_low(self):
        app._local_raw_launches[LAUNCH_ID] = _current()
        launch_insights.LL_HOURLY_BUDGET = 15
        launch_insights.LL_RESERVED_FOR_MAIN_FEED = 14
        with patch.object(launch_insights, "_ll_get", side_effect=AssertionError("LL2")), \
             patch.object(app, "fetch_launches", side_effect=AssertionError("launches")), \
             patch.object(app, "fetch_launch_details", side_effect=AssertionError("details")), \
             patch.object(app, "_load_launch_payload", side_effect=AssertionError("list")):
            body, hit = launch_insights.build_insights(LAUNCH_ID)
        self.assertTrue(hit)
        self.assertTrue(body["cached"])
        self.assertIn("events", body)
        self.assertIn("booster", body)
        self.assertNotIn("flights", body["booster"][0])
        self.assertEqual(body["booster"][0]["serial"], "B1100")

    def test_cache_hit_skips_ll2_and_keeps_separate_ttls(self):
        written = []
        real_set = launch_insights._cache_set

        def spy(key, data, ttl):
            written.append((key, ttl))
            return real_set(key, data, ttl)

        calls = {"n": 0}

        def fake(url, params, timeout):
            calls["n"] += 1
            return self._router()(url, params, timeout)

        with patch.object(launch_insights, "_ll_get", side_effect=fake), \
             patch.object(launch_insights, "_cache_set", side_effect=spy):
            first, first_hit = launch_insights.build_insights(LAUNCH_ID)
            second, second_hit = launch_insights.build_insights(LAUNCH_ID)
        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertEqual(calls["n"], 2)
        self.assertEqual(second["booster"][0]["flights"], first["booster"][0]["flights"])
        self.assertIn((launch_insights._launch_key(LAUNCH_ID), launch_insights.LAUNCH_TTL), written)
        self.assertIn((launch_insights._booster_key("B1100"), launch_insights.BOOSTER_TTL), written)
        self.assertEqual(launch_insights.LAUNCH_TTL, 6 * 3600)
        self.assertEqual(launch_insights.BOOSTER_TTL, 24 * 3600)

    def test_second_launch_reuses_the_booster_cache(self):
        other = "44444444-4444-4444-4444-444444444444"
        other_record = _launch(
            other,
            "Falcon 9 Block 5 | Next",
            "Next",
            "2024-03-01T00:00:00Z",
            [_stage("B1100", 4, 4, 3, 3, "2024-01-01T00:00:00Z", _asds(None))],
            timeline=[{"type": {"abbrev": "Liftoff"}, "relative_time": "PT0S"}],
        )
        calls = {"history": 0, "launch": 0}

        def fake(url, params, timeout):
            if params.get("serial_number"):
                calls["history"] += 1
                return 200, {"results": [_past(), _middle(), _current()]}
            calls["launch"] += 1
            if other in url:
                return 200, other_record
            return 200, _current()

        with patch.object(launch_insights, "_ll_get", side_effect=fake):
            launch_insights.build_insights(LAUNCH_ID)
            body, _hit = launch_insights.build_insights(other)
        self.assertEqual(calls["history"], 1)
        self.assertEqual(calls["launch"], 2)
        current = [flight for flight in body["booster"][0]["flights"] if flight["is_current"]]
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]["ll2_launch_id"], other)

    def test_heavy_cores_are_deduped_and_each_serial_is_fetched_once(self):
        heavy_id = "55555555-5555-5555-5555-555555555555"
        record = _launch(
            heavy_id,
            "Falcon Heavy | Test",
            "Test",
            "2024-04-01T00:00:00Z",
            [
                _stage("B1064", 1, 1, 1, 1, "2024-04-01T00:00:00Z", _asds(True)),
                _stage("B1065", 1, 1, 1, 1, "2024-04-01T00:00:00Z", _asds(True)),
                _stage("B1064", 1, 1, 1, 1, "2024-04-01T00:00:00Z", _asds(True)),
            ],
            timeline=[],
        )
        seen = []

        def fake(url, params, timeout):
            serial = params.get("serial_number")
            if serial:
                seen.append(serial)
                return 200, {"results": []}
            return 200, record

        with patch.object(launch_insights, "_ll_get", side_effect=fake):
            body, _hit = launch_insights.build_insights(heavy_id)
        self.assertEqual(seen, ["B1064", "B1065"])
        self.assertEqual([item["serial"] for item in body["booster"]], ["B1064", "B1065"])

    def test_rate_limit_returns_partial_200_without_retry(self):
        calls = {"n": 0}

        def fake(url, params, timeout):
            calls["n"] += 1
            return 429, None

        with patch.object(launch_insights, "_ll_get", side_effect=fake):
            first, hit = launch_insights.build_insights(LAUNCH_ID)
            second, second_hit = launch_insights.build_insights(LAUNCH_ID)
        self.assertEqual(calls["n"], 1)
        self.assertFalse(hit)
        self.assertTrue(second_hit)
        self.assertEqual(first, {"ll2_launch_id": LAUNCH_ID, "cached": False})
        self.assertEqual(second["ll2_launch_id"], LAUNCH_ID)
        self.assertNotIn("booster", first)

    def test_history_error_omits_flights_and_does_not_retry(self):
        calls = {"n": 0}

        def fake(url, params, timeout):
            calls["n"] += 1
            if params.get("serial_number"):
                return 500, None
            return 200, _current()

        with patch.object(launch_insights, "_ll_get", side_effect=fake):
            body, _hit = launch_insights.build_insights(LAUNCH_ID)
        self.assertEqual(calls["n"], 2)
        self.assertIn("events", body)
        self.assertIn("orbit", body)
        self.assertIn("landing", body)
        self.assertNotIn("flights", body["booster"][0])
        self.assertEqual(body["events"][0]["source"], "ll2")

    def test_low_surplus_skips_the_booster_fetch(self):
        calls = {"n": 0}

        def fake(url, params, timeout):
            calls["n"] += 1
            if params.get("serial_number"):
                raise AssertionError("booster fetch")
            return 200, _current()

        launch_insights.LL_HOURLY_BUDGET = 15
        launch_insights.LL_RESERVED_FOR_MAIN_FEED = 14
        with patch.object(launch_insights, "_ll_get", side_effect=fake):
            body, _hit = launch_insights.build_insights(LAUNCH_ID)
        self.assertEqual(calls["n"], 1)
        self.assertNotIn("flights", body["booster"][0])
        self.assertEqual(body["events"][0]["name"], "Liftoff")

    def test_unusable_history_page_is_not_cached(self):
        calls = {"n": 0}

        def fake(url, params, timeout):
            calls["n"] += 1
            if params.get("serial_number"):
                return 200, {"results": [_launch(
                    PAST_ID, "Other", "Other", "2024-01-01T00:00:00Z",
                    [_stage("B9999", 1, 1, 1, 1, "2024-01-01T00:00:00Z", _asds(True))],
                )]}
            return 200, _current()

        with patch.object(launch_insights, "_ll_get", side_effect=fake):
            first, _hit = launch_insights.build_insights(LAUNCH_ID)
            second, _hit2 = launch_insights.build_insights(LAUNCH_ID)
        self.assertNotIn("flights", first["booster"][0])
        self.assertEqual(calls["n"], 3)

    def test_http_route_returns_the_same_body(self):
        from fastapi.testclient import TestClient

        client = TestClient(app.app)
        with patch.object(launch_insights, "_ll_get", side_effect=self._router()):
            response = client.get(f"/launch/{LAUNCH_ID}/insights")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["booster"][0]["serial"], "B1100")
        self.assertNotIn("ascent_profile", body)
        self.assertEqual(body["events"][-1]["source"], "ll2")


class ExistingShapeRegressionTests(unittest.TestCase):
    """Paths and payloads that existed before /launch/{id}/insights."""

    FROZEN_PATHS = {
        "/",
        "/calendar",
        "/dashboard",
        "/docs",
        "/docs/oauth2-redirect",
        "/external_narratives",
        "/launch_details/{launch_id}",
        "/launch_raw/{launch_id}",
        "/launch_trends",
        "/launches",
        "/launches_slim",
        "/locations",
        "/metrics",
        "/next_launch",
        "/notify/copy",
        "/openapi.json",
        "/recent_launches_narratives",
        "/redoc",
        "/refresh",
        "/reset_metrics",
        "/satellites/deployed",
        "/satellites/gp",
        "/satellites/meta",
        "/satellites/starlink",
        "/seed_history",
        "/seed_status",
        "/starship/closures",
        "/starship/tracker",
        "/stop_seeding",
        "/trajectory",
        "/trajectory/{launch_id}",
        "/upcoming_launches",
        "/user_weather",
        "/weather/{location}",
        "/weather_all",
    }
    TRACKER_KEYS = {
        "show",
        "stale",
        "hold",
        "hide_planned",
        "launch_active",
        "vehicles",
        "telemetry",
        "position_source",
        "live_positions",
        "generated_at",
        "feed",
        "planned",
        "orbit",
    }
    CLOSURE_KEYS = {"show", "source", "areas", "generated_at", "kind", "fallback"}

    def setUp(self):
        app._reset_cache_coordination_for_tests()
        self._redis = app.r
        app.r = None

    def tearDown(self):
        app.r = self._redis

    def test_route_table_only_adds_the_insights_path(self):
        paths = {getattr(route, "path", None) for route in app.app.routes}
        paths.discard(None)
        self.assertEqual(paths - self.FROZEN_PATHS, {"/launch/{ll2_launch_id}/insights"})
        self.assertTrue(self.FROZEN_PATHS <= paths)

    def test_existing_endpoints_keep_their_shapes(self):
        from fastapi.testclient import TestClient

        launch = {
            "id": "shape-cape",
            "name": "Starlink shape pin",
            "mission": "Starlink shape pin",
            "net": "2026-12-01T00:00:00Z",
            "date": "2026-12-01",
            "time": "00:00",
            "status": "Go",
            "rocket": "Falcon 9",
            "orbit": "LEO",
            "pad": "Launch Complex 39A",
            "video_url": "https://example.com/watch",
            "x_video_url": "https://x.com/example",
            "trajectory_data": {
                "launch_site": {"lat": 28.6084, "lon": -80.6043, "name": "Cape Canaveral, FL"},
                "trajectory": [{"lat": 28.6084, "lon": -80.6043, "r": 1.01}],
                "booster_trajectory": [],
                "booster_ground_track": "along_track_return",
                "sep_idx": 0,
                "orbit_path": [],
                "orbit": "LEO",
                "mission": "Starlink shape pin",
                "pad": "Launch Complex 39A",
                "landing_type": "ASDS",
                "landing_location": "A Shortfall of Gravitas",
                "landing_site": None,
                "landing_downrange_km": None,
                "inclination_deg": 28.6,
            },
        }
        payload = {
            "upcoming": [launch],
            "previous": [],
            "last_updated": "2026-11-01T00:00:00Z",
        }
        client = TestClient(app.app)
        with patch("requests.get", side_effect=AssertionError("network")), \
             patch.object(app, "_load_launch_payload", return_value=payload), \
             patch.object(app, "fetch_launches", side_effect=AssertionError("launches")), \
             patch.object(app, "fetch_launch_details", side_effect=AssertionError("details")):
            launches = client.get("/launches")
            slim = client.get("/launches_slim")
            trajectory = client.get("/trajectory")
            tracker = client.get("/starship/tracker")
            closures = client.get("/starship/closures")
            locations = client.get("/locations")
            metrics = client.get("/metrics")
            narratives = client.get("/recent_launches_narratives")
            trends = client.get("/launch_trends")
            nxt = client.get("/next_launch")
            upcoming = client.get("/upcoming_launches")
            calendar = client.get("/calendar")

        self.assertEqual(launches.status_code, 200)
        self.assertEqual(set(launches.json()), {"upcoming", "previous", "last_updated"})
        row = launches.json()["upcoming"][0]
        self.assertNotIn("all_data", row)
        self.assertNotIn("booster", row)
        self.assertNotIn("events", row)
        self.assertNotIn("ascent_profile", row)
        for key in ("id", "name", "mission", "net", "rocket", "orbit", "pad", "trajectory_data"):
            self.assertIn(key, row)
        self.assertEqual(set(row["trajectory_data"]), {
            "launch_site", "trajectory", "booster_trajectory", "booster_ground_track",
            "sep_idx", "orbit_path", "orbit", "mission", "pad", "landing_type",
            "landing_location", "landing_site", "landing_downrange_km", "inclination_deg",
        })
        self.assertEqual(set(slim.json()), {"upcoming", "previous", "last_updated"})
        self.assertEqual(set(trajectory.json()), {"trajectory", "launch_id", "mission"})
        self.assertEqual(trajectory.json()["launch_id"], "shape-cape")
        self.assertEqual(tracker.status_code, 200)
        self.assertEqual(set(tracker.json()), self.TRACKER_KEYS)
        self.assertEqual(tracker.json()["position_source"], "none")
        self.assertEqual(tracker.json()["vehicles"], [])
        self.assertEqual(tracker.json()["planned"]["kind"], "planned")
        self.assertEqual(closures.status_code, 200)
        self.assertEqual(set(closures.json()), self.CLOSURE_KEYS)
        self.assertEqual(closures.json()["kind"], "closure")
        self.assertEqual(set(locations.json()), {"locations"})
        self.assertEqual(set(metrics.json()), {"current", "range_stats", "history", "hits_per_day"})
        self.assertEqual(set(narratives.json()), {"descriptions", "last_updated"})
        self.assertEqual(set(trends.json()), {"trends", "last_updated"})
        self.assertEqual(set(trends.json()["trends"]), {"year", "month", "cumulative", "rolling"})
        self.assertEqual(set(nxt.json()), {"next_launch", "timezone", "last_updated"})
        self.assertEqual(set(upcoming.json()), {"upcoming", "timezone", "last_updated"})
        self.assertEqual(set(calendar.json()), {"calendar", "timezone", "last_updated"})


def _warm_id(index):
    return f"00000000-0000-4000-8000-{index:012d}"


def _warm_net(index):
    return f"2026-12-{index + 1:02d}T00:00:00Z"


def _warm_detail(launch_id, serial, net):
    return {
        "id": launch_id,
        "name": f"Falcon 9 Block 5 | {serial}",
        "net": net,
        "mission": {"name": serial, "orbit": {"name": "Low Earth Orbit", "abbrev": "LEO"}},
        "rocket": {
            "configuration": {"name": "Falcon 9", "apogee": 200},
            "launcher_stage": [{
                "type": "Core",
                "launcher_flight_number": 2,
                "launcher": {
                    "serial_number": serial,
                    "flights": 2,
                    "successful_landings": 1,
                    "attempted_landings": 1,
                    "first_launch_date": "2024-01-01T00:00:00Z",
                },
                "landing": {
                    "attempt": True,
                    "success": None,
                    "downrange_distance": 600,
                    "type": {"name": "Autonomous Spaceport Drone Ship", "abbrev": "ASDS"},
                    "landing_location": {
                        "name": "A Shortfall of Gravitas",
                        "abbrev": "ASOG",
                        "successful_landings": 10,
                        "attempted_landings": 10,
                    },
                },
            }],
        },
        "timeline": [{"type": {"abbrev": "Liftoff"}, "relative_time": "PT0S"}],
    }


def _store_upcoming(launches):
    app._launches_mem["data"] = {
        "upcoming": launches,
        "previous": [],
        "last_updated": "2026-11-17T00:00:00Z",
    }
    app._launches_mem["at"] = 1_000_000_000


class _LockRedis:
    """Just enough Redis for the warm lock and the shared insights budget."""

    def __init__(self):
        self.store = {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    def get(self, key):
        value = self.store.get(key)
        return None if value is None else str(value)

    def delete(self, key):
        self.store.pop(key, None)
        return 1

    def incr(self, key):
        self.store[key] = int(self.store.get(key) or 0) + 1
        return self.store[key]

    def decr(self, key):
        self.store[key] = int(self.store.get(key) or 0) - 1
        return self.store[key]

    def expire(self, key, ttl):
        return True

    def setex(self, key, ttl, value):
        self.store[key] = value
        return True


class InsightsWarmTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()
        self._redis = app.r
        app.r = None
        self._warm_env = os.environ.get("INSIGHTS_WARM")
        os.environ["INSIGHTS_WARM"] = "1"
        launch_insights.LL_HOURLY_BUDGET = 15
        launch_insights.LL_RESERVED_FOR_MAIN_FEED = 12
        launch_insights.reset_state()

    def tearDown(self):
        app.r = self._redis
        if self._warm_env is None:
            os.environ.pop("INSIGHTS_WARM", None)
        else:
            os.environ["INSIGHTS_WARM"] = self._warm_env
        launch_insights.LL_HOURLY_BUDGET = int(os.getenv("LL_HOURLY_BUDGET", "15"))
        launch_insights.LL_RESERVED_FOR_MAIN_FEED = int(os.getenv("LL_RESERVED_FOR_MAIN_FEED", "12"))
        launch_insights.reset_state()

    def _upcoming(self, count):
        """Stored newest-first so a warmer that skips the sort warms the wrong launch."""
        launches = []
        details = {}
        soonest_first = []
        for index in range(count):
            launch_id = _warm_id(index)
            net = _warm_net(index)
            serial = f"B{index:04d}"
            soonest_first.append(launch_id)
            launches.append({
                "id": launch_id,
                "name": f"Falcon 9 | {serial}",
                "mission": serial,
                "net": net,
                "rocket": "Falcon 9",
                "orbit": "LEO",
            })
            details[launch_id] = _warm_detail(launch_id, serial, net)
        launches.reverse()
        _store_upcoming(launches)
        return soonest_first, details

    def _router(self, details, calls):
        def fake(url, params, timeout):
            calls.append((url, params.get("serial_number")))
            serial = params.get("serial_number")
            if serial:
                return 200, {"count": 0, "results": []}
            for launch_id, record in details.items():
                if launch_id in url:
                    return 200, record
            return 404, None

        return fake

    def test_warmer_spends_only_the_surplus_and_starts_with_the_soonest(self):
        soonest_first, details = self._upcoming(12)
        calls = []
        with patch.object(launch_insights, "_ll_get", side_effect=self._router(details, calls)), \
             patch.object(app, "fetch_launches", side_effect=AssertionError("list refresh")), \
             patch.object(app, "_load_launch_payload", side_effect=AssertionError("list load")):
            result = launch_insights.warm_upcoming_insights()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["considered"], 12)
        self.assertEqual(launch_insights._allowance(), 3)
        self.assertEqual(len(calls), 3)
        self.assertLess(len(calls), 12)
        detail_ids = [launch_id for url, serial in calls if not serial for launch_id in soonest_first if launch_id in url]
        self.assertEqual(detail_ids, soonest_first[:2])
        self.assertEqual(calls[1][1], "B0000")
        for launch_id in soonest_first[2:]:
            self.assertFalse(any(launch_id in url for url, _serial in calls))
        self.assertLessEqual(launch_insights._budget.remaining() + len(calls), launch_insights._allowance())
        self.assertEqual(launch_insights._budget.remaining(), 0)

    def test_warmer_covers_twelve_and_leaves_the_thirteenth(self):
        soonest_first, details = self._upcoming(13)
        launch_insights.LL_HOURLY_BUDGET = 100
        launch_insights.LL_RESERVED_FOR_MAIN_FEED = 0
        calls = []
        with patch.object(launch_insights, "_ll_get", side_effect=self._router(details, calls)):
            result = launch_insights.warm_upcoming_insights()
        detail_ids = [launch_id for url, serial in calls if not serial for launch_id in soonest_first if launch_id in url]
        self.assertEqual(detail_ids, soonest_first[:12])
        self.assertEqual(result["considered"], 12)
        self.assertNotIn(soonest_first[12], detail_ids)
        self.assertFalse(any(soonest_first[12] in url for url, _serial in calls))

    def test_repeated_serial_reuses_the_booster_cache(self):
        first, second = _warm_id(0), _warm_id(1)
        _store_upcoming([
            {"id": second, "name": "Later", "mission": "Later", "net": _warm_net(1), "rocket": "Falcon 9"},
            {"id": first, "name": "Sooner", "mission": "Sooner", "net": _warm_net(0), "rocket": "Falcon 9"},
        ])
        app._local_raw_launches[first] = _warm_detail(first, "B1067", _warm_net(0))
        app._local_raw_launches[second] = _warm_detail(second, "B1067", _warm_net(1))
        launch_insights.LL_HOURLY_BUDGET = 15
        launch_insights.LL_RESERVED_FOR_MAIN_FEED = 0
        calls = []

        def fake(url, params, timeout):
            calls.append(params.get("serial_number"))
            if params.get("serial_number"):
                return 200, {"results": []}
            raise AssertionError("detail fetch")

        with patch.object(launch_insights, "_ll_get", side_effect=fake):
            launch_insights.warm_upcoming_insights()
        self.assertEqual(calls, ["B1067"])

    def test_fresh_launch_is_skipped(self):
        soonest_first, details = self._upcoming(2)
        launch_id = soonest_first[0]
        record = details[launch_id]
        doc = launch_insights._sections_from_record(record, launch_id)
        doc["_detailed"] = True
        launch_insights._cache_set(launch_insights._launch_key(launch_id), doc, launch_insights.LAUNCH_TTL)
        launch_insights._cache_set(
            launch_insights._booster_key("B0000"),
            {"flights": []},
            launch_insights.BOOSTER_TTL,
        )
        calls = []
        with patch.object(launch_insights, "_ll_get", side_effect=self._router(details, calls)):
            result = launch_insights.warm_upcoming_insights()
        self.assertGreaterEqual(result["skipped"], 1)
        self.assertFalse(any(launch_id in url for url, _serial in calls))
        self.assertTrue(any(soonest_first[1] in url for url, _serial in calls))

    def test_disabled_flag_makes_no_upstream_calls(self):
        self._upcoming(4)
        os.environ["INSIGHTS_WARM"] = "0"
        with patch.object(launch_insights, "_ll_get", side_effect=AssertionError("LL2")):
            result = launch_insights.warm_upcoming_insights()
        self.assertEqual(result["status"], "disabled")
        self.assertEqual(result["warmed"], 0)

    def test_other_worker_holding_the_lock_does_not_fetch(self):
        self._upcoming(4)
        fake = _LockRedis()
        fake.store[launch_insights.WARM_LOCK_KEY] = "other-dyno"
        app.r = fake
        with patch.object(launch_insights, "_ll_get", side_effect=AssertionError("LL2")):
            result = launch_insights.warm_upcoming_insights()
        self.assertEqual(result["status"], "busy")
        self.assertEqual(fake.store[launch_insights.WARM_LOCK_KEY], "other-dyno")

    def test_finished_pass_releases_the_lock(self):
        self._upcoming(1)
        fake = _LockRedis()
        app.r = fake
        launch_insights.LL_HOURLY_BUDGET = 15
        launch_insights.LL_RESERVED_FOR_MAIN_FEED = 12
        with patch.object(launch_insights, "_ll_get", return_value=(429, None)):
            first = launch_insights.warm_upcoming_insights()
            second = launch_insights.warm_upcoming_insights()
        self.assertEqual(first["status"], "ok")
        self.assertNotIn(launch_insights.WARM_LOCK_KEY, fake.store)
        self.assertEqual(second["status"], "ok")


if __name__ == "__main__":
    unittest.main()
