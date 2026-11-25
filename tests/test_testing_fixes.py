"""Two faults found while testing the live server on 2026-11-18."""
import time
import unittest
from unittest.mock import MagicMock, patch

import app


def _response(status=200, payload=None, headers=None):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = payload
    response.headers = headers or {}
    response.text = ""
    return response


AUSTIN = [
    {"icaoId": "KHYI", "lat": 29.8958, "lon": -97.8664},
    {"icaoId": "KAUS", "lat": 30.1831, "lon": -97.6806},
    {"icaoId": "KATT", "lat": 30.3208, "lon": -97.7604},
    {"icaoId": "KATT", "lat": 30.3208, "lon": -97.7604},
    {"icaoId": "KGTU", "lat": 30.6809, "lon": -97.6767},
    {"icaoId": None, "lat": 30.0, "lon": -97.0},
    {"icaoId": "KBAD", "lat": None, "lon": -97.0},
]


class NearestStationTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()

    def test_stations_are_searched_by_bounding_box_and_ordered_by_distance(self):
        with patch.object(app.requests, "get", return_value=_response(payload=AUSTIN)) as get:
            stations = app.nearest_metar_stations(30.2672, -97.7431)
        self.assertEqual(stations, ["KATT", "KAUS", "KHYI"])
        url = get.call_args[0][0]
        self.assertIn("bbox=", url)
        self.assertIn("format=json", url)
        self.assertNotIn("radialDistance", url)

    def test_a_failed_search_is_an_empty_list(self):
        with patch.object(app.requests, "get", return_value=_response(status=400)):
            self.assertEqual(app.nearest_metar_stations(30.27, -97.74), [])
        with patch.object(app.requests, "get", side_effect=RuntimeError("timeout")):
            self.assertEqual(app.nearest_metar_stations(30.27, -97.74), [])

    def test_a_location_with_no_station_gets_modeled_weather_not_brownsville(self):
        forecast = {"current_weather": {"temperature": 12.5, "windspeed": 10, "winddirection": 180}}
        asked = []

        def get(url, **_kwargs):
            asked.append(url)
            return _response(payload=[])

        with patch.object(app.requests, "get", side_effect=get), \
             patch.object(app, "fetch_forecast", return_value=forecast):
            weather = app.fetch_weather(lat=47.0, lon=-110.0)
        self.assertEqual(weather["temperature_c"], 12.5)
        self.assertEqual(weather["source"], "Open-Meteo")
        self.assertFalse(any("KBRO" in url for url in asked))

    def test_starbase_by_name_still_uses_its_stations(self):
        asked = []

        def get(url, **_kwargs):
            asked.append(url)
            return _response(status=500)

        with patch.object(app.requests, "get", side_effect=get), \
             patch.object(app, "fetch_forecast", return_value=None):
            app.fetch_weather(location="Starbase")
        self.assertTrue(any("KBRO" in url for url in asked))


class LaunchLibraryBackoffTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()

    def tearDown(self):
        app._reset_cache_coordination_for_tests()

    def test_a_429_stops_single_launch_fetches_for_a_while(self):
        with patch.object(app.requests, "get", return_value=_response(status=429)) as get:
            self.assertIsNone(app.fetch_launch_details("u1"))
            self.assertIsNone(app.fetch_launch_details("u1"))
            self.assertIsNone(app.fetch_launch_details("u2"))
        self.assertEqual(get.call_count, 1)
        remaining = app._ll_backoff["until"] - time.time()
        self.assertGreater(remaining, app.LL_RATE_LIMIT_BACKOFF_SEC - 5)

    def test_retry_after_is_honoured_within_bounds(self):
        with patch.object(app.requests, "get", return_value=_response(status=429, headers={"Retry-After": "120"})):
            app.fetch_launch_details("u1")
        self.assertAlmostEqual(app._ll_backoff["until"] - time.time(), 120, delta=5)
        app._ll_backoff["until"] = 0.0
        with patch.object(app.requests, "get", return_value=_response(status=429, headers={"Retry-After": "86400"})):
            app.fetch_launch_details("u1")
        self.assertLessEqual(app._ll_backoff["until"] - time.time(), app.LL_RATE_LIMIT_BACKOFF_MAX_SEC + 1)

    def test_fetches_resume_after_the_backoff(self):
        app._ll_backoff["until"] = time.time() - 1
        with patch.object(app.requests, "get", return_value=_response(payload={"id": "u1"})) as get:
            self.assertEqual(app.fetch_launch_details("u1"), {"id": "u1"})
        self.assertEqual(get.call_count, 1)

    def test_the_hot_path_serves_its_stored_record_while_limited(self):
        record = {"id": "u1", "status": {"name": "Go"}}
        app._remember_raw_launch("u1", record)
        app._ll_backoff["until"] = time.time() + 60
        with patch.object(app, "r", None), patch.object(app.requests, "get") as get:
            app._local_raw_launches["u1"] = record
            self.assertEqual(app._fetch_and_store_hot_raw("u1"), record)
        get.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class LaunchLibraryBudgetTests(unittest.TestCase):
    """The account is allowed 45 requests an hour; the list refresh needs 12 of them."""

    def setUp(self):
        app._reset_cache_coordination_for_tests()

    def tearDown(self):
        app._reset_cache_coordination_for_tests()

    def test_single_launch_reads_stop_at_the_hourly_budget(self):
        with patch.object(app.requests, "get", return_value=_response(payload={"id": "u1"})) as get:
            for _ in range(app.LL_SINGLE_LAUNCH_BUDGET_PER_HOUR):
                self.assertEqual(app.fetch_launch_details("u1"), {"id": "u1"})
            self.assertIsNone(app.fetch_launch_details("u1"))
            self.assertIsNone(app.fetch_launch_details("u2"))
        self.assertEqual(get.call_count, app.LL_SINGLE_LAUNCH_BUDGET_PER_HOUR)

    def test_budget_frees_up_as_the_hour_rolls(self):
        start = 1_000_000.0
        for i in range(app.LL_SINGLE_LAUNCH_BUDGET_PER_HOUR):
            self.assertTrue(app._ll_single_launch_allowed(now=start + i))
        self.assertFalse(app._ll_single_launch_allowed(now=start + 100))
        self.assertTrue(app._ll_single_launch_allowed(now=start + 3601))

    def test_one_launch_and_the_list_refresh_fit_inside_the_limit(self):
        near = (app.HOT_RAW_NEAR_BEFORE_SEC + app.HOT_RAW_NEAR_AFTER_SEC) / app.HOT_RAW_TTL_NEAR
        far = (3600 - app.HOT_RAW_NEAR_BEFORE_SEC - app.HOT_RAW_NEAR_AFTER_SEC) / app.HOT_RAW_TTL
        self.assertLessEqual(near + far, app.LL_SINGLE_LAUNCH_BUDGET_PER_HOUR)
        self.assertLessEqual(app.LL_SINGLE_LAUNCH_BUDGET_PER_HOUR + 12, 45)

    def test_fresh_copy_is_reused_longer_away_from_liftoff(self):
        now = 1_800_000_000.0
        iso = lambda offset: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now + offset))
        self.assertEqual(app._hot_raw_ttl({"net": iso(40 * 60)}, now=now), app.HOT_RAW_TTL)
        self.assertEqual(app._hot_raw_ttl({"net": iso(10 * 60)}, now=now), app.HOT_RAW_TTL_NEAR)
        self.assertEqual(app._hot_raw_ttl({"net": iso(-5 * 60)}, now=now), app.HOT_RAW_TTL_NEAR)
        self.assertEqual(app._hot_raw_ttl({"net": iso(-20 * 60)}, now=now), app.HOT_RAW_TTL)
        self.assertEqual(app._hot_raw_ttl({"net": "not a date"}, now=now), app.HOT_RAW_TTL)
        self.assertEqual(app._hot_raw_ttl(None, now=now), app.HOT_RAW_TTL)
