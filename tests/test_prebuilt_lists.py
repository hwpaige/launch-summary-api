"""The big list responses are serialized once per version of their data."""
import gzip
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import app


def _launch(launch_id, name, **extra):
    data = {
        "id": launch_id,
        "name": name,
        "mission": name,
        "net": "2026-12-01T00:00:00Z",
        "status": "Go",
        "rocket": "Falcon 9",
        "orbit": "LEO",
        "pad": "SLC-40",
        "all_data": {"id": launch_id, "raw": True},
    }
    data.update(extra)
    return data


def _launches(stamp="2026-11-18T02:46:35Z", name="Next é"):
    return {
        "upcoming": [
            _launch("u1", name, trajectory_data={"trajectory": [{"lat": 1.5, "lon": 2.5}] * 50}),
            _launch("u2", "Later", trajectory_data={"trajectory": [{"lat": 9.0, "lon": 9.0}]}),
        ],
        "previous": [_launch("p1", "Past")],
        "last_updated": stamp,
    }


class PrebuiltLaunchListTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()
        self._redis = app.r
        app.r = None
        self.client = TestClient(app.app)

    def tearDown(self):
        app.r = self._redis
        app._reset_cache_coordination_for_tests()

    def _serving(self, data):
        return patch.object(app, "get_cached_data", return_value=data), \
            patch.object(app, "set_cached_data", return_value=True), \
            patch.object(app, "save_cache_to_file")

    def test_the_http_response_is_the_same_json_the_function_returns(self):
        a, b, c = self._serving(_launches())
        with a, b, c:
            over_http = self.client.get("/launches_slim")
            direct = app.get_launches_slim(internal=True)
            default = self.client.get("/launches")
        self.assertEqual(over_http.status_code, 200)
        self.assertEqual(over_http.headers["content-type"], "application/json")
        self.assertEqual(over_http.json(), direct)
        self.assertEqual(default.json(), direct)
        self.assertIn("trajectory_data", over_http.json()["upcoming"][0])
        self.assertNotIn("trajectory_data", over_http.json()["upcoming"][1])
        self.assertNotIn("all_data", over_http.json()["upcoming"][0])
        self.assertEqual(over_http.json()["upcoming"][0]["mission"], "Next é")

    def test_repeat_requests_do_not_rebuild_the_response(self):
        a, b, c = self._serving(_launches())
        with a, b, c, patch.object(app, "_slim_launch_payload", wraps=app._slim_launch_payload) as build:
            first = self.client.get("/launches_slim")
            for _ in range(25):
                again = self.client.get("/launches_slim")
            self.client.get("/launches")
        self.assertEqual(build.call_count, 1)
        self.assertEqual(again.content, first.content)

    def test_gzip_only_for_clients_that_accept_it(self):
        a, b, c = self._serving(_launches())
        with a, b, c:
            plain = self.client.get("/launches_slim", headers={"Accept-Encoding": "identity"})
        self.assertNotIn("content-encoding", plain.headers)
        expected = json.loads(plain.content)
        _version, body, zipped = app._prebuilt_json["launches_slim"]
        self.assertEqual(body, plain.content)
        self.assertEqual(json.loads(gzip.decompress(zipped)), expected)
        with a, b, c:
            compressed = self.client.get("/launches_slim", headers={"Accept-Encoding": "gzip"})
        self.assertEqual(compressed.headers["content-encoding"], "gzip")
        self.assertEqual(compressed.headers["vary"], "Accept-Encoding")
        self.assertEqual(compressed.json(), expected)

    def test_new_data_is_served_at_once(self):
        a, b, c = self._serving(_launches())
        with a, b, c:
            old = self.client.get("/launches_slim").json()
        app._launches_mem["at"] = 0.0
        a, b, c = self._serving(_launches(stamp="2026-11-18T02:56:35Z", name="Rescheduled"))
        with a, b, c:
            new = self.client.get("/launches_slim").json()
        self.assertEqual(old["upcoming"][0]["mission"], "Next é")
        self.assertEqual(new["upcoming"][0]["mission"], "Rescheduled")
        self.assertEqual(new["last_updated"], "2026-11-18T02:56:35Z")

    def test_a_change_made_in_place_shows_within_the_max_age(self):
        a, b, c = self._serving(_launches())
        with a, b, c:
            self.client.get("/launches_slim")
            app._launches_mem["data"]["upcoming"][0]["status"] = "Hold"
            now = app.time.time()
            with patch.object(app.time, "time", return_value=now + app.PREBUILT_JSON_MAX_AGE_SEC + 1):
                app._launches_mem["at"] = now + app.PREBUILT_JSON_MAX_AGE_SEC + 1
                later = self.client.get("/launches_slim").json()
        self.assertEqual(later["upcoming"][0]["status"], "Hold")

    def test_full_and_forced_requests_are_not_prebuilt(self):
        a, b, c = self._serving(_launches())
        with a, b, c, patch.object(app, "_get_raw_launch", return_value={"raw": True}):
            full = self.client.get("/launches", params={"full": "true"})
        self.assertEqual(full.status_code, 200)
        self.assertIn("all_data", full.json()["upcoming"][0])
        self.assertNotIn("launches_slim", app._prebuilt_json)

    def test_unavailable_is_still_a_503(self):
        with patch.object(app, "get_cached_data", side_effect=ConnectionError("down")):
            self.assertEqual(self.client.get("/launches_slim").status_code, 503)


class PrebuiltWeatherTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()
        self._redis = app.r
        app.r = None
        self.client = TestClient(app.app)

    def tearDown(self):
        app.r = self._redis
        app._reset_cache_coordination_for_tests()

    def test_site_weather_is_assembled_once_per_window(self):
        payload = {"weather": {"Cape": {"temperature_c": 20.0, "last_updated": "2026-11-18T02:54:11Z"}}, "last_updated": "2026-11-18T02:54:11Z"}
        with patch.object(app, "_load_weather_all", return_value=payload) as load:
            first = self.client.get("/weather_all")
            for _ in range(10):
                again = self.client.get("/weather_all")
            direct = app.get_all_weather(internal=True)
        self.assertEqual(first.json(), payload)
        self.assertEqual(again.json(), payload)
        self.assertEqual(direct, payload)
        self.assertEqual(load.call_count, 2)  # one for HTTP, one for the direct call

    def test_a_forced_refresh_is_not_served_from_the_prebuilt_bytes(self):
        stale = {"weather": {"Cape": {"temperature_c": 20.0, "last_updated": "x"}}, "last_updated": "x"}
        fresh = {"weather": {"Cape": {"temperature_c": 25.0, "last_updated": "y"}}, "last_updated": "y"}
        with patch.object(app, "_load_weather_all", return_value=stale), \
             patch.object(app, "refresh_weather_internal", return_value=fresh):
            self.client.get("/weather_all")
            forced = self.client.get("/weather_all", params={"force": "true"})
        self.assertEqual(forced.json()["weather"]["Cape"]["temperature_c"], 25.0)


if __name__ == "__main__":
    unittest.main()
