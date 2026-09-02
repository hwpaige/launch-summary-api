import re
import threading
import time
import unittest
from unittest.mock import Mock, patch

import app

app._background_enabled = False


def is_utc_iso8601(value):
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value))


def _sample_launch(launch_id, name, with_raw=True, with_traj=False):
    launch = {
        "id": launch_id,
        "name": name,
        "mission": name,
        "net": "2026-09-02T14:21:21Z",
    }
    if with_raw:
        launch["all_data"] = {"id": launch_id, "rocket": {"huge": "blob"}, "pad": {"more": "blob"}}
    if with_traj:
        launch["trajectory_data"] = {"orbit_path": [{"lat": 1, "lon": 2, "r": 1.1}] * 10}
    return launch


class AppRegressionTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()

    def test_parse_launch_data_prefers_image_url_and_sets_name_fields(self):
        launch = {
            "id": "launch-1",
            "name": "Test Mission",
            "net": "2026-09-02T14:21:21.428527+00:00",
            "window_start": "2026-09-02T14:20:00.123456+00:00",
            "window_end": "2026-09-02T15:20:00.654321+00:00",
            "status": {"name": "Go", "id": 1},
            "rocket": {
                "configuration": {"name": "Falcon 9"},
                "launcher_stage": [],
            },
            "mission": {"orbit": {"name": "LEO"}},
            "pad": {"name": "LC-39A"},
            "image": {
                "image_url": "https://example.com/image.jpg",
                "url": "https://example.com/url.jpg",
                "thumbnail_url": "https://example.com/thumb.jpg",
            },
        }

        parsed = app.parse_launch_data(launch)

        self.assertEqual(parsed["image"], "https://example.com/image.jpg")
        self.assertEqual(parsed["name"], "Test Mission")
        self.assertEqual(parsed["mission"], "Test Mission")
        self.assertTrue(is_utc_iso8601(parsed["net"]))
        self.assertTrue(is_utc_iso8601(parsed["window_start"]))
        self.assertTrue(is_utc_iso8601(parsed["window_end"]))
        self.assertIn("all_data", parsed)
        self.assertEqual(parsed["all_data"]["id"], "launch-1")

    def test_generate_narratives_keeps_full_existing_history(self):
        fake_response = Mock()
        fake_response.status_code = 200
        fake_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": 'launch_descriptions = ["9/2 1421: Fresh narrative"]'
                    }
                }
            ]
        }
        fake_launches = Mock()
        fake_launches.status_code = 200
        fake_launches.json.return_value = {
            "results": [
                {
                    "net": "2026-09-02T14:21:21Z",
                    "name": "Test Mission",
                    "pad": {"name": "LC-39A"},
                    "rocket": {"configuration": {"name": "Falcon 9"}},
                    "mission": {"orbit": {"name": "LEO"}},
                    "status": {"name": "Go"},
                }
            ]
        }

        existing = [f"9/1 1200: old narrative {idx}" for idx in range(40)]

        with patch.object(app.requests, "get", return_value=fake_launches), \
             patch.object(app.requests, "post", return_value=fake_response):
            result = app.generate_narratives(existing_narratives=existing)

        self.assertEqual(len(result), 41)
        self.assertEqual(result[0], "9/2 1421: Fresh narrative")
        self.assertEqual(result[1:], existing)

    def test_refresh_weather_internal_keeps_forecast_and_float_values(self):
        weather_payload = {
            "temperature_c": 31,
            "temperature_f": 87,
            "humidity": 50,
            "wind_speed_kts": 12,
            "wind_gust_kts": 15,
            "wind_direction": 180,
            "timestamp": "2026-09-02T14:21:21.428527+00:00",
        }
        forecast_payload = {
            "current_weather": {
                "temperature": 31,
                "windspeed": 12,
                "winddirection": 180,
            },
            "daily": {
                "temperature_2m_max": [31, 32],
                "temperature_2m_min": [21, 22],
                "weathercode": [1, 2],
            },
        }

        with patch.object(app, "r", None), \
             patch.object(app, "WEATHER_LOCATIONS", ["Cape"]), \
             patch.object(app, "fetch_weather", return_value=dict(weather_payload)), \
             patch.object(app, "fetch_forecast", return_value=forecast_payload):
            result = app.refresh_weather_internal()

        weather = result["weather"]["Cape"]
        self.assertTrue(is_utc_iso8601(weather["last_updated"]))
        self.assertEqual(weather["last_updated"], result["last_updated"])
        self.assertEqual(weather["temperature_c"], 31.0)
        self.assertEqual(weather["humidity"], 50.0)
        self.assertEqual(weather["forecast"]["current_weather"]["temperature"], 31.0)
        self.assertEqual(weather["forecast"]["daily"]["temperature_2m_max"][0], 31.0)
        self.assertTrue(is_utc_iso8601(weather["timestamp"]))

    def test_refresh_cache_uses_utc_timestamp(self):
        with patch.object(app, "refresh_narratives_internal", return_value=["a narrative"]):
            response = app.refresh_cache()

        self.assertEqual(response["count"], 1)
        self.assertTrue(is_utc_iso8601(response["timestamp"]))

    def test_launches_default_keeps_all_data_compat(self):
        payload = {
            "upcoming": [
                _sample_launch("u1", "Next", with_traj=True),
                _sample_launch("u2", "Later", with_traj=True),
            ],
            "previous": [_sample_launch("p1", "Past", with_traj=True)],
            "last_updated": "2026-09-02T14:21:21Z",
        }
        persisted = []

        def fake_persist(key, data, ttl=None):
            persisted.append((key, data))
            return True

        with patch.object(app, "get_cached_data", return_value=payload), \
             patch.object(app, "set_cached_data", side_effect=fake_persist), \
             patch.object(app, "r", None):
            result = app.get_launches(force=False, include_raw=False, internal=True)
            slim = app.get_launches(slim=True, internal=True)
            full = app.get_launches(full=True, internal=True)

        self.assertIn("all_data", result["upcoming"][0])
        self.assertEqual(result["upcoming"][0]["all_data"]["id"], "u1")
        self.assertIn("all_data", result["previous"][0])
        self.assertIn("all_data", full["upcoming"][0])
        self.assertNotIn("all_data", slim["upcoming"][0])
        self.assertNotIn("all_data", slim["previous"][0])
        self.assertIn("trajectory_data", result["upcoming"][0])
        self.assertNotIn("trajectory_data", result["upcoming"][1])
        self.assertNotIn("trajectory_data", result["previous"][0])
        list_persists = [item for item in persisted if item[0] == app.LAUNCHES_CACHE_KEY]
        self.assertTrue(list_persists)
        self.assertNotIn("all_data", list_persists[0][1]["upcoming"][0])
        self.assertIsNone(app._launches_mem["data"]["upcoming"][0].get("all_data"))

    def test_launches_slim_omits_all_data(self):
        payload = {
            "upcoming": [_sample_launch("u1", "Next", with_traj=True)],
            "previous": [_sample_launch("p1", "Past")],
            "last_updated": "2026-09-02T14:21:21Z",
        }
        with patch.object(app, "get_cached_data", return_value=payload), \
             patch.object(app, "set_cached_data", return_value=True), \
             patch.object(app, "r", None):
            default = app.get_launches(internal=True)
            slim = app.get_launches_slim(internal=True)

        self.assertEqual(default["upcoming"][0]["id"], slim["upcoming"][0]["id"])
        self.assertIn("all_data", default["upcoming"][0])
        self.assertNotIn("all_data", slim["upcoming"][0])
        self.assertNotIn("all_data", slim["previous"][0])

    def test_strip_heavy_fields_drops_raw_blobs(self):
        data = {
            "upcoming": [_sample_launch("u1", "Next", with_traj=True)],
            "previous": [_sample_launch("p1", "Past", with_traj=True)],
        }
        changed = app._strip_heavy_launch_fields(data)
        self.assertTrue(changed)
        self.assertNotIn("all_data", data["upcoming"][0])
        self.assertNotIn("all_data", data["previous"][0])
        self.assertIn("trajectory_data", data["upcoming"][0])
        self.assertNotIn("trajectory_data", data["previous"][0])

    def test_single_flight_coalesces_concurrent_calls(self):
        flight = app._SingleFlight()
        calls = []
        entered = threading.Barrier(5)

        def work():
            entered.wait(timeout=2)
            return flight.do(_slow)

        def _slow():
            calls.append(1)
            time.sleep(0.1)
            return "ok"

        results = []

        def worker():
            results.append(work())

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertEqual(calls, [1])
        self.assertEqual(results, ["ok"] * 5)

    def test_weather_refresh_single_flight_and_debounce(self):
        entered = threading.Barrier(3)
        weather_calls = []

        def fake_fetch(location=None, station_id=None, lat=None, lon=None):
            weather_calls.append(location)
            time.sleep(0.1)
            return {
                "temperature_c": 20,
                "temperature_f": 68,
                "humidity": 50,
                "wind_speed_kts": 5,
                "wind_gust_kts": 0,
                "wind_direction": 90,
            }

        forecast = {"current_weather": {"temperature": 20, "windspeed": 5, "winddirection": 90}}
        results = [None] * 3

        def worker(idx):
            entered.wait(timeout=2)
            results[idx] = app.refresh_weather_internal()

        with patch.object(app, "r", None), \
             patch.object(app, "WEATHER_LOCATIONS", ["Cape"]), \
             patch.object(app, "fetch_weather", side_effect=fake_fetch), \
             patch.object(app, "fetch_forecast", return_value=forecast):
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

            first_count = len(weather_calls)
            app.refresh_weather_internal()
            second_count = len(weather_calls)

        self.assertEqual(first_count, 1)
        self.assertEqual(second_count, 1)
        self.assertTrue(all(results))
        self.assertEqual(results[0]["weather"]["Cape"]["temperature_c"], 20.0)
        self.assertTrue(is_utc_iso8601(results[0]["weather"]["Cape"]["last_updated"]))


if __name__ == "__main__":
    unittest.main()
