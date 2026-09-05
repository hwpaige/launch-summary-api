import json
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
        "pad": "LC-39A",
        "video_url": "https://example.com/watch",
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

    def test_launches_default_omits_all_data_full_flag_restores_it(self):
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

        self.assertNotIn("all_data", result["upcoming"][0])
        self.assertNotIn("all_data", result["upcoming"][1])
        self.assertNotIn("all_data", result["previous"][0])
        self.assertEqual(result["upcoming"][0]["mission"], "Next")
        self.assertEqual(result["upcoming"][0]["net"], "2026-09-02T14:21:21Z")
        self.assertIn("trajectory_data", result["upcoming"][0])
        self.assertNotIn("trajectory_data", result["upcoming"][1])
        self.assertNotIn("trajectory_data", result["previous"][0])
        self.assertNotIn("all_data", slim["upcoming"][0])
        self.assertIn("all_data", full["upcoming"][0])
        self.assertEqual(full["upcoming"][0]["all_data"]["id"], "u1")
        self.assertIn("all_data", full["previous"][0])
        list_persists = [item for item in persisted if item[0] == app.LAUNCHES_CACHE_KEY]
        self.assertTrue(list_persists)
        self.assertNotIn("all_data", list_persists[0][1]["upcoming"][0])
        self.assertIsNone(app._launches_mem["data"]["upcoming"][0].get("all_data"))

    def test_launches_slim_matches_default_shape(self):
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
        self.assertEqual(default["upcoming"][0]["mission"], slim["upcoming"][0]["mission"])
        self.assertNotIn("all_data", default["upcoming"][0])
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


class NotifyCopyTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()

    def _payload(self, **overrides):
        data = {
            "launch_id": "launch-notify-1",
            "event": "t1h",
            "mission": "Starlink Group 10-20",
            "net": "2026-09-06T02:15:00Z",
            "status": "Go",
            "pad": "SLC-40",
            "rocket": "Falcon 9",
            "orbit": "LEO",
            "probability": 70,
        }
        data.update(overrides)
        return app.NotifyCopyRequest(**data)

    def test_fallback_includes_probability_for_t1h_t24h_scrub(self):
        t1h_title, t1h_body = app.fallback_notify_copy(self._payload(event="t1h", probability=70))
        t24h_title, t24h_body = app.fallback_notify_copy(self._payload(event="t24h", probability=80))
        scrub_title, scrub_body = app.fallback_notify_copy(
            self._payload(event="scrub", status="Hold", previous_status="Go", probability=40)
        )

        self.assertIn("T-1h", t1h_title)
        self.assertIn("70%", f"{t1h_title} {t1h_body}")
        self.assertLessEqual(len(t1h_title), app.NOTIFY_TITLE_MAX)
        self.assertLessEqual(len(t1h_body), app.NOTIFY_BODY_MAX)

        self.assertIn("T-24h", t24h_title)
        self.assertIn("80%", f"{t24h_title} {t24h_body}")
        self.assertIn("Starlink", t24h_body)

        self.assertIn("Hold", f"{scrub_title} {scrub_body}")
        self.assertIn("plans changed", scrub_body.lower())
        self.assertIn("40%", f"{scrub_title} {scrub_body}")

    def test_generate_notify_copy_uses_shared_grok_path_and_caches(self):
        calls = []

        def fake_grok(prompt, temperature=0.7, max_tokens=4000):
            calls.append(prompt)
            self.assertIn("t1h", prompt)
            self.assertIn("70", prompt)
            return '{"title":"T-1h: Starlink 10-20","body":"T-1 hour at SLC-40. 70% go."}'

        with patch.object(app, "r", None), \
             patch.object(app, "call_grok", side_effect=fake_grok):
            first = app.generate_notify_copy(self._payload())
            second = app.generate_notify_copy(self._payload())
            rescheduled = app.generate_notify_copy(self._payload(net="2026-09-06T04:15:00Z"))
            different_prob = app.generate_notify_copy(self._payload(probability=40))

        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(first["title"], second["title"])
        self.assertEqual(first["model"], app.GROK_MODEL)
        self.assertFalse(rescheduled["cached"])
        self.assertFalse(different_prob["cached"])
        self.assertEqual(len(calls), 3)
        self.assertIn("70%", first["body"])

    def test_generate_notify_copy_falls_back_when_grok_fails(self):
        with patch.object(app, "r", None), \
             patch.object(app, "call_grok", side_effect=ValueError("Grok down")):
            result = app.generate_notify_copy(self._payload(event="t24h", probability=55))

        self.assertFalse(result["cached"])
        self.assertEqual(result["event"], "t24h")
        self.assertIn("T-24h", result["title"])
        self.assertIn("55%", f"{result['title']} {result['body']}")
        self.assertEqual(result["model"], app.GROK_MODEL)

    def test_probability_injected_when_grok_omits_it(self):
        with patch.object(app, "r", None), \
             patch.object(app, "call_grok", return_value='{"title":"T-1h: Starlink","body":"T-1 hour at SLC-40. Stack is hot."}'):
            result = app.generate_notify_copy(self._payload(probability=70))

        self.assertIn("70%", f"{result['title']} {result['body']}")

    def test_http_get_and_post_notify_copy(self):
        from fastapi.testclient import TestClient

        grok_json = '{"title":"T-24h: Starlink 10-20","body":"24-hour clock at SLC-40. 80% go."}'
        with patch.object(app, "r", None), \
             patch.object(app, "call_grok", return_value=grok_json), \
             patch.object(app, "_background_enabled", False):
            client = TestClient(app.app)
            get_resp = client.get(
                "/notify/copy",
                params={
                    "launch_id": "launch-notify-1",
                    "event": "t24h",
                    "mission": "Starlink Group 10-20",
                    "pad": "SLC-40",
                    "probability": 80,
                },
            )
            post_resp = client.post(
                "/notify/copy",
                json={
                    "launch_id": "launch-notify-2",
                    "event": "scrub",
                    "mission": "Starlink Group 10-20",
                    "status": "Hold",
                    "previous_status": "Go",
                    "probability": 40,
                },
            )
            bad_event = client.get("/notify/copy", params={"launch_id": "x", "event": "liftoff"})

        self.assertEqual(get_resp.status_code, 200)
        get_body = get_resp.json()
        self.assertEqual(get_body["event"], "t24h")
        self.assertIn("80%", f"{get_body['title']} {get_body['body']}")
        self.assertEqual(get_body["model"], app.GROK_MODEL)

        self.assertEqual(post_resp.status_code, 200)
        post_body = post_resp.json()
        self.assertEqual(post_body["event"], "scrub")
        self.assertIn("40%", f"{post_body['title']} {post_body['body']}")
        self.assertLessEqual(len(post_body["title"]), 50)
        self.assertLessEqual(len(post_body["body"]), 150)

        self.assertEqual(bad_event.status_code, 422)

    def test_openapi_documents_notify_copy_events(self):
        schema = app.app.openapi()
        paths = schema["paths"]
        self.assertIn("/notify/copy", paths)
        self.assertIn("get", paths["/notify/copy"])
        self.assertIn("post", paths["/notify/copy"])
        dumped = json.dumps(schema)
        self.assertIn("t1h", dumped)
        self.assertIn("t24h", dumped)
        self.assertIn("scrub", dumped)
        self.assertIn("Notifications", dumped)
        self.assertIn(app.GROK_MODEL, dumped)


if __name__ == "__main__":
    unittest.main()
