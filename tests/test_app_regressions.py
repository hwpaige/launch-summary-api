import json
import os
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


class HotLaunchRawTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()

    def _list_payload(self, next_id="next-1", extra_id="later-2", net="2027-09-15T16:00:00Z"):
        return {
            "upcoming": [
                {
                    "id": next_id,
                    "name": "Next",
                    "mission": "Next",
                    "net": net,
                    "status": "Go",
                    "status_id": 1,
                },
                {
                    "id": extra_id,
                    "name": "Later",
                    "mission": "Later",
                    "net": "2027-09-20T16:00:00Z",
                    "status": "Go",
                    "status_id": 1,
                },
            ],
            "previous": [
                {
                    "id": "past-9",
                    "name": "Past",
                    "net": "2026-09-01T12:00:00Z",
                    "status": "Success",
                    "status_id": 3,
                }
            ],
            "last_updated": "2027-09-15T15:50:00Z",
        }

    def _remember(self, payload):
        app._remember_launches(payload)

    def test_default_launch_raw_returns_side_store_without_ll_fetch(self):
        stale = {"id": "next-1", "stale": True, "status": {"id": 1, "name": "Go"}}
        fresh = {"id": "next-1", "stale": False}
        self._remember(self._list_payload())
        app._local_raw_launches["next-1"] = stale

        with patch.object(app, "r", None), \
             patch.object(app, "fetch_launch_details", return_value=fresh) as fetch:
            result = app.get_launch_raw("next-1", internal=True)

        self.assertEqual(result["stale"], True)
        fetch.assert_not_called()

    def test_hot_query_refreshes_current_next_then_serves_short_ttl(self):
        stale = {"id": "next-1", "stale": True}
        fresh = {"id": "next-1", "stale": False, "status": {"id": 1, "name": "Go"}}
        self._remember(self._list_payload())
        app._local_raw_launches["next-1"] = stale

        with patch.object(app, "r", None), \
             patch.object(app, "fetch_launch_details", return_value=fresh) as fetch:
            first = app.get_launch_raw("next-1", hot=True, internal=True)
            second = app.get_launch_raw("next-1", hot=True, internal=True)

        self.assertEqual(first["stale"], False)
        self.assertEqual(second["stale"], False)
        self.assertEqual(fetch.call_count, 1)

    def test_hot_cache_expires_and_refetches(self):
        fresh_a = {"id": "next-1", "rev": 1}
        fresh_b = {"id": "next-1", "rev": 2}
        self._remember(self._list_payload())
        app._local_raw_launches["next-1"] = {"id": "next-1", "rev": 0}

        with patch.object(app, "r", None), \
             patch.object(app, "fetch_launch_details", side_effect=[fresh_a, fresh_b]) as fetch:
            first = app.get_launch_raw("next-1", hot=True, internal=True)
            data, _expires = app._local_hot_raw["next-1"]
            app._local_hot_raw["next-1"] = (data, time.time() - 1)
            second = app.get_launch_raw("next-1", hot=True, internal=True)

        self.assertEqual(first["rev"], 1)
        self.assertEqual(second["rev"], 2)
        self.assertEqual(fetch.call_count, 2)

    def test_hot_is_noop_for_non_current_ids(self):
        stale = {"id": "later-2", "stale": True}
        self._remember(self._list_payload())
        app._local_raw_launches["later-2"] = stale

        with patch.object(app, "r", None), \
             patch.object(app, "fetch_launch_details", return_value={"stale": False}) as fetch:
            result = app.get_launch_raw("later-2", hot=True, internal=True)

        self.assertEqual(result["stale"], True)
        fetch.assert_not_called()

    def test_hot_single_flight_coalesces_concurrent_ll_gets(self):
        self._remember(self._list_payload())
        entered = threading.Barrier(4)
        calls = []

        def fake_fetch(launch_id):
            calls.append(launch_id)
            time.sleep(0.08)
            return {"id": launch_id, "live": True}

        results = [None] * 4

        def worker(idx):
            entered.wait(timeout=2)
            results[idx] = app.get_launch_raw("next-1", hot=True, internal=True)

        with patch.object(app, "r", None), \
             patch.object(app, "fetch_launch_details", side_effect=fake_fetch):
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

        self.assertEqual(calls, ["next-1"])
        self.assertTrue(all(item and item.get("live") for item in results))

    def test_launches_slim_does_not_hit_hot_raw_or_ll_details(self):
        payload = self._list_payload()
        with patch.object(app, "r", None), \
             patch.object(app, "get_cached_data", return_value=payload), \
             patch.object(app, "set_cached_data", return_value=True), \
             patch.object(app, "fetch_launch_details") as fetch, \
             patch.object(app, "fetch_launches") as list_fetch:
            slim = app.get_launches_slim(internal=True)

        self.assertEqual(slim["upcoming"][0]["id"], "next-1")
        self.assertNotIn("all_data", slim["upcoming"][0])
        fetch.assert_not_called()
        list_fetch.assert_not_called()

    def test_fetch_launch_details_sends_ll_auth_token(self):
        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"id": "next-1", "ok": True}

        def fake_get(url, headers=None, timeout=10, verify=True):
            captured["url"] = url
            captured["headers"] = headers
            return FakeResponse()

        with patch.object(app.requests, "get", side_effect=fake_get):
            result = app.fetch_launch_details("next-1")

        self.assertEqual(result["ok"], True)
        self.assertEqual(
            captured["headers"],
            {"Authorization": f"Token {app.LL_API_KEY}"},
        )
        self.assertEqual(captured["headers"], app._ll_request_headers())

    def test_http_hot_query_param_and_default_compat(self):
        from fastapi.testclient import TestClient

        stale = {"id": "next-1", "source": "side"}
        live = {"id": "next-1", "source": "ll2"}
        self._remember(self._list_payload())
        app._local_raw_launches["next-1"] = stale

        with patch.object(app, "r", None), \
             patch.object(app, "fetch_launch_details", return_value=live) as fetch, \
             patch.object(app, "_background_enabled", False):
            client = TestClient(app.app)
            default_resp = client.get("/launch_raw/next-1")
            hot_resp = client.get("/launch_raw/next-1", params={"hot": 1})

        self.assertEqual(default_resp.status_code, 200)
        self.assertEqual(default_resp.json()["source"], "side")
        self.assertEqual(hot_resp.status_code, 200)
        self.assertEqual(hot_resp.json()["source"], "ll2")
        self.assertEqual(fetch.call_count, 1)

    def test_hot_uses_redis_single_flight_name(self):
        self._remember(self._list_payload())
        live = {"id": "next-1", "live": True}

        with patch.object(app, "r", None), \
             patch.object(app, "fetch_launch_details", return_value=live), \
             patch.object(app, "_redis_single_flight", side_effect=lambda name, fn: fn()) as lock:
            app.get_launch_raw("next-1", hot=True, internal=True)

        lock.assert_called()
        self.assertEqual(lock.call_args[0][0], "hot_raw_next-1")


def _max_cross_track_km(leg, reference):
    """Largest distance from a leg point to the nearest reference point."""
    worst = 0.0
    for point in leg:
        nearest = min(app._distance_km(point, other) for other in reference)
        if nearest > worst:
            worst = nearest
    return worst


class LaunchSiteTrajectoryTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()

    def tearDown(self):
        cache_path = app.TRAJECTORY_CACHE_FILE
        if os.path.exists(cache_path):
            os.remove(cache_path)

    def test_olp2_and_boca_chica_aliases_resolve_to_starbase(self):
        for pad in (
            "Orbital Launch Pad 2",
            "OLP-2",
            "OLP 2",
            "OLM-2",
            "Boca Chica",
            "Starbase",
            "Orbital Launch Mount 2",
        ):
            site, _key = app.resolve_launch_site(pad)
            self.assertIsNotNone(site, pad)
            self.assertIn("Starbase", site["name"])
            self.assertAlmostEqual(site["lat"], app.STARBASE_LAUNCH_SITE["lat"], places=5)
            self.assertAlmostEqual(site["lon"], app.STARBASE_LAUNCH_SITE["lon"], places=5)
            self.assertGreater(abs(site["lat"] - 28.6084), 1.0)

    def test_unknown_pad_is_not_lc39a(self):
        site, key = app.resolve_launch_site("Some Future Pad")
        self.assertIsNone(site)
        self.assertIsNone(key)
        site, key = app.resolve_launch_site("")
        self.assertIsNone(site)
        self.assertIsNone(key)

    def test_known_cape_and_vandenberg_pads_still_resolve(self):
        cape, _key = app.resolve_launch_site("Launch Complex 39A")
        self.assertEqual(cape["name"], "Cape Canaveral, FL")
        self.assertAlmostEqual(cape["lat"], 28.6084, places=4)
        slc40, _key = app.resolve_launch_site("Space Launch Complex 40")
        self.assertEqual(slc40["name"], "Cape Canaveral, FL")
        self.assertAlmostEqual(slc40["lat"], 28.5619, places=4)
        vandy, _key = app.resolve_launch_site("Space Launch Complex 4E")
        self.assertIn("Vandenberg", vandy["name"])
        self.assertAlmostEqual(vandy["lon"], -120.6107, places=4)

    def test_upstream_pad_coordinates_override_hardcoded_defaults(self):
        site, _key = app.resolve_launch_site(
            "Orbital Launch Pad 2",
            latitude="25.99677",
            longitude="-97.15799",
            location_name="SpaceX Starbase, TX, USA",
        )
        self.assertAlmostEqual(site["lat"], 25.99677, places=5)
        self.assertAlmostEqual(site["lon"], -97.15799, places=5)
        self.assertIn("Starbase", site["name"])

        # Pad name used to miss every alias and fall through to LC-39A.
        site, _key = app.resolve_launch_site(
            "Unlisted Starbase Pad",
            latitude=25.99677,
            longitude=-97.15799,
        )
        self.assertAlmostEqual(site["lat"], 25.99677, places=5)
        self.assertAlmostEqual(site["lon"], -97.15799, places=5)
        self.assertIn("Starbase", site["name"])

        cape, _key = app.resolve_launch_site(
            "Launch Complex 39A",
            latitude=28.608389,
            longitude=-80.604333,
        )
        self.assertAlmostEqual(cape["lat"], 28.608389, places=6)
        self.assertAlmostEqual(cape["lon"], -80.604333, places=6)
        self.assertEqual(cape["name"], "Cape Canaveral, FL")

    def test_parse_launch_data_keeps_pad_coordinates(self):
        parsed = app.parse_launch_data({
            "id": "7d1afb26-6f9c-429b-9ccf-29012fd1e519",
            "name": "Starship Flight 14",
            "net": "2026-10-01T00:00:00Z",
            "status": {"name": "Go", "id": 1},
            "rocket": {"configuration": {"name": "Starship"}, "launcher_stage": []},
            "mission": {"orbit": {"name": "LEO"}},
            "pad": {
                "name": "Orbital Launch Pad 2",
                "latitude": 25.99677,
                "longitude": -97.15799,
                "location": {"name": "SpaceX Starbase, TX, USA"},
            },
        })
        self.assertEqual(parsed["pad"], "Orbital Launch Pad 2")
        self.assertEqual(parsed["pad_latitude"], 25.99677)
        self.assertEqual(parsed["pad_longitude"], -97.15799)
        self.assertIn("Starbase", parsed["pad_location"])

    def test_flight14_trajectory_starts_at_starbase(self):
        launch = {
            "id": "7d1afb26-6f9c-429b-9ccf-29012fd1e519",
            "mission": "Starship | Starlink Group 31-1 (Starship Flight 14)",
            "pad": "Orbital Launch Pad 2",
            "pad_latitude": 25.99677,
            "pad_longitude": -97.15799,
            "pad_location": "SpaceX Starbase, TX, USA",
            "orbit": "Low Earth Orbit",
        }
        with patch.object(app, "save_cache_to_file"), patch.object(app, "r", None):
            traj = app.get_launch_trajectory_data(launch)
        site = traj["launch_site"]
        origin = traj["trajectory"][0]
        self.assertIn("Starbase", site["name"])
        self.assertAlmostEqual(site["lat"], 25.99677, places=4)
        self.assertAlmostEqual(site["lon"], -97.15799, places=4)
        self.assertAlmostEqual(origin["lat"], site["lat"], places=2)
        self.assertAlmostEqual(origin["lon"], site["lon"], places=2)
        self.assertGreater(abs(origin["lat"] - 28.6084), 1.0)
        self.assertGreater(abs(origin["lon"] - (-80.6043)), 1.0)

    def test_slim_regenerates_cached_cape_trajectory_for_olp2(self):
        payload = {
            "upcoming": [{
                "id": "7d1afb26-6f9c-429b-9ccf-29012fd1e519",
                "mission": "Starship Flight 14",
                "name": "Starship Flight 14",
                "pad": "Orbital Launch Pad 2",
                "orbit": "LEO",
                "net": "2026-10-01T00:00:00Z",
                "trajectory_data": {
                    "launch_site": {
                        "lat": 28.6084,
                        "lon": -80.6043,
                        "name": "Cape Canaveral, FL",
                    },
                    "trajectory": [{"lat": 28.6084, "lon": -80.6043, "r": 1.0}],
                },
            }],
            "previous": [],
            "last_updated": "2026-09-28T00:00:00Z",
        }
        persisted = []

        def fake_persist(key, data, ttl=None):
            persisted.append((key, data))
            return True

        with patch.object(app, "get_cached_data", return_value=payload), \
             patch.object(app, "set_cached_data", side_effect=fake_persist), \
             patch.object(app, "save_cache_to_file"), \
             patch.object(app, "r", None):
            slim = app.get_launches_slim(internal=True)

        site = slim["upcoming"][0]["trajectory_data"]["launch_site"]
        origin = slim["upcoming"][0]["trajectory_data"]["trajectory"][0]
        self.assertIn("Starbase", site["name"])
        self.assertAlmostEqual(site["lat"], app.STARBASE_LAUNCH_SITE["lat"], places=3)
        self.assertAlmostEqual(site["lon"], app.STARBASE_LAUNCH_SITE["lon"], places=3)
        self.assertLess(abs(origin["lat"] - site["lat"]), 0.05)
        self.assertLess(abs(origin["lon"] - site["lon"]), 0.05)
        self.assertGreater(abs(origin["lat"] - 28.6084), 1.0)
        self.assertTrue(any(item[0] == app.LAUNCHES_CACHE_KEY for item in persisted))

    def test_matching_cape_trajectory_is_not_regenerated(self):
        payload = {
            "upcoming": [{
                "id": "cape-1",
                "mission": "Falcon",
                "pad": "Launch Complex 39A",
                "orbit": "LEO",
                "trajectory_data": {
                    "launch_site": {
                        "lat": 28.6084,
                        "lon": -80.6043,
                        "name": "Cape Canaveral, FL",
                    },
                    "trajectory": [{"lat": 28.6084, "lon": -80.6043, "r": 1.0}],
                    "booster_ground_track": None,
                },
            }],
            "previous": [],
            "last_updated": "2026-09-28T00:00:00Z",
        }
        with patch.object(app, "get_cached_data", return_value=payload), \
             patch.object(app, "set_cached_data", return_value=True), \
             patch.object(app, "get_launch_trajectory_data") as generate, \
             patch.object(app, "r", None):
            slim = app.get_launches_slim(internal=True)

        generate.assert_not_called()
        self.assertEqual(
            slim["upcoming"][0]["trajectory_data"]["launch_site"]["name"],
            "Cape Canaveral, FL",
        )

    def test_parse_keeps_published_landing_zone_over_expended_core(self):
        parsed = app.parse_launch_data({
            "id": "fh-1",
            "name": "Falcon Heavy | NROL-97",
            "net": "2026-10-02T03:53:00Z",
            "status": {"name": "Go", "id": 1},
            "rocket": {
                "configuration": {"name": "Falcon Heavy"},
                "launcher_stage": [
                    {
                        "type": "Strap-On Booster",
                        "landing": {
                            "type": {"name": "Return to Launch Site"},
                            "downrange_distance": 14.9,
                            "landing_location": {
                                "name": "Landing Zone 1",
                                "latitude": 28.485712,
                                "longitude": -80.542963,
                            },
                        },
                    },
                    {
                        "type": "Core",
                        "landing": {
                            "type": {"name": "Expended"},
                            "downrange_distance": None,
                            "landing_location": {
                                "name": "Atlantic Ocean",
                                "latitude": None,
                                "longitude": None,
                            },
                        },
                    },
                ],
            },
            "mission": {"orbit": {"name": "Unknown"}, "description": "Classified payload."},
            "pad": {"name": "Launch Complex 39A", "latitude": 28.60822681, "longitude": -80.60428186},
        })
        self.assertEqual(parsed["landing_type"], "Return to Launch Site")
        self.assertEqual(parsed["landing_location"], "Landing Zone 1")
        self.assertAlmostEqual(parsed["landing_latitude"], 28.485712, places=5)
        self.assertAlmostEqual(parsed["landing_longitude"], -80.542963, places=5)
        self.assertAlmostEqual(parsed["landing_downrange_km"], 14.9, places=2)

    def test_crew_iss_ground_track_uses_station_inclination(self):
        launch = {
            "mission": "Falcon 9 Block 5 | Crew-13",
            "name": "Falcon 9 Block 5 | Crew-13",
            "description": (
                "SpaceX Crew-13 is the thirteenth crewed operational flight of a "
                "Crew Dragon spacecraft to the International Space Station."
            ),
            "pad": "Space Launch Complex 40",
            "pad_latitude": 28.56194122,
            "pad_longitude": -80.57735736,
            "orbit": "Low Earth Orbit",
            "landing_type": "Return to Launch Site",
            "landing_location": "Landing Zone 40",
            "landing_latitude": 28.5634384,
            "landing_longitude": -80.5752619,
            "landing_downrange_km": 0.3,
        }
        with patch.object(app, "save_cache_to_file"), patch.object(app, "r", None):
            traj = app.get_launch_trajectory_data(launch)
        lats = [p["lat"] for p in traj["orbit_path"]]
        self.assertGreater(max(lats), 48.0)
        self.assertLess(max(lats), 55.0)
        self.assertAlmostEqual(traj["inclination_deg"], 51.6, places=1)
        ascent_km = app._polyline_length_km(traj["trajectory"])
        # Orbital-rate coast for 9 minutes is ~4,000 km. The ramp covers about half.
        self.assertGreater(ascent_km, 800.0)
        self.assertLess(ascent_km, 2600.0)
        self.assertGreater(max(p["lat"] for p in traj["trajectory"]), 34.0)
        boost = traj["booster_trajectory"]
        self.assertEqual(traj["booster_ground_track"], "along_track_return")
        self.assertGreater(len(boost), 5)
        pad = traj["launch_site"]
        self.assertGreater(app._distance_km(pad, boost[0]), 60.0)
        self.assertLess(app._distance_km(pad, boost[0]), 400.0)
        self.assertAlmostEqual(boost[-1]["lat"], 28.5634384, places=3)
        self.assertAlmostEqual(boost[-1]["lon"], -80.5752619, places=3)
        # The return stays on the ascent ground track. It does not bow off it.
        self.assertLess(_max_cross_track_km(boost, traj["trajectory"]), 25.0)

    def test_rtls_return_follows_the_ascent_track_and_ends_at_lz1(self):
        launch = {
            "mission": "Falcon Heavy | NROL-97",
            "pad": "Launch Complex 39A",
            "pad_latitude": 28.60822681,
            "pad_longitude": -80.60428186,
            "orbit": "Unknown",
            "landing_type": "Return to Launch Site",
            "landing_location": "Landing Zone 1",
            "landing_latitude": 28.485712,
            "landing_longitude": -80.542963,
            "landing_downrange_km": 14.9,
        }
        with patch.object(app, "save_cache_to_file"), patch.object(app, "r", None):
            traj = app.get_launch_trajectory_data(launch)
        end = traj["booster_trajectory"][-1]
        pad = traj["launch_site"]
        start = traj["booster_trajectory"][0]
        self.assertEqual(traj["booster_ground_track"], "along_track_return")
        self.assertAlmostEqual(end["lat"], 28.485712, places=3)
        self.assertAlmostEqual(end["lon"], -80.542963, places=3)
        # Staging is well downrange. The line is the return, not the 15 km pad offset.
        self.assertGreater(app._distance_km(pad, start), 60.0)
        self.assertLess(app._distance_km(pad, start), 400.0)
        self.assertGreater(
            max(app._distance_km(pad, p) for p in traj["booster_trajectory"]),
            60.0,
        )
        self.assertLess(_max_cross_track_km(traj["booster_trajectory"], traj["trajectory"]), 25.0)

    def test_asds_uses_published_downrange_not_a_650_km_default(self):
        launch = {
            "mission": "Falcon 9 Block 5 | Starlink Group 15-25",
            "pad": "Space Launch Complex 4E",
            "pad_latitude": 34.632,
            "pad_longitude": -120.611,
            "pad_location": "Vandenberg SFB, CA, USA",
            "orbit": "Low Earth Orbit",
            "landing_type": "Autonomous Spaceport Drone Ship",
            "landing_location": "Of Course I Still Love You",
            "landing_downrange_km": 574.0,
        }
        with patch.object(app, "save_cache_to_file"), patch.object(app, "r", None):
            traj = app.get_launch_trajectory_data(launch)
        end = traj["booster_trajectory"][-1]
        start = traj["booster_trajectory"][0]
        pad = traj["launch_site"]
        end_km = app._distance_km(pad, end)
        start_km = app._distance_km(pad, start)
        self.assertEqual(traj["booster_ground_track"], "along_track_downrange")
        self.assertAlmostEqual(end_km, 574.0, delta=15.0)
        self.assertGreater(abs(end_km - 650.0), 40.0)
        self.assertGreater(start_km, 40.0)
        self.assertLess(start_km, end_km)
        self.assertLess(_max_cross_track_km(traj["booster_trajectory"], traj["trajectory"]), 25.0)

    def test_ocean_splashdown_without_coordinates_has_no_booster_track(self):
        launch = {
            "mission": "Starship | Starlink Group 31-1 (Starship Flight 14)",
            "pad": "Orbital Launch Pad 2",
            "pad_latitude": 25.99677,
            "pad_longitude": -97.15799,
            "pad_location": "SpaceX Starbase, TX, USA",
            "orbit": "Low Earth Orbit",
            "landing_type": "Ocean",
            "landing_location": "Gulf of Mexico",
            "description": (
                "The booster’s primary test objective on Flight 14 will be a "
                "landing burn at an offshore landing point in the Gulf."
            ),
        }
        with patch.object(app, "save_cache_to_file"), patch.object(app, "r", None):
            traj = app.get_launch_trajectory_data(launch)
        self.assertIn("Starbase", traj["launch_site"]["name"])
        self.assertEqual(traj["booster_trajectory"], [])
        self.assertIsNone(traj["booster_ground_track"])
        self.assertIsNone(traj["landing_site"])

    def test_stale_booster_model_is_regenerated_on_read(self):
        payload = {
            "upcoming": [{
                "id": "crew-13",
                "mission": "Falcon 9 Block 5 | Crew-13",
                "pad": "Space Launch Complex 40",
                "pad_latitude": 28.56194122,
                "pad_longitude": -80.57735736,
                "orbit": "Low Earth Orbit",
                "landing_type": "Return to Launch Site",
                "landing_location": "Landing Zone 40",
                "landing_latitude": 28.5634384,
                "landing_longitude": -80.5752619,
                "trajectory_data": {
                    "launch_site": {
                        "lat": 28.56194122,
                        "lon": -80.57735736,
                        "name": "Cape Canaveral, FL",
                    },
                    "trajectory": [
                        {"lat": 28.56194122, "lon": -80.57735736, "r": 1.0},
                        {"lat": 29.0, "lon": -70.0, "r": 1.04},
                    ],
                    "booster_trajectory": [
                        {"lat": 29.0, "lon": -70.0, "r": 1.04},
                        {"lat": 28.56194122, "lon": -80.57735736, "r": 1.0},
                    ],
                },
            }],
            "previous": [],
            "last_updated": "2026-09-28T00:00:00Z",
        }
        persisted = []

        def fake_persist(key, data, ttl=None):
            persisted.append(key)
            return True

        with patch.object(app, "get_cached_data", return_value=payload), \
             patch.object(app, "set_cached_data", side_effect=fake_persist), \
             patch.object(app, "save_cache_to_file"), \
             patch.object(app, "r", None):
            slim = app.get_launches_slim(internal=True)

        traj = slim["upcoming"][0]["trajectory_data"]
        self.assertEqual(traj["booster_ground_track"], "along_track_return")
        boost = traj["booster_trajectory"]
        self.assertGreater(len(boost), 5)
        # Regenerated return stays near the Cape. The cached arc reached 70°W.
        self.assertGreater(min(p["lon"] for p in boost), -85.0)
        self.assertTrue(any(key == app.LAUNCHES_CACHE_KEY for key in persisted))


if __name__ == "__main__":
    unittest.main()
