"""The poller publishes feed fixes. Requests only read the cached document."""
import gzip
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import app
import starship_http
from starlette.requests import Request
from starship_tracker import synthetic_inflight_feeds, utc_to_gps

app._background_enabled = False


class Resp:
    def __init__(self, status, body=None, headers=None):
        self.status_code = status
        self._body = {} if body is None else body
        self.headers = headers or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _feed(moment, lat, lon, alt=80_000.0, speed=1_500.0, mission=40.0):
    gps = utc_to_gps(moment)
    return {
        "ship42": {
            "current": {
                "gps_time": gps,
                "mission_time": mission,
                "altitude": alt,
                "speed": speed,
                "latitude": lat,
                "longitude": lon,
                "r_ecef": [1_000_000.0, 2_000_000.0, 3_000_000.0],
            },
            "trajectory": [
                {
                    "time": gps + 60,
                    "latitude": lat + 0.02,
                    "longitude": lon + 0.02,
                    "altitude": alt + 200,
                    "r_ecef": [0, 0, 0],
                },
                {
                    "time": gps + 120,
                    "latitude": lat + 0.04,
                    "longitude": lon + 0.04,
                    "altitude": alt + 400,
                    "r_ecef": [0, 0, 0],
                },
            ],
        },
        "metadata": {
            "generation_time": gps,
            "trajectory_version": 1,
            "coordinate_frame": "ECEF",
            "sample_period": 60,
        },
    }


class StarshipPollerTests(unittest.TestCase):
    def setUp(self):
        app._background_enabled = False
        self._redis = app.r
        app.r = None
        self.tmp = tempfile.TemporaryDirectory()
        self.calls = []
        self.now = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
        self.launches = {
            "upcoming": [{
                "net": (self.now - timedelta(minutes=5)).isoformat(),
                "rocket": "Starship",
                "mission": "Starship Flight 16",
            }],
            "previous": [],
        }
        self.service = starship_http.StarshipService(
            launch_loader=lambda: self.launches,
            path=os.path.join(self.tmp.name, "tracker.json"),
        )
        starship_http.reset_service_for_tests(self.service)

    def tearDown(self):
        app.r = self._redis
        starship_http.reset_service_for_tests(None)
        self.tmp.cleanup()

    def test_positions_come_from_the_feed_and_predictions_are_marked(self):
        samples = [
            (self.now, 26.2, -96.4),
            (self.now + timedelta(seconds=10), 26.4, -95.8),
        ]
        pending = [
            _feed(moment, lat, lon, mission=30 + index * 10)
            for index, (moment, lat, lon) in enumerate(samples)
        ]

        def get(url, headers=None, timeout=None):
            self._get_headers = headers
            if "ship-trail" in url or "history" in url:
                return Resp(404, {})
            if not url.endswith("starship_tracker_public.json"):
                return Resp(404, {})
            if not pending:
                return Resp(304, None)
            return Resp(200, pending.pop(0), headers={"ETag": "v"})

        self.assertTrue(callable(synthetic_inflight_feeds))
        poller_src = open(
            os.path.join(os.path.dirname(__file__), "..", "starship_http.py"),
            encoding="utf-8",
        ).read()
        self.assertNotIn("synthetic_inflight_feeds(", poller_src)
        first = self.service.poll_once(
            now=samples[0][0], get=get, refresh_closures=False
        )
        second = self.service.poll_once(
            now=samples[1][0], get=get, refresh_closures=False
        )
        self.assertGreaterEqual(first, 5)
        self.assertGreaterEqual(second, 5)
        payload = self.service.tracker_payload()
        self.assertTrue(payload["live_positions"])
        self.assertEqual(payload["position_source"], "feed")
        self.assertEqual(payload["feed"]["kind"], "official")
        self.assertIn("content.spacex.com", payload["feed"]["url"])
        self.assertEqual(payload["planned"]["kind"], "planned")
        self.assertEqual(payload["orbit"]["kind"], "derived")
        self.assertFalse(payload.get("test", False))
        vehicle = payload["vehicles"][0]
        self.assertEqual(vehicle["id"], "ship42")
        self.assertEqual(vehicle["position_kind"], "feed")
        self.assertAlmostEqual(vehicle["lat"], 26.4)
        self.assertAlmostEqual(vehicle["lon"], -95.8)
        self.assertAlmostEqual(vehicle["alt_km"], 80.0)
        self.assertAlmostEqual(vehicle["speed_kmh"], 1_500.0 * 3.6)
        self.assertIn(",", vehicle["labels"]["speed"])
        self.assertTrue(vehicle["labels"]["altitude"].endswith("km"))
        self.assertNotIn("mi", vehicle["labels"]["altitude"].lower())
        self.assertEqual(vehicle["predicted"][0]["kind"], "predicted")
        self.assertNotEqual(vehicle["predicted"][0]["lat"], vehicle["lat"])
        self.assertTrue(vehicle["trail"])
        self.assertTrue(all(point["kind"] == "flown" for point in vehicle["trail"]))
        held_lat = vehicle["lat"]

        def unchanged(url, headers=None, timeout=None):
            if "ship-trail" in url or "history" in url:
                return Resp(404, {})
            return Resp(304, None)

        self.service.poll_once(
            now=samples[1][0] + timedelta(seconds=10),
            get=unchanged,
            refresh_closures=False,
        )
        again = self.service.tracker_payload()
        self.assertAlmostEqual(again["vehicles"][0]["lat"], held_lat)
        self.assertEqual(again["position_source"], "feed")

    def test_a_feed_with_no_fix_does_not_invent_a_vehicle(self):
        def empty(url, headers=None, timeout=None):
            if url.endswith("starship_tracker_public.json"):
                return Resp(200, {})
            return Resp(404, {})

        self.service.poll_once(now=self.now, get=empty, refresh_closures=False)
        payload = self.service.tracker_payload()
        self.assertEqual(payload["vehicles"], [])
        self.assertFalse(payload["live_positions"])
        self.assertEqual(payload["position_source"], "none")
        self.assertEqual(payload["feed"]["kind"], "none")
        self.assertIsNone(payload["feed"]["url"])

    def test_fixture_is_not_a_live_position(self):
        payload = None
        for _ in range(40):
            self.service._fixture_step_at = 0
            payload = self.service.fixture_payload("once")
            if payload.get("vehicles"):
                break
        self.assertTrue(payload["vehicles"])
        self.assertTrue(payload["test"])
        self.assertEqual(payload["position_source"], "fixture")
        self.assertFalse(payload["live_positions"])
        self.assertEqual(payload["feed"]["kind"], "fixture")
        self.assertIsNone(payload["feed"]["url"])
        for vehicle in payload["vehicles"]:
            self.assertEqual(vehicle["position_kind"], "fixture")
            self.assertIn("TEST", vehicle["name"])
            self.assertTrue(all(point.get("kind") == "predicted" for point in vehicle["predicted"]))
            self.assertTrue(all(point.get("kind") == "flown" for point in vehicle["trail"]))
        live = self.service.tracker_payload()
        self.assertEqual(live["position_source"], "none")
        self.assertEqual(live["vehicles"], [])

    def test_retry_after_slows_the_poll(self):
        def limited(url, headers=None, timeout=None):
            return Resp(429, {}, headers={"Retry-After": "30"})

        interval = self.service.poll_once(now=self.now, get=limited, refresh_closures=False)
        self.assertGreaterEqual(interval, 30)
        self.assertLessEqual(interval, 300)

    def test_request_does_not_fetch_or_load_launches(self):
        scope = {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/starship/tracker",
            "raw_path": b"/starship/tracker",
            "query_string": b"",
            "headers": [(b"accept-encoding", b"gzip")],
            "client": ("127.0.0.1", 123),
            "server": ("test", 80),
        }
        request = Request(scope)
        rendered = self.service.rendered_tracker()
        response = starship_http._json_response(
            rendered, request, starship_http.TRACKER_CACHE_CONTROL, "cache"
        )
        self.assertEqual(response.headers["cache-control"], "public, max-age=5")
        self.assertEqual(response.headers["content-encoding"], "gzip")
        self.assertTrue(response.body.startswith(b"\x1f\x8b"))
        self.assertEqual(json.loads(gzip.decompress(response.body))["position_source"], "none")

        from fastapi.testclient import TestClient

        client = TestClient(app.app)
        with patch("requests.get", side_effect=AssertionError("network")), \
             patch.object(app, "_load_launch_payload", side_effect=AssertionError("launches")):
            live = client.get("/starship/tracker")
            closures = client.get("/starship/closures")
            fixture = client.get("/starship/tracker", params={"fixture": "flight_test"})
        self.assertEqual(live.status_code, 200)
        self.assertEqual(live.headers["cache-control"], "public, max-age=5")
        self.assertEqual(live.json()["position_source"], "none")
        self.assertEqual(live.json()["vehicles"], [])
        self.assertIn("show", live.json())
        self.assertEqual(closures.status_code, 200)
        self.assertEqual(closures.headers["cache-control"], "public, max-age=60")
        body = closures.json()
        self.assertEqual(body["kind"], "closure")
        self.assertIn(body["source"], ("fallback", "cache", "live", "mixed"))
        self.assertIn("areas", body)
        self.assertEqual(fixture.status_code, 200)
        self.assertEqual(fixture.headers["cache-control"], "no-store")
        self.assertEqual(fixture.json()["position_source"], "fixture")
        self.assertFalse(fixture.json()["live_positions"])
        self.assertTrue(fixture.json()["test"])


if __name__ == "__main__":
    unittest.main()
