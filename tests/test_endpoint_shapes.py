"""Pin the response shapes Launch Buddy and the dashboard Pis already parse.

Starship routes are additive. These tests fail if an existing payload gains,
drops, or renames a field.
"""
import os
import unittest
from unittest.mock import patch

import app

app._background_enabled = False

SLIM_TOP = {"upcoming", "previous", "last_updated"}
TRAJECTORY_WRAP = {"trajectory", "launch_id", "mission"}
TRAJECTORY_BODY = {
    "launch_site",
    "trajectory",
    "booster_trajectory",
    "booster_ground_track",
    "sep_idx",
    "orbit_path",
    "orbit",
    "mission",
    "pad",
    "landing_type",
    "landing_location",
    "landing_site",
    "landing_downrange_km",
    "inclination_deg",
}
NOTIFY_KEYS = {"title", "body", "launch_id", "event", "cached", "model"}
SATELLITE_KEYS = {
    "group",
    "fetched_at",
    "ttl_seconds",
    "count",
    "stale",
    "source",
    "note",
    "satellites",
}
SATELLITE_ROW = {"name", "norad_id", "tle_line1", "tle_line2"}
META_KEYS = {
    "group",
    "fetched_at",
    "count",
    "ttl_seconds",
    "age_seconds",
    "stale",
    "source",
    "note",
    "allowed_groups",
}
DEPLOYED_KEYS = {
    "launch_date",
    "generation",
    "mission",
    "fetched_at",
    "ttl_seconds",
    "count",
    "catalog_count",
    "stale",
    "empty",
    "source",
    "note",
    "satellites",
}
# Surveyed Starbase point this API already serves. It is the same pad as
# 25.9975, -97.1566 (well under 2°) and it is not LC-39A. The constant stays
# so cached /trajectory documents do not move.
STARBASE_LAT = 25.99677
STARBASE_LON = -97.15799
CAPE_LAT = 28.6084
CAPE_LON = -80.6043


def _cape_launch():
    return {
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
        "all_data": {"id": "shape-cape", "huge": True},
        "trajectory_data": {
            "launch_site": {"lat": CAPE_LAT, "lon": CAPE_LON, "name": "Cape Canaveral, FL"},
            "trajectory": [{"lat": CAPE_LAT, "lon": CAPE_LON, "r": 1.01, "sentinel": "kept"}],
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


class ExistingShapeTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()
        self._redis = app.r
        app.r = None

    def tearDown(self):
        app.r = self._redis
        cache_path = app.TRAJECTORY_CACHE_FILE
        if os.path.exists(cache_path):
            os.remove(cache_path)

    def _serve_launches(self, data):
        return patch.object(app, "get_cached_data", return_value=data), patch.object(
            app, "set_cached_data", return_value=True
        ), patch.object(app, "save_cache_to_file")

    def test_launches_slim_keeps_its_keys_and_drops_raw(self):
        first = _cape_launch()
        second = _cape_launch()
        second["id"] = "shape-cape-2"
        second["all_data"] = {"id": "shape-cape-2"}
        data = {
            "upcoming": [first, second],
            "previous": [_cape_launch()],
            "last_updated": "2026-11-01T00:00:00Z",
        }
        data["previous"][0]["id"] = "shape-prev"
        cached, persist, save = self._serve_launches(data)
        with cached, persist, save:
            slim = app.get_launches_slim(internal=True)

        self.assertEqual(set(slim), SLIM_TOP)
        launch = slim["upcoming"][0]
        self.assertNotIn("all_data", launch)
        self.assertNotIn("position_source", launch)
        self.assertNotIn("vehicles", launch)
        for key in (
            "id", "name", "mission", "net", "date", "time", "status",
            "rocket", "orbit", "pad", "video_url", "x_video_url", "trajectory_data",
        ):
            self.assertIn(key, launch)
        self.assertEqual(launch["trajectory_data"]["trajectory"][0]["sentinel"], "kept")
        self.assertNotIn("trajectory_data", slim["upcoming"][1])
        self.assertNotIn("all_data", slim["upcoming"][1])
        self.assertNotIn("trajectory_data", slim["previous"][0])
        self.assertNotIn("all_data", slim["previous"][0])

    def test_trajectory_wrapper_and_body_keys(self):
        data = {
            "upcoming": [_cape_launch()],
            "previous": [],
            "last_updated": "2026-11-01T00:00:00Z",
        }
        cached, persist, save = self._serve_launches(data)
        with cached, persist, save:
            body = app.get_trajectory(internal=True)
        self.assertEqual(set(body), TRAJECTORY_WRAP)
        self.assertEqual(body["launch_id"], "shape-cape")
        self.assertEqual(body["mission"], "Starlink shape pin")
        self.assertEqual(body["trajectory"]["trajectory"][0]["sentinel"], "kept")
        self.assertAlmostEqual(body["trajectory"]["launch_site"]["lat"], CAPE_LAT, places=4)
        self.assertAlmostEqual(body["trajectory"]["launch_site"]["lon"], CAPE_LON, places=4)

    def test_generated_trajectory_keys_and_cape_coordinates(self):
        launch = {
            "id": "shape-generated",
            "mission": "Starlink generated",
            "pad": "LC-39A",
            "orbit": "LEO",
            "net": "2026-12-01T00:00:00Z",
        }
        with patch.object(app, "save_cache_to_file"), patch.object(app, "r", None):
            traj = app.get_launch_trajectory_data(launch)
        self.assertEqual(set(traj), TRAJECTORY_BODY)
        self.assertAlmostEqual(traj["launch_site"]["lat"], CAPE_LAT, places=4)
        self.assertAlmostEqual(traj["launch_site"]["lon"], CAPE_LON, places=4)
        self.assertAlmostEqual(traj["trajectory"][0]["lat"], CAPE_LAT, places=2)

    def test_notify_copy_keys(self):
        payload = app.NotifyCopyRequest(
            launch_id="shape-notify",
            event="t1h",
            mission="Starlink Group 10-20",
            probability=70,
        )
        result = app._notify_copy_result(payload, "Starlink in 1 hour", "Liftoff 70% go.", False)
        self.assertEqual(set(result), NOTIFY_KEYS)
        self.assertEqual(result["launch_id"], "shape-notify")
        self.assertEqual(result["event"], "t1h")
        self.assertIs(result["cached"], False)
        self.assertEqual(result["model"], app.GROK_MODEL)
        self.assertLessEqual(len(result["title"]), 50)
        self.assertLessEqual(len(result["body"]), 150)

    def test_satellite_gp_meta_and_deployed_keys(self):
        catalog = app._satellite_payload("starlink", [{
            "name": "ISS (ZARYA)",
            "norad_id": 25544,
            "tle_line1": "1 25544U 98067A   26262.30395647  .00006211  00000+0  12007-3 0  9997",
            "tle_line2": "2 25544  51.6308 194.2901 0004815 157.3949 202.7252 15.49175317586346",
        }])
        self.assertEqual(set(catalog), SATELLITE_KEYS)
        self.assertEqual(set(catalog["satellites"][0]), SATELLITE_ROW)
        app._local_satellites["starlink"] = catalog
        served = app.get_satellites_gp(group="starlink", internal=True, request=None)
        self.assertEqual(set(served), SATELLITE_KEYS)
        self.assertEqual(served["satellites"], catalog["satellites"])
        meta = app.get_satellites_meta(group="starlink", internal=True)
        self.assertEqual(set(meta), META_KEYS)
        self.assertEqual(meta["group"], "starlink")
        self.assertEqual(meta["count"], 1)

        empty = app._no_v3_launch_payload()
        self.assertEqual(set(empty), DEPLOYED_KEYS)
        with patch.object(app, "_serve_launched_v3_union", return_value=(empty, True)):
            deployed = app.get_satellites_deployed(internal=True)
        self.assertEqual(set(deployed), DEPLOYED_KEYS)
        self.assertEqual(deployed["satellites"], [])
        self.assertNotIn("flights", deployed)
        self.assertNotIn("position_source", deployed)


class StarbasePadTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()

    def tearDown(self):
        cache_path = app.TRAJECTORY_CACHE_FILE
        if os.path.exists(cache_path):
            os.remove(cache_path)

    def test_starbase_family_is_not_lc39a_and_sits_on_the_surveyed_pad(self):
        self.assertAlmostEqual(app.STARBASE_LAUNCH_SITE["lat"], STARBASE_LAT, places=5)
        self.assertAlmostEqual(app.STARBASE_LAUNCH_SITE["lon"], STARBASE_LON, places=5)
        table = {"lat": 25.9975, "lon": -97.1566}
        cape = {"lat": CAPE_LAT, "lon": CAPE_LON}
        for pad in (
            "OLP-2",
            "Boca",
            "Boca Chica",
            "Orbital Launch Pad",
            "Orbital Launch Pad 2",
            "OLM-1",
            "OLM-2",
            "OLM-A",
        ):
            site, _key = app.resolve_launch_site(pad)
            self.assertIn("Starbase", site["name"], pad)
            self.assertLess(app._ang_dist_deg(site, table), 0.05, pad)
            self.assertGreater(app._ang_dist_deg(site, cape), 2.0, pad)
            self.assertAlmostEqual(site["lat"], STARBASE_LAT, places=5)
            self.assertAlmostEqual(site["lon"], STARBASE_LON, places=5)

    def test_api_track_more_than_two_degrees_from_the_pad_is_not_served(self):
        far_lat = STARBASE_LAT + 3.0
        launch = {
            "id": "olp-far",
            "mission": "Starship Flight 14",
            "pad": "Orbital Launch Pad 2",
            "orbit": "Low Earth Orbit",
            "net": "2026-12-01T00:00:00Z",
            "trajectory_data": {
                "launch_site": {"lat": far_lat, "lon": STARBASE_LON, "name": "Cape Canaveral, FL"},
                "trajectory": [
                    {"lat": far_lat, "lon": STARBASE_LON, "r": 1.01, "sentinel": "cape-track"},
                    {"lat": 20.0, "lon": -50.0, "r": 1.05},
                ],
                "booster_trajectory": [],
                "booster_ground_track": "along_track_return",
                "orbit_path": [],
            },
        }
        data = {"upcoming": [launch], "previous": [], "last_updated": "2026-11-01T00:00:00Z"}
        with patch.object(app, "get_cached_data", return_value=data), \
             patch.object(app, "set_cached_data", return_value=True), \
             patch.object(app, "save_cache_to_file"), \
             patch.object(app, "r", None):
            body = app.get_trajectory(internal=True)
        origin = body["trajectory"]["trajectory"][0]
        self.assertNotIn("sentinel", origin)
        self.assertIn("Starbase", body["trajectory"]["launch_site"]["name"])
        self.assertAlmostEqual(origin["lat"], STARBASE_LAT, delta=0.05)
        self.assertAlmostEqual(origin["lon"], STARBASE_LON, delta=0.05)
        self.assertGreater(abs(origin["lat"] - far_lat), 2.0)
        self.assertGreater(abs(origin["lat"] - CAPE_LAT), 1.0)
        self.assertGreater(abs(origin["lon"] - CAPE_LON), 1.0)

    def test_supplied_track_is_ignored_by_generation(self):
        """get_launch_trajectory_data builds from the pad. It does not replay a Cape track."""
        launch = {
            "mission": "Starship Flight 14",
            "pad": "OLP-2",
            "orbit": "LEO",
            "trajectory_data": {
                "trajectory": [
                    {"lat": CAPE_LAT, "lon": CAPE_LON, "r": 1.01},
                    {"lat": 20.0, "lon": -50.0, "r": 1.05},
                ],
            },
        }
        with patch.object(app, "save_cache_to_file"), patch.object(app, "r", None):
            traj = app.get_launch_trajectory_data(launch)
        origin = traj["trajectory"][0]
        self.assertIn("Starbase", traj["launch_site"]["name"])
        self.assertAlmostEqual(origin["lat"], STARBASE_LAT, delta=0.05)
        self.assertGreater(app._ang_dist_deg(origin, {"lat": CAPE_LAT, "lon": CAPE_LON}), 2.0)


if __name__ == "__main__":
    unittest.main()
