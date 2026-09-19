import json
import threading
import time
import unittest
from unittest.mock import Mock, patch

import app

app._background_enabled = False

ISS_OMM = {
    "OBJECT_NAME": "ISS (ZARYA)",
    "OBJECT_ID": "1998-067A",
    "EPOCH": "2026-09-19T07:17:41.839008",
    "MEAN_MOTION": 15.49175317,
    "ECCENTRICITY": 0.0004815,
    "INCLINATION": 51.6308,
    "RA_OF_ASC_NODE": 194.2901,
    "ARG_OF_PERICENTER": 157.3949,
    "MEAN_ANOMALY": 202.7252,
    "EPHEMERIS_TYPE": 0,
    "CLASSIFICATION_TYPE": "U",
    "NORAD_CAT_ID": 25544,
    "ELEMENT_SET_NO": 999,
    "REV_AT_EPOCH": 58634,
    "BSTAR": 0.00012007,
    "MEAN_MOTION_DOT": 6.211e-5,
    "MEAN_MOTION_DDOT": 0,
}

STARLINK_OMM = {
    "OBJECT_NAME": "STARLINK-1008",
    "OBJECT_ID": "2019-074B",
    "EPOCH": "2026-09-19T12:00:00.000000",
    "MEAN_MOTION": 15.06400000,
    "ECCENTRICITY": 0.0001234,
    "INCLINATION": 53.0500,
    "RA_OF_ASC_NODE": 100.0000,
    "ARG_OF_PERICENTER": 50.0000,
    "MEAN_ANOMALY": 310.0000,
    "EPHEMERIS_TYPE": 0,
    "CLASSIFICATION_TYPE": "U",
    "NORAD_CAT_ID": 44714,
    "ELEMENT_SET_NO": 999,
    "REV_AT_EPOCH": 12345,
    "BSTAR": 0.0001,
    "MEAN_MOTION_DOT": 1.0e-5,
    "MEAN_MOTION_DDOT": 0,
}

FREGAT_OMM = {
    "OBJECT_NAME": "FREGAT DEB",
    "OBJECT_ID": "2011-037PF",
    "EPOCH": "2026-09-19T04:41:33.191520",
    "MEAN_MOTION": 12.44551487,
    "ECCENTRICITY": 0.09440644,
    "INCLINATION": 51.6468,
    "RA_OF_ASC_NODE": 78.697,
    "ARG_OF_PERICENTER": 131.6853,
    "MEAN_ANOMALY": 236.8792,
    "EPHEMERIS_TYPE": 0,
    "CLASSIFICATION_TYPE": "U",
    "NORAD_CAT_ID": 49271,
    "ELEMENT_SET_NO": 999,
    "REV_AT_EPOCH": 23674,
    "BSTAR": 0.015154405,
    "MEAN_MOTION_DOT": 0.00010835,
    "MEAN_MOTION_DDOT": 0,
}

HIGH_NORAD_OMM = {
    "OBJECT_NAME": "SOYUZ-MS 29",
    "OBJECT_ID": "2026-162A",
    "EPOCH": "2026-09-19T04:11:54.904992",
    "MEAN_MOTION": 15.49173263,
    "ECCENTRICITY": 0.00048063,
    "INCLINATION": 51.6309,
    "RA_OF_ASC_NODE": 194.9288,
    "ARG_OF_PERICENTER": 156.4521,
    "MEAN_ANOMALY": 203.6688,
    "EPHEMERIS_TYPE": 0,
    "CLASSIFICATION_TYPE": "U",
    "NORAD_CAT_ID": 100057,
    "ELEMENT_SET_NO": 999,
    "REV_AT_EPOCH": 58674,
    "BSTAR": 0.00011723079,
    "MEAN_MOTION_DOT": 6.053e-5,
    "MEAN_MOTION_DDOT": 0,
}


def _json_response(payload, status_code=200):
    response = Mock()
    response.status_code = status_code
    response.raise_for_status = Mock()
    if status_code >= 400:
        response.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    response.json.return_value = payload
    return response


class SatelliteGpTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()

    def test_omm_to_tle_matches_celestrak_iss_fields(self):
        line1, line2 = app.omm_to_tle_lines(ISS_OMM)

        self.assertEqual(len(line1), 69)
        self.assertEqual(len(line2), 69)
        self.assertEqual(line1[0], "1")
        self.assertEqual(line2[0], "2")
        self.assertEqual(app._tle_checksum(line1), line1[68])
        self.assertEqual(app._tle_checksum(line2), line2[68])
        self.assertIn("25544U", line1)
        self.assertIn("98067A", line1)
        self.assertIn("26262.30395647", line1)
        self.assertIn(".00006211", line1)
        self.assertIn("12007-3", line1)
        self.assertIn("51.6308", line2)
        self.assertIn("194.2901", line2)
        self.assertIn("0004815", line2)
        self.assertIn("15.49175317", line2)
        self.assertIn("58634", line2)

    def test_omm_to_tle_handles_long_object_id_and_alpha5_norad(self):
        fregat1, fregat2 = app.omm_to_tle_lines(FREGAT_OMM)
        high1, high2 = app.omm_to_tle_lines(HIGH_NORAD_OMM)

        self.assertIn("11037PF", fregat1)
        self.assertIn("0944064", fregat2)
        self.assertIn("15154-1", fregat1)
        self.assertTrue(high1.startswith("1 A0057U"))
        self.assertTrue(high2.startswith("2 A0057"))
        self.assertEqual(len(high1), 69)
        self.assertEqual(len(high2), 69)

    def test_slim_gp_record_is_compact_tle_shape(self):
        record = app.slim_gp_record(ISS_OMM)
        self.assertEqual(set(record), {"name", "norad_id", "tle_line1", "tle_line2"})
        self.assertEqual(record["name"], "ISS (ZARYA)")
        self.assertEqual(record["norad_id"], 25544)
        self.assertEqual(len(record["tle_line1"]), 69)
        self.assertEqual(len(record["tle_line2"]), 69)

    def test_group_validation_rejects_unknown_and_accepts_allowlist(self):
        self.assertEqual(app._normalize_satellite_group("Starlink"), "starlink")
        self.assertEqual(app._normalize_satellite_group("stations"), "stations")
        with self.assertRaises(app.HTTPException) as ctx:
            app._normalize_satellite_group("active")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Allowed", ctx.exception.detail)

    def test_cache_miss_fetches_celestrak_and_hit_skips_upstream(self):
        fake = _json_response([ISS_OMM, STARLINK_OMM])
        with patch.object(app, "r", None), \
             patch.object(app.requests, "get", return_value=fake) as get:
            first = app.get_satellites_gp(group="starlink", internal=True)
            second = app.get_satellites_gp(group="starlink", internal=True)

        self.assertEqual(get.call_count, 1)
        get.assert_called_with(
            app.CELESTRAK_GP_URL,
            params={"GROUP": "starlink", "FORMAT": "JSON"},
            headers=app.CELESTRAK_HEADERS,
            timeout=app.SATELLITES_FETCH_TIMEOUT,
        )
        self.assertEqual(first["count"], 2)
        self.assertEqual(first["group"], "starlink")
        self.assertFalse(first["stale"])
        self.assertEqual(first["satellites"][0]["name"], "ISS (ZARYA)")
        self.assertIn("tle_line1", first["satellites"][0])
        self.assertNotIn("MEAN_MOTION", first["satellites"][0])
        self.assertEqual(second["count"], 2)
        self.assertEqual(second["satellites"][1]["norad_id"], 44714)
        self.assertIn("SGP4", first["note"])

    def test_stale_cache_served_when_celestrak_fails(self):
        ok = _json_response([ISS_OMM])
        with patch.object(app, "r", None), \
             patch.object(app.requests, "get", return_value=ok):
            fresh = app.refresh_satellites_internal("stations")

        fresh["fetched_at"] = "2020-01-01T00:00:00Z"
        app._local_satellites["stations"] = fresh

        with patch.object(app, "r", None), \
             patch.object(app.requests, "get", side_effect=RuntimeError("celestrak down")):
            stale = app.get_satellites_gp(group="stations", internal=True)

        self.assertTrue(stale["stale"])
        self.assertEqual(stale["count"], 1)
        self.assertEqual(stale["satellites"][0]["norad_id"], 25544)

    def test_upstream_failure_without_cache_is_502(self):
        from fastapi.testclient import TestClient

        with patch.object(app, "r", None), \
             patch.object(app.requests, "get", side_effect=RuntimeError("celestrak down")), \
             patch.object(app, "_background_enabled", False):
            client = TestClient(app.app)
            response = client.get("/satellites/gp", params={"group": "starlink"})

        self.assertEqual(response.status_code, 502)
        self.assertIn("unavailable", response.json()["detail"].lower())

    def test_http_alias_meta_and_group_validation(self):
        from fastapi.testclient import TestClient

        fake = _json_response([STARLINK_OMM])
        with patch.object(app, "r", None), \
             patch.object(app.requests, "get", return_value=fake) as get, \
             patch.object(app, "_background_enabled", False):
            client = TestClient(app.app)
            alias = client.get("/satellites/starlink")
            gp = client.get("/satellites/gp", params={"group": "starlink"})
            meta = client.get("/satellites/meta", params={"group": "starlink"})
            bad = client.get("/satellites/gp", params={"group": "iridium"})
            empty_meta = client.get("/satellites/meta", params={"group": "oneweb"})

        self.assertEqual(alias.status_code, 200)
        self.assertEqual(gp.status_code, 200)
        self.assertEqual(alias.json()["group"], "starlink")
        self.assertEqual(alias.json()["satellites"][0]["name"], "STARLINK-1008")
        self.assertEqual(get.call_count, 1)

        meta_body = meta.json()
        self.assertEqual(meta_body["group"], "starlink")
        self.assertEqual(meta_body["count"], 1)
        self.assertFalse(meta_body["stale"])
        self.assertIsNotNone(meta_body["fetched_at"])
        self.assertEqual(meta_body["ttl_seconds"], app.SATELLITES_CACHE_TTL)
        self.assertIn("starlink", meta_body["allowed_groups"])
        self.assertIn("SGP4", meta_body["note"])

        self.assertEqual(bad.status_code, 400)
        self.assertIn("Allowed", bad.json()["detail"])
        self.assertEqual(empty_meta.status_code, 200)
        self.assertEqual(empty_meta.json()["count"], 0)
        self.assertTrue(empty_meta.json()["stale"])
        self.assertIsNone(empty_meta.json()["fetched_at"])

    def test_force_bypasses_fresh_cache(self):
        first = _json_response([ISS_OMM])
        second = _json_response([STARLINK_OMM])
        with patch.object(app, "r", None), \
             patch.object(app.requests, "get", side_effect=[first, second]) as get:
            initial = app.get_satellites_gp(group="visual", internal=True)
            forced = app.get_satellites_gp(group="visual", force=True, internal=True)

        self.assertEqual(get.call_count, 2)
        self.assertEqual(initial["satellites"][0]["norad_id"], 25544)
        self.assertEqual(forced["satellites"][0]["norad_id"], 44714)

    def test_uses_set_cached_data_with_stale_ttl(self):
        persisted = []

        def fake_set(key, data, ttl=None):
            persisted.append((key, data, ttl))
            return True

        fake = _json_response([ISS_OMM])
        with patch.object(app, "r", None), \
             patch.object(app, "get_cached_data", return_value=None), \
             patch.object(app, "set_cached_data", side_effect=fake_set), \
             patch.object(app.requests, "get", return_value=fake):
            app.refresh_satellites_internal("weather")

        self.assertEqual(len(persisted), 1)
        key, data, ttl = persisted[0]
        self.assertEqual(key, "satellites_gp_v1:weather")
        self.assertEqual(ttl, app.SATELLITES_STALE_TTL)
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["satellites"][0]["name"], "ISS (ZARYA)")

    def test_redis_stale_fallback_when_memory_empty(self):
        stale_payload = {
            "group": "gps-ops",
            "fetched_at": "2020-01-01T00:00:00Z",
            "ttl_seconds": app.SATELLITES_CACHE_TTL,
            "count": 1,
            "stale": False,
            "satellites": [app.slim_gp_record(ISS_OMM)],
        }
        with patch.object(app, "r", None), \
             patch.object(app, "get_cached_data", return_value=stale_payload), \
             patch.object(app.requests, "get", side_effect=RuntimeError("down")):
            result = app.get_satellites_gp(group="gps-ops", internal=True)

        self.assertTrue(result["stale"])
        self.assertEqual(result["satellites"][0]["norad_id"], 25544)

    def test_single_flight_coalesces_concurrent_gp_fetches(self):
        entered = threading.Barrier(4)
        calls = []

        def fake_get(*args, **kwargs):
            calls.append(1)
            time.sleep(0.08)
            return _json_response([STARLINK_OMM])

        results = [None] * 4

        def worker(idx):
            entered.wait(timeout=2)
            results[idx] = app.refresh_satellites_internal("oneweb")

        with patch.object(app, "r", None), \
             patch.object(app.requests, "get", side_effect=fake_get):
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

        self.assertEqual(calls, [1])
        self.assertTrue(all(item and item["count"] == 1 for item in results))

    def test_openapi_documents_satellite_routes(self):
        schema = app.app.openapi()
        paths = schema["paths"]
        self.assertIn("/satellites/gp", paths)
        self.assertIn("/satellites/starlink", paths)
        self.assertIn("/satellites/meta", paths)
        dumped = json.dumps(schema)
        self.assertIn("Satellites", dumped)
        self.assertIn("starlink", dumped)
        self.assertIn("SGP4", dumped)


if __name__ == "__main__":
    unittest.main()
