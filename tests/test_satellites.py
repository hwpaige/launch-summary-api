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


def _text_response(text, status_code=200):
    response = Mock()
    response.status_code = status_code
    response.text = text
    response.raise_for_status = Mock()
    if status_code >= 400:
        response.raise_for_status.side_effect = Exception(f"HTTP {status_code}")

    def iter_lines(decode_unicode=False):
        for line in text.splitlines():
            yield line

    response.iter_lines = iter_lines
    response.close = Mock()
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

        keys = [item[0] for item in persisted]
        self.assertIn("satellites_gp_v1:weather", keys)
        self.assertIn("satellites_gp_http_v1:weather", keys)
        gp = next(item for item in persisted if item[0] == "satellites_gp_v1:weather")
        http = next(item for item in persisted if item[0] == "satellites_gp_http_v1:weather")
        self.assertEqual(gp[2], app.SATELLITES_STALE_TTL)
        self.assertEqual(http[2], app.SATELLITES_STALE_TTL)
        self.assertEqual(gp[1]["count"], 1)
        self.assertEqual(gp[1]["satellites"][0]["name"], "ISS (ZARYA)")
        self.assertIn("gzip_fresh", http[1])
        self.assertIn("gzip_stale", http[1])

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
        self.assertIn("/satellites/deployed", paths)
        dumped = json.dumps(schema)
        self.assertIn("Satellites", dumped)
        self.assertIn("starlink", dumped)
        self.assertIn("SGP4", dumped)

    def test_stale_catalog_does_not_wait_on_celestrak(self):
        payload = app._satellite_payload(
            "starlink", [app.slim_gp_record(STARLINK_OMM)]
        )
        payload["fetched_at"] = "2020-01-01T00:00:00Z"
        app._local_satellites["starlink"] = payload

        def slow_get(*args, **kwargs):
            time.sleep(5)
            return _json_response([ISS_OMM])

        from fastapi.testclient import TestClient

        started = time.perf_counter()
        with patch.object(app, "r", None), \
             patch.object(app.requests, "get", side_effect=slow_get) as get, \
             patch.object(app, "_background_enabled", False):
            client = TestClient(app.app)
            response = client.get(
                "/satellites/starlink",
                headers={"Accept-Encoding": "identity"},
            )
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 2.0)
        self.assertEqual(get.call_count, 0)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("content-encoding", response.headers)
        self.assertEqual(response.headers.get("x-satellite-cache"), "stale")
        body = response.json()
        self.assertTrue(body["stale"])
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["satellites"][0]["norad_id"], 44714)
        self.assertEqual(response.content, app._local_satellite_http["starlink"]["json_stale"])

    def test_gzip_starlink_body_is_precomputed(self):
        import gzip as gzip_mod
        from fastapi.testclient import TestClient

        payload = app._satellite_payload("starlink", [app.slim_gp_record(ISS_OMM)])
        app._write_satellite_cache("starlink", payload)
        with patch.object(app, "r", None), \
             patch.object(app.requests, "get", side_effect=AssertionError("upstream")), \
             patch.object(app, "_background_enabled", False):
            client = TestClient(app.app)
            response = client.get(
                "/satellites/gp",
                params={"group": "starlink"},
                headers={"Accept-Encoding": "gzip"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("content-encoding"), "gzip")
        self.assertEqual(response.headers.get("x-satellite-cache"), "hit")
        self.assertIn("accept-encoding", response.headers.get("vary", "").lower())
        raw = response.content
        # httpx may already have decompressed. Accept either the gzip bytes
        # or the decoded JSON, and require both to match the precomputed body.
        doc = app._local_satellite_http["starlink"]
        if raw[:2] == b"\x1f\x8b":
            self.assertEqual(raw, doc["gzip_fresh"])
            decoded = gzip_mod.decompress(raw)
        else:
            decoded = raw
        self.assertEqual(decoded, doc["json_fresh"])
        self.assertFalse(json.loads(decoded)["stale"])
        self.assertEqual(json.loads(decoded)["count"], 1)

    def test_cold_gp_deadline_is_json_502_not_a_hang(self):
        from fastapi.testclient import TestClient

        def slow_get(*args, **kwargs):
            time.sleep(3)
            return _json_response([STARLINK_OMM])

        started = time.perf_counter()
        with patch.object(app, "r", None), \
             patch.object(app, "SATELLITES_REQUEST_BUDGET_SEC", 0.35), \
             patch.object(app.requests, "get", side_effect=slow_get), \
             patch.object(app, "_background_enabled", False):
            client = TestClient(app.app)
            response = client.get("/satellites/starlink")
        elapsed = time.perf_counter() - started

        self.assertEqual(response.status_code, 502)
        self.assertLess(elapsed, 2.0)
        self.assertIn("unavailable", response.json()["detail"].lower())
        self.assertIn("application/json", response.headers.get("content-type", ""))

    def test_stale_cache_schedules_one_background_refresh(self):
        payload = app._satellite_payload("stations", [app.slim_gp_record(ISS_OMM)])
        payload["fetched_at"] = "2020-01-01T00:00:00Z"
        app._local_satellites["stations"] = payload
        started = threading.Event()

        def fake_refresh(group="starlink", force=False, wait_timeout=None, deadline=None):
            started.set()
            return payload

        with patch.object(app, "r", None), \
             patch.object(app, "_background_enabled", True), \
             patch.object(app, "refresh_satellites_internal", side_effect=fake_refresh):
            first = app.get_satellites_gp(group="stations", internal=True)
            second = app.get_satellites_gp(group="stations", internal=True)
            self.assertTrue(started.wait(1.0))

        self.assertTrue(first["stale"])
        self.assertTrue(second["stale"])
        self.assertEqual(first["satellites"][0]["norad_id"], 25544)

    def test_http_bytes_roundtrip_without_satellite_objects(self):
        payload = app._satellite_payload("oneweb", [app.slim_gp_record(STARLINK_OMM)])
        doc = app._render_satellite_http(payload)
        blob = {
            "fetched_at": doc["fetched_at"],
            "count": doc["count"],
            "gzip_fresh": __import__("base64").b64encode(doc["gzip_fresh"]).decode("ascii"),
            "gzip_stale": __import__("base64").b64encode(doc["gzip_stale"]).decode("ascii"),
        }

        def cached(key):
            if key == app._satellite_http_cache_key("oneweb"):
                return blob
            raise AssertionError(f"unexpected cache read {key}")

        from fastapi.testclient import TestClient

        with patch.object(app, "r", None), \
             patch.object(app, "get_cached_data", side_effect=cached), \
             patch.object(app.requests, "get", side_effect=AssertionError("upstream")), \
             patch.object(app, "_background_enabled", False):
            client = TestClient(app.app)
            response = client.get(
                "/satellites/gp",
                params={"group": "oneweb"},
                headers={"Accept-Encoding": "identity"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, doc["json_fresh"])
        self.assertEqual(response.json()["count"], 1)
        self.assertEqual(response.json()["satellites"][0]["name"], "STARLINK-1008")


def _satcat_row(name, norad, object_id, launch_date, object_type="PAY", decay=""):
    return {
        "OBJECT_NAME": name,
        "OBJECT_ID": object_id,
        "NORAD_CAT_ID": norad,
        "OBJECT_TYPE": object_type,
        "LAUNCH_DATE": launch_date,
        "DECAY_DATE": decay,
    }


def _v3_omm(norad=100753, name="STARLINK-38381", object_id="2026-219A"):
    omm = dict(STARLINK_OMM)
    omm["OBJECT_NAME"] = name
    omm["OBJECT_ID"] = object_id
    omm["NORAD_CAT_ID"] = norad
    return omm


FLIGHT_14 = {
    "id": "flight-14",
    "mission": "Starship | Starlink Group 31-1 (Starship Flight 14)",
    "name": "Starship | Starlink Group 31-1 (Starship Flight 14)",
    "rocket": "Starship",
    "net": "2026-09-28T12:48:59Z",
    "status": "Launch in Flight",
}


class DeployedSatelliteTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()

    def test_flight_14_group_31_is_the_v3_launch(self):
        from datetime import datetime, timezone

        now = datetime(2026, 9, 28, 16, 0, tzinfo=timezone.utc)
        flight_13 = {
            "id": "flight-13",
            "mission": "Starship | Flight 13",
            "rocket": "Starship",
            "net": "2026-07-24T22:51:00Z",
        }
        chosen = app.select_v3_starship([FLIGHT_14], [flight_13, FLIGHT_14], now=now)
        self.assertEqual(chosen["id"], "flight-14")
        self.assertEqual(app.deployed_generation(app._launch_blob(chosen)), "v3")
        self.assertIsNone(app.deployed_generation(app._launch_blob(flight_13)))

    def test_slim_satcat_index_skips_decayed_debris_and_other_dates(self):
        from datetime import datetime, timezone

        now = datetime(2026, 9, 28, tzinfo=timezone.utc)
        rows = [
            _satcat_row("STARLINK-38381", 100753, "2026-219A", "2026-09-28"),
            _satcat_row("STARLINK-38381", 100753, "2026-219A", "2026-09-28"),
            _satcat_row("STARLINK-DEAD", 100701, "2026-219C", "2026-09-28", decay="2026-09-28"),
            _satcat_row("STARLINK R/B", 100702, "2026-219D", "2026-09-28", object_type="R/B"),
            _satcat_row("STARLINK DEB", 100703, "2026-219E", "2026-09-28", object_type="DEB"),
            _satcat_row("STARLINK-OLD", 100700, "2026-200A", "2026-09-20"),
            _satcat_row("NOT-A-STARLINK", 100799, "2026-219F", "2026-09-28"),
        ]
        index = app.slim_satcat_index(rows, now=now, extra_dates=["2020-01-01"])
        self.assertEqual([row["norad_id"] for row in index["dates"]["2026-09-28"]], [100753])
        self.assertEqual(index["dates"]["2026-09-20"][0]["norad_id"], 100700)
        self.assertEqual(index["dates"]["2020-01-01"], [])
        self.assertNotIn("2020-01-02", index["dates"])

    def _route_celestrak(
        self,
        satcat_rows,
        gp_by_intdes=None,
        sup_by_intdes=None,
        manifest="",
        ephemerides=None,
    ):
        gp_by_intdes = gp_by_intdes or {}
        sup_by_intdes = sup_by_intdes or {}
        ephemerides = ephemerides or {}
        calls = []

        def fake_get(url, params=None, **kwargs):
            calls.append((url, dict(params or {})))
            if url == app.CELESTRAK_SATCAT_URL:
                return _json_response(satcat_rows)
            if url == app.SPACEX_MANIFEST_URL:
                return _text_response(manifest)
            if isinstance(url, str) and url.startswith(app.SPACEX_EPHEM_BASE):
                name = url.rsplit("/", 1)[-1]
                if name not in ephemerides:
                    raise AssertionError(url)
                return _text_response(ephemerides[name])
            intdes = (params or {}).get("INTDES")
            if url == app.CELESTRAK_GP_URL:
                return _json_response(gp_by_intdes.get(intdes, []))
            if url == app.CELESTRAK_SUP_GP_URL:
                return _json_response(sup_by_intdes.get(intdes, []))
            raise AssertionError(url)

        return fake_get, calls

    def test_flight_14_empty_satcat_is_honest_and_does_not_fetch_gp(self):
        rows = [_satcat_row("STARLINK-OLD", 100700, "2026-200A", "2026-09-20")]
        fake_get, calls = self._route_celestrak(rows)
        launches = {"upcoming": [FLIGHT_14], "previous": [FLIGHT_14], "last_updated": None}
        with patch.object(app, "r", None), \
             patch.object(app, "_load_launch_payload", return_value=launches), \
             patch.object(app.requests, "get", side_effect=fake_get):
            body = app.get_satellites_deployed(internal=True)

        self.assertEqual(body["launch_date"], "2026-09-28")
        self.assertEqual(body["generation"], "v3")
        self.assertIn("Group 31-1", body["mission"])
        self.assertEqual(body["satellites"], [])
        self.assertEqual(body["count"], 0)
        self.assertEqual(body["catalog_count"], 0)
        self.assertTrue(body["empty"])
        self.assertFalse(body["stale"])
        self.assertEqual(body["source"], "celestrak-satcat")
        self.assertEqual(body["ttl_seconds"], app.DEPLOYED_EMPTY_TTL)
        self.assertNotIn("unavailable", body["note"].lower())
        self.assertIn("2026-09-28", body["note"])
        self.assertIn("SGP4", body["note"])
        self.assertEqual(
            [url for url, _params in calls],
            [app.CELESTRAK_SATCAT_URL, app.SPACEX_MANIFEST_URL],
        )

    def test_joins_real_tle_and_ignores_unrelated_gp_objects(self):
        rows = [
            _satcat_row("STARLINK-38381", 100753, "2026-219A", "2026-09-28"),
            _satcat_row("STARLINK-OLD", 100700, "2026-200A", "2026-09-20"),
            _satcat_row("STARLINK-DEAD", 100701, "2026-219C", "2026-09-28", decay="2026-09-28"),
        ]
        # ISS is a real OMM but it is not in the SATCAT match, so it must not appear.
        fake_get, calls = self._route_celestrak(
            rows,
            gp_by_intdes={"2026-219": [_v3_omm(), ISS_OMM]},
        )
        launches = {"upcoming": [FLIGHT_14], "previous": [], "last_updated": None}
        with patch.object(app, "r", None), \
             patch.object(app, "_load_launch_payload", return_value=launches), \
             patch.object(app.requests, "get", side_effect=fake_get):
            body = app.get_satellites_deployed(launch_date="2026-09-28", internal=True)

        self.assertEqual(body["catalog_count"], 1)
        self.assertEqual(body["count"], 1)
        self.assertFalse(body["empty"])
        self.assertFalse(body["stale"])
        sat = body["satellites"][0]
        self.assertEqual(set(sat), {"name", "norad_id", "tle_line1", "tle_line2", "generation"})
        self.assertEqual(sat["norad_id"], 100753)
        self.assertEqual(sat["name"], "STARLINK-38381")
        self.assertEqual(sat["generation"], "v3")
        self.assertEqual(len(sat["tle_line1"]), 69)
        self.assertEqual(len(sat["tle_line2"]), 69)
        self.assertTrue(sat["tle_line1"].startswith("1 A0753U"))
        self.assertTrue(sat["tle_line2"].startswith("2 A0753"))
        self.assertNotIn(25544, [row["norad_id"] for row in body["satellites"]])
        self.assertFalse(any(url == app.SPACEX_MANIFEST_URL for url, _params in calls))
        gp_calls = [params for url, params in calls if url == app.CELESTRAK_GP_URL]
        self.assertEqual(gp_calls, [{"INTDES": "2026-219", "FORMAT": "JSON"}])
        self.assertFalse(any(url == app.CELESTRAK_SUP_GP_URL for url, _params in calls))

    def test_supplemental_gp_fills_norads_missing_from_main_gp(self):
        rows = [_satcat_row("STARLINK-38381", 100753, "2026-219A", "2026-09-28")]
        fake_get, calls = self._route_celestrak(
            rows,
            gp_by_intdes={"2026-219": []},
            sup_by_intdes={"2026-219": [_v3_omm()]},
        )
        with patch.object(app, "r", None), \
             patch.object(app.requests, "get", side_effect=fake_get):
            body = app.get_satellites_deployed(launch_date="2026-09-28", internal=True)

        self.assertEqual(body["count"], 1)
        self.assertEqual(body["satellites"][0]["norad_id"], 100753)
        self.assertTrue(any(url == app.CELESTRAK_SUP_GP_URL for url, _params in calls))

    def test_cached_starlink_gp_skips_intdes_fetch(self):
        omm = _v3_omm()
        app._local_satellites["starlink"] = {
            "fetched_at": app._utc_isoformat(),
            "satellites": [app.slim_gp_record(omm)],
        }
        rows = [_satcat_row("STARLINK-38381", 100753, "2026-219A", "2026-09-28")]
        fake_get, calls = self._route_celestrak(rows, gp_by_intdes={"2026-219": [_v3_omm()]})
        with patch.object(app, "r", None), \
             patch.object(app.requests, "get", side_effect=fake_get):
            body = app.get_satellites_deployed(launch_date="2026-09-28", internal=True)

        self.assertEqual(body["satellites"][0]["norad_id"], 100753)
        self.assertEqual([url for url, _params in calls], [app.CELESTRAK_SATCAT_URL])

    def test_catalog_without_tle_is_not_unavailable(self):
        rows = [_satcat_row("STARLINK-38381", 100753, "2026-219A", "2026-09-28")]
        fake_get, _calls = self._route_celestrak(rows)
        with patch.object(app, "r", None), \
             patch.object(app.requests, "get", side_effect=fake_get):
            body = app.get_satellites_deployed(launch_date="2026-09-28", internal=True)

        self.assertEqual(body["catalog_count"], 1)
        self.assertEqual(body["count"], 0)
        self.assertEqual(body["satellites"], [])
        self.assertTrue(body["empty"])
        self.assertFalse(body["stale"])
        self.assertNotIn("unavailable", body["note"].lower())
        self.assertIn("no GP TLE", body["note"])

    def test_satcat_outage_without_cache_is_empty_and_flagged(self):
        from fastapi.testclient import TestClient

        with patch.object(app, "r", None), \
             patch.object(app.requests, "get", side_effect=RuntimeError("celestrak down")), \
             patch.object(app, "_background_enabled", False):
            client = TestClient(app.app)
            response = client.get("/satellites/deployed", params={"launch_date": "2026-09-28"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["satellites"], [])
        self.assertEqual(body["catalog_count"], 0)
        self.assertTrue(body["empty"])
        self.assertTrue(body["stale"])
        self.assertIn("unavailable", body["note"].lower())
        self.assertEqual(body["source"], "celestrak-satcat")

    def test_satcat_outage_serves_stale_cache_without_changing_the_note(self):
        rows = [_satcat_row("STARLINK-38381", 100753, "2026-219A", "2026-09-20")]
        fake_get, _calls = self._route_celestrak(
            rows,
            gp_by_intdes={"2026-219": [_v3_omm()]},
        )
        with patch.object(app, "r", None), \
             patch.object(app.requests, "get", side_effect=fake_get):
            fresh = app.get_satellites_deployed(launch_date="2026-09-20", internal=True)
        self.assertEqual(fresh["count"], 1)
        fresh_note = fresh["note"]
        app._local_deployed["2026-09-20"]["fetched_at"] = "2020-01-01T00:00:00Z"
        app._local_satcat_index["fetched_at"] = "2020-01-01T00:00:00Z"

        with patch.object(app, "r", None), \
             patch.object(app.requests, "get", side_effect=RuntimeError("celestrak down")):
            stale = app.get_satellites_deployed(launch_date="2026-09-20", internal=True)

        self.assertTrue(stale["stale"])
        self.assertEqual(stale["satellites"][0]["norad_id"], 100753)
        self.assertEqual(stale["note"], fresh_note)
        self.assertNotIn("unavailable", stale["note"].lower())

    def test_fresh_cache_skips_a_second_satcat_download(self):
        fake_get, calls = self._route_celestrak([])
        with patch.object(app, "r", None), \
             patch.object(app.requests, "get", side_effect=fake_get):
            first = app.get_satellites_deployed(launch_date="2026-09-28", internal=True)
            second = app.get_satellites_deployed(launch_date="2026-09-28", internal=True)

        self.assertTrue(first["empty"])
        self.assertTrue(second["empty"])
        self.assertEqual(
            [url for url, _params in calls],
            [app.CELESTRAK_SATCAT_URL, app.SPACEX_MANIFEST_URL],
        )

    def test_no_v3_launch_does_not_query_celestrak(self):
        launches = {
            "upcoming": [{
                "mission": "Falcon 9 Block 5 | Starlink Group 10-20",
                "rocket": "Falcon 9",
                "net": "2026-09-28T00:00:00Z",
            }],
            "previous": [{
                "mission": "Starship | Flight 13",
                "rocket": "Starship",
                "net": "2026-07-24T22:51:00Z",
            }],
        }
        with patch.object(app, "r", None), \
             patch.object(app, "_load_launch_payload", return_value=launches), \
             patch.object(app.requests, "get", side_effect=AssertionError("upstream")):
            body = app.get_satellites_deployed(internal=True)

        self.assertIsNone(body["launch_date"])
        self.assertIsNone(body["generation"])
        self.assertTrue(body["empty"])
        self.assertFalse(body["stale"])
        self.assertEqual(body["satellites"], [])
        self.assertNotIn("unavailable", body["note"].lower())
        self.assertIn("No Starship V3", body["note"])

    def test_bad_launch_date_is_400(self):
        from fastapi.testclient import TestClient

        with patch.object(app, "r", None), \
             patch.object(app, "_background_enabled", False):
            client = TestClient(app.app)
            response = client.get("/satellites/deployed", params={"launch_date": "09-28-2026"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("YYYY-MM-DD", response.json()["detail"])

    def test_same_day_second_launch_is_included(self):
        rows = [
            _satcat_row("STARLINK-38381", 100753, "2026-219A", "2026-09-28"),
            _satcat_row("STARLINK-100", 44714, "2026-220A", "2026-09-28"),
        ]
        fake_get, calls = self._route_celestrak(
            rows,
            gp_by_intdes={
                "2026-219": [_v3_omm()],
                "2026-220": [_v3_omm(44714, "STARLINK-100", "2026-220A")],
            },
        )
        with patch.object(app, "r", None), \
             patch.object(app.requests, "get", side_effect=fake_get):
            body = app.get_satellites_deployed(launch_date="2026-09-28", internal=True)

        self.assertEqual(body["catalog_count"], 2)
        self.assertEqual(
            [sat["norad_id"] for sat in body["satellites"]],
            [44714, 100753],
        )
        intdes = sorted(params["INTDES"] for url, params in calls if url == app.CELESTRAK_GP_URL)
        self.assertEqual(intdes, ["2026-219", "2026-220"])


from datetime import datetime, timezone


# Two inertial samples bracketing 2026-09-28 16:59:12 UTC (Flight 14 MEME shape).
_MEME_SAMPLE = """created:2026-09-28 17:01:25 UTC
ephemeris_start:2026-09-28 16:58:42 UTC ephemeris_stop:2026-09-30 16:57:42 UTC step_size:60
ephemeris_source:blend
UVW
2026271165842.000 6514.9476854708 -391.0827173447 1215.5516556776 -0.2750437930 6.8311421476 3.6645843404
0 0 0 0 0 0 0
0 0 0 0 0 0 0
0 0 0 0 0 0 0
2026271165942.000 6482.4712625142 19.4108315434 1432.2542472462 -0.8070548990 6.8463729972 3.5558690285
0 0 0 0 0 0 0
0 0 0 0 0 0 0
0 0 0 0 0 0 0
"""

_MEME_OLD = """created:2020-01-01 00:00:00 UTC
ephemeris_start:2020-01-01 00:00:00 UTC ephemeris_stop:2020-01-02 00:00:00 UTC step_size:60
ephemeris_source:blend
UVW
2020001000000.000 6514.9476854708 -391.0827173447 1215.5516556776 -0.2750437930 6.8311421476 3.6645843404
0 0 0 0 0 0 0
0 0 0 0 0 0 0
0 0 0 0 0 0 0
"""

# Announced Flight 14 V3 set: 40075–40088, 40090–40095, 40097–40101, 40103.
F14_STARLINK_IDS = (
    list(range(40075, 40089))
    + list(range(40090, 40096))
    + list(range(40097, 40102))
    + [40103]
)


def _f14_filename(sid):
    return f"MEME_{sid}_STARLINK-{sid}_2711658_Operational_1_UNCLASSIFIED.txt"


class _FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        stamp = cls(2026, 9, 28, 16, 59, 12, tzinfo=timezone.utc)
        if tz is not None:
            return stamp.astimezone(tz)
        return stamp


class ManifestEphemerisTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()

    def test_manifest_keeps_only_the_post_falcon_band(self):
        text = "\n".join([
            "MEME_1_STARLINK-1008_2710000_Operational_1_UNCLASSIFIED.txt",
            "MEME_2_STARLINK-38451_2710000_Operational_1_UNCLASSIFIED.txt",
            _f14_filename(40075),
            _f14_filename(40103),
            "not a file",
        ])
        self.assertEqual(
            [sid for sid, _name in app.manifest_starlink_filenames(text)],
            [40075, 40103],
        )

    def test_meme_sample_interpolates_geodetic_and_fits_a_tle(self):
        from datetime import datetime, timezone

        when = datetime(2026, 9, 28, 16, 58, 42, tzinfo=timezone.utc)
        state = app.meme_state_from_lines(_MEME_SAMPLE.splitlines(), "2026-09-28", when)
        self.assertIsNotNone(state)
        self.assertEqual(state["t"], when)
        record = app._satellite_from_inertial_state(40075, state, "v3")
        self.assertEqual(record["name"], "STARLINK-40075")
        self.assertEqual(record["id"], "STARLINK-40075")
        self.assertNotIn("norad_id", record)
        self.assertEqual(record["generation"], "v3")
        self.assertAlmostEqual(record["lat"], 10.61684, places=3)
        self.assertAlmostEqual(record["lon"], 94.40681, places=3)
        self.assertAlmostEqual(record["alt_km"], 261.488, places=2)
        self.assertEqual(len(record["tle_line1"]), 69)
        self.assertEqual(len(record["tle_line2"]), 69)
        self.assertTrue(record["tle_line1"].startswith("1 00000U"))
        self.assertTrue(record["tle_line2"].startswith("2 00000"))
        self.assertIn(" 30.4", record["tle_line2"])
        self.assertEqual(record["tle_line1"][-1], app._tle_checksum(record["tle_line1"]))
        self.assertEqual(record["tle_line2"][-1], app._tle_checksum(record["tle_line2"]))

        midpoint = datetime(2026, 9, 28, 16, 59, 12, tzinfo=timezone.utc)
        mid = app.meme_state_from_lines(_MEME_SAMPLE.splitlines(), "2026-09-28", midpoint)
        mid_rec = app._satellite_from_inertial_state(40075, mid, "v3")
        self.assertAlmostEqual(mid_rec["lat"], 11.58256, places=3)
        self.assertAlmostEqual(mid_rec["lon"], 96.07876, places=3)
        self.assertAlmostEqual(mid_rec["alt_km"], 257.511, places=2)

        # A launch well outside the roll-forward window must not inherit this file.
        self.assertIsNone(
            app.meme_state_from_lines(_MEME_SAMPLE.splitlines(), "2026-08-01", when)
        )
        self.assertIsNone(
            app.meme_state_from_lines(_MEME_OLD.splitlines(), "2026-09-28", when)
        )

    def test_rolled_forward_window_still_matches_the_launch(self):
        from datetime import datetime, timezone

        rolled = {
            "created": datetime(2026, 9, 29, 2, 11, 53, tzinfo=timezone.utc),
            "start": datetime(2026, 9, 29, 2, 8, 42, tzinfo=timezone.utc),
            "stop": datetime(2026, 10, 1, 2, 7, 42, tzinfo=timezone.utc),
        }
        now = datetime(2026, 9, 29, 4, 0, tzinfo=timezone.utc)
        self.assertFalse(app._meme_window_covers(rolled, "2026-09-28"))
        self.assertTrue(app._meme_window_covers(rolled, "2026-09-28", now))
        before_launch = {
            "start": datetime(2026, 9, 27, 0, 0, tzinfo=timezone.utc),
            "stop": datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc),
        }
        self.assertTrue(app._meme_window_covers(before_launch, "2026-09-28", now))
        too_old = {
            "start": datetime(2020, 1, 1, tzinfo=timezone.utc),
            "stop": datetime(2020, 1, 2, tzinfo=timezone.utc),
        }
        self.assertFalse(app._meme_window_covers(too_old, "2026-09-28", now))

    def test_satcat_miss_manifest_hit_returns_flight_14_set(self):
        manifest_lines = [
            "MEME_1_STARLINK-1008_2710000_Operational_1_UNCLASSIFIED.txt",
            _f14_filename(40110),
        ]
        ephemerides = {_f14_filename(40110): _MEME_OLD}
        for sid in F14_STARLINK_IDS:
            name = _f14_filename(sid)
            manifest_lines.append(name)
            ephemerides[name] = _MEME_SAMPLE
        router = DeployedSatelliteTests()
        fake_get, calls = router._route_celestrak(
            [],
            manifest="\n".join(manifest_lines) + "\n",
            ephemerides=ephemerides,
        )
        launches = {"upcoming": [], "previous": [FLIGHT_14], "last_updated": None}
        with patch.object(app, "r", None), \
             patch.object(app, "datetime", _FrozenDateTime), \
             patch.object(app, "_load_launch_payload", return_value=launches), \
             patch.object(app.requests, "get", side_effect=fake_get):
            body = app.get_satellites_deployed(launch_date="2026-09-28", internal=True)

        self.assertEqual(body["source"], "spacex-manifest-ephemeris")
        self.assertEqual(body["count"], 26)
        self.assertEqual(body["catalog_count"], 26)
        self.assertFalse(body["empty"])
        self.assertFalse(body["stale"])
        self.assertEqual(body["ttl_seconds"], app.DEPLOYED_MANIFEST_TTL)
        self.assertEqual(body["launch_date"], "2026-09-28")
        self.assertEqual(body["generation"], "v3")
        self.assertNotIn("unavailable", body["note"].lower())
        self.assertNotIn("manifest_checked", body)
        names = [sat["name"] for sat in body["satellites"]]
        self.assertEqual(names, [f"STARLINK-{sid}" for sid in F14_STARLINK_IDS])
        self.assertNotIn("STARLINK-1008", names)
        self.assertNotIn("STARLINK-40110", names)
        sat = body["satellites"][0]
        self.assertEqual(set(sat), {
            "name", "id", "lat", "lon", "alt_km", "epoch",
            "tle_line1", "tle_line2", "generation",
        })
        self.assertNotIn("norad_id", sat)
        self.assertAlmostEqual(sat["lat"], 11.58256, places=3)
        self.assertAlmostEqual(sat["lon"], 96.07876, places=3)
        self.assertGreater(sat["alt_km"], 200)
        self.assertLess(sat["alt_km"], 400)
        self.assertEqual(len(sat["tle_line1"]), 69)
        self.assertEqual(len(sat["tle_line2"]), 69)
        file_calls = [
            url for url, _params in calls
            if url.startswith(app.SPACEX_EPHEM_BASE) and not url.endswith("MANIFEST.txt")
        ]
        self.assertEqual(len(file_calls), 27)  # 26 live + one expired window

    def test_empty_satcat_cache_does_not_block_manifest(self):
        app._local_deployed["2026-09-28"] = {
            "launch_date": "2026-09-28",
            "generation": "v3",
            "mission": FLIGHT_14["mission"],
            "fetched_at": "2026-09-28T16:50:00Z",
            "ttl_seconds": 3600,
            "count": 0,
            "catalog_count": 0,
            "stale": False,
            "empty": True,
            "source": "celestrak-satcat",
            "note": "No Starlink SATCAT objects with LAUNCH_DATE 2026-09-28 yet.",
            "satellites": [],
        }
        name = _f14_filename(40075)
        router = DeployedSatelliteTests()
        fake_get, _calls = router._route_celestrak(
            [],
            manifest=name + "\n",
            ephemerides={name: _MEME_SAMPLE},
        )
        with patch.object(app, "r", None), \
             patch.object(app, "datetime", _FrozenDateTime), \
             patch.object(app.requests, "get", side_effect=fake_get):
            body = app.get_satellites_deployed(launch_date="2026-09-28", internal=True)

        self.assertEqual(body["count"], 1)
        self.assertEqual(body["source"], "spacex-manifest-ephemeris")
        self.assertFalse(body["empty"])
        self.assertEqual(body["satellites"][0]["id"], "STARLINK-40075")

    def test_both_sources_empty_stays_honest(self):
        router = DeployedSatelliteTests()
        fake_get, calls = router._route_celestrak(
            [_satcat_row("STARLINK-OLD", 100700, "2026-200A", "2026-09-20")],
            manifest="MEME_1_STARLINK-1008_2710000_Operational_1_UNCLASSIFIED.txt\n",
        )
        with patch.object(app, "r", None), \
             patch.object(app.requests, "get", side_effect=fake_get):
            body = app.get_satellites_deployed(launch_date="2026-09-28", internal=True)

        self.assertEqual(body["satellites"], [])
        self.assertEqual(body["count"], 0)
        self.assertEqual(body["catalog_count"], 0)
        self.assertTrue(body["empty"])
        self.assertFalse(body["stale"])
        self.assertEqual(body["source"], "celestrak-satcat")
        self.assertNotIn("unavailable", body["note"].lower())
        self.assertEqual(
            [url for url, _params in calls],
            [app.CELESTRAK_SATCAT_URL, app.SPACEX_MANIFEST_URL],
        )

    def test_rolled_forward_manifest_fills_flight_14(self):
        """SpaceX republishes MEMEs the day after launch. The window no longer
        contains the launch date, but the post-Falcon ids are still this flight.
        """
        from datetime import datetime, timezone

        class _RolledNow(datetime):
            @classmethod
            def now(cls, tz=None):
                stamp = cls(2026, 9, 29, 4, 0, 30, tzinfo=timezone.utc)
                if tz is not None:
                    return stamp.astimezone(tz)
                return stamp

        rolled = _MEME_SAMPLE.replace(
            "created:2026-09-28 17:01:25 UTC",
            "created:2026-09-29 02:11:53 UTC",
        ).replace(
            "ephemeris_start:2026-09-28 16:58:42 UTC ephemeris_stop:2026-09-30 16:57:42 UTC step_size:60",
            "ephemeris_start:2026-09-29 02:08:42 UTC ephemeris_stop:2026-10-01 02:07:42 UTC step_size:60",
        ).replace(
            "2026271165842.000",
            "2026272035950.000",
        ).replace(
            "2026271165942.000",
            "2026272040050.000",
        )
        name = _f14_filename(40075)
        router = DeployedSatelliteTests()
        fake_get, calls = router._route_celestrak(
            [],
            manifest=name + "\n",
            ephemerides={name: rolled},
        )
        with patch.object(app, "r", None), \
             patch.object(app, "datetime", _RolledNow), \
             patch.object(app.requests, "get", side_effect=fake_get):
            body = app.get_satellites_deployed(launch_date="2026-09-28", internal=True)

        self.assertEqual(body["source"], "spacex-manifest-ephemeris")
        self.assertEqual(body["count"], 1)
        self.assertFalse(body["empty"])
        self.assertEqual(body["satellites"][0]["id"], "STARLINK-40075")
        self.assertEqual(len(body["satellites"][0]["tle_line1"]), 69)
        self.assertTrue(any(url == app.SPACEX_MANIFEST_URL for url, _params in calls))


_SPACE_TRACK_ENV = {
    "SPACE_TRACK_IDENTITY": "operator@example.com",
    "SPACE_TRACK_PASSWORD": "unit-test-space-track-password",
}


def _space_track_session(get_responses, login_status=200, cookie="chocolatechip-session"):
    session = Mock()
    session.cookies = Mock()
    session.cookies.get.return_value = cookie
    login = Mock()
    login.status_code = login_status
    session.post.return_value = login
    session.get.side_effect = list(get_responses)
    return session


class SpaceTrackPrimaryTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()

    def _creds(self):
        return patch.dict("os.environ", _SPACE_TRACK_ENV, clear=False)

    def test_starlink_gp_uses_space_track_and_skips_celestrak(self):
        official_1 = "1 44714U 19074B   26262.50000000  .00001000  00000+0  10000-3 0  9991"
        official_2 = "2 44714  53.0500 100.0000 0001234  50.0000 310.0000 15.06400000 12345"
        self.assertEqual(len(official_1), 69)
        self.assertEqual(len(official_2), 69)
        payload = dict(STARLINK_OMM)
        payload["NORAD_CAT_ID"] = "44714"
        payload["EPOCH"] = "2026-09-19 12:00:00.000000"
        payload["TLE_LINE1"] = official_1
        payload["TLE_LINE2"] = official_2
        payload_with_stray = [dict(ISS_OMM), payload]
        session = _space_track_session([_json_response(payload_with_stray)])
        with self._creds(), \
             patch.object(app, "r", None), \
             patch.object(app.requests, "Session", return_value=session), \
             patch.object(app.requests, "get", side_effect=AssertionError("celestrak")) as get:
            body = app.get_satellites_gp(group="starlink", internal=True)

        self.assertEqual(get.call_count, 0)
        self.assertEqual(session.post.call_count, 1)
        login_url = session.post.call_args.args[0]
        login_data = session.post.call_args.kwargs["data"]
        self.assertEqual(login_url, app.SPACE_TRACK_LOGIN_URL)
        self.assertEqual(login_data["identity"], _SPACE_TRACK_ENV["SPACE_TRACK_IDENTITY"])
        self.assertEqual(login_data["password"], _SPACE_TRACK_ENV["SPACE_TRACK_PASSWORD"])
        gp_url = session.get.call_args.args[0]
        self.assertIn("/class/gp/", gp_url)
        self.assertIn("OBJECT_NAME/STARLINK~~", gp_url)
        self.assertIn("decay_date/null-val", gp_url)
        self.assertIn("epoch/%3Enow-10", gp_url)
        self.assertIn("format/json", gp_url)
        self.assertNotIn(_SPACE_TRACK_ENV["SPACE_TRACK_PASSWORD"], gp_url)
        self.assertEqual(body["source"], "space-track")
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["satellites"][0]["norad_id"], 44714)
        self.assertEqual(body["satellites"][0]["tle_line1"], official_1)
        self.assertEqual(body["satellites"][0]["tle_line2"], official_2)
        self.assertFalse(body["stale"])

    def test_non_200_and_tls_eof_fall_back_to_celestrak(self):
        tls = _space_track_session([
            requests_ssl_error(),
        ])
        celestrak = _json_response([STARLINK_OMM])
        with self._creds(), \
             patch.object(app, "r", None), \
             patch.object(app.requests, "Session", return_value=tls), \
             patch.object(app.requests, "get", return_value=celestrak) as get:
            tls_body = app.refresh_satellites_internal("starlink", force=True)
        self.assertEqual(tls_body["source"], "celestrak")
        self.assertEqual(tls_body["count"], 1)
        self.assertEqual(get.call_count, 1)

        app._reset_cache_coordination_for_tests()
        denied = Mock()
        denied.status_code = 503
        session = _space_track_session([denied])
        with self._creds(), \
             patch.object(app, "r", None), \
             patch.object(app.requests, "Session", return_value=session), \
             patch.object(app.requests, "get", return_value=celestrak):
            body = app.refresh_satellites_internal("starlink", force=True)
        self.assertEqual(body["source"], "celestrak")
        self.assertEqual(body["satellites"][0]["norad_id"], 44714)

    def test_401_relogin_once_then_uses_the_catalog(self):
        denied = Mock()
        denied.status_code = 401
        session = _space_track_session([denied, _json_response([STARLINK_OMM])])
        with self._creds(), \
             patch.object(app, "r", None), \
             patch.object(app.requests, "Session", return_value=session), \
             patch.object(app.requests, "get", side_effect=AssertionError("celestrak")):
            body = app.refresh_satellites_internal("starlink", force=True)
        self.assertEqual(session.post.call_count, 2)
        self.assertEqual(session.get.call_count, 2)
        self.assertEqual(body["source"], "space-track")
        self.assertEqual(body["count"], 1)

    def test_missing_cookie_does_not_accept_the_login(self):
        session = _space_track_session([_json_response([STARLINK_OMM])], cookie=None)
        celestrak = _json_response([ISS_OMM])
        with self._creds(), \
             patch.object(app, "r", None), \
             patch.object(app.requests, "Session", return_value=session), \
             patch.object(app.requests, "get", return_value=celestrak):
            body = app.refresh_satellites_internal("stations", force=True)
        # stations are not the Starlink GP query
        self.assertEqual(body["source"], "celestrak")
        self.assertEqual(session.post.call_count, 0)

        app._reset_cache_coordination_for_tests()
        with self._creds(), \
             patch.object(app, "r", None), \
             patch.object(app.requests, "Session", return_value=session), \
             patch.object(app.requests, "get", return_value=celestrak):
            body = app.refresh_satellites_internal("starlink", force=True)
        self.assertEqual(body["source"], "celestrak")
        self.assertEqual(body["satellites"][0]["norad_id"], 25544)
        self.assertGreaterEqual(session.post.call_count, 1)

    def test_hourly_gp_limit_uses_celestrak_for_the_next_force(self):
        session = _space_track_session([
            _json_response([STARLINK_OMM]),
            _json_response([ISS_OMM]),
        ])
        celestrak = _json_response([ISS_OMM])
        with self._creds(), \
             patch.object(app, "r", None), \
             patch.object(app.requests, "Session", return_value=session), \
             patch.object(app.requests, "get", return_value=celestrak) as get:
            first = app.refresh_satellites_internal("starlink", force=True)
            second = app.refresh_satellites_internal("starlink", force=True)
        self.assertEqual(first["source"], "space-track")
        self.assertEqual(second["source"], "celestrak")
        self.assertEqual(second["satellites"][0]["norad_id"], 25544)
        self.assertEqual(session.get.call_count, 1)
        self.assertEqual(get.call_count, 1)

    def test_empty_or_failed_catalog_does_not_invent_positions(self):
        from fastapi.testclient import TestClient

        session = _space_track_session([_json_response([])])
        with self._creds(), \
             patch.object(app, "r", None), \
             patch.object(app.requests, "Session", return_value=session), \
             patch.object(app.requests, "get", side_effect=RuntimeError("celestrak down")), \
             patch.object(app, "_background_enabled", False):
            client = TestClient(app.app)
            response = client.get("/satellites/gp", params={"group": "starlink"})
        self.assertEqual(response.status_code, 502)
        self.assertIn("unavailable", response.json()["detail"].lower())
        self.assertNotIn(_SPACE_TRACK_ENV["SPACE_TRACK_PASSWORD"], response.text)
        cached = app._local_satellites.get("starlink")
        self.assertTrue(cached is None or not (cached.get("satellites") or []))

    def test_both_sources_fail_serves_last_good_without_logging_the_password(self):
        import io
        from contextlib import redirect_stdout

        stored = app._satellite_payload("starlink", [app.slim_gp_record(STARLINK_OMM)])
        stored["fetched_at"] = "2020-01-01T00:00:00Z"
        app._local_satellites["starlink"] = stored
        boom = RuntimeError(
            "tls eof while sending " + _SPACE_TRACK_ENV["SPACE_TRACK_PASSWORD"]
        )
        session = _space_track_session([boom])
        stdout = io.StringIO()
        with self._creds(), \
             patch.object(app, "r", None), \
             patch.object(app.requests, "Session", return_value=session), \
             patch.object(app.requests, "get", side_effect=RuntimeError("celestrak down")), \
             redirect_stdout(stdout):
            body = app.refresh_satellites_internal("starlink", force=True)
        logged = stdout.getvalue()
        self.assertNotIn(_SPACE_TRACK_ENV["SPACE_TRACK_PASSWORD"], logged)
        self.assertIn("[redacted]", logged)
        self.assertIn("satellite_catalog_stale", logged)
        self.assertTrue(body["stale"])
        self.assertEqual(body["satellites"][0]["norad_id"], 44714)
        self.assertEqual(len(body["satellites"]), 1)

    def test_request_rate_limit_stops_before_thirty_per_minute(self):
        now = time.monotonic()
        app._space_track._request_times = [now] * app.SPACE_TRACK_MAX_PER_MINUTE
        with self.assertRaises(app._SpaceTrackError):
            app._space_track._reserve_request()

    def test_unconfigured_starlink_still_uses_celestrak(self):
        fake = _json_response([STARLINK_OMM])
        with patch.dict("os.environ", {"SPACE_TRACK_IDENTITY": "", "SPACE_TRACK_PASSWORD": ""}, clear=False), \
             patch.object(app, "r", None), \
             patch.object(app.requests, "Session", side_effect=AssertionError("no session")), \
             patch.object(app.requests, "get", return_value=fake) as get:
            body = app.refresh_satellites_internal("starlink", force=True)
        self.assertEqual(get.call_count, 1)
        self.assertEqual(body["source"], "celestrak")
        self.assertEqual(body["count"], 1)

    def test_deployed_last_good_returns_immediately_and_alerts_when_old(self):
        import io
        from contextlib import redirect_stdout

        record = app.slim_gp_record(_v3_omm())
        record["generation"] = "v3"
        app._local_deployed["2026-09-28"] = {
            "launch_date": "2026-09-28",
            "generation": "v3",
            "mission": FLIGHT_14["mission"],
            "fetched_at": "2020-01-01T00:00:00Z",
            "ttl_seconds": app.SATELLITES_CACHE_TTL,
            "count": 1,
            "catalog_count": 1,
            "stale": False,
            "empty": False,
            "source": "celestrak-satcat",
            "note": "Starlink SATCAT objects with LAUNCH_DATE 2026-09-28, joined to CelesTrak GP TLEs.",
            "satellites": [record],
        }
        snapshot = dict(app._local_deployed["2026-09-28"])
        started_refresh = threading.Event()
        finished_refresh = threading.Event()

        def slow_refresh(launch_date=None, force=False):
            started_refresh.set()
            time.sleep(0.3)
            finished_refresh.set()
            return snapshot

        stdout = io.StringIO()
        started = time.perf_counter()
        with patch.object(app, "r", None), \
             patch.object(app, "_background_enabled", True), \
             patch.object(app, "refresh_deployed_satellites_internal", side_effect=slow_refresh), \
             patch.object(app.requests, "get", side_effect=AssertionError("upstream")), \
             redirect_stdout(stdout):
            body = app.get_satellites_deployed(launch_date="2026-09-28", internal=True)
            elapsed = time.perf_counter() - started
            self.assertTrue(started_refresh.wait(1.0))
            self.assertTrue(finished_refresh.wait(1.0))
        self.assertLess(elapsed, 1.0)
        self.assertTrue(body["stale"])
        self.assertEqual(body["satellites"][0]["norad_id"], 100753)
        self.assertNotIn("manifest_checked", body)
        logged = stdout.getvalue()
        self.assertIn("satellite_catalog_stale", logged)
        self.assertIn("2026-09-28", logged)
        self.assertNotIn("unit-test-space-track-password", logged)

    def test_deployed_satcat_falls_back_to_space_track_without_inventing_rows(self):
        import requests

        session = _space_track_session([_json_response([
            {
                "SATNAME": "STARLINK-38381",
                "INTLDES": "2026-219A",
                "NORAD_CAT_ID": "100753",
                "OBJECT_TYPE": "PAYLOAD",
                "LAUNCH": "2026-09-28",
                "DECAY": None,
            },
            {
                "SATNAME": "STARLINK DEB",
                "INTLDES": "2026-219E",
                "NORAD_CAT_ID": 100703,
                "OBJECT_TYPE": "DEBRIS",
                "LAUNCH": "2026-09-28",
                "DECAY": None,
            },
            {
                "SATNAME": "NOT-STARLINK",
                "INTLDES": "2026-219F",
                "NORAD_CAT_ID": 100799,
                "OBJECT_TYPE": "PAYLOAD",
                "LAUNCH": "2026-09-28",
                "DECAY": None,
            },
        ])])

        def fake_get(url, params=None, **kwargs):
            if url == app.CELESTRAK_SATCAT_URL:
                raise requests.exceptions.SSLError("TLS EOF")
            if url == app.CELESTRAK_GP_URL:
                return _json_response([_v3_omm()])
            raise AssertionError(url)

        with self._creds(), \
             patch.object(app, "r", None), \
             patch.object(app.requests, "Session", return_value=session), \
             patch.object(app.requests, "get", side_effect=fake_get):
            body = app.get_satellites_deployed(launch_date="2026-09-28", internal=True)
            again = app.get_satellites_deployed(
                launch_date="2026-09-28", force=True, internal=True
            )

        self.assertEqual(body["count"], 1)
        self.assertEqual(body["catalog_count"], 1)
        self.assertEqual(body["satellites"][0]["norad_id"], 100753)
        self.assertEqual(body["source"], "celestrak-satcat")
        self.assertNotIn("unavailable", body["note"].lower())
        self.assertEqual(again["satellites"][0]["norad_id"], 100753)
        self.assertEqual(session.get.call_count, 1)
        satcat_url = session.get.call_args.args[0]
        self.assertIn("/class/satcat/", satcat_url)
        self.assertIn("SATNAME/STARLINK~~", satcat_url)
        self.assertNotIn(_SPACE_TRACK_ENV["SPACE_TRACK_PASSWORD"], satcat_url)


def requests_ssl_error():
    import requests
    return requests.exceptions.SSLError("EOF occurred in violation of protocol (_ssl.c: TLS EOF)")


if __name__ == "__main__":
    unittest.main()
