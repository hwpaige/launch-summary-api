"""Trajectory cache: per-entry Redis keys with a TTL, no ever-growing blob.

The old single ``trajectory_cache_v2`` blob never expired and filled the
25 MB Heroku Redis (noeviction), so its writes failed with
``used memory > 'maxmemory'``. These tests pin the replacement.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app  # noqa: E402


class FakeRedis:
    def __init__(self, used=1_000_000, maxmemory=25 * 1024 * 1024):
        self.store = {}
        self.ttls = {}
        self.used = used
        self.maxmemory = maxmemory
        self.writes = []

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return False
        self.store[key] = value
        self.ttls[key] = ex if ex else -1
        self.writes.append(key)
        return True

    def setex(self, key, ttl, value):
        self.store[key] = value
        self.ttls[key] = ttl
        self.writes.append(key)
        return True

    def delete(self, *keys):
        n = 0
        for key in keys:
            if key in self.store:
                n += 1
            self.store.pop(key, None)
            self.ttls.pop(key, None)
        return n

    def exists(self, key):
        return int(key in self.store)

    def ttl(self, key):
        return self.ttls.get(key, -2)

    def info(self, section=None):
        return {"used_memory": self.used, "maxmemory": self.maxmemory}

    def incr(self, key):
        self.store[key] = str(int(self.store.get(key) or 0) + 1)
        return int(self.store[key])

    def expire(self, key, ttl):
        self.ttls[key] = ttl
        return True


LAUNCH = {
    "id": "traj-cache-test",
    "mission": "Falcon 9 Block 5 | Starlink Group 15-15",
    "pad": "Space Launch Complex 4E",
    "pad_latitude": 34.632,
    "pad_longitude": -120.611,
    "pad_location": "Vandenberg SFB, CA, USA",
    "orbit": "SSO",
    "landing_type": "ASDS",
    "landing_location": "Of Course I Still Love You",
}


class TrajectoryCacheTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()
        self.tmp = tempfile.TemporaryDirectory()
        self.cache_file = os.path.join(self.tmp.name, "trajectory_cache.json")
        self.patches = [patch.object(app, "TRAJECTORY_CACHE_FILE", self.cache_file)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()
        app._reset_cache_coordination_for_tests()

    def _traj_keys(self, fake):
        return [k for k in fake.store if k.startswith(app.TRAJECTORY_REDIS_PREFIX)]

    def test_entry_has_ttl_and_second_worker_serves_identical_payload(self):
        fake = FakeRedis()
        with patch.object(app, "r", fake):
            first = app.get_launch_trajectory_data(dict(LAUNCH))
            keys = self._traj_keys(fake)
            self.assertEqual(len(keys), 1)
            self.assertEqual(fake.ttl(keys[0]), app.TRAJECTORY_CACHE_TTL)
            self.assertNotIn(app.TRAJECTORY_LEGACY_REDIS_KEY, fake.store)

            # A different worker: empty process memory, no disk file.
            os.remove(self.cache_file)
            app._TRAJECTORY_DATA_CACHE = {}
            writes_before = len(fake.writes)
            second = app.get_launch_trajectory_data(dict(LAUNCH))
        self.assertEqual(
            json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True)
        )
        self.assertEqual(len(fake.writes), writes_before)  # served from Redis, not recomputed

    def test_cache_hit_matches_fresh_computation_without_redis(self):
        with patch.object(app, "r", None):
            miss = app.get_launch_trajectory_data(dict(LAUNCH))
            hit = app.get_launch_trajectory_data(dict(LAUNCH))
        self.assertEqual(json.dumps(miss, sort_keys=True), json.dumps(hit, sort_keys=True))

    def test_legacy_blob_is_migrated_then_deleted(self):
        fake = FakeRedis()
        current_key = f"{app.TRAJECTORY_CACHE_VERSION}:site:SSO:97.6:ASDS:x:none:574.0"
        old_key = "v200-along-track:site:SSO:97.6:ASDS:x:none:574.0"
        entry = {"trajectory": [{"lat": 1.0, "lon": 2.0}], "launch_site": {"name": "x"}}
        legacy = {old_key: entry, current_key: entry}
        fake.store[app.TRAJECTORY_LEGACY_REDIS_KEY] = app._encode_for_cache(legacy)
        fake.ttls[app.TRAJECTORY_LEGACY_REDIS_KEY] = -1
        with patch.object(app, "r", fake):
            loaded = app.load_cache_from_file(self.cache_file)["data"]
            self.assertNotIn(app.TRAJECTORY_LEGACY_REDIS_KEY, fake.store)
            self.assertEqual(list(loaded), [current_key])
            self.assertEqual(app._load_trajectory_entry(current_key), entry)
            self.assertIsNone(app._load_trajectory_entry(old_key))
            for key in self._traj_keys(fake):
                self.assertEqual(fake.ttl(key), app.TRAJECTORY_CACHE_TTL)
            # Runs once per process.
            fake.store[app.TRAJECTORY_LEGACY_REDIS_KEY] = app._encode_for_cache(legacy)
            app.load_cache_from_file(self.cache_file)
            self.assertIn(app.TRAJECTORY_LEGACY_REDIS_KEY, fake.store)

    def test_trajectory_write_skipped_when_reserve_would_be_used(self):
        fake = FakeRedis(used=20 * 1024 * 1024)
        with patch.object(app, "r", fake):
            traj = app.get_launch_trajectory_data(dict(LAUNCH))
        self.assertTrue(traj["trajectory"])
        self.assertEqual(self._traj_keys(fake), [])

    def test_starlink_catalog_write_ignores_trajectory_reserve(self):
        fake = FakeRedis(used=20 * 1024 * 1024)
        payload = {
            "group": "starlink",
            "fetched_at": "2026-10-20T16:44:45Z",
            "count": 1,
            "source": "space-track",
            "satellites": [{"name": "S", "norad_id": 1, "tle_line1": "1", "tle_line2": "2"}],
        }
        with patch.object(app, "r", fake), patch.object(app, "_background_enabled", False):
            app._write_satellite_cache("starlink", payload)
        self.assertIn(app._satellite_cache_key("starlink"), fake.store)
        self.assertIn(app._satellite_stamp_key("starlink"), fake.store)

    def test_prune_drops_old_versions_and_caps_size(self):
        data = {f"old-version:{i}": {"a": i} for i in range(5)}
        for i in range(app.TRAJECTORY_LOCAL_MAX + 10):
            data[f"{app.TRAJECTORY_CACHE_VERSION}:{i}"] = {"a": i}
        pruned = app._prune_trajectory_cache(data)
        self.assertEqual(len(pruned), app.TRAJECTORY_LOCAL_MAX)
        self.assertTrue(all(k.startswith(app.TRAJECTORY_CACHE_VERSION) for k in pruned))
        self.assertIn(f"{app.TRAJECTORY_CACHE_VERSION}:{app.TRAJECTORY_LOCAL_MAX + 9}", pruned)
        with patch.object(app, "r", None):
            app.save_cache_to_file(self.cache_file, data)
        with open(self.cache_file) as f:
            self.assertEqual(len(json.load(f)), app.TRAJECTORY_LOCAL_MAX)

    def test_set_cached_data_format_unchanged(self):
        fake = FakeRedis()
        with patch.object(app, "r", fake):
            app.set_cached_data("k", {"a": [1, 2]}, ttl=5)
            self.assertEqual(app.get_cached_data("k"), {"a": [1, 2]})
        import base64, zlib
        self.assertEqual(
            json.loads(zlib.decompress(base64.b64decode(fake.store["k"]))), {"a": [1, 2]}
        )


if __name__ == "__main__":
    unittest.main()
