"""Work that is the same for every device is done once, and tokens are not lost.

- Scheduled jobs and the push sender run in one process per dyno.
- Countdown alert text ships on the launch list instead of one request per alert.
- Nearby weather is shared by devices in the same area.
- Live Activity tokens are stored one record per activity, optionally in SQL.
"""
import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import app
import live_activity_push as push


def _launch(launch_id="u1", **overrides):
    data = {
        "id": launch_id,
        "name": "Falcon 9 Block 5 | Starlink Group 10-20",
        "mission": "Falcon 9 Block 5 | Starlink Group 10-20",
        "net": "2026-12-01T00:00:00Z",
        "status": "Go for Launch",
        "rocket": "Falcon 9 Block 5",
        "orbit": "Low Earth Orbit",
        "pad": "Space Launch Complex 40",
        "probability": None,
    }
    data.update(overrides)
    return data


class BackgroundLeaderTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()
        app._release_background_leadership_for_tests()
        self._dir = tempfile.TemporaryDirectory()
        self._env = patch.dict(
            os.environ, {"BACKGROUND_LEADER_LOCK": os.path.join(self._dir.name, "leader.lock")}
        )
        self._env.start()

    def tearDown(self):
        app._release_background_leadership_for_tests()
        self._env.stop()
        self._dir.cleanup()
        app._reset_cache_coordination_for_tests()

    def test_only_one_process_on_the_dyno_is_the_leader(self):
        self.assertTrue(app.is_background_leader())
        self.assertTrue(app.is_background_leader())

        # A second worker process: it has no handle of its own and the file is locked.
        first, app._background_leader_handle = app._background_leader_handle, None
        try:
            self.assertFalse(app.is_background_leader())
            self.assertFalse(app.is_background_leader())
            # The leader exits; the OS drops its lock and the next check takes over.
            first.close()
            self.assertTrue(app.is_background_leader())
        finally:
            if not first.closed:
                first.close()

    def test_a_follower_reads_the_shared_lists_into_memory_without_upstream_calls(self):
        launches = {"upcoming": [_launch()], "previous": [], "last_updated": "2026-11-18T02:46:35Z"}

        def stored(key, strict=False):
            if key == app.LAUNCHES_CACHE_KEY:
                return launches
            if key == app.CACHE_KEY:
                return ["11/17 0246: Falcon 9 flew."]
            if key == app.CACHE_TIME_KEY:
                return "2026-11-18T02:43:50Z"
            return None

        with patch.object(app, "get_cached_data", side_effect=stored), \
             patch.object(app, "set_cached_data", return_value=True), \
             patch.object(app, "save_cache_to_file"), \
             patch.object(app, "fetch_launches", side_effect=AssertionError("called Launch Library")), \
             patch.object(app, "generate_narratives", side_effect=AssertionError("called Grok")):
            app.warm_shared_lists()

        self.assertEqual(app._launches_mem["data"]["upcoming"][0]["id"], "u1")
        self.assertEqual(app._narratives_mem["payload"]["descriptions"], ["11/17 0246: Falcon 9 flew."])
        # Which is what the list route serves if Redis then fails.
        with patch.object(app, "get_cached_data", side_effect=ConnectionError("down")):
            app._launches_mem["at"] = 0.0
            response = TestClient(app.app).get("/launches_slim")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["upcoming"][0]["id"], "u1")

    def test_the_push_loop_only_sends_from_the_leader(self):
        import inspect

        source = inspect.getsource(push.start_worker)
        self.assertIn("is_background_leader()", source)


class ListNotifyCopyTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()
        self._redis = app.r
        app.r = None

    def tearDown(self):
        app.r = self._redis
        app._reset_cache_coordination_for_tests()

    def test_every_upcoming_launch_gets_text_for_each_countdown_alert(self):
        data = {"upcoming": [_launch("u1"), _launch("u2", probability=70)], "previous": [_launch("p1")]}
        app.attach_notify_copy(data)

        for launch in data["upcoming"]:
            self.assertEqual(sorted(launch["notify_copy"]), ["t1h", "t24h", "t5m"])
            for event, copy in launch["notify_copy"].items():
                self.assertTrue(copy["title"] and copy["body"], event)
                self.assertLessEqual(len(copy["title"]), app.NOTIFY_TITLE_MAX)
                self.assertLessEqual(len(copy["body"]), app.NOTIFY_BODY_MAX)
        self.assertIn("T-5m", data["upcoming"][0]["notify_copy"]["t5m"]["title"])
        self.assertIn("70%", json.dumps(data["upcoming"][1]["notify_copy"]["t1h"]))
        self.assertNotIn("notify_copy", data["previous"][0])

    def test_cached_text_is_used_and_shares_the_key_a_client_request_would_use(self):
        launch = _launch("u1")
        client_request = app.NotifyCopyRequest(
            launch_id="u1",
            event="t1h",
            mission=launch["mission"],
            net=launch["net"],
            status=launch["status"],
            pad=launch["pad"],
            rocket=launch["rocket"],
            orbit=launch["orbit"],
        )
        app._set_notify_copy_cached(
            app._notify_copy_cache_key_for(client_request),
            {"title": "T-1h: Starlink 10-20", "body": "Last coffee before liftoff."},
        )
        data = {"upcoming": [launch], "previous": []}
        app.attach_notify_copy(data)
        self.assertEqual(
            data["upcoming"][0]["notify_copy"]["t1h"],
            {"title": "T-1h: Starlink 10-20", "body": "Last coffee before liftoff."},
        )
        self.assertIn("T-24h", data["upcoming"][0]["notify_copy"]["t24h"]["title"])

    def test_a_probability_outside_0_to_100_is_treated_as_unknown(self):
        for bad in (-1, 250, "80", True):
            data = {"upcoming": [_launch("u1", probability=bad)], "previous": []}
            app.attach_notify_copy(data)
            self.assertEqual(len(data["upcoming"][0]["notify_copy"]), 3, bad)

    def test_a_refresh_stores_the_text_with_the_list_and_the_slim_route_serves_it(self):
        fetched = {"upcoming": [_launch("u1")], "previous": [_launch("p1")]}
        stored = {}
        with patch.object(app, "get_cached_data", return_value=None), \
             patch.object(app, "set_cached_data", side_effect=lambda key, data, ttl=None: stored.__setitem__(key, data) or True), \
             patch.object(app, "save_cache_to_file"), \
             patch.object(app, "fetch_launches", return_value=fetched), \
             patch.object(app, "get_launch_trajectory_data", return_value=None):
            app._refresh_launches_uncached()
            response = TestClient(app.app).get("/launches_slim")

        self.assertIn("t5m", stored[app.LAUNCHES_CACHE_KEY]["upcoming"][0]["notify_copy"])
        served = response.json()["upcoming"][0]["notify_copy"]
        self.assertEqual(sorted(served), ["t1h", "t24h", "t5m"])
        self.assertEqual(sorted(served["t24h"]), ["body", "title"])

    def test_no_grok_call_is_started_while_background_work_is_off(self):
        with patch.object(app, "_begin_notify_copy_flight") as flight:
            app.attach_notify_copy({"upcoming": [_launch("u1")], "previous": []})
        flight.assert_not_called()


class UserWeatherCellTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()
        self._redis = app.r
        app.r = None
        self.client = TestClient(app.app)
        self.metar_calls = []

    def tearDown(self):
        app.r = self._redis
        app._reset_cache_coordination_for_tests()

    def _upstream(self):
        def metar(station_id=None, lat=None, lon=None, location=None):
            self.metar_calls.append((lat, lon))
            return {"temperature_c": 21.0, "wind_speed_kts": 5, "raw": "METAR KAUS"}

        return patch.object(app, "fetch_weather", side_effect=metar), \
            patch.object(app, "fetch_forecast", return_value={"daily": {}})

    def test_devices_in_the_same_area_share_one_upstream_fetch(self):
        metar, forecast = self._upstream()
        with metar, forecast as forecast_mock:
            first = self.client.get("/user_weather", params={"lat": 30.2672, "lon": -97.7431})
            second = self.client.get("/user_weather", params={"lat": 30.2901, "lon": -97.7102})
            other = self.client.get("/user_weather", params={"lat": 29.7604, "lon": -95.3698})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.json(), first.json())
        self.assertEqual(first.json()["temperature_c"], 21.0)
        self.assertEqual(other.status_code, 200)
        self.assertEqual(len(self.metar_calls), 2)
        self.assertEqual(forecast_mock.call_count, 2)

    def test_a_station_request_is_cached_apart_from_the_plain_one(self):
        self.assertNotEqual(
            app._user_weather_key(30.27, -97.74, None), app._user_weather_key(30.27, -97.74, "kaus")
        )
        self.assertEqual(
            app._user_weather_key(30.27, -97.74, "kaus"), app._user_weather_key(30.31, -97.69, "KAUS")
        )
        self.assertEqual(app._user_weather_key(-0.01, 0.01, None), app._user_weather_key(0.01, -0.01, None))

    def test_an_upstream_failure_serves_the_last_answer_for_the_cell(self):
        metar, forecast = self._upstream()
        with metar, forecast:
            good = self.client.get("/user_weather", params={"lat": 30.27, "lon": -97.74})
        key = app._user_weather_key(30.27, -97.74, None)
        data, _expires = app._user_weather_mem[key]
        app._user_weather_mem[key] = (data, time.time() - 1)

        with patch.object(app, "fetch_weather", side_effect=RuntimeError("upstream down")):
            stale = self.client.get("/user_weather", params={"lat": 30.27, "lon": -97.74})
            nothing = self.client.get("/user_weather", params={"lat": 10.0, "lon": 10.0})

        self.assertEqual(stale.status_code, 200)
        self.assertEqual(stale.json(), good.json())
        self.assertEqual(nothing.status_code, 503)

    def test_the_memory_copy_is_bounded(self):
        for index in range(app.USER_WEATHER_MEM_MAX + 25):
            app._remember_user_weather(f"cell-{index}", {"temperature_c": index})
        self.assertLessEqual(len(app._user_weather_mem), app.USER_WEATHER_MEM_MAX)


class _FakeRedis:
    """The few commands the token store uses, with `decode_responses=True` results."""

    def __init__(self):
        self.values = {}
        self.hashes = {}
        self.hgetall_calls = 0

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, **_kwargs):
        self.values[key] = value
        return True

    def hgetall(self, key):
        self.hgetall_calls += 1
        return dict(self.hashes.get(key, {}))

    def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    def hset(self, key, field, value):
        self.hashes.setdefault(key, {})[field] = value
        return 1

    def hdel(self, key, field):
        return 1 if self.hashes.get(key, {}).pop(field, None) is not None else 0


def _register(activity_id, launch_id="u1"):
    return push.upsert_token(
        token="ab" * 32, launch_id=launch_id, activity_id=activity_id, net="2026-12-01T00:00:00Z", content_state=None
    )


class TokenRedisStoreTests(unittest.TestCase):
    def setUp(self):
        push.reset_for_tests()
        self._redis = app.r
        self.fake = _FakeRedis()
        app.r = self.fake
        self._legacy = patch.object(app, "get_cached_data", return_value=None)
        self._legacy.start()

    def tearDown(self):
        self._legacy.stop()
        app.r = self._redis
        push.reset_for_tests()

    def test_each_activity_is_its_own_hash_field(self):
        _register("a1")
        _register("a2")
        self.assertEqual(sorted(self.fake.hashes[push.HASH_KEY]), ["a1", "a2"])
        self.assertTrue(push.delete_token("a1"))
        self.assertEqual(sorted(self.fake.hashes[push.HASH_KEY]), ["a2"])
        self.assertFalse(push.delete_token("a1"))

    def test_registering_does_not_read_every_other_record(self):
        for index in range(5):
            _register(f"a{index}")
        self.assertEqual(self.fake.hgetall_calls, 0)

    def test_two_processes_registering_do_not_drop_each_others_tokens(self):
        _register("from-worker-1")
        # A second worker: nothing in its memory, same Redis.
        push._memory.clear()
        _register("from-worker-2")
        self.assertEqual(sorted(push._load()), ["from-worker-1", "from-worker-2"])

    def test_a_re_registration_keeps_what_was_already_pushed(self):
        _register("a1")
        stored = json.loads(self.fake.hashes[push.HASH_KEY]["a1"])
        stored["pushed_net_unix"] = 123.0
        self.fake.hashes[push.HASH_KEY]["a1"] = json.dumps(stored)
        again = _register("a1")
        self.assertEqual(again["pushed_net_unix"], 123.0)
        self.assertEqual(again["registered_at"], stored["registered_at"])

    def test_more_than_forty_activities_are_kept(self):
        for index in range(60):
            _register(f"a{index}")
        self.assertEqual(len(push._load()), 60)
        self.assertGreaterEqual(push.MAX_RECORDS, 1000)

    def test_the_old_single_document_is_copied_in_once(self):
        self._legacy.stop()
        old = {"old-1": {"token": "cd" * 32, "launch_id": "u1", "activity_id": "old-1", "registered_at": 1.0}}
        with patch.object(app, "get_cached_data", return_value=old):
            self.assertEqual(sorted(push._load()), ["old-1"])
            self.assertTrue(push.delete_token("old-1"))
            # A restart: the activity has ended and must not come back from the old copy.
            push.reset_for_tests()
            self.assertEqual(push._load(), {})
        self._legacy.start()

    def test_a_read_failure_serves_this_process_s_last_copy(self):
        _register("a1")
        push._load()
        with patch.object(self.fake, "hgetall", side_effect=ConnectionError("down")):
            self.assertEqual(sorted(push._load()), ["a1"])


class TokenSqlStoreTests(unittest.TestCase):
    def setUp(self):
        push.reset_for_tests()
        self._redis = app.r
        self.fake = _FakeRedis()
        app.r = self.fake
        self._legacy = patch.object(app, "get_cached_data", return_value=None)
        self._legacy.start()
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._dir.name, "tokens.db")
        self.connections = 0

        def connect():
            self.connections += 1
            return sqlite3.connect(self.path, check_same_thread=False)

        self.store = push.SqlTokenStore(connect, placeholder="?")
        push._sql_store = self.store
        push._sql_store_checked = True

    def tearDown(self):
        self._legacy.stop()
        app.r = self._redis
        push.reset_for_tests()
        self._dir.cleanup()

    def test_tokens_are_rows_and_survive_losing_redis_and_the_process(self):
        _register("a1")
        _register("a2")
        self.assertTrue(push.delete_token("a2"))

        # Redis is wiped and the process restarts with nothing in memory.
        self.fake.hashes.clear()
        self.fake.values.clear()
        push._memory.clear()
        fresh = push.SqlTokenStore(lambda: sqlite3.connect(self.path), placeholder="?")
        self.assertEqual(sorted(fresh.all()), ["a1"])
        self.assertEqual(fresh.get("a1")["launch_id"], "u1")
        self.assertIsNone(fresh.get("a2"))
        self.assertEqual(sorted(push._load()), ["a1"])
        self.assertEqual(self.fake.hashes.get(push.HASH_KEY, {}), {})

    def test_an_update_replaces_the_row(self):
        _register("a1", launch_id="u1")
        _register("a1", launch_id="u2")
        self.assertEqual(len(self.store.all()), 1)
        self.assertEqual(self.store.get("a1")["launch_id"], "u2")

    def test_tokens_already_in_redis_move_into_the_database_once(self):
        self.fake.hashes[push.HASH_KEY] = {
            "from-redis": json.dumps({"token": "ef" * 32, "launch_id": "u1", "activity_id": "from-redis"})
        }
        self.assertEqual(sorted(push._load()), ["from-redis"])
        self.assertEqual(sorted(self.store.all()), ["from-redis"])
        self.assertTrue(push.delete_token("from-redis"))
        push.reset_for_tests()
        push._sql_store, push._sql_store_checked = self.store, True
        self.assertEqual(push._load(), {})

    def test_a_dropped_connection_is_reopened(self):
        _register("a1")
        self.store._connection.close()
        _register("a2")
        self.assertEqual(sorted(self.store.all()), ["a1", "a2"])
        self.assertEqual(self.connections, 2)

    def test_a_database_outage_does_not_fail_a_registration(self):
        _register("a1")
        with patch.object(self.store, "_connect", side_effect=ConnectionError("db down")):
            self.store._connection.close()
            self.store._connection = None
            record = _register("a2")
            self.assertEqual(record["activity_id"], "a2")
            self.assertIn("a1", push._load())

    def test_the_tick_reads_the_store_once_per_pass(self):
        for index in range(3):
            _register(f"a{index}")
        reads = []
        original = self.store.all

        def counted():
            reads.append(1)
            return original()

        with patch.object(self.store, "all", side_effect=counted), \
             patch.object(push, "apns_configured", return_value=True):
            push.tick(lookup=lambda launch_id: (None, None), sender=lambda *a, **k: {"sent": False})
        self.assertLessEqual(len(reads), 2)

    def test_no_database_url_means_no_sql_store(self):
        push.reset_for_tests()
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(push._configured_sql_store())
            self.assertIsInstance(push._store(), push._RedisHashStore)


if __name__ == "__main__":
    unittest.main()
