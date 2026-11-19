"""Redis outages and connection limits must not empty the app's launch list.

Regressions for the 2026-11-18 outage: a burst of concurrent requests opened more
Redis connections than the add-on allows, every read then failed, and the list
routes answered 200 with empty lists, which clients showed as "no upcoming launches".
"""
import socketserver
import threading
import time
import unittest
from unittest.mock import patch

import redis
from fastapi.testclient import TestClient

import app


def _launch(launch_id, name):
    return {
        "id": launch_id,
        "name": name,
        "mission": name,
        "net": "2026-12-01T00:00:00Z",
        "status": "Go",
        "rocket": "Falcon 9",
        "orbit": "LEO",
        "pad": "SLC-40",
    }


def _launches():
    return {
        "upcoming": [_launch("u1", "Next")],
        "previous": [_launch("p1", "Past")],
        "last_updated": "2026-11-18T02:46:35Z",
    }


class _CountingRedisServer(socketserver.ThreadingTCPServer):
    """Answers every command, slowly, and records how many clients are connected at once."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self):
        self.lock = threading.Lock()
        self.connected = 0
        self.peak = 0
        super().__init__(("127.0.0.1", 0), _CountingRedisHandler)


class _CountingRedisHandler(socketserver.StreamRequestHandler):
    def handle(self):
        server = self.server
        with server.lock:
            server.connected += 1
            server.peak = max(server.peak, server.connected)
        try:
            while True:
                header = self.rfile.readline()
                if not header:
                    return
                parts = []
                for _ in range(int(header[1:].strip())):
                    size = int(self.rfile.readline()[1:].strip())
                    parts.append(self.rfile.read(size + 2)[:-2])
                command = parts[0].upper()
                if command == b"PING":
                    self.wfile.write(b"+PONG\r\n")
                elif command == b"HELLO":
                    self.wfile.write(b"%1\r\n$5\r\nproto\r\n:3\r\n")
                elif command == b"GET":
                    time.sleep(0.01)
                    self.wfile.write(b"_\r\n")
                else:
                    self.wfile.write(b"+OK\r\n")
        except (ConnectionError, ValueError, IndexError):
            return
        finally:
            with server.lock:
                server.connected -= 1


class RedisPoolTests(unittest.TestCase):
    def test_client_is_built_on_a_capped_blocking_pool(self):
        client = app._build_redis_client("rediss://:secret@example.com:6380")
        pool = client.connection_pool
        self.assertIsInstance(pool, redis.BlockingConnectionPool)
        self.assertEqual(pool.max_connections, app.REDIS_MAX_CONNECTIONS)
        self.assertEqual(pool.timeout, app.REDIS_POOL_WAIT_SEC)
        self.assertEqual(pool.connection_kwargs["socket_connect_timeout"], app.REDIS_CONNECT_TIMEOUT_SEC)
        self.assertEqual(pool.connection_kwargs["socket_timeout"], app.REDIS_SOCKET_TIMEOUT_SEC)
        self.assertEqual(pool.connection_kwargs["health_check_interval"], app.REDIS_HEALTH_CHECK_SEC)
        self.assertIsNone(pool.connection_kwargs["ssl_cert_reqs"])
        # Comfortably under the 20-connection add-on, with room for a second dyno.
        self.assertLessEqual(app.REDIS_MAX_CONNECTIONS, 9)

    def test_a_burst_of_threads_never_holds_more_connections_than_the_cap(self):
        server = _CountingRedisServer()
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            port = server.server_address[1]
            client = app._build_redis_client(f"redis://127.0.0.1:{port}")
            errors = []

            def burst():
                try:
                    for _ in range(5):
                        client.get("launches_cache_v2")
                except Exception as exc:  # pragma: no cover - reported below
                    errors.append(exc)

            # The burst that took the server down was 44 requests in one second.
            threads = [threading.Thread(target=burst) for _ in range(44)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)
            client.connection_pool.disconnect()
        finally:
            server.shutdown()
            server.server_close()

        self.assertEqual(errors, [])
        self.assertGreater(server.peak, 1)
        self.assertLessEqual(server.peak, app.REDIS_MAX_CONNECTIONS)


class _FlakyClient:
    def __init__(self):
        self.calls = 0
        self.error = None

    def get(self, key):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return "value"


class RedisCircuitTests(unittest.TestCase):
    def test_connection_errors_open_the_circuit_and_calls_fail_without_reaching_redis(self):
        client = _FlakyClient()
        circuit = app._RedisCircuit(client, cooldown=60)
        client.error = redis.exceptions.ConnectionError("over limit")
        with self.assertRaises(redis.exceptions.ConnectionError):
            circuit.get("k")
        self.assertTrue(circuit.is_open())
        for _ in range(5):
            with self.assertRaises(redis.exceptions.ConnectionError):
                circuit.get("k")
        self.assertEqual(client.calls, 1)

    def test_one_probe_after_the_cooldown_closes_the_circuit(self):
        client = _FlakyClient()
        circuit = app._RedisCircuit(client, cooldown=0.05)
        client.error = redis.exceptions.TimeoutError("slow")
        with self.assertRaises(redis.exceptions.TimeoutError):
            circuit.get("k")
        client.error = None
        time.sleep(0.06)
        self.assertEqual(circuit.get("k"), "value")
        self.assertFalse(circuit.is_open())
        self.assertEqual(circuit.get("k"), "value")
        self.assertEqual(client.calls, 3)

    def test_a_failed_probe_keeps_the_circuit_open(self):
        client = _FlakyClient()
        circuit = app._RedisCircuit(client, cooldown=0.05)
        client.error = redis.exceptions.ConnectionError("down")
        with self.assertRaises(redis.exceptions.ConnectionError):
            circuit.get("k")
        time.sleep(0.06)
        with self.assertRaises(redis.exceptions.ConnectionError):
            circuit.get("k")
        with self.assertRaises(redis.exceptions.ConnectionError):
            circuit.get("k")
        self.assertEqual(client.calls, 2)

    def test_other_errors_do_not_open_the_circuit(self):
        client = _FlakyClient()
        circuit = app._RedisCircuit(client, cooldown=60)
        client.error = redis.exceptions.ResponseError("wrong type")
        with self.assertRaises(redis.exceptions.ResponseError):
            circuit.get("k")
        self.assertFalse(circuit.is_open())

    def test_startup_keeps_a_configured_client_that_is_unreachable(self):
        with patch.dict("os.environ", {"REDIS_URL": "redis://127.0.0.1:1"}, clear=True):
            client = app.get_redis_client()
        self.assertIsInstance(client, app._RedisCircuit)


class LaunchListAvailabilityTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()
        self.client = TestClient(app.app)

    def tearDown(self):
        app._reset_cache_coordination_for_tests()

    def _redis_down(self):
        return patch.object(app, "get_cached_data", side_effect=redis.exceptions.ConnectionError("down"))

    def test_a_failed_read_serves_the_list_held_in_memory(self):
        with patch.object(app, "get_cached_data", return_value=_launches()), \
             patch.object(app, "set_cached_data", return_value=True), \
             patch.object(app, "save_cache_to_file"):
            self.assertEqual(self.client.get("/launches_slim").status_code, 200)
        app._launches_mem["at"] = time.time() - app.LAUNCHES_MEM_TTL - 1

        with self._redis_down() as read:
            first = self.client.get("/launches_slim")
            second = self.client.get("/launches_slim")

        self.assertEqual(first.status_code, 200)
        self.assertEqual([l["id"] for l in first.json()["upcoming"]], ["u1"])
        self.assertEqual(first.json()["last_updated"], "2026-11-18T02:46:35Z")
        self.assertEqual(second.json(), first.json())
        # Redis is left alone for a while after the failure.
        self.assertEqual(read.call_count, 1)

    def test_a_failed_read_with_nothing_held_is_503_not_an_empty_list(self):
        with self._redis_down():
            for path in ("/launches_slim", "/launches"):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 503, path)
                self.assertEqual(response.headers["Retry-After"], app.CACHE_UNAVAILABLE_RETRY_AFTER)
                self.assertNotIn("upcoming", response.json())

    def test_a_missing_key_on_a_reachable_redis_is_still_an_empty_list(self):
        with patch.object(app, "get_cached_data", return_value=None):
            response = self.client.get("/launches_slim")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"upcoming": [], "previous": [], "last_updated": None})

    def test_internal_callers_get_an_empty_payload_and_no_exception(self):
        with self._redis_down():
            self.assertEqual(
                app._load_launch_payload(),
                {"upcoming": [], "previous": [], "last_updated": None},
            )
            self.assertFalse(app._is_current_or_next_launch("u1"))

    def test_health_reports_whether_a_launch_list_is_held(self):
        empty = self.client.get("/health")
        self.assertEqual(empty.status_code, 503)
        self.assertEqual(empty.json()["status"], "unavailable")

        with patch.object(app, "get_cached_data", return_value=_launches()), \
             patch.object(app, "set_cached_data", return_value=True), \
             patch.object(app, "save_cache_to_file"):
            self.client.get("/launches_slim")
        held = self.client.get("/health")
        self.assertEqual(held.status_code, 200)
        self.assertTrue(held.json()["launches_held"])
        self.assertEqual(held.json()["launches_last_updated"], "2026-11-18T02:46:35Z")

    def test_large_lists_are_compressed_for_clients_that_accept_it(self):
        payload = _launches()
        payload["upcoming"] = [_launch(f"u{i}", f"Launch {i}") for i in range(60)]
        with patch.object(app, "get_cached_data", return_value=payload), \
             patch.object(app, "set_cached_data", return_value=True), \
             patch.object(app, "save_cache_to_file"):
            response = self.client.get("/launches_slim", headers={"Accept-Encoding": "gzip"})
        self.assertEqual(response.headers.get("content-encoding"), "gzip")
        self.assertEqual(len(response.json()["upcoming"]), 60)


class NarrativesAvailabilityTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()
        self._local = dict(app._local_cache)
        app._local_cache.update({"launch_narratives": None, "last_updated": None})
        self.client = TestClient(app.app)

    def tearDown(self):
        app._local_cache.update(self._local)
        app._reset_cache_coordination_for_tests()

    @staticmethod
    def _stored(key, strict=False):
        return ["11/17 0246: Falcon 9 flew."] if key == app.CACHE_KEY else "2026-11-18T02:43:50Z"

    def test_a_failed_read_serves_the_narratives_held_in_memory(self):
        with patch.object(app, "get_cached_data", side_effect=self._stored):
            good = self.client.get("/recent_launches_narratives")
        self.assertEqual(good.status_code, 200)
        app._narratives_mem["at"] = time.time() - app.LAUNCHES_MEM_TTL - 1

        with patch.object(app, "get_cached_data", side_effect=redis.exceptions.ConnectionError("down")) as read:
            first = self.client.get("/recent_launches_narratives")
            second = self.client.get("/recent_launches_narratives")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json(), good.json())
        self.assertEqual(second.json(), good.json())
        self.assertEqual(read.call_count, 1)

    def test_a_failed_read_with_nothing_held_is_503_not_an_empty_list(self):
        with patch.object(app, "get_cached_data", side_effect=redis.exceptions.ConnectionError("down")):
            response = self.client.get("/recent_launches_narratives")
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("descriptions", response.json())

    def test_a_missing_key_on_a_reachable_redis_is_still_an_empty_list(self):
        with patch.object(app, "get_cached_data", return_value=None):
            response = self.client.get("/recent_launches_narratives")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"descriptions": [], "last_updated": None})


class _CountingRedis:
    def __init__(self):
        self.increments = []
        self.fail = False

    def hincrby(self, key, field, amount):
        if self.fail:
            raise redis.exceptions.ConnectionError("down")
        self.increments.append((key, field, amount))
        return amount

    def hgetall(self, key):
        return {}


class MetricsBatchingTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()
        self._redis = app.r
        self.fake = _CountingRedis()
        app.r = self.fake
        app._metric_last_flush = time.time()

    def tearDown(self):
        app.r = self._redis
        app._reset_cache_coordination_for_tests()

    def test_counting_a_request_does_not_touch_redis_until_the_flush(self):
        for _ in range(25):
            app.increment_metric("total_requests")
        app.increment_metric("cache_hits")
        self.assertEqual(self.fake.increments, [])

        app.flush_metrics()
        self.assertEqual(
            sorted(self.fake.increments),
            [(app.METRICS_KEY, "cache_hits", 1), (app.METRICS_KEY, "total_requests", 25)],
        )

    def test_a_failed_flush_keeps_the_counts_for_the_next_one(self):
        app.increment_metric("total_requests")
        app.increment_metric("total_requests")
        self.fake.fail = True
        app.flush_metrics()
        self.fake.fail = False
        app.increment_metric("total_requests")
        app.flush_metrics()
        self.assertEqual(self.fake.increments, [(app.METRICS_KEY, "total_requests", 3)])

    def test_the_first_count_after_the_interval_flushes(self):
        app._metric_last_flush = time.time() - app.METRICS_FLUSH_SEC - 1
        app.increment_metric("total_requests")
        self.assertEqual(self.fake.increments, [(app.METRICS_KEY, "total_requests", 1)])


class LaunchRecordFallbackTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()
        self._redis = app.r
        app.r = object()

    def tearDown(self):
        app.r = self._redis
        app._reset_cache_coordination_for_tests()

    def test_a_record_read_once_survives_a_failed_redis_read_without_a_live_fetch(self):
        record = {"id": "u1", "name": "Next"}
        with patch.object(app, "get_cached_data", return_value=record):
            self.assertEqual(app._get_raw_launch("u1"), record)
        with patch.object(app, "get_cached_data", return_value=None), \
             patch.object(app, "fetch_launch_details") as live:
            served, hit = app._serve_cached_or_live_raw("u1")
        self.assertEqual(served, record)
        self.assertTrue(hit)
        live.assert_not_called()

    def test_a_live_fetch_is_reused_for_a_few_minutes(self):
        record = {"id": "x9", "name": "Uncached"}
        with patch.object(app, "get_cached_data", return_value=None), \
             patch.object(app, "fetch_launch_details", return_value=record) as live:
            app._serve_cached_or_live_raw("x9")
            served, hit = app._serve_cached_or_live_raw("x9")
        self.assertEqual(served, record)
        self.assertTrue(hit)
        self.assertEqual(live.call_count, 1)

    def test_the_hot_record_is_read_from_memory_before_redis(self):
        record = {"id": "u1", "status": {"name": "Go"}}
        with patch.object(app, "set_cached_data", return_value=True):
            app._write_hot_raw("u1", record)
        with patch.object(app, "get_cached_data", side_effect=AssertionError("Redis was read")):
            self.assertEqual(app._read_hot_raw("u1"), record)

    def test_only_the_most_recent_records_are_held(self):
        for index in range(app.RAW_LAUNCH_MEM_MAX + 10):
            app._remember_raw_launch(f"id-{index}", {"id": index})
        self.assertEqual(len(app._raw_launch_mem), app.RAW_LAUNCH_MEM_MAX)
        self.assertIsNone(app._remembered_raw_launch("id-0"))
        self.assertIsNotNone(app._remembered_raw_launch(f"id-{app.RAW_LAUNCH_MEM_MAX + 9}"))


class NotifyFiveMinuteTests(unittest.TestCase):
    def setUp(self):
        app._reset_cache_coordination_for_tests()

    def test_t5m_is_accepted_and_has_its_own_template(self):
        payload = app.NotifyCopyRequest(
            launch_id="launch-notify-5m",
            event="t5m",
            mission="Starlink Group 10-20",
            pad="SLC-40",
            probability=70,
        )
        title, body = app.fallback_notify_copy(payload)
        self.assertIn("T-5m", title)
        self.assertIn("5 minutes", body)
        self.assertIn("70%", f"{title} {body}")
        self.assertIn("t5m", app._event_prompt_instructions("t5m"))

    def test_post_with_t5m_is_not_rejected(self):
        with patch.object(app, "r", None), \
             patch.object(app, "call_grok", side_effect=RuntimeError("no credits")):
            response = TestClient(app.app).post(
                "/notify/copy",
                json={"launch_id": "launch-notify-5m", "event": "t5m", "mission": "Starlink Group 10-20"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn("T-5m", response.json()["title"])


if __name__ == "__main__":
    unittest.main()
