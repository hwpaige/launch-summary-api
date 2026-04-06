import os
import json
from datetime import datetime, timedelta, timezone

# Mock Redis-like behavior for testing
class MockRedis:
    def __init__(self):
        self.metrics = {}
        self.history = []

    def hincrby(self, key, field, amount):
        if key not in self.metrics:
            self.metrics[key] = {}
        self.metrics[key][field] = self.metrics[key].get(field, 0) + amount

    def hgetall(self, key):
        return self.metrics.get(key, {})

    def lpush(self, key, value):
        self.history.insert(0, value)

    def ltrim(self, key, start, stop):
        self.history = self.history[start:stop+1]

    def lrange(self, key, start, stop):
        if stop == -1:
            return self.history[start:]
        return self.history[start:stop+1]

r = MockRedis()
METRICS_KEY = "app_metrics_v2"
METRICS_HISTORY_KEY = "app_metrics_history_v2"
HISTORY_LIMIT = 100

def increment_metric(field):
    r.hincrby(METRICS_KEY, field, 1)

def get_metrics(include_history=True, range_type="1h"):
    current = {k: int(v) for k, v in r.hgetall(METRICS_KEY).items()}
    if not current:
        current = {"total_requests": 0, "cache_hits": 0, "cache_misses": 0, "api_calls": 0}
        
    if not include_history:
        return current
        
    history_data = r.lrange(METRICS_HISTORY_KEY, 0, -1)
    history = [json.loads(s) for s in history_data]
    history.reverse() 

    now = datetime.now(timezone.utc)
    if range_type == "1h":
        start_time = now - timedelta(hours=1)
        history = [h for h in history if datetime.fromisoformat(h['timestamp'].replace('Z', '+00:00')) > start_time]

    range_stats = {
        "total_requests": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "api_calls": 0
    }
    if history:
        first = history[0]['data']
        last = history[-1]['data']
        range_stats = {
            "total_requests": last.get('total_requests', 0) - first.get('total_requests', 0),
            "cache_hits": last.get('cache_hits', 0) - first.get('cache_hits', 0),
            "cache_misses": last.get('cache_misses', 0) - first.get('cache_misses', 0),
            "api_calls": last.get('api_calls', 0) - first.get('api_calls', 0)
        }
    
    hits_per_day = 0
    if len(history) > 1:
        first_h = history[0]
        last_h = history[-1]
        t1 = datetime.fromisoformat(first_h['timestamp'].replace('Z', '+00:00'))
        t2 = datetime.fromisoformat(last_h['timestamp'].replace('Z', '+00:00'))
        duration_hours = (t2 - t1).total_seconds() / 3600
        if duration_hours > 0.1:
            hits_diff = last_h['data']['total_requests'] - first_h['data']['total_requests']
            hits_per_day = (hits_diff / duration_hours) * 24

    return {
        "current": current,
        "range_stats": range_stats,
        "history": history,
        "hits_per_day": round(hits_per_day, 1)
    }

def record_snapshot(ts=None):
    current_metrics = {k: int(v) for k, v in r.hgetall(METRICS_KEY).items()}
    if not current_metrics:
        current_metrics = {"total_requests": 0, "cache_hits": 0, "cache_misses": 0, "api_calls": 0}
    snapshot = {
        "timestamp": ts or datetime.now(timezone.utc).isoformat(),
        "data": current_metrics
    }
    r.lpush(METRICS_HISTORY_KEY, json.dumps(snapshot))

# Test Scenario
print("--- Initial state ---")
print(get_metrics())

print("\n--- 10 requests, all hits ---")
for _ in range(10):
    increment_metric("total_requests")
    increment_metric("cache_hits")

now = datetime.now(timezone.utc)
record_snapshot((now - timedelta(minutes=5)).isoformat())

print("\n--- 5 more requests, all misses ---")
for _ in range(5):
    increment_metric("total_requests")
    increment_metric("cache_misses")

record_snapshot(now.isoformat())

metrics = get_metrics()
print(f"Absolute live metrics: {metrics['current']}")
print(f"Range-relative metrics (should reflect diff): {metrics['range_stats']}")
print(f"Hits per day: {metrics['hits_per_day']}")

# Problem demonstration: Only one snapshot
r.history = []
record_snapshot(now.isoformat())
metrics = get_metrics()
print(f"\n--- One snapshot only ---")
print(f"Absolute live metrics: {metrics['current']}")
print(f"Range-relative metrics: {metrics['range_stats']}")
print(f"Note: Absolute metrics are correct (15), range metrics are 0 (expected for single snapshot)")
