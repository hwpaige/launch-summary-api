import os
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import requests
import ast
import re
import redis
import json
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env if present

app = FastAPI()

# Redis Configuration for persistence across Heroku builds
def get_redis_client():
    # List of potential Redis environment variables used by various Heroku add-ons
    redis_env_vars = ["REDIS_URL", "REDISCLOUD_URL", "REDISTOGO_URL"]
    
    # Also check for HEROKU_REDIS_*_URL
    for key in os.environ:
        if key.startswith("HEROKU_REDIS_") and key.endswith("_URL"):
            redis_env_vars.append(key)

    for var in redis_env_vars:
        url = os.getenv(var)
        if url:
            try:
                # Heroku Redis often requires SSL with cert verification disabled for self-signed certs
                if url.startswith("rediss://"):
                    client = redis.from_url(url, decode_responses=True, ssl_cert_reqs=None)
                else:
                    client = redis.from_url(url, decode_responses=True)
                client.ping()
                print(f"Connected to Redis via {var}")
                return client
            except Exception as e:
                print(f"Failed to connect to Redis via {var}: {e}")
    
    print("No Redis instance found or connection failed. Using in-memory fallback (non-persistent).")
    return None

r = get_redis_client()

# In-memory storage fallback (used only if Redis is unavailable)
_local_cache = {
    "launch_narratives": None,
    "last_updated": None
}
_local_metrics = {
    "total_requests": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "api_calls": 0
}
_local_metrics_history = []

CACHE_KEY = "launch_narratives_v2"
CACHE_TIME_KEY = "last_updated_v2"
METRICS_KEY = "app_metrics_v2"
METRICS_HISTORY_KEY = "app_metrics_history_v2"
CACHE_TTL = 3600  # 1 hour TTL in seconds
_last_snapshot_time = 0

def increment_metric(field):
    """Increment a metric in Redis or memory."""
    if r:
        try:
            r.hincrby(METRICS_KEY, field, 1)
            return
        except Exception as e:
            print(f"Redis error in increment_metric: {e}")
    
    # Fallback to in-memory
    if field in _local_metrics:
        _local_metrics[field] += 1

def record_snapshot():
    """Record a snapshot of current metrics for historical tracking."""
    global _last_snapshot_time
    now = datetime.now(timezone.utc).timestamp()
    if now - _last_snapshot_time < 25: # Throttle to ~30s
        return
    _last_snapshot_time = now
    
    current_metrics = get_metrics(include_history=False)
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": current_metrics
    }
    
    if r:
        try:
            r.lpush(METRICS_HISTORY_KEY, json.dumps(snapshot))
            r.ltrim(METRICS_HISTORY_KEY, 0, 29) # Keep last 30 snapshots
        except Exception as e:
            print(f"Redis error in record_snapshot: {e}")
    
    _local_metrics_history.append(snapshot)
    if len(_local_metrics_history) > 30:
        _local_metrics_history.pop(0)

def get_metrics(include_history=True):
    """Retrieve metrics from Redis or memory."""
    current = {
        "total_requests": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "api_calls": 0
    }
    
    if r:
        try:
            data = r.hgetall(METRICS_KEY)
            if data:
                current = {k: int(v) for k, v in data.items()}
        except Exception as e:
            print(f"Redis error in get_metrics: {e}")
    else:
        current = _local_metrics.copy()
        
    if not include_history:
        return current
        
    history = []
    if r:
        try:
            history_data = r.lrange(METRICS_HISTORY_KEY, 0, -1)
            history = [json.loads(s) for s in history_data]
            history.reverse() # Oldest first for charting
        except Exception as e:
            print(f"Redis error fetching history: {e}")
    else:
        history = _local_metrics_history
        
    return {"current": current, "history": history}

def generate_narratives(existing_narratives=None):
    """Fetch launches and generate narratives using Grok, appending new ones only."""
    increment_metric("api_calls")
    current_time = datetime.now(timezone.utc)
    three_months_ago = current_time - timedelta(days=90)
    # Note: Using LL 2.0.0 as requested in the snippet
    url = (
        f"https://ll.thespacedevs.com/2.0.0/launch/previous/"
        f"?lsp__name=SpaceX"
        f"&net__gte={three_months_ago.strftime('%Y-%m-%d')}"
        f"&net__lte={current_time.strftime('%Y-%m-%d')}"
        f"&limit=40"
        f"&ordering=-net"
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch launches: {response.status_code}")
    
    data = response.json().get('results', [])
    
    launches = []
    for launch in data:
        net_str = launch['net']
        # Handle cases where Z might be missing or other ISO formats
        try:
            net_dt = datetime.fromisoformat(net_str.replace('Z', '+00:00'))
        except ValueError:
            # Fallback if the format is slightly different
            net_dt = datetime.strptime(net_str, "%Y-%m-%dT%H:%M:%SZ")
            
        date_time = net_dt.strftime("%m/%d %H%M")
        
        mission = launch['name']
        pad = launch['pad']['name']
        rocket = launch['rocket']['configuration']['name']
        orbit = launch.get('mission', {}).get('orbit', {}).get('name', 'Unknown')
        status = launch['status']['name']
        
        launches.append({
            "date_time": date_time,
            "mission": mission,
            "pad": pad,
            "rocket": rocket,
            "orbit": orbit,
            "status": status
        })
    
    if not launches:
        return existing_narratives if existing_narratives else []

    # Identify new launches not in the current cache
    existing_keys = set()
    if existing_narratives:
        for narr in existing_narratives:
            # Extract "MM/DD HHMM" from the start of the narrative
            parts = narr.split(': ', 1)
            if parts:
                existing_keys.add(parts[0])
    
    new_launches = [l for l in launches if l['date_time'] not in existing_keys]
    
    if existing_narratives and not new_launches:
        print("No new launches found. Cache is up to date.")
        return existing_narratives

    # If we have existing narratives, only process the new ones to append
    # If no cache exists, process all fetched launches
    launches_to_process = new_launches if existing_narratives else launches
    
    launch_list = "\n".join([
        f"{l['date_time']}: {l['mission']} from {l['pad']}, {l['rocket']} to {l['orbit']}, status {l['status']}"
        for l in launches_to_process
    ])
    
    prompt = f"""Generate a list of short news like descriptions for these SpaceX launches:
{launch_list}

In the style of Cities Skylines notifications: kind of witty and dry. Factual, complete, somewhat technical - think Kerbal Space Program.

IMPORTANT: Return ONLY a Python list assignment and nothing else. Be extremely concise for each entry to avoid truncation. No conversational filler, no introductory text, no markdown formatting.

Examples:
- Falcon 9 hoists MTG-S1/Sentinel-4A to geosync from LC-39A; Ariane's loss is our nominal gain, booster recovered without drama.
- 500th Falcon 9 ignites with 27 Starlinks from SLC-40; B1067 clocks 29th flight, orbit insertion as predictable as gravity.
- Starship Flight 10 ignites from Starbase; hot-staging clean, ship splashes precisely in Indian Ocean, Super Heavy boosts back nominally.

Format each as: month/day HHMM: description

Output as a Python list assignment: launch_descriptions = [...]"""
    
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "grok-4-1-fast-reasoning",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 4000
        }
        response = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload)
        if response.status_code != 200:
            raise ValueError(f"Grok API call failed: {response.status_code} - {response.text}")
        
        data = response.json()
        generated_text = data['choices'][0]['message']['content']
        print(f"DEBUG: Raw Grok response: {generated_text}")
    except Exception as e:
        raise ValueError(f"Grok API call failed: {str(e)}")
    
    try:
        # Robust extraction: find all strings that match the pattern "month/day HHMM: description"
        new_descriptions = re.findall(r'["\'](\d{1,2}/\d{1,2} \d{4}: .*?)["\']', generated_text)
        
        if not new_descriptions:
            # Fallback for alternative formatting
            start_idx = generated_text.find('[')
            end_idx = generated_text.rfind(']') + 1
            if start_idx != -1 and end_idx > start_idx:
                try:
                    list_str = generated_text[start_idx:end_idx]
                    new_descriptions = ast.literal_eval(list_str)
                except:
                    pass
        
        if not new_descriptions:
            raise ValueError("No valid launch descriptions found in response")
        
        if not isinstance(new_descriptions, list) or not all(isinstance(d, str) for d in new_descriptions):
            raise ValueError("Parsed content is not a list of strings")
            
        print(f"Successfully generated {len(new_descriptions)} new narratives.")
        
        if existing_narratives:
            # Prepend new ones and limit the total list size
            combined = new_descriptions + existing_narratives
            # We assume they are mostly sorted, but we could do a final sort if needed.
            # However, since we don't have the year in the string, a simple string sort is risky.
            # Prepending preserves the newest-first order from the API.
            return combined[:50]
        
        return new_descriptions
    except Exception as e:
        raise ValueError(f"Failed to parse Grok response: {str(e)}")

@app.get("/recent_launches_narratives")
def get_narratives():
    """Serve from cache if available; generate otherwise."""
    increment_metric("total_requests")
    
    current_time = datetime.now(timezone.utc)
    cached_narratives = None
    last_updated = None

    # Try to get from Redis
    if r:
        try:
            data = r.get(CACHE_KEY)
            time_str = r.get(CACHE_TIME_KEY)
            if data and time_str:
                cached_narratives = json.loads(data)
                last_updated = datetime.fromisoformat(time_str)
        except Exception as e:
            print(f"Redis error in get_narratives: {e}")
    else:
        # Fallback to in-memory
        cached_narratives = _local_cache["launch_narratives"]
        last_updated = _local_cache["last_updated"]

    # Check if cache is valid (not None and within TTL)
    if cached_narratives and last_updated:
        elapsed = (current_time - last_updated).total_seconds()
        if elapsed < CACHE_TTL:
            increment_metric("cache_hits")
            return {"descriptions": cached_narratives}
    
    increment_metric("cache_misses")
    descriptions = generate_narratives(existing_narratives=cached_narratives)
    
    # Update cache (Redis and/or in-memory)
    if r:
        try:
            r.set(CACHE_KEY, json.dumps(descriptions))
            r.set(CACHE_TIME_KEY, current_time.isoformat())
        except Exception as e:
            print(f"Redis error updating cache: {e}")
    
    _local_cache["launch_narratives"] = descriptions
    _local_cache["last_updated"] = current_time
        
    return {"descriptions": descriptions}

@app.get("/metrics")
def get_app_metrics():
    """Endpoint to fetch application metrics."""
    record_snapshot()
    return get_metrics()

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    """Serve the dashboard UI."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SpaceX Launch Narratives Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; }
        .ticker-item { border-left: 4px solid #3b82f6; }
        .chart-container { height: 60px; width: 100%; margin-top: 1rem; }
    </style>
</head>
<body class="bg-slate-950 text-slate-200 min-h-screen">
    <nav class="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex items-center justify-between h-16">
                <div class="flex items-center gap-2">
                    <i data-lucide="rocket" class="text-blue-500 w-8 h-8"></i>
                    <span class="text-xl font-bold tracking-tight">SpaceX Narratives</span>
                </div>
                <div class="flex items-center gap-4">
                    <button onclick="refreshData()" class="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 transition-colors rounded-lg font-semibold text-sm">
                        <i data-lucide="refresh-cw" class="w-4 h-4" id="refresh-icon"></i>
                        Force Refresh
                    </button>
                </div>
            </div>
        </div>
    </nav>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <!-- Metrics Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
            <!-- Total Requests Card -->
            <div class="bg-slate-900 border border-slate-800 p-6 rounded-2xl flex flex-col justify-between">
                <div>
                    <div class="flex items-center justify-between mb-4">
                        <span class="text-slate-400 text-sm font-medium">Total Requests</span>
                        <i data-lucide="activity" class="text-blue-400 w-5 h-5"></i>
                    </div>
                    <div class="text-3xl font-bold" id="metric-total-requests">0</div>
                </div>
                <div class="chart-container">
                    <canvas id="chart-requests"></canvas>
                </div>
            </div>

            <!-- Cache Hits Card -->
            <div class="bg-slate-900 border border-slate-800 p-6 rounded-2xl flex flex-col justify-between">
                <div>
                    <div class="flex items-center justify-between mb-4">
                        <span class="text-slate-400 text-sm font-medium">Cache Hits</span>
                        <i data-lucide="database" class="text-emerald-400 w-5 h-5"></i>
                    </div>
                    <div class="text-3xl font-bold text-emerald-400" id="metric-cache-hits">0</div>
                </div>
                <div class="chart-container">
                    <canvas id="chart-hits"></canvas>
                </div>
            </div>

            <!-- API Calls Card -->
            <div class="bg-slate-900 border border-slate-800 p-6 rounded-2xl flex flex-col justify-between">
                <div>
                    <div class="flex items-center justify-between mb-4">
                        <span class="text-slate-400 text-sm font-medium">Grok API Calls</span>
                        <i data-lucide="brain-circuit" class="text-purple-400 w-5 h-5"></i>
                    </div>
                    <div class="text-3xl font-bold text-purple-400" id="metric-api-calls">0</div>
                </div>
                <div class="chart-container">
                    <canvas id="chart-api"></canvas>
                </div>
            </div>

            <!-- Efficiency Card -->
            <div class="bg-slate-900 border border-slate-800 p-6 rounded-2xl flex flex-col justify-between">
                <div>
                    <div class="flex items-center justify-between mb-4">
                        <span class="text-slate-400 text-sm font-medium">Cache Efficiency</span>
                        <i data-lucide="zap" class="text-yellow-400 w-5 h-5"></i>
                    </div>
                    <div class="text-3xl font-bold text-yellow-400" id="metric-efficiency">0%</div>
                </div>
                <div class="chart-container">
                    <canvas id="chart-efficiency"></canvas>
                </div>
            </div>
        </div>

        <!-- Narratives Section -->
        <div class="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
            <div class="px-6 py-4 border-b border-slate-800 bg-slate-800/30 flex items-center justify-between">
                <h2 class="text-lg font-semibold flex items-center gap-2">
                    <i data-lucide="list" class="w-5 h-5 text-blue-500"></i>
                    Current Narratives
                </h2>
                <span class="text-xs text-slate-500 uppercase tracking-widest font-bold" id="last-updated">Updating...</span>
            </div>
            <div class="p-6">
                <div id="narratives-list" class="space-y-4">
                    <!-- Loaded dynamically -->
                    <div class="animate-pulse flex space-x-4">
                        <div class="flex-1 space-y-4 py-1">
                            <div class="h-4 bg-slate-800 rounded w-3/4"></div>
                            <div class="h-4 bg-slate-800 rounded"></div>
                            <div class="h-4 bg-slate-800 rounded w-5/6"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <script>
        lucide.createIcons();

        // Chart instances
        const charts = {};

        function initChart(id, color) {
            const ctx = document.getElementById(id).getContext('2d');
            return new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Array(30).fill(''),
                    datasets: [{
                        data: Array(30).fill(null),
                        borderColor: color,
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.4,
                        fill: true,
                        backgroundColor: color + '20'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false }, tooltip: { enabled: true } },
                    scales: {
                        x: { display: false },
                        y: { display: false, beginAtZero: true }
                    }
                }
            });
        }

        charts.requests = initChart('chart-requests', '#3b82f6');
        charts.hits = initChart('chart-hits', '#10b981');
        charts.api = initChart('chart-api', '#a855f7');
        charts.efficiency = initChart('chart-efficiency', '#eab308');

        async function fetchMetrics() {
            try {
                const response = await fetch('/metrics');
                const data = await response.json();
                
                const current = data.current || {};
                const history = data.history || [];
                
                const total = current.total_requests || 0;
                const hits = current.cache_hits || 0;
                const apiCalls = current.api_calls || 0;
                
                document.getElementById('metric-total-requests').textContent = total.toLocaleString();
                document.getElementById('metric-cache-hits').textContent = hits.toLocaleString();
                document.getElementById('metric-api-calls').textContent = apiCalls.toLocaleString();
                
                const efficiency = total > 0 ? Math.round((hits / total) * 100) : 0;
                document.getElementById('metric-efficiency').textContent = efficiency + '%';

                // Update charts
                if (history.length > 0) {
                    const updateChart = (chart, key, isEfficiency = false) => {
                        const values = history.map(h => {
                            if (isEfficiency) {
                                const t = h.data.total_requests || 0;
                                return t > 0 ? (h.data.cache_hits / t) * 100 : 0;
                            }
                            return h.data[key] || 0;
                        });
                        
                        // Pad with nulls if history is short
                        const padded = [...Array(30 - values.length).fill(null), ...values];
                        chart.data.datasets[0].data = padded;
                        chart.update('none');
                    };

                    updateChart(charts.requests, 'total_requests');
                    updateChart(charts.hits, 'cache_hits');
                    updateChart(charts.api, 'api_calls');
                    updateChart(charts.efficiency, '', true);
                }
            } catch (error) {
                console.error('Error fetching metrics:', error);
            }
        }

        async function fetchNarratives() {
            try {
                const response = await fetch('/recent_launches_narratives');
                const data = await response.json();
                const list = document.getElementById('narratives-list');
                list.innerHTML = '';
                
                if (data.descriptions && data.descriptions.length > 0) {
                    data.descriptions.forEach(desc => {
                        const parts = desc.split(': ', 2);
                        const date = parts[0];
                        const text = parts[1] || '';
                        
                        const div = document.createElement('div');
                        div.className = 'ticker-item bg-slate-800/40 p-4 rounded-r-lg border-l-4 border-blue-500 hover:bg-slate-800/60 transition-colors';
                        div.innerHTML = `
                            <div class="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4">
                                <span class="text-blue-400 font-mono font-bold whitespace-nowrap">${date}</span>
                                <p class="text-slate-200 leading-relaxed">${text}</p>
                            </div>
                        `;
                        list.appendChild(div);
                    });
                } else {
                    list.innerHTML = '<p class="text-slate-500 text-center py-8">No narratives available.</p>';
                }
                
                document.getElementById('last-updated').textContent = 'Last updated: ' + new Date().toLocaleTimeString();
            } catch (error) {
                console.error('Error fetching narratives:', error);
            }
        }

        async function refreshData() {
            const icon = document.getElementById('refresh-icon');
            icon.classList.add('animate-spin');
            
            try {
                await fetch('/refresh', { method: 'POST' });
                await Promise.all([fetchMetrics(), fetchNarratives()]);
            } catch (error) {
                console.error('Error refreshing data:', error);
            } finally {
                setTimeout(() => icon.classList.remove('animate-spin'), 500);
            }
        }

        // Initial load
        fetchMetrics();
        fetchNarratives();

        // Auto refresh metrics every 30s
        setInterval(fetchMetrics, 30000);
    </script>
</body>
</html>
    """

@app.post("/refresh")
def refresh_cache():
    """Force refresh the cache (call this via scheduler)."""
    print("Manual cache refresh triggered.")
    
    # Try to get the current cache to pass it for incremental update
    cached_narratives = None
    if r:
        try:
            data = r.get(CACHE_KEY)
            if data:
                cached_narratives = json.loads(data)
        except:
            pass
    
    if not cached_narratives:
        cached_narratives = _local_cache["launch_narratives"]

    descriptions = generate_narratives(existing_narratives=cached_narratives)
    current_time = datetime.now(timezone.utc)
    
    # Update Redis
    if r:
        try:
            r.set(CACHE_KEY, json.dumps(descriptions))
            r.set(CACHE_TIME_KEY, current_time.isoformat())
        except Exception as e:
            print(f"Redis error in refresh_cache: {e}")
            
    # Update in-memory fallback
    _local_cache["launch_narratives"] = descriptions
    _local_cache["last_updated"] = current_time
    
    count = len(descriptions)
    print(f"Cache successfully refreshed with {count} narratives.")
    return {
        "status": "Cache refreshed",
        "count": count,
        "timestamp": current_time.isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))