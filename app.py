import os
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import requests
import ast
import re
import redis
import json
import math
import time
import threading
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
HISTORY_LIMIT = 10080 # 7 days at 1 minute intervals
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
    if now - _last_snapshot_time < 55: # Throttle to ~1m
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
            r.ltrim(METRICS_HISTORY_KEY, 0, HISTORY_LIMIT - 1)
        except Exception as e:
            print(f"Redis error in record_snapshot: {e}")
    
    _local_metrics_history.append(snapshot)
    if len(_local_metrics_history) > HISTORY_LIMIT:
        _local_metrics_history.pop(0)

def get_metrics(include_history=True, range_type="1h"):
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
        history = _local_metrics_history.copy()
        
    # Filter by range
    now = datetime.now(timezone.utc)
    if range_type == "1h":
        start_time = now - timedelta(hours=1)
        history = [h for h in history if datetime.fromisoformat(h['timestamp'].replace('Z', '+00:00')) > start_time]
    elif range_type == "24h":
        start_time = now - timedelta(hours=24)
        history = [h for h in history if datetime.fromisoformat(h['timestamp'].replace('Z', '+00:00')) > start_time]
        if len(history) > 100:
            history = history[::15] # ~15m intervals
    elif range_type == "7d":
        start_time = now - timedelta(days=7)
        history = [h for h in history if datetime.fromisoformat(h['timestamp'].replace('Z', '+00:00')) > start_time]
        if len(history) > 200:
            history = history[::60] # ~1h intervals

    # Calculate live hits/day (last 24h or available range)
    hits_per_day = 0
    if len(history) > 1:
        first = history[0]
        last = history[-1]
        try:
            t1 = datetime.fromisoformat(first['timestamp'].replace('Z', '+00:00'))
            t2 = datetime.fromisoformat(last['timestamp'].replace('Z', '+00:00'))
            duration_hours = (t2 - t1).total_seconds() / 3600
            if duration_hours > 0.1: # At least 6 mins of data
                hits_diff = last['data']['total_requests'] - first['data']['total_requests']
                hits_per_day = (hits_diff / duration_hours) * 24
        except: pass

    return {
        "current": current, 
        "history": history, 
        "hits_per_day": round(hits_per_day, 1)
    }

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

# --- Ported Fetch Functions from functions.py ---

LL_API_KEY = os.getenv("LL_API_KEY", "9b91363961799d7f79aabe547ed0f7be914664dd")

def parse_launch_data(launch: dict, is_detailed: bool = False) -> dict:
    """Helper to parse raw API launch data into the dashboard's internal format."""
    launcher_stage = launch.get('rocket', {}).get('launcher_stage', [])
    landing_type = None
    landing_location = None
    if isinstance(launcher_stage, list) and len(launcher_stage) > 0:
        landing = launcher_stage[0].get('landing')
        if landing:
            landing_type = landing.get('type', {}).get('name')
            landing_location = landing.get('landing_location', {}).get('name')
            if not landing_location:
                landing_location = landing.get('location', {}).get('name')
    
    mission_data = launch.get('mission') or {}
    
    return {
        'id': launch.get('id'),
        'mission': launch.get('name', 'Unknown'),
        'date': launch.get('net').split('T')[0] if launch.get('net') else 'TBD',
        'time': launch.get('net').split('T')[1].split('Z')[0] if launch.get('net') and 'T' in launch.get('net') else 'TBD',
        'net': launch.get('net'),
        'status': launch.get('status', {}).get('name', 'Unknown'),
        'rocket': launch.get('rocket', {}).get('configuration', {}).get('name', 'Unknown'),
        'orbit': mission_data.get('orbit', {}).get('name', 'Unknown'),
        'pad': launch.get('pad', {}).get('name', 'Unknown'),
        'video_url': launch.get('vidURLs', [{}])[0].get('url', '') if launch.get('vidURLs') else '',
        'x_video_url': next((v['url'] for v in launch.get('vidURLs', []) if v.get('url') and ('x.com' in v['url'].lower() or 'twitter.com' in v['url'].lower())), '') if launch.get('vidURLs') else '',
        'landing_type': landing_type,
        'landing_location': landing_location,
        'is_detailed': is_detailed,
        # New enriched fields for "ALL data" view
        'description': mission_data.get('description', ''),
        'image': launch.get('image', ''),
        'window_start': launch.get('window_start'),
        'window_end': launch.get('window_end'),
        'probability': launch.get('probability'),
        'holdreason': launch.get('holdreason'),
        'failreason': launch.get('failreason'),
        # Ensure every field returned by the API is cached and available
        'all_data': launch
    }

def fetch_launch_details(launch_id: str):
    """Fetch detailed information for a single launch."""
    if not launch_id:
        return None
    url = f"https://ll.thespacedevs.com/2.3.0/launches/{launch_id}/?mode=detailed"
    try:
        headers = {'Authorization': f'Token {LL_API_KEY}'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Failed to fetch launch details for {launch_id}: {e}")
        return None

def fetch_launches():
    """Fetch SpaceX launch data (v2.3.0) using detailed mode to get all fields."""
    try:
        headers = {'Authorization': f'Token {LL_API_KEY}'}
        
        # Fetch previous launches with mode=detailed to get ALL data fields
        prev_url = 'https://ll.thespacedevs.com/2.3.0/launches/previous/?lsp__name=SpaceX&limit=15&mode=detailed'
        prev_response = requests.get(prev_url, headers=headers, timeout=15)
        prev_response.raise_for_status()
        prev_data = prev_response.json().get('results', [])
        
        # Fetch upcoming launches with mode=detailed to get ALL data fields
        up_url = 'https://ll.thespacedevs.com/2.3.0/launches/upcoming/?lsp__name=SpaceX&limit=15&mode=detailed'
        up_response = requests.get(up_url, headers=headers, timeout=15)
        up_response.raise_for_status()
        up_data = up_response.json().get('results', [])
        
        return {
            'previous': [parse_launch_data(l, is_detailed=True) for l in prev_data],
            'upcoming': [parse_launch_data(l, is_detailed=True) for l in up_data]
        }
    except Exception as e:
        print(f"Error in fetch_launches: {e}")
        return {'previous': [], 'upcoming': []}

def parse_metar(raw_metar: str):
    """Parse METAR string to extract weather data."""
    temperature_c = 25
    wind_speed_kts = 0
    wind_direction = 0
    cloud_cover = 0

    try:
        # Extract temperature
        temp_match = re.search(r'(\d{1,3})/', raw_metar)
        if temp_match:
            temperature_c = int(temp_match.group(1))
            if temp_match.start() > 0 and raw_metar[temp_match.start()-1] == 'M':
                temperature_c = -temperature_c

        # Extract wind
        wind_match = re.search(r'(\d{3})(\d{2,3})(?:G\d{2,3})?KT', raw_metar)
        if wind_match:
            wind_direction = int(wind_match.group(1))
            wind_speed_kts = int(wind_match.group(2))

        # Cloud cover estimation
        if 'SKC' in raw_metar or 'CLR' in raw_metar:
            cloud_cover = 0
        elif 'FEW' in raw_metar:
            cloud_cover = 25
        elif 'SCT' in raw_metar:
            cloud_cover = 50
        elif 'BKN' in raw_metar:
            cloud_cover = 75
        elif 'OVC' in raw_metar:
            cloud_cover = 100
        else:
            cloud_cover = 50
    except Exception as e:
        print(f"Error parsing METAR: {e}")

    return {
        'temperature_c': temperature_c,
        'temperature_f': temperature_c * 9 / 5 + 32,
        'wind_speed_ms': wind_speed_kts * 0.514444,
        'wind_speed_kts': wind_speed_kts,
        'wind_direction': wind_direction,
        'cloud_cover': cloud_cover,
        'raw': raw_metar
    }

def fetch_weather(location: str):
    """Fetch METAR weather data for a given location."""
    metar_stations = {
        'Starbase': 'KBRO',
        'Vandy': 'KVBG',
        'Cape': 'KMLB',
        'Hawthorne': 'KHHR'
    }
    station_id = metar_stations.get(location, 'KBRO')
    url = f"https://aviationweather.gov/api/data/metar?ids={station_id}&format=raw"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        raw_metar = response.text.strip()
        if not raw_metar:
            raise ValueError("Empty METAR response")
        return parse_metar(raw_metar)
    except Exception as e:
        print(f"Error fetching weather for {location}: {e}")
        return {
            'temperature_c': 25, 'temperature_f': 77,
            'wind_speed_ms': 5, 'wind_speed_kts': 9.7,
            'wind_direction': 90, 'cloud_cover': 50,
            'error': str(e)
        }

def fetch_external_narratives():
    """Fetch narratives from the external API as specified in functions.py."""
    url = "https://launch-narrative-api-dafccc521fb8.herokuapp.com/recent_launches_narratives"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and 'descriptions' in data:
            return data['descriptions']
        return data
    except Exception as e:
        print(f"Error fetching external narratives: {e}")
        return []

# --- New Endpoints ---

@app.get("/launches")
def get_launches(force: bool = False):
    increment_metric("total_requests")
    cache_key = "launches_cache_v1"
    if r and not force:
        try:
            cached = r.get(cache_key)
            if cached:
                increment_metric("cache_hits")
                return json.loads(cached)
        except: pass
    
    increment_metric("cache_misses")
    data = fetch_launches()
    if r and (data['previous'] or data['upcoming']):
        try:
            r.setex(cache_key, 600, json.dumps(data))
        except: pass
    return data

@app.get("/weather/{location}")
def get_weather(location: str, force: bool = False):
    increment_metric("total_requests")
    cache_key = f"weather_cache_{location}"
    if r and not force:
        try:
            cached = r.get(cache_key)
            if cached:
                increment_metric("cache_hits")
                return json.loads(cached)
        except: pass
    
    increment_metric("cache_misses")
    data = fetch_weather(location)
    if r:
        try:
            r.setex(cache_key, 900, json.dumps(data))
        except: pass
    return data

@app.get("/weather_all")
def get_all_weather(force: bool = False):
    increment_metric("total_requests")
    locations = ['Starbase', 'Vandy', 'Cape', 'Hawthorne']
    return {loc: fetch_weather(loc) for loc in locations}

@app.get("/launch_details/{launch_id}")
def get_launch_details(launch_id: str):
    increment_metric("total_requests")
    return fetch_launch_details(launch_id)

@app.get("/external_narratives")
def get_all_narratives():
    increment_metric("total_requests")
    return {"descriptions": fetch_external_narratives()}

@app.get("/recent_launches_narratives")
def get_narratives(force: bool = False):
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
    if not force and cached_narratives and last_updated:
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
def get_app_metrics(range: str = "1h"):
    """Endpoint to fetch application metrics."""
    record_snapshot()
    return get_metrics(range_type=range)

def start_metrics_background_thread():
    def run():
        # Wait a bit for the app to start
        time.sleep(10)
        while True:
            try:
                record_snapshot()
            except Exception as e:
                print(f"Background metrics error: {e}")
            time.sleep(30) # Check every 30s, record_snapshot will throttle to 1m
    thread = threading.Thread(target=run, daemon=True)
    thread.start()

start_metrics_background_thread()

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
        .tab-active { border-bottom: 2px solid #3b82f6; color: #3b82f6; }
        .range-active { background-color: #2563eb !important; color: white !important; }
        .hidden { display: none; }
        .launch-card:hover { transform: translateY(-1px); }
        pre::-webkit-scrollbar { width: 6px; height: 6px; }
        pre::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
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
                    <!-- Global refresh removed, now per-tab -->
                </div>
            </div>
        </div>
    </nav>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <!-- Metrics Header & Time Range -->
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
            <div>
                <h2 class="text-2xl font-bold tracking-tight">System Metrics</h2>
                <p class="text-slate-400 text-sm">Real-time performance and API usage tracking.</p>
            </div>
            <div class="flex bg-slate-900 border border-slate-800 p-1 rounded-xl">
                <button onclick="changeRange('1h')" id="range-1h" class="px-4 py-1.5 rounded-lg text-sm font-medium transition-all range-active">1h</button>
                <button onclick="changeRange('24h')" id="range-24h" class="px-4 py-1.5 rounded-lg text-sm font-medium transition-all text-slate-400 hover:text-slate-200">24h</button>
                <button onclick="changeRange('7d')" id="range-7d" class="px-4 py-1.5 rounded-lg text-sm font-medium transition-all text-slate-400 hover:text-slate-200">7d</button>
            </div>
        </div>

        <!-- Key Stats Row -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
             <div class="bg-slate-900/50 border border-slate-800 p-4 rounded-2xl flex items-center gap-4">
                <div class="p-3 bg-blue-500/10 rounded-xl">
                    <i data-lucide="trending-up" class="text-blue-500 w-6 h-6"></i>
                </div>
                <div>
                    <p class="text-slate-500 text-[10px] font-bold uppercase tracking-wider">Live Hits / Day</p>
                    <p class="text-xl font-bold" id="stat-hits-day">0</p>
                </div>
             </div>
             <div class="bg-slate-900/50 border border-slate-800 p-4 rounded-2xl flex items-center gap-4">
                <div class="p-3 bg-emerald-500/10 rounded-xl">
                    <i data-lucide="clock" class="text-emerald-500 w-6 h-6"></i>
                </div>
                <div>
                    <p class="text-slate-500 text-[10px] font-bold uppercase tracking-wider">Uptime</p>
                    <p class="text-xl font-bold" id="stat-uptime">Nominal</p>
                </div>
             </div>
             <div class="bg-slate-900/50 border border-slate-800 p-4 rounded-2xl flex items-center gap-4">
                <div class="p-3 bg-purple-500/10 rounded-xl">
                    <i data-lucide="server" class="text-purple-500 w-6 h-6"></i>
                </div>
                <div>
                    <p class="text-slate-500 text-[10px] font-bold uppercase tracking-wider">Storage</p>
                    <p class="text-xl font-bold">Redis Cloud</p>
                </div>
             </div>
        </div>

        <!-- Refresh Schedule Row -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
             <div class="bg-slate-900/30 border border-slate-800/50 p-3 rounded-xl flex items-center justify-between">
                <div class="flex items-center gap-2">
                    <i data-lucide="list" class="text-blue-500 w-4 h-4"></i>
                    <span class="text-[10px] font-bold uppercase tracking-wider text-slate-500">Narrative Refresh</span>
                </div>
                <span id="timer-narratives" class="text-sm font-mono font-bold text-blue-400">00:00</span>
             </div>
             <div class="bg-slate-900/30 border border-slate-800/50 p-3 rounded-xl flex items-center justify-between">
                <div class="flex items-center gap-2">
                    <i data-lucide="rocket" class="text-emerald-500 w-4 h-4"></i>
                    <span class="text-[10px] font-bold uppercase tracking-wider text-slate-500">Launch Refresh</span>
                </div>
                <span id="timer-launches" class="text-sm font-mono font-bold text-emerald-400">00:00</span>
             </div>
             <div class="bg-slate-900/30 border border-slate-800/50 p-3 rounded-xl flex items-center justify-between">
                <div class="flex items-center gap-2">
                    <i data-lucide="cloud-sun" class="text-blue-400 w-4 h-4"></i>
                    <span class="text-[10px] font-bold uppercase tracking-wider text-slate-500">Weather Refresh</span>
                </div>
                <span id="timer-weather" class="text-sm font-mono font-bold text-blue-400">00:00</span>
             </div>
        </div>

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

        <!-- Tabs -->
        <div class="flex gap-8 mb-6 border-b border-slate-800">
            <button onclick="showTab('narratives')" id="tab-narratives" class="pb-2 font-semibold transition-colors tab-active">Narratives</button>
            <button onclick="showTab('launches')" id="tab-launches" class="pb-2 font-semibold text-slate-400 hover:text-slate-200 transition-colors">Launches</button>
            <button onclick="showTab('weather')" id="tab-weather" class="pb-2 font-semibold text-slate-400 hover:text-slate-200 transition-colors">Weather</button>
        </div>

        <!-- Narratives Section -->
        <div id="content-narratives" class="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
            <div class="px-6 py-4 border-b border-slate-800 bg-slate-800/30 flex items-center justify-between">
                <h2 class="text-lg font-semibold flex items-center gap-2">
                    <i data-lucide="list" class="w-5 h-5 text-blue-500"></i>
                    Current Narratives
                </h2>
                <div class="flex items-center gap-4">
                    <span class="text-xs text-slate-500 uppercase tracking-widest font-bold" id="last-updated">Updating...</span>
                    <button onclick="refreshTab('narratives')" class="p-1.5 hover:bg-slate-700/50 rounded-lg transition-colors text-slate-400 hover:text-blue-400" title="Force Refresh Narratives">
                        <i data-lucide="refresh-cw" class="w-4 h-4" id="refresh-icon-narratives"></i>
                    </button>
                </div>
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

        <!-- Launches Section -->
        <div id="content-launches" class="hidden space-y-8">
            <div class="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
                <div class="px-6 py-4 border-b border-slate-800 bg-slate-800/30 flex items-center justify-between">
                    <h2 class="text-lg font-semibold flex items-center gap-2">
                        <i data-lucide="rocket" class="w-5 h-5 text-emerald-500"></i>
                        Upcoming Launches
                    </h2>
                    <button onclick="refreshTab('launches')" class="p-1.5 hover:bg-slate-700/50 rounded-lg transition-colors text-slate-400 hover:text-emerald-400" title="Force Refresh Launches">
                        <i data-lucide="refresh-cw" class="w-4 h-4" id="refresh-icon-launches"></i>
                    </button>
                </div>
                <div class="p-6" id="upcoming-launches-list">
                    <div class="animate-pulse space-y-4">
                        <div class="h-12 bg-slate-800 rounded w-full"></div>
                        <div class="h-12 bg-slate-800 rounded w-full"></div>
                    </div>
                </div>
            </div>
            
            <div class="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
                <div class="px-6 py-4 border-b border-slate-800 bg-slate-800/30">
                    <h2 class="text-lg font-semibold flex items-center gap-2">
                        <i data-lucide="history" class="w-5 h-5 text-blue-500"></i>
                        Previous Launches
                    </h2>
                </div>
                <div class="p-6" id="previous-launches-list">
                    <div class="animate-pulse space-y-4">
                        <div class="h-12 bg-slate-800 rounded w-full"></div>
                        <div class="h-12 bg-slate-800 rounded w-full"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Weather Section -->
        <div id="content-weather" class="hidden space-y-6">
            <div class="flex items-center justify-between">
                <h2 class="text-2xl font-bold tracking-tight">Weather Conditions</h2>
                <button onclick="refreshTab('weather')" class="flex items-center gap-2 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-800 transition-colors rounded-lg font-semibold text-xs text-slate-300">
                    <i data-lucide="refresh-cw" class="w-3.5 h-3.5" id="refresh-icon-weather"></i>
                    Refresh Weather
                </button>
            </div>
            <div id="weather-grid" class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- Weather cards will be loaded here -->
            </div>
        </div>
    </main>

    <script>
        lucide.createIcons();

        // Chart instances
        const charts = {};
        let currentRange = '1h';

        function initChart(id, color) {
            const ctx = document.getElementById(id).getContext('2d');
            return new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        data: [],
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

        // Timer management
        const refreshIntervals = {
            metrics: 30,
            narratives: 900, // 15m
            launches: 600,   // 10m
            weather: 300     // 5m
        };

        // Initialize next refresh targets
        const nextRefresh = {
            metrics: Date.now() + refreshIntervals.metrics * 1000,
            narratives: Date.now() + refreshIntervals.narratives * 1000,
            launches: Date.now() + refreshIntervals.launches * 1000,
            weather: Date.now() + refreshIntervals.weather * 1000
        };

        function updateTimers() {
            const now = Date.now();
            
            const formatTime = (ms) => {
                const totalSeconds = Math.max(0, Math.floor(ms / 1000));
                const minutes = Math.floor(totalSeconds / 60);
                const seconds = totalSeconds % 60;
                return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            };

            if (document.getElementById('timer-narratives')) {
                document.getElementById('timer-narratives').textContent = formatTime(nextRefresh.narratives - now);
            }
            if (document.getElementById('timer-launches')) {
                document.getElementById('timer-launches').textContent = formatTime(nextRefresh.launches - now);
            }
            if (document.getElementById('timer-weather')) {
                document.getElementById('timer-weather').textContent = formatTime(nextRefresh.weather - now);
            }

            // Check if any timer expired
            if (now >= nextRefresh.metrics) {
                fetchMetrics();
                nextRefresh.metrics = now + refreshIntervals.metrics * 1000;
            }
            if (now >= nextRefresh.narratives) {
                fetchNarratives();
                nextRefresh.narratives = now + refreshIntervals.narratives * 1000;
            }
            if (now >= nextRefresh.launches) {
                fetchLaunches();
                nextRefresh.launches = now + refreshIntervals.launches * 1000;
            }
            if (now >= nextRefresh.weather) {
                fetchWeatherAll();
                nextRefresh.weather = now + refreshIntervals.weather * 1000;
            }
        }

        function changeRange(range) {
            currentRange = range;
            ['1h', '24h', '7d'].forEach(r => {
                const btn = document.getElementById(`range-${r}`);
                if (r === range) {
                    btn.classList.add('range-active');
                    btn.classList.remove('text-slate-400', 'hover:text-slate-200');
                    btn.classList.add('text-white');
                } else {
                    btn.classList.remove('range-active', 'text-white');
                    btn.classList.add('text-slate-400', 'hover:text-slate-200');
                }
            });
            fetchMetrics();
            // Reset metrics timer
            nextRefresh.metrics = Date.now() + refreshIntervals.metrics * 1000;
        }

        function showTab(tab) {
            ['narratives', 'launches', 'weather'].forEach(t => {
                document.getElementById(`tab-${t}`).classList.remove('tab-active');
                document.getElementById(`tab-${t}`).classList.add('text-slate-400');
                document.getElementById(`content-${t}`).classList.add('hidden');
            });
            
            document.getElementById(`tab-${tab}`).classList.add('tab-active');
            document.getElementById(`tab-${tab}`).classList.remove('text-slate-400');
            document.getElementById(`content-${tab}`).classList.remove('hidden');
            
            // Note: Automatic fetches removed from here to make it strictly timer-based
            // and prevent unnecessary API calls/loading states when switching tabs.
        }

        async function fetchMetrics() {
            try {
                const response = await fetch(`/metrics?range=${currentRange}`);
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

                document.getElementById('stat-hits-day').textContent = (data.hits_per_day || 0).toLocaleString();

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
                        
                        chart.data.labels = history.map(h => '');
                        chart.data.datasets[0].data = values;
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

        function formatValue(val) {
            if (val === null || val === undefined) return '<span class="text-slate-600 italic text-[10px]">null</span>';
            if (val === '') return '<span class="text-slate-600 italic text-[10px]">empty</span>';
            if (typeof val === 'boolean') return `<span class="${val ? 'text-emerald-400' : 'text-red-400'} font-bold text-[10px]">${val}</span>`;
            if (typeof val === 'string' && (val.startsWith('http://') || val.startsWith('https://'))) {
                return `<a href="${val}" target="_blank" class="text-blue-400 hover:underline truncate block max-w-full text-[10px]">${val}</a>`;
            }
            return `<span class="text-slate-300 break-words text-[10px]">${val}</span>`;
        }

        function renderSubData(val, depth = 0) {
            if (val === null || val === undefined || val === '') return formatValue(val);
            if (depth > 3) {
                const str = JSON.stringify(val);
                return `<span class="text-[9px] text-slate-500 italic font-mono" title='${str.replace(/'/g, "&apos;")}'>${str.length > 40 ? str.substring(0, 40) + '...' : str}</span>`;
            }
            
            if (Array.isArray(val)) {
                if (val.length === 0) return formatValue('');
                return val.map(v => `
                    <div class="pl-2 border-l border-slate-800/30 my-1">
                        ${(typeof v === 'object' && v !== null) ? renderSubData(v, depth + 1) : formatValue(v)}
                    </div>
                `).join('');
            }
            
            if (typeof val === 'object') {
                return Object.entries(val).map(([k, v]) => {
                    return `
                        <div class="flex flex-col mt-1">
                            <p class="text-[9px] text-slate-600 uppercase font-medium tracking-tight">${k.replace(/_/g, ' ')}</p>
                            ${(typeof v === 'object' && v !== null) ? renderSubData(v, depth + 1) : formatValue(v)}
                        </div>
                    `;
                }).join('');
            }
            
            return formatValue(val);
        }

        function renderDataCards(data) {
            if (!data || typeof data !== 'object') return '';
            
            let html = '<div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">';
            
            // Prioritize these fields to appear first
            const priority = ['status', 'net', 'window_start', 'window_end', 'probability', 'holdreason', 'failreason', 'rocket', 'mission', 'pad'];
            
            const keys = Object.keys(data).sort((a, b) => {
                const aPrio = priority.indexOf(a);
                const bPrio = priority.indexOf(b);
                if (aPrio !== -1 && bPrio !== -1) return aPrio - bPrio;
                if (aPrio !== -1) return -1;
                if (bPrio !== -1) return 1;
                
                const aIsObj = typeof data[a] === 'object' && data[a] !== null;
                const bIsObj = typeof data[b] === 'object' && data[b] !== null;
                return aIsObj - bIsObj;
            });

            for (const key of keys) {
                const value = data[key];

                html += `
                    <div class="bg-slate-950/40 p-3 rounded-xl border border-slate-800/50 space-y-1.5 flex flex-col">
                        <p class="text-[10px] text-slate-500 uppercase font-bold tracking-wider border-b border-slate-800/50 pb-1">${key.replace(/_/g, ' ')}</p>
                        <div class="flex-1">
                            ${(typeof value === 'object' && value !== null) ? renderSubData(value) : formatValue(value)}
                        </div>
                    </div>
                `;
            }
            
            html += '</div>';
            return html;
        }

        async function fetchNarratives(force = false) {
            try {
                const url = force ? '/recent_launches_narratives?force=true' : '/recent_launches_narratives';
                const response = await fetch(url);
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

        async function fetchLaunches(force = false) {
            const upList = document.getElementById('upcoming-launches-list');
            const prevList = document.getElementById('previous-launches-list');
            
            try {
                const url = force ? '/launches?force=true' : '/launches';
                const response = await fetch(url);
                const data = await response.json();
                
                const renderList = (el, launches) => {
                    el.innerHTML = '';
                    if (!launches || launches.length === 0) {
                        el.innerHTML = '<p class="text-slate-500 text-center py-4">No launches found.</p>';
                        return;
                    }
                    launches.forEach(l => {
                        const div = document.createElement('div');
                        div.className = 'launch-card border-b border-slate-800 last:border-0 hover:bg-slate-800/10 transition-all';
                        
                        const id = 'details-' + l.id;
                        const rawId = 'raw-' + l.id;
                        
                        div.innerHTML = `
                            <div class="p-4 cursor-pointer" onclick="document.getElementById('${id}').classList.toggle('hidden'); lucide.createIcons();">
                                <div class="flex items-center justify-between">
                                    <div class="flex flex-col">
                                        <span class="font-bold text-slate-100">${l.mission}</span>
                                        <span class="text-xs text-slate-400">${l.rocket} • ${l.pad}</span>
                                    </div>
                                    <div class="flex flex-col items-end text-right">
                                        <div class="flex items-center gap-2 mb-1">
                                            <span class="text-sm font-mono text-blue-400">${l.date} ${l.time}</span>
                                            <i data-lucide="chevron-down" class="w-4 h-4 text-slate-500"></i>
                                        </div>
                                        <span class="text-[10px] px-2 py-0.5 rounded bg-slate-800 uppercase tracking-tighter ${l.status === 'Success' ? 'text-emerald-400' : 'text-yellow-400'}">${l.status}</span>
                                    </div>
                                </div>
                            </div>
                            
                            <div id="${id}" class="hidden px-4 pb-6 space-y-4 border-t border-slate-800/50 pt-4 bg-slate-900/30">
                                ${l.image ? `<img src="${l.image}" class="w-full h-48 object-cover rounded-xl border border-slate-700 shadow-lg" onerror="this.style.display='none'">` : ''}
                                
                                ${l.description ? `<p class="text-sm text-slate-300 leading-relaxed bg-slate-950/50 p-4 rounded-xl border border-slate-800">${l.description}</p>` : ''}
                                
                                <div class="space-y-4">
                                    <h3 class="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-2 border-b border-slate-800 pb-1">Detailed Mission Data</h3>
                                    ${renderDataCards(l.all_data)}
                                </div>

                                <div class="flex flex-wrap gap-3">
                                    ${l.video_url ? `
                                    <a href="${l.video_url}" target="_blank" class="flex items-center gap-2 px-3 py-1.5 bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded-lg transition-colors border border-red-500/20 text-xs font-semibold">
                                        <i data-lucide="play-circle" class="w-4 h-4"></i> Webcast
                                    </a>` : ''}
                                    ${l.x_video_url ? `
                                    <a href="${l.x_video_url}" target="_blank" class="flex items-center gap-2 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg transition-colors border border-slate-700 text-xs font-semibold">
                                        <i data-lucide="twitter" class="w-4 h-4"></i> X Update
                                    </a>` : ''}
                                    <button onclick="document.getElementById('${rawId}').classList.toggle('hidden')" class="flex items-center gap-2 px-3 py-1.5 bg-blue-500/10 hover:bg-blue-500/20 text-blue-400 rounded-lg transition-colors border border-blue-500/20 text-xs font-semibold ml-auto">
                                        <i data-lucide="code" class="w-4 h-4"></i> View Raw Data
                                    </button>
                                </div>

                                <div id="${rawId}" class="hidden mt-4">
                                    <div class="flex items-center justify-between mb-2">
                                        <span class="text-[10px] text-slate-500 uppercase font-bold">API Response Object</span>
                                        <button onclick="navigator.clipboard.writeText(this.parentElement.nextElementSibling.textContent)" class="text-[10px] text-blue-400 hover:text-blue-300 uppercase font-bold">Copy JSON</button>
                                    </div>
                                    <pre class="bg-slate-950 p-4 rounded-xl border border-slate-800 text-[10px] text-slate-400 overflow-x-auto font-mono max-h-60">${JSON.stringify(l.all_data, null, 2)}</pre>
                                </div>
                            </div>
                        `;
                        el.appendChild(div);
                    });
                    lucide.createIcons();
                };
                
                renderList(upList, data.upcoming);
                renderList(prevList, data.previous);
            } catch (error) {
                upList.innerHTML = prevList.innerHTML = '<p class="text-red-400 text-center py-4">Error loading launches.</p>';
            }
        }

        async function fetchWeatherAll(force = false) {
            const locations = ['Starbase', 'Vandy', 'Cape', 'Hawthorne'];
            const container = document.getElementById('weather-grid');
            container.innerHTML = '';
            
            for (const loc of locations) {
                const card = document.createElement('div');
                card.className = 'bg-slate-900 border border-slate-800 p-6 rounded-2xl flex flex-col gap-4';
                card.innerHTML = `<div class="animate-pulse h-32 bg-slate-800 rounded"></div>`;
                container.appendChild(card);
                
                fetchWeather(loc, card, force);
            }
        }

        async function fetchWeather(loc, card, force = false) {
            try {
                const url = force ? `/weather/${loc}?force=true` : `/weather/${loc}`;
                const response = await fetch(url);
                const data = await response.json();
                
                card.innerHTML = `
                    <div class="flex items-center justify-between mb-2">
                        <h3 class="text-xl font-bold">${loc}</h3>
                        <i data-lucide="cloud-sun" class="text-blue-400 w-6 h-6"></i>
                    </div>
                    <div class="grid grid-cols-2 gap-y-4 gap-x-6">
                        <div>
                            <p class="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1">Temperature</p>
                            <p class="text-2xl font-semibold">${Math.round(data.temperature_f)}°F <span class="text-sm text-slate-400 font-normal">/ ${Math.round(data.temperature_c)}°C</span></p>
                        </div>
                        <div>
                            <p class="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1">Wind Speed</p>
                            <p class="text-2xl font-semibold">${Math.round(data.wind_speed_kts)} <span class="text-sm text-slate-400 font-normal">kts</span></p>
                        </div>
                        <div>
                            <p class="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1">Cloud Cover</p>
                            <p class="text-2xl font-semibold">${data.cloud_cover}%</p>
                        </div>
                        <div>
                            <p class="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1">Wind Direction</p>
                            <p class="text-2xl font-semibold">${data.wind_direction}°</p>
                        </div>
                    </div>
                    <div class="mt-2 pt-3 border-t border-slate-800">
                        <p class="text-[9px] font-mono text-slate-600 truncate uppercase" title="${data.raw || ''}">${data.raw || 'No raw METAR data'}</p>
                    </div>
                `;
                lucide.createIcons();
            } catch (error) {
                card.innerHTML = `<p class="text-red-400">Error loading weather for ${loc}</p>`;
            }
        }

        async function refreshTab(tab) {
            const icon = document.getElementById(`refresh-icon-${tab}`);
            if (icon) icon.classList.add('animate-spin');
            
            try {
                if (tab === 'narratives') await fetchNarratives(true);
                if (tab === 'launches') await fetchLaunches(true);
                if (tab === 'weather') await fetchWeatherAll(true);
                
                // Reset timer for this specific tab after manual refresh
                nextRefresh[tab] = Date.now() + refreshIntervals[tab] * 1000;
                
                // Also update metrics as a force refresh counts as API calls
                fetchMetrics();
                nextRefresh.metrics = Date.now() + refreshIntervals.metrics * 1000;
            } catch (error) {
                console.error(`Error refreshing ${tab}:`, error);
            } finally {
                if (icon) {
                    setTimeout(() => icon.classList.remove('animate-spin'), 500);
                }
            }
        }

        // Initial load - Fetch all data once to bootstrap the UI
        fetchMetrics();
        fetchNarratives();
        fetchLaunches();
        fetchWeatherAll();

        // Start the master timer loop (updates UI countdowns and triggers refreshes)
        setInterval(updateTimers, 1000);
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