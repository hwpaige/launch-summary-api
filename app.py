import os
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import requests
import ast
import redis
import re
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env if present

app = FastAPI()

# Initialize Redis (Heroku will set REDIS_URL env var after adding the add-on)
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
r = redis.from_url(redis_url)
CACHE_KEY = "launch_narratives"
METRICS_KEY = "launch_metrics"
CACHE_TTL = 3600  # 1 hour TTL in seconds; adjust as needed

def increment_metric(field):
    """Increment a metric in Redis."""
    try:
        r.hincrby(METRICS_KEY, field, 1)
    except Exception as e:
        print(f"Metrics error: {e}")

def get_metrics():
    """Retrieve metrics from Redis."""
    try:
        metrics = r.hgetall(METRICS_KEY)
        return {k.decode('utf-8'): int(v.decode('utf-8')) for k, v in metrics.items()}
    except Exception as e:
        print(f"Metrics error: {e}")
        return {}

def generate_narratives():
    """Fetch launches and generate narratives using Grok."""
    increment_metric("api_calls")
    current_time = datetime.now(timezone.utc)
    three_months_ago = current_time - timedelta(days=90)
    # Note: Using LL 2.0.0 as requested in the snippet
    url = (
        f"https://ll.thespacedevs.com/2.0.0/launch/previous/"
        f"?lsp__name=SpaceX"
        f"&net__gte={three_months_ago.strftime('%Y-%m-%d')}"
        f"&net__lte={current_time.strftime('%Y-%m-%d')}"
        f"&limit=50"
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
        return []
    
    launch_list = "\n".join([
        f"{l['date_time']}: {l['mission']} from {l['pad']}, {l['rocket']} to {l['orbit']}, status {l['status']}"
        for l in launches
    ])
    
    prompt = f"""Generate a list of short news like descriptions for these recent SpaceX launches:
{launch_list}

In the style of Cities Skylines notifications: kind of witty and dry. Factual, complete, somewhat technical - think Kerbal Space Program.

Examples of the desired style:
- Falcon 9 hoists MTG-S1/Sentinel-4A to geosync from LC-39A; Ariane's loss is our nominal gain, booster recovered without drama.
- 500th Falcon 9 ignites with 27 Starlinks from SLC-40; B1067 clocks 29th flight, orbit insertion as predictable as gravity.
- Another 28 Starlinks flung to LEO via Falcon 9 at SLC-40; deployment flawless, booster sticks the landing like it's bored.
- 24 KuiperSats for Amazon lofted by Falcon 9 at SLC-40; ironic assist to rivals, payloads separate cleanly in orbit.
- Starship Flight 10 ignites from Starbase; hot-staging clean, ship splashes precisely in Indian Ocean, Super Heavy boosts back nominally.

Format each as: month/day HHMM: description

Output as a Python list assignment: launch_descriptions = [...]"""
    
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "grok-beta",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        response = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload)
        if response.status_code != 200:
            raise ValueError(f"Grok API call failed: {response.status_code} - {response.text}")
        
        data = response.json()
        generated_text = data['choices'][0]['message']['content']
    except Exception as e:
        raise ValueError(f"Grok API call failed: {str(e)}")
    
    try:
        # Try to find a python code block first
        code_block_match = re.search(r"```(?:python)?\s*(launch_descriptions\s*=\s*\[.*?\])```", generated_text, re.DOTALL)
        if code_block_match:
            code_content = code_block_match.group(1)
            # Extract just the list part from the assignment
            list_match = re.search(r"\[.*\]", code_content, re.DOTALL)
            if list_match:
                list_str = list_match.group(0)
            else:
                list_str = code_content
        else:
            # Fallback to finding brackets
            start_idx = generated_text.find('[')
            end_idx = generated_text.rfind(']') + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No list found in response")
            list_str = generated_text[start_idx:end_idx]
        
        descriptions = ast.literal_eval(list_str)
        
        if not isinstance(descriptions, list) or not all(isinstance(d, str) for d in descriptions):
            raise ValueError("Parsed content is not a list of strings")
    except Exception as e:
        raise ValueError(f"Failed to parse Grok response: {str(e)}")
    
    return descriptions

@app.get("/recent_launches_narratives")
def get_narratives():
    """Serve from cache if available; generate otherwise."""
    increment_metric("total_requests")
    try:
        cached = r.get(CACHE_KEY)
        if cached:
            increment_metric("cache_hits")
            return {"descriptions": ast.literal_eval(cached.decode('utf-8'))}
    except Exception as e:
        # Fallback if redis fails
        print(f"Redis error: {e}")
    
    increment_metric("cache_misses")
    descriptions = generate_narratives()
    try:
        r.setex(CACHE_KEY, CACHE_TTL, str(descriptions))
    except Exception as e:
        print(f"Redis error: {e}")
        
    return {"descriptions": descriptions}

@app.get("/metrics")
def get_app_metrics():
    """Endpoint to fetch application metrics."""
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
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; }
        .ticker-item { border-left: 4px solid #3b82f6; }
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
            <div class="bg-slate-900 border border-slate-800 p-6 rounded-2xl">
                <div class="flex items-center justify-between mb-4">
                    <span class="text-slate-400 text-sm font-medium">Total Requests</span>
                    <i data-lucide="activity" class="text-blue-400 w-5 h-5"></i>
                </div>
                <div class="text-3xl font-bold" id="metric-total-requests">0</div>
            </div>
            <div class="bg-slate-900 border border-slate-800 p-6 rounded-2xl">
                <div class="flex items-center justify-between mb-4">
                    <span class="text-slate-400 text-sm font-medium">Cache Hits</span>
                    <i data-lucide="database" class="text-emerald-400 w-5 h-5"></i>
                </div>
                <div class="text-3xl font-bold text-emerald-400" id="metric-cache-hits">0</div>
            </div>
            <div class="bg-slate-900 border border-slate-800 p-6 rounded-2xl">
                <div class="flex items-center justify-between mb-4">
                    <span class="text-slate-400 text-sm font-medium">Grok API Calls</span>
                    <i data-lucide="brain-circuit" class="text-purple-400 w-5 h-5"></i>
                </div>
                <div class="text-3xl font-bold text-purple-400" id="metric-api-calls">0</div>
            </div>
            <div class="bg-slate-900 border border-slate-800 p-6 rounded-2xl">
                <div class="flex items-center justify-between mb-4">
                    <span class="text-slate-400 text-sm font-medium">Cache Efficiency</span>
                    <i data-lucide="zap" class="text-yellow-400 w-5 h-5"></i>
                </div>
                <div class="text-3xl font-bold" id="metric-efficiency">0%</div>
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

        async function fetchMetrics() {
            try {
                const response = await fetch('/metrics');
                const data = await response.json();
                
                const total = data.total_requests || 0;
                const hits = data.cache_hits || 0;
                const apiCalls = data.api_calls || 0;
                
                document.getElementById('metric-total-requests').textContent = total.toLocaleString();
                document.getElementById('metric-cache-hits').textContent = hits.toLocaleString();
                document.getElementById('metric-api-calls').textContent = apiCalls.toLocaleString();
                
                const efficiency = total > 0 ? Math.round((hits / total) * 100) : 0;
                document.getElementById('metric-efficiency').textContent = efficiency + '%';
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
    descriptions = generate_narratives()
    r.setex(CACHE_KEY, CACHE_TTL, str(descriptions))
    return {"status": "Cache refreshed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))