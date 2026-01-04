# SpaceX Launch Summary API

A witty, high-performance API that provides "Cities Skylines" style narratives for recent SpaceX launches. It uses the xAI Grok API to transform technical launch data into dry, humorous descriptions suitable for scrolling tickers, dashboards, and monitoring tools.

## Live API Access

The API is live and can be accessed at:
**[https://launch-narrative-api-dafccc521fb8.herokuapp.com/](https://launch-narrative-api-dafccc521fb8.herokuapp.com/)**

---

## API Endpoints

### 1. Get Launch Narratives
Returns a chronological list (newest first) of witty descriptions for recent SpaceX launches.

*   **Endpoint:** `GET /recent_launches_narratives`
*   **Parameters:** `force=true` (optional, bypasses cache)
*   **Response Format:** JSON
*   **Fields:**
    *   `descriptions`: (array) List of witty strings in "month/day HHMM: description" format.
*   **Caching:** Results are cached for 1 hour. The API uses incremental generation to append new launches without changing existing witty descriptions.

### 2. Get Detailed Launch Data
Returns structured data for both upcoming and previous SpaceX launches. Uses `mode=detailed` (LL API v2.3.0) to ensure every field (mission descriptions, images, technical metrics) is included.

*   **Endpoint:** `GET /launches`
*   **Parameters:** `force=true` (optional)
*   **Response Format:** JSON
*   **Fields (Launch Object):**
    *   `id`: (string) Unique identifier for the launch.
    *   `mission`: (string) Name of the mission.
    *   `date`: (string) NET date in YYYY-MM-DD.
    *   `time`: (string) NET time in HH:MM:SS.
    *   `net`: (string) Full ISO8601 timestamp.
    *   `status`: (string) Current status (e.g., Success, TBD).
    *   `rocket`: (string) Rocket configuration name.
    *   `orbit`: (string) Target orbit name.
    *   `pad`: (string) Launch pad name.
    *   `video_url`: (string) Primary webcast URL.
    *   `x_video_url`: (string) X (Twitter) update/webcast URL.
    *   `landing_type`: (string) Type of landing (e.g., ASDS, RTLS).
    *   `landing_location`: (string) Specific landing site name.
    *   `description`: (string) Full mission description.
    *   `image`: (string) Primary mission image URL.
    *   `window_start`: (string) ISO8601 timestamp for window open.
    *   `window_end`: (string) ISO8601 timestamp for window close.
    *   `probability`: (int) Launch probability percentage.
    *   `holdreason`: (string) Reason for a hold, if any.
    *   `failreason`: (string) Reason for a failure, if any.
    *   `all_data`: (object) Complete, unparsed raw response from the source API.
*   **Caching:** 10 minutes.

### 3. Get Weather Data
Returns parsed METAR weather data for SpaceX launch and development sites (Starbase, Vandy, Cape, Hawthorne).

*   **Endpoint:** `GET /weather/{location}` or `GET /weather_all`
*   **Parameters:** `force=true` (optional)
*   **Response Format:** JSON
*   **Fields:**
    *   For `/weather/{location}`: A single weather object (see below).
    *   For `/weather_all`: A dictionary mapping location names to weather objects.
    *   **Weather Object Fields:**
        *   `temperature_c`: (int) Temperature in Celsius.
        *   `temperature_f`: (float) Temperature in Fahrenheit.
        *   `wind_speed_ms`: (float) Wind speed in meters per second.
        *   `wind_speed_kts`: (int) Wind speed in knots.
        *   `wind_direction`: (int) Wind direction in degrees.
        *   `cloud_cover`: (int) Percentage of cloud cover estimation.
        *   `raw`: (string) Raw METAR string from the weather service.
*   **Caching:** 15 minutes.

### 4. Get API Metrics
Provides real-time and historical performance data, including request counts, cache efficiency, and interactive history.

*   **Endpoint:** `GET /metrics`
*   **Parameters:** `range=1h` (default), `24h`, or `7d`
*   **Response Format:** JSON
*   **Fields:**
    *   `current`: (object) Current counters for `total_requests`, `cache_hits`, `cache_misses`, and `api_calls`.
    *   `history`: (array) List of snapshots containing `timestamp` and `data` (current metrics at that time).
    *   `hits_per_day`: (float) Rolling average of requests projected to a 24-hour period.

### 5. Force Cache Refresh
Manually triggers the API to poll for new launches and generate new narratives using Grok.

*   **Endpoint:** `POST /refresh`
*   **Response Format:** JSON
*   **Fields:**
    *   `status`: (string) Confirmation message.
    *   `count`: (int) Total number of narratives in the cache.
    *   `timestamp`: (string) ISO8601 update time.
*   **Behavior:** Incremental. It only generates narratives for launches not already in the cache.

### 6. Utility Endpoints
*   **Get Single Launch Details:** `GET /launch_details/{launch_id}` (Returns full raw API response for a specific launch).
*   **Get External Narratives:** `GET /external_narratives` (Returns `{"descriptions": [...]}` proxying secondary source).

---

## Interactive Dashboard

Access the root URL (`/`) in any web browser to view the **API Status Dashboard**.
*   **Tabbed Interface:** Switch between **Narratives**, **Launches**, and **Weather** views.
*   **Real-time Monitoring:** Interactive sparkline charts for traffic and efficiency with selectable time ranges (1h, 24h, 7d).
*   **Live Metrics:** View live performance indicators like "Live Hits / Day" and system uptime.
*   **Detailed Launch Cards:** Click any launch to see comprehensive technical data, images, mission descriptions, and raw API responses.
*   **Refresh Schedule:** Visual countdown timers show exactly when each data category is scheduled to refresh.
*   **Manual Control:** Per-tab refresh buttons to bypass cache and fetch fresh data instantly.

---

## Usage Examples

### cURL
```bash
curl https://launch-narrative-api-dafccc521fb8.herokuapp.com/recent_launches_narratives
```

### JavaScript (Fetch API)
```javascript
fetch('https://launch-narrative-api-dafccc521fb8.herokuapp.com/recent_launches_narratives')
  .then(response => response.json())
  .then(data => console.log(data.descriptions));
```

### Python (Requests)
```python
import requests

url = "https://launch-narrative-api-dafccc521fb8.herokuapp.com/recent_launches_narratives"
response = requests.get(url)
launches = response.json().get("descriptions", [])

for launch in launches:
    print(launch)
```

---

## Data Refresh Policy
The API utilizes a **Timer-Based Refresh Strategy** to ensure stability and speed:
1.  **Background Refresh:** A dedicated worker thread in the backend automatically refreshes the cache for Narratives (15m), Launches (10m), and Weather (5m).
2.  **Manual Force:** Users can trigger an immediate refresh via the dashboard buttons or by appending `?force=true` to API requests.
3.  **Incremental History:** Previous launch data is never fully replaced; new launches are appended to the existing historical cache to preserve a continuous record.

---

## License
MIT
