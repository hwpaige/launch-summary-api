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
    *   `last_updated`: (string) ISO8601 timestamp of when the narratives were last generated.
*   **Sample Response:**
```json
{
  "descriptions": [
    "01/04 1200: Falcon 9 launches Starlink 12-1 from SLC-40; booster B1080 nails the landing on A Shortfall of Gravitas, mission nominal.",
    "12/30 1530: Falcon 9 lofted O3b mPOWER 7 & 8 from SLC-40; B1078 completes 12th flight, orbital insertion confirmed."
  ],
  "last_updated": "2026-01-04T15:00:00.000000+00:00"
}
```
*   **Caching:** Results are cached for 1 hour. The API uses incremental generation to append new launches without changing existing witty descriptions.

### 2. Get Detailed Launch Data
Returns exhaustive structured data for both upcoming and previous SpaceX launches. This endpoint includes the full, raw response from the Launch Library v2.3.0 API in the `all_data` field for every launch.

**NOTE:** This endpoint can return payloads exceeding 85MB. For performance-critical applications or web dashboards, use `GET /launches_slim` instead.

*   **Endpoint:** `GET /launches`
*   **Parameters:** `force=true` (optional)
*   **Response Format:** JSON
*   **Fields:**
    *   `upcoming`: (array) List of upcoming launch objects.
    *   `previous`: (array) List of historical launch objects.
    *   `last_updated`: (string) ISO8601 timestamp of when the launch data was last fetched.
*   **Fields (Launch Object - Top Level):**
    *   `id`: (string) Unique UUID for the launch.
    *   `name`: (string) Full name of the mission.
    *   `all_data`: (object) Complete recursive map of ALL fields returned by the source API.
    *   *(See /launches_slim for other convenience fields)*
*   **Caching:** 10 minutes.

### 3. Get Optimized Launch Data (Recommended for Dashboards)
A performance-optimized version of the launches endpoint that strips the heavy `all_data` field. This reduces the transfer size from ~85MB to less than 1MB.

*   **Endpoint:** `GET /launches_slim`
*   **Parameters:** `force=true` (optional)
*   **Response Format:** JSON
*   **Fields:** Same as `/launches`, but each launch object excludes `all_data`.
*   **Caching:** 10 minutes.

### 4. Get Raw Launch Details
Returns the full, unpruned raw API response for a specific launch from the cache. Use this to get deep details for a single launch on-demand.

*   **Endpoint:** `GET /launch_raw/{launch_id}`
*   **Response Format:** JSON
*   **Sample Response:** (Large nested JSON object)

### 5. Get Weather Data
Returns parsed METAR weather data for SpaceX launch and development sites (Starbase, Vandy, Cape, Hawthorne).

*   **Endpoint:** `GET /weather/{location}` or `GET /weather_all`
*   **Parameters:** `force=true` (optional)
*   **Response Format:** JSON
*   **Fields:**
    *   For `/weather/{location}`: A single weather object (see below).
    *   For `/weather_all`: A dictionary mapping location names to weather objects, plus a global `last_updated` field.
    *   **Weather Object Fields:**
        *   `temperature_c`: (int) Temperature in Celsius.
        *   `temperature_f`: (float) Temperature in Fahrenheit.
        *   `dewpoint_c`: (int) Dewpoint in Celsius.
        *   `dewpoint_f`: (float) Dewpoint in Fahrenheit.
        *   `humidity`: (int) Relative humidity percentage.
        *   `wind_speed_kts`: (int) Wind speed in knots.
        *   `wind_gust_kts`: (int) Wind gust speed in knots (0 if none).
        *   `wind_direction`: (int) Wind direction in degrees.
        *   `visibility_sm`: (float) Visibility in statute miles.
        *   `altimeter_inhg`: (float) Altimeter setting in inches of mercury.
        *   `cloud_cover`: (int) Percentage of cloud cover estimation.
        *   `flight_category`: (string) Estimated flight category (VFR, MVFR, IFR, LIFR).
        *   `raw`: (string) Raw METAR string from the weather service.
        *   `forecast`: (object) 7-day forecast data from Open-Meteo.
            *   `daily`: (object) Daily forecast including `time`, `temperature_2m_max`, `temperature_2m_min`, and `weathercode`.
            *   `hourly`: (object) Hourly temperature data including `time` and `temperature_2m`.
        *   `last_updated`: (string) ISO8601 timestamp of the weather fetch.
*   **Sample Response (`/weather_all`):**
```json
{
  "weather": {
    "Starbase": {
      "temperature_c": 18,
      "temperature_f": 64.4,
      "dewpoint_c": 14,
      "dewpoint_f": 57.2,
      "humidity": 77,
      "wind_speed_kts": 8,
      "wind_gust_kts": 0,
      "wind_direction": 160,
      "visibility_sm": 10.0,
      "altimeter_inhg": 30.12,
      "cloud_cover": 25,
      "flight_category": "VFR",
      "raw": "KBRO 041453Z 16008KT 10SM FEW025 18/14 A3012 RMK AO2 SLP198 T01830139",
      "forecast": {
        "daily": {
          "time": ["2026-01-04", "2026-01-05", "..."],
          "temperature_2m_max": [22.5, 23.1, "..."],
          "temperature_2m_min": [15.2, 14.8, "..."],
          "weathercode": [0, 1, "..."]
        },
        "hourly": {
          "time": ["2026-01-04T00:00", "2026-01-04T01:00", "..."],
          "temperature_2m": [18.5, 18.2, "..."]
        }
      },
      "last_updated": "2026-01-04T14:55:00Z"
    },
    "Vandy": { "temperature_c": 12, "last_updated": "2026-01-04T14:55:00Z" },
    "Cape": { "temperature_c": 22, "last_updated": "2026-01-04T14:55:00Z" },
    "Hawthorne": { "temperature_c": 19, "last_updated": "2026-01-04T14:55:00Z" }
  },
  "last_updated": "2026-01-04T14:55:00Z"
}
```
*   **Caching:** 5 minutes.

### 4. Get API Metrics
Provides real-time and historical performance data, including request counts, cache efficiency, and interactive history.

*   **Endpoint:** `GET /metrics`
*   **Parameters:** `range=1h` (default), `24h`, `7d`, or `30d`
*   **Response Format:** JSON
*   **Fields:**
    *   `current`: (object) Current counters for:
        *   `total_requests`: Total HTTP requests received.
        *   `cache_hits`: Number of requests served from cache.
        *   `cache_misses`: Number of requests that required a backend fetch.
        *   `api_calls`: Number of calls made to external APIs (Grok, Launch Library v2.3.0, and Aviation Weather).
    *   `history`: (array) List of snapshots containing `timestamp` and `data` (current metrics at that time).
    *   `hits_per_day`: (float) Rolling average of requests projected to a 24-hour period based on the selected range.
*   **Sample Response:**
```json
{
  "current": {
    "total_requests": 1520,
    "cache_hits": 1450,
    "cache_misses": 70,
    "api_calls": 125
  },
  "history": [
    {
      "timestamp": "2026-01-04T14:59:00Z",
      "data": {
        "total_requests": 1518,
        "cache_hits": 1448,
        "cache_misses": 70,
        "api_calls": 124
      }
    }
  ],
  "hits_per_day": 36480.0
}
```

### 5. Force Cache Refresh
Manually triggers the API to poll for new launches and generate new narratives using Grok.

*   **Endpoint:** `POST /refresh`
*   **Response Format:** JSON
*   **Fields:**
    *   `status`: (string) Confirmation message.
    *   `count`: (int) Total number of narratives currently in the cache.
    *   `timestamp`: (string) ISO8601 update time.
*   **Sample Response:**
```json
{
  "status": "Cache refreshed",
  "count": 42,
  "timestamp": "2026-01-04T15:00:00Z"
}
```
*   **Behavior:** Incremental. It only generates narratives for launches not already in the cache.

### 6. Utility Endpoints
*   **Get Single Launch Details:** `GET /launch_details/{launch_id}`
    *   Returns full raw API response for a specific launch from the LL API.
    *   **Sample Response:** (Large JSON object containing technical mission/rocket/pad details)
*   **Get External Narratives:** `GET /external_narratives`
    *   Returns witty descriptions from a secondary narrative source.
    *   **Fields:**
        *   `descriptions`: (array) List of narrative strings.
    *   **Sample Response:**
```json
{
  "descriptions": [
    "01/01 0000: Sample external narrative entry.",
    "12/31 2359: Another external sample entry."
  ]
}
```

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
