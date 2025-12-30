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
*   **Response Format:** JSON
*   **Caching:** Results are cached for 1 hour. The API uses incremental generation to append new launches without changing existing witty descriptions.

**Example Response:**
```json
{
  "descriptions": [
    "12/17 1527: Falcon 9 Block 5 launches Starlink Group 15-13 from SLC-4E; another batch to LEO, booster lands with textbook precision, mission nominal.",
    "12/17 1342: Falcon 9 Block 5 deploys Starlink Group 6-99 from LC-39A; LEO constellation grows, booster recovery a routine snooze, success confirmed.",
    "12/15 0525: Falcon 9 Block 5 sends Starlink Group 6-82 to LEO from SLC-40; satellites deployed without a hitch, booster sticks the landing, all systems go."
  ]
}
```

### 2. Get API Metrics
Provides real-time and historical performance data, including request counts, cache efficiency, and interactive history.

*   **Endpoint:** `GET /metrics`
*   **Response Format:** JSON

**Example Response:**
```json
{
  "current": {
    "total_requests": 450,
    "cache_hits": 412,
    "cache_misses": 38,
    "api_calls": 12
  },
  "history": [
    {
      "timestamp": "2025-12-29T20:30:00Z",
      "data": { "total_requests": 420, "cache_hits": 385 }
    }
  ]
}
```

### 3. Force Cache Refresh
Manually triggers the API to poll for new launches and generate new narratives using Grok.

*   **Endpoint:** `POST /refresh`
*   **Behavior:** Incremental. It only generates narratives for launches not already in the cache.

---

## Interactive Dashboard

Access the root URL (`/`) in any web browser to view the **API Status Dashboard**.
*   **Real-time Monitoring:** Interactive sparkline charts for traffic and efficiency.
*   **Live Feed:** A visual list of the latest generated narratives.
*   **Manual Control:** A "Force Refresh" button to trigger the `/refresh` endpoint.

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
The API utilizes a **Hybrid Refresh Strategy** to ensure stability and speed:
1.  **Lazy Refresh:** The cache automatically refreshes if the data is older than 1 hour when the narratives endpoint is hit.
2.  **Proactive Refresh:** Designed to be used with a scheduler (e.g., Heroku Scheduler) hitting the `/refresh` endpoint hourly to keep the cache warm.

---

## License
MIT
