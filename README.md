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

### 2. Get Launch Data (default slim)
Returns structured upcoming and previous SpaceX launches. **Default payload is slim** (no `all_data`): the same convenience fields spacex-dashboard already keeps after `pop('all_data')` — `mission`, `net`, `pad`, `video_url`, next-launch `trajectory_data`, etc.

This is the production cutover: default `/launches` drops the ~19MB raw LL blobs. LaunchBuddy `fetchLaunchesFull` should call `GET /launches?full=true` (or use `/launch_details/{id}` / `/launch_raw/{id}` for one launch).

*   **Endpoint:** `GET /launches`
*   **Parameters:**
    *   `force=true` (optional) — refresh the list cache
    *   `full=true` / `include_raw=true` (optional) — restore the legacy giant payload with `all_data` on every launch
    *   `slim=true` (optional) — explicit alias for the default slim shape
*   **Response Format:** JSON
*   **Fields:**
    *   `upcoming`: (array) List of upcoming launch objects.
    *   `previous`: (array) List of historical launch objects.
    *   `last_updated`: (string) ISO8601 timestamp of when the launch data was last fetched.
*   **Fields (Launch Object - Top Level):**
    *   `id`: (string) Unique UUID for the launch.
    *   `name`: (string) Full name of the mission.
    *   `mission`: (string) Same as `name` (dashboard fast-path field).
    *   `net`, `date`, `time`, `status`, `rocket`, `orbit`, `pad`
    *   `video_url`, `x_video_url`
    *   `trajectory_data`: (object, optional) Modeled ground track for the next launch. Ascent follows the pad and an inclination (51.6° for ISS crew/cargo, otherwise the orbit family, or a figure stated in the mission text). It is not a Flight Club or webcast telemetry path.
        *   `trajectory`: (array) List of `{lat, lon, r}` points for ascent (starts at surface, radius 1.0).
        *   `orbit_path`: (array) List of `{lat, lon, r}` points for the rest of one modeled orbit.
        *   `booster_trajectory`: (array) Empty unless Launch Library published an endpoint. A fixed zone more than 2 km from the pad is the geodesic between those surveyed points. A droneship with a downrange and no coordinates is the ascent azimuth out to that distance. Ocean splashdowns and on-pad RTLS zones do not get an invented curve.
        *   `booster_ground_track`: (string or null) `landing_zone_offset`, `published_downrange`, or null.
        *   `landing_site`: (object or null) Published landing latitude/longitude when Launch Library has them.
        *   `inclination_deg`: (number) Inclination used for the ground track.
        *   `sep_idx`: (int or null) `0` when `booster_trajectory` is a standalone leg.
        *   `launch_site`: (object) Coordinates and name of the launch site.
        *   `landing_location`: (string) Name of the landing zone or droneship.
        *   `landing_type`: (string) Type of landing (ASDS, RTLS, Ocean, etc.).
        *   `landing_latitude`, `landing_longitude`, `landing_downrange_km`: published landing geometry from Launch Library, when present.
        *   `orbit`: (string) Normalized orbit type (LEO-Equatorial, LEO-Polar, GTO, etc.).
        *   `mission`: (string) Mission name.
        *   `pad`: (string) Full name of the launch pad.
    *   `all_data` is **omitted by default**. Use `?full=true` or `GET /launch_details/{id}`.
*   **Caching:** 10 minutes. List cache is slim; `all_data` is hydrated only for `?full=true` from `launch_raw_v2:{id}`.

### 3. Get Optimized Launch Data (LaunchBuddy primary path)
Same slim launch list as default `GET /launches`. Field names and Swift-safe values are unchanged.

*   **Endpoint:** `GET /launches_slim`
*   **Parameters:** `force=true` (optional)
*   **Response Format:** JSON
*   **Fields:** Same as default `/launches` (no `all_data`; trajectory kept on the next launch).
*   **Caching:** 10 minutes.

### 4. Get Raw Launch Details
Returns the full, unpruned Launch Library record for a specific launch from the side store, leftover cache, or a single-launch LL fetch.

*   **Endpoint:** `GET /launch_raw/{launch_id}`
*   **Parameters:**
    *   `hot=1` (optional) — for the **current/next launch only**, serve a short-lived (~20s) stale-while-revalidate cache or one single-flight Launch Library GET. Other launch ids ignore `hot` and use the normal side store. Without `hot`, behavior is unchanged (side-store leftover, which can be as stale as the 10-minute list refresh).
*   **Response Format:** JSON
*   **Sample Response:** (Large nested JSON object)
*   **Also:** `GET /launch_details/{launch_id}` always fetches the current LL record (authenticated with the same LL token as the list fetch).

### 5. Get Weather Data
Returns parsed METAR weather data for SpaceX launch and development sites (Starbase, Vandy, Cape, Hawthorne, Bastrop), enhanced with high-frequency live wind data from the National Weather Service (NWS) API.

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
        *   `wind_speed_kts`: (int) Wind speed in knots (from METAR).
        *   `wind_gust_kts`: (int) Wind gust speed in knots (0 if none).
        *   `wind_direction`: (int) Wind direction in degrees.
        *   `live_wind`: (object, optional) High-frequency data from NWS observations.
            *   `speed_kts`: (float) Real-time wind speed.
            *   `gust_kts`: (float) Real-time wind gust.
            *   `direction`: (int) Real-time wind direction.
            *   `timestamp`: (string) Observation time.
            *   `source`: (string) "NWS Real-time".
        *   `visibility_sm`: (float) Visibility in statute miles.
        *   `altimeter_inhg`: (float) Altimeter setting in inches of mercury.
        *   `cloud_cover`: (int) Percentage of cloud cover estimation.
        *   `flight_category`: (string) Estimated flight category (VFR, MVFR, IFR, LIFR).
        *   `raw`: (string) Raw METAR string from the weather service.
        *   `forecast`: (object) 7-day forecast data from Open-Meteo.
            *   `daily`: (object) Daily forecast including `time`, `temperature_2m_max`, `temperature_2m_min`, and `weathercode`.
            *   `hourly`: (object) Hourly data including `time`, `temperature_2m`, `windspeed_10m`, and `winddirection_10m`.
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
          "temperature_2m": [18.5, 18.2, "..."],
          "windspeed_10m": [12.5, 11.8, "..."],
          "winddirection_10m": [160, 155, "..."]
        }
      },
      "last_updated": "2026-01-04T14:55:00Z"
    },
    "Vandy": { "temperature_c": 12, "last_updated": "2026-01-04T14:55:00Z" },
    "Cape": { "temperature_c": 22, "last_updated": "2026-01-04T14:55:00Z" },
    "Hawthorne": { "temperature_c": 19, "last_updated": "2026-01-04T14:55:00Z" },
    "Bastrop": { "temperature_c": 28, "last_updated": "2026-01-04T14:55:00Z" }
  },
  "last_updated": "2026-01-04T14:55:00Z"
}
```
*   **Caching:** 5 minutes. Concurrent refreshes are single-flight + debounced (~20s) so `/weather_all`, `/dashboard`, and the background worker cannot stampede Open-Meteo/METAR.

### 6. Get User-Specific Weather Data
Returns METAR and 7-day forecast data for any user-provided location.

*   **Endpoint:** `GET /user_weather`
*   **Parameters:**
    *   `lat`: (float, required) Latitude of the location.
    *   `lon`: (float, required) Longitude of the location.
    *   `station_id`: (string, optional) ICAO METAR station ID (e.g., "KBRO").
*   **Response Format:** JSON
*   **Fields:** Combined METAR (if `station_id` provided) and Forecast data.
*   **Caching:** None. Always fetches fresh data.

### 7. Get API Metrics
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

### 8. Force Cache Refresh
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

### 9. Dashboard Snapshot (Recommended for Kiosk / PWA)
One response with every cloud-side payload the dashboard needs. Device hardware, Wi-Fi, brightness, hostname, and local settings stay on the Pi.

*   **Endpoint:** `GET /dashboard`
*   **Parameters:**
    *   `tz` (optional IANA timezone, e.g. `America/Chicago`)
    *   `location` (optional site name: `Starbase`, `Vandy`, `Cape`, `Hawthorne`, `Bastrop` — used to pick timezone if `tz` is omitted)
    *   `include_calendar=true` (optional, default true)
    *   `force=true` (optional)
*   **Response Fields:**
    *   `locations`: site coords, timezones, METAR stations, Windy embed URLs
    *   `launches`: slim upcoming/previous launches (`all_data` stripped; trajectory kept on the next launch)
    *   `weather`: all sites including Bastrop
    *   `narratives.descriptions` / `narratives.prepared`: raw ticker strings plus timezone-adjusted rows with launch metadata
    *   `next_launch`: next T- launch, or in-window T+ launch if nothing is upcoming
    *   `upcoming`: next 10 dated launches with `local_date` / `local_time`
    *   `trends`: current-year cumulative + rolling 12-month series for Starship / Falcon 9 / Falcon Heavy
    *   `calendar`: `YYYY-MM-DD` → slim launches (omitted if `include_calendar=false`)
    *   `trajectory`: next-launch globe path
    *   `closest_x_video_url`: nearest-in-time X/Twitter webcast
    *   `last_updated`: per-source timestamps

### 10. Derived Launch Helpers
These are the same fields as `/dashboard`, split out for clients that already have launch data cached.

*   **Site metadata:** `GET /locations`
*   **Next launch:** `GET /next_launch?tz=America/Chicago` (or `location=Starbase`)
*   **Upcoming list:** `GET /upcoming_launches?tz=...&limit=10`
*   **Calendar:** `GET /calendar?tz=...`
*   **Trends:** `GET /launch_trends` or `GET /launch_trends?mode=cumulative` / `mode=rolling`
*   **Trajectory:** `GET /trajectory` (next launch) or `GET /trajectory/{launch_id}`

### 11. Utility Endpoints
*   **Get Single Launch Details:** `GET /launch_details/{launch_id}`
    *   Returns the full raw Launch Library record for one launch (on-demand LL fetch). Default `GET /launches` is slim; use `?full=true` for the old list-wide `all_data` payload.
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

### 12. Satellite GP / TLE (Starlink globe)
Cached CelesTrak General Perturbations (GP / OMM) for the SpaceX dashboard Three.js globe. **Pi clients should poll these endpoints** so they never hit CelesTrak directly (avoids CORS and CelesTrak rate limits).

**These are SGP4 predictions from GP/TLE element sets, not live telemetry.** Propagate on the client with [satellite.js](https://github.com/shashwatak/satellite-js) (`twoline2satrec`) or an equivalent SGP4 library. CelesTrak asks not to hammer their GP API; this service caches for ~1 hour and keeps a longer stale copy if upstream fails.

*   **List:** `GET /satellites/gp?group=starlink`
    *   **Parameters:**
        *   `group` (optional, default `starlink`) — allowlisted CelesTrak `GROUP`: `starlink`, `stations` (ISS / CSS), `visual`, `oneweb`, `gps-ops`, `weather`
        *   `force=true` (optional) — bypass the fresh cache and refetch
    *   **Caching:** Fresh for 1 hour (Redis via `set_cached_data` / `get_cached_data`, in-memory fallback). Stale copies are kept ~48 hours and returned immediately; CelesTrak is refreshed in the background and is not on the request path while a copy exists. A cold cache fetches with an ~18s budget so the handler returns JSON before Heroku's ~30s router timeout. Starlink + stations are warmed on startup and refreshed hourly by the background worker. Responses are pre-rendered; clients that send `Accept-Encoding: gzip` get `Content-Encoding: gzip` (same JSON document). `X-Satellite-Cache` is `hit`, `stale`, or `miss`.
    *   **Response fields:**
        *   `group`, `fetched_at`, `ttl_seconds`, `count`, `stale`, `source`, `note`
        *   `satellites`: compact `{name, norad_id, tle_line1, tle_line2}` (unused OMM keywords omitted so ~6–10k Starlinks stay small)
*   **Starlink alias:** `GET /satellites/starlink` — same payload as `/satellites/gp?group=starlink`
*   **Metadata only:** `GET /satellites/meta?group=starlink` — `fetched_at`, `count`, `group`, `ttl_seconds`, `age_seconds`, `stale`, `allowed_groups` (does not call CelesTrak)
*   **Deployed Starlink (Starship V3 / Group 31):** `GET /satellites/deployed`
    *   Used by spacex-dashboard `globe.html` (`ingestDeployedPayload`) to recolor this flight's sats inside the existing constellation Points shell. Pi clients proxy this path; they do not call CelesTrak.
    *   **Parameters:**
        *   `launch_date` (optional, `YYYY-MM-DD`) — UTC SATCAT launch date. Omit to use the NET date of the Starship launch whose mission text contains `v3` or `group 31-` (Flight 14 is `Starship | Starlink Group 31-1`).
        *   `force=true` (optional) — bypass the fresh cache and refetch SATCAT.
    *   **Source:** CelesTrak SATCAT `GROUP=starlink`, filtered to that `LAUNCH_DATE`, is preferred once it lists the flight. Decayed objects, rocket bodies, and debris are dropped. Remaining payloads are joined to GP TLEs by NORAD (the cached Starlink GP feed, then `gp.php?INTDES=`, then supplemental GP only for catalog numbers the main set does not have). Same-day Starlink rows are included. **No positions are invented.**
    *   **Before SATCAT catalogs the flight:** SpaceX public MEME ephemerides (`https://api.starlink.com/public-files/ephemerides/MANIFEST.txt`, files only — not the operator portal). Candidates are `STARLINK-` ids in the post-Falcon band (id ≥ 40000) whose ephemeris window covers the launch date, or whose window has rolled forward (starts on or after the launch day) and still covers the current time. Each file's inertial state is interpolated at request time into `lat` / `lon` / `alt_km` and an osculating TLE so the existing globe propagator can draw it. `source` is `spacex-manifest-ephemeris`. `norad_id` is omitted until CelesTrak assigns one; `id` / `name` are the SpaceX `STARLINK-` name. When SATCAT later has TLEs for that date, the same endpoint switches back to `celestrak-satcat`.
    *   **Caching:** SATCAT-backed copies are fresh for 1 hour (Redis key `satellites_deployed_v2:{date}`, in-memory fallback). Manifest-backed copies and honest empties are fresh for 10 minutes so a newly published ephemeris is not stuck behind an empty SATCAT cache. Stale copies are kept ~48 hours. Raw multi-megabyte files are not stored; only the parsed satellite records are.
    *   **Response fields the globe reads:**
        *   `satellites`: `{name, norad_id, tle_line1, tle_line2}` for SATCAT, or `{name, id, lat, lon, alt_km, epoch, tle_line1, tle_line2}` for the manifest bridge. Both TLE lines are required; `generation` is `v3` on each row when the launch text says so. Rows without both TLE lines are ignored by the globe.
        *   `catalog_count`: SATCAT matches before the TLE join, or the ephemeris count on the manifest path. `catalog_count > 0` with no TLE lines → dashboard `catalog-no-tle`.
        *   `note`: contains `unavailable` only when SATCAT could not be fetched and nothing is cached → dashboard `satcat-unavailable`. An honest empty catalog does **not** say unavailable → dashboard `awaiting-catalog`.
        *   `source`: `celestrak-satcat` or `spacex-manifest-ephemeris`.
    *   **Also returned:** `launch_date`, `generation`, `mission`, `fetched_at`, `ttl_seconds`, `count`, `stale`, `empty`.
    *   **Sample (SATCAT has no rows for the flight date yet):**
```json
{
  "launch_date": "2026-09-28",
  "generation": "v3",
  "mission": "Starship | Starlink Group 31-1 (Starship Flight 14)",
  "fetched_at": "2026-09-28T14:00:00Z",
  "ttl_seconds": 600,
  "count": 0,
  "catalog_count": 0,
  "stale": false,
  "empty": true,
  "source": "celestrak-satcat",
  "note": "No Starlink SATCAT objects with LAUNCH_DATE 2026-09-28 yet. Positions are SGP4 predictions from GP/TLE element sets, not live telemetry.",
  "satellites": []
}
```
*   **Sample Response (`GET /satellites/gp?group=stations`):**
```json
{
  "group": "stations",
  "fetched_at": "2026-09-19T23:00:00Z",
  "ttl_seconds": 3600,
  "count": 1,
  "stale": false,
  "source": "celestrak",
  "note": "Positions are SGP4 predictions from GP/TLE element sets, not live telemetry.",
  "satellites": [
    {
      "name": "ISS (ZARYA)",
      "norad_id": 25544,
      "tle_line1": "1 25544U 98067A   26262.30395647  .00006211  00000+0  12007-3 0  9997",
      "tle_line2": "2 25544  51.6308 194.2901 0004815 157.3949 202.7252 15.49175317586346"
    }
  ]
}
```

---

## Interactive Dashboard

Access the root URL (`/`) in any web browser to view the **API Status Dashboard**.
*   **Tabbed Interface:** Switch between **Narratives**, **Launches**, and **Weather** views (Starbase, Vandy, Cape, Hawthorne, Bastrop).
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
1.  **Background Refresh:** A dedicated worker thread in the backend automatically refreshes the cache for Narratives (15m), Launches (10m), Weather (2m for high-frequency wind), and Starlink / stations GP (1h). `/launches` and `/launches_slim` stay on this 10-minute list cadence.
2.  **Hot raw (opt-in):** `GET /launch_raw/{id}?hot=1` refreshes **only the current/next launch** from Launch Library on a ~20s stale-while-revalidate TTL (single-flight per id). This is not a full-list refresh. Other ids, and `/launch_raw/{id}` without `hot`, keep the long-lived side store.
3.  **Manual Force:** Users can trigger an immediate refresh via the dashboard buttons or by appending `?force=true` to API requests.
4.  **Incremental History:** Previous launch data is never fully replaced; new launches are appended to the existing historical cache to preserve a continuous record.
5.  **Satellite GP:** `/satellites/gp`, `/satellites/starlink`, and `/satellites/meta` are a CelesTrak proxy with a 1-hour freshness window. A stale Starlink catalog is returned immediately (gzip when the client accepts it) and refreshed in the background, so a cold dyno does not rebuild the catalog on the request. `/satellites/deployed` prefers the Starship V3 SATCAT join, then SpaceX public ephemerides. Element sets are GP/TLE; globe dots are SGP4 predictions, not live positions. An empty deployed list means neither SATCAT nor the manifest had plottable sats for that launch date.

---

## License
MIT
