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
import zlib
import base64
import pytz
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

def set_cached_data(key, data, ttl=None):
    """Set data in Redis with compression and better error handling."""
    if not r:
        return False
    try:
        json_str = json.dumps(data)
        # Compress and base64 encode to store as string in the current Redis client
        compressed = zlib.compress(json_str.encode('utf-8'))
        encoded = base64.b64encode(compressed).decode('utf-8')
        if ttl:
            r.setex(key, ttl, encoded)
        else:
            r.set(key, encoded)
        return True
    except Exception as e:
        print(f"Redis write error for {key}: {e}")
        return False

def get_cached_data(key):
    """Get data from Redis with decompression support."""
    if not r:
        return None
    try:
        data = r.get(key)
        if not data:
            return None
        
        # Try to decompress (it will be base64 encoded string)
        try:
            decoded = base64.b64decode(data)
            decompressed = zlib.decompress(decoded)
            return json.loads(decompressed)
        except Exception:
            # Fallback for old uncompressed data
            try:
                return json.loads(data)
            except:
                return None
    except Exception as e:
        print(f"Redis read error for {key}: {e}")
        return None

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
CACHE_TTL = 900  # 15 minutes TTL in seconds (aligned with dashboard)
HISTORY_LIMIT = 43200 # 30 days at 1 minute intervals
SEEDING_STATUS_KEY = "seeding_status_v2"
SEEDING_STOP_SIGNAL_KEY = "seeding_stop_signal_v2"
_last_snapshot_time = 0

# Trajectory Helpers and Cache
class Profiler:
    def mark(self, msg):
        print(f"[PROFILER] {msg}")

class Logger:
    def info(self, msg):
        print(f"[INFO] {msg}")
    def warning(self, msg):
        print(f"[WARNING] {msg}")

profiler = Profiler()
logger = Logger()

_TRAJECTORY_DATA_CACHE = None
TRAJECTORY_CACHE_FILE = "trajectory_cache.json"

def load_cache_from_file(filename):
    """Load cache from Redis instead of file if possible, or fallback to file."""
    if r:
        data = get_cached_data("trajectory_cache_v2")
        if data:
            return {"data": data}
    
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                return {"data": json.load(f)}
        except Exception as e:
            print(f"Error loading cache file: {e}")
    return {"data": {}}

def save_cache_to_file(filename, data, updated_at=None):
    """Save cache to Redis and local file."""
    if r:
        set_cached_data("trajectory_cache_v2", data)
    
    try:
        with open(filename, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error saving cache file: {e}")

def compute_orbit_radius(orbit_label):
    """Compute orbital radius based on orbit type."""
    label = (orbit_label or '').lower()
    EARTH_RADIUS = 1.0 # Normalized
    if 'gto' in label:
        return 1.15 # Geostationary Transfer
    if 'suborbital' in label:
        return 1.05
    return 1.08 # Standard LEO

def _ang_dist_deg(p1, p2):
    """Calculate angular distance between two points in degrees."""
    lat1, lon1 = math.radians(p1['lat']), math.radians(p1['lon'])
    lat2, lon2 = math.radians(p2['lat']), math.radians(p2['lon'])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1-a)))
    return math.degrees(c)

def get_launch_trajectory_data(upcoming_launches, previous_launches=None):
    """
    Get trajectory data for the next upcoming launch or a specific launch.
    If upcoming_launches is a dict, treat it as a single launch object.
    If upcoming_launches is a list, use the first item (existing behavior).
    Standalone version of Backend.get_launch_trajectory.
    """
    profiler.mark("get_launch_trajectory_data Start")
    logger.info("get_launch_trajectory_data called")
    
    # Handle single launch object (dict) vs list of launches
    if isinstance(upcoming_launches, dict):
        # Single launch object
        display_launches = [upcoming_launches]
    else:
        # List of launches (existing behavior)
        display_launches = upcoming_launches
        if not display_launches:
            logger.info("No upcoming launches, trying recent launches")
            if previous_launches:
                recent_launches = previous_launches[:5]
                if recent_launches:
                    display_launches = [{
                        'mission': launch.get('mission', 'Unknown'),
                        'pad': launch.get('pad', 'Cape Canaveral'),
                        'orbit': launch.get('orbit', 'LEO'),
                        'net': launch.get('net', ''),
                        'landing_type': launch.get('landing_type'),
                        'landing_location': launch.get('landing_location')
                    } for launch in recent_launches]
                    logger.info(f"Using {len(display_launches)} recent launches for demo")

    if not display_launches:
        logger.info("No launches available at all")
        return None

    next_launch = display_launches[0]
    mission_name = next_launch.get('mission', 'Unknown')
    pad = next_launch.get('pad', '')
    orbit = next_launch.get('orbit', '')
    logger.info(f"Next launch: {mission_name} from {pad}")

    # Launch site coordinates
    launch_sites = {
        'LC-39A': {'lat': 28.6084, 'lon': -80.6043, 'name': 'Cape Canaveral, FL'},
        'LC-40': {'lat': 28.5619, 'lon': -80.5773, 'name': 'Cape Canaveral, FL'},
        'SLC-4E': {'lat': 34.6321, 'lon': -120.6107, 'name': 'Vandenberg, CA'},
        'Starbase': {'lat': 25.9975, 'lon': -97.1566, 'name': 'Starbase, TX'},
        'Launch Complex 39A': {'lat': 28.6084, 'lon': -80.6043, 'name': 'Cape Canaveral, FL'},
        'Launch Complex 40': {'lat': 28.5619, 'lon': -80.5773, 'name': 'Cape Canaveral, FL'},
        'Space Launch Complex 4E': {'lat': 34.6321, 'lon': -120.6107, 'name': 'Vandenberg, CA'}
    }

    # Find launch site coordinates
    launch_site = None
    matched_site_key = None
    for site_key, site_data in launch_sites.items():
        if site_key in pad:
            launch_site = site_data
            matched_site_key = site_key
            break

    if not launch_site:
        launch_site = launch_sites['LC-39A']
        matched_site_key = 'LC-39A'
        logger.info(f"Using default launch site: {launch_site}")

    def _normalize_orbit(orbit_label: str, site_name: str) -> str:
        try:
            label = (orbit_label or '').lower()
            if 'gto' in label or 'geostationary' in label:
                return 'GTO'
            if 'suborbital' in label:
                return 'Suborbital'
            # Explicit Polar/SSO detection
            if any(k in label for k in ['polar', 'sso', 'sun-synchronous']):
                return 'LEO-Polar'
            if 'leo' in label or 'low earth orbit' in label:
                if 'Vandenberg' in site_name:
                    return 'LEO-Polar'
                return 'LEO-Equatorial'
            return 'Default'
        except Exception:
            return 'Default'

    normalized_orbit = _normalize_orbit(orbit, launch_site.get('name', ''))

    # Resolve an inclination assumption
    def _resolve_inclination_deg(norm_orbit: str, site_name: str, site_lat: float) -> float:
        try:
            label = (orbit or '').lower()
            if 'iss' in label:
                return 51.6
            if norm_orbit == 'LEO-Polar' or 'sso' in label or 'sun-synchronous' in label:
                return 97.5 # Better average for SSO
            if norm_orbit == 'LEO-Equatorial':
                base = abs(site_lat)
                return max(20.0, min(60.0, base + 0.5))
            if norm_orbit == 'GTO':
                return max(20.0, min(35.0, abs(site_lat)))
            if norm_orbit == 'Suborbital':
                return max(10.0, min(45.0, abs(site_lat)))
        except Exception:
            pass
        return 30.0

    assumed_incl = _resolve_inclination_deg(
        normalized_orbit,
        launch_site.get('name', ''),
        launch_site.get('lat', 0.0)
    )

    ORBIT_CACHE_VERSION = 'v230-hybrid-ro'
    landing_type = next_launch.get('landing_type')
    landing_loc = next_launch.get('landing_location')
    cache_key = f"{ORBIT_CACHE_VERSION}:{matched_site_key}:{normalized_orbit}:{round(assumed_incl,1)}:{landing_type}:{landing_loc}"
    
    global _TRAJECTORY_DATA_CACHE
    if _TRAJECTORY_DATA_CACHE is None:
        logger.info("Loading trajectory cache from disk...")
        cache_loaded = load_cache_from_file(TRAJECTORY_CACHE_FILE)
        if cache_loaded and isinstance(cache_loaded.get('data'), dict):
            _TRAJECTORY_DATA_CACHE = cache_loaded['data']
        else:
            # Handle direct dictionary without 'data' wrapper if it exists
            _TRAJECTORY_DATA_CACHE = cache_loaded if isinstance(cache_loaded, dict) else {}

    if cache_key in _TRAJECTORY_DATA_CACHE:
        cached = _TRAJECTORY_DATA_CACHE[cache_key]
        logger.info(f"Trajectory in-memory cache hit for {cache_key}")
        return {
            'launch_site': cached.get('launch_site', launch_site),
            'trajectory': cached.get('trajectory', []),
            'booster_trajectory': cached.get('booster_trajectory', []),
            'sep_idx': cached.get('sep_idx'),
            'orbit_path': cached.get('orbit_path', []),
            'orbit': orbit or cached.get('orbit', normalized_orbit),
            'mission': mission_name,
            'pad': pad,
            'landing_type': landing_type,
            'landing_location': cached.get('landing_location', next_launch.get('landing_location'))
        }

    traj_cache = _TRAJECTORY_DATA_CACHE
    logger.info(f"Trajectory cache miss for {cache_key}; generating new trajectory")

    def get_radius(progress, target_radius, offset=0.0):
        """Calculate visual radius with a smooth quadratic ease-out to avoid janky dips."""
        TRAJ_START_RADIUS = 1.012 
        # Use quadratic ease-out: 2t - t^2. This ensures vertical velocity is zero at the end.
        climb_progress = 2 * progress - progress**2
        radius = TRAJ_START_RADIUS + (target_radius - TRAJ_START_RADIUS) * climb_progress + offset
        return radius

    def generate_curved_trajectory(start_point, end_point, num_points, orbit_type='default', end_bearing_deg=None):
        points = []
        start_lat = start_point['lat']
        start_lon = start_point['lon']
        end_lat = end_point['lat']
        end_lon = end_point['lon']

        if end_bearing_deg is not None:
            lat1 = math.radians(start_lat); lon1 = math.radians(start_lon)
            lat2 = math.radians(end_lat);   lon2 = math.radians(end_lon)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1-a)))
            ang_deg = math.degrees(c)
            # Tighter control point distance (L) to avoid "flat" segments near insertion
            L = min(15.0, max(3.0, ang_deg / 4.0))
            br = math.radians(end_bearing_deg)
            cos_lat = max(1e-6, math.cos(math.radians(end_lat)))
            dlat_deg = L * math.cos(br)
            dlon_deg = (L * math.sin(br)) / cos_lat
            control_lat = end_lat - dlat_deg
            control_lon = (end_lon - dlon_deg + 180.0) % 360.0 - 180.0
        else:
            mid_lat = (start_lat + end_lat) / 2
            mid_lon = (start_lon + end_lon) / 2
            dist = _ang_dist_deg(start_point, end_point)
            
            # Scale control point offset based on distance
            offset = max(5, min(30, dist * 0.4))
            
            if orbit_type == 'polar':
                control_lat = max(-85.0, mid_lat - offset)
                control_lon = mid_lon - offset/2
            elif orbit_type == 'equatorial':
                # Aim more towards equator
                target_equator_lat = 0
                control_lat = (mid_lat + target_equator_lat) / 2
                control_lon = mid_lon + offset
            elif orbit_type == 'gto':
                control_lat = mid_lat + offset
                control_lon = mid_lon + offset * 2
            elif orbit_type == 'suborbital':
                # Suborbital/Booster return needs a tighter arc
                control_lat = mid_lat + offset/4
                control_lon = mid_lon + offset/4
            else:
                control_lat = mid_lat + offset
                control_lon = mid_lon + offset * 1.5

        for i in range(num_points + 1):
            t = i / num_points
            lat = (1-t)**2 * start_lat + 2*(1-t)*t * control_lat + t**2 * end_lat
            lon = (1-t)**2 * start_lon + 2*(1-t)*t * control_lon + t**2 * end_lon
            lon = (lon + 180) % 360 - 180
            points.append({'lat': lat, 'lon': lon})
        return points

    def generate_spherical_path(start_point, inclination_deg, num_points=2000, descending=False):
        """Generate a full Great Circle path passing through start_point with given inclination."""
        lat0 = float(start_point['lat']); lon0 = float(start_point['lon'])
        # Handle nearly-polar or nearly-equatorial cases with small epsilon
        eff_i_deg = max(0.1, min(179.9, abs(inclination_deg)))
        i_rad = math.radians(eff_i_deg)
        lat0_rad = math.radians(lat0); lon0_rad = math.radians(lon0)
        
        # Position vector
        x0 = math.cos(lat0_rad) * math.cos(lon0_rad)
        y0 = math.cos(lat0_rad) * math.sin(lon0_rad)
        z0 = math.sin(lat0_rad)
        
        sin_i = math.sin(i_rad)
        # Argument of latitude u0 at start_point
        u0 = math.asin(max(-1.0, min(1.0, z0 / (sin_i or 1e-6))))
        
        # If we want to be on the descending part of the orbit (going South)
        if descending:
            u0 = math.pi - u0
            
        # Longitude of Ascending Node (Omega)
        Omega = math.atan2(y0, x0) - math.atan2(math.sin(u0) * math.cos(i_rad), math.cos(u0))
        
        cosO = math.cos(Omega); sinO = math.sin(Omega)
        cosi = math.cos(i_rad); sili = math.sin(i_rad)
        
        points = []
        for k in range(num_points):
            u = u0 + (2.0 * math.pi * k) / num_points
            cu = math.cos(u); su = math.sin(u)
            # Transform from orbital frame to ECEF-like lat/lon
            x = cosO * cu - sinO * (su * cosi)
            y = sinO * cu + cosO * (su * cosi)
            z = su * sili
            points.append({
                'lat': math.degrees(math.atan2(z, max(1e-12, math.hypot(x, y)))), 
                'lon': (math.degrees(math.atan2(y, x)) + 180) % 360 - 180
            })
        return points

    # Main generation
    target_r = compute_orbit_radius(orbit)
    
    # Heuristic for launch direction: VAFB launches are usually descending (Southward)
    is_desc = False
    if 'Vandenberg' in launch_site.get('name', '') or 'SLC-4E' in pad:
        is_desc = True
    
    # Generate the Master Path (The Orbit)
    # We use 2000 points for a smooth full circle
    master_path = generate_spherical_path(launch_site, assumed_incl, num_points=2000, descending=is_desc)
    
    # The ascent trajectory is a segment of the SAME Master Path
    # LEO ascent usually takes ~10% of an orbit. Let's use 12.5% for visual length.
    traj_len = int(len(master_path) * 0.125)
    trajectory = [p.copy() for p in master_path[:traj_len]]
    
    # The orbit path is the REMAINING part of the Master Path to avoid overlap and "double touching"
    if normalized_orbit != 'Suborbital':
        # Starts at the insertion point (traj_len-1) and goes around to the end of the master path circle.
        # This prevents the orbit line from being drawn over the ascent segment.
        orbit_path = [p.copy() for p in master_path[traj_len-1:]]
    else:
        orbit_path = []

    # Set radii for the orbit path
    for p in orbit_path:
        p['r'] = target_r

    # Add radii to main trajectory
    for i, p in enumerate(trajectory):
        progress = i / (len(trajectory) - 1)
        p['r'] = get_radius(progress, target_r)

    # Booster Return Trajectory (RTLS vs ASDS vs Expendable simulation)
    booster_trajectory = []
    sep_idx = None
    if trajectory and len(trajectory) > 5:
        try:
            # Separation usually around 1/5th of the way to orbit
            sep_idx = max(2, len(trajectory) // 5)
            # Create booster ascent with a tiny radius offset to ensure it's visible outside main trajectory
            ascent_part = []
            for i, p in enumerate(trajectory[:sep_idx+1]):
                bp = p.copy()
                # Apply a subtle 0.001 offset (approx 6km) to keep booster visible
                bp['r'] = (bp.get('r') or 1.012) + 0.001
                ascent_part.append(bp)
                
            sep_point = ascent_part[-1]
            sep_radius = sep_point['r']

            l_type = (landing_type or '').upper()
            l_loc = (next_launch.get('landing_location') or '').upper()
            combined_landing_info = f"{l_type} {l_loc}"
            
            # Expanded detection for ASDS and RTLS
            asds_keywords = ['ASDS', 'DRONE', 'SHIP', 'OCISLY', 'JRTI', 'ASOG', 'GRAVITAS', 'INSTRUCTIONS', 'STILL LOVE YOU']
            rtls_keywords = ['RTLS', 'LAUNCH SITE', 'CATCH', 'TOWER', 'LZ', 'LANDING ZONE']
            
            if any(k in combined_landing_info for k in asds_keywords):
                # Droneship landing: continues downrange to a point further along the trajectory
                landing_idx = min(len(trajectory) - 1, max(sep_idx + 3, len(trajectory) // 3))
                landing_point = trajectory[landing_idx]
                return_part = generate_curved_trajectory(sep_point, landing_point, 100, orbit_type='suborbital')
                
                # Add radius to return part (ballistic arc)
                for i, p in enumerate(return_part):
                    prog = i / (len(return_part) - 1)
                    # Parabolic arc from sep_radius to 1.01, peaking at sep_radius + 0.02
                    p['r'] = sep_radius + (1.012 - sep_radius) * prog + 0.02 * math.sin(prog * math.pi)
                
                booster_trajectory = return_part
                sep_idx = 0
                logger.info(f"Generated ASDS booster trajectory (info: {combined_landing_info})")
            elif any(k in combined_landing_info for k in rtls_keywords):
                # RTLS/Catch: returns to launch site
                return_part = generate_curved_trajectory(sep_point, launch_site, 100, orbit_type='suborbital')
                for i, p in enumerate(return_part):
                    prog = i / (len(return_part) - 1)
                    # Higher arc for RTLS boostback (~300km peak)
                    p['r'] = sep_radius + (1.012 - sep_radius) * prog + 0.03 * math.sin(prog * math.pi)
                
                booster_trajectory = return_part
                sep_idx = 0
                logger.info(f"Generated RTLS/Catch booster trajectory (info: {combined_landing_info})")
            elif any(k in combined_landing_info for k in ['OCEAN', 'SPLASHDOWN']):
                landing_idx = min(len(trajectory) - 1, max(sep_idx + 2, len(trajectory) // 4))
                landing_point = trajectory[landing_idx]
                return_part = generate_curved_trajectory(sep_point, landing_point, 100, orbit_type='suborbital')
                for i, p in enumerate(return_part):
                    prog = i / (len(return_part) - 1)
                    p['r'] = sep_radius + (1.012 - sep_radius) * prog + 0.015 * math.sin(prog * math.pi)
                booster_trajectory = return_part
                sep_idx = 0
                logger.info(f"Generated Ocean splashdown booster trajectory (type: {landing_type})")
            else:
                # Expendable/Unknown: no return trajectory to show
                booster_trajectory = []
                sep_idx = None
                logger.info(f"Skipping booster trajectory for unknown/expendable type: {landing_type}")
        except Exception as e:
            logger.warning(f"Booster trajectory generation failed: {e}")

    result = {
        'launch_site': launch_site,
        'trajectory': trajectory,
        'booster_trajectory': booster_trajectory,
        'sep_idx': sep_idx,
        'orbit_path': orbit_path,
        'orbit': orbit,
        'mission': mission_name,
        'pad': pad,
        'landing_type': landing_type,
        'landing_location': next_launch.get('landing_location')
    }

    # Persist to cache
    try:
        traj_cache[cache_key] = {
            'launch_site': launch_site,
            'trajectory': trajectory,
            'booster_trajectory': booster_trajectory,
            'sep_idx': sep_idx,
            'orbit_path': orbit_path,
            'orbit': normalized_orbit,
            'inclination_deg': assumed_incl,
            'landing_type': landing_type,
            'landing_location': next_launch.get('landing_location'),
            'model': 'v11-non-overlapping-orbit'
        }
        save_cache_to_file(TRAJECTORY_CACHE_FILE, traj_cache, datetime.now(pytz.utc))
    except Exception as e:
        logger.warning(f"Failed to save trajectory cache: {e}")

    return result

_local_seeding_status = {"is_running": False, "last_status": "Idle", "total_pulled": 0, "oldest_launch": None}
_stop_seeding_requested = False

def update_seeding_status(is_running, last_status, total_pulled=0, oldest_launch=None):
    """Update the seeding status in Redis and memory."""
    global _local_seeding_status
    status = {
        "is_running": is_running,
        "last_status": last_status,
        "total_pulled": total_pulled,
        "oldest_launch": oldest_launch,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    _local_seeding_status = status
    if r:
        try:
            r.set(SEEDING_STATUS_KEY, json.dumps(status))
        except Exception as e:
            print(f"Redis error in update_seeding_status: {e}")
    return status

def reset_stuck_seeding():
    """Reset seeding status if it's marked as running but the app just started."""
    if r:
        try:
            data = r.get(SEEDING_STATUS_KEY)
            if data:
                status = json.loads(data)
                if status.get("is_running"):
                    print("Detected stuck seeding status at startup. Resetting.")
                    update_seeding_status(False, "Interrupted (App Restart)", status.get("total_pulled", 0), status.get("oldest_launch"))
        except Exception as e:
            print(f"Error resetting stuck seeding: {e}")

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

@app.post("/reset_metrics")
def reset_app_metrics():
    """Endpoint to reset all metrics and history."""
    global _local_metrics, _local_metrics_history, _last_snapshot_time
    
    # Reset in-memory
    _local_metrics = {
        "total_requests": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "api_calls": 0
    }
    _local_metrics_history = []
    _last_snapshot_time = 0
    
    # Reset Redis
    if r:
        try:
            r.delete(METRICS_KEY)
            r.delete(METRICS_HISTORY_KEY)
        except Exception as e:
            print(f"Redis error in reset_app_metrics: {e}")
            return {"status": "Error", "message": str(e)}
            
    return {"status": "Success", "message": "All metrics and history have been reset."}

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
    elif range_type == "7d":
        start_time = now - timedelta(days=7)
        history = [h for h in history if datetime.fromisoformat(h['timestamp'].replace('Z', '+00:00')) > start_time]
    elif range_type == "30d":
        start_time = now - timedelta(days=30)
        history = [h for h in history if datetime.fromisoformat(h['timestamp'].replace('Z', '+00:00')) > start_time]

    # Calculate range-relative current stats if we have history
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

    # Calculate live hits/day BEFORE downsampling for better accuracy
    hits_per_day = 0
    if len(history) > 1:
        first_h = history[0]
        last_h = history[-1]
        try:
            t1 = datetime.fromisoformat(first_h['timestamp'].replace('Z', '+00:00'))
            t2 = datetime.fromisoformat(last_h['timestamp'].replace('Z', '+00:00'))
            duration_hours = (t2 - t1).total_seconds() / 3600
            if duration_hours > 0.1: # At least 6 mins of data
                hits_diff = last_h['data']['total_requests'] - first_h['data']['total_requests']
                hits_per_day = (hits_diff / duration_hours) * 24
        except: pass

    # Downsample history for charting
    if range_type == "24h" and len(history) > 100:
        history = history[::15] # ~15m intervals
    elif range_type == "7d" and len(history) > 200:
        history = history[::60] # ~1h intervals
    elif range_type == "30d" and len(history) > 300:
        history = history[::240] # ~4h intervals

    return {
        "current": current, 
        "range_stats": range_stats,
        "history": history, 
        "hits_per_day": round(hits_per_day, 1)
    }

def generate_narratives(existing_narratives=None):
    """Fetch launches and generate narratives using Grok, appending new ones only."""
    increment_metric("api_calls")
    current_time = datetime.now(timezone.utc)
    three_months_ago = current_time - timedelta(days=90)
    # Use v2.3.0
    url = (
        f"https://ll.thespacedevs.com/2.3.0/launches/previous/"
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
    
    # API v2.3.0 uses vid_urls, while v2.0.0 uses vidURLs
    raw_vid_urls = launch.get('vid_urls') or launch.get('vidURLs') or []
    vid_urls = raw_vid_urls if isinstance(raw_vid_urls, list) else []
    
    return {
        'id': launch.get('id'),
        'mission': launch.get('name', 'Unknown'),
        'date': launch.get('net').split('T')[0] if launch.get('net') else 'TBD',
        'time': launch.get('net').split('T')[1].split('Z')[0] if launch.get('net') and 'T' in launch.get('net') else 'TBD',
        'net': launch.get('net'),
        'status': launch.get('status', {}).get('name', 'Unknown'),
        'status_id': launch.get('status', {}).get('id'),
        'rocket': launch.get('rocket', {}).get('configuration', {}).get('name', 'Unknown'),
        'orbit': mission_data.get('orbit', {}).get('name', 'Unknown'),
        'pad': launch.get('pad', {}).get('name', 'Unknown'),
        'video_url': vid_urls[0].get('url', '') if vid_urls else '',
        'x_video_url': next((v['url'] for v in vid_urls if v.get('url') and ('x.com' in v['url'].lower() or 'twitter.com' in v['url'].lower())), '') if vid_urls else '',
        'landing_type': landing_type,
        'landing_location': landing_location,
        'is_detailed': is_detailed,
        # New enriched fields for "ALL data" view
        'description': mission_data.get('description', ''),
        'image': launch.get('image', '') if isinstance(launch.get('image'), str) else (launch.get('image', {}).get('url', '') if isinstance(launch.get('image'), dict) else ''),
        'window_start': launch.get('window_start'),
        'window_end': launch.get('window_end'),
        'probability': launch.get('probability'),
        'holdreason': launch.get('holdreason'),
        'failreason': launch.get('failreason'),
        # Ensure every field returned by the API is available, but prune redundant URLs to save space
        'all_data': {k: v for k, v in launch.items() if k not in ['vidURLs', 'infoURLs', 'vid_urls', 'info_urls', 'infographic']}
    }

def fetch_launch_details(launch_id: str):
    """Fetch detailed information for a single launch to get vidURLs."""
    if not launch_id:
        return None
    increment_metric("api_calls")
    # Use v2.3.0 for detailed fetch
    url = f"https://ll.thespacedevs.com/2.3.0/launches/{launch_id}/"
    print(f"Fetching details for launch {launch_id}")
    try:
        try:
            response = requests.get(url, timeout=10, verify=True)
        except Exception:
            # Fallback for SSL issues
            response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Failed to fetch launch details for {launch_id}: {e}")
        return None

def fetch_launches(existing_previous=None, existing_upcoming=None):
    """Fetch SpaceX launch data (v2.3.0) using detailed mode to get all fields."""
    headers = {'Authorization': f'Token {LL_API_KEY}'}
    
    combined_prev = existing_previous or []
    combined_up = existing_upcoming or []
    
    # 1. Fetch Previous (Incremental)
    try:
        increment_metric("api_calls")
        prev_url = 'https://ll.thespacedevs.com/2.3.0/launches/previous/?lsp__name=SpaceX&limit=15&mode=detailed'
        prev_response = requests.get(prev_url, headers=headers, timeout=15)
        prev_response.raise_for_status()
        prev_data = prev_response.json().get('results', [])
        parsed_prev = [parse_launch_data(l, is_detailed=True) for l in prev_data]
        
        if existing_previous:
            # Incremental update: Replace existing ones with fresh data if present in latest fetch
            new_ids = {l['id'] for l in parsed_prev if 'id' in l}
            filtered_old = [l for l in existing_previous if l.get('id') not in new_ids]
            # Maintain newest-first order by sorting after merge
            combined_prev = sorted(parsed_prev + filtered_old, key=lambda x: x.get('net') or '', reverse=True)
            # Limit history to 2000 items for efficiency and to allow seeding
            combined_prev = combined_prev[:2000]
        else:
            combined_prev = sorted(parsed_prev, key=lambda x: x.get('net') or '', reverse=True)
    except Exception as e:
        print(f"Error fetching previous launches: {e}")

    # 2. Fetch Upcoming (Full Refresh)
    try:
        # Upcoming is always fully refreshed as statuses and dates shift frequently
        increment_metric("api_calls")
        up_url = 'https://ll.thespacedevs.com/2.3.0/launches/upcoming/?lsp__name=SpaceX&limit=15&mode=detailed'
        up_response = requests.get(up_url, headers=headers, timeout=15)
        up_response.raise_for_status()
        up_data = up_response.json().get('results', [])
        
        parsed_up = [parse_launch_data(l, is_detailed=True) for l in up_data]
        # Filter out finished launches from upcoming list to satisfy user request
        # Status IDs: 3=Success, 4=Failure, 7=Partial Failure
        # This keeps "active" launches like "In Flight" (ID 6) and "Go" (ID 1)
        combined_up = [l for l in parsed_up if l.get('status_id') not in [3, 4, 7]]
    except Exception as e:
        print(f"Error fetching upcoming launches: {e}")
        
    return {
        'previous': combined_prev,
        'upcoming': combined_up
    }

def seed_historical_launches():
    """Seed the historical launch cache by pulling increasingly older launches in batches of 5 until we hit the api limit."""
    print("Starting historical launch seeding...")
    cache_key = "launches_cache_v2"
    
    # Get initial state
    existing_previous = []
    cached_data = get_cached_data(cache_key)
    if cached_data:
        existing_previous = cached_data.get('previous', [])
    
    # Sort to find the actual oldest launch
    if existing_previous:
        existing_previous.sort(key=lambda x: x.get('net') or '', reverse=True)
        oldest_so_far = existing_previous[-1].get('net')
    else:
        oldest_so_far = None
        
    total_so_far = len(existing_previous)
    update_seeding_status(True, "Starting batch fetch...", total_so_far, oldest_so_far)

    # We use a loop to keep fetching until we hit a limit or run out of data
    upcoming = []
    while True:
        # Check for stop signal
        global _stop_seeding_requested
        stop_signal = _stop_seeding_requested
        if r:
            try:
                if r.get(SEEDING_STOP_SIGNAL_KEY) == "true":
                    stop_signal = True
            except: pass
            
        if stop_signal:
            print("Seeding: Stop signal received. Terminating.")
            update_seeding_status(False, "Stopped by user.", total_so_far, oldest_so_far)
            _stop_seeding_requested = False
            if r:
                try: r.delete(SEEDING_STOP_SIGNAL_KEY)
                except: pass
            break

        # Get current state from cache to determine where we are
        # This allows us to pick up any new launches added by the regular refresh
        cached_data = get_cached_data(cache_key)
        if cached_data:
            existing_previous = cached_data.get('previous', [])
            upcoming = cached_data.get('upcoming', [])
        
        if existing_previous:
            # Crucial: Always sort to ensure the last item is truly the oldest
            existing_previous.sort(key=lambda x: x.get('net') or '', reverse=True)
            oldest_so_far = existing_previous[-1].get('net')
        else:
            oldest_so_far = None
        
        total_so_far = len(existing_previous)

        if total_so_far >= 2000:
            print(f"Seeding: Already at history limit ({total_so_far}). Stopping.")
            update_seeding_status(False, f"Complete. Hit history limit ({total_so_far}).", total_so_far, oldest_so_far)
            break
        
        # Use date-based pagination for robustness against cache shifts
        status_msg = f"Fetching launches older than {oldest_so_far.split('T')[0] if oldest_so_far else 'now'}..."
        update_seeding_status(True, status_msg, total_so_far, oldest_so_far)
        print(f"Seeding: {status_msg}")
        
        try:
            headers = {'Authorization': f'Token {LL_API_KEY}'}
            params = {
                'lsp__name': 'SpaceX',
                'limit': 20,
                'mode': 'detailed',
                'ordering': '-net'
            }
            if oldest_so_far:
                params['net__lt'] = oldest_so_far

            # Pull increasingly older launches in batches
            increment_metric("api_calls")
            url = 'https://ll.thespacedevs.com/2.3.0/launches/previous/'
            response = requests.get(url, headers=headers, params=params, timeout=15)
            
            if response.status_code == 429:
                print("Hit API rate limit (429) during seeding. Stopping for now.")
                update_seeding_status(False, "Hit API rate limit (429).", total_so_far, oldest_so_far)
                break
                
            response.raise_for_status()
            data = response.json()
            results = data.get('results', [])
            
            if not results:
                print("No more historical launches found. Seeding complete.")
                update_seeding_status(False, "Seeding complete. No more data.", total_so_far, oldest_so_far)
                break
                
            parsed_new = [parse_launch_data(l, is_detailed=True) for l in results]
            
            # Filter out duplicates (possible if multiple launches have exact same timestamp)
            existing_ids = {l['id'] for l in existing_previous if 'id' in l}
            unique_new = [l for l in parsed_new if l.get('id') not in existing_ids]
            
            if not unique_new and results:
                print("Seeding: All fetched results are duplicates. Stopping.")
                update_seeding_status(False, "Stopped to avoid duplicate loop.", total_so_far, oldest_so_far)
                break

            # Append older launches and maintain sorted order
            combined_prev = sorted(existing_previous + unique_new, key=lambda x: x.get('net') or '', reverse=True)
            
            # Limit history to 2000 items for efficiency
            combined_prev = combined_prev[:2000]
            
            # Update cache in Redis
            result = {
                "upcoming": upcoming,
                "previous": combined_prev,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            set_cached_data(cache_key, result)
            
            total_so_far = len(combined_prev)
            oldest_so_far = combined_prev[-1].get('net')
            print(f"Added {len(unique_new)} historical launches. Total: {total_so_far}")
            update_seeding_status(True, f"Added {len(unique_new)} launches.", total_so_far, oldest_so_far)
            
            # Brief sleep between batches
            time.sleep(1)
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print("Hit API rate limit (429) during seeding. Stopping.")
                update_seeding_status(False, "Hit API rate limit (429).", total_so_far, oldest_so_far)
                break
            else:
                msg = f"HTTP error during seeding: {e}"
                print(msg)
                update_seeding_status(False, msg, total_so_far, oldest_so_far)
                break
        except Exception as e:
            msg = f"Unexpected error during seeding: {e}"
            print(msg)
            update_seeding_status(False, msg, total_so_far, oldest_so_far)
            break

def parse_metar(raw_metar: str):
    """Parse METAR string to extract weather data."""
    temperature_c = 25
    dewpoint_c = 15
    wind_speed_kts = 0
    wind_gust_kts = 0
    wind_direction = 0
    cloud_cover = 0
    visibility_sm = 10
    altimeter_inhg = 29.92
    
    try:
        # Extract temperature and dewpoint
        # Format: 18/14 or M01/M05
        temp_dew_match = re.search(r'(M?\d{2})/(M?\d{2})', raw_metar)
        if temp_dew_match:
            t_str = temp_dew_match.group(1)
            d_str = temp_dew_match.group(2)
            
            temperature_c = int(t_str.replace('M', '-'))
            dewpoint_c = int(d_str.replace('M', '-'))

        # Extract wind
        # Format: 16008KT or 16008G15KT or VRB05KT
        wind_match = re.search(r'(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?KT', raw_metar)
        if wind_match:
            dir_str = wind_match.group(1)
            wind_direction = int(dir_str) if dir_str != 'VRB' else 0
            wind_speed_kts = int(wind_match.group(2))
            if wind_match.group(3):
                wind_gust_kts = int(wind_match.group(3))

        # Extract visibility
        # Format: 10SM or 1/2SM
        vis_match = re.search(r'(\d+(?:\s\d/\d)?SM)', raw_metar)
        if vis_match:
            vis_str = vis_match.group(1).replace('SM', '')
            if ' ' in vis_str:
                parts = vis_str.split(' ')
                visibility_sm = float(parts[0]) + (eval(parts[1]) if '/' in parts[1] else 0)
            elif '/' in vis_str:
                visibility_sm = eval(vis_str)
            else:
                visibility_sm = float(vis_str)

        # Extract altimeter
        # Format: A3012
        alt_match = re.search(r'A(\d{4})', raw_metar)
        if alt_match:
            altimeter_inhg = int(alt_match.group(1)) / 100.0

        # Cloud cover estimation and ceiling
        ceiling_ft = 10000
        if 'SKC' in raw_metar or 'CLR' in raw_metar or 'NCD' in raw_metar:
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

        # Extract ceiling (lowest BKN or OVC layer)
        ceiling_match = re.search(r'(BKN|OVC)(\d{3})', raw_metar)
        if ceiling_match:
            ceiling_ft = int(ceiling_match.group(2)) * 100

        # Humidity calculation (August-Roche-Magnus)
        import math
        es = 6.112 * math.exp((17.67 * temperature_c) / (temperature_c + 243.5))
        e = 6.112 * math.exp((17.67 * dewpoint_c) / (dewpoint_c + 243.5))
        humidity = min(100, max(0, int(100 * (e / es))))

        # Flight category
        if visibility_sm > 5 and ceiling_ft > 3000:
            flight_category = "VFR"
        elif visibility_sm >= 3 and ceiling_ft >= 1000:
            flight_category = "MVFR"
        elif visibility_sm >= 1 and ceiling_ft >= 500:
            flight_category = "IFR"
        else:
            flight_category = "LIFR"

    except Exception as e:
        print(f"Error parsing METAR: {e}")
        humidity = 50
        flight_category = "VFR"

    return {
        'temperature_c': temperature_c,
        'temperature_f': round(temperature_c * 9 / 5 + 32, 1),
        'dewpoint_c': dewpoint_c,
        'dewpoint_f': round(dewpoint_c * 9 / 5 + 32, 1),
        'humidity': humidity,
        'wind_speed_kts': wind_speed_kts,
        'wind_gust_kts': wind_gust_kts,
        'wind_direction': wind_direction,
        'visibility_sm': visibility_sm,
        'altimeter_inhg': altimeter_inhg,
        'cloud_cover': cloud_cover,
        'flight_category': flight_category,
        'raw': raw_metar
    }

def fetch_forecast(location: str):
    """Fetch 7-day forecast (daily + hourly) from Open-Meteo."""
    increment_metric("api_calls")
    coords = {
        'Starbase': (25.997, -97.156),
        'Vandy': (34.742, -120.572),
        'Cape': (28.483, -80.577),
        'Hawthorne': (33.921, -118.330)
    }
    lat, lon = coords.get(location, (25.997, -97.156))
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,weathercode&hourly=temperature_2m,windspeed_10m,winddirection_10m&timezone=auto"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching forecast for {location}: {e}")
        return None

def fetch_weather(location: str):
    """Fetch METAR weather data for a given location."""
    increment_metric("api_calls")
    metar_stations = {
        'Starbase': 'KBRO',
        'Vandy': 'KVBG',
        'Cape': 'KMLB',
        'Hawthorne': 'KHHR'
    }
    station_id = metar_stations.get(location, 'KBRO')
    
    # Try to fetch live wind from NWS API for higher frequency (often includes SPECI or more frequent updates)
    live_wind = {}
    try:
        # User-Agent is required for NWS API
        nws_headers = {'User-Agent': '(my-launch-app, contact@example.com)'}
        nws_url = f"https://api.weather.gov/stations/{station_id}/observations/latest"
        nws_res = requests.get(nws_url, headers=nws_headers, timeout=5)
        if nws_res.status_code == 200:
            nws_data = nws_res.json().get('properties', {})
            wind_speed = nws_data.get('windSpeed', {}).get('value') # km/h
            wind_gust = nws_data.get('windGust', {}).get('value')   # km/h
            wind_dir = nws_data.get('windDirection', {}).get('value') # degrees
            
            if wind_speed is not None:
                # NWS API returns speed in km/h (wmoUnit:km_h-1)
                live_wind['speed_kts'] = round(wind_speed * 0.539957, 1)
            if wind_gust is not None:
                live_wind['gust_kts'] = round(wind_gust * 0.539957, 1)
            if wind_dir is not None:
                live_wind['direction'] = int(wind_dir)
            
            live_wind['timestamp'] = nws_data.get('timestamp')
            live_wind['source'] = 'NWS Real-time'
    except Exception as e:
        print(f"Error fetching NWS live wind for {location}: {e}")

    url = f"https://aviationweather.gov/api/data/metar?ids={station_id}&format=raw"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        raw_metar = response.text.strip()
        if not raw_metar:
            raise ValueError("Empty METAR response")
        
        parsed = parse_metar(raw_metar)
        if live_wind:
            parsed['live_wind'] = live_wind
        return parsed
    except Exception as e:
        print(f"Error fetching weather for {location}: {e}")
        return {
            'temperature_c': 25, 'temperature_f': 77,
            'dewpoint_c': 15, 'dewpoint_f': 59,
            'humidity': 50,
            'wind_speed_kts': 0, 'wind_gust_kts': 0,
            'wind_direction': 0, 'visibility_sm': 10,
            'altimeter_inhg': 29.92, 'cloud_cover': 0,
            'flight_category': 'VFR', 'error': str(e)
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

# --- Background Refresh Helpers ---

def refresh_narratives_internal():
    """Internal helper to refresh narratives cache."""
    print("Refreshing narratives cache...")
    cached_narratives = get_cached_data(CACHE_KEY)
    if not cached_narratives:
        cached_narratives = _local_cache["launch_narratives"]

    try:
        descriptions = generate_narratives(existing_narratives=cached_narratives)
        current_time = datetime.now(timezone.utc)
        set_cached_data(CACHE_KEY, descriptions)
        set_cached_data(CACHE_TIME_KEY, current_time.isoformat())
        
        _local_cache["launch_narratives"] = descriptions
        _local_cache["last_updated"] = current_time
        return descriptions
    except Exception as e:
        print(f"Error in refresh_narratives_internal: {e}")
        return cached_narratives

def refresh_launches_internal():
    """Internal helper to refresh launches cache."""
    print("Refreshing launches cache...")
    cache_key = "launches_cache_v2"
    existing_previous = None
    existing_upcoming = None
    
    cached_data = get_cached_data(cache_key)
    if cached_data:
        existing_previous = cached_data.get('previous')
        existing_upcoming = cached_data.get('upcoming')

    try:
        data = fetch_launches(existing_previous=existing_previous, existing_upcoming=existing_upcoming)
        
        # Add trajectory data to the first upcoming launch
        upcoming = data.get("upcoming", [])
        previous = data.get("previous", [])
        if upcoming:
            traj = get_launch_trajectory_data(upcoming[0], previous)
            if traj:
                upcoming[0]['trajectory_data'] = traj
        elif previous:
            # Fallback to most recent previous launch if no upcoming
            traj = get_launch_trajectory_data(previous[0], previous)
            if traj:
                previous[0]['trajectory_data'] = traj

        last_updated = datetime.now(timezone.utc).isoformat()
        result = {
            "upcoming": upcoming,
            "previous": previous,
            "last_updated": last_updated
        }
        set_cached_data(cache_key, result)
        return result
    except Exception as e:
        print(f"Error in refresh_launches_internal: {e}")
        return None

def refresh_weather_internal():
    """Internal helper to refresh all weather cache."""
    print("Refreshing weather cache...")
    locations = ['Starbase', 'Vandy', 'Cape', 'Hawthorne']
    weather_results = {}
    timestamps = []
    
    for loc in locations:
        data = fetch_weather(loc)
        forecast = fetch_forecast(loc)
        data['forecast'] = forecast
        last_updated = datetime.now(timezone.utc).isoformat()
        data['last_updated'] = last_updated
        weather_results[loc] = data
        timestamps.append(last_updated)
        
        # Update individual cache
        if r:
            try:
                r.setex(f"weather_cache_v2_{loc}", 300, json.dumps(data))
            except: pass
            
    return {
        "weather": weather_results,
        "last_updated": min(timestamps) if timestamps else datetime.now(timezone.utc).isoformat()
    }

# --- New Endpoints ---

@app.get("/launches")
def get_launches(force: bool = False, internal: bool = False):
    if not internal:
        increment_metric("total_requests")
    cache_key = "launches_cache_v2"
    if force:
        increment_metric("cache_misses")
        return refresh_launches_internal()

    cached = get_cached_data(cache_key)
    if cached:
        increment_metric("cache_hits")
        return cached
    
    increment_metric("cache_misses")
    # Return empty if not in cache and not forcing (timer will populate it)
    return {"upcoming": [], "previous": [], "last_updated": None}

@app.get("/launches_slim")
def get_launches_slim(force: bool = False, internal: bool = False):
    """Dashboard-optimized endpoint that strips 'all_data'."""
    if not internal:
        increment_metric("total_requests")
    cache_key = "launches_cache_v2"
    
    data = None
    if force:
        increment_metric("cache_misses")
        data = refresh_launches_internal()
    else:
        cached = get_cached_data(cache_key)
        if cached:
            increment_metric("cache_hits")
            data = cached
        else:
            increment_metric("cache_misses")
            return {"upcoming": [], "previous": [], "last_updated": None}
    
    if data:
        def strip_bloat(l):
            return {k: v for k, v in l.items() if k != 'all_data'}
            
        return {
            "upcoming": [strip_bloat(l) for l in data.get("upcoming", [])],
            "previous": [strip_bloat(l) for l in data.get("previous", [])],
            "last_updated": data.get("last_updated")
        }
    return {"upcoming": [], "previous": [], "last_updated": None}

@app.get("/launch_raw/{launch_id}")
def get_launch_raw(launch_id: str, internal: bool = False):
    """Fetch the raw 'all_data' for a specific launch from the cache."""
    if not internal:
        increment_metric("total_requests")
    cache_key = "launches_cache_v2"
    cached = get_cached_data(cache_key)
    if not cached:
        return {"error": "Cache empty"}
    
    # Search in both upcoming and previous
    for l in cached.get('upcoming', []) + cached.get('previous', []):
        if l.get('id') == launch_id:
            increment_metric("cache_hits")
            return l.get('all_data', {})
            
    increment_metric("cache_misses")
    return {"error": "Launch not found"}

def _get_weather_cached(location: str, force: bool = False):
    """Internal helper to fetch weather with v2 caching metadata."""
    cache_key = f"weather_cache_v2_{location}"
    if r and not force:
        try:
            cached = r.get(cache_key)
            if cached:
                return json.loads(cached), True
        except: pass
    
    if force:
        data = fetch_weather(location)
        last_updated = datetime.now(timezone.utc).isoformat()
        data['last_updated'] = last_updated
        if r:
            try:
                r.setex(cache_key, 300, json.dumps(data))
            except: pass
        return data, False

    # Return default empty if not in cache (timer will populate)
    return {
        "temperature_c": 25, "temperature_f": 77, 
        "dewpoint_c": 15, "dewpoint_f": 59,
        "humidity": 50, "wind_speed_kts": 0, 
        "flight_category": "VFR", "last_updated": None
    }, False

@app.get("/weather/{location}")
def get_weather(location: str, force: bool = False, internal: bool = False):
    if not internal:
        increment_metric("total_requests")
    res, is_hit = _get_weather_cached(location, force)
    if is_hit:
        increment_metric("cache_hits")
    else:
        increment_metric("cache_misses")
    return res

@app.get("/weather_all")
def get_all_weather(force: bool = False, internal: bool = False):
    if not internal:
        increment_metric("total_requests")
    if force:
        increment_metric("cache_misses")
        return refresh_weather_internal()

    locations = ['Starbase', 'Vandy', 'Cape', 'Hawthorne']
    weather_results = {}
    timestamps = []
    
    hit_count = 0
    for loc in locations:
        res, is_hit = _get_weather_cached(loc, force)
        weather_results[loc] = res
        if res.get('last_updated'):
            timestamps.append(res.get('last_updated'))
        if is_hit:
            hit_count += 1
            
    if hit_count == len(locations):
        increment_metric("cache_hits")
    else:
        increment_metric("cache_misses")
        
    return {
        "weather": weather_results, 
        "last_updated": min(timestamps) if timestamps else None
    }

@app.get("/launch_details/{launch_id}")
def get_launch_details(launch_id: str, internal: bool = False):
    if not internal:
        increment_metric("total_requests")
    increment_metric("cache_misses") # Detailed fetch is always a direct API call/miss in this impl
    return fetch_launch_details(launch_id)

@app.get("/external_narratives")
def get_all_narratives(internal: bool = False):
    if not internal:
        increment_metric("total_requests")
    increment_metric("cache_misses") # This endpoint always fetches from external source
    return {"descriptions": fetch_external_narratives()}

@app.get("/recent_launches_narratives")
def get_narratives(force: bool = False, internal: bool = False):
    """Serve from cache only (timer-based refresh)."""
    if not internal:
        increment_metric("total_requests")
    
    if force:
        increment_metric("cache_misses")
        descriptions = refresh_narratives_internal()
        return {"descriptions": descriptions, "last_updated": datetime.now(timezone.utc).isoformat()}

    # Try to get from Redis
    data = get_cached_data(CACHE_KEY)
    time_str = get_cached_data(CACHE_TIME_KEY)
    
    if data and time_str:
        increment_metric("cache_hits")
        return {"descriptions": data, "last_updated": time_str}
    
    # Fallback to in-memory
    cached_narratives = _local_cache["launch_narratives"]
    last_updated = _local_cache["last_updated"]

    if cached_narratives and last_updated:
        increment_metric("cache_hits")
        return {"descriptions": cached_narratives, "last_updated": last_updated.isoformat()}
    
    increment_metric("cache_misses")
    return {"descriptions": [], "last_updated": None}

@app.get("/metrics")
def get_app_metrics(range: str = "1h", internal: bool = False):
    """Endpoint to fetch application metrics."""
    if not internal:
        record_snapshot()
    return get_metrics(range_type=range)

@app.post("/reset_metrics")
def reset_app_metrics():
    """Endpoint to reset all metrics and history."""
    global _local_metrics, _local_metrics_history, _last_snapshot_time
    
    # Reset in-memory
    _local_metrics = {
        "total_requests": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "api_calls": 0
    }
    _local_metrics_history = []
    _last_snapshot_time = 0
    
    # Reset Redis
    if r:
        try:
            r.delete(METRICS_KEY)
            r.delete(METRICS_HISTORY_KEY)
        except Exception as e:
            print(f"Redis error in reset_app_metrics: {e}")
            return {"status": "Error", "message": str(e)}
            
    return {"status": "Success", "message": "All metrics and history have been reset."}

@app.get("/seed_status")
def get_seed_status():
    """Endpoint to check the status of historical seeding."""
    if r:
        try:
            data = r.get(SEEDING_STATUS_KEY)
            if data:
                return json.loads(data)
        except Exception as e:
            print(f"Redis error in get_seed_status: {e}")
    return _local_seeding_status

@app.post("/seed_history")
def trigger_seed_history():
    """Endpoint to manually trigger historical launch seeding."""
    status = get_seed_status()
    if status.get("is_running"):
        return {"status": "Seeding already in progress"}
    
    # Clear stop signal if it exists
    global _stop_seeding_requested
    _stop_seeding_requested = False
    if r:
        try:
            r.delete(SEEDING_STOP_SIGNAL_KEY)
        except: pass
    
    # Start seeding in a background thread
    seeding_thread = threading.Thread(target=seed_historical_launches, daemon=True)
    seeding_thread.start()
    return {"status": "Historical seeding started"}

@app.post("/stop_seeding")
def stop_seeding():
    """Endpoint to stop the historical launch seeding."""
    global _stop_seeding_requested
    _stop_seeding_requested = True
    if r:
        try:
            r.set(SEEDING_STOP_SIGNAL_KEY, "true")
        except: pass
    return {"status": "Stop signal sent"}

def start_background_worker():
    def run():
        # Wait a bit for the app to start
        time.sleep(5)
        
        # Initial bootstrap (populate empty caches)
        print("Starting background worker bootstrap...")
        try:
            reset_stuck_seeding()
            refresh_narratives_internal()
            refresh_launches_internal()
            refresh_weather_internal()
            
        except Exception as e:
            print(f"Bootstrap error: {e}")

        last_run = {
            "narratives": time.time(),
            "launches": time.time(),
            "weather": time.time()
        }
        
        while True:
            try:
                now = time.time()
                
                # Metrics (every 30s)
                record_snapshot()
                
                # Narratives (every 15m)
                if now - last_run["narratives"] >= 900:
                    refresh_narratives_internal()
                    last_run["narratives"] = now
                
                # Launches (every 10m)
                if now - last_run["launches"] >= 600:
                    refresh_launches_internal()
                    last_run["launches"] = now
                    
                # Weather (every 2m for higher frequency wind updates)
                if now - last_run["weather"] >= 120:
                    refresh_weather_internal()
                    last_run["weather"] = now
                    
            except Exception as e:
                print(f"Background worker error: {e}")
            
            time.sleep(30) # Loop interval
            
    thread = threading.Thread(target=run, daemon=True)
    thread.start()

start_background_worker()

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    """Serve the dashboard UI."""
    return r"""
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
        .forecast-day:hover { background-color: rgba(30, 41, 59, 0.5); }
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
                    <div class="flex items-center gap-2 px-3 py-1 bg-slate-950 border border-slate-800 rounded-lg">
                        <i data-lucide="globe" class="text-slate-500 w-4 h-4"></i>
                        <span id="utc-clock" class="text-xs font-mono font-bold text-slate-400">00:00:00 UTC</span>
                    </div>
                </div>
            </div>
        </div>
    </nav>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <!-- Metrics Header & Time Range -->
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
            <div>
                <h2 class="text-2xl font-bold tracking-tight">System Metrics</h2>
                <p class="text-slate-400 text-sm">Real-time performance and absolute API usage tracking.</p>
            </div>
            <div class="flex flex-col sm:flex-row gap-3">
                <div class="flex bg-slate-900 border border-slate-800 p-1 rounded-xl">
                    <button onclick="changeRange('1h')" id="range-1h" class="px-4 py-1.5 rounded-lg text-sm font-medium transition-all range-active">1h</button>
                    <button onclick="changeRange('24h')" id="range-24h" class="px-4 py-1.5 rounded-lg text-sm font-medium transition-all text-slate-400 hover:text-slate-200">24h</button>
                    <button onclick="changeRange('7d')" id="range-7d" class="px-4 py-1.5 rounded-lg text-sm font-medium transition-all text-slate-400 hover:text-slate-200">7d</button>
                    <button onclick="changeRange('30d')" id="range-30d" class="px-4 py-1.5 rounded-lg text-sm font-medium transition-all text-slate-400 hover:text-slate-200">30d</button>
                </div>
                <button onclick="resetMetrics()" class="flex items-center justify-center gap-2 px-4 py-1.5 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 rounded-xl text-red-400 text-sm font-bold transition-all" title="Reset all metrics and history">
                    <i data-lucide="trash-2" class="w-4 h-4"></i>
                    Reset
                </button>
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
                <div class="flex flex-col">
                    <div class="flex items-center gap-2">
                        <i data-lucide="list" class="text-blue-500 w-4 h-4"></i>
                        <span class="text-[10px] font-bold uppercase tracking-wider text-slate-500">Narrative Refresh</span>
                    </div>
                    <span class="text-[9px] text-slate-600 mt-1 font-medium">Last: <span id="last-ref-narratives" class="text-slate-400">--:--:--</span></span>
                </div>
                <span id="timer-narratives" class="text-sm font-mono font-bold text-blue-400">00:00</span>
             </div>
             <div class="bg-slate-900/30 border border-slate-800/50 p-3 rounded-xl flex items-center justify-between">
                <div class="flex flex-col">
                    <div class="flex items-center gap-2">
                        <i data-lucide="rocket" class="text-emerald-500 w-4 h-4"></i>
                        <span class="text-[10px] font-bold uppercase tracking-wider text-slate-500">Launch Refresh</span>
                    </div>
                    <span class="text-[9px] text-slate-600 mt-1 font-medium">Last: <span id="last-ref-launches" class="text-slate-400">--:--:--</span></span>
                </div>
                <span id="timer-launches" class="text-sm font-mono font-bold text-emerald-400">00:00</span>
             </div>
             <div class="bg-slate-900/30 border border-slate-800/50 p-3 rounded-xl flex items-center justify-between">
                <div class="flex flex-col">
                    <div class="flex items-center gap-2">
                        <i data-lucide="cloud-sun" class="text-blue-400 w-4 h-4"></i>
                        <span class="text-[10px] font-bold uppercase tracking-wider text-slate-500">Weather Refresh</span>
                    </div>
                    <span class="text-[9px] text-slate-600 mt-1 font-medium">Last: <span id="last-ref-weather" class="text-slate-400">--:--:--</span></span>
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
            <!-- Historical Seeding Control -->
            <div class="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden p-6 mb-8">
                <div class="flex flex-col md:flex-row items-center justify-between gap-6">
                    <div class="flex items-start gap-4">
                        <div class="p-3 bg-purple-500/10 rounded-xl">
                            <i data-lucide="history" class="w-6 h-6 text-purple-400"></i>
                        </div>
                        <div>
                            <h3 class="text-lg font-semibold flex items-center gap-2 mb-1">
                                Historical Data Seeding
                            </h3>
                            <p class="text-sm text-slate-400 max-w-md">Pull increasingly older launches from SpaceX history to populate the cache. This operation respects API rate limits and runs in the background.</p>
                        </div>
                    </div>
                    <div class="flex flex-col sm:flex-row items-center gap-6 w-full md:w-auto">
                        <div id="seeding-stats" class="bg-slate-800/50 px-4 py-2 rounded-xl border border-slate-700/50 min-w-[200px]">
                            <div class="flex items-center justify-between mb-1">
                                <span class="text-[10px] font-bold uppercase tracking-wider text-slate-500">Seeding Stats</span>
                                <span id="seed-status-tag" class="text-[10px] font-bold uppercase px-1.5 py-0.5 rounded bg-slate-700 text-slate-400">Idle</span>
                            </div>
                            <div class="text-sm font-mono flex flex-col">
                                <span class="flex justify-between gap-4">Pulled: <span id="seed-count" class="text-white font-bold">0</span></span>
                                <span class="flex justify-between gap-4">Oldest: <span id="seed-oldest" class="text-white font-bold text-xs">--</span></span>
                            </div>
                        </div>
                        <div class="flex flex-col sm:flex-row gap-3 w-full sm:w-auto">
                            <button id="btn-seed-history" onclick="triggerSeeding()" class="w-full sm:w-auto px-6 py-3 bg-purple-600 hover:bg-purple-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-xl font-bold transition-all shadow-lg shadow-purple-900/20 flex items-center justify-center gap-2 whitespace-nowrap">
                                <i data-lucide="database-zap" class="w-5 h-5"></i>
                                Seed History
                            </button>
                            <button id="btn-stop-seeding" onclick="stopSeeding()" class="hidden w-full sm:w-auto px-6 py-3 bg-red-600 hover:bg-red-500 text-white rounded-xl font-bold transition-all shadow-lg shadow-red-900/20 flex items-center justify-center gap-2 whitespace-nowrap">
                                <i data-lucide="square" class="w-5 h-5"></i>
                                Stop Seeding
                            </button>
                        </div>
                    </div>
                </div>
            </div>

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

    <!-- Weather Hourly Modal -->
    <div id="weather-modal" class="fixed inset-0 z-[60] hidden bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
        <div class="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-2xl max-h-[80vh] flex flex-col shadow-2xl">
            <div class="p-6 border-b border-slate-800 flex items-center justify-between">
                <div>
                    <h3 id="modal-title" class="text-xl font-bold text-white">Hourly Forecast</h3>
                    <p id="modal-subtitle" class="text-sm text-slate-400">Loading...</p>
                </div>
                <button onclick="closeWeatherModal()" class="p-2 hover:bg-slate-800 rounded-lg transition-colors">
                    <i data-lucide="x" class="w-6 h-6 text-slate-400"></i>
                </button>
            </div>
            <div id="modal-content" class="p-6 overflow-y-auto grid grid-cols-4 sm:grid-cols-6 gap-4">
                <!-- Hourly data will be injected here -->
            </div>
        </div>
    </div>

    <script>
        lucide.createIcons();

        // Chart instances
        const charts = {};
        let currentRange = '1h';

        // Launch data management for lazy loading and pagination
        let allPreviousLaunches = [];
        let displayedPreviousCount = 20;

        async function toggleLaunch(id) {
            const detailsEl = document.getElementById('details-' + id);
            if (!detailsEl) return;
            
            const isExpanding = detailsEl.classList.contains('hidden');
            
            // Toggle visibility
            detailsEl.classList.toggle('hidden');
            
            if (isExpanding) {
                const contentEl = document.getElementById('details-content-' + id);
                if (contentEl && contentEl.getAttribute('data-loaded') !== 'true') {
                    await fetchLaunchDetailsOnDemand(id);
                }
                lucide.createIcons();
            }
        }

        async function fetchLaunchDetailsOnDemand(id) {
            const contentEl = document.getElementById('details-content-' + id);
            const rawEl = document.getElementById('raw-content-' + id);
            
            if (!contentEl) return;
            
            try {
                const response = await fetch('/launch_raw/' + id + '?internal=true');
                const rawData = await response.json();
                
                if (rawData.error) throw new Error(rawData.error);
                
                contentEl.innerHTML = renderDataCards(rawData);
                contentEl.setAttribute('data-loaded', 'true');
                if (rawEl) rawEl.textContent = JSON.stringify(rawData, null, 2);
            } catch (error) {
                console.error('Error fetching launch raw data:', error);
                contentEl.innerHTML = `<div class="p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-xs flex items-center gap-2">
                    <i data-lucide="alert-circle" class="w-4 h-4"></i>
                    Failed to load detailed mission data: ${error.message}
                </div>`;
                lucide.createIcons();
            }
        }

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

        function syncTimer(category, lastUpdatedIso) {
            if (!lastUpdatedIso) return;
            const lastUpdatedDate = new Date(lastUpdatedIso);
            const lastUpdated = lastUpdatedDate.getTime();
            const interval = refreshIntervals[category] * 1000;
            
            // Update the next refresh target based on when the backend last updated
            // Robustness: Ensure we don't set a target in the past, which causes infinite refresh loops
            const target = lastUpdated + interval;
            const now = Date.now();
            
            if (target <= now + 5000) { // If expired or expiring in next 5s
                // Backend is lagging. Set next refresh to 30s from now to avoid spamming
                nextRefresh[category] = now + 30000;
            } else {
                nextRefresh[category] = target;
            }

            // Update the "Last Refreshed" display in the timer cards (in UTC to match clock)
            const lastRefEl = document.getElementById(`last-ref-${category}`);
            if (lastRefEl) {
                const utcStr = lastUpdatedDate.getUTCHours().toString().padStart(2, '0') + ':' + 
                               lastUpdatedDate.getUTCMinutes().toString().padStart(2, '0') + ':' + 
                               lastUpdatedDate.getUTCSeconds().toString().padStart(2, '0');
                lastRefEl.textContent = utcStr + ' UTC';
            }
        }

        function updateTimers() {
            const now = Date.now();
            
            // Update UTC Clock
            const nowDate = new Date();
            const utcString = nowDate.getUTCHours().toString().padStart(2, '0') + ':' + 
                             nowDate.getUTCMinutes().toString().padStart(2, '0') + ':' + 
                             nowDate.getUTCSeconds().toString().padStart(2, '0');
            const utcClockEl = document.getElementById('utc-clock');
            if (utcClockEl) {
                utcClockEl.textContent = utcString + ' UTC';
            }

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
            ['1h', '24h', '7d', '30d'].forEach(r => {
                const btn = document.getElementById(`range-${r}`);
                if (btn) {
                    if (r === range) {
                        btn.classList.add('range-active');
                        btn.classList.remove('text-slate-400', 'hover:text-slate-200');
                        btn.classList.add('text-white');
                    } else {
                        btn.classList.remove('range-active', 'text-white');
                        btn.classList.add('text-slate-400', 'hover:text-slate-200');
                    }
                }
            });
            fetchMetrics();
            // Reset metrics timer
            nextRefresh.metrics = Date.now() + refreshIntervals.metrics * 1000;
        }

        async function resetMetrics() {
            if (!confirm('Are you sure you want to reset all metrics and history? This cannot be undone.')) {
                return;
            }
            
            try {
                const response = await fetch('/reset_metrics', { method: 'POST' });
                const result = await response.json();
                if (result.status === 'Success') {
                    // Force refresh metrics immediately
                    fetchMetrics();
                    alert('Metrics have been reset successfully.');
                } else {
                    alert('Error: ' + result.message);
                }
            } catch (error) {
                console.error('Error resetting metrics:', error);
                alert('Failed to reset metrics. See console for details.');
            }
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
                const response = await fetch(`/metrics?range=${currentRange}&internal=true`);
                const data = await response.json();
                
                const current = data.current || {};
                const rangeStats = data.range_stats || {};
                const history = data.history || [];
        
                // Use live absolute totals for the main metrics
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
                        const values = [];
                        for (let i = 0; i < history.length; i++) {
                            if (isEfficiency) {
                                const t = history[i].data.total_requests || 0;
                                values.push(t > 0 ? (history[i].data.cache_hits / t) * 100 : 0);
                            } else {
                                if (i === 0) {
                                    values.push(0);
                                } else {
                                    const diff = (history[i].data[key] || 0) - (history[i-1].data[key] || 0);
                                    values.push(Math.max(0, diff));
                                }
                            }
                        }
                        
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
                const isImage = /\.(jpg|jpeg|png|gif|webp|bmp|svg)($|\?)/i.test(val);
                if (isImage) {
                    return `
                        <div class="flex flex-col gap-2">
                            <a href="${val}" target="_blank" class="text-blue-400 hover:underline truncate block max-w-full text-[10px]">${val}</a>
                            <img src="${val}" class="max-w-full h-auto rounded-lg border border-slate-800 shadow-sm mt-1" onerror="this.style.display='none'">
                        </div>
                    `;
                }
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
                const url = force ? '/recent_launches_narratives?force=true&internal=true' : '/recent_launches_narratives?internal=true';
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
                
                if (data.last_updated) {
                    syncTimer('narratives', data.last_updated);
                    const luDate = new Date(data.last_updated);
                    const luUtc = luDate.getUTCHours().toString().padStart(2, '0') + ':' + 
                                 luDate.getUTCMinutes().toString().padStart(2, '0') + ':' + 
                                 luDate.getUTCSeconds().toString().padStart(2, '0') + ' UTC';
                    document.getElementById('last-updated').textContent = 'Last updated: ' + luUtc;
                }
            } catch (error) {
                console.error('Error fetching narratives:', error);
            }
        }

        async function fetchLaunches(force = false) {
            const upList = document.getElementById('upcoming-launches-list');
            
            try {
                const url = force ? '/launches_slim?force=true&internal=true' : '/launches_slim?internal=true';
                const response = await fetch(url);
                const data = await response.json();
                
                allPreviousLaunches = data.previous || [];
                
                renderList(upList, data.upcoming);
                renderPreviousList(false);
                
                if (data.last_updated) {
                    syncTimer('launches', data.last_updated);
                }
            } catch (error) {
                console.error('Error fetching launches:', error);
                const upList = document.getElementById('upcoming-launches-list');
                const prevList = document.getElementById('previous-launches-list');
                if (upList) upList.innerHTML = '<p class="text-red-400 text-center py-4">Error loading upcoming launches.</p>';
                if (prevList) prevList.innerHTML = '<p class="text-red-400 text-center py-4">Error loading previous launches.</p>';
            }
        }

        function renderList(el, launches, isHistorical = false) {
            if (!isHistorical) el.innerHTML = '';
            if (!launches || launches.length === 0) {
                if (!isHistorical) el.innerHTML = '<p class="text-slate-500 text-center py-4">No launches found.</p>';
                return;
            }
            launches.forEach(l => {
                const div = document.createElement('div');
                div.className = 'launch-card border-b border-slate-800 last:border-0 hover:bg-slate-800/10 transition-all';
                
                const id = 'details-' + l.id;
                const rawId = 'raw-' + l.id;
                
                div.innerHTML = `
                    <div class="p-4 cursor-pointer" onclick="toggleLaunch('${l.id}')">
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
                        ${(l.image && typeof l.image === 'string') ? `<img src="${l.image}" class="w-full h-48 object-cover rounded-xl border border-slate-700 shadow-lg" onerror="this.style.display='none'">` : ''}
                        
                        ${l.description ? `<p class="text-sm text-slate-300 leading-relaxed bg-slate-950/50 p-4 rounded-xl border border-slate-800">${l.description}</p>` : ''}
                        
                        <div class="space-y-4">
                            <h3 class="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-2 border-b border-slate-800 pb-1">Detailed Mission Data</h3>
                            <div id="details-content-${l.id}" class="min-h-[50px] flex flex-col justify-center">
                                <div class="animate-pulse flex space-x-2">
                                    <div class="h-4 bg-slate-800 rounded w-full"></div>
                                </div>
                                <span class="text-[10px] text-slate-600 italic mt-2 text-center">Loading detailed mission data...</span>
                            </div>
                        </div>

                        <div class="flex flex-wrap gap-3">
                            ${(l.video_url && typeof l.video_url === 'string') ? `
                            <a href="${l.video_url}" target="_blank" class="flex items-center gap-2 px-3 py-1.5 bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded-lg transition-colors border border-red-500/20 text-xs font-semibold">
                                <i data-lucide="play-circle" class="w-4 h-4"></i> Webcast
                            </a>` : ''}
                            ${(l.x_video_url && typeof l.x_video_url === 'string') ? `
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
                            <pre id="raw-content-${l.id}" class="bg-slate-950 p-4 rounded-xl border border-slate-800 text-[10px] text-slate-400 overflow-x-auto font-mono max-h-60">Loading JSON...</pre>
                        </div>
                    </div>
                `;
                el.appendChild(div);
            });
            lucide.createIcons();
        }

        function renderPreviousList(append = false) {
            const el = document.getElementById('previous-launches-list');
            if (!append) {
                el.innerHTML = '';
                displayedPreviousCount = 0;
            }
            
            const existingBtn = document.getElementById('btn-load-more');
            if (existingBtn) existingBtn.remove();

            const start = displayedPreviousCount;
            const end = Math.min(start + (append ? 50 : 20), allPreviousLaunches.length);
            const launches = allPreviousLaunches.slice(start, end);
            
            renderList(el, launches, true);
            displayedPreviousCount = end;
            
            if (displayedPreviousCount < allPreviousLaunches.length) {
                const btn = document.createElement('button');
                btn.id = 'btn-load-more';
                btn.className = 'w-full py-8 text-slate-400 hover:text-white font-bold transition-all border-t border-slate-800 bg-slate-900/10 hover:bg-slate-900/30 flex items-center justify-center gap-2';
                btn.innerHTML = `<i data-lucide="plus-circle" class="w-5 h-5"></i> Load More (${allPreviousLaunches.length - displayedPreviousCount} remaining)`;
                btn.onclick = () => {
                    renderPreviousList(true);
                };
                el.appendChild(btn);
                lucide.createIcons();
            }
        }

        function renderWeatherCard(card, loc, data) {
            const getFlightCategoryColor = (cat) => {
                switch(cat) {
                    case 'VFR': return 'text-emerald-400';
                    case 'MVFR': return 'text-blue-400';
                    case 'IFR': return 'text-yellow-400';
                    case 'LIFR': return 'text-red-400';
                    default: return 'text-slate-400';
                }
            };

            const getWeatherIcon = (code) => {
                // WMO Weather interpretation codes (WW)
                if (code === 0) return 'sun';
                if (code <= 3) return 'cloud-sun';
                if (code <= 48) return 'cloud';
                if (code <= 67) return 'cloud-rain';
                if (code <= 77) return 'cloud-snow';
                if (code <= 82) return 'cloud-rain';
                if (code <= 86) return 'cloud-snow';
                if (code <= 99) return 'cloud-lightning';
                return 'cloud-sun';
            };

            let forecastHtml = '';
            if (data.forecast && data.forecast.daily) {
                forecastHtml = `
                    <div class="mt-4 pt-4 border-t border-slate-800">
                        <p class="text-[10px] font-bold uppercase tracking-wider text-slate-500 mb-3">7-Day Forecast</p>
                        <div class="grid grid-cols-7 gap-1">
                            ${data.forecast.daily.time.map((time, i) => {
                                const date = new Date(time + 'T00:00:00');
                                const dayName = date.toLocaleDateString('en-US', { weekday: 'short' });
                                const maxTemp = Math.round(data.forecast.daily.temperature_2m_max[i]);
                                const minTemp = Math.round(data.forecast.daily.temperature_2m_min[i]);
                                const code = data.forecast.daily.weathercode[i];
                                return `
                                    <div class="flex flex-col items-center p-1.5 rounded-lg forecast-day cursor-pointer transition-colors" 
                                         onclick="showHourly('${loc}', '${time}', ${i})">
                                        <span class="text-[9px] text-slate-500 font-bold uppercase">${dayName}</span>
                                        <i data-lucide="${getWeatherIcon(code)}" class="w-4 h-4 my-1 text-blue-400"></i>
                                        <span class="text-[10px] font-bold">${maxTemp}°</span>
                                        <span class="text-[9px] text-slate-500">${minTemp}°</span>
                                    </div>
                                `;
                            }).join('')}
                        </div>
                    </div>
                `;
            }

            card.innerHTML = `
                <div class="flex items-center justify-between mb-4">
                    <div class="flex flex-col">
                        <h3 class="text-xl font-bold">${loc}</h3>
                        <span class="text-[10px] font-bold uppercase tracking-widest ${getFlightCategoryColor(data.flight_category)}">${data.flight_category || 'Unknown'}</span>
                    </div>
                    <i data-lucide="cloud-sun" class="text-blue-400 w-6 h-6"></i>
                </div>
                <div class="grid grid-cols-3 gap-y-4 gap-x-4">
                    <div>
                        <p class="text-[9px] text-slate-500 uppercase font-bold tracking-wider mb-1">Temp</p>
                        <p class="text-lg font-semibold">${Math.round(data.temperature_f)}°F <span class="text-[10px] text-slate-400 font-normal">/ ${Math.round(data.temperature_c)}°C</span></p>
                    </div>
                    <div>
                        <p class="text-[9px] text-slate-500 uppercase font-bold tracking-wider mb-1">Dewpoint</p>
                        <p class="text-lg font-semibold">${Math.round(data.dewpoint_f)}°F <span class="text-[10px] text-slate-400 font-normal">/ ${Math.round(data.dewpoint_c)}°C</span></p>
                    </div>
                    <div>
                        <p class="text-[9px] text-slate-500 uppercase font-bold tracking-wider mb-1">Humidity</p>
                        <p class="text-lg font-semibold">${data.humidity}%</p>
                    </div>
                    <div>
                        <p class="text-[9px] text-slate-500 uppercase font-bold tracking-wider mb-1">Wind</p>
                        <p class="text-lg font-semibold">${Math.round(data.live_wind && data.live_wind.speed_kts !== undefined ? data.live_wind.speed_kts : data.wind_speed_kts)}${ (data.live_wind && data.live_wind.gust_kts) ? `<span class="text-red-400 text-sm ml-1">G${Math.round(data.live_wind.gust_kts)}</span>` : (data.wind_gust_kts ? `<span class="text-red-400 text-sm ml-1">G${data.wind_gust_kts}</span>` : '')} <span class="text-[10px] text-slate-400 font-normal">kts</span></p>
                        ${data.live_wind ? `<p class="text-[8px] text-blue-500 font-bold mt-0.5"><i data-lucide="zap" class="w-2 h-2 inline-block"></i> LIVE</p>` : ''}
                    </div>
                    <div>
                        <p class="text-[9px] text-slate-500 uppercase font-bold tracking-wider mb-1">Direction</p>
                        <p class="text-lg font-semibold">${data.live_wind && data.live_wind.direction !== undefined ? data.live_wind.direction : data.wind_direction}°</p>
                    </div>
                    <div>
                        <p class="text-[9px] text-slate-500 uppercase font-bold tracking-wider mb-1">Visibility</p>
                        <p class="text-lg font-semibold">${data.visibility_sm} <span class="text-[10px] text-slate-400 font-normal">sm</span></p>
                    </div>
                    <div>
                        <p class="text-[9px] text-slate-500 uppercase font-bold tracking-wider mb-1">Pressure</p>
                        <p class="text-lg font-semibold">${data.altimeter_inhg ? data.altimeter_inhg.toFixed(2) : '29.92'} <span class="text-[10px] text-slate-400 font-normal">inHg</span></p>
                    </div>
                    <div>
                        <p class="text-[9px] text-slate-500 uppercase font-bold tracking-wider mb-1">Clouds</p>
                        <p class="text-lg font-semibold">${data.cloud_cover}%</p>
                    </div>
                </div>
                ${forecastHtml}
                <div class="mt-4 pt-3 border-t border-slate-800">
                    <p class="text-[9px] font-mono text-slate-600 truncate uppercase" title="${data.raw || ''}">${data.raw || 'No raw METAR data'}</p>
                </div>
            `;
        }

        let lastWeatherData = null;

        function showHourly(loc, dateStr, dayIndex) {
            if (!lastWeatherData || !lastWeatherData[loc] || !lastWeatherData[loc].forecast) return;
            const forecast = lastWeatherData[loc].forecast;
            const modal = document.getElementById('weather-modal');
            const content = document.getElementById('modal-content');
            const title = document.getElementById('modal-title');
            const subtitle = document.getElementById('modal-subtitle');

            const date = new Date(dateStr + 'T00:00:00');
            title.textContent = `${loc} - ${date.toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' })}`;
            subtitle.textContent = "Hourly Temperature (°C)";

            content.innerHTML = '';
            
            // Hourly data starts from index (dayIndex * 24)
            const startIndex = dayIndex * 24;
            for (let i = 0; i < 24; i++) {
                const idx = startIndex + i;
                if (idx >= forecast.hourly.time.length) break;
                
                const timeStr = forecast.hourly.time[idx];
                const hour = new Date(timeStr).getHours();
                const temp = Math.round(forecast.hourly.temperature_2m[idx]);
                const windSpeed = Math.round(forecast.hourly.windspeed_10m[idx]);
                const windDir = forecast.hourly.winddirection_10m[idx];
                
                const item = document.createElement('div');
                item.className = 'flex flex-col items-center p-3 bg-slate-800/50 rounded-xl border border-slate-700/50';
                item.innerHTML = `
                    <span class="text-[10px] font-bold text-slate-400 mb-1">${hour}:00</span>
                    <span class="text-sm font-bold text-white">${temp}°C</span>
                    <span class="text-[9px] text-slate-500 mt-1">${Math.round(temp * 9/5 + 32)}°F</span>
                    <div class="mt-2 pt-2 border-t border-slate-700/50 w-full flex flex-col items-center">
                         <span class="text-[10px] font-bold text-blue-400">${windSpeed} km/h</span>
                         <span class="text-[8px] text-slate-500 uppercase">${windDir}°</span>
                    </div>
                `;
                content.appendChild(item);
            }

            modal.classList.remove('hidden');
            document.body.style.overflow = 'hidden';
            lucide.createIcons();
        }

        function closeWeatherModal() {
            const modal = document.getElementById('weather-modal');
            modal.classList.add('hidden');
            document.body.style.overflow = 'auto';
        }

        async function fetchWeatherAll(force = false) {
            const container = document.getElementById('weather-grid');
            
            // Show loaders
            container.innerHTML = '';
            ['Starbase', 'Vandy', 'Cape', 'Hawthorne'].forEach(loc => {
                const card = document.createElement('div');
                card.className = 'bg-slate-900 border border-slate-800 p-6 rounded-2xl flex flex-col gap-4';
                card.innerHTML = `<div class="animate-pulse h-32 bg-slate-800 rounded"></div>`;
                container.appendChild(card);
            });
            
            try {
                const url = force ? `/weather_all?force=true&internal=true` : `/weather_all?internal=true`;
                const response = await fetch(url);
                const data = await response.json();
                lastWeatherData = data.weather;
                
                container.innerHTML = '';
                if (data.weather) {
                    Object.entries(data.weather).forEach(([loc, weatherData]) => {
                        const card = document.createElement('div');
                        card.className = 'bg-slate-900 border border-slate-800 p-6 rounded-2xl flex flex-col gap-4';
                        renderWeatherCard(card, loc, weatherData);
                        container.appendChild(card);
                    });
                    lucide.createIcons();
                }
                
                if (data.last_updated) {
                    syncTimer('weather', data.last_updated);
                }
            } catch (error) {
                container.innerHTML = '<p class="text-red-400 text-center py-4">Error loading weather.</p>';
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

        // --- Seeding Logic ---
        let seedingPollInterval = null;

        async function triggerSeeding() {
            const btn = document.getElementById('btn-seed-history');
            btn.disabled = true;
            btn.innerHTML = '<i data-lucide="loader-2" class="w-5 h-5 animate-spin"></i> Triggering...';
            lucide.createIcons();

            try {
                const response = await fetch('/seed_history', { method: 'POST' });
                const data = await response.json();
                console.log('Seeding response:', data);
                
                // Small delay to let the background thread start and update status
                setTimeout(startSeedingPoll, 1000);
            } catch (error) {
                console.error('Error triggering seeding:', error);
                btn.disabled = false;
                btn.innerHTML = '<i data-lucide="database-zap" class="w-5 h-5"></i> Seed History';
                lucide.createIcons();
            }
        }

        async function stopSeeding() {
            const btn = document.getElementById('btn-stop-seeding');
            btn.disabled = true;
            btn.innerHTML = '<i data-lucide="loader-2" class="w-5 h-5 animate-spin"></i> Stopping...';
            lucide.createIcons();

            try {
                const response = await fetch('/stop_seeding', { method: 'POST' });
                const data = await response.json();
                console.log('Stop response:', data);
            } catch (error) {
                console.error('Error stopping seeding:', error);
                btn.disabled = false;
                btn.innerHTML = '<i data-lucide="square" class="w-5 h-5"></i> Stop Seeding';
                lucide.createIcons();
            }
        }

        function startSeedingPoll() {
            if (seedingPollInterval) clearInterval(seedingPollInterval);
            pollSeedingStatus();
            seedingPollInterval = setInterval(pollSeedingStatus, 3000);
        }

        async function pollSeedingStatus() {
            try {
                const response = await fetch('/seed_status');
                const data = await response.json();
                
                const btn = document.getElementById('btn-seed-history');
                const stopBtn = document.getElementById('btn-stop-seeding');
                const tag = document.getElementById('seed-status-tag');
                const count = document.getElementById('seed-count');
                const oldest = document.getElementById('seed-oldest');

                count.textContent = (data.total_pulled || 0).toLocaleString();
                oldest.textContent = data.oldest_launch ? data.oldest_launch.split('T')[0] : '--';
                
                if (data.is_running) {
                    btn.disabled = true;
                    btn.innerHTML = '<i data-lucide="loader-2" class="w-5 h-5 animate-spin"></i> Seeding...';
                    stopBtn.classList.remove('hidden');
                    // Only reset stop button text if not already in "Stopping..." state
                    if (!stopBtn.disabled) {
                        stopBtn.innerHTML = '<i data-lucide="square" class="w-5 h-5"></i> Stop Seeding';
                    }
                    tag.textContent = 'Active';
                    tag.className = 'text-[10px] font-bold uppercase px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-400 animate-pulse';
                } else {
                    btn.disabled = false;
                    btn.innerHTML = '<i data-lucide="database-zap" class="w-5 h-5"></i> Seed History';
                    stopBtn.classList.add('hidden');
                    tag.textContent = data.last_status || 'Idle';
                    tag.className = 'text-[10px] font-bold uppercase px-1.5 py-0.5 rounded bg-slate-700 text-slate-400';
                    
                    // If it stopped and we are polling, we can stop if it's completed or hit limit
                    if (!data.is_running && seedingPollInterval && data.last_status !== 'Starting batch fetch...') {
                        // We keep it polling just in case, or stop it to save resources?
                        // Let's keep it but at a much slower rate? No, let's just stop if it's Idle.
                    }
                }
                lucide.createIcons();
            } catch (error) {
                console.error('Error polling seeding status:', error);
            }
        }

        // Initial load - Fetch all data once to bootstrap the UI
        fetchMetrics();
        fetchNarratives();
        fetchLaunches();
        fetchWeatherAll();
        startSeedingPoll();

        // Start the master timer loop (updates UI countdowns and triggers refreshes)
        setInterval(updateTimers, 1000);
    </script>
</body>
</html>
    """

@app.post("/refresh")
def refresh_cache():
    """Force refresh the cache (call this via scheduler)."""
    print("Manual cache refresh triggered via POST /refresh.")
    descriptions = refresh_narratives_internal()
    count = len(descriptions) if descriptions else 0
    return {
        "status": "Cache refreshed",
        "count": count,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))