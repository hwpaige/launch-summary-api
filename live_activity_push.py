"""ActivityKit push tokens and the end/update scheduler.

Launch Buddy requests each Live Activity with a push token and POSTs it
here. This process sends an APNs ``liveactivity`` ``event: end`` at the
flight milestone (later of SECO-1 and stage-1 landing, else T+9, never
past T+20) and on scrub/failure, plus an ``update`` when NET or that
milestone changes.

If the APNs env vars are missing the routes still store tokens and every
send is a no-op. Nothing in here reads a key file off disk.

Required Heroku config vars (names only; values stay on the dyno):

- ``APNS_AUTH_KEY`` — contents of the Apple .p8 auth key
- ``APNS_KEY_ID``
- ``APNS_TEAM_ID``
- ``APNS_LIVE_ACTIVITY_TOPIC`` — ``<bundle id>.push-type.liveactivity``

Optional: ``APNS_ENVIRONMENT`` ``production`` (default, TestFlight and
App Store) or ``sandbox`` (Xcode installs).
"""

from __future__ import annotations

import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Swift's default JSONEncoder writes Date as seconds since 2001-01-01 UTC.
# APNs `timestamp` / `stale-date` / `dismissal-date` are Unix seconds.
SWIFT_REFERENCE_UNIX = 978307200

HARD_CAP_SEC = 20 * 60
DEFAULT_END_SEC = 9 * 60
FINAL_LINGER_SEC = 2 * 60
PROMPT_LINGER_SEC = 60
LATE_DISMISS_SEC = 15
NET_CHANGE_SEC = 2
STORE_KEY = "live_activity_push_v1"
MAX_RECORDS = 40
DROP_AFTER_NET_SEC = 12 * 3600

_lock = threading.Lock()
_memory: dict = {}
_logged_disabled = False
_jwt_cache: dict = {"token": None, "exp": 0}

_SECO1 = re.compile(r"seco[\s-]*1\b", re.IGNORECASE)
