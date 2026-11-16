"""On-disk cache directory for the ported Starship poller.

The dashboard keeps this under the kiosk home directory. On the Heroku API
the log is local to the dyno; a restart backfills the trail from history.
"""
from __future__ import annotations

import os

CACHE_DIR = os.environ.get("STARSHIP_CACHE_DIR") or os.path.join(
    "/tmp", "launch-narrative-starship"
)
os.makedirs(CACHE_DIR, exist_ok=True)
