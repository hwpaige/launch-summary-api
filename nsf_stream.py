"""Flight number used on the tracker strip. The kiosk NSF player is not ported."""
from __future__ import annotations

import re

_FLIGHT = re.compile(r"flight\s*(\d+)", re.IGNORECASE)


def launch_flight_number(launch):
    """``15`` from a mission title such as ``Starship Flight 15``, else ``''``."""
    if not isinstance(launch, dict):
        return ""
    blob = " ".join(
        str(launch.get(key) or "") for key in ("mission", "name", "rocket")
    )
    match = _FLIGHT.search(blob)
    return match.group(1) if match else ""
