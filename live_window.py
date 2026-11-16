"""Starship identification for the tracker and closure window."""
from __future__ import annotations


def is_starship_launch(launch):
    """True for a Starship flight (rocket name or mission title)."""
    if not isinstance(launch, dict):
        return False
    rocket = str(launch.get("rocket") or "")
    mission = str(launch.get("mission") or launch.get("name") or "")
    blob = f"{rocket} {mission}".lower()
    return "starship" in blob
