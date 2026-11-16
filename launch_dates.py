"""Parse launch NET strings for the Starship tracker. No disk cache."""
from __future__ import annotations

from datetime import datetime, timezone


def _get_parsed_dt(net_str):
    """Aware UTC datetime, or None when the string is missing or not ISO-8601."""
    if not net_str:
        return None
    text = str(net_str).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)
