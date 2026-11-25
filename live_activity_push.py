"""ActivityKit push tokens and the end/update scheduler.

Launch Buddy requests each Live Activity with a push token and POSTs it
here. This process sends an APNs ``liveactivity`` ``event: end`` at the
flight milestone (later of the last ascent SECO and the last
first-stage recovery, else T+9, never past T+80) once Launch Library
shows the vehicle flew, and promptly on scrub/failure. Ascent SECO is
SECO-1, or SECO-2 when that cutoff is at or before T+75; SECO-3 and
later burns are ignored. A hold is not terminal. Every end is
re-checked against a fresh Launch Library read before it is sent, so a
recycled T0 reschedules instead of ending early. An ``update`` goes out
when NET or that milestone changes, and when a hold starts or clears. A
countdown that is on screen and whose NET moves more than four hours away
(a scrub to another day) ends with ``Delayed``. The content state carries
``milestoneSource`` (for example ``SECO-2``) with the milestone time.

From T-1h until the flight ends the loop reads the launch from the short
single-flight copy (90s near liftoff, else 180s) instead of the list
refresh, which is ten minutes apart. A slip called in the last minutes
used to arrive after the old T0. The cadence is set by Launch Library's
hourly request limit; see ``HOT_RAW_TTL`` in ``app.py``.

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

import json
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

HARD_CAP_SEC = 80 * 60
# SECO-2 at or before this counts as ascent (Starlink deorbit ~T+53,
# GTO ~T+27–30). Later than this, and SECO-3+, fall back to SECO-1
# or first-stage recovery. The hard cap above still clips the push.
ASCENT_SECO_LIMIT_SEC = 75 * 60
DEFAULT_END_SEC = 9 * 60
FINAL_LINGER_SEC = 2 * 60
PROMPT_LINGER_SEC = 60
LATE_DISMISS_SEC = 15
# No In Flight / Success seen this long after NET: end anyway so a stale
# Launch Library status cannot strand the activity. Never applies to a hold.
UNCONFIRMED_END_SEC = 30 * 60
# LL2 status ids: 3 Success, 4 Failure, 6 In Flight, 7 Partial Failure,
# 9 Payload Deployed.
FLOWN_STATUS_IDS = (3, 4, 6, 7, 9)
NET_CHANGE_SEC = 2
# The app shows the countdown from T-1h.
START_WINDOW_SEC = 60 * 60
# A countdown already on screen ends when its NET moves further away than this.
# Not one hour: the app can only schedule the next countdown while it is open,
# so ending on a slip inside the launch window would leave a locked phone with
# no countdown at liftoff. A scrub to another day is well past it.
DELAYED_END_SEC = 4 * 60 * 60
DELAYED_LABEL = "Delayed"
HOLD_LABEL = "On Hold"
HOLD_ABBREV = "Hold"
# A hold publishes a far end date so the card never goes stale into ``SECO ✓``.
# Same horizon as the app: the later of NET+36h and six hours from now.
HOLD_NET_HORIZON_SEC = 36 * 3600
HOLD_STALE_HORIZON_SEC = 6 * 3600
# APNs reasons that mean this token will never work. Anything else is retried.
DEAD_TOKEN_REASONS = ("BadDeviceToken", "DeviceTokenNotForTopic", "Unregistered", "ExpiredToken")
SEND_RETRY_BASE_SEC = 15
SEND_RETRY_MAX_SEC = 600
# v1 kept every record in one JSON value: each change rewrote the whole document, two
# processes registering at once could drop a token, and the list was capped at 40.
STORE_KEY = "live_activity_push_v1"
# v2: one Redis hash field per activity, or one SQL row when DATABASE_URL is set.
HASH_KEY = "live_activity_push_v2"
SQL_TABLE = "live_activity_tokens"
MAX_RECORDS = 5000
DROP_AFTER_NET_SEC = 12 * 3600

_lock = threading.Lock()
_memory: dict = {}
_logged_disabled = False
_jwt_cache: dict = {"token": None, "exp": 0}
# activity_id -> (failed sends in a row, earliest next attempt). This process only.
_send_backoff: dict = {}

# SECO-1 / bare SECO only. SECO-2 is selected in ``activity_end_plan``
# when it falls at or before ``ASCENT_SECO_LIMIT_SEC``. Ship SECO on
# Starship is labelled SECO-1 in Launch Library; a bare ``SECO`` still counts.
_SECO1 = re.compile(r"\bseco(?:[\s-]*1)?(?![\s-]*\d)", re.IGNORECASE)
# Group 1 is the cutoff number when present (``SECO-2`` → 2, ``SECO-10`` → 10).
_SECO_NUMBER = re.compile(r"\bseco(?:[\s-]*(\d+))?\b", re.IGNORECASE)
# First-stage recovery: F9 ``Stage 1 Landing``, FH ``Boosters Landing`` and a
# core landing, Super Heavy ``Stage 1 Landing``/``Splashdown``/catch.
_RECOVERY_STAGE = re.compile(
    r"\b(stage[\s-]*1|first[\s-]*stage|boosters?|side[\s-]*boosters?|"
    r"(center[\s-]*)?core|super[\s-]*heavy)\b",
    re.IGNORECASE,
)
_RECOVERY_EVENT = re.compile(r"\b(landing|landed|splashdown|catch|caught|touchdown)\b", re.IGNORECASE)
_NOT_RECOVERY = re.compile(r"burn|flip|starship|\bship\b|dragon|capsule|fairing", re.IGNORECASE)


class LiveActivityTokenIn(BaseModel):
    token: str = Field(min_length=16)
    launch_id: str = Field(min_length=1)
    activity_id: str = Field(min_length=1)
    net: Optional[str] = None
    content_state: Optional[dict] = None


def install(fastapi_app: FastAPI) -> None:
    @fastapi_app.post("/live_activity/token", tags=["Live Activity"])
    def register_live_activity_token(body: LiveActivityTokenIn):
        token = _normalize_token(body.token)
        if token is None:
            raise HTTPException(status_code=400, detail="token must be hex")
        record = upsert_token(
            token=token,
            launch_id=body.launch_id.strip(),
            activity_id=body.activity_id.strip(),
            net=body.net,
            content_state=body.content_state,
        )
        return {"stored": True, "push": "enabled" if apns_configured() else "disabled", "activity_id": record["activity_id"]}

    @fastapi_app.delete("/live_activity/token/{activity_id}", tags=["Live Activity"])
    def delete_live_activity_token(activity_id: str):
        removed = delete_token(activity_id)
        return {"removed": removed}


def start_worker() -> None:
    def run():
        time.sleep(5)
        app_module = _app()
        if app_module is None or not getattr(app_module, "_background_enabled", False):
            return
        while getattr(app_module, "_background_enabled", False):
            try:
                # One sender per dyno: every uvicorn worker used to run this loop
                # and decide on the same tokens.
                if app_module.is_background_leader():
                    tick()
            except Exception as exc:
                print(f"Live Activity push tick error: {exc}")
            time.sleep(15)

    threading.Thread(target=run, daemon=True, name="live-activity-push").start()


def apns_configured() -> bool:
    return all(_env(name) for name in (
        "APNS_AUTH_KEY",
        "APNS_KEY_ID",
        "APNS_TEAM_ID",
        "APNS_LIVE_ACTIVITY_TOPIC",
    ))


def missing_apns_env() -> list:
    return [
        name for name in (
            "APNS_AUTH_KEY",
            "APNS_KEY_ID",
            "APNS_TEAM_ID",
            "APNS_LIVE_ACTIVITY_TOPIC",
        )
        if not _env(name)
    ]


def upsert_token(*, token: str, launch_id: str, activity_id: str, net: Optional[str], content_state: Optional[dict]) -> dict:
    now = time.time()
    with _lock:
        previous = _get(activity_id) or {}
        content = content_state if isinstance(content_state, dict) else previous.get("content_state")
        record = {
            "token": token,
            "launch_id": launch_id,
            "activity_id": activity_id,
            "net": net or previous.get("net"),
            "content_state": content,
            "registered_at": previous.get("registered_at") or now,
            "updated_at": now,
            "pushed_net_unix": previous.get("pushed_net_unix"),
            "pushed_dismiss_unix": previous.get("pushed_dismiss_unix"),
            # The app's own state says whether it shows a hold; ours only until then.
            "pushed_hold": None if isinstance(content_state, dict) else previous.get("pushed_hold"),
        }
        _put(activity_id, record)
        return dict(record)


def delete_token(activity_id: str) -> bool:
    activity_id = (activity_id or "").strip()
    if not activity_id:
        return False
    with _lock:
        return _remove(activity_id)


def decide(record: dict, launch: Optional[dict], raw: Optional[dict], now: datetime) -> Optional[dict]:
    """Pure decision for one registration. None means wait."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    net = _launch_net(launch, raw, record)
    if net is None:
        registered = record.get("registered_at") or 0
        if registered and now.timestamp() - float(registered) > 36 * 3600:
            return {"kind": "drop"}
        return None
    if now.timestamp() > net.timestamp() + DROP_AFTER_NET_SEC:
        return {"kind": "drop"}

    status, status_id = _status(launch, raw)
    events = timeline_events(raw)
    plan = activity_end_plan(net, events)
    content = record.get("content_state") if isinstance(record.get("content_state"), dict) else None

    if _is_prompt_terminal(status, status_id, net, now) and not (
        _is_success(status, status_id) and now < plan["milestone"]
    ):
        label = _terminal_text(status, status_id)
        dismissal = now + timedelta(seconds=PROMPT_LINGER_SEC)
        if _is_success(status, status_id) and plan["dismissal"] > dismissal:
            # Success past the milestone keeps the planned linger.
            dismissal = plan["dismissal"]
        return {
            "kind": "end",
            "priority": 10,
            "net_unix": net.timestamp(),
            "dismiss_unix": dismissal.timestamp(),
            "payload": _end_payload(record, launch, content, net, plan, label, dismissal, now),
        }

    flown = _has_flown(status, status_id)
    unconfirmed_due = (
        not _is_hold(status)
        and now.timestamp() >= net.timestamp() + UNCONFIRMED_END_SEC
    )
    if now >= plan["milestone"] and (flown or unconfirmed_due):
        if now >= plan["dismissal"]:
            dismissal = now + timedelta(seconds=LATE_DISMISS_SEC)
        else:
            dismissal = plan["dismissal"]
        return {
            "kind": "end",
            "priority": 10,
            "net_unix": net.timestamp(),
            "dismiss_unix": dismissal.timestamp(),
            "payload": _end_payload(record, launch, content, net, plan, plan["label"], dismissal, now),
        }

    net_unix = net.timestamp()
    now_unix = now.timestamp()
    pushed_net = record.get("pushed_net_unix")
    pushed_dismiss = record.get("pushed_dismiss_unix")
    content_net = _content_net_unix(content) if content is not None else None
    shown_net = pushed_net if pushed_net is not None else content_net

    # The countdown is on screen (its NET was inside T-1h) and the new NET is
    # hours away: a scrub to another day. Without this the update below left a
    # 23-hour countdown on the Lock Screen.
    if (
        shown_net is not None
        and not flown
        and net_unix - now_unix > DELAYED_END_SEC
        and float(shown_net) - now_unix <= START_WINDOW_SEC
        and net_unix - float(shown_net) > NET_CHANGE_SEC
    ):
        dismissal = now + timedelta(seconds=PROMPT_LINGER_SEC)
        return {
            "kind": "end",
            "priority": 10,
            "net_unix": net_unix,
            "dismiss_unix": dismissal.timestamp(),
            "payload": _end_payload(
                record, launch, content, net, plan, DELAYED_LABEL, dismissal, now, final_now=True
            ),
        }

    if content is None:
        return None
    content_dismiss = _content_dismiss_unix(content)
    first = pushed_net is None and pushed_dismiss is None
    nets_match = content_net is not None and abs(content_net - net_unix) <= NET_CHANGE_SEC
    if pushed_net is None:
        net_changed = not nets_match
    else:
        net_changed = abs(float(pushed_net) - net_unix) > NET_CHANGE_SEC
    shown_hold = record.get("pushed_hold")
    if shown_hold is None:
        shown_hold = _content_is_hold(content)

    if _is_hold(status):
        # A hold changes the status, not the NET. Nothing used to go out, so the
        # card counted through zero as if the vehicle had left.
        horizon = _hold_horizon(net, now)
        if shown_hold and not net_changed:
            if first:
                return {
                    "kind": "note",
                    "net_unix": net_unix,
                    "dismiss_unix": content_dismiss if content_dismiss is not None else horizon.timestamp(),
                    "hold": True,
                    "payload": {"aps": {"content-state": content}},
                }
            return None
        base = _update_content(content, net, plan)
        return {
            "kind": "update",
            "priority": 10,
            "net_unix": net_unix,
            "dismiss_unix": horizon.timestamp(),
            "hold": True,
            "store_content": base,
            "payload": {
                "aps": {
                    "timestamp": int(now_unix),
                    "event": "update",
                    "stale-date": int(horizon.timestamp()),
                    "content-state": _hold_content(base, horizon),
                }
            },
        }

    published = _published_end(net, plan, flown or unconfirmed_due)
    dismiss_unix = published["dismissal"].timestamp()
    if first and not shown_hold:
        dismiss_match = content_dismiss is not None and abs(content_dismiss - dismiss_unix) <= NET_CHANGE_SEC
        if nets_match and (dismiss_match or content_dismiss is None):
            return {
                "kind": "note",
                "net_unix": net_unix,
                "dismiss_unix": content_dismiss if content_dismiss is not None else dismiss_unix,
                "hold": False,
                "payload": {"aps": {"content-state": content}},
            }
    dismiss_changed = pushed_dismiss is None or abs(float(pushed_dismiss) - dismiss_unix) > NET_CHANGE_SEC
    if not net_changed and not dismiss_changed and not shown_hold:
        return None
    updated = _update_content(content, net, plan, published)
    if shown_hold:
        updated = _release_hold(updated, status)
    return {
        "kind": "update",
        "priority": 10,
        "net_unix": net_unix,
        "dismiss_unix": dismiss_unix,
        "hold": False,
        "payload": {
            "aps": {
                "timestamp": int(now_unix),
                "event": "update",
                "stale-date": int(dismiss_unix),
                "content-state": updated,
            }
        },
    }


def _published_end(net: datetime, plan: dict, liftoff_known: bool) -> dict:
    """The end the card may show now. Same gate as the app.

    Until Launch Library shows the vehicle flew (or T+30 passes with no word),
    a milestone before T+30 is withheld and T+30 stands in. Publishing
    ``SECO ✓`` at T+11 on a Go row froze the card into that line while the
    rocket was still on the pad.
    """
    gate = net + timedelta(seconds=UNCONFIRMED_END_SEC)
    if liftoff_known or plan["milestone"] >= gate:
        return {"milestone": plan["milestone"], "dismissal": plan["dismissal"]}
    return {"milestone": gate, "dismissal": gate}


def _hold_horizon(net: datetime, now: datetime) -> datetime:
    return max(
        net + timedelta(seconds=HOLD_NET_HORIZON_SEC),
        now + timedelta(seconds=HOLD_STALE_HORIZON_SEC),
    )


def _content_is_hold(content: Optional[dict]) -> bool:
    if not isinstance(content, dict):
        return False
    return "hold" in str(content.get("statusLabel") or "").lower()


def _hold_content(content: dict, horizon: datetime) -> dict:
    """The card for a hold: ``On Hold`` before and after the old T0, no flight end.

    The timeline is left out on purpose. The app re-reads the phase names
    from it when it decodes the state, which would put ``Engine Chill`` and
    ``Liftoff`` back over the hold text.
    """
    held = dict(content)
    held.pop("timelineEvents", None)
    held["phase"] = "scrubbed"
    held["statusLabel"] = HOLD_LABEL
    held["preLiftoffLabel"] = HOLD_LABEL
    held["preLiftoffAbbrev"] = HOLD_ABBREV
    held["postLiftoffLabel"] = HOLD_LABEL
    held["postLiftoffAbbrev"] = HOLD_ABBREV
    held["finalPhaseLabel"] = HOLD_LABEL
    held["activityMilestone"] = _unix_to_swift(horizon.timestamp())
    held["activityDismissAt"] = _unix_to_swift(horizon.timestamp())
    return held


def _release_hold(content: dict, status: Optional[str]) -> dict:
    """Back to a countdown after a hold clears. Hold text must not outlive it."""
    released = dict(content)
    if _content_is_hold(released) or released.get("phase") == "scrubbed":
        released["phase"] = "upcoming"
        released["statusLabel"] = (status or "").strip() or "Go for Launch"
    # The app refills these from the timeline, or with its defaults.
    for label, abbrev in (("preLiftoffLabel", "preLiftoffAbbrev"), ("postLiftoffLabel", "postLiftoffAbbrev")):
        if "hold" in str(released.get(label) or "").lower():
            released.pop(label, None)
            released.pop(abbrev, None)
    return released


def activity_end_plan(net: datetime, events: list) -> dict:
    """Later of the last ascent SECO and the last first-stage recovery.

    Ascent SECO is SECO-1 (or a bare SECO), and SECO-2 when that cutoff
    is at or before T+75. Starlink's deorbit SECO-2 (~T+53) and a GTO
    SECO-2 (~T+27–30) both count. SECO-3 and later, including a direct-GEO
    coast such as GOES-U SECO-3, are ignored, as is any SECO past T+75;
    the milestone then falls back to SECO-1 or recovery. Falcon Heavy
    lists side boosters and (when recovered) the core separately, so
    recoveries take the latest. T+9 when neither is present. The
    milestone and the two-minute dismissal never run past T+80.

    ``milestone_source`` is the winning timeline label (``SECO-2``,
    ``Center Core Landing``, …) or ``T+9`` for the fallback. ``label``
    stays the short Live Activity string (``SECO ✓`` or ``Booster landed``);
    core and side-booster landings share that text.
    """
    seco = None
    recovery = None
    for label, seconds in events:
        if seconds < 0:
            continue
        if _is_ascent_seco(label, seconds):
            if seco is None or seconds > seco[0]:
                seco = (seconds, label)
        if _is_stage1_landing(label):
            if recovery is None or seconds > recovery[0]:
                recovery = (seconds, label)
    if seco is not None and recovery is not None and recovery[0] > seco[0]:
        seconds, milestone_source, label = recovery[0], recovery[1], "Booster landed"
    elif seco is not None:
        seconds, milestone_source, label = seco[0], seco[1], "SECO \u2713"
    elif recovery is not None:
        seconds, milestone_source, label = recovery[0], recovery[1], "Booster landed"
    else:
        seconds, milestone_source, label = DEFAULT_END_SEC, "T+9", "SECO \u2713"
    clamped = min(max(seconds, 0), HARD_CAP_SEC)
    milestone = net + timedelta(seconds=clamped)
    dismissal = min(
        milestone + timedelta(seconds=FINAL_LINGER_SEC),
        net + timedelta(seconds=HARD_CAP_SEC),
    )
    return {
        "milestone": milestone,
        "dismissal": dismissal,
        "label": label,
        "milestone_source": milestone_source,
        "milestone_unix": milestone.timestamp(),
        "dismissal_unix": dismissal.timestamp(),
    }


def timeline_events(raw: Optional[dict]) -> list:
    if not isinstance(raw, dict):
        return []
    rows = raw.get("timeline") or []
    if not isinstance(rows, list):
        return []
    events = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = ""
        kind = row.get("type")
        if isinstance(kind, dict):
            label = (kind.get("abbrev") or kind.get("description") or "").strip()
        if not label:
            label = str(row.get("label") or row.get("abbrev") or "").strip()
        seconds = _parse_relative(row.get("relative_time"))
        if seconds is None:
            seconds = _parse_relative(row.get("relative_seconds"))
        if label and seconds is not None:
            events.append((label, float(seconds)))
    return events


def send_push(token: str, payload: dict, priority: int = 10) -> dict:
    """POST one liveactivity notification. Safe when APNs is not configured."""
    if not apns_configured():
        return {"sent": False, "reason": "disabled"}
    try:
        import httpx
    except ImportError:
        print("Live Activity push skipped; httpx is not installed")
        return {"sent": False, "reason": "no-httpx"}
    try:
        bearer = _apns_jwt()
    except Exception as exc:
        print(f"Live Activity APNs token error: {exc}")
        return {"sent": False, "reason": "jwt"}
    host = _apns_host()
    topic = _env("APNS_LIVE_ACTIVITY_TOPIC")
    url = f"https://{host}/3/device/{token}"
    headers = {
        "authorization": f"bearer {bearer}",
        "apns-topic": topic,
        "apns-push-type": "liveactivity",
        "apns-priority": str(priority),
    }
    try:
        with httpx.Client(http2=True, timeout=10.0) as client:
            response = client.post(url, headers=headers, json=payload)
    except Exception as exc:
        print(f"Live Activity APNs transport error: {exc}")
        return {"sent": False, "reason": "transport"}
    if response.status_code == 200:
        return {"sent": True}
    body = (response.text or "")[:300]
    reason = _apns_reason(body)
    if response.status_code == 410 or reason in DEAD_TOKEN_REASONS:
        # A debug build's sandbox token sent to production is BadDeviceToken. It
        # was retried every 15 seconds until twelve hours after NET.
        print(f"Live Activity APNs {response.status_code}: {reason or 'gone'}; dropping the token")
        return {"sent": False, "unregistered": True, "status": response.status_code, "reason": reason}
    print(f"Live Activity APNs {response.status_code}: {body}")
    return {"sent": False, "status": response.status_code, "reason": reason}


def _apns_reason(body: str) -> Optional[str]:
    try:
        parsed = json.loads(body or "")
    except ValueError:
        return None
    reason = parsed.get("reason") if isinstance(parsed, dict) else None
    return reason if isinstance(reason, str) else None


def _end_payload(record, launch, content, net, plan, label, dismissal, now, final_now: bool = False) -> dict:
    """The last card. ``final_now`` shows ``label`` at once instead of at the milestone."""
    state = dict(content or {})
    state["missionName"] = state.get("missionName") or _mission_name(launch, record) or "Launch"
    state["net"] = _unix_to_swift(net.timestamp())
    state["statusLabel"] = label
    state["phase"] = "complete"
    state["finalPhaseLabel"] = label
    state["activityMilestone"] = _unix_to_swift(plan["milestone"].timestamp())
    state["activityDismissAt"] = _unix_to_swift(dismissal.timestamp())
    state["milestoneSource"] = plan.get("milestone_source")
    if final_now:
        # The milestone is on the new NET, far ahead; the app reveals the final
        # line there. Put it just behind now and drop the names it would
        # otherwise re-read from the timeline.
        state.pop("timelineEvents", None)
        state["activityMilestone"] = _unix_to_swift(now.timestamp() - 1)
        for key in ("preLiftoffLabel", "preLiftoffAbbrev", "postLiftoffLabel", "postLiftoffAbbrev"):
            state[key] = label
    state = _fit_content(state)
    return {
        "aps": {
            "timestamp": int(now.timestamp()),
            "event": "end",
            "dismissal-date": int(dismissal.timestamp()),
            "stale-date": int(dismissal.timestamp()),
            "content-state": state,
        }
    }


def _update_content(content: dict, net: datetime, plan: dict, published: Optional[dict] = None) -> dict:
    published = published or plan
    updated = dict(content)
    old_net = _content_net_unix(content)
    if old_net is not None:
        delta = net.timestamp() - old_net
        if abs(delta) > 0.5:
            _shift_swift_dates(updated, delta)
    updated["net"] = _unix_to_swift(net.timestamp())
    updated["activityMilestone"] = _unix_to_swift(published["milestone"].timestamp())
    updated["activityDismissAt"] = _unix_to_swift(published["dismissal"].timestamp())
    updated["finalPhaseLabel"] = plan["label"]
    updated["milestoneSource"] = plan.get("milestone_source")
    return _fit_content(updated)


def _shift_swift_dates(content: dict, delta_unix: float) -> None:
    for key in ("net", "activityMilestone", "activityDismissAt"):
        if isinstance(content.get(key), (int, float)):
            content[key] = float(content[key]) + delta_unix
    for key in ("timerInterval", "radarLoopInterval"):
        span = content.get(key)
        if isinstance(span, list) and len(span) == 2 and all(
            isinstance(bound, (int, float)) and not isinstance(bound, bool) for bound in span
        ):
            # Swift encodes ClosedRange<Date> as [lower, upper]. This is the shape
            # the app sends; the keyed form below never came from it.
            content[key] = [float(span[0]) + delta_unix, float(span[1]) + delta_unix]
        elif isinstance(span, dict):
            shifted = dict(span)
            for bound in ("lowerBound", "upperBound"):
                if isinstance(shifted.get(bound), (int, float)):
                    shifted[bound] = float(shifted[bound]) + delta_unix
            content[key] = shifted


def _fit_content(content: dict) -> dict:
    events = content.get("timelineEvents")
    if isinstance(events, list) and len(events) > 4:
        content = dict(content)
        content["timelineEvents"] = events[:4]
    return content


def _note_action_applied(records, activity_id, action) -> None:
    stored = records.get(activity_id)
    if not stored or action.get("kind") != "note":
        return
    stored["pushed_net_unix"] = action.get("net_unix")
    stored["pushed_dismiss_unix"] = action.get("dismiss_unix")
    stored["pushed_hold"] = action.get("hold")


def _wants_fresh(record: dict, launch: Optional[dict], raw: Optional[dict], now: datetime) -> bool:
    """True from T-1h until the flight ends (or while a hold is up).

    Outside that the ten-minute list copy is close enough. A launch id the
    server has never seen (a debug simulation) is never worth a Launch Library
    read: each one was a 404 that counted against the hourly limit.
    """
    if launch is None and raw is None:
        return False
    net = _launch_net(launch, raw, record)
    if net is None:
        return False
    status, _status_id = _status(launch, raw)
    after = HOLD_STALE_HORIZON_SEC if _is_hold(status) else HARD_CAP_SEC + FINAL_LINGER_SEC
    return net.timestamp() - START_WINDOW_SEC <= now.timestamp() <= net.timestamp() + after


def tick(
    now: Optional[datetime] = None,
    lookup: Optional[Callable] = None,
    sender: Optional[Callable] = None,
    refresh: Optional[Callable] = None,
) -> dict:
    """Send due end/update pushes. No-ops when APNs is not configured.

    The cached lookup can be ~10 minutes old (list refresh cadence). Before
    any ``end`` goes out, ``refresh`` re-reads the launch from Launch Library
    and the decision is re-run, so a recycled T0 or a scrub seen late turns
    into an update instead of an early end.
    """
    global _logged_disabled
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    if not apns_configured():
        if not _logged_disabled:
            missing = ", ".join(missing_apns_env())
            print(f"Live Activity push disabled; missing {missing}")
            _logged_disabled = True
        return {"push": "disabled", "sent": 0, "stored": _count()}

    find = lookup or _default_lookup
    fresh = refresh or (_default_refresh if lookup is None else None)
    send = sender or send_push
    sent = 0
    with _lock:
        snapshot = {key: dict(value) for key, value in _load().items() if isinstance(value, dict)}
        _prune(snapshot)
    # One lookup per launch per pass, however many phones are watching it.
    seen: dict = {}

    def resolve(launch_id, want_fresh: bool):
        entry = seen.get(launch_id)
        if entry is None:
            entry = {"data": find(launch_id), "fresh": False}
            seen[launch_id] = entry
        if want_fresh and not entry["fresh"] and fresh is not None:
            entry["fresh"] = True
            try:
                entry["data"] = fresh(launch_id)
            except Exception as exc:
                print(f"Live Activity fresh lookup error: {exc}")
        return entry["data"]

    for activity_id in [key for key in _send_backoff if key not in snapshot]:
        _send_backoff.pop(activity_id, None)
    for activity_id, record in snapshot.items():
        if not record:
            continue
        launch_id = record.get("launch_id")
        launch, raw = resolve(launch_id, False)
        if _wants_fresh(record, launch, raw, moment):
            launch, raw = resolve(launch_id, True)
        action = decide(record, launch, raw, moment)
        if action is not None and action["kind"] == "end" and not seen[launch_id]["fresh"] and fresh is not None:
            launch, raw = resolve(launch_id, True)
            action = decide(record, launch, raw, moment)
        if action is None:
            continue
        kind = action["kind"]
        if kind == "note":
            with _lock:
                _note_action_applied(snapshot, activity_id, action)
                _put(activity_id, snapshot[activity_id])
            continue
        if kind == "drop":
            delete_token(activity_id)
            _send_backoff.pop(activity_id, None)
            continue
        waiting = _send_backoff.get(activity_id)
        if waiting is not None and moment.timestamp() < waiting[1]:
            continue
        result = send(record["token"], action["payload"], priority=action.get("priority", 10))
        if result.get("sent"):
            sent += 1
        if result.get("unregistered") or (result.get("sent") and kind == "end"):
            delete_token(activity_id)
            _send_backoff.pop(activity_id, None)
            continue
        if not result.get("sent"):
            # Not a dead token: try again, but not on every pass.
            failures = (waiting[0] if waiting is not None else 0) + 1
            delay = min(SEND_RETRY_BASE_SEC * (2 ** failures), SEND_RETRY_MAX_SEC)
            _send_backoff[activity_id] = (failures, moment.timestamp() + delay)
            continue
        _send_backoff.pop(activity_id, None)
        if kind == "update":
            with _lock:
                stored = snapshot.get(activity_id)
                if stored:
                    stored["pushed_net_unix"] = action.get("net_unix")
                    stored["pushed_dismiss_unix"] = action.get("dismiss_unix")
                    stored["pushed_hold"] = action.get("hold")
                    stored["content_state"] = action.get("store_content") or action["payload"]["aps"]["content-state"]
                    stored["updated_at"] = time.time()
                    _put(activity_id, stored)
    return {"push": "enabled", "sent": sent, "stored": _count()}


def _is_seco1(label: str) -> bool:
    return _SECO1.search(label or "") is not None


def _seco_number(label: str) -> Optional[int]:
    """1 for bare SECO / SECO-1, 2 for SECO-2, and so on. None when it is not a SECO."""
    match = _SECO_NUMBER.search(label or "")
    if match is None:
        return None
    digits = match.group(1)
    if not digits:
        return 1
    try:
        return int(digits)
    except ValueError:
        return None


def _is_ascent_seco(label: str, seconds: float) -> bool:
    """SECO-1, or SECO-2 at or before T+75. SECO-3+ and anything later are not ascent."""
    number = _seco_number(label)
    if number not in (1, 2):
        return False
    return seconds <= ASCENT_SECO_LIMIT_SEC


def _is_stage1_landing(label: str) -> bool:
    text = label or ""
    if _NOT_RECOVERY.search(text):
        return False
    return bool(_RECOVERY_STAGE.search(text) and _RECOVERY_EVENT.search(text))


def _is_scrub(status: Optional[str]) -> bool:
    """Scrub-like and terminal. A hold is not: it can recycle and fly."""
    text = (status or "").lower()
    return any(word in text for word in ("scrub", "delay", "postpone"))


def _is_hold(status: Optional[str]) -> bool:
    return "hold" in (status or "").lower()


def _is_success(status: Optional[str], status_id) -> bool:
    try:
        if int(status_id) == 3:
            return True
    except (TypeError, ValueError):
        pass
    return "success" in (status or "").lower()


def _has_flown(status: Optional[str], status_id) -> bool:
    """Launch Library shows liftoff happened (In Flight or a final status)."""
    if _is_hold(status) or _is_scrub(status):
        return False
    try:
        if int(status_id) in FLOWN_STATUS_IDS:
            return True
    except (TypeError, ValueError):
        pass
    text = (status or "").lower()
    return any(word in text for word in ("in flight", "success", "fail", "deployed"))


def _is_complete(status: Optional[str], status_id) -> bool:
    text = (status or "").lower()
    if any(word in text for word in ("go", "in flight", "liftoff")) or _is_scrub(status) or _is_hold(status):
        return False
    try:
        if int(status_id) in (3, 4, 7):
            return True
    except (TypeError, ValueError):
        pass
    return "success" in text or "fail" in text


def _is_prompt_terminal(status, status_id, net: datetime, now: datetime) -> bool:
    if _is_complete(status, status_id):
        return True
    if _is_scrub(status) and net <= now:
        return True
    return False


def _terminal_text(status, status_id) -> str:
    text = (status or "").lower()
    try:
        numeric = int(status_id)
    except (TypeError, ValueError):
        numeric = None
    if "fail" in text or numeric in (4, 7):
        return "Failure"
    if "success" in text or numeric == 3:
        return "Success"
    if "scrub" in text:
        return "Scrubbed"
    return "Hold"


def _parse_relative(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    sign = 1.0
    if text[0] == "-":
        sign = -1.0
        text = text[1:]
    elif text[0] == "+":
        text = text[1:]
    if not text or text[0] != "P":
        try:
            return float(value)
        except ValueError:
            return None
    text = text[1:]
    total = 0.0
    number = ""
    units = {"W": 7 * 86400, "D": 86400, "H": 3600, "M": 60, "S": 1}

    def flush(unit: str) -> bool:
        nonlocal total, number
        if number == "":
            return False
        try:
            amount = float(number)
        except ValueError:
            return False
        number = ""
        scale = units.get(unit)
        if scale is None:
            return False
        total += amount * scale
        return True

    for ch in text:
        if ch == "T":
            continue
        if ch.isdigit() or ch == ".":
            number += ch
            continue
        if not flush(ch):
            return None
    if number:
        return None
    return sign * total


def _launch_net(launch, raw, record) -> Optional[datetime]:
    for source in (raw, launch):
        if isinstance(source, dict):
            parsed = _parse_net(source.get("net"))
            if parsed is not None:
                return parsed
    return _parse_net(record.get("net"))


def _parse_net(value) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _status(launch, raw):
    status = None
    status_id = None
    if isinstance(raw, dict):
        raw_status = raw.get("status")
        if isinstance(raw_status, dict):
            status = raw_status.get("name") or raw_status.get("abbrev")
            status_id = raw_status.get("id")
        elif isinstance(raw_status, str):
            status = raw_status
        if status_id is None:
            status_id = raw.get("status_id")
    if isinstance(launch, dict):
        if not status and isinstance(launch.get("status"), str):
            status = launch.get("status")
        if status_id is None:
            status_id = launch.get("status_id")
    return status, status_id


def _mission_name(launch, record) -> Optional[str]:
    content = record.get("content_state") if isinstance(record.get("content_state"), dict) else {}
    if content.get("missionName"):
        return str(content["missionName"])
    if isinstance(launch, dict):
        for key in ("mission", "name"):
            if launch.get(key):
                return str(launch[key])
    return None


def _content_net_unix(content: dict) -> Optional[float]:
    net = content.get("net")
    if isinstance(net, (int, float)):
        return _swift_to_unix(float(net))
    return None


def _content_dismiss_unix(content: dict) -> Optional[float]:
    dismiss = content.get("activityDismissAt")
    if isinstance(dismiss, (int, float)):
        return _swift_to_unix(float(dismiss))
    return None


def _unix_to_swift(unix: float) -> float:
    return float(unix) - SWIFT_REFERENCE_UNIX


def _swift_to_unix(swift: float) -> float:
    return float(swift) + SWIFT_REFERENCE_UNIX


def _normalize_token(token: str) -> Optional[str]:
    text = (token or "").strip().lower()
    if len(text) < 16 or any(ch not in "0123456789abcdef" for ch in text):
        return None
    return text


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _apns_host() -> str:
    environment = _env("APNS_ENVIRONMENT").lower()
    if environment in ("sandbox", "development"):
        return "api.sandbox.push.apple.com"
    return "api.push.apple.com"


def _apns_key_pem() -> str:
    raw = os.getenv("APNS_AUTH_KEY") or ""
    text = raw.strip()
    if "\\n" in text and "BEGIN" in text:
        text = text.replace("\\n", "\n")
    return text


def _apns_jwt() -> str:
    now = int(time.time())
    cached = _jwt_cache.get("token")
    if cached and now < int(_jwt_cache.get("exp") or 0) - 60:
        return cached
    import jwt
    token = jwt.encode(
        {"iss": _env("APNS_TEAM_ID"), "iat": now},
        _apns_key_pem(),
        algorithm="ES256",
        headers={"alg": "ES256", "kid": _env("APNS_KEY_ID")},
    )
    if isinstance(token, bytes):
        token = token.decode()
    _jwt_cache["token"] = token
    _jwt_cache["exp"] = now + 50 * 60
    return token


def _default_lookup(launch_id):
    app_module = _app()
    if app_module is None or not launch_id:
        return None, None
    launch = None
    try:
        data = app_module._load_launch_payload(force=False)
    except Exception:
        data = None
    if isinstance(data, dict):
        for bucket in ("upcoming", "previous"):
            for row in data.get(bucket) or []:
                if isinstance(row, dict) and str(row.get("id")) == str(launch_id):
                    launch = row
                    break
            if launch is not None:
                break
    raw = None
    try:
        raw = app_module._get_raw_launch(launch_id)
    except Exception:
        raw = None
    return launch, raw


def _default_refresh(launch_id):
    """Same as the cached lookup, but raw comes from the short-lived hot single-flight."""
    launch, raw = _default_lookup(launch_id)
    app_module = _app()
    if app_module is None or not launch_id:
        return launch, raw
    if launch is None and raw is None:
        # Not a launch this server knows. Asking Launch Library is a 404 that
        # still counts against the hourly limit.
        return launch, raw
    try:
        hot = app_module._refresh_hot_launch_raw(launch_id)
    except Exception:
        hot = None
    return launch, (hot if isinstance(hot, dict) else raw)


def _app():
    try:
        import app as app_module
    except Exception:
        return None
    return app_module


class _RedisHashStore:
    """One hash field per activity: a change touches that activity only."""

    def __init__(self, client):
        self._client = client

    def all(self) -> dict:
        records = {}
        for activity_id, blob in (self._client.hgetall(HASH_KEY) or {}).items():
            try:
                record = json.loads(blob)
            except (TypeError, ValueError):
                continue
            if isinstance(record, dict):
                records[str(activity_id)] = record
        return records

    def get(self, activity_id: str) -> Optional[dict]:
        blob = self._client.hget(HASH_KEY, activity_id)
        if not blob:
            return None
        try:
            record = json.loads(blob)
        except (TypeError, ValueError):
            return None
        return record if isinstance(record, dict) else None

    def put(self, activity_id: str, record: dict) -> None:
        self._client.hset(HASH_KEY, activity_id, json.dumps(record))

    def delete(self, activity_id: str) -> bool:
        return bool(self._client.hdel(HASH_KEY, activity_id))


class SqlTokenStore:
    """One row per activity in a SQL table, so tokens survive a Redis restart.

    The Redis plan in use keeps nothing on disk: a restart or failover there used to
    forget every registered Live Activity. `connect` returns a DB-API connection
    (psycopg for Postgres; sqlite3 in tests) and `placeholder` is its parameter mark.
    One connection, reopened after any error.
    """

    def __init__(self, connect: Callable, placeholder: str = "%s"):
        self._connect = connect
        self._placeholder = placeholder
        self._lock = threading.Lock()
        self._connection = None

    def _run(self, sql: str, params: tuple = (), fetch: bool = False):
        sql = sql.replace("?", self._placeholder)
        with self._lock:
            for attempt in (1, 2):
                try:
                    if self._connection is None:
                        self._connection = self._connect()
                        cursor = self._connection.cursor()
                        cursor.execute(
                            f"CREATE TABLE IF NOT EXISTS {SQL_TABLE} ("
                            "activity_id TEXT PRIMARY KEY, "
                            "record TEXT NOT NULL, "
                            "updated_at DOUBLE PRECISION NOT NULL)"
                        )
                        self._connection.commit()
                    cursor = self._connection.cursor()
                    cursor.execute(sql, params)
                    result = cursor.fetchall() if fetch else cursor.rowcount
                    self._connection.commit()
                    return result
                except Exception:
                    try:
                        if self._connection is not None:
                            self._connection.close()
                    except Exception:
                        pass
                    self._connection = None
                    if attempt == 2:
                        raise

    def all(self) -> dict:
        records = {}
        for activity_id, blob in self._run(f"SELECT activity_id, record FROM {SQL_TABLE}", fetch=True):
            try:
                record = json.loads(blob)
            except (TypeError, ValueError):
                continue
            if isinstance(record, dict):
                records[str(activity_id)] = record
        return records

    def get(self, activity_id: str) -> Optional[dict]:
        rows = self._run(
            f"SELECT record FROM {SQL_TABLE} WHERE activity_id = ?", (activity_id,), fetch=True
        )
        if not rows:
            return None
        try:
            record = json.loads(rows[0][0])
        except (TypeError, ValueError):
            return None
        return record if isinstance(record, dict) else None

    def put(self, activity_id: str, record: dict) -> None:
        self._run(
            f"INSERT INTO {SQL_TABLE} (activity_id, record, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT (activity_id) DO UPDATE SET record = excluded.record, updated_at = excluded.updated_at",
            (activity_id, json.dumps(record), float(record.get("updated_at") or time.time())),
        )

    def delete(self, activity_id: str) -> bool:
        return bool(self._run(f"DELETE FROM {SQL_TABLE} WHERE activity_id = ?", (activity_id,)))


_sql_store: Optional[SqlTokenStore] = None
_sql_store_checked = False
_migrated: set = set()


def _database_url() -> str:
    return _env("LIVE_ACTIVITY_DATABASE_URL") or _env("DATABASE_URL")


def _configured_sql_store() -> Optional[SqlTokenStore]:
    """The SQL store when a database URL is set and its driver is installed."""
    global _sql_store, _sql_store_checked
    if _sql_store_checked:
        return _sql_store
    _sql_store_checked = True
    url = _database_url()
    if not url:
        return None
    try:
        import psycopg
    except ImportError:
        print("Live Activity tokens: DATABASE_URL is set but psycopg is not installed; using Redis")
        return None
    _sql_store = SqlTokenStore(lambda: psycopg.connect(url, connect_timeout=5))
    print("Live Activity tokens: stored in the SQL database")
    return _sql_store


def _redis_store() -> Optional[_RedisHashStore]:
    app_module = _app()
    client = getattr(app_module, "r", None) if app_module is not None else None
    return _RedisHashStore(client) if client else None


def _store():
    return _configured_sql_store() or _redis_store()


def _migrate_once(store, records: dict) -> dict:
    """Copy records from the store this one replaces, once.

    A marker in Redis records that the copy happened, so an activity that has since
    ended is not brought back from the old copy at the next restart. Without Redis
    there is nothing to copy from.
    """
    name = type(store).__name__
    if name in _migrated:
        return records
    _migrated.add(name)
    app_module = _app()
    client = getattr(app_module, "r", None) if app_module is not None else None
    if not client:
        return records
    marker = f"{HASH_KEY}:migrated:{name}"
    if client.get(marker):
        return records
    older: dict = {}
    legacy = app_module.get_cached_data(STORE_KEY, strict=True)
    if isinstance(legacy, dict):
        older.update({str(k): v for k, v in legacy.items() if isinstance(v, dict)})
    if isinstance(store, SqlTokenStore):
        older.update(_RedisHashStore(client).all())
    moved = 0
    for activity_id, record in older.items():
        if activity_id not in records:
            store.put(activity_id, record)
            records[activity_id] = record
            moved += 1
    client.set(marker, str(int(time.time())))
    if moved:
        print(f"Live Activity tokens: moved {moved} record(s) into {name}")
    return records


def _load() -> dict:
    """Every stored record. This process's last copy when the store cannot be read."""
    store = _store()
    if store is not None:
        try:
            records = store.all()
        except Exception as exc:
            print(f"Live Activity token read error: {exc}")
            return _memory
        try:
            records = _migrate_once(store, records)
        except Exception as exc:
            # Try again at the next read: the old copy could not be read just now.
            _migrated.discard(type(store).__name__)
            print(f"Live Activity token migration error: {exc}")
        _memory.clear()
        _memory.update(records)
    return _memory


def _get(activity_id: str) -> Optional[dict]:
    """One record, without reading every other one."""
    store = _store()
    if store is not None:
        try:
            record = store.get(activity_id)
        except Exception as exc:
            print(f"Live Activity token read error: {exc}")
        else:
            if record is not None:
                _memory[activity_id] = record
            return record
    return _memory.get(activity_id)


def _put(activity_id: str, record: dict) -> None:
    _memory[activity_id] = dict(record)
    store = _store()
    if store is None:
        return
    try:
        store.put(activity_id, record)
    except Exception as exc:
        print(f"Live Activity token store error: {exc}")


def _remove(activity_id: str) -> bool:
    removed = _memory.pop(activity_id, None) is not None
    store = _store()
    if store is None:
        return removed
    try:
        return store.delete(activity_id) or removed
    except Exception as exc:
        print(f"Live Activity token store error: {exc}")
        return removed


def _count() -> int:
    with _lock:
        return len(_load())


def _prune(records: dict) -> None:
    """Drop the oldest registrations past MAX_RECORDS."""
    if len(records) <= MAX_RECORDS:
        return
    ordered = sorted(records.items(), key=lambda item: item[1].get("registered_at") or 0)
    for activity_id, _record in ordered[: len(records) - MAX_RECORDS]:
        records.pop(activity_id, None)
        _remove(activity_id)


def reset_for_tests() -> None:
    global _logged_disabled, _sql_store, _sql_store_checked
    with _lock:
        _memory.clear()
        _migrated.clear()
        _sql_store = None
        _sql_store_checked = False
        _logged_disabled = False
        _send_backoff.clear()
        _jwt_cache["token"] = None
        _jwt_cache["exp"] = 0
