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
        records = _load()
        previous = records.get(activity_id) or {}
        content = content_state if isinstance(content_state, dict) else previous.get("content_state")
        records[activity_id] = {
            "token": token,
            "launch_id": launch_id,
            "activity_id": activity_id,
            "net": net or previous.get("net"),
            "content_state": content,
            "registered_at": previous.get("registered_at") or now,
            "updated_at": now,
            "pushed_net_unix": previous.get("pushed_net_unix"),
            "pushed_dismiss_unix": previous.get("pushed_dismiss_unix"),
        }
        _prune(records, now)
        _save(records)
        return dict(records[activity_id])


def delete_token(activity_id: str) -> bool:
    activity_id = (activity_id or "").strip()
    if not activity_id:
        return False
    with _lock:
        records = _load()
        removed = records.pop(activity_id, None) is not None
        if removed:
            _save(records)
        return removed


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

    if _is_prompt_terminal(status, status_id, net, now):
        label = _terminal_text(status, status_id)
        dismissal = now + timedelta(seconds=PROMPT_LINGER_SEC)
        return {
            "kind": "end",
            "priority": 10,
            "net_unix": net.timestamp(),
            "dismiss_unix": dismissal.timestamp(),
            "payload": _end_payload(record, launch, content, net, plan, label, dismissal, now),
        }

    if now >= plan["milestone"]:
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

    if content is None:
        return None
    dismiss_unix = plan["dismissal"].timestamp()
    net_unix = net.timestamp()
    pushed_net = record.get("pushed_net_unix")
    pushed_dismiss = record.get("pushed_dismiss_unix")
    content_net = _content_net_unix(content)
    content_dismiss = _content_dismiss_unix(content)
    if pushed_net is None and pushed_dismiss is None:
        nets_match = content_net is not None and abs(content_net - net_unix) <= NET_CHANGE_SEC
        dismiss_match = content_dismiss is not None and abs(content_dismiss - dismiss_unix) <= NET_CHANGE_SEC
        if nets_match and (dismiss_match or content_dismiss is None):
            return {
                "kind": "note",
                "net_unix": net_unix,
                "dismiss_unix": content_dismiss if content_dismiss is not None else dismiss_unix,
                "payload": {"aps": {"content-state": content}},
            }
    net_changed = pushed_net is None or abs(float(pushed_net) - net_unix) > NET_CHANGE_SEC
    if content_net is not None and abs(content_net - net_unix) <= NET_CHANGE_SEC and pushed_net is None:
        net_changed = False
    dismiss_changed = pushed_dismiss is None or abs(float(pushed_dismiss) - dismiss_unix) > NET_CHANGE_SEC
    if not net_changed and not dismiss_changed:
        return None
    updated = _update_content(content, net, plan)
    return {
        "kind": "update",
        "priority": 10,
        "net_unix": net_unix,
        "dismiss_unix": dismiss_unix,
        "payload": {
            "aps": {
                "timestamp": int(now.timestamp()),
                "event": "update",
                "stale-date": int(dismiss_unix),
                "content-state": updated,
            }
        },
    }


def activity_end_plan(net: datetime, events: list) -> dict:
    """Later of SECO-1 and stage-1 landing. T+9 when neither is present."""
    seco1 = None
    landing = None
    for label, seconds in events:
        if seconds < 0:
            continue
        if _is_seco1(label):
            seco1 = seconds if seco1 is None else min(seco1, seconds)
        if _is_stage1_landing(label):
            landing = seconds if landing is None else min(landing, seconds)
    if seco1 is not None and landing is not None and landing > seco1:
        seconds, label = landing, "Booster landed"
    elif seco1 is not None:
        seconds, label = seco1, "SECO \u2713"
    elif landing is not None:
        seconds, label = landing, "Booster landed"
    else:
        seconds, label = DEFAULT_END_SEC, "SECO \u2713"
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
    if response.status_code == 410:
        return {"sent": False, "unregistered": True}
    body = (response.text or "")[:300]
    print(f"Live Activity APNs {response.status_code}: {body}")
    return {"sent": False, "status": response.status_code}


def _end_payload(record, launch, content, net, plan, label, dismissal, now) -> dict:
    state = dict(content or {})
    state["missionName"] = state.get("missionName") or _mission_name(launch, record) or "Launch"
    state["net"] = _unix_to_swift(net.timestamp())
    state["statusLabel"] = label
    state["phase"] = "complete"
    state["finalPhaseLabel"] = label
    state["activityMilestone"] = _unix_to_swift(plan["milestone"].timestamp())
    state["activityDismissAt"] = _unix_to_swift(dismissal.timestamp())
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


def _update_content(content: dict, net: datetime, plan: dict) -> dict:
    updated = dict(content)
    old_net = _content_net_unix(content)
    if old_net is not None:
        delta = net.timestamp() - old_net
        if abs(delta) > 0.5:
            _shift_swift_dates(updated, delta)
    updated["net"] = _unix_to_swift(net.timestamp())
    updated["activityMilestone"] = _unix_to_swift(plan["milestone"].timestamp())
    updated["activityDismissAt"] = _unix_to_swift(plan["dismissal"].timestamp())
    updated["finalPhaseLabel"] = plan["label"]
    return _fit_content(updated)


def _shift_swift_dates(content: dict, delta_unix: float) -> None:
    for key in ("net", "activityMilestone", "activityDismissAt"):
        if isinstance(content.get(key), (int, float)):
            content[key] = float(content[key]) + delta_unix
    for key in ("timerInterval", "radarLoopInterval"):
        span = content.get(key)
        if isinstance(span, dict):
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


def tick(now: Optional[datetime] = None, lookup: Optional[Callable] = None, sender: Optional[Callable] = None) -> dict:
    """Send due end/update pushes. No-ops when APNs is not configured."""
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
    send = sender or send_push
    sent = 0
    with _lock:
        activity_ids = list(_load().keys())
    for activity_id in activity_ids:
        with _lock:
            record = dict((_load().get(activity_id) or {}))
        if not record:
            continue
        launch, raw = find(record.get("launch_id"))
        action = decide(record, launch, raw, moment)
        if action is None:
            continue
        kind = action["kind"]
        if kind == "note":
            with _lock:
                current = _load()
                _note_action_applied(current, activity_id, action)
                _save(current)
            continue
        if kind == "drop":
            delete_token(activity_id)
            continue
        result = send(record["token"], action["payload"], priority=action.get("priority", 10))
        if result.get("sent"):
            sent += 1
        if result.get("unregistered") or (result.get("sent") and kind == "end"):
            delete_token(activity_id)
            continue
        if result.get("sent") and kind == "update":
            with _lock:
                current = _load()
                stored = current.get(activity_id)
                if stored:
                    stored["pushed_net_unix"] = action.get("net_unix")
                    stored["pushed_dismiss_unix"] = action.get("dismiss_unix")
                    stored["content_state"] = action["payload"]["aps"]["content-state"]
                    stored["updated_at"] = time.time()
                    _save(current)
    return {"push": "enabled", "sent": sent, "stored": _count()}


def _is_seco1(label: str) -> bool:
    return _SECO1.search(label or "") is not None


def _is_stage1_landing(label: str) -> bool:
    text = (label or "").lower()
    if "burn" in text or "starship" in text:
        return False
    if "stage 1 landing" in text or "first stage landing" in text:
        return True
    if "booster landing" in text or "boosters landing" in text:
        return True
    return False


def _is_scrub(status: Optional[str]) -> bool:
    text = (status or "").lower()
    return any(word in text for word in ("scrub", "hold", "delay", "postpone"))


def _is_complete(status: Optional[str], status_id) -> bool:
    text = (status or "").lower()
    if any(word in text for word in ("go", "in flight", "liftoff")) or _is_scrub(status):
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


def _app():
    try:
        import app as app_module
    except Exception:
        return None
    return app_module


def _load() -> dict:
    app_module = _app()
    if app_module is not None:
        try:
            cached = app_module.get_cached_data(STORE_KEY)
        except Exception:
            cached = None
        if isinstance(cached, dict):
            _memory.clear()
            _memory.update(cached)
    return _memory


def _save(records: dict) -> None:
    snapshot = {}
    for key, value in records.items():
        snapshot[key] = dict(value) if isinstance(value, dict) else value
    _memory.clear()
    _memory.update(snapshot)
    app_module = _app()
    if app_module is None:
        return
    try:
        app_module.set_cached_data(STORE_KEY, snapshot)
    except Exception as exc:
        print(f"Live Activity token store error: {exc}")


def _count() -> int:
    with _lock:
        return len(_load())


def _prune(records: dict, now: float) -> None:
    if len(records) <= MAX_RECORDS:
        return
    ordered = sorted(records.items(), key=lambda item: item[1].get("registered_at") or 0)
    for activity_id, _record in ordered[: len(records) - MAX_RECORDS]:
        records.pop(activity_id, None)
    _ = now


def reset_for_tests() -> None:
    global _logged_disabled
    with _lock:
        _memory.clear()
        _logged_disabled = False
        _jwt_cache["token"] = None
        _jwt_cache["exp"] = 0
