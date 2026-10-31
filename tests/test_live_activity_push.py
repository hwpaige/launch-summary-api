import os
import unittest
from datetime import datetime, timedelta, timezone

import live_activity_push as push


NET = datetime(2026, 10, 31, 10, 23, tzinfo=timezone.utc)
SWIFT = push.SWIFT_REFERENCE_UNIX


def _iso(moment):
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _content(net, dismiss):
    return {
        "missionName": "Starlink",
        "net": net.timestamp() - SWIFT,
        "statusLabel": "In Flight",
        "phase": "live",
        "activityMilestone": (dismiss - timedelta(minutes=2)).timestamp() - SWIFT,
        "activityDismissAt": dismiss.timestamp() - SWIFT,
        "finalPhaseLabel": "SECO \u2713",
        "timerInterval": {
            "lowerBound": net.timestamp() - SWIFT,
            "upperBound": (net + timedelta(hours=12)).timestamp() - SWIFT,
        },
    }


class LiveActivityPushTests(unittest.TestCase):
    def setUp(self):
        push.reset_for_tests()
        self._env = {
            name: os.environ.pop(name, None)
            for name in (
                "APNS_AUTH_KEY",
                "APNS_KEY_ID",
                "APNS_TEAM_ID",
                "APNS_LIVE_ACTIVITY_TOPIC",
                "APNS_ENVIRONMENT",
            )
        }

    def tearDown(self):
        for name, value in self._env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        push.reset_for_tests()

    def test_end_plan_uses_later_of_seco_and_landing(self):
        events = [
            ("Liftoff", 0),
            ("Stage 1 Landing", 7 * 60 + 30),
            ("SECO-1", 8 * 60 + 39),
            ("Payload Deploy", 52 * 60),
            ("SECO-2", 53 * 60),
            ("SECO-10", 80 * 60),
            ("Starship Landing", 10 * 3600),
        ]
        plan = push.activity_end_plan(NET, events)
        self.assertEqual(plan["label"], "SECO \u2713")
        self.assertEqual(plan["milestone"], NET + timedelta(seconds=8 * 60 + 39))
        self.assertEqual(plan["dismissal"], NET + timedelta(seconds=8 * 60 + 39 + 120))

    def test_landing_after_seco_wins(self):
        plan = push.activity_end_plan(NET, [
            ("SECO-1", 8 * 60),
            ("Boosters Landing", 9 * 60),
        ])
        self.assertEqual(plan["label"], "Booster landed")
        self.assertEqual(plan["milestone"], NET + timedelta(minutes=9))

    def test_missing_timeline_is_t_plus_9(self):
        plan = push.activity_end_plan(NET, [])
        self.assertEqual(plan["label"], "SECO \u2713")
        self.assertEqual(plan["milestone"], NET + timedelta(minutes=9))
        self.assertEqual(plan["dismissal"], NET + timedelta(minutes=11))

    def test_hard_cap_clips_linger(self):
        plan = push.activity_end_plan(NET, [("SECO-1", 50 * 60)])
        self.assertEqual(plan["milestone"], NET + timedelta(minutes=20))
        self.assertEqual(plan["dismissal"], NET + timedelta(minutes=20))

    def test_relative_time_duration(self):
        raw = {"timeline": [
            {"type": {"abbrev": "SECO-1"}, "relative_time": "PT8M39S"},
            {"type": {"abbrev": "Stage 1 Landing"}, "relative_time": "-PT7M"},
        ]}
        events = push.timeline_events(raw)
        self.assertEqual(events[0], ("SECO-1", 8 * 60 + 39))
        self.assertEqual(events[1][0], "Stage 1 Landing")
        self.assertAlmostEqual(events[1][1], -7 * 60)

    def test_end_at_milestone_uses_short_dismissal_and_swift_dates(self):
        milestone = NET + timedelta(minutes=9)
        now = milestone + timedelta(seconds=5)
        dismiss = NET + timedelta(minutes=11)
        record = {
            "token": "ab" * 16,
            "launch_id": "launch-1",
            "activity_id": "activity-1",
            "content_state": _content(NET, dismiss),
            "pushed_net_unix": NET.timestamp(),
            "pushed_dismiss_unix": dismiss.timestamp(),
        }
        launch = {"id": "launch-1", "mission": "Starlink", "net": _iso(NET), "status": "In Flight", "status_id": 6}
        action = push.decide(record, launch, {"timeline": []}, now)
        self.assertEqual(action["kind"], "end")
        aps = action["payload"]["aps"]
        self.assertEqual(aps["event"], "end")
        self.assertEqual(aps["dismissal-date"], int(dismiss.timestamp()))
        self.assertLess(aps["dismissal-date"] - int(now.timestamp()), 180)
        state = aps["content-state"]
        self.assertEqual(state["phase"], "complete")
        self.assertEqual(state["finalPhaseLabel"], "SECO \u2713")
        self.assertAlmostEqual(state["net"], NET.timestamp() - SWIFT, places=3)
        self.assertNotAlmostEqual(state["net"], NET.timestamp(), delta=1000)

    def test_late_end_dismissal_is_only_a_few_seconds(self):
        now = NET + timedelta(minutes=30)
        record = {
            "token": "ab" * 16,
            "launch_id": "launch-1",
            "content_state": _content(NET, NET + timedelta(minutes=11)),
            "pushed_net_unix": NET.timestamp(),
            "pushed_dismiss_unix": (NET + timedelta(minutes=11)).timestamp(),
        }
        launch = {"net": _iso(NET), "status": "In Flight", "status_id": 6}
        action = push.decide(record, launch, None, now)
        self.assertEqual(action["kind"], "end")
        dismissal = action["payload"]["aps"]["dismissal-date"]
        self.assertEqual(dismissal, int((now + timedelta(seconds=15)).timestamp()))

    def test_scrub_and_failure_end_immediately(self):
        now = NET + timedelta(minutes=2)
        record = {"token": "ab" * 16, "launch_id": "launch-1", "content_state": _content(NET, NET + timedelta(minutes=11))}
        launch = {"net": _iso(NET), "status": "Scrubbed", "status_id": 1}
        action = push.decide(record, launch, None, now)
        self.assertEqual(action["kind"], "end")
        self.assertEqual(action["payload"]["aps"]["content-state"]["statusLabel"], "Scrubbed")

        launch["status"] = "Failure"
        launch["status_id"] = 4
        action = push.decide(record, launch, None, now)
        self.assertEqual(action["payload"]["aps"]["content-state"]["finalPhaseLabel"], "Failure")

    def test_hold_before_net_does_not_end(self):
        now = NET - timedelta(hours=2)
        record = {"token": "ab" * 16, "launch_id": "launch-1", "content_state": _content(NET, NET + timedelta(minutes=11))}
        launch = {"net": _iso(NET), "status": "Hold", "status_id": 1}
        action = push.decide(record, launch, None, now)
        self.assertEqual(action["kind"], "note")

    def test_net_change_sends_update_with_shifted_range(self):
        slipped = NET + timedelta(minutes=20)
        dismiss = NET + timedelta(minutes=11)
        record = {
            "token": "cd" * 16,
            "launch_id": "launch-1",
            "content_state": _content(NET, dismiss),
            "pushed_net_unix": NET.timestamp(),
            "pushed_dismiss_unix": dismiss.timestamp(),
        }
        launch = {"net": _iso(slipped), "status": "Go", "status_id": 1, "mission": "Starlink"}
        now = NET - timedelta(minutes=30)
        action = push.decide(record, launch, {"timeline": [{"type": {"abbrev": "SECO-1"}, "relative_time": "PT8M"}]}, now)
        self.assertEqual(action["kind"], "update")
        state = action["payload"]["aps"]["content-state"]
        self.assertAlmostEqual(state["net"], slipped.timestamp() - SWIFT, places=3)
        self.assertEqual(action["payload"]["aps"]["event"], "update")
        self.assertIsInstance(action["payload"]["aps"]["stale-date"], int)
        self.assertGreater(state["timerInterval"]["lowerBound"], _content(NET, dismiss)["timerInterval"]["lowerBound"])
        self.assertEqual(state["finalPhaseLabel"], "SECO \u2713")

    def test_timeline_correction_pushes_real_milestone(self):
        phone_dismiss = NET + timedelta(minutes=11)
        record = {
            "token": "ab" * 16,
            "launch_id": "launch-1",
            "content_state": _content(NET, phone_dismiss),
        }
        launch = {"net": _iso(NET), "status": "In Flight", "status_id": 6}
        raw = {"timeline": [{"type": {"abbrev": "SECO-1"}, "relative_time": "PT8M39S"}]}
        now = NET + timedelta(minutes=1)
        action = push.decide(record, launch, raw, now)
        self.assertEqual(action["kind"], "update")
        state = action["payload"]["aps"]["content-state"]
        expected = (NET + timedelta(seconds=8 * 60 + 39 + 120)).timestamp() - SWIFT
        self.assertAlmostEqual(state["activityDismissAt"], expected, places=3)

    def test_tick_is_a_noop_without_apns_env(self):
        push.upsert_token(
            token="ab" * 16,
            launch_id="launch-1",
            activity_id="activity-1",
            net=_iso(NET),
            content_state=_content(NET, NET + timedelta(minutes=11)),
        )
        calls = []
        result = push.tick(
            now=NET + timedelta(minutes=12),
            lookup=lambda _id: ({"net": _iso(NET), "status": "In Flight"}, None),
            sender=lambda *args, **kwargs: calls.append(args) or {"sent": True},
        )
        self.assertEqual(result["push"], "disabled")
        self.assertEqual(calls, [])
        self.assertEqual(push.missing_apns_env(), [
            "APNS_AUTH_KEY",
            "APNS_KEY_ID",
            "APNS_TEAM_ID",
            "APNS_LIVE_ACTIVITY_TOPIC",
        ])

    def test_tick_sends_end_and_drops_the_token(self):
        os.environ["APNS_AUTH_KEY"] = "test-key"
        os.environ["APNS_KEY_ID"] = "KEYID"
        os.environ["APNS_TEAM_ID"] = "TEAMID"
        os.environ["APNS_LIVE_ACTIVITY_TOPIC"] = "hp.launchBuddy.push-type.liveactivity"
        push._logged_disabled = False
        push.upsert_token(
            token="ab" * 16,
            launch_id="launch-1",
            activity_id="activity-1",
            net=_iso(NET),
            content_state=_content(NET, NET + timedelta(minutes=11)),
        )
        sent = []

        def sender(token, payload, priority=10):
            sent.append((token, payload, priority))
            return {"sent": True}

        result = push.tick(
            now=NET + timedelta(minutes=12),
            lookup=lambda _id: ({"net": _iso(NET), "status": "In Flight", "status_id": 6, "mission": "Starlink"}, None),
            sender=sender,
        )
        self.assertEqual(result["sent"], 1)
        self.assertEqual(sent[0][2], 10)
        self.assertEqual(sent[0][1]["aps"]["event"], "end")
        self.assertEqual(result["stored"], 0)

    def test_matching_registration_does_not_push_until_something_changes(self):
        os.environ["APNS_AUTH_KEY"] = "test-key"
        os.environ["APNS_KEY_ID"] = "KEYID"
        os.environ["APNS_TEAM_ID"] = "TEAMID"
        os.environ["APNS_LIVE_ACTIVITY_TOPIC"] = "hp.launchBuddy.push-type.liveactivity"
        push._logged_disabled = False
        dismiss = NET + timedelta(minutes=11)
        push.upsert_token(
            token="ab" * 16,
            launch_id="launch-1",
            activity_id="activity-1",
            net=_iso(NET),
            content_state=_content(NET, dismiss),
        )
        sent = []
        lookup = lambda _id: ({"net": _iso(NET), "status": "Go", "status_id": 1}, None)
        now = NET - timedelta(minutes=40)
        first = push.tick(now=now, lookup=lookup, sender=lambda *a, **k: sent.append(a) or {"sent": True})
        self.assertEqual(first["sent"], 0)
        self.assertEqual(sent, [])
        second = push.tick(now=now + timedelta(seconds=20), lookup=lookup, sender=lambda *a, **k: sent.append(a) or {"sent": True})
        self.assertEqual(second["sent"], 0)
        self.assertEqual(push._count(), 1)

    def test_rejects_non_hex_token(self):
        self.assertIsNone(push._normalize_token("not a token!!!!!"))
        self.assertEqual(push._normalize_token("AB" * 16), "ab" * 16)


if __name__ == "__main__":
    unittest.main()
