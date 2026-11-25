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
        # SECO-2 at T+53 is the last ascent cutoff. SECO-10 and Starship Landing are not.
        self.assertEqual(plan["milestone_source"], "SECO-2")
        self.assertEqual(plan["milestone"], NET + timedelta(seconds=53 * 60))
        self.assertEqual(plan["dismissal"], NET + timedelta(seconds=53 * 60 + 120))

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
        self.assertEqual(plan["milestone_source"], "T+9")
        self.assertEqual(plan["milestone"], NET + timedelta(minutes=9))
        self.assertEqual(plan["dismissal"], NET + timedelta(minutes=11))

    def test_hard_cap_clips_linger(self):
        # A T+50 SECO-1 is inside the ascent window, so the old T+20 cap no longer clips it.
        plan = push.activity_end_plan(NET, [("SECO-1", 50 * 60)])
        self.assertEqual(plan["milestone_source"], "SECO-1")
        self.assertEqual(plan["milestone"], NET + timedelta(minutes=50))
        self.assertEqual(plan["dismissal"], NET + timedelta(minutes=52))
        # The two-minute linger still cannot run past T+80.
        plan = push.activity_end_plan(NET, [("Stage 1 Landing", 90 * 60)])
        self.assertEqual(plan["milestone"], NET + timedelta(minutes=80))
        self.assertEqual(plan["dismissal"], NET + timedelta(minutes=80))

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

        launch["status"] = "Partial Failure"
        launch["status_id"] = 7
        action = push.decide(record, launch, None, now)
        self.assertEqual(action["kind"], "end")
        self.assertEqual(action["payload"]["aps"]["content-state"]["finalPhaseLabel"], "Failure")

    def test_hold_before_net_does_not_end(self):
        now = NET - timedelta(hours=2)
        record = {"token": "ab" * 16, "launch_id": "launch-1", "content_state": _content(NET, NET + timedelta(minutes=11))}
        launch = {"net": _iso(NET), "status": "Hold", "status_id": 1}
        action = push.decide(record, launch, None, now)
        # The phone is not showing a hold yet, so it is told. It is not ended.
        self.assertEqual(action["kind"], "update")
        self.assertEqual(action["payload"]["aps"]["event"], "update")

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
        self.assertEqual(state["milestoneSource"], "SECO-1")

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
        # A Go row before liftoff: the app publishes T+30, not the T+11 flight end.
        dismiss = NET + timedelta(minutes=30)
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

    # --- end-rule audit (real LL2 timelines, Oct 2026) ---

    def _apns_env(self):
        os.environ["APNS_AUTH_KEY"] = "test-key"
        os.environ["APNS_KEY_ID"] = "KEYID"
        os.environ["APNS_TEAM_ID"] = "TEAMID"
        os.environ["APNS_LIVE_ACTIVITY_TOPIC"] = "hp.launchBuddy.push-type.liveactivity"
        push._logged_disabled = False

    def test_falcon_heavy_core_landing_after_side_boosters_wins(self):
        # No in-window SECO-2: the later core landing beats the side boosters and SECO-1.
        # Core Landing and Center Core Landing use the same booster-landed text.
        for core in ("Core Landing", "Center Core Landing"):
            plan = push.activity_end_plan(NET, [
                ("Boosters Landing", 8 * 60 + 11),
                ("SECO-1", 8 * 60 + 23),
                (core, 9 * 60 + 40),
                ("SECO-3", 4 * 3600 + 21 * 60 + 51),
            ])
            self.assertEqual(plan["label"], "Booster landed", core)
            self.assertEqual(plan["milestone_source"], core)
            self.assertEqual(plan["milestone"], NET + timedelta(seconds=9 * 60 + 40), core)
            self.assertEqual(plan["dismissal"], NET + timedelta(seconds=9 * 60 + 40 + 120), core)

    def test_starship_bare_seco_and_catch_names(self):
        plan = push.activity_end_plan(NET, [
            ("Stage 1 Splashdown", 6 * 60 + 50),
            ("Booster Catch", 6 * 60 + 56),
            ("SECO", 8 * 60 + 27),
            ("SEB-2", 37 * 60),
            ("Landing Flip", 65 * 60),
            ("Starship Landing", 65 * 60 + 34),
        ])
        self.assertEqual(plan["label"], "SECO \u2713")
        self.assertEqual(plan["milestone_source"], "SECO")
        self.assertEqual(plan["milestone"], NET + timedelta(seconds=8 * 60 + 27))

    def test_matchers_ignore_burns_ship_and_later_secos(self):
        for label in ("Stage 1 Landing Burn", "Boosters Landing Burn", "Starship Landing",
                      "Landing Flip", "Dragon Splashdown", "Starship Splashdown"):
            self.assertFalse(push._is_stage1_landing(label), label)
        for label in ("Super Heavy Catch", "Super Heavy Landing", "Center Core Landing",
                      "Core Landing", "Stage 1 Splashdown", "Booster Catch"):
            self.assertTrue(push._is_stage1_landing(label), label)
        for label in ("SECO-2", "SECO 2", "SECO-10", "SBECO", "MECO", "SEB-2"):
            self.assertFalse(push._is_seco1(label), label)
        for label in ("SECO-1", "SECO 1", "SECO", "Ship SECO"):
            self.assertTrue(push._is_seco1(label), label)

    def test_hold_after_net_does_not_end(self):
        # Starship Flight 9: hold at T-40s, flew ~6 minutes late.
        record = {"token": "ab" * 16, "launch_id": "launch-1", "content_state": _content(NET, NET + timedelta(minutes=11))}
        launch = {"net": _iso(NET), "status": "On Hold", "status_id": 5}
        for later in (timedelta(seconds=30), timedelta(minutes=12), timedelta(minutes=45)):
            action = push.decide(record, launch, None, NET + later)
            self.assertNotEqual((action or {}).get("kind"), "end", later)

    def test_milestone_waits_for_liftoff_evidence(self):
        record = {"token": "ab" * 16, "launch_id": "launch-1", "content_state": _content(NET, NET + timedelta(minutes=11))}
        launch = {"net": _iso(NET), "status": "Go for Launch", "status_id": 1}
        self.assertNotEqual((push.decide(record, launch, None, NET + timedelta(minutes=10)) or {}).get("kind"), "end")
        launch = {"net": _iso(NET), "status": "Launch in Flight", "status_id": 6}
        self.assertEqual(push.decide(record, launch, None, NET + timedelta(minutes=10))["kind"], "end")
        launch = {"net": _iso(NET), "status": "Go for Launch", "status_id": 1}
        self.assertEqual(push.decide(record, launch, None, NET + timedelta(minutes=31))["kind"], "end")

    def test_success_before_milestone_waits_for_milestone(self):
        record = {"token": "ab" * 16, "launch_id": "launch-1", "content_state": _content(NET, NET + timedelta(minutes=11))}
        raw = {"net": _iso(NET), "status": {"id": 3, "name": "Launch Successful"},
               "timeline": [{"type": {"abbrev": "Stage 1 Landing"}, "relative_time": "PT8M20S"},
                            {"type": {"abbrev": "SECO-1"}, "relative_time": "PT8M39S"}]}
        self.assertNotEqual((push.decide(record, None, raw, NET + timedelta(minutes=8)) or {}).get("kind"), "end")
        action = push.decide(record, None, raw, NET + timedelta(seconds=8 * 60 + 40))
        self.assertEqual(action["kind"], "end")
        self.assertEqual(action["payload"]["aps"]["dismissal-date"], int((NET + timedelta(seconds=8 * 60 + 39 + 120)).timestamp()))

    def test_tick_rechecks_fresh_data_before_ending(self):
        # Cache still has the old T0; Launch Library already moved it 6 minutes.
        self._apns_env()
        push.upsert_token(token="ab" * 16, launch_id="launch-1", activity_id="activity-1",
                          net=_iso(NET), content_state=_content(NET, NET + timedelta(minutes=11)))
        recycled = NET + timedelta(minutes=6)
        sent = []
        result = push.tick(
            now=NET + timedelta(minutes=9),
            lookup=lambda _id: ({"net": _iso(NET), "status": "Launch in Flight", "status_id": 6}, None),
            refresh=lambda _id: ({"net": _iso(recycled), "status": "Launch in Flight", "status_id": 6}, None),
            sender=lambda token, payload, priority=10: sent.append(payload) or {"sent": True},
        )
        self.assertTrue(all(p["aps"].get("event") != "end" for p in sent))
        self.assertEqual(result["stored"], 1)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["aps"]["event"], "update")
        self.assertAlmostEqual(sent[0]["aps"]["content-state"]["net"], recycled.timestamp() - SWIFT, places=3)

    def test_late_scrub_without_confirmation_waits_until_t_plus_30(self):
        # Scrub already happened; Launch Library still says Go (audit scenario C).
        record = {"token": "ab" * 16, "launch_id": "launch-1", "content_state": _content(NET, NET + timedelta(minutes=11))}
        launch = {"net": _iso(NET), "status": "Go for Launch", "status_id": 1}
        self.assertNotEqual(
            (push.decide(record, launch, None, NET + timedelta(seconds=10 * 60 + 40)) or {}).get("kind"),
            "end",
        )
        self.assertEqual(push.decide(record, launch, None, NET + timedelta(minutes=30))["kind"], "end")
        held = {"net": _iso(NET), "status": "On Hold", "status_id": 5}
        self.assertNotEqual(
            (push.decide(record, held, None, NET + timedelta(minutes=30)) or {}).get("kind"),
            "end",
        )

    def test_starlink_seco2_ends_near_t_plus_53(self):
        # Starlink 15-18: SECO-2 deorbit at T+53:08, after SECO-1 and the droneship.
        raw = {"timeline": [
            {"type": {"abbrev": "Stage 1 Landing Burn"}, "relative_time": "PT7M58S"},
            {"type": {"abbrev": "Stage 1 Landing"}, "relative_time": "PT8M20S"},
            {"type": {"abbrev": "SECO-1"}, "relative_time": "PT8M39S"},
            {"type": {"abbrev": "SECO-2"}, "relative_time": "PT53M8S"},
        ]}
        record = {"token": "ab" * 16, "launch_id": "launch-1", "content_state": _content(NET, NET + timedelta(minutes=11))}
        flown = {"net": _iso(NET), "status": "Launch in Flight", "status_id": 6}
        self.assertNotEqual(
            (push.decide(record, flown, raw, NET + timedelta(seconds=8 * 60 + 39)) or {}).get("kind"),
            "end",
        )
        # An early Success still waits for that later ascent SECO.
        success = {"net": _iso(NET), "status": "Launch Successful", "status_id": 3}
        self.assertNotEqual((push.decide(record, success, raw, NET + timedelta(minutes=10)) or {}).get("kind"), "end")
        action = push.decide(record, flown, raw, NET + timedelta(seconds=53 * 60 + 8))
        self.assertEqual(action["kind"], "end")
        state = action["payload"]["aps"]["content-state"]
        self.assertEqual(state["milestoneSource"], "SECO-2")
        self.assertEqual(state["finalPhaseLabel"], "SECO \u2713")
        milestone = NET + timedelta(seconds=53 * 60 + 8)
        self.assertEqual(action["payload"]["aps"]["dismissal-date"], int((milestone + timedelta(seconds=120)).timestamp()))
        self.assertAlmostEqual(state["activityMilestone"], milestone.timestamp() - SWIFT, places=3)
        # Starlink 15-16 is the same shape, SECO-2 at T+53:03.
        plan = push.activity_end_plan(NET, [
            ("Stage 1 Landing", 8 * 60 + 23),
            ("SECO-1", 8 * 60 + 39),
            ("SECO-2", 53 * 60 + 3),
        ])
        self.assertEqual(plan["milestone_source"], "SECO-2")
        self.assertEqual(plan["milestone"], NET + timedelta(seconds=53 * 60 + 3))

    def test_gto_seco2_around_t_plus_28_is_the_end(self):
        # Representative GTO second burn, and O3b mPower 11-13's real T+30:38.
        plan = push.activity_end_plan(NET, [
            ("SECO-1", 8 * 60 + 2),
            ("Stage 1 Landing", 8 * 60 + 46),
            ("SECO-2", 28 * 60),
        ])
        self.assertEqual(plan["label"], "SECO \u2713")
        self.assertEqual(plan["milestone_source"], "SECO-2")
        self.assertEqual(plan["milestone"], NET + timedelta(minutes=28))
        self.assertEqual(plan["dismissal"], NET + timedelta(minutes=30))
        plan = push.activity_end_plan(NET, [
            ("SECO-1", 8 * 60 + 2),
            ("Stage 1 Landing Burn", 8 * 60 + 23),
            ("Stage 1 Landing", 8 * 60 + 46),
            ("SECO-2", 30 * 60 + 38),
        ])
        self.assertEqual(plan["milestone_source"], "SECO-2")
        self.assertEqual(plan["milestone"], NET + timedelta(seconds=30 * 60 + 38))
        self.assertEqual(plan["dismissal"], NET + timedelta(seconds=30 * 60 + 38 + 120))

    def test_goes_u_ignores_seco3_and_keeps_seco2(self):
        # Falcon Heavy GOES-U: side boosters RTLS, core expended, SECO-3 at T+4:21:51.
        events = [
            ("Boosters Landing Burn", 7 * 60 + 53),
            ("Boosters Landing", 8 * 60 + 11),
            ("SECO-1", 8 * 60 + 23),
            ("SECO-2", 27 * 60 + 46),
            ("SECO-3", 4 * 3600 + 21 * 60 + 51),
        ]
        plan = push.activity_end_plan(NET, events)
        self.assertEqual(plan["milestone_source"], "SECO-2")
        self.assertEqual(plan["label"], "SECO \u2713")
        self.assertEqual(plan["milestone"], NET + timedelta(seconds=27 * 60 + 46))
        self.assertEqual(plan["dismissal"], NET + timedelta(seconds=27 * 60 + 46 + 120))
        # The same SECO-2 also beats a recovered center core.
        plan = push.activity_end_plan(NET, events + [("Center Core Landing", 9 * 60 + 40)])
        self.assertEqual(plan["milestone_source"], "SECO-2")
        self.assertEqual(plan["milestone"], NET + timedelta(seconds=27 * 60 + 46))
        # SECO-3 before T+75 is still not an ascent cutoff. SECO-2 past T+75 falls back.
        plan = push.activity_end_plan(NET, [
            ("SECO-1", 8 * 60 + 23),
            ("Boosters Landing", 8 * 60 + 11),
            ("SECO-3", 50 * 60),
            ("SECO-10", 40 * 60),
            ("SECO-2", 75 * 60 + 1),
        ])
        self.assertEqual(plan["milestone_source"], "SECO-1")
        self.assertEqual(plan["milestone"], NET + timedelta(seconds=8 * 60 + 23))
        plan = push.activity_end_plan(NET, [("SECO-1", 8 * 60 + 23), ("SECO-2", 75 * 60)])
        self.assertEqual(plan["milestone_source"], "SECO-2")
        self.assertEqual(plan["milestone"], NET + timedelta(minutes=75))
        self.assertEqual(plan["dismissal"], NET + timedelta(minutes=77))

    def test_europa_clipper_ends_at_seco2(self):
        # All three cores expended. SECO-2 at T+50:56 is the ascent cutoff.
        plan = push.activity_end_plan(NET, [
            ("Boosters Separation", 3 * 60 + 7),
            ("SECO-1", 7 * 60 + 53),
            ("SECO-2", 50 * 60 + 56),
        ])
        self.assertEqual(plan["milestone_source"], "SECO-2")
        self.assertEqual(plan["label"], "SECO \u2713")
        self.assertEqual(plan["milestone"], NET + timedelta(seconds=50 * 60 + 56))
        self.assertEqual(plan["dismissal"], NET + timedelta(seconds=50 * 60 + 56 + 120))

    def test_starship_flights_end_at_ship_seco_not_the_landing(self):
        # Ship SECO is SECO-1 in Launch Library. Starship Landing stays ignored.
        flights = {
            "IFT-5": [
                ("Stage 1 Splashdown", 6 * 60 + 50),
                ("Stage 1 Landing", 6 * 60 + 56),
                ("SECO-1", 8 * 60 + 27),
                ("Landing Flip", 65 * 60 + 15),
                ("Starship Landing", 65 * 60 + 34),
            ],
            "Flight 7": [
                ("Stage 1 Landing Burn", 6 * 60 + 35),
                ("Stage 1 Landing", 6 * 60 + 55),
                ("SECO-1", 8 * 60 + 53),
                ("SEB-2", 37 * 60 + 33),
                ("Landing Flip", 66 * 60 + 13),
                ("Starship Landing", 66 * 60 + 39),
            ],
            "Flight 10": [
                ("Stage 1 Landing", 6 * 60 + 40),
                ("SECO-1", 8 * 60 + 57),
                ("SEB-2", 37 * 60 + 48),
                ("Landing Flip", 66 * 60 + 14),
                ("Starship Landing Burn", 66 * 60 + 20),
                ("Starship Landing", 66 * 60 + 30),
            ],
            "Flight 14": [
                ("Stage 1 Landing Burn", 6 * 60 + 35),
                ("Stage 1 Landing", 7 * 60 + 1),
                ("SECO-1", 8 * 60 + 11),
                ("Landing Flip", 9 * 3600 + 50 * 60 + 13),
                ("Starship Landing", 9 * 3600 + 50 * 60 + 30),
            ],
        }
        expected = {
            "IFT-5": 8 * 60 + 27,
            "Flight 7": 8 * 60 + 53,
            "Flight 10": 8 * 60 + 57,
            "Flight 14": 8 * 60 + 11,
        }
        for name, events in flights.items():
            plan = push.activity_end_plan(NET, events)
            self.assertEqual(plan["milestone_source"], "SECO-1", name)
            self.assertEqual(plan["label"], "SECO \u2713", name)
            self.assertEqual(plan["milestone"], NET + timedelta(seconds=expected[name]), name)
            self.assertEqual(plan["dismissal"], NET + timedelta(seconds=expected[name] + 120), name)
        # A ship label that drops the -1, and a Super Heavy catch, still end on the ship SECO.
        plan = push.activity_end_plan(NET, [
            ("Super Heavy Catch", 6 * 60 + 56),
            ("Ship SECO", 8 * 60 + 27),
            ("Starship Landing", 65 * 60 + 34),
        ])
        self.assertEqual(plan["milestone_source"], "Ship SECO")
        self.assertEqual(plan["label"], "SECO \u2713")
        self.assertEqual(plan["milestone"], NET + timedelta(seconds=8 * 60 + 27))

    def test_remaining_audit_flights_keep_seco1_recovery_or_t_plus_9(self):
        cases = [
            ("USSF-481", [("Stage 1 Landing Burn", 8 * 60), ("Stage 1 Landing", 8 * 60 + 9)], 8 * 60 + 9, "Stage 1 Landing", "Booster landed"),
            ("CRS SpX-35", [("Stage 1 Landing", 7 * 60 + 21), ("SECO-1", 8 * 60 + 30)], 8 * 60 + 30, "SECO-1", "SECO \u2713"),
            ("SDA T1TL-A", [("Stage 1 Landing", 7 * 60 + 19)], 7 * 60 + 19, "Stage 1 Landing", "Booster landed"),
            ("Galileo L12", [("MECO", 2 * 60 + 45), ("SES-1", 2 * 60 + 55), ("Fairing Separation", 3 * 60 + 39)], 9 * 60, "T+9", "SECO \u2713"),
            ("NROL-97", [("Boosters Landing Burn", 7 * 60 + 30), ("Boosters Landing", 7 * 60 + 47)], 7 * 60 + 47, "Boosters Landing", "Booster landed"),
            ("Roman", [("Boosters Landing", 7 * 60 + 40), ("SECO-1", 8 * 60 + 28), ("SECO-2", 26 * 60 + 32)], 26 * 60 + 32, "SECO-2", "SECO \u2713"),
        ]
        for name, events, seconds, source, label in cases:
            plan = push.activity_end_plan(NET, events)
            self.assertEqual(plan["milestone_source"], source, name)
            self.assertEqual(plan["label"], label, name)
            self.assertEqual(plan["milestone"], NET + timedelta(seconds=seconds), name)
            self.assertEqual(plan["dismissal"], NET + timedelta(seconds=seconds + 120), name)

    # --- slips, holds and rejected pushes close to liftoff (2026-11-24) ---

    def _app_content(self, net, registered):
        """What a real phone registers before liftoff: Go, T+30 gate, [lower, upper] ranges."""
        gate = net + timedelta(minutes=30)
        return {
            "missionName": "NROL-136",
            "net": net.timestamp() - SWIFT,
            "statusLabel": "Go for Launch",
            "phase": "upcoming",
            "timelineEvents": [
                {"label": "Engine Chill", "relativeSeconds": -420},
                {"label": "Liftoff", "relativeSeconds": 0},
                {"label": "Max-Q", "relativeSeconds": 62},
            ],
            "preLiftoffLabel": "Engine Chill",
            "preLiftoffAbbrev": "Eng Chill",
            "postLiftoffLabel": "Liftoff",
            "postLiftoffAbbrev": "Liftoff",
            "timerInterval": [registered.timestamp() - SWIFT, net.timestamp() - SWIFT],
            "timerCountsDown": True,
            "activityMilestone": gate.timestamp() - SWIFT,
            "activityDismissAt": gate.timestamp() - SWIFT,
            "finalPhaseLabel": "SECO \u2713",
            "padWeather": {"f": 46, "w": 25},
        }

    def _settled(self, net, now):
        """A registration the loop has already looked at once."""
        return {
            "token": "ab" * 16,
            "launch_id": "launch-1",
            "content_state": self._app_content(net, now - timedelta(minutes=30)),
            "pushed_net_unix": net.timestamp(),
            "pushed_dismiss_unix": (net + timedelta(minutes=30)).timestamp(),
            "pushed_hold": False,
        }

    def _raw(self, net, status="Go for Launch", status_id=1):
        return {
            "net": _iso(net),
            "status": {"id": status_id, "name": status},
            "timeline": [
                {"type": {"abbrev": "Stage 1 Landing"}, "relative_time": "PT7M34S"},
                {"type": {"abbrev": "SECO-1"}, "relative_time": "PT8M40S"},
            ],
        }

    def test_go_row_registration_matches_the_liftoff_gate(self):
        # The flight end is T+8:40. Before liftoff is known the card must not carry it.
        now = NET - timedelta(minutes=20)
        record = {"token": "ab" * 16, "launch_id": "launch-1",
                  "content_state": self._app_content(NET, now - timedelta(minutes=30))}
        self.assertEqual(push.decide(record, None, self._raw(NET), now)["kind"], "note")

    def test_liftoff_confirmation_publishes_the_real_end(self):
        now = NET + timedelta(minutes=1)
        action = push.decide(self._settled(NET, now), None, self._raw(NET, "Launch in Flight", 6), now)
        self.assertEqual(action["kind"], "update")
        expected = (NET + timedelta(seconds=8 * 60 + 40 + 120)).timestamp()
        self.assertEqual(action["payload"]["aps"]["stale-date"], int(expected))
        self.assertAlmostEqual(action["payload"]["aps"]["content-state"]["activityDismissAt"], expected - SWIFT, places=3)

    def test_slip_keeps_the_gate_and_shifts_the_apps_range_shape(self):
        now = NET - timedelta(minutes=20)
        slipped = NET + timedelta(minutes=15)
        record = self._settled(NET, now)
        action = push.decide(record, None, self._raw(slipped), now)
        self.assertEqual(action["kind"], "update")
        state = action["payload"]["aps"]["content-state"]
        self.assertAlmostEqual(state["net"], slipped.timestamp() - SWIFT, places=3)
        gate = (slipped + timedelta(minutes=30)).timestamp()
        self.assertAlmostEqual(state["activityDismissAt"], gate - SWIFT, places=3)
        self.assertEqual(action["payload"]["aps"]["stale-date"], int(gate))
        before = record["content_state"]["timerInterval"]
        self.assertEqual(state["timerInterval"], [before[0] + 900, before[1] + 900])
        self.assertEqual(state["padWeather"], {"f": 46, "w": 25})

    def test_hold_is_pushed_once_and_cleared_when_it_ends(self):
        now = NET - timedelta(minutes=2)
        record = self._settled(NET, now)
        held = self._raw(NET, "On Hold", 5)
        action = push.decide(record, None, held, now)
        self.assertEqual(action["kind"], "update")
        self.assertTrue(action["hold"])
        state = action["payload"]["aps"]["content-state"]
        self.assertEqual(state["phase"], "scrubbed")
        self.assertEqual(state["statusLabel"], "On Hold")
        self.assertEqual(state["preLiftoffLabel"], "On Hold")
        self.assertEqual(state["postLiftoffLabel"], "On Hold")
        self.assertEqual(state["finalPhaseLabel"], "On Hold")
        self.assertNotIn("timelineEvents", state)
        # Never stale into a flight-end line while holding.
        self.assertGreater(action["payload"]["aps"]["stale-date"], (NET + timedelta(hours=5)).timestamp())
        # The stored copy keeps the countdown so the hold can be undone.
        self.assertEqual(action["store_content"]["phase"], "upcoming")
        self.assertIn("timelineEvents", action["store_content"])

        record.update(pushed_hold=True, content_state=action["store_content"])
        for later in (timedelta(seconds=15), timedelta(minutes=12), timedelta(minutes=45)):
            self.assertIsNone(push.decide(record, None, held, now + later), later)

        recycled = NET + timedelta(minutes=6)
        action = push.decide(record, None, self._raw(recycled), NET + timedelta(minutes=1))
        self.assertEqual(action["kind"], "update")
        self.assertFalse(action["hold"])
        state = action["payload"]["aps"]["content-state"]
        self.assertEqual(state["phase"], "upcoming")
        self.assertEqual(state["statusLabel"], "Go for Launch")
        self.assertEqual(state["postLiftoffLabel"], "Liftoff")
        self.assertAlmostEqual(state["net"], recycled.timestamp() - SWIFT, places=3)

    def test_hold_the_phone_already_shows_is_not_pushed_again(self):
        now = NET - timedelta(minutes=2)
        content = self._app_content(NET, now - timedelta(minutes=30))
        content.update(phase="scrubbed", statusLabel="On Hold", postLiftoffLabel="On Hold", postLiftoffAbbrev="Hold")
        record = {"token": "ab" * 16, "launch_id": "launch-1", "content_state": content}
        action = push.decide(record, None, self._raw(NET, "On Hold", 5), now)
        self.assertEqual(action["kind"], "note")
        self.assertTrue(action["hold"])
        # Cleared from the app's own hold state: the hold words go, the app refills them.
        record.update(pushed_net_unix=NET.timestamp(), pushed_dismiss_unix=action["dismiss_unix"], pushed_hold=True)
        action = push.decide(record, None, self._raw(NET + timedelta(minutes=10)), now)
        state = action["payload"]["aps"]["content-state"]
        self.assertEqual(state["phase"], "upcoming")
        self.assertNotIn("postLiftoffLabel", state)
        self.assertEqual(state["preLiftoffLabel"], "Engine Chill")

    def test_new_registration_from_the_app_resets_the_hold_flag(self):
        push.upsert_token(token="ab" * 16, launch_id="launch-1", activity_id="a1", net=_iso(NET), content_state={"net": 1.0})
        with push._lock:
            stored = push._get("a1")
            stored["pushed_hold"] = True
            push._put("a1", stored)
        self.assertTrue(push.upsert_token(token="ab" * 16, launch_id="launch-1", activity_id="a1", net=None, content_state=None)["pushed_hold"])
        self.assertIsNone(push.upsert_token(token="ab" * 16, launch_id="launch-1", activity_id="a1", net=None, content_state={"net": 2.0})["pushed_hold"])

    def test_scrub_to_another_day_ends_the_countdown(self):
        now = NET - timedelta(minutes=20)
        tomorrow = NET + timedelta(hours=24)
        action = push.decide(self._settled(NET, now), None, self._raw(tomorrow, "To Be Confirmed", 8), now)
        self.assertEqual(action["kind"], "end")
        aps = action["payload"]["aps"]
        self.assertEqual(aps["event"], "end")
        self.assertEqual(aps["dismissal-date"], int((now + timedelta(seconds=60)).timestamp()))
        state = aps["content-state"]
        self.assertEqual(state["statusLabel"], "Delayed")
        self.assertEqual(state["phase"], "complete")
        self.assertEqual(state["preLiftoffLabel"], "Delayed")
        self.assertNotIn("timelineEvents", state)
        self.assertLess(state["activityMilestone"], now.timestamp() - SWIFT)
        # Also after the old T0 has passed on the phone.
        late = NET + timedelta(minutes=3)
        self.assertEqual(push.decide(self._settled(NET, late), None, self._raw(tomorrow), late)["kind"], "end")

    def test_slip_inside_the_launch_window_keeps_the_countdown(self):
        # The app cannot bring a countdown back while the phone is locked.
        now = NET - timedelta(minutes=20)
        for slip in (timedelta(minutes=35), timedelta(minutes=90), timedelta(hours=3)):
            action = push.decide(self._settled(NET, now), None, self._raw(NET + slip), now)
            self.assertEqual(action["kind"], "update", slip)
        action = push.decide(self._settled(NET, now), None, self._raw(NET + timedelta(hours=5)), now)
        self.assertEqual(action["kind"], "end")

    def test_countdown_not_on_screen_yet_is_not_ended_by_a_slip(self):
        # Scheduled to appear at T-1h, five hours away. Its NET moving is not an end.
        now = NET - timedelta(hours=5)
        action = push.decide(self._settled(NET, now), None, self._raw(NET + timedelta(hours=24)), now)
        self.assertEqual(action["kind"], "update")
        fresh = {"token": "ab" * 16, "launch_id": "launch-1",
                 "content_state": self._app_content(NET, now)}
        self.assertEqual(push.decide(fresh, None, self._raw(NET), now)["kind"], "note")

    def test_loop_reads_fresh_data_inside_the_hour(self):
        self._apns_env()
        now = NET - timedelta(minutes=20)
        slipped = NET + timedelta(minutes=15)
        for activity in ("a1", "a2"):
            push.upsert_token(token="ab" * 16, launch_id="launch-1", activity_id=activity, net=_iso(NET),
                              content_state=self._app_content(NET, now - timedelta(minutes=30)))
        looked, refreshed, sent = [], [], []
        result = push.tick(
            now=now,
            lookup=lambda _id: (looked.append(_id) or ({"net": _iso(NET), "status": "Go for Launch", "status_id": 1}, self._raw(NET))),
            refresh=lambda _id: (refreshed.append(_id) or (None, self._raw(slipped))),
            sender=lambda token, payload, priority=10: sent.append(payload) or {"sent": True},
        )
        self.assertEqual(result["sent"], 2)
        # Two phones, one launch: one cached lookup and one fresh read.
        self.assertEqual(looked, ["launch-1"])
        self.assertEqual(refreshed, ["launch-1"])
        self.assertAlmostEqual(sent[0]["aps"]["content-state"]["net"], slipped.timestamp() - SWIFT, places=3)

    def test_loop_uses_the_list_copy_outside_the_hour_and_for_unknown_launches(self):
        self._apns_env()
        refreshed = []
        push.upsert_token(token="ab" * 16, launch_id="launch-1", activity_id="a1", net=_iso(NET),
                          content_state=self._app_content(NET, NET - timedelta(hours=4)))
        push.tick(
            now=NET - timedelta(hours=3),
            lookup=lambda _id: ({"net": _iso(NET), "status": "Go for Launch", "status_id": 1}, self._raw(NET)),
            refresh=lambda _id: refreshed.append(_id) or (None, self._raw(NET)),
            sender=lambda *a, **k: {"sent": True},
        )
        self.assertEqual(refreshed, [])
        # A debug simulation: no list row, no stored record. Inside the hour, still no read.
        self.assertFalse(push._wants_fresh({"net": _iso(NET)}, None, None, NET - timedelta(minutes=5)))
        self.assertTrue(push._wants_fresh({}, None, self._raw(NET), NET - timedelta(minutes=5)))
        self.assertTrue(push._wants_fresh({}, None, self._raw(NET, "On Hold", 5), NET + timedelta(hours=3)))
        self.assertFalse(push._wants_fresh({}, None, self._raw(NET), NET + timedelta(hours=3)))

    def test_default_refresh_never_asks_launch_library_about_an_unknown_id(self):
        class Stub:
            @staticmethod
            def _load_launch_payload(force=False):
                return {"upcoming": [{"id": "real-1", "net": _iso(NET)}], "previous": []}

            @staticmethod
            def _get_raw_launch(launch_id):
                return None

            asked = []

            @classmethod
            def _refresh_hot_launch_raw(cls, launch_id):
                cls.asked.append(launch_id)
                return {"net": _iso(NET)}

        original = push._app
        push._app = lambda: Stub
        try:
            self.assertEqual(push._default_refresh("debug-sim-1"), (None, None))
            self.assertEqual(Stub.asked, [])
            launch, raw = push._default_refresh("real-1")
            self.assertEqual(Stub.asked, ["real-1"])
            self.assertEqual(raw, {"net": _iso(NET)})
        finally:
            push._app = original

    def test_rejected_token_is_dropped_and_other_failures_back_off(self):
        self._apns_env()
        now = NET - timedelta(minutes=20)
        lookup = lambda _id: (None, self._raw(NET + timedelta(minutes=15)))

        def register(activity):
            push.upsert_token(token="ab" * 16, launch_id="launch-1", activity_id=activity, net=_iso(NET),
                              content_state=self._app_content(NET, now - timedelta(minutes=30)))
            with push._lock:
                stored = push._get(activity)
                stored.update(pushed_net_unix=NET.timestamp(), pushed_dismiss_unix=(NET + timedelta(minutes=30)).timestamp())
                push._put(activity, stored)

        register("dead")
        tries = []
        push.tick(now=now, lookup=lookup,
                  sender=lambda *a, **k: tries.append(1) or {"sent": False, "unregistered": True, "reason": "BadDeviceToken"})
        self.assertEqual(len(tries), 1)
        self.assertEqual(push._count(), 0)

        register("flaky")
        tries = []
        for step in range(9):  # 0s … 120s, a pass every 15 seconds
            push.tick(now=now + timedelta(seconds=15 * step), lookup=lookup,
                      sender=lambda *a, **k: tries.append(1) or {"sent": False, "status": 503})
        # 0s, then 30s later, then 60s after that: three tries, not nine.
        self.assertEqual(len(tries), 3)
        self.assertEqual(push._count(), 1)
        sent = []
        push.tick(now=now + timedelta(seconds=200), lookup=lookup,
                  sender=lambda token, payload, priority=10: sent.append(payload) or {"sent": True})
        self.assertEqual(sent, [])  # the third failure waits 120s
        push.tick(now=now + timedelta(seconds=215), lookup=lookup,
                  sender=lambda token, payload, priority=10: sent.append(payload) or {"sent": True})
        self.assertEqual(len(sent), 1)
        self.assertNotIn("flaky", push._send_backoff)

    def test_apns_reason_is_read_from_the_body(self):
        self.assertEqual(push._apns_reason('{"reason":"BadDeviceToken"}'), "BadDeviceToken")
        self.assertIn("BadDeviceToken", push.DEAD_TOKEN_REASONS)
        self.assertNotIn("TooManyRequests", push.DEAD_TOKEN_REASONS)
        self.assertIsNone(push._apns_reason(""))
        self.assertIsNone(push._apns_reason("not json"))

    def test_rejects_non_hex_token(self):
        self.assertIsNone(push._normalize_token("not a token!!!!!"))
        self.assertEqual(push._normalize_token("AB" * 16), "ab" * 16)


if __name__ == "__main__":
    unittest.main()
