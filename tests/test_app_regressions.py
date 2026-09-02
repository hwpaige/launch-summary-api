import re
import unittest
from unittest.mock import Mock, patch

import app


def is_utc_iso8601(value):
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value))


class AppRegressionTests(unittest.TestCase):
    def test_parse_launch_data_prefers_image_url_and_sets_name_fields(self):
        launch = {
            "id": "launch-1",
            "name": "Test Mission",
            "net": "2026-09-02T14:21:21.428527+00:00",
            "window_start": "2026-09-02T14:20:00.123456+00:00",
            "window_end": "2026-09-02T15:20:00.654321+00:00",
            "status": {"name": "Go", "id": 1},
            "rocket": {
                "configuration": {"name": "Falcon 9"},
                "launcher_stage": [],
            },
            "mission": {"orbit": {"name": "LEO"}},
            "pad": {"name": "LC-39A"},
            "image": {
                "image_url": "https://example.com/image.jpg",
                "url": "https://example.com/url.jpg",
                "thumbnail_url": "https://example.com/thumb.jpg",
            },
        }

        parsed = app.parse_launch_data(launch)

        self.assertEqual(parsed["image"], "https://example.com/image.jpg")
        self.assertEqual(parsed["name"], "Test Mission")
        self.assertEqual(parsed["mission"], "Test Mission")
        self.assertTrue(is_utc_iso8601(parsed["net"]))
        self.assertTrue(is_utc_iso8601(parsed["window_start"]))
        self.assertTrue(is_utc_iso8601(parsed["window_end"]))

    def test_generate_narratives_keeps_full_existing_history(self):
        fake_response = Mock()
        fake_response.status_code = 200
        fake_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": 'launch_descriptions = ["9/2 1421: Fresh narrative"]'
                    }
                }
            ]
        }
        fake_launches = Mock()
        fake_launches.status_code = 200
        fake_launches.json.return_value = {
            "results": [
                {
                    "net": "2026-09-02T14:21:21Z",
                    "name": "Test Mission",
                    "pad": {"name": "LC-39A"},
                    "rocket": {"configuration": {"name": "Falcon 9"}},
                    "mission": {"orbit": {"name": "LEO"}},
                    "status": {"name": "Go"},
                }
            ]
        }

        existing = [f"9/1 1200: old narrative {idx}" for idx in range(40)]

        with patch.object(app.requests, "get", return_value=fake_launches), \
             patch.object(app.requests, "post", return_value=fake_response):
            result = app.generate_narratives(existing_narratives=existing)

        self.assertEqual(len(result), 41)
        self.assertEqual(result[0], "9/2 1421: Fresh narrative")
        self.assertEqual(result[1:], existing)

    def test_refresh_weather_internal_keeps_forecast_and_float_values(self):
        weather_payload = {
            "temperature_c": 31,
            "temperature_f": 87,
            "humidity": 50,
            "wind_speed_kts": 12,
            "wind_gust_kts": 15,
            "wind_direction": 180,
            "timestamp": "2026-09-02T14:21:21.428527+00:00",
        }
        forecast_payload = {
            "current_weather": {
                "temperature": 31,
                "windspeed": 12,
                "winddirection": 180,
            },
            "daily": {
                "temperature_2m_max": [31, 32],
                "temperature_2m_min": [21, 22],
                "weathercode": [1, 2],
            },
        }

        with patch.object(app, "r", None), \
             patch.object(app, "WEATHER_LOCATIONS", ["Cape"]), \
             patch.object(app, "fetch_weather", return_value=dict(weather_payload)), \
             patch.object(app, "fetch_forecast", return_value=forecast_payload):
            result = app.refresh_weather_internal()

        weather = result["weather"]["Cape"]
        self.assertTrue(is_utc_iso8601(weather["last_updated"]))
        self.assertEqual(weather["last_updated"], result["last_updated"])
        self.assertEqual(weather["temperature_c"], 31.0)
        self.assertEqual(weather["humidity"], 50.0)
        self.assertEqual(weather["forecast"]["current_weather"]["temperature"], 31.0)
        self.assertEqual(weather["forecast"]["daily"]["temperature_2m_max"][0], 31.0)
        self.assertTrue(is_utc_iso8601(weather["timestamp"]))

    def test_refresh_cache_uses_utc_timestamp(self):
        with patch.object(app, "refresh_narratives_internal", return_value=["a narrative"]):
            response = app.refresh_cache()

        self.assertEqual(response["count"], 1)
        self.assertTrue(is_utc_iso8601(response["timestamp"]))


if __name__ == "__main__":
    unittest.main()
