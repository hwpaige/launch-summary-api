"""Heroku web entry. Installs Live Activity routes on the FastAPI app.

``uvicorn app:app`` would skip the push worker. This module is the Procfile
target so token registration and the end-push loop start with the dyno.
"""

import app as app_module
import live_activity_push

live_activity_push.install(app_module.app)
if getattr(app_module, "_background_enabled", False):
    live_activity_push.start_worker()

app = app_module.app
