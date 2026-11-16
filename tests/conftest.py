"""Keep the Heroku background poller off while the suite imports the app."""
import os

os.environ["STARSHIP_POLLER"] = "0"

import app  # noqa: E402

app._background_enabled = False
