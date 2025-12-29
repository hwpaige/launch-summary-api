# SpaceX Launch Summary API

A FastAPI application that fetches recent SpaceX launches and generates witty, "Cities Skylines" style narratives using the xAI Grok API.

## Features

- **Witty Narratives**: Generates short, dry, and technical descriptions of SpaceX launches.
- **FastAPI**: Modern, high-performance web framework.
- **Beautiful Dashboard**: Built-in UI to monitor narratives and API metrics (Requests, Cache Hits, Grok API usage).
- **Redis Caching**: Caches narratives to improve performance and minimize API costs.
- **Heroku Ready**: Optimized for Heroku deployment.

## API Endpoints

- `GET /`: The Dashboard UI.
- `GET /recent_launches_narratives`: Returns narratives as a JSON list.
- `GET /metrics`: Returns application performance metrics.
- `POST /refresh`: Forces a refresh of the narrative cache.

## Local Setup

1. **Clone the repository.**
2. **Create a virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
4. **Set environment variables:**
   Create a `.env` file:
   ```env
   XAI_API_KEY=your_xai_api_key_here
   REDIS_URL=redis://localhost:6379/0
   ```
5. **Run the application:**
   ```powershell
   python app.py
   ```
   Access the dashboard at [http://localhost:5000](http://localhost:5000).

   *Note: If you see "Could not find platform independent libraries <prefix>" on Windows, ensure your `PYTHONHOME` is set or run via the virtual environment's python directly: `.\.venv\Scripts\python.exe app.py`.*

## Deployment to Heroku

1. **Create a new Heroku app:**
   ```bash
   heroku create your-app-name
   ```
2. **Add Heroku Redis add-on:**
   ```bash
   heroku addons:create heroku-redis:mini -a your-app-name
   ```
3. **Set the xAI API Key:**
   ```bash
   heroku config:set XAI_API_KEY=your_xai_api_key_here -a your-app-name
   ```
4. **Deploy the code:**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push heroku main
   ```
5. **Set up periodic refreshes (Optional):**
   Use the Heroku Scheduler add-on to call the `/refresh` endpoint periodically:
   ```bash
   heroku addons:create scheduler:standard -a your-app-name
   ```
   Configure the scheduler to run:
   ```bash
   curl -X POST https://your-app-name.herokuapp.com/refresh
   ```

## Monitoring

Visit your Heroku app's root URL (e.g., `https://your-app-name.herokuapp.com/`) to access the real-time monitoring dashboard.

## License

MIT
