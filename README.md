# SpaceX Launch Summary API

A FastAPI application that fetches recent SpaceX launches and generates witty, "Cities Skylines" style narratives using the xAI Grok API.

## Features

- **Witty Narratives**: Generates short, dry, and technical descriptions of SpaceX launches.
- **FastAPI**: Modern, high-performance web framework.
- **Beautiful Dashboard**: Built-in UI to monitor narratives and API metrics (Requests, Cache Hits, Grok API usage).
- **Persistent Caching**: Uses Heroku Key Value Store (Redis) to cache narratives and metrics across Heroku builds and dyno restarts.
- **Heroku Ready**: Optimized for Heroku deployment with automatic database discovery.

## Live API

The API is live and can be accessed at:
`https://launch-narrative-api-dafccc521fb8.herokuapp.com/`

## API Endpoints

- **Dashboard UI**: `GET /`
  - URL: `https://launch-narrative-api-dafccc521fb8.herokuapp.com/`
- **Narratives List**: `GET /recent_launches_narratives`
  - URL: `https://launch-narrative-api-dafccc521fb8.herokuapp.com/recent_launches_narratives`
  - Returns a JSON object with a list of witty launch descriptions.
- **Metrics**: `GET /metrics`
  - URL: `https://launch-narrative-api-dafccc521fb8.herokuapp.com/metrics`
  - Returns real-time application performance metrics.
- **Cache Refresh**: `POST /refresh`
  - URL: `https://launch-narrative-api-dafccc521fb8.herokuapp.com/refresh`
  - Triggers a manual refresh of the launch data and Grok narratives.

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
   REDIS_URL=redis://localhost:6379  # Optional: for persistent local caching
   ```
5. **Run the application:**
   ```powershell
   python app.py
   ```
   Access the dashboard at [http://localhost:5000](http://localhost:5000).

   *Note: If you see "Could not find platform independent libraries <prefix>" on Windows, ensure your `PYTHONHOME` is set or run via the virtual environment's python directly: `.\.venv\Scripts\python.exe app.py`.*

## Deployment to Heroku

### Option A: Heroku CLI (Recommended)

1. **Create a new Heroku app:**
   ```bash
   heroku create your-app-name
   ```
2. **Add Heroku Key Value Store (Redis):**
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
   git commit -m "Add Redis for persistent caching"
   git push heroku main
   ```

### Option B: Heroku Dashboard (GUI)

1. **Log in** to the [Heroku Dashboard](https://dashboard.heroku.com/).
2. **Select your app** from the list.
3. **Add Heroku Key Value Store (formerly Heroku Redis):**
   - Click the **Resources** tab.
   - In the **Add-ons** search box, type `Heroku Key Value Store`.
   - Select it, choose a plan (e.g., `Mini`), and click **Submit Order Form**.
4. **Configure API Keys:**
   - Click the **Settings** tab.
   - Click **Reveal Config Vars**.
   - Add a new variable:
     - **KEY**: `XAI_API_KEY`
     - **VALUE**: `your_actual_xai_api_key`
   - Click **Add**.
5. **Deploy:**
   - Go to the **Deploy** tab.
   - Connect your GitHub repository or follow the **Heroku Git** instructions to push your code.

## Scheduled Updates (Optional)

The application features built-in "lazy" refreshing: whenever a user hits the API and the cache is older than 1 hour, it automatically pulls new data and regenerates narratives. 

To keep your narratives proactively fresh (ensuring no visitor ever waits for a refresh), set up the Heroku Scheduler:

1. **Add the Scheduler:**
   - **CLI:** `heroku addons:create scheduler:standard -a your-app-name`
   - **GUI:** Go to the **Resources** tab, search for `Heroku Scheduler`, and add it.
2. **Configure the Job:**
   - Open the Scheduler (click on it in the **Resources** tab).
   - Click **Add Job**.
   - Set the command to: `curl -X POST https://your-app-name.herokuapp.com/refresh`
   - Set the frequency (e.g., **Every Hour**).
   - Click **Save Job**.

## Monitoring

Visit your Heroku app's root URL (e.g., `https://your-app-name.herokuapp.com/`) to access the real-time monitoring dashboard. Metrics and narratives are persisted in the Heroku Key Value Store (Redis), so they will survive new builds and restarts.

## License

MIT
