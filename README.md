# AI-farming-recommendation
An Ai-farmer agent crop analysis

## Website / Deployment

I added a simple Flask web app to provide an easy interface and make the model deployable.

Files added:
- `web_app.py` — Flask application that trains the sample model on startup and exposes a web form to get crop recommendations.
- `templates/index.html` — Form to input N, P, K, temperature, humidity, pH and rainfall.
- `templates/result.html` — Result page showing recommended crop, confidence and basic soil/weather analysis.
- `requirements.txt` — Python dependencies.
- `Procfile` — for Heroku-style deployments using gunicorn.

Run locally:

1. Create a virtual environment and install dependencies:

   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Start the app:

   python web_app.py

3. Open http://localhost:5000

Deploy (example to Heroku):

1. Create a Heroku app and push this repository, or use any other Python host (Render, Railway, etc.).
2. Ensure `requirements.txt` and `Procfile` are present. Use `gunicorn web_app:app` as the web command.

Notes:
- The app uses a small sample dataset for demonstration. Replace with a production dataset and persist a trained model (pickle) for better performance.
- If you want, I can add Dockerfile, CI, or instructions for Render/GitHub Pages (static frontend) + backend separately.
