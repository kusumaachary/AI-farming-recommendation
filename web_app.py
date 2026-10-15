from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# --- Prepare dataset and train model on startup (small sample dataset) ---
data = pd.DataFrame({
    "N": [90, 85, 60, 74, 78, 95, 40, 50, 30, 100,
          35, 65, 80, 55, 70, 45, 90, 25, 60, 75],

    "P": [42, 58, 55, 35, 42, 45, 40, 30, 25, 50,
          20, 45, 50, 35, 40, 30, 55, 20, 42, 48],

    "K": [43, 41, 44, 40, 42, 38, 35, 25, 30, 45,
          20, 40, 48, 32, 38, 28, 45, 18, 35, 42],

    "temperature": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                    30, 31, 32, 24, 25, 27, 23, 26, 28, 30],

    "humidity": [80, 82, 78, 75, 72, 70, 68, 65, 60, 58,
                 55, 52, 50, 73, 70, 65, 76, 62, 59, 54],

    "ph": [6.5, 6.2, 6.8, 6.4, 6.6, 6.7, 6.3, 6.0, 5.8, 6.1,
           7.0, 6.5, 6.2, 6.4, 6.7, 6.0, 6.6, 5.9, 6.3, 6.8],

    "rainfall": [200, 210, 190, 180, 220, 250, 160, 140, 120, 230,
                 100, 170, 200, 190, 210, 150, 240, 110, 160, 180],

    "crop": [
        "Rice", "Rice", "Rice", "Wheat", "Wheat",
        "Maize", "Maize", "Millet", "Millet", "Sugarcane",
        "Cotton", "Cotton", "Sugarcane", "Wheat", "Maize",
        "Millet", "Rice", "Cotton", "Maize", "Sugarcane"
    ]
})

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

X = data[FEATURES]
y = data["crop"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = [float(request.form.get(f)) for f in FEATURES]
    except (TypeError, ValueError):
        return render_template('result.html', error='Please enter valid numerical values for all fields.')

    farmer_data = pd.DataFrame([values], columns=FEATURES)
    prediction = model.predict(farmer_data)[0]
    probabilities = model.predict_proba(farmer_data)[0]
    confidence = round(max(probabilities) * 100, 2)

    N = values[0]
    P = values[1]
    K = values[2]
    temperature = values[3]
    humidity = values[4]
    ph = values[5]
    rainfall = values[6]

    # Soil analysis
    if N < 50:
        N_status = "LOW"
    elif N <= 100:
        N_status = "MEDIUM"
    else:
        N_status = "HIGH"

    if P < 30:
        P_status = "LOW"
    elif P <= 60:
        P_status = "MEDIUM"
    else:
        P_status = "HIGH"

    if K < 30:
        K_status = "LOW"
    elif K <= 60:
        K_status = "MEDIUM"
    else:
        K_status = "HIGH"

    if ph < 5.5:
        ph_status = "Strongly Acidic"
    elif ph < 6.5:
        ph_status = "Slightly Acidic"
    elif ph <= 7.5:
        ph_status = "Neutral"
    else:
        ph_status = "Alkaline"

    # Weather analysis
    if temperature < 15:
        temp_status = "Low"
    elif temperature <= 30:
        temp_status = "Suitable"
    else:
        temp_status = "High"

    if humidity < 40:
        hum_status = "Low"
    elif humidity <= 80:
        hum_status = "Suitable"
    else:
        hum_status = "High"

    if rainfall < 100:
        rain_status = "Low"
    elif rainfall <= 250:
        rain_status = "Suitable"
    else:
        rain_status = "High"

    return render_template('result.html',
                           recommended_crop=prediction,
                           confidence=confidence,
                           N_status=N_status,
                           P_status=P_status,
                           K_status=K_status,
                           ph_status=ph_status,
                           temp_status=temp_status,
                           hum_status=hum_status,
                           rain_status=rain_status)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
