# ============================================================
# AI BASED FARMER CROP RECOMMENDATION SYSTEM
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------------------------
# STEP 1: Load Dataset
# ------------------------------------------------------------

# If you have your own CSV file, use:
# data = pd.read_csv("Crop_recommendation.csv")

# Sample dataset for demonstration
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

print("\n==============================")
print("       DATASET")
print("==============================")

print(data.head())

# ------------------------------------------------------------
# STEP 2: Separate Input and Output
# ------------------------------------------------------------

X = data[
    ["N", "P", "K", "temperature",
     "humidity", "ph", "rainfall"]
]

y = data["crop"]

# ------------------------------------------------------------
# STEP 3: Split Dataset
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ------------------------------------------------------------
# STEP 4: Create AI Model
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nAI Model Training Completed!")

# ------------------------------------------------------------
# STEP 5: Test Model Accuracy
# ------------------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:",
      round(accuracy * 100, 2), "%")

# ------------------------------------------------------------
# STEP 6: Get Farmer Input
# ------------------------------------------------------------

print("\n===================================")
print("      FARMER INPUT")
print("===================================")

try:
    N = float(input("Enter Nitrogen (N): "))
    P = float(input("Enter Phosphorus (P): "))
    K = float(input("Enter Potassium (K): "))

    temperature = float(
        input("Enter Temperature (°C): ")
    )

    humidity = float(
        input("Enter Humidity (%): ")
    )

    ph = float(
        input("Enter Soil pH: ")
    )

    rainfall = float(
        input("Enter Rainfall (mm): ")
    )

    # --------------------------------------------------------
    # STEP 7: Create Farmer Data
    # --------------------------------------------------------

    farmer_data = pd.DataFrame([[
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    ]], columns=X.columns)

    # --------------------------------------------------------
    # STEP 8: Predict Crop
    # --------------------------------------------------------

    prediction = model.predict(farmer_data)

    recommended_crop = prediction[0]

    # --------------------------------------------------------
    # STEP 9: Prediction Probability
    # --------------------------------------------------------

    probabilities = model.predict_proba(farmer_data)

    confidence = max(probabilities[0]) * 100

    # --------------------------------------------------------
    # STEP 10: Display Recommendation
    # --------------------------------------------------------

    print("\n===================================")
    print("       AI CROP RECOMMENDATION")
    print("===================================")

    print("Recommended Crop :", recommended_crop)

    print(
        "AI Confidence    :",
        round(confidence, 2),
        "%"
    )

    # --------------------------------------------------------
    # STEP 11: Basic Soil Analysis
    # --------------------------------------------------------

    print("\n===================================")
    print("          SOIL ANALYSIS")
    print("===================================")

    if N < 50:
        print("Nitrogen : LOW")
    elif N <= 100:
        print("Nitrogen : MEDIUM")
    else:
        print("Nitrogen : HIGH")

    if P < 30:
        print("Phosphorus : LOW")
    elif P <= 60:
        print("Phosphorus : MEDIUM")
    else:
        print("Phosphorus : HIGH")

    if K < 30:
        print("Potassium : LOW")
    elif K <= 60:
        print("Potassium : MEDIUM")
    else:
        print("Potassium : HIGH")

    if ph < 5.5:
        print("Soil pH : Strongly Acidic")
    elif ph < 6.5:
        print("Soil pH : Slightly Acidic")
    elif ph <= 7.5:
        print("Soil pH : Neutral")
    else:
        print("Soil pH : Alkaline")

    # --------------------------------------------------------
    # STEP 12: Weather Analysis
    # --------------------------------------------------------

    print("\n===================================")
    print("        WEATHER ANALYSIS")
    print("===================================")

    if temperature < 15:
        print("Temperature : Low")
    elif temperature <= 30:
        print("Temperature : Suitable")
    else:
        print("Temperature : High")

    if humidity < 40:
        print("Humidity : Low")
    elif humidity <= 80:
        print("Humidity : Suitable")
    else:
        print("Humidity : High")

    if rainfall < 100:
        print("Rainfall : Low")
    elif rainfall <= 250:
        print("Rainfall : Suitable")
    else:
        print("Rainfall : High")

except ValueError:
    print("\nPlease enter valid numerical values.")

# ============================================================
# END
# ============================================================
