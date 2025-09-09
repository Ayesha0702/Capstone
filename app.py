from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from datetime import timedelta

app = Flask(__name__)

# Load trained model
try:
    model = joblib.load("xgb_simple_model.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Failed to load model:", e)
    model = None

@app.route("/")
def home():
    return render_template("solar.html")

@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON received"}), 400

        horizon = int(data.get("horizon", 24))
        gen_rows = data.get("generation", [])
        weather_rows = data.get("weather", [])

        if not gen_rows or not weather_rows:
            return jsonify({"error": "Both generation and weather data are required"}), 400

        # Convert to DataFrames
        gen = pd.DataFrame(gen_rows)
        weather = pd.DataFrame(weather_rows)

        # Ensure required columns exist
        if "timestamp" not in gen.columns or "value" not in gen.columns:
            return jsonify({"error": "Generation data missing 'timestamp' or 'value'"}), 400
        if "timestamp" not in weather.columns or "value" not in weather.columns:
            return jsonify({"error": "Weather data missing 'timestamp' or 'value'"}), 400

        gen["timestamp"] = pd.to_datetime(gen["timestamp"])
        weather["timestamp"] = pd.to_datetime(weather["timestamp"])

        # Merge on timestamp
        df = pd.merge(gen, weather, on="timestamp", how="inner", suffixes=("_gen","_weather"))
        if df.empty:
            return jsonify({"error": "No matching timestamps in generation and weather data"}), 400

        # Rename to match model features
        df = df.rename(columns={"value_gen": "DC_POWER", "value_weather": "IRRADIATION"})

        # Time features
        df["hour"] = df["timestamp"].dt.hour
        df["dayofweek"] = df["timestamp"].dt.dayofweek

        last_time = df["timestamp"].max()
        last_row = df.iloc[-1]

        predictions = []
        dc_power = last_row["DC_POWER"]

        for i in range(horizon):
            next_time = last_time + timedelta(hours=i+1)
            # Find irradiation for next_time from weather DataFrame
            irradiation_row = weather[weather["timestamp"] == next_time]
            if not irradiation_row.empty:
                irradiation = irradiation_row.iloc[0]["value"]
            else:
                irradiation = last_row["IRRADIATION"]  # fallback: use last value

            features = pd.DataFrame([{
                "hour": next_time.hour,
                "dayofweek": next_time.dayofweek,
                "DC_POWER": dc_power,
                "IRRADIATION": irradiation
            }])

            pred = model.predict(features)[0]
            predictions.append({"timestamp": next_time.isoformat(), "power": float(pred)})
            dc_power = pred

        return jsonify({"forecast": predictions})

    except Exception as e:
        print("❌ Error in /forecast:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
