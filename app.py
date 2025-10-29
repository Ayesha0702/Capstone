from flask import Flask, request, jsonify, render_template, send_file
import joblib
import pandas as pd
from datetime import timedelta
import os
from flask_cors import CORS
import io
from flask import send_from_directory
import traceback

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


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

@app.route("/ping")
def ping():
    return "Server is alive ✅"

@app.route('/data/<filename>')
def serve_data(filename):
    return send_from_directory('data', filename)

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
        df = pd.merge(gen, weather, on="timestamp", how="inner", suffixes=("_gen", "_weather"))
        if df.empty:
            return jsonify({"error": "No matching timestamps in generation and weather data"}), 400

        # Rename columns if they exist, else fallback
        if "value_gen" in df.columns:
            df.rename(columns={"value_gen": "DC_POWER"}, inplace=True)
        elif "value" in gen.columns:
            df.rename(columns={"value": "DC_POWER"}, inplace=True)
        else:
            return jsonify({"error": "Cannot find DC_POWER column"}), 400

        if "value_weather" in df.columns:
            df.rename(columns={"value_weather": "IRRADIATION"}, inplace=True)
        elif "value" in weather.columns:
            df.rename(columns={"value": "IRRADIATION"}, inplace=True)
        else:
            return jsonify({"error": "Cannot find IRRADIATION column"}), 400

        # Add time-based features
        df["hour"] = df["timestamp"].dt.hour
        df["dayofweek"] = df["timestamp"].dt.dayofweek

        last_time = df["timestamp"].max()
        last_row = df.iloc[-1]

        predictions = []
        dc_power = last_row["DC_POWER"]

        for i in range(horizon):
            next_time = last_time + timedelta(hours=i + 1)

            # Find irradiation for next_time from weather DataFrame
            irradiation_row = weather[weather["timestamp"] == next_time]
            if not irradiation_row.empty:
                irradiation = irradiation_row.iloc[0]["value"]
            else:
                irradiation = last_row["IRRADIATION"]  # fallback to last value

            features = pd.DataFrame([{
                "hour": next_time.hour,
                "dayofweek": next_time.dayofweek,
                "DC_POWER": dc_power,
                "IRRADIATION": irradiation
            }])

            pred = model.predict(features)[0]
            predictions.append({
                "timestamp": next_time.strftime("%Y-%m-%d %H:%M:%S"),
                "power": float(pred)
            })
            dc_power = pred

        # Calculate total forecasted energy (kWh)
        total_energy = sum(p["power"] for p in predictions) / 1000

        return jsonify({
            "message": f"✅ Forecast completed for {horizon} hours.",
            "total_energy": f"{total_energy:.2f} kWh",
            "forecast": predictions
        })

    except Exception as e:
     traceback.print_exc()
    return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/download_forecast", methods=["POST"])
def download_forecast():
    try:
        data = request.get_json()
        forecast_data = data.get("forecast_data")

        if not forecast_data:
            return jsonify({"error": "No forecast data received"}), 400

        df = pd.DataFrame(forecast_data)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

        # Create in-memory CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        return send_file(
            io.BytesIO(csv_buffer.getvalue().encode()),
            mimetype="text/csv",
            as_attachment=True,
            download_name="solarml_forecast.csv"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        print("❌ Error in /download_forecast:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

