# prometheus_prediction_service.py
from prometheus_client import start_http_server, Gauge
from flask import Flask
import joblib
import os
import time

app = Flask(__name__)
MODEL_DIR = os.getenv("MODEL_DIR", "/models")

# Define Gauges for predicted metrics
predicted_cpu_usage = Gauge('predicted_cpu_usage', 'Predicted CPU usage')
predicted_memory_usage = Gauge('predicted_memory_usage', 'Predicted memory usage')
predicted_request_rate = Gauge('predicted_request_rate', 'Predicted request rate')

def load_models_and_predict():
    """Load Prophet models and update Prometheus Gauges with predictions."""
    models = ["cpu_usage", "memory_usage", "request_rate"]
    predictions = {}

    for metric in models:
        model_path = os.path.join(MODEL_DIR, f"{metric}_model.pkl")
        if not os.path.exists(model_path):
            print(f"Model for {metric} not found")
            continue

        model = joblib.load(model_path)
        future = model.make_future_dataframe(periods=1, freq='H')
        forecast = model.predict(future)
        predictions[metric] = forecast.iloc[-1]['yhat']

    # Update Prometheus Gauges
    predicted_cpu_usage.set(predictions.get("cpu_usage", 0))
    predicted_memory_usage.set(predictions.get("memory_usage", 0))
    predicted_request_rate.set(predictions.get("request_rate", 0))

@app.route('/refresh', methods=['POST'])
def refresh_predictions():
    """Refresh predictions on demand."""
    load_models_and_predict()
    return "Predictions refreshed", 200

if __name__ == '__main__':
    # Start Prometheus metrics endpoint
    start_http_server(8001)
    # Periodically refresh predictions
    while True:
        load_models_and_predict()
        time.sleep(3600)  # Refresh predictions hourly
