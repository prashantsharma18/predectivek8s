import os
import logging
import joblib
from prophet import Prophet
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import traceback
from prometheus_client import Gauge

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Implement the code to get Prometheus configuration from environment variables
PROMETHEUS_URL = os.getenv('PROMETHEUS_URL', 'http://prometheus-kube-prometheus-prometheus.monitoring:9090')

# location of trained model for CPU, memory, request rate, http requests etc metrics
MODEL_DIR = os.getenv('MODEL_DIR', '/models')

# provide configuration to understand for what all metrics (CPU, memory, request rate, http requests etc) we need to train the model
METRICS_TO_TRAIN = [
    'cpu_usage',
    'memory_usage',
    'request_rate',
    'http_requests'
]

def fetch_data_from_prometheus(metric):
    """Fetch the data from Prometheus for the given metric"""
    logger.debug(f"Fetching data for metric: {metric}")
    query = f'{PROMETHEUS_URL}/api/v1/query?query={metric}'
    try:
        response = requests.get(query)
        response.raise_for_status()
        data = response.json()
        if data['status'] != 'success':
            raise Exception(f'Failed to fetch data from Prometheus: {data["error"]}')
        logger.debug(f"Data fetched for {metric}: {data['data']['result']}")
        return data['data']['result']
    except Exception as e:
        logger.error(f'Failed to fetch data from Prometheus: {e}')
        logger.error(traceback.format_exc())
        return []

def train_model(metric, data):
    """Train a model based on the data fetched from Prometheus for the given metric."""
    logger.debug(f"Training model for metric: {metric} with data: {data}")

    # Convert the data to a DataFrame
    df = pd.DataFrame(data, columns=['ds', 'y'])

    # Initialize a Prophet model
    model = Prophet()

    # Fit the model with the data
    try:
        model.fit(df)
    except Exception as e:
        logger.error(f'Failed to fit model for {metric}: {e}')
        logger.error(traceback.format_exc())
        return None

    # Save the model to the specified directory
    model_path = os.path.join(MODEL_DIR, f"{metric}_model.pkl")
    try:
        joblib.dump(model, model_path)
    except Exception as e:
        logger.error(f'Failed to save model for {metric} to {model_path}: {e}')
        logger.error(traceback.format_exc())
        return None
    logger.info(f"Model for {metric} trained and saved to {model_path}")
    return model

def generate_dummy_data(metric, num_points=100):
    """Generate dummy data for the given metric"""
    logger.debug(f"Generating dummy data for metric: {metric}")

    # Generate a list of datetime values
    date_range = [datetime.now() - timedelta(hours=i) for i in range(num_points)][::-1]

    # Generate random values for the metric
    random_values = np.random.rand(num_points) * 100  # Scale as needed

    # Combine into a list of tuples (ds, y)
    dummy_data = list(zip(date_range, random_values))
    logger.info(f"Generated dummy data for {metric}")

    return dummy_data

def refresh_predictions(metric, model):
    """Refresh the predictions based on the trained model"""
    logger.debug(f"Refreshing predictions for metric: {metric}")

    # Make future predictions
    future = model.make_future_dataframe(periods=1, freq='H')
    forecast = model.predict(future)

    # Save the predictions to the specified directory
    predictions_path = os.path.join(MODEL_DIR, f"{metric}_predictions.pkl")
    try:
        joblib.dump(forecast, predictions_path)
    except Exception as e:
        logger.error(f'Failed to save predictions for {metric} to {predictions_path}: {e}')
        logger.error(traceback.format_exc())
        return None
    logger.info(f"Predictions for {metric} refreshed and saved to {predictions_path}")

def push_predictions_to_prometheus(metric, predictions):
    """Push the predictions to Prometheus"""
    logger.debug(f"Pushing predictions to Prometheus for metric: {metric}")

    # Assuming predictions is a DataFrame with the latest forecast
    latest_prediction = predictions.iloc[-1]['yhat']

    # Define a Gauge for the metric if not already defined
    gauge_name = f'predicted_{metric}'
    if gauge_name not in globals():
        globals()[gauge_name] = Gauge(gauge_name, f'Predicted {metric}')

    # Set the latest prediction to the Prometheus Gauge
    globals()[gauge_name].set(latest_prediction)

    # Push the metric to the Pushgateway
    push_gateway = os.getenv('PUSHGATEWAY_URL', 'http://localhost:9091')
    url = f"{push_gateway}/metrics/job/{metric}"
    headers = {'Content-Type': 'text/plain'}
    try:
        requests.post(url, data=globals()[gauge_name].generate_latest(), headers=headers)
    except Exception as e:
        logger.error(f'Failed to push metric {metric} to Pushgateway: {e}')
        logger.error(traceback.format_exc())
        return None
    
    logger.info(f"Prediction for {metric} pushed to Prometheus: {latest_prediction}")


def main():
    """Main function to train the model and refresh the predictions"""
    logger.info("Starting model training and prediction refresh process")
    # Train models for all the specified metrics
    for metric in METRICS_TO_TRAIN:
        logger.debug(f"Processing metric: {metric}")
        # Fetch dummy data instead of fetching from prometheus as the data is not sufficient in prometheus
        data = generate_dummy_data(metric)
        model = train_model(metric, data)
        if model is not None:
            refresh_predictions(metric, model)
            # Push the predictions to prometheus
            predictions_path = os.path.join(MODEL_DIR, f"{metric}_predictions.pkl")
            try:
                predictions = joblib.load(predictions_path)
                push_predictions_to_prometheus(metric, predictions)
            except FileNotFoundError:
                logger.error(f"Predictions not found for {metric} at {predictions_path}")

if __name__ == '__main__':
    main()

