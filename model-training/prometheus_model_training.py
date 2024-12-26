import os
import logging
import joblib
from prophet import Prophet
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import traceback

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

# Configuration for services and their metrics
SERVICES_CONFIG = {
    'service1': {
        'metrics': {
            'cpu_usage': 'container_cpu_usage_seconds_total{container="memory-intensive"}',
            'memory_usage': 'container_memory_usage_bytes{container="memory-intensive"}',
        }
    },
    'service2': {
        'metrics': {
            'cpu_usage': 'container_cpu_usage_seconds_total{container="cpu-intensive"}',
            'memory_usage': 'container_memory_usage_bytes{container="cpu-intensive"}',
        }
    }
}

# Model hyperparameters for different metrics
MODEL_PARAMS = {
    'cpu_usage': {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'seasonality_mode': 'multiplicative'
    },
    'memory_usage': {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'seasonality_mode': 'additive'
    },
}

def fetch_data_from_prometheus(service, metric):
    """
    Fetch metric data from Prometheus for a specific service
    """
    try:
        # Get the Prometheus query for this service and metric
        query = SERVICES_CONFIG[service]['metrics'][metric]
        logger.info(f"Fetching data for {service}/{metric} with query: {query}")
        
        # Calculate the time range (last 2 days)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=2)
        
        # Convert to Unix timestamp
        start_timestamp = int(start_time.timestamp())
        end_timestamp = int(end_time.timestamp())
        
        # Construct the query URL with step of 5 minutes
        url = f"{PROMETHEUS_URL}/api/v1/query_range"
        params = {
            'query': query,
            'start': start_timestamp,
            'end': end_timestamp,
            'step': '5m'
        }
        
        logger.info(f"Making request to {url} with params: {params}")
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        result = response.json()
        
        if result['status'] == 'success' and len(result['data']['result']) > 0:
            # Extract the values
            values = result['data']['result'][0]['values']
            logger.info(f"Found {len(values)} data points for {service}/{metric}")
            
            # Convert to DataFrame
            df = pd.DataFrame(values, columns=['ds', 'y'])
            df['ds'] = pd.to_datetime(df['ds'], unit='s')
            df['y'] = df['y'].astype(float)
            
            # For CPU metrics, calculate the rate of change
            if 'cpu' in metric:
                df['y'] = df['y'].diff().fillna(0)
                df = df[df['y'] >= 0]  # Remove negative values from counter resets
            
            # For memory metrics, convert to MB
            if 'memory' in metric:
                df['y'] = df['y'] / (1024 * 1024)  # Convert to MB
            
            return df
            
        logger.warning(f"No data found for {service}/{metric}")
        return None
            
    except Exception as e:
        logger.error(f"Error fetching data for {service}/{metric}: {str(e)}")
        return None

def train_model(service, metric, data):
    """
    Train a Prophet model for a specific service and metric
    """
    try:
        if data is None or len(data) < 5:
            logger.warning(f"Insufficient data for {service}/{metric}, skipping model creation")
            return None
            
        logger.info(f"Training model for {service}/{metric} with {len(data)} data points")
        
        # Adjust model parameters based on data availability
        if len(data) < 50:  # Limited data
            model = Prophet(
                changepoint_prior_scale=0.01,
                seasonality_prior_scale=0.1,
                seasonality_mode='additive',
                daily_seasonality=True,
                weekly_seasonality=False,
                interval_width=0.95
            )
        else:  # Sufficient data
            model_params = MODEL_PARAMS[metric]
            model = Prophet(
                changepoint_prior_scale=model_params['changepoint_prior_scale'],
                seasonality_prior_scale=model_params['seasonality_prior_scale'],
                seasonality_mode=model_params['seasonality_mode'],
                daily_seasonality=True,
                weekly_seasonality=True
            )
        
        model.fit(data)
        
        # Save the model
        model_path = os.path.join(MODEL_DIR, f"{service}_{metric}_model.pkl")
        joblib.dump(model, model_path)
        
        logger.info(f"Successfully trained and saved model for {service}/{metric}")
        return model
        
    except Exception as e:
        logger.error(f"Error training model for {service}/{metric}: {str(e)}")
        return None

def refresh_predictions(service, metric, model):
    """
    Generate and save predictions for the model
    """
    try:
        if model is None:
            logger.warning(f"No model available for {service}/{metric}, skipping predictions")
            return None
            
        # Generate future dataframe for next 24 hours
        future = model.make_future_dataframe(periods=288, freq='5T')
        forecast = model.predict(future)
        
        # Save predictions
        predictions_path = os.path.join(MODEL_DIR, f"{service}_{metric}_predictions.pkl")
        joblib.dump(forecast, predictions_path)
        
        logger.info(f"Successfully generated and saved predictions for {service}/{metric}")
        return forecast
        
    except Exception as e:
        logger.error(f"Error refreshing predictions for {service}_{metric}: {str(e)}")
        return None

def main():
    """
    Main function to train models for all services and metrics
    """
    for service, config in SERVICES_CONFIG.items():
        for metric in config['metrics']:
            logger.info(f"Processing service: {service}, metric: {metric}")
            
            # Fetch data
            data = fetch_data_from_prometheus(service, metric)
            
            # Train model if we have data
            model = train_model(service, metric, data)
            
            # Generate predictions if we have a model
            if model is not None:
                refresh_predictions(service, metric, model)
            else:
                logger.warning(f"Skipping predictions for {service}/{metric} due to insufficient data")

if __name__ == '__main__':
    main()
