# prediction_service.py
from flask import Flask, jsonify, Response, request
import joblib
import os
from datetime import datetime
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Gauge
import json

app = Flask(__name__)
MODEL_DIR = os.getenv("MODEL_DIR", "/models")

# Create Prometheus gauges for each metric
service1_cpu_prediction = Gauge('service1_predicted_cpu_usage', 'Predicted CPU usage for service1')
service1_memory_prediction = Gauge('service1_predicted_memory_usage', 'Predicted memory usage for service1')
service2_cpu_prediction = Gauge('service2_predicted_cpu_usage', 'Predicted CPU usage for service2')
service2_memory_prediction = Gauge('service2_predicted_memory_usage', 'Predicted memory usage for service2')

def get_predictions(service, metric):
    """Get predictions for a specific service and metric."""
    try:
        predictions_path = os.path.join(MODEL_DIR, f"{service}_{metric}_predictions.pkl")
        if not os.path.exists(predictions_path):
            return None
            
        predictions = joblib.load(predictions_path)
        
        # Get the prediction for the current timestamp
        current_time = datetime.now()
        future_predictions = predictions[predictions['ds'] > current_time]
        
        if len(future_predictions) > 0:
            next_prediction = future_predictions.iloc[0]
            return {
                'value': float(next_prediction['yhat']),
                'timestamp': next_prediction['ds'].isoformat(),
                'lower_bound': float(next_prediction['yhat_lower']),
                'upper_bound': float(next_prediction['yhat_upper'])
            }
        return None
        
    except Exception as e:
        print(f"Error loading predictions for {service}_{metric}: {str(e)}")
        return None

def update_prometheus_metrics():
    """Update all Prometheus metrics with latest predictions"""
    services = ['service1', 'service2']
    metrics = ['cpu_usage', 'memory_usage']
    
    for service in services:
        for metric in metrics:
            prediction = get_predictions(service, metric)
            if prediction:
                if service == 'service1':
                    if metric == 'cpu_usage':
                        service1_cpu_prediction.set(prediction['value'])
                    else:
                        service1_memory_prediction.set(prediction['value'])
                else:
                    if metric == 'cpu_usage':
                        service2_cpu_prediction.set(prediction['value'])
                    else:
                        service2_memory_prediction.set(prediction['value'])

@app.route('/predict/<service>/<metric>')
def predict(service, metric):
    """API endpoint to get predictions."""
    prediction = get_predictions(service, metric)
    if prediction:
        return jsonify(prediction)
    return jsonify({'error': 'Prediction not available'}), 404

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

@app.route('/models')
def list_models():
    """List all available models and predictions."""
    try:
        files = os.listdir(MODEL_DIR)
        models = {
            'models': [f for f in files if f.endswith('_model.pkl')],
            'predictions': [f for f in files if f.endswith('_predictions.pkl')]
        }
        return jsonify(models)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    update_prometheus_metrics()
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/api/v1/query', methods=['GET'])
def query():
    """Prometheus query API endpoint"""
    query = request.args.get('query', '')
    
    # Update metrics before responding
    update_prometheus_metrics()
    
    # Get the current timestamp
    timestamp = datetime.now().timestamp()
    
    # Match the query to our metrics
    value = None
    if query == 'service1_predicted_cpu_usage':
        value = service1_cpu_prediction._value.get()
    elif query == 'service1_predicted_memory_usage':
        value = service1_memory_prediction._value.get()
    elif query == 'service2_predicted_cpu_usage':
        value = service2_cpu_prediction._value.get()
    elif query == 'service2_predicted_memory_usage':
        value = service2_memory_prediction._value.get()
    
    if value is not None:
        # Format response in Prometheus API format
        response = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [{
                    "metric": {"__name__": query},
                    "value": [timestamp, str(value)]
                }]
            }
        }
        return jsonify(response)
    
    return jsonify({"status": "error", "error": "Unknown metric"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
