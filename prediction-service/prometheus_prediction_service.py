# prediction_service.py
from flask import Flask, jsonify
import joblib
import os
from datetime import datetime

app = Flask(__name__)
MODEL_DIR = os.getenv("MODEL_DIR", "/models")

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
