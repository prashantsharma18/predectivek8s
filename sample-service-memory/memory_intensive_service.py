from flask import Flask
import time
import prometheus_client
from prometheus_client import Counter, Gauge, start_http_server

app = Flask(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests', ['service'])
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory Usage', ['service'])
ACTIVE_REQUESTS = Gauge('active_requests', 'Number of active requests', ['service'])

# Start Prometheus metrics endpoint on a different port
start_http_server(8001)

# Global list to store data (simulating memory usage)
data_store = []

@app.route('/allocate/<int:size>')
def allocate_memory(size):
    REQUEST_COUNT.labels(service='memory-intensive').inc()
    ACTIVE_REQUESTS.labels(service='memory-intensive').inc()
    
    try:
        # Allocate memory by creating a list of specified size (in MB)
        chunk = [0] * (size * 131072)  # Approximately 1MB per 131072 elements
        data_store.append(chunk)
        
        # Update memory usage metric (approximate)
        memory_mb = len(data_store) * size
        MEMORY_USAGE.labels(service='memory-intensive').set(memory_mb * 1024 * 1024)
        
        return {'status': 'success', 'allocated_mb': size}
    finally:
        ACTIVE_REQUESTS.labels(service='memory-intensive').dec()

@app.route('/clear')
def clear_memory():
    REQUEST_COUNT.labels(service='memory-intensive').inc()
    global data_store
    data_store = []
    MEMORY_USAGE.labels(service='memory-intensive').set(0)
    return {'status': 'cleared'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)