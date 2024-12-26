from flask import Flask
import time
import math
import prometheus_client
from prometheus_client import Counter, Histogram, start_http_server

app = Flask(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests', ['service'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'Request latency', ['service'])
CPU_LOAD = Counter('cpu_load_total', 'CPU Load', ['service'])

# Start Prometheus metrics endpoint
start_http_server(8000)

def compute_intensive_task(n):
    """CPU intensive task - calculating prime numbers"""
    start = time.time()
    primes = []
    for num in range(2, n + 1):
        if all(num % i != 0 for i in range(2, int(math.sqrt(num)) + 1)):
            primes.append(num)
    CPU_LOAD.labels(service='cpu-intensive').inc(time.time() - start)
    return primes

@app.route('/compute/<int:n>')
def compute(n):
    REQUEST_COUNT.labels(service='cpu-intensive').inc()
    with REQUEST_LATENCY.labels(service='cpu-intensive').time():
        primes = compute_intensive_task(n)
    return {'primes': primes}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)