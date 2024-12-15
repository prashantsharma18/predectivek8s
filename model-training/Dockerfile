# Use a lightweight Python image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && apt-get clean

# Create working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the script into the container
COPY prometheus_model_training.py /app/prometheus_model_training.py

# Set the default command
CMD ["python", "prometheus_model_training.py"]
