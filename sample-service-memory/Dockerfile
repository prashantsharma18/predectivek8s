FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY memory_intensive_service.py .
EXPOSE 5001 8001
CMD ["python", "memory_intensive_service.py"]