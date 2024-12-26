FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY cpu_intensive_service.py .
EXPOSE 5000 8000
CMD ["python", "cpu_intensive_service.py"]