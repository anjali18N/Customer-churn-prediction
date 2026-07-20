FROM python:3.11-slim

WORKDIR /code

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and model artifact
COPY app ./app
COPY models ./models

EXPOSE 8080

# Cloud Run / most cloud platforms expect the app to listen on $PORT (default 8080)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
