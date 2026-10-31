# Lightweight Python base image (same Python version as the training environment)
FROM python:3.13-slim

# libgomp is needed by xgboost / scikit-learn for parallel processing
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Working directory inside the container
WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code, the shared feature code, the market index and the serialized model
COPY . .

# The API listens on port 7860
EXPOSE 7860

# Serve the Flask app with gunicorn (2 workers) -> module "app", Flask object "freight_api"
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "--timeout", "120", "app:freight_api"]
