FROM python:3.11-slim

# Install system dependencies for OpenCV / Ultralytics
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Render provides the PORT env variable
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:${PORT}", "--workers", "2"]
