# AgroVision API + CLI — Dockerfile (v0.7c)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1

# System deps for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY veg_product_api_v07c.py veg_product_api_v07c.py
COPY veg_product_cli.py veg_product_cli.py
COPY model_utils.py model_utils.py
COPY report_backend_auto.html report_backend_auto.html

# Install Python deps
RUN pip install --no-cache-dir fastapi uvicorn opencv-python-headless numpy joblib scikit-learn

# Runs directory
RUN mkdir -p /app/runs
VOLUME ["/app/runs"]

EXPOSE 8000

CMD ["uvicorn", "veg_product_api_v07c:app", "--host", "0.0.0.0", "--port", "8000"]
