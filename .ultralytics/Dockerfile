# Build Stage
FROM python:3.11-slim-bookworm AS builder 

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libgl1-mesa-dev \
        libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime Stage
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY wms_main.py wms_gen_video.py wms_camera.py wms_model.py wms_logging.py ./
COPY model/model-WMS.pt ./model/

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000

CMD ["uvicorn", "wms_main:app", "--host", "0.0.0.0", "--port", "8000"]