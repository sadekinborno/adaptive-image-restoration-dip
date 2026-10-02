# ==============================================================================
# Adaptive Image Quality Assessment and Restoration System (DIP Project)
# Lightweight Classical Computer Vision Container
# ==============================================================================
FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr and bytecode generation
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENCV_LOG_LEVEL=SILENT \
    PYTHONPATH=/app/src:$PYTHONPATH

WORKDIR /app

# Install minimal OS dependencies for headless OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code into container
COPY src/ /app/src/

# Create default mount points for volume binding
RUN mkdir -p /app/input /app/output

# Volume mounts for host input/output directories
VOLUME ["/app/input", "/app/output"]

# Default entrypoint runs the unified restoration pipeline
ENTRYPOINT ["python", "src/pipeline.py"]
CMD ["--input", "/app/input", "--output", "/app/output"]
