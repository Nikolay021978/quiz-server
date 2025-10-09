# Base image and working directory
FROM python:3.10-slim
WORKDIR /app

# Install system build deps (if native wheels required)
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker layer cache
COPY requirements.txt .

# Create virtual environment and install Python dependencies
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt

# Copy application source code
COPY . .

# Ensure virtualenv binaries are in PATH
ENV PATH="/opt/venv/bin:$PATH"

# Expose application port
EXPOSE 8000

# Run with Gunicorn using aiohttp worker; replace workers count if needed
CMD ["gunicorn", "run_server:app", "-k", "aiohttp.GunicornWebWorker", "--bind", "0.0.0.0:8000", "--workers", "2"]

