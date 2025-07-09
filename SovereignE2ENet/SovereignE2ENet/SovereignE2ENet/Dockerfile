# Use a base image with Python 3.8 (compatible with MPS)
FROM python:3.8-slim

# Set working directory in container
WORKDIR /app

# Install necessary dependencies for MPS and transformers
RUN apt-get update && apt-get install -y     git     curl     && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Expose API port
EXPOSE 8000

# Command to run API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
