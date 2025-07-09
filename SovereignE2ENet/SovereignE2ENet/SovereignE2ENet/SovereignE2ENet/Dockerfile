# Use a base image with CUDA support (for GPU usage)
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Install dependencies
RUN apt-get update && apt-get install -y     python3     python3-pip     python3-dev     git     curl     && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Set working directory
WORKDIR /app

# Expose API port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
