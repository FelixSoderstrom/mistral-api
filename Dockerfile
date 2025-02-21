# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    cmake \
    build-essential \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip3 install --no-cache-dir llama-cpp-python==0.2.23
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install Gunicorn
RUN pip3 install gunicorn

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV CUDA_LAUNCH_BLOCKING=1

# Expose port
EXPOSE 8000

# Start the application with Gunicorn
CMD ["gunicorn", "main:app", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "300"] 