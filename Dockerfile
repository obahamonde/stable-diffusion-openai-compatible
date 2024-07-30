# Use the NVIDIA CUDA runtime as the base image
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH /usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Update and install essential packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    build-essential \
    curl \
    wget \
    software-properties-common \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port for the FastAPI app
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
