# Multi-stage Dockerfile for Verilog Code Generator API
# Optimized for AWS deployment with GPU support

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_CACHE=/app/model_cache \
    HF_HOME=/app/model_cache \
    TORCH_HOME=/app/model_cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    software-properties-common \
    nodejs \
    npm \
    # Verilog simulation tools
    iverilog \
    gtkwave \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install Yosys (Verilog synthesis tool)
RUN apt-get update && apt-get install -y \
    yosys \
    && rm -rf /var/lib/apt/lists/*

# Install netlistsvg globally via npm
RUN npm install -g netlistsvg

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN python3.10 -m pip install --upgrade pip setuptools wheel && \
    python3.10 -m pip install \
    transformers>=4.30.0 \
    torch>=2.0.0 \
    peft>=0.4.0 \
    bitsandbytes>=0.39.0 \
    fastapi>=0.95.0 \
    uvicorn>=0.21.0 \
    python-dotenv>=1.0.0 \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    tqdm>=4.65.0 \
    matplotlib>=3.7.0 \
    vcdvcd>=2.0.0 \
    google-genai \
    pydantic>=2.0.0 \
    pillow>=10.0.0 \
    accelerate>=0.20.0 \
    sentencepiece>=0.1.99 \
    protobuf>=3.20.0

# Copy application code
COPY app.py .
COPY final.py .
COPY main.py .
COPY main2.py .
COPY main3.py .
COPY schematic.py .
COPY schematic2.py .
COPY schematic3.py .
COPY schematic4.py .
COPY test.py .
COPY test2.py .
COPY test3.py .

# Copy .env file if it exists (optional, better to use docker-compose env)
COPY .env* ./

# Create necessary directories
RUN mkdir -p /app/model_cache \
    /app/frontend/public/images \
    /app/outputs \
    /app/logs

# Pre-download models (optional - commented out to reduce build time)
# Uncomment if you want models baked into the image
# RUN python3.10 -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
#     AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-hf'); \
#     AutoModelForCausalLM.from_pretrained('codellama/CodeLlama-7b-hf', device_map='cpu')"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Set the entrypoint
CMD ["python3.10", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
