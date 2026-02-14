# Quick Start - Docker Deployment

This is a quick start guide for deploying the Verilog Code Generator API using Docker.

## 🚀 Quick Start (5 minutes)

### 1. Prerequisites
- Docker Desktop with GPU support
- NVIDIA GPU with CUDA support
- GEMINI_API_KEY from [Google AI Studio](https://makersuite.google.com/app/apikey)

### 2. Setup Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your GEMINI_API_KEY
# nano .env  (Linux/Mac)
# notepad .env  (Windows)
```

### 3. Build and Run

```bash
# Build the Docker image (10-15 minutes)
docker build -t verilog-generator:latest .

# Run the container
docker run -d \
  --gpus all \
  --name verilog-api \
  -p 8000:8000 \
  --env-file .env \
  verilog-generator:latest

# Or use docker-compose (easier)
docker-compose up -d
```

### 4. Test the API

Wait for the model to download (first run only, ~10-15 minutes), then test:

```bash
# Check health
curl http://localhost:8000/

# Generate Verilog code
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Generate Verilog code for a 2-input AND gate."}'
```

### 5. View Logs

```bash
# Using docker-compose
docker-compose logs -f

# Using docker
docker logs -f verilog-api
```

## 📖 Full Documentation

See [deployment.md](deployment.md) for complete AWS deployment instructions including:
- AWS ECS deployment
- AWS EC2 deployment
- Production configuration
- Troubleshooting guide

## 🛠️ Common Commands

```bash
# Stop the container
docker-compose down

# Rebuild after code changes
docker-compose up -d --build

# View resource usage
docker stats verilog-api

# Execute commands inside container
docker exec -it verilog-api bash

# Check GPU access
docker exec -it verilog-api python3.10 -c "import torch; print(torch.cuda.is_available())"
```

## 📊 Resource Requirements

- **CPU**: 4+ cores
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **Storage**: 100GB+
- **Network**: Fast internet for initial model download

## 🔍 Troubleshooting

**Container won't start?**
```bash
docker logs verilog-api
```

**GPU not detected?**
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi
```

**Out of memory?**
- Use a larger instance or reduce model size in `app.py`

## 📝 API Endpoints

- `GET /` - Health check
- `POST /generate` - Generate Verilog
- `POST /testbench` - Generate testbench
- `POST /simulate` - Run simulation
- `POST /visualize` - Generate diagram
- `POST /complete-workflow` - Full pipeline

Access API docs at: http://localhost:8000/docs
