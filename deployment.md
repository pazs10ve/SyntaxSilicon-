# Verilog Code Generator API - AWS Deployment Guide

This guide covers deploying the Verilog Code Generator API to AWS using Docker.

## Prerequisites

- Docker installed locally
- AWS CLI configured
- NVIDIA Docker runtime (for GPU support)
- AWS account with appropriate permissions
- GEMINI_API_KEY from Google AI Studio

## Local Development & Testing

### 1. Environment Setup

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
HF_TOKEN=your_huggingface_token_here  # Optional, for private models
```

### 2. Build the Docker Image

```bash
# Build the image
docker build -t verilog-generator:latest .

# This may take 10-15 minutes depending on your internet connection
```

### 3. Run Locally with Docker Compose

```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

**Note**: First startup will take 10-15 minutes as the models download from HuggingFace (~13GB).

### 4. Test the API

```bash
# Check health
curl http://localhost:8000/

# Test Verilog generation
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Generate Verilog code for a 2-input AND gate."}'
```

## AWS Deployment Options

### Option 1: AWS ECS with ECR (Recommended)

#### Step 1: Create ECR Repository

```bash
# Create repository
aws ecr create-repository --repository-name verilog-generator

# Get login command
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
```

#### Step 2: Tag and Push Image

```bash
# Tag the image
docker tag verilog-generator:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/verilog-generator:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/verilog-generator:latest
```

#### Step 3: Create ECS Task Definition

Create a task definition JSON file `ecs-task-definition.json`:

```json
{
  "family": "verilog-generator-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["EC2"],
  "cpu": "4096",
  "memory": "16384",
  "containerDefinitions": [
    {
      "name": "verilog-generator",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/verilog-generator:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "GEMINI_API_KEY",
          "value": "your_api_key_here"
        },
        {
          "name": "TRANSFORMERS_CACHE",
          "value": "/app/model_cache"
        }
      ],
      "resourceRequirements": [
        {
          "type": "GPU",
          "value": "1"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/verilog-generator",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/ || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 120
      }
    }
  ]
}
```

#### Step 4: Create ECS Cluster

```bash
# Create cluster
aws ecs create-cluster --cluster-name verilog-generator-cluster

# Register task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json
```

#### Step 5: Create ECS Service

```bash
aws ecs create-service \
  --cluster verilog-generator-cluster \
  --service-name verilog-generator-service \
  --task-definition verilog-generator-task \
  --desired-count 1 \
  --launch-type EC2 \
  --load-balancers targetGroupArn=<your-target-group-arn>,containerName=verilog-generator,containerPort=8000
```

### Option 2: AWS EC2 with Docker

#### Step 1: Launch GPU-Enabled EC2 Instance

1. Go to AWS EC2 Console
2. Launch instance with:
   - **AMI**: Deep Learning AMI GPU PyTorch (Ubuntu 22.04)
   - **Instance Type**: g4dn.xlarge or p3.2xlarge
   - **Storage**: At least 100GB
   - **Security Group**: Allow inbound traffic on port 8000

#### Step 2: Connect and Install Docker

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<ec2-public-ip>

# Update and install Docker (if not already installed)
sudo apt-get update
sudo apt-get install -y docker.io docker-compose

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Step 3: Deploy Application

```bash
# Clone repository or copy files
git clone <your-repo> && cd SyntaxSilicon-

# Create .env file
echo "GEMINI_API_KEY=your_key_here" > .env

# Pull or build image
docker-compose up -d

# Check logs
docker-compose logs -f
```

#### Step 4: Configure Reverse Proxy (Optional)

For production, use Nginx as a reverse proxy:

```bash
sudo apt-get install -y nginx

# Create Nginx config
sudo nano /etc/nginx/sites-available/verilog-generator
```

Nginx configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Increase timeouts for model inference
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/verilog-generator /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key for testbench generation |
| `HF_TOKEN` | No | HuggingFace token for private models |
| `TRANSFORMERS_CACHE` | No | Path for model cache (default: `/app/model_cache`) |
| `TORCH_HOME` | No | PyTorch cache directory |

## Resource Requirements

### Minimum Requirements
- **CPU**: 4 vCPUs
- **RAM**: 16GB
- **GPU**: 1x NVIDIA GPU with 8GB+ VRAM (e.g., T4, V100, A10G)
- **Storage**: 100GB
- **Network**: 10GB download for initial model download

### Recommended Instance Types
- **AWS EC2**: g4dn.xlarge, g4dn.2xlarge, p3.2xlarge
- **AWS ECS**: EC2 launch type with GPU-enabled instances

## API Endpoints

Once deployed, the following endpoints are available:

- `GET /` - API information and health check
- `POST /generate` - Generate Verilog code from prompt
- `POST /testbench` - Generate testbench for Verilog file
- `POST /simulate` - Simulate Verilog with testbench
- `POST /visualize` - Generate SVG visualization
- `POST /complete-workflow` - Run complete workflow
- `GET /download/{file_path}` - Download generated files

## Monitoring & Troubleshooting

### View Logs

```bash
# Docker Compose
docker-compose logs -f

# Docker
docker logs verilog-api -f

# ECS
aws logs tail /ecs/verilog-generator --follow
```

### Common Issues

#### 1. Model Download Timeout
**Problem**: Container health check fails during initial startup  
**Solution**: Increase `start_period` in health check to 300s for first run

#### 2. Out of Memory
**Problem**: Container crashes with OOM error  
**Solution**: Increase instance memory or reduce model size in `app.py`

#### 3. GPU Not Detected
**Problem**: Model loads on CPU instead of GPU  
**Solution**: 
- Verify NVIDIA Docker runtime: `docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi`
- Check ECS task definition includes GPU resource requirement

#### 4. Slow Inference
**Problem**: API responses take too long  
**Solution**: 
- Verify GPU is being used
- Check instance type has sufficient VRAM
- Consider using larger instance or optimizing model parameters

### Testing GPU Access

```bash
# Inside container
docker exec -it verilog-api python3.10 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## Scaling Considerations

For production deployments:

1. **Load Balancing**: Use Application Load Balancer with multiple ECS tasks
2. **Auto Scaling**: Configure ECS auto-scaling based on CPU/Memory metrics
3. **Model Caching**: Use EFS for persistent model cache across instances
4. **API Gateway**: Add AWS API Gateway for rate limiting and authentication
5. **CloudWatch**: Set up alarms for health check failures and high memory usage

## Cost Optimization

- Use **Spot Instances** for ECS/EC2 to reduce costs by up to 70%
- Implement **auto-shutdown** during off-hours
- Use **Savings Plans** for predictable workloads
- Monitor and optimize **data transfer costs**

## Security Best Practices

1. Store API keys in **AWS Secrets Manager**
2. Use **IAM roles** instead of hardcoded credentials
3. Enable **VPC** endpoints for private communication
4. Implement **API authentication** (JWT, API keys)
5. Use **HTTPS** with SSL/TLS certificates
6. Regular security updates and vulnerability scanning

## Support

For issues or questions:
- Check logs using the commands above
- Review the API documentation at `http://<your-endpoint>/docs`
- Test endpoints using the provided curl commands
