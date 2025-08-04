# AWS EC2 Deployment Guide for Redis-Enhanced RAG System

This guide will help you deploy the Redis-enhanced RAG system on an AWS EC2 instance.

## Prerequisites

- AWS Account with EC2 access
- SSH key pair for EC2 access
- Domain/DNS setup (optional, for custom domain)

## AWS EC2 Setup

### 1. Launch EC2 Instance

**Recommended Instance Type:** `t3.large` or `t3.xlarge`
- **vCPUs:** 2-4 cores
- **Memory:** 8-16 GB RAM
- **Storage:** 50-100 GB SSD

**Operating System:** Ubuntu 22.04 LTS

### 2. Security Group Configuration

Configure your security group to allow:

| Type | Protocol | Port Range | Source | Description |
|------|----------|------------|--------|-------------|
| SSH | TCP | 22 | Your IP | SSH access |
| HTTP | TCP | 8000 | 0.0.0.0/0 | RAG API access |
| Custom TCP | TCP | 6379 | 127.0.0.1/32 | Redis (localhost only) |

### 3. Connect to Instance

```bash
ssh -i your-key.pem ubuntu@your-instance-ip
```

## Deployment Steps

### 1. Upload Files to Instance

**Option A: Using SCP**
```bash
# From your local machine
scp -i your-key.pem -r paany_instance/* ubuntu@your-instance-ip:~/
```

**Option B: Using Git (if repository is public)**
```bash
# On EC2 instance
git clone https://github.com/yourusername/your-repo.git
cd your-repo/paany_instance
```

**Option C: Manual Upload via Terminal**
```bash
# Create directory on instance
mkdir -p ~/rag-system
cd ~/rag-system

# Copy files manually (upload each file via your preferred method)
```

### 2. Run Setup Script

```bash
# Make script executable
chmod +x setup.sh

# Run setup (this will install Docker, dependencies, and start services)
./setup.sh
```

### 3. Configure Environment

```bash
# Edit environment file
nano .env

# Add your Google API key:
GOOGLE_API_KEY=your_actual_google_api_key_here
```

### 4. Start Services

```bash
# Start the system
./start.sh
```

## Verification

### Test the Deployment

```bash
# Get your public IP
curl http://checkip.amazonaws.com

# Test health endpoint
curl http://your-public-ip:8000/health

# Test comprehensive health
curl http://your-public-ip:8000/api/health

# Test main API
curl -X POST http://your-public-ip:8000/api/v1/hackrx/run \
  -H "Content-Type: application/json" \
  -d '{"query": "What is artificial intelligence?", "documents": ["AI is a field of computer science"]}'
```

## Management Commands

### Start/Stop Services
```bash
./start.sh    # Start all services
./stop.sh     # Stop all services
```

### View Logs
```bash
sudo docker-compose logs -f              # All services
sudo docker-compose logs -f rag-system   # RAG system only
sudo docker-compose logs -f redis        # Redis only
```

### Service Status
```bash
sudo docker-compose ps    # Container status
sudo docker stats          # Resource usage
```

### Restart Services
```bash
sudo docker-compose restart              # Restart all
sudo docker-compose restart rag-system   # Restart RAG only
```

## Monitoring

### Health Endpoints
- **Basic Health:** `http://your-ip:8000/health`
- **Comprehensive Health:** `http://your-ip:8000/api/health`
- **Redis Status:** `http://your-ip:8000/redis-status`
- **Performance Stats:** `http://your-ip:8000/performance/stats`

### System Resources
```bash
# Check system resources
htop

# Check disk usage
df -h

# Check memory usage
free -h

# Check Docker stats
sudo docker stats
```

## Troubleshooting

### Common Issues

**1. Services Won't Start**
```bash
# Check logs
sudo docker-compose logs

# Check Docker daemon
sudo systemctl status docker

# Restart Docker
sudo systemctl restart docker
```

**2. API Returns Errors**
```bash
# Check environment variables
cat .env

# Check container logs
sudo docker-compose logs rag-system

# Test Redis connection
sudo docker exec rag-redis redis-cli ping
```

**3. Memory Issues**
```bash
# Check memory usage
free -h

# Clear Docker cache
sudo docker system prune -f

# Restart with memory optimization
sudo docker-compose restart
```

**4. Port Access Issues**
```bash
# Check if port is open
sudo netstat -tlnp | grep 8000

# Check security group settings in AWS console
# Ensure port 8000 is open to 0.0.0.0/0
```

### Log Analysis
```bash
# Search for errors
sudo docker-compose logs | grep -i error

# Follow real-time logs
sudo docker-compose logs -f --tail=100

# Check specific timeframe
sudo docker-compose logs --since="2h"
```

## Production Optimization

### 1. SSL/TLS Setup (Recommended)

**Using Nginx Reverse Proxy:**
```bash
# Install Nginx
sudo apt install nginx certbot python3-certbot-nginx

# Configure Nginx (create /etc/nginx/sites-available/rag-system)
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Enable site and get SSL certificate
sudo ln -s /etc/nginx/sites-available/rag-system /etc/nginx/sites-enabled/
sudo certbot --nginx -d your-domain.com
```

### 2. Auto-Start on Boot
```bash
# Create systemd service
sudo nano /etc/systemd/system/rag-system.service

[Unit]
Description=RAG System
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/bin/docker-compose -f /home/ubuntu/rag-system/docker-compose.yml up -d
ExecStop=/usr/local/bin/docker-compose -f /home/ubuntu/rag-system/docker-compose.yml down
WorkingDirectory=/home/ubuntu/rag-system

[Install]
WantedBy=multi-user.target

# Enable service
sudo systemctl enable rag-system.service
```

### 3. Monitoring Setup
```bash
# Set up log rotation
sudo nano /etc/logrotate.d/docker-compose

/home/ubuntu/rag-system/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 ubuntu ubuntu
}
```

## Backup and Maintenance

### Backup Redis Data
```bash
# Create backup
sudo docker exec rag-redis redis-cli BGSAVE

# Copy backup file
sudo docker cp rag-redis:/data/dump.rdb ./backup-$(date +%Y%m%d).rdb
```

### Update System
```bash
# Pull latest images
sudo docker-compose pull

# Rebuild and restart
sudo docker-compose up --build -d
```

## Support

For issues:
1. Check service logs: `sudo docker-compose logs`
2. Verify health endpoints
3. Check AWS security group settings
4. Review system resources
5. Check environment configuration
