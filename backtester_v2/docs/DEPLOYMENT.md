# 🚀 Production Deployment Guide

> Complete guide for deploying Enterprise GPU Backtester v7.1 to production environments

## 📋 Prerequisites

### System Requirements
- **Node.js**: 18.17+ (LTS recommended)
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **Storage**: 50GB+ SSD (100GB+ recommended)
- **CPU**: 4+ cores (8+ cores recommended)
- **GPU**: NVIDIA GPU with CUDA support (for HeavyDB)

### External Services
- **HeavyDB**: GPU-accelerated database instance
- **MySQL**: Historical data storage
- **Redis**: Caching and session storage (optional but recommended)
- **SMTP**: Email service for notifications
- **SSL Certificate**: For HTTPS in production

## 🔧 Environment Setup

### 1. Environment Variables

Create a `.env.production` file:

```env
# Application
NODE_ENV=production
NEXT_PUBLIC_APP_URL=https://your-domain.com
PORT=3000

# Security
NEXTAUTH_SECRET=your-super-secure-secret-key-min-32-chars
NEXTAUTH_URL=https://your-domain.com

# Database - HeavyDB (Primary)
HEAVYDB_HOST=your-heavydb-host
HEAVYDB_PORT=6274
HEAVYDB_USER=admin
HEAVYDB_PASSWORD=your-secure-password
HEAVYDB_DATABASE=heavyai
HEAVYDB_SSL=true
HEAVYDB_POOL_SIZE=10

# Database - MySQL (Historical)
MYSQL_HOST=your-mysql-host
MYSQL_PORT=3306
MYSQL_USER=backtester
MYSQL_PASSWORD=your-secure-password
MYSQL_DATABASE=historicaldb
MYSQL_SSL=true
MYSQL_POOL_SIZE=20

# Redis (Caching)
REDIS_HOST=your-redis-host
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password
REDIS_DB=0
REDIS_TLS=true

# Rate Limiting
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=900000
RATE_LIMIT_SKIP_FAILED_REQUESTS=true

# Security Headers
CSP_REPORTING_ENDPOINT=https://your-domain.com/api/csp-report
SECURITY_HEADERS_ENABLED=true

# Performance Monitoring
SENTRY_DSN=your-sentry-dsn
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=0.1

# Email Configuration
SMTP_HOST=your-smtp-host
SMTP_PORT=587
SMTP_USER=your-smtp-user
SMTP_PASSWORD=your-smtp-password
SMTP_FROM=noreply@your-domain.com

# WebSocket
WS_RATE_LIMIT=100
WS_CONNECTION_LIMIT=1000

# File Upload
MAX_FILE_SIZE=50MB
ALLOWED_FILE_TYPES=.xlsx,.csv,.json
UPLOAD_DIR=/app/uploads

# Monitoring
HEALTH_CHECK_INTERVAL=30000
PERFORMANCE_MONITORING=true
AUDIT_LOGGING=true
```

### 2. SSL Certificate Setup

For HTTPS in production:

```bash
# Using Let's Encrypt with Certbot
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com

# Or upload your SSL certificates
mkdir -p /etc/ssl/private
cp your-certificate.crt /etc/ssl/certs/
cp your-private-key.key /etc/ssl/private/
```

## 🐳 Docker Deployment

### 1. Dockerfile

```dockerfile
# Use Node.js LTS alpine image
FROM node:18-alpine AS base

# Install dependencies only when needed
FROM base AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./
RUN npm ci --only=production

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Set build environment
ENV NEXT_TELEMETRY_DISABLED 1
ENV NODE_ENV production

# Build application
RUN npm run build

# Production image, copy all the files and run next
FROM base AS runner
WORKDIR /app

ENV NODE_ENV production
ENV NEXT_TELEMETRY_DISABLED 1

# Create non-root user
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy necessary files
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# Create upload directory
RUN mkdir -p /app/uploads && chown nextjs:nodejs /app/uploads

USER nextjs

EXPOSE 3000

ENV PORT 3000
ENV HOSTNAME "0.0.0.0"

CMD ["node", "server.js"]
```

### 2. Docker Compose

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
    env_file:
      - .env.production
    volumes:
      - ./uploads:/app/uploads
      - ./logs:/app/logs
    restart: unless-stopped
    depends_on:
      - redis
    networks:
      - backtester-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    depends_on:
      - app
    restart: unless-stopped
    networks:
      - backtester-network

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis-data:/data
    restart: unless-stopped
    networks:
      - backtester-network

volumes:
  redis-data:

networks:
  backtester-network:
    driver: bridge
```

### 3. Build and Deploy

```bash
# Build Docker image
docker build -t enterprise-gpu-backtester:latest .

# Run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f app
```

## 🌐 Nginx Configuration

### nginx.conf

```nginx
events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    
    access_log /var/log/nginx/access.log main;

    # Performance settings
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    # Compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        application/javascript
        application/json
        text/css
        text/javascript
        text/xml
        text/plain
        application/xml+rss;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login:10m rate=1r/s;

    # Upstream backend
    upstream backend {
        server app:3000;
        keepalive 32;
    }

    # HTTP to HTTPS redirect
    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        # SSL Configuration
        ssl_certificate /etc/ssl/certs/your-certificate.crt;
        ssl_certificate_key /etc/ssl/private/your-private-key.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        ssl_prefer_server_ciphers on;

        # Security headers
        add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
        add_header X-Content-Type-Options nosniff;
        add_header X-Frame-Options DENY;
        add_header X-XSS-Protection "1; mode=block";

        # Client upload limit
        client_max_body_size 50M;

        # Static files
        location /_next/static/ {
            alias /app/.next/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }

        location /static/ {
            alias /app/public/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }

        # API routes with rate limiting
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;
        }

        # Login rate limiting
        location /api/auth/ {
            limit_req zone=login burst=5 nodelay;
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # WebSocket support
        location /ws/ {
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # All other requests
        location / {
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;
        }
    }
}
```

## 📊 Database Setup

### HeavyDB Production Configuration

```sql
-- Create production database
CREATE DATABASE heavyai_prod;

-- Create production user
CREATE USER backtester_prod WITH PASSWORD 'secure_password';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE heavyai_prod TO backtester_prod;

-- Optimize for production
ALTER DATABASE heavyai_prod SET shared_preload_libraries = 'pg_stat_statements';
ALTER DATABASE heavyai_prod SET max_connections = 200;
ALTER DATABASE heavyai_prod SET shared_buffers = '2GB';
ALTER DATABASE heavyai_prod SET effective_cache_size = '6GB';
```

### MySQL Production Configuration

```sql
-- Create production database
CREATE DATABASE historicaldb_prod;

-- Create production user
CREATE USER 'backtester_prod'@'%' IDENTIFIED BY 'secure_password';

-- Grant permissions
GRANT ALL PRIVILEGES ON historicaldb_prod.* TO 'backtester_prod'@'%';
FLUSH PRIVILEGES;

-- Optimize for production
SET GLOBAL innodb_buffer_pool_size = 2147483648; -- 2GB
SET GLOBAL max_connections = 200;
SET GLOBAL query_cache_size = 268435456; -- 256MB
```

## 🔄 Process Management

### PM2 Configuration

```javascript
// ecosystem.config.js
module.exports = {
  apps: [{
    name: 'enterprise-gpu-backtester',
    script: './node_modules/.bin/next',
    args: 'start',
    instances: 'max',
    exec_mode: 'cluster',
    env: {
      NODE_ENV: 'development'
    },
    env_production: {
      NODE_ENV: 'production',
      PORT: 3000
    },
    error_file: './logs/err.log',
    out_file: './logs/out.log',
    log_file: './logs/combined.log',
    time: true,
    max_memory_restart: '2G',
    node_args: '--max-old-space-size=2048'
  }]
}
```

### Start with PM2

```bash
# Install PM2 globally
npm install -g pm2

# Start application
pm2 start ecosystem.config.js --env production

# Save PM2 configuration
pm2 save

# Setup PM2 startup script
pm2 startup

# Monitor processes
pm2 monit

# View logs
pm2 logs enterprise-gpu-backtester
```

## 📈 Monitoring & Logging

### 1. Application Monitoring

```bash
# Health check endpoint
curl https://your-domain.com/api/health

# Performance metrics
curl https://your-domain.com/api/metrics

# System status
curl https://your-domain.com/api/status
```

### 2. Log Configuration

```javascript
// logger.config.js
const winston = require('winston')

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'enterprise-gpu-backtester' },
  transports: [
    new winston.transports.File({ 
      filename: './logs/error.log', 
      level: 'error' 
    }),
    new winston.transports.File({ 
      filename: './logs/combined.log' 
    })
  ]
})

if (process.env.NODE_ENV !== 'production') {
  logger.add(new winston.transports.Console({
    format: winston.format.simple()
  }))
}

module.exports = logger
```

### 3. Error Tracking (Sentry)

```javascript
// sentry.config.js
import * as Sentry from '@sentry/nextjs'

Sentry.init({
  dsn: process.env.SENTRY_DSN,
  environment: process.env.NODE_ENV,
  tracesSampleRate: parseFloat(process.env.SENTRY_TRACES_SAMPLE_RATE || '0.1'),
  beforeSend(event) {
    // Filter out sensitive information
    if (event.exception) {
      const error = event.exception.values[0]
      if (error.value && error.value.includes('password')) {
        return null
      }
    }
    return event
  }
})
```

## 🛡️ Security Hardening

### 1. System Security

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install fail2ban
sudo apt install fail2ban

# Configure UFW firewall
sudo ufw enable
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw deny 3000   # Block direct access to Node.js
```

### 2. Application Security

```javascript
// security.config.js
module.exports = {
  // Helmet configuration
  helmet: {
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        scriptSrc: ["'self'", "'unsafe-inline'", "https://tradingview.com"],
        styleSrc: ["'self'", "'unsafe-inline'"],
        imgSrc: ["'self'", "data:", "https:"],
        connectSrc: ["'self'", "wss:", "https:"],
        fontSrc: ["'self'"],
        objectSrc: ["'none'"],
        mediaSrc: ["'self'"],
        frameSrc: ["https://tradingview.com"]
      }
    },
    hsts: {
      maxAge: 31536000,
      includeSubDomains: true,
      preload: true
    }
  },
  
  // Rate limiting
  rateLimit: {
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 1000, // requests per window
    message: 'Too many requests from this IP'
  }
}
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - run: npm ci
      - run: npm run type-check
      - run: npm run lint
      - run: npm run test:coverage
      - run: npm run build

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to Production
        uses: appleboy/ssh-action@v0.1.7
        with:
          host: ${{ secrets.HOST }}
          username: ${{ secrets.USERNAME }}
          key: ${{ secrets.SSH_KEY }}
          script: |
            cd /app/enterprise-gpu-backtester
            git pull origin main
            npm ci --only=production
            npm run build
            pm2 reload ecosystem.config.js --env production
```

## 📋 Deployment Checklist

### Pre-deployment
- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Database connections tested
- [ ] All tests passing
- [ ] Security audit completed
- [ ] Performance benchmarks verified

### Deployment
- [ ] Build application (`npm run build`)
- [ ] Deploy to production server
- [ ] Update environment variables
- [ ] Restart application services
- [ ] Verify health endpoints
- [ ] Check application logs

### Post-deployment
- [ ] Smoke tests completed
- [ ] Performance monitoring active
- [ ] Error tracking functional
- [ ] Backup systems verified
- [ ] Documentation updated
- [ ] Team notified of deployment

## 🆘 Troubleshooting

### Common Issues

#### 1. Build Failures
```bash
# Clear cache and rebuild
rm -rf .next node_modules
npm install
npm run build
```

#### 2. Database Connection Issues
```bash
# Test database connectivity
node -e "console.log('Testing DB connection...')"
# Check environment variables
env | grep -E "(HEAVYDB|MYSQL)"
```

#### 3. Memory Issues
```bash
# Increase Node.js memory limit
export NODE_OPTIONS="--max-old-space-size=4096"
npm run build
```

#### 4. SSL Certificate Issues
```bash
# Check certificate validity
openssl x509 -in /etc/ssl/certs/your-certificate.crt -text -noout
# Test SSL configuration
openssl s_client -connect your-domain.com:443
```

### Performance Issues

#### 1. Slow API Responses
- Check database query performance
- Review Redis cache hit rates
- Monitor memory usage
- Analyze slow query logs

#### 2. High Memory Usage
- Check for memory leaks
- Review garbage collection logs
- Monitor process memory usage
- Consider scaling horizontally

## 📞 Support

### Production Support
- **Emergency Contact**: +1-XXX-XXX-XXXX
- **Email**: support@your-domain.com
- **Slack**: #production-support
- **On-call Schedule**: Available 24/7

### Monitoring Dashboards
- **Application**: https://your-domain.com/admin/monitoring
- **Infrastructure**: https://grafana.your-domain.com
- **Logs**: https://kibana.your-domain.com
- **Uptime**: https://status.your-domain.com

---

**For additional support, please refer to the [main documentation](../README.md) or contact the development team.**