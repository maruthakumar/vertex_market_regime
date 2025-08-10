# 🌐 API Documentation

> Complete API reference for Enterprise GPU Backtester v7.1

## 📋 Overview

The Enterprise GPU Backtester API provides comprehensive endpoints for managing trading strategies, executing backtests, accessing real-time market data, and administrative functions. All endpoints follow RESTful principles with WebSocket support for real-time features.

### Base URL
```
Production: https://your-domain.com/api
Development: http://localhost:3000/api
```

### Authentication
All API endpoints (except public endpoints) require authentication using JWT tokens provided by NextAuth.js.

```http
Authorization: Bearer <jwt_token>
```

## 🔐 Authentication Endpoints

### POST /api/auth/signin
Authenticate user and obtain JWT token.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**Response:**
```json
{
  "user": {
    "id": "user-123",
    "email": "user@example.com",
    "name": "John Doe",
    "role": "user",
    "permissions": ["dashboard:read", "strategy:read"]
  },
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires": "2024-12-31T23:59:59.999Z"
}
```

### POST /api/auth/signout
Sign out current user.

**Response:**
```json
{
  "success": true,
  "message": "Successfully signed out"
}
```

### GET /api/auth/session
Get current user session.

**Response:**
```json
{
  "user": {
    "id": "user-123",
    "email": "user@example.com",
    "name": "John Doe",
    "role": "user"
  },
  "expires": "2024-12-31T23:59:59.999Z"
}
```

## 🎯 Strategy Management

### GET /api/strategies
Get all strategies for the authenticated user.

**Query Parameters:**
- `page` (number): Page number (default: 1)
- `limit` (number): Items per page (default: 20)
- `type` (string): Strategy type filter
- `status` (string): Status filter

**Response:**
```json
{
  "strategies": [
    {
      "id": "strategy-123",
      "name": "TBS Strategy 1",
      "type": "TBS",
      "status": "active",
      "description": "Time-based strategy",
      "parameters": {
        "maxLoss": -5000,
        "maxProfit": 10000,
        "exitTime": "15:25"
      },
      "createdAt": "2024-01-01T00:00:00Z",
      "updatedAt": "2024-01-01T00:00:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 50,
    "pages": 3
  }
}
```

### GET /api/strategies/:id
Get specific strategy by ID.

**Response:**
```json
{
  "id": "strategy-123",
  "name": "TBS Strategy 1",
  "type": "TBS",
  "status": "active",
  "description": "Time-based strategy",
  "parameters": {
    "maxLoss": -5000,
    "maxProfit": 10000,
    "exitTime": "15:25"
  },
  "configuration": {
    "portfolioSettings": { /* Excel data */ },
    "strategySettings": { /* Excel data */ }
  },
  "backtests": [
    {
      "id": "backtest-456",
      "name": "Test Run 1",
      "status": "completed",
      "totalReturn": 0.15
    }
  ],
  "createdAt": "2024-01-01T00:00:00Z",
  "updatedAt": "2024-01-01T00:00:00Z"
}
```

### POST /api/strategies
Create a new strategy.

**Request:**
```json
{
  "name": "New TBS Strategy",
  "type": "TBS",
  "description": "A new time-based strategy",
  "parameters": {
    "maxLoss": -3000,
    "maxProfit": 8000,
    "exitTime": "15:20"
  }
}
```

**Response:**
```json
{
  "id": "strategy-124",
  "name": "New TBS Strategy",
  "type": "TBS",
  "status": "draft",
  "createdAt": "2024-01-02T00:00:00Z"
}
```

### PUT /api/strategies/:id
Update existing strategy.

**Request:**
```json
{
  "name": "Updated Strategy Name",
  "description": "Updated description",
  "parameters": {
    "maxLoss": -4000,
    "maxProfit": 9000
  }
}
```

### DELETE /api/strategies/:id
Delete strategy.

**Response:**
```json
{
  "success": true,
  "message": "Strategy deleted successfully"
}
```

## 🔄 Backtest Management

### GET /api/backtests
Get all backtests for the authenticated user.

**Query Parameters:**
- `page` (number): Page number
- `limit` (number): Items per page
- `strategyId` (string): Filter by strategy
- `status` (string): Filter by status
- `startDate` (string): Filter by start date
- `endDate` (string): Filter by end date

**Response:**
```json
{
  "backtests": [
    {
      "id": "backtest-456",
      "strategyId": "strategy-123",
      "name": "NIFTY Backtest Jan 2024",
      "status": "completed",
      "startDate": "2024-01-01",
      "endDate": "2024-01-31",
      "symbol": "NIFTY",
      "initialCapital": 100000,
      "results": {
        "totalReturn": 0.15,
        "annualizedReturn": 1.95,
        "sharpeRatio": 1.2,
        "maxDrawdown": -0.08,
        "volatility": 0.16,
        "trades": 45,
        "winRate": 0.67,
        "avgWin": 2800,
        "avgLoss": -1200,
        "profitFactor": 1.8
      },
      "createdAt": "2024-01-01T00:00:00Z",
      "completedAt": "2024-01-31T23:59:59Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 25,
    "pages": 2
  }
}
```

### GET /api/backtests/:id
Get specific backtest with detailed results.

**Response:**
```json
{
  "id": "backtest-456",
  "strategyId": "strategy-123",
  "strategy": {
    "id": "strategy-123",
    "name": "TBS Strategy 1",
    "type": "TBS"
  },
  "name": "NIFTY Backtest Jan 2024",
  "status": "completed",
  "startDate": "2024-01-01",
  "endDate": "2024-01-31",
  "symbol": "NIFTY",
  "initialCapital": 100000,
  "results": {
    "totalReturn": 0.15,
    "annualizedReturn": 1.95,
    "sharpeRatio": 1.2,
    "maxDrawdown": -0.08,
    "volatility": 0.16,
    "trades": 45,
    "winRate": 0.67,
    "avgWin": 2800,
    "avgLoss": -1200,
    "profitFactor": 1.8,
    "calmarRatio": 24.4,
    "sortinoRatio": 1.9
  },
  "trades": [
    {
      "id": "trade-789",
      "entryTime": "2024-01-02T09:15:00Z",
      "exitTime": "2024-01-02T15:25:00Z",
      "symbol": "NIFTY",
      "type": "long",
      "quantity": 50,
      "entryPrice": 21000,
      "exitPrice": 21150,
      "pnl": 7500,
      "commission": 25,
      "netPnl": 7475
    }
  ],
  "performance": {
    "daily": [ /* Daily P&L data */ ],
    "monthly": [ /* Monthly performance */ ],
    "drawdown": [ /* Drawdown curve */ ]
  },
  "createdAt": "2024-01-01T00:00:00Z",
  "completedAt": "2024-01-31T23:59:59Z"
}
```

### POST /api/backtests
Start a new backtest.

**Request:**
```json
{
  "strategyId": "strategy-123",
  "name": "NIFTY Backtest Feb 2024",
  "startDate": "2024-02-01",
  "endDate": "2024-02-29",
  "symbol": "NIFTY",
  "initialCapital": 100000,
  "parameters": {
    "maxLoss": -5000,
    "maxProfit": 10000
  }
}
```

**Response:**
```json
{
  "id": "backtest-457",
  "status": "running",
  "estimatedCompletion": "2024-02-01T10:30:00Z",
  "progressUrl": "/api/backtests/457/progress"
}
```

### GET /api/backtests/:id/progress
Get real-time backtest progress.

**Response:**
```json
{
  "id": "backtest-457",
  "status": "running",
  "progress": 45.7,
  "currentDate": "2024-02-15",
  "estimatedCompletion": "2024-02-01T10:25:00Z",
  "statistics": {
    "tradesExecuted": 23,
    "currentPnL": 3250,
    "processedRows": 1256000
  }
}
```

### POST /api/backtests/:id/stop
Stop a running backtest.

**Response:**
```json
{
  "success": true,
  "message": "Backtest stopped successfully"
}
```

## 📊 Market Data

### GET /api/market-data/live
Get live market data.

**Query Parameters:**
- `symbols` (string): Comma-separated symbols (default: NIFTY)

**Response:**
```json
{
  "data": [
    {
      "symbol": "NIFTY",
      "price": 21000,
      "change": 150,
      "changePercent": 0.72,
      "volume": 125000000,
      "high": 21200,
      "low": 20850,
      "open": 20900,
      "previousClose": 20850,
      "timestamp": "2024-01-15T09:30:00Z",
      "status": "open"
    }
  ],
  "timestamp": "2024-01-15T09:30:00Z"
}
```

### GET /api/market-data/historical
Get historical market data.

**Query Parameters:**
- `symbol` (string): Symbol (required)
- `startDate` (string): Start date (YYYY-MM-DD)
- `endDate` (string): End date (YYYY-MM-DD)
- `interval` (string): Data interval (1m, 3m, 5m, 15m, 1h, 1d)

**Response:**
```json
{
  "symbol": "NIFTY",
  "interval": "3m",
  "data": [
    {
      "timestamp": "2024-01-15T09:15:00Z",
      "open": 20900,
      "high": 20950,
      "low": 20880,
      "close": 20920,
      "volume": 1250000
    }
  ]
}
```

### GET /api/market-data/option-chain
Get option chain data.

**Query Parameters:**
- `symbol` (string): Underlying symbol (required)
- `expiry` (string): Expiry date (YYYY-MM-DD)
- `strikes` (string): Comma-separated strike prices

**Response:**
```json
{
  "symbol": "NIFTY",
  "expiry": "2024-01-25",
  "spotPrice": 21000,
  "options": [
    {
      "strike": 21000,
      "callPrice": 125.50,
      "putPrice": 98.75,
      "callIV": 0.18,
      "putIV": 0.19,
      "callVolume": 15000,
      "putVolume": 12000,
      "callOI": 125000,
      "putOI": 98000,
      "callDelta": 0.52,
      "putDelta": -0.48,
      "gamma": 0.008,
      "theta": -12.5,
      "vega": 45.2
    }
  ]
}
```

## 🤖 ML Training & Analytics

### GET /api/ml/models
Get available ML models.

**Response:**
```json
{
  "models": [
    {
      "id": "model-123",
      "name": "NIFTY Volatility Predictor",
      "type": "regression",
      "status": "trained",
      "accuracy": 0.87,
      "lastTrained": "2024-01-01T00:00:00Z"
    }
  ]
}
```

### POST /api/ml/train
Start ML model training.

**Request:**
```json
{
  "modelType": "volatility_prediction",
  "symbol": "NIFTY",
  "features": ["price", "volume", "volatility"],
  "trainingPeriod": {
    "startDate": "2023-01-01",
    "endDate": "2023-12-31"
  },
  "parameters": {
    "learningRate": 0.001,
    "epochs": 100
  }
}
```

### GET /api/ml/predictions
Get ML predictions.

**Query Parameters:**
- `modelId` (string): Model ID
- `symbol` (string): Symbol
- `date` (string): Prediction date

**Response:**
```json
{
  "modelId": "model-123",
  "symbol": "NIFTY",
  "predictions": [
    {
      "date": "2024-01-16",
      "volatility": 0.18,
      "confidence": 0.85,
      "direction": "up",
      "probability": 0.72
    }
  ]
}
```

## 📈 Analytics & Reporting

### GET /api/analytics/performance
Get performance analytics.

**Query Parameters:**
- `strategyIds` (string): Comma-separated strategy IDs
- `startDate` (string): Start date
- `endDate` (string): End date
- `groupBy` (string): Group by period (day, week, month)

**Response:**
```json
{
  "performance": {
    "totalReturn": 0.156,
    "sharpeRatio": 1.23,
    "maxDrawdown": -0.08,
    "winRate": 0.67,
    "profitFactor": 1.85
  },
  "breakdown": [
    {
      "strategyId": "strategy-123",
      "strategyName": "TBS Strategy 1",
      "return": 0.12,
      "trades": 45,
      "winRate": 0.65
    }
  ],
  "timeSeries": [
    {
      "date": "2024-01-01",
      "cumulativeReturn": 0.01,
      "dailyPnL": 1250
    }
  ]
}
```

### GET /api/analytics/correlation
Get strategy correlation analysis.

**Response:**
```json
{
  "correlationMatrix": [
    {
      "strategy1": "strategy-123",
      "strategy2": "strategy-124",
      "correlation": 0.45,
      "pValue": 0.002
    }
  ],
  "diversificationScore": 0.78
}
```

## ⚙️ Configuration Management

### GET /api/config/excel-templates
Get available Excel templates.

**Response:**
```json
{
  "templates": [
    {
      "id": "template-tbs",
      "name": "TBS Configuration Template",
      "type": "TBS",
      "files": [
        "TBS_PORTFOLIO_SETTING_1.0.0.xlsx",
        "TBS_STRATEGY_SETTING_1.0.0.xlsx"
      ],
      "version": "1.0.0"
    }
  ]
}
```

### POST /api/config/upload
Upload Excel configuration files.

**Request:** Multipart form data with file uploads

**Response:**
```json
{
  "uploadId": "upload-789",
  "files": [
    {
      "originalName": "TBS_PORTFOLIO_SETTING_1.0.0.xlsx",
      "fileName": "upload-789-portfolio.xlsx",
      "size": 52348,
      "status": "uploaded"
    }
  ],
  "validationUrl": "/api/config/validate/upload-789"
}
```

### POST /api/config/validate/:uploadId
Validate uploaded Excel files.

**Response:**
```json
{
  "uploadId": "upload-789",
  "status": "valid",
  "files": [
    {
      "fileName": "upload-789-portfolio.xlsx",
      "sheets": [
        {
          "name": "PortfolioSetting",
          "rows": 25,
          "columns": 8,
          "valid": true
        }
      ]
    }
  ],
  "configuration": {
    "portfolioSettings": { /* Parsed data */ },
    "strategySettings": { /* Parsed data */ }
  }
}
```

## 🔴 Live Trading

### GET /api/live/positions
Get current live positions.

**Response:**
```json
{
  "positions": [
    {
      "id": "position-123",
      "symbol": "NIFTY",
      "strategy": "TBS Strategy 1",
      "type": "long",
      "quantity": 50,
      "entryPrice": 21000,
      "currentPrice": 21150,
      "unrealizedPnL": 7500,
      "entryTime": "2024-01-15T09:15:00Z"
    }
  ],
  "summary": {
    "totalPositions": 5,
    "totalUnrealizedPnL": 12500,
    "dayPnL": 3250
  }
}
```

### POST /api/live/orders
Place a live order.

**Request:**
```json
{
  "strategyId": "strategy-123",
  "symbol": "NIFTY",
  "type": "market",
  "side": "buy",
  "quantity": 50,
  "price": 21000
}
```

**Response:**
```json
{
  "orderId": "order-456",
  "status": "submitted",
  "message": "Order submitted successfully"
}
```

## 🛡️ Admin Endpoints

### GET /api/admin/users
Get all users (Admin only).

**Response:**
```json
{
  "users": [
    {
      "id": "user-123",
      "email": "user@example.com",
      "name": "John Doe",
      "role": "user",
      "status": "active",
      "lastLogin": "2024-01-15T08:30:00Z",
      "createdAt": "2024-01-01T00:00:00Z"
    }
  ]
}
```

### GET /api/admin/system-stats
Get system statistics.

**Response:**
```json
{
  "system": {
    "uptime": 86400,
    "memoryUsage": {
      "used": 2.1,
      "total": 8.0,
      "percentage": 26.25
    },
    "cpuUsage": 15.3
  },
  "database": {
    "heavydb": {
      "status": "connected",
      "activeConnections": 15,
      "queryTime": 45.2
    },
    "mysql": {
      "status": "connected",
      "activeConnections": 8,
      "queryTime": 12.8
    }
  },
  "api": {
    "requestsPerMinute": 150,
    "averageResponseTime": 85.3,
    "errorRate": 0.02
  }
}
```

## 📡 WebSocket Events

### Connection
```javascript
const ws = new WebSocket('wss://your-domain.com/ws')

// Authentication after connection
ws.send(JSON.stringify({
  type: 'auth',
  token: 'your-jwt-token'
}))
```

### Market Data Updates
```javascript
// Subscribe to market data
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'market-data',
  symbols: ['NIFTY', 'BANKNIFTY']
}))

// Receive updates
ws.onmessage = (event) => {
  const data = JSON.parse(event.data)
  if (data.type === 'market-data') {
    console.log('Market update:', data.data)
  }
}
```

### Backtest Progress
```javascript
// Subscribe to backtest progress
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'backtest-progress',
  backtestId: 'backtest-456'
}))

// Receive progress updates
ws.onmessage = (event) => {
  const data = JSON.parse(event.data)
  if (data.type === 'backtest-progress') {
    console.log('Progress:', data.data.progress + '%')
  }
}
```

## 🚨 Error Responses

### Standard Error Format
```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Invalid request parameters",
    "details": {
      "field": "startDate",
      "issue": "Invalid date format"
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "requestId": "req-123456"
  }
}
```

### HTTP Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `422` - Validation Error
- `429` - Rate Limited
- `500` - Internal Server Error
- `503` - Service Unavailable

## 📊 Rate Limits

### Default Limits
- **Authentication**: 5 requests per minute
- **API Endpoints**: 1000 requests per 15 minutes
- **WebSocket**: 100 messages per minute
- **File Upload**: 10 uploads per hour

### Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642248000
```

## 🔧 SDK Examples

### JavaScript/TypeScript
```typescript
import { BacktesterAPI } from '@enterprise-gpu-backtester/sdk'

const api = new BacktesterAPI({
  baseURL: 'https://your-domain.com/api',
  token: 'your-jwt-token'
})

// Get strategies
const strategies = await api.strategies.getAll()

// Start backtest
const backtest = await api.backtests.create({
  strategyId: 'strategy-123',
  startDate: '2024-01-01',
  endDate: '2024-01-31'
})
```

### Python
```python
from enterprise_gpu_backtester import BacktesterAPI

api = BacktesterAPI(
    base_url='https://your-domain.com/api',
    token='your-jwt-token'
)

# Get strategies
strategies = api.strategies.get_all()

# Start backtest
backtest = api.backtests.create(
    strategy_id='strategy-123',
    start_date='2024-01-01',
    end_date='2024-01-31'
)
```

## 📝 Changelog

### v7.1.0
- ✅ Complete API redesign with RESTful principles
- ✅ WebSocket real-time capabilities
- ✅ Enhanced authentication and authorization
- ✅ Comprehensive error handling
- ✅ Rate limiting and security features
- ✅ OpenAPI/Swagger documentation

---

**For more information, see the [main documentation](../README.md) or contact the API team.**