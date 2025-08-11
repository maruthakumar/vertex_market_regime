# Vertex Market Regime - Cloud-Native 8-Component System

🚀 **Next-Generation Market Regime Classification System**

Cloud-native, GPU-accelerated, adaptive learning framework for sophisticated options trading regime analysis.

## 🎯 **System Overview**

- **8-Component Architecture**: Modular, independent component system
- **774 Features**: Expert-optimized feature engineering pipeline  
- **Cloud-Native**: Google Cloud / Vertex AI integration
- **Adaptive Learning**: Continuous improvement through performance feedback
- **Configuration Bridge**: Seamless Excel configuration compatibility

## 🏗️ **Architecture**

```
Parquet Data → Apache Arrow → GPU Processing → 8 Components → Master Integration → 8 Regime Classification
```

### 8 Strategic Regimes
- **LVLD**: Low Volatility Low Delta
- **HVC**: High Volatility Contraction
- **VCPE**: Volatility Contraction Price Expansion
- **TBVE**: Trend Breaking Volatility Expansion
- **TBVS**: Trend Breaking Volatility Suppression
- **SCGS**: Strong Correlation Good Sentiment
- **PSED**: Poor Sentiment Elevated Divergence
- **CBV**: Choppy Breakout Volatility

## 📊 **Performance Targets**

- **Processing Time**: <600ms total analysis
- **Memory Usage**: <2.5GB optimized
- **Accuracy**: >87% regime classification
- **Features**: 774 expert-selected features
- **Throughput**: 1000+ requests/minute

## 🚀 **Quick Start**

### Prerequisites
- Python 3.9+
- Google Cloud SDK authenticated
- GPU support (optional, recommended)

### Installation
```bash
cd vertex_market_regime
pip install -r requirements.txt
python setup.py develop
```

### Configuration
```bash
# Copy your Excel configurations to configs/excel/
# Run configuration migration
python scripts/migrate_configs.py
```

### Run Analysis
```python
from src.api.fastapi_app import app
import uvicorn

uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 📁 **Directory Structure**

```
vertex_market_regime/
├── configs/           # Configuration management
├── src/              # Core source code
│   ├── components/   # 8-component modular system
│   ├── cloud/        # Google Cloud integration  
│   ├── data/         # Data processing pipeline
│   ├── ml/           # ML training & inference
│   └── api/          # FastAPI application
├── tests/            # Comprehensive testing
├── docs/             # Documentation
└── deployment/       # Cloud deployment
```

## 🔧 **Component Status**

| Component | Status | Implementation |
|-----------|--------|----------------|
| 1. Triple Straddle | 🟡 Migrating | DTE learning + weight optimization |
| 2. Greeks Sentiment | 🔴 Fixing | Gamma weight correction (0.0→1.5) |
| 3. OI-PA Trending | 🔴 Building | ATM ±7 cumulative analysis |
| 4. IV Skew | 🟡 Enhancing | Dual DTE framework |
| 5. ATR-EMA-CPR | 🔴 Building | Dual asset analysis |
| 6. Correlation | 🟡 Optimizing | 774-feature matrix |
| 7. Support/Resistance | 🔴 Building | Dynamic level detection |
| 8. Master Integration | 🔴 Building | 8-regime classification |

## 🌐 **API Endpoints**

- `POST /api/v1/analyze` - Main regime analysis
- `GET /api/v1/components/{id}/health` - Component health
- `POST /api/v1/config/migrate` - Configuration migration
- `GET /api/v1/models/status` - ML model status

## 🔗 **Integration**

### Legacy Backtester Bridge
```python
from src.legacy_bridge import BacktesterConnector

connector = BacktesterConnector()
result = await connector.get_regime_analysis(symbol="NIFTY")
```

### Excel Configuration Bridge
```python
from configs.excel.excel_parser import ExcelConfigurationBridge

parser = ExcelConfigurationBridge()
config = parser.migrate_excel_to_cloud_config("MR_CONFIG_REGIME_1.0.0.xlsx")
```

## 📈 **Monitoring**

- **Grafana**: Component performance dashboards
- **Prometheus**: Metrics collection and alerting
- **Cloud Monitoring**: Google Cloud native monitoring

## 🚀 **Deployment**

### Local Development
```bash
docker-compose up -d
```

### Google Cloud
```bash
./scripts/deploy_to_gcp.sh
```

### Kubernetes
```bash
kubectl apply -f deployment/kubernetes/
```

## 📚 **Documentation**

- [Architecture Overview](docs/architecture.md)
- [Component Guides](docs/components/)
- [API Documentation](docs/api/)
- [Migration Guide](docs/migration/)

## 🤝 **Contributing**

1. Run tests: `./scripts/run_tests.sh`
2. Check performance: `python scripts/benchmark_performance.py`
3. Validate configs: `python scripts/validate_configs.py`

---

**Status**: 🏗️ **Under Active Development**  
**Target**: Production-ready cloud-native market regime system  
**Timeline**: 12 weeks to full deployment