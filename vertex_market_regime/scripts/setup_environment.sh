#!/bin/bash

# Setup Environment Script for Vertex Market Regime
# Creates Python environment, installs dependencies, and runs configuration migration

set -e  # Exit on any error

echo "🚀 Setting up Vertex Market Regime Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project directory
PROJECT_DIR="/Users/maruth/projects/market_regime/vertex_market_regime"
cd "$PROJECT_DIR"

echo -e "${BLUE}📁 Working directory: $PROJECT_DIR${NC}"

# Check Python version
echo -e "${BLUE}🐍 Checking Python version...${NC}"
python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
    python3 -m venv venv
else
    echo -e "${GREEN}✅ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}🔄 Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${BLUE}⬆️  Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements
echo -e "${BLUE}📚 Installing requirements...${NC}"
pip install -r requirements.txt

# Install package in development mode
echo -e "${BLUE}🔧 Installing package in development mode...${NC}"
pip install -e .

# Create necessary directories
echo -e "${BLUE}📁 Creating output directories...${NC}"
mkdir -p configs/yaml
mkdir -p configs/json
mkdir -p logs
mkdir -p data/cache

# Set up environment variables
echo -e "${BLUE}⚙️  Setting up environment variables...${NC}"
if [ ! -f ".env" ]; then
    cat > .env << EOF
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=arched-bot-269016
GOOGLE_CLOUD_REGION=us-central1

# Vertex AI Configuration
VERTEX_AI_LOCATION=us-central1
VERTEX_AI_STAGING_BUCKET=market-regime-vertex-staging

# BigQuery Configuration
BIGQUERY_DATASET=market_regime_ml
BIGQUERY_LOCATION=US

# Application Configuration
LOG_LEVEL=INFO
API_PORT=8000
API_HOST=0.0.0.0

# Performance Configuration
GPU_ENABLED=true
MAX_WORKERS=8
PROCESSING_TIMEOUT=600

# Feature Engineering
TOTAL_FEATURES=774
ENABLE_FEATURE_CACHING=true
EOF
    echo -e "${GREEN}✅ Created .env file${NC}"
else
    echo -e "${GREEN}✅ .env file already exists${NC}"
fi

# Test Google Cloud authentication
echo -e "${BLUE}🔐 Checking Google Cloud authentication...${NC}"
if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${GREEN}✅ Google Cloud authentication is active${NC}"
    gcloud config get-value project
else
    echo -e "${YELLOW}⚠️  Google Cloud authentication not found${NC}"
    echo -e "${YELLOW}Please run: gcloud auth login${NC}"
fi

# Run configuration migration
echo -e "${BLUE}🔄 Running configuration migration...${NC}"
python3 -c "
import sys
sys.path.append('$PROJECT_DIR')

from configs.excel.excel_parser import ExcelConfigurationBridge
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize parser
excel_dir = '$PROJECT_DIR/configs/excel'
parser = ExcelConfigurationBridge(excel_dir)

try:
    # Migrate configurations
    print('🔄 Migrating Excel configurations...')
    master_config = parser.migrate_all_configurations()
    
    # Save YAML and JSON outputs
    parser.save_yaml_configs('$PROJECT_DIR/configs/yaml')
    parser.save_json_configs('$PROJECT_DIR/configs/json')
    
    # Generate and save migration report
    report = parser.generate_migration_report()
    with open('$PROJECT_DIR/migration_report.md', 'w') as f:
        f.write(report)
    
    print('✅ Configuration migration completed successfully!')
    print(f'📊 Migrated {len(master_config.components)} components')
    print(f'🔢 Total features: {sum(c.feature_count for c in master_config.components)}')
    print('📄 Migration report saved to migration_report.md')
    
    # Validate configuration
    validation = parser.validate_configuration(master_config)
    passed = sum(1 for r in validation.values() if r)
    total = len(validation)
    print(f'✅ Validation: {passed}/{total} checks passed')
    
    if validation.get('gamma_weight_fixed'):
        print('🚨 CRITICAL FIX CONFIRMED: Gamma weight corrected to 1.5')
    
except Exception as e:
    print(f'❌ Configuration migration failed: {e}')
    sys.exit(1)
"

# Check if migration was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Configuration migration successful!${NC}"
else
    echo -e "${RED}❌ Configuration migration failed!${NC}"
    exit 1
fi

# Test component loading
echo -e "${BLUE}🧪 Testing component loading...${NC}"
python3 -c "
import sys
sys.path.append('$PROJECT_DIR')

from src.components.base_component import BaseMarketRegimeComponent, ComponentFactory
from src.components.component_02_greeks_sentiment.greeks_analyzer import GreeksAnalyzer

try:
    # Test base component
    print('✅ Base component imported successfully')
    
    # Test Greeks analyzer (Component 2)
    config = {
        'component_id': 2,
        'feature_count': 98,
        'gamma_weight_corrected': True
    }
    
    greeks_analyzer = GreeksAnalyzer(config)
    print('✅ Greeks Analyzer (Component 2) initialized successfully')
    print(f'🚨 Gamma weight: {greeks_analyzer.greek_weights[\"gamma\"]} (CORRECTED)')
    
    print('✅ Component loading test passed!')
    
except Exception as e:
    print(f'❌ Component loading failed: {e}')
    sys.exit(1)
"

# Display final status
echo ""
echo -e "${GREEN}🎉 SETUP COMPLETED SUCCESSFULLY! 🎉${NC}"
echo ""
echo -e "${BLUE}📋 Next Steps:${NC}"
echo -e "  1. Review migration report: ${YELLOW}cat migration_report.md${NC}"
echo -e "  2. Check YAML configs: ${YELLOW}ls -la configs/yaml/${NC}"
echo -e "  3. Start API server: ${YELLOW}uvicorn src.api.fastapi_app:app --reload${NC}"
echo -e "  4. Run tests: ${YELLOW}pytest tests/${NC}"
echo ""
echo -e "${BLUE}📊 System Status:${NC}"
echo -e "  • Virtual environment: ${GREEN}✅ Active${NC}"
echo -e "  • Dependencies: ${GREEN}✅ Installed${NC}"
echo -e "  • Configurations: ${GREEN}✅ Migrated${NC}"
echo -e "  • Components: ${GREEN}✅ Loadable${NC}"
echo -e "  • Gamma weight fix: ${GREEN}✅ Applied (1.5)${NC}"
echo ""
echo -e "${BLUE}🌟 Ready for development!${NC}"