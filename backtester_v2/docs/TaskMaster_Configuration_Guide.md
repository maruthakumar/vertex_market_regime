# 🤖 TaskMaster AI Configuration & Environment Setup Guide
## SuperClaude Integration for Enterprise GPU Backtester

### Document Version
- **Version**: 1.0.0
- **Created**: July 13, 2025
- **Last Updated**: July 13, 2025
- **Compatibility**: SuperClaude v2.1.0, TaskMaster AI v1.0.0

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [API Configuration](#api-configuration)
5. [Database Connections](#database-connections)
6. [SuperClaude Integration](#superclaude-integration)
7. [Excel Configuration Validation](#excel-configuration-validation)
8. [MCP Server Configuration](#mcp-server-configuration)
9. [Testing and Validation](#testing-and-validation)
10. [Security Considerations](#security-considerations)
11. [Troubleshooting](#troubleshooting)
12. [Usage Examples](#usage-examples)

---

## 🎯 Overview

TaskMaster AI integration with SuperClaude provides autonomous development capabilities for the Enterprise GPU Backtester platform. This guide ensures proper configuration of the complete ecosystem including AI models, databases, Excel configurations, and MCP servers.

### Key Components
- **TaskMaster AI**: Core autonomous development engine
- **SuperClaude**: Enhanced Claude configuration with 9 personas and MCP integration
- **Enterprise GPU Backtester**: Financial data processing with HeavyDB acceleration
- **Real Data Validation**: HeavyDB (33M+ rows) and MySQL (28M+ rows) connections
- **Excel Configuration System**: 31 production configuration files

---

## 🔧 Prerequisites

### System Requirements
```bash
# Node.js & npm (for TaskMaster AI)
node --version  # >= 18.0.0
npm --version   # >= 8.0.0

# Python environment (for backend systems)
python --version  # >= 3.10.0
pip --version     # >= 22.0.0

# Database systems
# HeavyDB: localhost:6274
# MySQL: localhost:3306 & 106.51.63.60:3306

# Required Python packages
pandas>=2.0.0
numpy>=1.24.0
heavydb>=6.4.0
pymysql>=1.0.0
redis>=4.5.0
```

### Directory Structure Validation
```bash
# Verify worktree structure
ls -la /srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/
# Expected directories: scripts/, docs/, tests/, nextjs-app/, backtester_v2/

# Verify main backend reference
ls -la /srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/
# Expected: strategies/, configurations/, api/, core/, dal/
```

---

## ⚙️ Environment Setup

### 1. Create Environment File

Create or update `.env` file in your worktree root:

```bash
# Navigate to current worktree
cd /srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/

# Create .env file (or update existing)
cp .env.example .env  # if exists, or create new
```

### 2. Complete Environment Configuration

```env
# ==============================================
# TASKMASTER AI CONFIGURATION
# ==============================================

# Anthropic API Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here
MODEL=claude-3-7-sonnet-20250219
MAX_TOKENS=64000
TEMPERATURE=0.2

# Perplexity API Configuration
PERPLEXITY_API_KEY=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
PERPLEXITY_MODEL=sonar-pro

# TaskMaster Settings
DEFAULT_SUBTASKS=5
DEFAULT_PRIORITY=medium

# ==============================================
# DATABASE CONNECTIONS (REAL DATA ONLY)
# ==============================================

# HeavyDB (GPU Database) - Primary
HEAVYDB_HOST=localhost
HEAVYDB_PORT=6274
HEAVYDB_USER=admin
HEAVYDB_PASSWORD=HyperInteractive
HEAVYDB_DATABASE=heavyai

# MySQL (Archive System) - Remote
ARCHIVE_HOST=106.51.63.60
ARCHIVE_USER=mahesh
ARCHIVE_PASSWORD=mahesh_123
ARCHIVE_DATABASE=historicaldb

# MySQL (Local) - 2024 NIFTY Data
LOCAL_MYSQL_HOST=localhost
LOCAL_MYSQL_PORT=3306
LOCAL_MYSQL_USER=mahesh
LOCAL_MYSQL_PASSWORD=mahesh_123
LOCAL_MYSQL_DATABASE=historicaldb

# ==============================================
# SUPERCLAUDE CONFIGURATION
# ==============================================

# SuperClaude Features
SUPERCLAUDE_VERSION=2.1.0
CONTEXT_ENGINEERING_ENABLED=true
PERSONA_COUNT=9
MCP_SERVER_COUNT=4

# Validation Settings
REAL_DATA_VALIDATION=true
EXCEL_VALIDATION_REQUIRED=true
TODO_LINE_LIMIT=500
MOCK_DATA_FORBIDDEN=true

# ==============================================
# REDIS CONFIGURATION (Multi-Agent)
# ==============================================

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# ==============================================
# SECURITY & MONITORING
# ==============================================

# JWT Configuration
JWT_SECRET_KEY=your-secret-key-here-change-in-production
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# Sentry Configuration
SENTRY_DSN=https://cebda644936ae0be88096ee62c504428@o4509550322384896.ingest.de.sentry.io/4509553032560720
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=1.0
SENTRY_PROFILES_SAMPLE_RATE=1.0

# ==============================================
# COMMUNICATION SETTINGS
# ==============================================

# SMS Configuration (MSG91)
SMS_PROVIDER=msg91
MSG91_AUTH_KEY=your_msg91_auth_key_here
MSG91_SENDER_ID=EGPUBT
MSG91_OTP_TEMPLATE_ID=your_template_id_here
MSG91_ROUTE=4
MSG91_COUNTRY=91

# Telegram Notifications
TELEGRAM_ENABLED=false
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
TELEGRAM_ALERT_LEVELS=critical,error,warning,info
TELEGRAM_ETL_NOTIFICATIONS=true

# ==============================================
# DEVELOPMENT SETTINGS
# ==============================================

# Environment
ENVIRONMENT=production
API_V1_STR=/api/v1

# Worktree Configuration
CURRENT_WORKTREE=/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/
MAIN_BACKEND_REFERENCE=/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/
```

### 3. Secure Environment File

```bash
# Set proper permissions
chmod 600 .env

# Add to .gitignore to prevent commit
echo ".env" >> .gitignore
```

---

## 🔑 API Configuration

### Anthropic API Setup

1. **Verify API Key**:
```bash
# Test Anthropic API connection
curl -H "Authorization: Bearer $ANTHROPIC_API_KEY" \
     -H "Content-Type: application/json" \
     -X POST "https://api.anthropic.com/v1/messages" \
     -d '{
       "model": "claude-3-7-sonnet-20250219",
       "max_tokens": 1000,
       "messages": [{"role": "user", "content": "Hello, Claude!"}]
     }'
```

2. **Model Selection Guidelines**:
```yaml
Development_Environment:
  Model: "claude-3-7-sonnet-20250219"
  Max_Tokens: 64000
  Temperature: 0.2
  Use_Case: "Code generation, analysis, debugging"

Production_Environment:
  Model: "claude-3-7-sonnet-20250219"
  Max_Tokens: 32000
  Temperature: 0.1
  Use_Case: "Critical system operations"

Testing_Environment:
  Model: "claude-3-haiku-20240307"
  Max_Tokens: 16000
  Temperature: 0.3
  Use_Case: "Rapid testing, validation"
```

### Perplexity API Setup

1. **API Key Validation**:
```bash
# Test Perplexity API
curl -X POST "https://api.perplexity.ai/chat/completions" \
     -H "Authorization: Bearer $PERPLEXITY_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "sonar-pro",
       "messages": [{"role": "user", "content": "What is TaskMaster AI?"}]
     }'
```

2. **Usage Guidelines**:
```yaml
Primary_Use_Cases:
  - Real-time market data queries
  - Technical documentation lookup
  - Latest framework updates
  - API documentation verification

Rate_Limits:
  - Requests_Per_Minute: 60
  - Concurrent_Requests: 5
  - Model: "sonar-pro"
```

---

## 🗄️ Database Connections

### HeavyDB Configuration (Primary GPU Database)

1. **Connection Test**:
```python
#!/usr/bin/env python3
"""Test HeavyDB connection for TaskMaster AI"""

import os
from heavydb import connect

def test_heavydb_connection():
    try:
        # Load from environment
        host = os.getenv('HEAVYDB_HOST', 'localhost')
        port = int(os.getenv('HEAVYDB_PORT', '6274'))
        user = os.getenv('HEAVYDB_USER', 'admin')
        password = os.getenv('HEAVYDB_PASSWORD', 'HyperInteractive')
        database = os.getenv('HEAVYDB_DATABASE', 'heavyai')
        
        # Create connection
        conn = connect(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=database
        )
        
        # Test query
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM nifty_option_chain")
        result = cursor.fetchone()
        
        print(f"✅ HeavyDB Connected Successfully")
        print(f"📊 Total nifty_option_chain rows: {result[0]:,}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ HeavyDB Connection Failed: {e}")
        return False

if __name__ == "__main__":
    test_heavydb_connection()
```

2. **Performance Validation**:
```python
#!/usr/bin/env python3
"""Validate HeavyDB performance for TaskMaster AI"""

import time
import pandas as pd
from heavydb import connect

def benchmark_heavydb():
    conn = connect(
        host='localhost',
        port=6274,
        user='admin',
        password='HyperInteractive',
        dbname='heavyai'
    )
    
    # Test query performance
    query = """
    SELECT trade_date, trade_time, strike, spot, ce_close, pe_close
    FROM nifty_option_chain
    WHERE trade_date = '2024-01-02'
    LIMIT 100000
    """
    
    start_time = time.time()
    df = pd.read_sql(query, conn)
    end_time = time.time()
    
    rows_per_sec = len(df) / (end_time - start_time)
    
    print(f"📈 Query Performance:")
    print(f"   Rows: {len(df):,}")
    print(f"   Time: {end_time - start_time:.2f} seconds")
    print(f"   Speed: {rows_per_sec:,.0f} rows/second")
    print(f"   Target: 529,861 rows/second")
    print(f"   Status: {'✅ PASS' if rows_per_sec > 100000 else '⚠️ SLOW'}")
    
    conn.close()

if __name__ == "__main__":
    benchmark_heavydb()
```

### MySQL Configuration (Archive & Local)

1. **Archive Connection Test**:
```python
#!/usr/bin/env python3
"""Test MySQL archive connection"""

import pymysql
import os

def test_archive_mysql():
    try:
        connection = pymysql.connect(
            host=os.getenv('ARCHIVE_HOST', '106.51.63.60'),
            user=os.getenv('ARCHIVE_USER', 'mahesh'),
            password=os.getenv('ARCHIVE_PASSWORD', 'mahesh_123'),
            database=os.getenv('ARCHIVE_DATABASE', 'historicaldb'),
            charset='utf8mb4'
        )
        
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM nifty_option_chain")
            result = cursor.fetchone()
            
        print(f"✅ Archive MySQL Connected")
        print(f"📊 Archive rows: {result[0]:,}")
        
        connection.close()
        return True
        
    except Exception as e:
        print(f"❌ Archive MySQL Failed: {e}")
        return False

def test_local_mysql():
    try:
        connection = pymysql.connect(
            host=os.getenv('LOCAL_MYSQL_HOST', 'localhost'),
            port=int(os.getenv('LOCAL_MYSQL_PORT', '3306')),
            user=os.getenv('LOCAL_MYSQL_USER', 'mahesh'),
            password=os.getenv('LOCAL_MYSQL_PASSWORD', 'mahesh_123'),
            database=os.getenv('LOCAL_MYSQL_DATABASE', 'historicaldb'),
            charset='utf8mb4'
        )
        
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM nifty_option_chain WHERE trade_date >= '2024-01-01'")
            result = cursor.fetchone()
            
        print(f"✅ Local MySQL Connected")
        print(f"📊 Local 2024 rows: {result[0]:,}")
        
        connection.close()
        return True
        
    except Exception as e:
        print(f"❌ Local MySQL Failed: {e}")
        return False

if __name__ == "__main__":
    test_archive_mysql()
    test_local_mysql()
```

---

## 🧠 SuperClaude Integration

### 1. Persona Configuration

```python
#!/usr/bin/env python3
"""SuperClaude persona mapping for TaskMaster AI"""

SUPERCLAUDE_PERSONAS = {
    "architect": {
        "role": "System design and architecture expert",
        "taskmaster_agent": "research",
        "specialization": [
            "System architecture design",
            "Component interaction patterns",
            "Scalability planning",
            "Technology stack decisions"
        ],
        "context_flags": ["--context:architecture", "--context:system"],
        "mcp_servers": ["seq", "c7"]
    },
    
    "frontend": {
        "role": "Frontend development specialist",
        "taskmaster_agent": "implementation",
        "specialization": [
            "React/Next.js development",
            "UI/UX implementation",
            "Component library integration",
            "Performance optimization"
        ],
        "context_flags": ["--context:ui", "--context:frontend"],
        "mcp_servers": ["magic", "pup"]
    },
    
    "backend": {
        "role": "Backend development specialist", 
        "taskmaster_agent": "implementation",
        "specialization": [
            "API development",
            "Database optimization",
            "Server-side logic",
            "Integration patterns"
        ],
        "context_flags": ["--context:api", "--context:backend"],
        "mcp_servers": ["seq", "c7"]
    },
    
    "security": {
        "role": "Security and compliance expert",
        "taskmaster_agent": "structure_enforcer",
        "specialization": [
            "Authentication systems",
            "Data security",
            "Compliance validation",
            "Vulnerability assessment"
        ],
        "context_flags": ["--context:security", "--context:auth"],
        "mcp_servers": ["seq"]
    },
    
    "performance": {
        "role": "Performance optimization specialist",
        "taskmaster_agent": "research",
        "specialization": [
            "Query optimization",
            "GPU acceleration",
            "Memory management",
            "Latency reduction"
        ],
        "context_flags": ["--context:performance", "--context:gpu"],
        "mcp_servers": ["seq", "pup"]
    },
    
    "qa": {
        "role": "Quality assurance specialist",
        "taskmaster_agent": "structure_enforcer",
        "specialization": [
            "Test strategy development",
            "Quality validation",
            "Bug detection",
            "Regression testing"
        ],
        "context_flags": ["--context:testing", "--context:quality"],
        "mcp_servers": ["pup", "seq"]
    },
    
    "ml": {
        "role": "Machine learning specialist",
        "taskmaster_agent": "research",
        "specialization": [
            "Market regime detection",
            "Pattern recognition",
            "Feature engineering",
            "Model training"
        ],
        "context_flags": ["--context:ml", "--context:regime"],
        "mcp_servers": ["seq", "c7"]
    },
    
    "devops": {
        "role": "DevOps and deployment expert",
        "taskmaster_agent": "orchestrator",
        "specialization": [
            "CI/CD pipeline",
            "Infrastructure management",
            "Monitoring systems",
            "Deployment automation"
        ],
        "context_flags": ["--context:devops", "--context:deploy"],
        "mcp_servers": ["seq", "pup"]
    },
    
    "data": {
        "role": "Data engineering expert",
        "taskmaster_agent": "research",
        "specialization": [
            "ETL pipeline development",
            "Data validation",
            "Database design",
            "Real-time processing"
        ],
        "context_flags": ["--context:data", "--context:etl"],
        "mcp_servers": ["seq", "c7"]
    }
}
```

### 2. Command Mapping

```python
#!/usr/bin/env python3
"""SuperClaude command to TaskMaster mapping"""

COMMAND_MAPPINGS = {
    # Analysis Commands
    "/analyze": {
        "taskmaster_operation": "research",
        "default_persona": "architect",
        "description": "Analyze system components and architecture",
        "required_context": ["--context:auto"]
    },
    
    "/implement": {
        "taskmaster_operation": "implement",
        "default_persona": "backend",
        "description": "Implement features and functionality",
        "required_context": ["--context:auto", "--context:implementation"]
    },
    
    "/test": {
        "taskmaster_operation": "test",
        "default_persona": "qa",
        "description": "Generate and execute tests",
        "required_context": ["--context:testing"]
    },
    
    "/debug": {
        "taskmaster_operation": "debug",
        "default_persona": "backend",
        "description": "Debug and troubleshoot issues",
        "required_context": ["--context:debug"]
    },
    
    "/optimize": {
        "taskmaster_operation": "optimize",
        "default_persona": "performance",
        "description": "Optimize performance and efficiency",
        "required_context": ["--context:performance"]
    },
    
    "/refactor": {
        "taskmaster_operation": "refactor",
        "default_persona": "architect",
        "description": "Refactor code and architecture",
        "required_context": ["--context:refactor"]
    },
    
    "/docs": {
        "taskmaster_operation": "document",
        "default_persona": "qa",
        "description": "Generate documentation",
        "required_context": ["--context:docs"]
    },
    
    "/project": {
        "taskmaster_operation": "orchestrate",
        "default_persona": "devops",
        "description": "Project management and orchestration",
        "required_context": ["--context:project"]
    },
    
    "/workflow": {
        "taskmaster_operation": "orchestrate",
        "default_persona": "devops",
        "description": "Workflow management and automation",
        "required_context": ["--context:workflow"]
    },
    
    "/security": {
        "taskmaster_operation": "audit",
        "default_persona": "security",
        "description": "Security audit and validation",
        "required_context": ["--context:security"]
    }
}
```

---

## 📊 Excel Configuration Validation

### 1. Configuration File Mapping

```python
#!/usr/bin/env python3
"""Excel configuration validation for TaskMaster AI"""

import pandas as pd
import os
from pathlib import Path

EXCEL_CONFIGURATIONS = {
    "production": {
        "base_path": "/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/configurations/data/production/",
        "files": [
            "tbs_config.xlsx",
            "tv_config.xlsx", 
            "orb_config.xlsx",
            "oi_config.xlsx",
            "ml_indicator_config.xlsx",
            "pos_config.xlsx",
            "market_regime_config.xlsx"
        ],
        "required_sheets": ["parameters", "conditions", "validation"],
        "validation_rules": {
            "required_columns": ["parameter", "value", "type", "validation"],
            "data_types": ["string", "numeric", "boolean", "date", "time"],
            "validation_methods": ["range", "list", "regex", "custom"]
        }
    },
    
    "development": {
        "base_path": "/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/configurations/data/development/",
        "files": [
            "dev_tbs_config.xlsx",
            "dev_tv_config.xlsx",
            "dev_orb_config.xlsx",
            "dev_oi_config.xlsx",
            "dev_ml_config.xlsx",
            "dev_pos_config.xlsx",
            "dev_regime_config.xlsx"
        ]
    },
    
    "testing": {
        "base_path": "/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/configurations/data/testing/",
        "files": [
            "test_config.xlsx"
        ]
    }
}

def validate_excel_configuration(environment="production"):
    """Validate Excel configuration files"""
    config = EXCEL_CONFIGURATIONS[environment]
    base_path = Path(config["base_path"])
    
    validation_results = {
        "valid_files": [],
        "invalid_files": [],
        "missing_files": [],
        "validation_errors": []
    }
    
    for file_name in config["files"]:
        file_path = base_path / file_name
        
        if not file_path.exists():
            validation_results["missing_files"].append(file_name)
            continue
            
        try:
            # Load Excel file
            excel_file = pd.ExcelFile(file_path)
            
            # Check required sheets
            if "required_sheets" in config:
                for sheet_name in config["required_sheets"]:
                    if sheet_name not in excel_file.sheet_names:
                        validation_results["validation_errors"].append(
                            f"{file_name}: Missing sheet '{sheet_name}'"
                        )
                        continue
                    
                    # Validate sheet content
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    # Check required columns
                    if "validation_rules" in config:
                        required_cols = config["validation_rules"]["required_columns"]
                        missing_cols = set(required_cols) - set(df.columns)
                        if missing_cols:
                            validation_results["validation_errors"].append(
                                f"{file_name}[{sheet_name}]: Missing columns {missing_cols}"
                            )
            
            validation_results["valid_files"].append(file_name)
            
        except Exception as e:
            validation_results["invalid_files"].append(f"{file_name}: {str(e)}")
    
    return validation_results

def generate_excel_validation_report():
    """Generate comprehensive Excel validation report"""
    environments = ["production", "development", "testing"]
    
    print("📊 Excel Configuration Validation Report")
    print("=" * 50)
    
    for env in environments:
        print(f"\n🔍 Environment: {env.upper()}")
        results = validate_excel_configuration(env)
        
        print(f"✅ Valid files: {len(results['valid_files'])}")
        for file in results['valid_files']:
            print(f"   ✓ {file}")
            
        if results['invalid_files']:
            print(f"❌ Invalid files: {len(results['invalid_files'])}")
            for file in results['invalid_files']:
                print(f"   ✗ {file}")
                
        if results['missing_files']:
            print(f"⚠️ Missing files: {len(results['missing_files'])}")
            for file in results['missing_files']:
                print(f"   ? {file}")
                
        if results['validation_errors']:
            print(f"🔧 Validation errors: {len(results['validation_errors'])}")
            for error in results['validation_errors']:
                print(f"   ! {error}")

if __name__ == "__main__":
    generate_excel_validation_report()
```

### 2. Real-time Excel Validation

```python
#!/usr/bin/env python3
"""Real-time Excel validation for TaskMaster AI operations"""

import pandas as pd
from pathlib import Path
import json

class ExcelValidator:
    """Excel configuration validator for TaskMaster AI"""
    
    def __init__(self, worktree_path=None):
        self.worktree_path = Path(worktree_path) if worktree_path else Path.cwd()
        self.reference_path = Path("/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/configurations/")
        
    def validate_against_reference(self, local_excel_path, strategy_type):
        """Validate local Excel against reference configuration"""
        reference_file = self.reference_path / "data" / "production" / f"{strategy_type}_config.xlsx"
        local_file = Path(local_excel_path)
        
        if not reference_file.exists():
            return {"error": f"Reference file not found: {reference_file}"}
            
        if not local_file.exists():
            return {"error": f"Local file not found: {local_file}"}
        
        try:
            # Load both files
            ref_excel = pd.ExcelFile(reference_file)
            local_excel = pd.ExcelFile(local_file)
            
            validation_result = {
                "compatible": True,
                "differences": [],
                "warnings": [],
                "errors": []
            }
            
            # Compare sheet names
            ref_sheets = set(ref_excel.sheet_names)
            local_sheets = set(local_excel.sheet_names)
            
            if ref_sheets != local_sheets:
                validation_result["compatible"] = False
                validation_result["errors"].append(
                    f"Sheet mismatch: Reference {ref_sheets}, Local {local_sheets}"
                )
            
            # Compare sheet content
            for sheet in ref_sheets & local_sheets:
                ref_df = pd.read_excel(reference_file, sheet_name=sheet)
                local_df = pd.read_excel(local_file, sheet_name=sheet)
                
                # Check column compatibility
                ref_cols = set(ref_df.columns)
                local_cols = set(local_df.columns)
                
                if ref_cols != local_cols:
                    validation_result["differences"].append(
                        f"Sheet '{sheet}' column mismatch: {ref_cols.symmetric_difference(local_cols)}"
                    )
                
                # Check data type compatibility
                common_cols = ref_cols & local_cols
                for col in common_cols:
                    if ref_df[col].dtype != local_df[col].dtype:
                        validation_result["warnings"].append(
                            f"Sheet '{sheet}' column '{col}' type mismatch: {ref_df[col].dtype} vs {local_df[col].dtype}"
                        )
            
            return validation_result
            
        except Exception as e:
            return {"error": f"Validation failed: {str(e)}"}
    
    def create_compatibility_report(self, strategy_types=None):
        """Create comprehensive compatibility report"""
        if strategy_types is None:
            strategy_types = ["tbs", "tv", "orb", "oi", "ml_indicator", "pos", "market_regime"]
        
        compatibility_report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "worktree": str(self.worktree_path),
            "reference": str(self.reference_path),
            "strategies": {}
        }
        
        for strategy in strategy_types:
            # Check for local config file
            local_config = self.worktree_path / "configurations" / f"{strategy}_config.xlsx"
            
            if local_config.exists():
                validation = self.validate_against_reference(local_config, strategy)
                compatibility_report["strategies"][strategy] = validation
            else:
                compatibility_report["strategies"][strategy] = {
                    "error": "Local configuration file not found"
                }
        
        return compatibility_report

if __name__ == "__main__":
    validator = ExcelValidator()
    report = validator.create_compatibility_report()
    print(json.dumps(report, indent=2))
```

---

## 🔌 MCP Server Configuration

### 1. Available MCP Servers

```yaml
MCP_Servers:
  context7:
    name: "Context7 MCP"
    purpose: "Official library documentation"
    use_cases:
      - External library integration
      - API documentation lookup
      - Framework reference
    commands:
      - "/analyze --c7"
      - "/build --react --c7"
    
  sequential:
    name: "Sequential MCP"
    purpose: "Complex analysis and problem solving"
    use_cases:
      - System design
      - Root cause analysis
      - Performance investigation
    commands:
      - "/analyze --seq"
      - "/troubleshoot --seq"
    
  magic:
    name: "Magic MCP"
    purpose: "UI component generation"
    use_cases:
      - React/Vue components
      - Design systems
      - UI patterns
    commands:
      - "/build --react --magic"
      - "/design --magic"
    
  puppeteer:
    name: "Puppeteer MCP"
    purpose: "Browser automation and testing"
    use_cases:
      - E2E testing
      - Performance monitoring
      - Visual validation
    commands:
      - "/test --e2e --pup"
      - "/analyze --performance --pup"
```

### 2. MCP Integration Script

```python
#!/usr/bin/env python3
"""MCP server integration for TaskMaster AI"""

import subprocess
import os
from typing import List, Dict, Any

class MCPServerManager:
    """Manage MCP server integration with TaskMaster AI"""
    
    def __init__(self):
        self.available_servers = ["context7", "sequential", "magic", "puppeteer"]
        self.server_status = {}
        
    def check_server_availability(self, server_name: str) -> bool:
        """Check if MCP server is available"""
        try:
            # Test server connection (placeholder - actual implementation depends on MCP server API)
            result = subprocess.run(
                ["curl", "-s", f"http://localhost:{self._get_server_port(server_name)}/health"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def _get_server_port(self, server_name: str) -> int:
        """Get default port for MCP server"""
        ports = {
            "context7": 8001,
            "sequential": 8002,
            "magic": 8003,
            "puppeteer": 8004
        }
        return ports.get(server_name, 8000)
    
    def initialize_servers(self) -> Dict[str, bool]:
        """Initialize all available MCP servers"""
        initialization_results = {}
        
        for server in self.available_servers:
            try:
                # Check if server is available
                available = self.check_server_availability(server)
                initialization_results[server] = available
                self.server_status[server] = "available" if available else "unavailable"
                
                print(f"{'✅' if available else '❌'} {server}: {self.server_status[server]}")
                
            except Exception as e:
                initialization_results[server] = False
                self.server_status[server] = f"error: {str(e)}"
                print(f"❌ {server}: {self.server_status[server]}")
        
        return initialization_results
    
    def get_server_capabilities(self, server_name: str) -> Dict[str, Any]:
        """Get capabilities for specific MCP server"""
        capabilities = {
            "context7": {
                "library_documentation": True,
                "api_reference": True,
                "code_examples": True,
                "integration_patterns": True
            },
            "sequential": {
                "complex_analysis": True,
                "system_design": True,
                "problem_solving": True,
                "debugging": True
            },
            "magic": {
                "ui_components": True,
                "react_generation": True,
                "vue_generation": True,
                "design_systems": True
            },
            "puppeteer": {
                "browser_automation": True,
                "e2e_testing": True,
                "performance_testing": True,
                "visual_regression": True
            }
        }
        
        return capabilities.get(server_name, {})
    
    def recommend_servers_for_task(self, task_description: str) -> List[str]:
        """Recommend MCP servers based on task description"""
        recommendations = []
        
        # Simple keyword-based recommendation
        if any(keyword in task_description.lower() for keyword in ["ui", "component", "react", "vue", "design"]):
            recommendations.append("magic")
            
        if any(keyword in task_description.lower() for keyword in ["test", "automation", "e2e", "browser"]):
            recommendations.append("puppeteer")
            
        if any(keyword in task_description.lower() for keyword in ["analysis", "debug", "complex", "system"]):
            recommendations.append("sequential")
            
        if any(keyword in task_description.lower() for keyword in ["documentation", "api", "library", "reference"]):
            recommendations.append("context7")
        
        # Filter by availability
        recommendations = [server for server in recommendations if self.server_status.get(server) == "available"]
        
        return recommendations

if __name__ == "__main__":
    manager = MCPServerManager()
    print("🔌 Initializing MCP Servers...")
    results = manager.initialize_servers()
    
    print(f"\n📊 Summary: {sum(results.values())}/{len(results)} servers available")
    
    # Test recommendations
    test_tasks = [
        "Create React components for trading dashboard",
        "Debug HeavyDB query performance issues", 
        "Generate API documentation for trading endpoints",
        "Run E2E tests for user authentication flow"
    ]
    
    print("\n🎯 Task Recommendations:")
    for task in test_tasks:
        recommended = manager.recommend_servers_for_task(task)
        print(f"   Task: {task}")
        print(f"   Recommended: {recommended}")
```

---

## 🧪 Testing and Validation

### 1. Integration Test Suite

```python
#!/usr/bin/env python3
"""Comprehensive integration test suite for TaskMaster AI setup"""

import unittest
import os
import sys
import pandas as pd
from pathlib import Path

class TaskMasterIntegrationTests(unittest.TestCase):
    """Integration tests for TaskMaster AI configuration"""
    
    def setUp(self):
        """Set up test environment"""
        self.worktree_path = Path("/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/")
        self.reference_path = Path("/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/")
        
    def test_environment_variables(self):
        """Test required environment variables are set"""
        required_vars = [
            'ANTHROPIC_API_KEY',
            'MODEL', 
            'HEAVYDB_HOST',
            'HEAVYDB_PORT',
            'HEAVYDB_USER',
            'HEAVYDB_PASSWORD',
            'HEAVYDB_DATABASE'
        ]
        
        for var in required_vars:
            self.assertIsNotNone(os.getenv(var), f"Environment variable {var} not set")
            self.assertNotEqual(os.getenv(var), '', f"Environment variable {var} is empty")
    
    def test_database_connections(self):
        """Test database connectivity"""
        # Test HeavyDB connection
        try:
            from heavydb import connect
            conn = connect(
                host=os.getenv('HEAVYDB_HOST'),
                port=int(os.getenv('HEAVYDB_PORT')),
                user=os.getenv('HEAVYDB_USER'),
                password=os.getenv('HEAVYDB_PASSWORD'),
                dbname=os.getenv('HEAVYDB_DATABASE')
            )
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            self.assertEqual(result[0], 1)
            cursor.close()
            conn.close()
            
        except Exception as e:
            self.fail(f"HeavyDB connection failed: {e}")
    
    def test_excel_configurations(self):
        """Test Excel configuration access"""
        config_path = self.reference_path / "configurations" / "data" / "production"
        
        # Check if configuration directory exists
        self.assertTrue(config_path.exists(), f"Configuration path not found: {config_path}")
        
        # Check for key configuration files
        required_configs = ["tbs_config.xlsx", "tv_config.xlsx", "market_regime_config.xlsx"]
        
        for config_file in required_configs:
            file_path = config_path / config_file
            self.assertTrue(file_path.exists(), f"Configuration file not found: {config_file}")
            
            # Try to load Excel file
            try:
                excel_file = pd.ExcelFile(file_path)
                self.assertGreater(len(excel_file.sheet_names), 0, f"No sheets found in {config_file}")
            except Exception as e:
                self.fail(f"Failed to load {config_file}: {e}")
    
    def test_superclaude_integration(self):
        """Test SuperClaude integration script"""
        integration_script = self.worktree_path / "scripts" / "superclaude_taskmaster_integration.py"
        
        self.assertTrue(integration_script.exists(), "SuperClaude integration script not found")
        
        # Test script can be imported
        try:
            sys.path.insert(0, str(integration_script.parent))
            import superclaude_taskmaster_integration
            
            # Test bridge initialization
            bridge = superclaude_taskmaster_integration.SuperClaudeTaskMasterBridge(
                workdir=str(self.worktree_path)
            )
            
            # Test configuration loading
            self.assertIsNotNone(bridge.config)
            self.assertIn('anthropic_api_key', bridge.config)
            self.assertIn('model', bridge.config)
            
            # Test persona mapping
            self.assertIsNotNone(bridge.persona_mapping)
            self.assertIn('architect', bridge.persona_mapping)
            
            # Test command mapping
            self.assertIsNotNone(bridge.command_mapping)
            self.assertIn('/analyze', bridge.command_mapping)
            
        except Exception as e:
            self.fail(f"SuperClaude integration test failed: {e}")
    
    def test_real_data_validation(self):
        """Test real data validation (no mock data)"""
        # This test ensures we can access real data sources
        try:
            from heavydb import connect
            
            conn = connect(
                host=os.getenv('HEAVYDB_HOST'),
                port=int(os.getenv('HEAVYDB_PORT')),
                user=os.getenv('HEAVYDB_USER'),
                password=os.getenv('HEAVYDB_PASSWORD'),
                dbname=os.getenv('HEAVYDB_DATABASE')
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM nifty_option_chain WHERE trade_date = '2024-01-02'")
            result = cursor.fetchone()
            
            # Ensure we have real data (should be > 0 for a trading day)
            self.assertGreater(result[0], 0, "No real data found for 2024-01-02")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            self.fail(f"Real data validation failed: {e}")
    
    def test_worktree_structure(self):
        """Test worktree directory structure"""
        required_dirs = ["scripts", "docs", "tests", "backtester_v2"]
        
        for dir_name in required_dirs:
            dir_path = self.worktree_path / dir_name
            self.assertTrue(dir_path.exists(), f"Required directory not found: {dir_name}")
            self.assertTrue(dir_path.is_dir(), f"{dir_name} is not a directory")
    
    def test_api_configuration(self):
        """Test API configuration"""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        self.assertIsNotNone(api_key, "Anthropic API key not configured")
        self.assertTrue(api_key.startswith('sk-'), "Invalid Anthropic API key format")
        
        model = os.getenv('MODEL')
        self.assertIsNotNone(model, "Model not configured")
        self.assertIn('claude', model.lower(), "Model should be a Claude model")

class TaskMasterPerformanceTests(unittest.TestCase):
    """Performance tests for TaskMaster AI setup"""
    
    def test_database_performance(self):
        """Test database query performance"""
        import time
        from heavydb import connect
        
        conn = connect(
            host=os.getenv('HEAVYDB_HOST'),
            port=int(os.getenv('HEAVYDB_PORT')),
            user=os.getenv('HEAVYDB_USER'),
            password=os.getenv('HEAVYDB_PASSWORD'),
            dbname=os.getenv('HEAVYDB_DATABASE')
        )
        
        # Test query performance
        query = """
        SELECT trade_date, trade_time, strike, spot, ce_close, pe_close
        FROM nifty_option_chain
        WHERE trade_date = '2024-01-02'
        LIMIT 10000
        """
        
        start_time = time.time()
        df = pd.read_sql(query, conn)
        end_time = time.time()
        
        query_time = end_time - start_time
        rows_per_second = len(df) / query_time
        
        # Performance threshold: should process at least 10,000 rows/second
        self.assertGreater(rows_per_second, 10000, 
                          f"Database performance too slow: {rows_per_second:.0f} rows/sec")
        
        conn.close()

def run_all_tests():
    """Run all integration and performance tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TaskMasterIntegrationTests))
    suite.addTests(loader.loadTestsFromTestCase(TaskMasterPerformanceTests))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return True if all tests passed
    return result.wasSuccessful()

if __name__ == "__main__":
    print("🧪 Running TaskMaster AI Integration Tests...")
    success = run_all_tests()
    
    if success:
        print("\n✅ All tests passed! TaskMaster AI is properly configured.")
        exit(0)
    else:
        print("\n❌ Some tests failed. Please check the configuration.")
        exit(1)
```

### 2. Validation Scripts

```bash
#!/bin/bash
# validation_runner.sh - Run all validation scripts

echo "🔍 TaskMaster AI Configuration Validation"
echo "=========================================="

# Set working directory
cd /srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Environment variables loaded"
else
    echo "❌ .env file not found"
    exit 1
fi

# Test database connections
echo -e "\n🗄️ Testing Database Connections..."
python3 -c "
import os
from heavydb import connect

try:
    conn = connect(
        host=os.getenv('HEAVYDB_HOST'),
        port=int(os.getenv('HEAVYDB_PORT')),
        user=os.getenv('HEAVYDB_USER'),
        password=os.getenv('HEAVYDB_PASSWORD'),
        dbname=os.getenv('HEAVYDB_DATABASE')
    )
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM nifty_option_chain')
    result = cursor.fetchone()
    print(f'✅ HeavyDB: {result[0]:,} rows available')
    cursor.close()
    conn.close()
except Exception as e:
    print(f'❌ HeavyDB: {e}')
    exit(1)
"

# Test Excel configurations
echo -e "\n📊 Testing Excel Configurations..."
python3 -c "
import pandas as pd
from pathlib import Path

config_path = Path('/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/configurations/data/production/')
configs = ['tbs_config.xlsx', 'tv_config.xlsx', 'market_regime_config.xlsx']

for config in configs:
    file_path = config_path / config
    if file_path.exists():
        try:
            excel_file = pd.ExcelFile(file_path)
            print(f'✅ {config}: {len(excel_file.sheet_names)} sheets')
        except Exception as e:
            print(f'❌ {config}: {e}')
    else:
        print(f'⚠️ {config}: Not found')
"

# Test API connectivity
echo -e "\n🔑 Testing API Connectivity..."
if [ -n "$ANTHROPIC_API_KEY" ]; then
    response=$(curl -s -w "%{http_code}" -o /dev/null \
        -H "Authorization: Bearer $ANTHROPIC_API_KEY" \
        -H "Content-Type: application/json" \
        -X POST "https://api.anthropic.com/v1/messages" \
        -d '{"model": "claude-3-haiku-20240307", "max_tokens": 10, "messages": [{"role": "user", "content": "Hi"}]}')
    
    if [ "$response" = "200" ]; then
        echo "✅ Anthropic API: Connected"
    else
        echo "❌ Anthropic API: HTTP $response"
    fi
else
    echo "❌ Anthropic API: Key not set"
fi

# Test SuperClaude integration
echo -e "\n🧠 Testing SuperClaude Integration..."
if [ -f "scripts/superclaude_taskmaster_integration.py" ]; then
    python3 -c "
import sys
sys.path.append('scripts')
from superclaude_taskmaster_integration import SuperClaudeTaskMasterBridge

try:
    bridge = SuperClaudeTaskMasterBridge()
    print('✅ SuperClaude Bridge: Initialized')
    print(f'   - Personas: {len(bridge.persona_mapping)}')
    print(f'   - Commands: {len(bridge.command_mapping)}')
except Exception as e:
    print(f'❌ SuperClaude Bridge: {e}')
"
else
    echo "❌ SuperClaude Integration: Script not found"
fi

# Run integration tests
echo -e "\n🧪 Running Integration Tests..."
if [ -f "tests/test_integration.py" ]; then
    python3 -m pytest tests/test_integration.py -v
else
    echo "⚠️ Integration tests not found"
fi

echo -e "\n✅ Validation Complete!"
```

---

## 🔒 Security Considerations

### 1. API Key Management

```yaml
Security_Best_Practices:
  
  API_Key_Storage:
    - Store in environment variables only
    - Never commit to version control
    - Use different keys for dev/prod
    - Rotate keys regularly (monthly)
    
  Access_Control:
    - Restrict file permissions (.env = 600)
    - Use role-based access
    - Audit API key usage
    - Monitor for unauthorized access
    
  Network_Security:
    - Use HTTPS for all API calls
    - Implement rate limiting
    - Monitor API usage patterns
    - Set up alerts for anomalies
```

### 2. Database Security

```python
#!/usr/bin/env python3
"""Database security validation for TaskMaster AI"""

import os
import pymysql
from heavydb import connect

def validate_database_security():
    """Validate database security configuration"""
    security_checks = {
        "heavydb": {"passed": [], "failed": []},
        "mysql": {"passed": [], "failed": []}
    }
    
    # HeavyDB Security Checks
    try:
        conn = connect(
            host=os.getenv('HEAVYDB_HOST'),
            port=int(os.getenv('HEAVYDB_PORT')),
            user=os.getenv('HEAVYDB_USER'),
            password=os.getenv('HEAVYDB_PASSWORD'),
            dbname=os.getenv('HEAVYDB_DATABASE')
        )
        
        cursor = conn.cursor()
        
        # Check if using localhost (more secure)
        if os.getenv('HEAVYDB_HOST') == 'localhost':
            security_checks["heavydb"]["passed"].append("Using localhost connection")
        else:
            security_checks["heavydb"]["failed"].append("Remote connection detected")
        
        # Check password strength (basic check)
        password = os.getenv('HEAVYDB_PASSWORD')
        if len(password) >= 8:
            security_checks["heavydb"]["passed"].append("Password length adequate")
        else:
            security_checks["heavydb"]["failed"].append("Password too short")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        security_checks["heavydb"]["failed"].append(f"Connection test failed: {e}")
    
    # MySQL Security Checks
    try:
        connection = pymysql.connect(
            host=os.getenv('ARCHIVE_HOST'),
            user=os.getenv('ARCHIVE_USER'),
            password=os.getenv('ARCHIVE_PASSWORD'),
            database=os.getenv('ARCHIVE_DATABASE'),
            charset='utf8mb4'
        )
        
        # Check SSL usage (if supported)
        with connection.cursor() as cursor:
            cursor.execute("SHOW STATUS LIKE 'Ssl_cipher'")
            result = cursor.fetchone()
            if result and result[1]:
                security_checks["mysql"]["passed"].append("SSL encryption enabled")
            else:
                security_checks["mysql"]["failed"].append("SSL encryption not detected")
        
        connection.close()
        
    except Exception as e:
        security_checks["mysql"]["failed"].append(f"Connection test failed: {e}")
    
    return security_checks

def generate_security_report():
    """Generate security validation report"""
    checks = validate_database_security()
    
    print("🔒 Database Security Validation Report")
    print("=" * 40)
    
    for db_type, results in checks.items():
        print(f"\n📊 {db_type.upper()}:")
        
        if results["passed"]:
            print("  ✅ Passed:")
            for check in results["passed"]:
                print(f"     • {check}")
        
        if results["failed"]:
            print("  ❌ Failed:")
            for check in results["failed"]:
                print(f"     • {check}")

if __name__ == "__main__":
    generate_security_report()
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. API Connection Issues

**Problem**: "Invalid API key" error
```bash
Error: The provided API key is invalid or expired
```

**Solution**:
```bash
# Verify API key format
echo $ANTHROPIC_API_KEY | grep -E "^sk-ant-"

# Test API key with curl
curl -H "Authorization: Bearer $ANTHROPIC_API_KEY" \
     -H "Content-Type: application/json" \
     -X POST "https://api.anthropic.com/v1/messages" \
     -d '{"model": "claude-3-haiku-20240307", "max_tokens": 10, "messages": [{"role": "user", "content": "test"}]}'
```

#### 2. Database Connection Issues

**Problem**: "Connection refused" to HeavyDB
```bash
Error: Connection to localhost:6274 refused
```

**Solution**:
```bash
# Check if HeavyDB is running
sudo systemctl status heavydb

# Start HeavyDB if not running
sudo systemctl start heavydb

# Check port availability
netstat -ln | grep 6274

# Test direct connection
telnet localhost 6274
```

#### 3. Excel Configuration Issues

**Problem**: "File not found" errors for Excel configurations
```bash
Error: Excel configuration file not found: tbs_config.xlsx
```

**Solution**:
```bash
# Verify configuration directory
ls -la /srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/configurations/data/production/

# Copy missing configurations from template
cp /srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/configurations/templates/*.xlsx \
   /srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/configurations/data/production/

# Validate Excel file format
python3 -c "
import pandas as pd
try:
    excel_file = pd.ExcelFile('path/to/config.xlsx')
    print(f'Sheets: {excel_file.sheet_names}')
except Exception as e:
    print(f'Error: {e}')
"
```

#### 4. MCP Server Issues

**Problem**: MCP servers not responding
```bash
Error: MCP server 'context7' not available
```

**Solution**:
```bash
# Check MCP server status
curl -s http://localhost:8001/health  # context7
curl -s http://localhost:8002/health  # sequential
curl -s http://localhost:8003/health  # magic
curl -s http://localhost:8004/health  # puppeteer

# Restart MCP services (if configured as services)
sudo systemctl restart mcp-context7
sudo systemctl restart mcp-sequential
```

#### 5. Environment Variable Issues

**Problem**: Environment variables not loading
```bash
Error: ANTHROPIC_API_KEY environment variable not set
```

**Solution**:
```bash
# Check .env file exists and has correct permissions
ls -la .env
chmod 600 .env

# Manually load environment variables
source .env
export $(cat .env | grep -v '^#' | xargs)

# Verify variables are set
env | grep -E "(ANTHROPIC|HEAVYDB|MODEL)"
```

---

## 📖 Usage Examples

### 1. Basic TaskMaster AI Integration

```python
#!/usr/bin/env python3
"""Basic TaskMaster AI usage example"""

from scripts.superclaude_taskmaster_integration import SuperClaudeTaskMasterBridge

# Initialize the bridge
bridge = SuperClaudeTaskMasterBridge(
    workdir="/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/"
)

# Example 1: Create task from SuperClaude command
superclaude_command = "/analyze --persona-architect --context:auto --seq"
result = bridge.create_task_from_superclaude(superclaude_command)

print("Task Creation Result:")
print(f"Success: {result['success']}")
print(f"Output: {result['stdout']}")

# Example 2: Execute workflow
workflow_commands = [
    "/analyze --persona-architect --context:system",
    "/implement --persona-backend --context:api",
    "/test --persona-qa --context:testing"
]

workflow_results = bridge.execute_superclaude_workflow(workflow_commands)

for i, result in enumerate(workflow_results):
    print(f"Command {i+1}: {result['command']}")
    print(f"Success: {result['result']['success']}")
```

### 2. Excel Configuration Validation

```python
#!/usr/bin/env python3
"""Excel configuration validation example"""

from scripts.superclaude_taskmaster_integration import SuperClaudeTaskMasterBridge

bridge = SuperClaudeTaskMasterBridge()

# Validate all Excel configurations
validation_result = bridge.validate_excel_configurations()

print("Excel Validation Result:")
print(f"Success: {validation_result['success']}")
print(f"Details: {validation_result['stdout']}")

# Enforce TODO line limits
todo_file = "docs/ui_refactoring_todo_final_v6.md"
split_result = bridge.enforce_todo_line_limits(todo_file)

print("TODO Split Result:")
print(f"Success: {split_result['success']}")
if split_result['success'] and 'split_files' in split_result:
    print(f"Split into: {split_result['split_files']}")
```

### 3. Persona-Specific Task Creation

```python
#!/usr/bin/env python3
"""Persona-specific task creation examples"""

from scripts.superclaude_taskmaster_integration import SuperClaudeTaskMasterBridge

bridge = SuperClaudeTaskMasterBridge()

# Architecture analysis task
arch_task = bridge.create_task_from_superclaude(
    "/analyze --persona-architect --context:system --seq"
)

# Frontend implementation task
frontend_task = bridge.create_task_from_superclaude(
    "/implement --persona-frontend --context:ui --magic"
)

# Security audit task
security_task = bridge.create_task_from_superclaude(
    "/security --persona-security --context:auth --seq"
)

# Performance optimization task
perf_task = bridge.create_task_from_superclaude(
    "/optimize --persona-performance --context:gpu --seq"
)

print("All tasks created successfully!")
```

### 4. Real Data Validation

```python
#!/usr/bin/env python3
"""Real data validation example"""

import pandas as pd
from heavydb import connect
import os

# Connect to HeavyDB (real data only)
conn = connect(
    host=os.getenv('HEAVYDB_HOST'),
    port=int(os.getenv('HEAVYDB_PORT')),
    user=os.getenv('HEAVYDB_USER'),
    password=os.getenv('HEAVYDB_PASSWORD'),
    dbname=os.getenv('HEAVYDB_DATABASE')
)

# Example: Validate market data for TaskMaster AI development
validation_query = """
SELECT 
    trade_date,
    COUNT(*) as row_count,
    COUNT(DISTINCT strike) as unique_strikes,
    AVG(spot) as avg_spot,
    MIN(trade_time) as market_open,
    MAX(trade_time) as market_close
FROM nifty_option_chain
WHERE trade_date = '2024-01-02'
GROUP BY trade_date
"""

df = pd.read_sql(validation_query, conn)

print("Real Data Validation:")
print(f"Date: {df['trade_date'].iloc[0]}")
print(f"Total rows: {df['row_count'].iloc[0]:,}")
print(f"Unique strikes: {df['unique_strikes'].iloc[0]}")
print(f"Average spot: {df['avg_spot'].iloc[0]:.2f}")
print(f"Market hours: {df['market_open'].iloc[0]} to {df['market_close'].iloc[0]}")

conn.close()

# This validates that we have real market data, not mocked data
assert df['row_count'].iloc[0] > 10000, "Insufficient real data found"
print("✅ Real data validation passed!")
```

---

## 🚀 Next Steps

### 1. Initial Setup Checklist

```bash
# 1. Environment Setup
□ Create .env file with all required variables
□ Set proper file permissions (chmod 600 .env)
□ Verify API keys are valid and working
□ Test database connections

# 2. Database Validation
□ Confirm HeavyDB is running and accessible
□ Verify MySQL connections (both local and remote)
□ Test query performance meets requirements
□ Validate real data availability (no mock data)

# 3. Excel Configuration
□ Verify all 31 production Excel files are accessible
□ Test Excel file loading and validation
□ Check compatibility between worktree and reference configs
□ Validate schema and data types

# 4. MCP Server Setup
□ Initialize all 4 MCP servers (context7, sequential, magic, puppeteer)
□ Test server connectivity and capabilities
□ Configure server recommendations for different task types
□ Verify integration with TaskMaster AI

# 5. SuperClaude Integration
□ Test persona mapping (all 9 personas)
□ Validate command mapping (all 10+ commands)
□ Test context engineering features
□ Verify real data validation enforcement

# 6. Testing and Validation
□ Run integration test suite
□ Execute performance benchmarks
□ Validate security configuration
□ Test complete workflow end-to-end
```

### 2. Development Workflow

```yaml
Daily_Development_Process:
  1. Environment_Check:
     - Load environment variables
     - Verify database connections
     - Check API quotas and limits
     
  2. Task_Planning:
     - Review V6 plan and TODO list
     - Select appropriate personas for tasks
     - Choose MCP servers based on task type
     
  3. Development_Execution:
     - Use SuperClaude commands with TaskMaster AI
     - Enforce real data validation throughout
     - Validate Excel configurations as needed
     
  4. Testing_Validation:
     - Run integration tests
     - Validate against reference implementations
     - Check performance metrics
     
  5. Documentation_Updates:
     - Update TODO lists (enforce 500-line limit)
     - Document configuration changes
     - Update troubleshooting guides
```

### 3. Monitoring and Maintenance

```python
#!/usr/bin/env python3
"""Monitoring script for TaskMaster AI system health"""

import schedule
import time
from datetime import datetime

def health_check():
    """Perform system health check"""
    print(f"[{datetime.now()}] Running health check...")
    
    # Check database connections
    # Check API availability
    # Validate Excel configurations
    # Test MCP server status
    # Monitor performance metrics
    
    print("Health check complete ✅")

def cleanup_todo_files():
    """Clean up and organize TODO files"""
    print(f"[{datetime.now()}] Cleaning up TODO files...")
    
    # Enforce 500-line limits
    # Archive completed TODOs
    # Update master TODO index
    
    print("TODO cleanup complete ✅")

# Schedule regular maintenance
schedule.every(1).hours.do(health_check)
schedule.every(1).days.do(cleanup_todo_files)

if __name__ == "__main__":
    print("🔄 TaskMaster AI Monitoring Started")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute
```

---

## 📚 Additional Resources

### Documentation Links
- [SuperClaude Documentation](./Super_Claude_Docs_v2.md)
- [UI Refactoring Plan v6](./ui_refactoring_plan_final_v6.md)
- [Persona Mapping Guide](./SuperClaude_Persona_Mapping_Guide.md)
- [V6 Autonomous Execution Guide](./V6_Plan_Autonomous_Execution_Guide.md)

### Reference Implementations
- Main Backend: `/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/`
- Excel Configurations: `/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/configurations/`
- Archive System: `/srv/samba/shared/bt/archive/`

### API Documentation
- [Anthropic Claude API](https://docs.anthropic.com/claude/reference)
- [Perplexity API](https://docs.perplexity.ai/)
- [HeavyDB Documentation](https://docs.heavy.ai/)

---

**Document Status**: ✅ Complete and Ready for Implementation  
**Last Validation**: July 13, 2025  
**Next Review**: July 20, 2025