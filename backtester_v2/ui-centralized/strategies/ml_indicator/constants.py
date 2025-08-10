"""
Constants for ML Indicator Strategy
"""

# Default strategy parameters
DEFAULT_LOOKBACK_PERIOD = 20
DEFAULT_PREDICTION_HORIZON = 5
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_RISK_PERCENT = 2.0
DEFAULT_POSITION_SIZE = 100000

# Indicator group names
INDICATOR_GROUPS = ["TREND", "MOMENTUM", "VOLATILITY", "VOLUME", "PATTERN", "SMC"]

# ML models supported
ML_MODELS = ["XGBOOST", "LIGHTGBM", "CATBOOST", "RANDOM_FOREST", "NEURAL_NET", "LSTM", "ENSEMBLE"]

# Query templates
QUERY_TEMPLATES = {}

# Database column mappings
DB_COLUMN_MAPPINGS = {
    "timestamp": "trade_date || ' ' || trade_time",
    "open": "open_price",
    "high": "high_price", 
    "low": "low_price",
    "close": "close_price",
    "volume": "volume"
}

# Timeframe to minutes mapping
TIMEFRAME_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440
}

# Error messages
ERROR_MESSAGES = {
    "INVALID_INDICATOR": "Invalid indicator: {}",
    "INVALID_TIMEFRAME": "Invalid timeframe: {}",
    "MISSING_DATA": "Missing required data for calculation"
}

# Indicator categories
INDICATOR_CATEGORIES = {
    "TREND": ["SMA", "EMA", "WMA", "DEMA", "TEMA", "KAMA", "T3", "SAR"],
    "MOMENTUM": ["RSI", "STOCH", "MACD", "ADX", "CCI", "MFI", "WILLR", "ROC", "MOM"],
    "VOLATILITY": ["BBANDS", "ATR", "NATR", "TRANGE"],
    "VOLUME": ["OBV", "AD", "ADOSC", "MFI"],
    "PATTERN": ["CDLDOJI", "CDLHAMMER", "CDLENGULFING", "CDLMORNINGSTAR"],
    "SMC": ["BOS", "CHOCH", "ORDER_BLOCK", "FVG", "LIQUIDITY_POOL"]
}

# TA-Lib indicator parameters
TALIB_PARAMS = {
    "SMA": {"timeperiod": 20},
    "EMA": {"timeperiod": 20},
    "RSI": {"timeperiod": 14},
    "MACD": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
    "BBANDS": {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2, "matype": 0},
    "STOCH": {"fastk_period": 5, "slowk_period": 3, "slowd_period": 3},
    "ADX": {"timeperiod": 14},
    "ATR": {"timeperiod": 14},
    "CCI": {"timeperiod": 20},
    "MFI": {"timeperiod": 14},
    "WILLR": {"timeperiod": 14},
    "SAR": {"acceleration": 0.02, "maximum": 0.2}
}

# SMC (Smart Money Concepts) parameters
SMC_PARAMS = {
    "BOS": {
        "lookback": 20,
        "min_swing_strength": 3,
        "confirmation_candles": 2
    },
    "CHOCH": {
        "lookback": 20,
        "trend_lookback": 50,
        "min_reversal_strength": 0.005
    },
    "ORDER_BLOCK": {
        "lookback": 50,
        "min_volume_multiplier": 1.5,
        "max_retest_count": 3
    },
    "FVG": {
        "min_gap_size": 0.001,
        "max_fill_ratio": 0.5,
        "lookback": 20
    },
    "LIQUIDITY_POOL": {
        "lookback": 100,
        "cluster_threshold": 0.002,
        "min_touches": 2
    }
}

# Candlestick patterns
CANDLESTICK_PATTERNS = {
    "BULLISH": [
        "CDLHAMMER", "CDLMORNINGSTAR", "CDLENGULFING", "CDLPIERCING",
        "CDLHARAMI", "CDLMARUBOZU", "CDLDRAGONFLYDOJI"
    ],
    "BEARISH": [
        "CDLSHOOTINGSTAR", "CDLEVENINGSTAR", "CDLENGULFING", "CDLDARKCLOUDCOVER",
        "CDLHARAMI", "CDLMARUBOZU", "CDLGRAVESTONEDOJI"
    ],
    "NEUTRAL": [
        "CDLDOJI", "CDLSPINNINGTOP", "CDLHIGHWAVE", "CDLLONGLEGGEDDOJI"
    ]
}

# ML model default parameters
ML_MODEL_PARAMS = {
    "XGBOOST": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss"
    },
    "LIGHTGBM": {
        "num_leaves": 31,
        "max_depth": -1,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "objective": "binary",
        "metric": "binary_logloss"
    },
    "RANDOM_FOREST": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt"
    },
    "NEURAL_NET": {
        "hidden_layers": [64, 32, 16],
        "activation": "relu",
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50
    }
}

# Feature engineering settings
FEATURE_ENGINEERING = {
    "PRICE_FEATURES": {
        "returns": [1, 5, 10, 20, 50],
        "log_returns": [1, 5, 10, 20],
        "volatility": [10, 20, 50],
        "price_ratios": ["high_low", "close_open", "close_vwap"]
    },
    "TECHNICAL_FEATURES": {
        "moving_averages": [5, 10, 20, 50, 200],
        "momentum_periods": [5, 10, 14, 20],
        "volatility_periods": [10, 20, 30]
    },
    "MICROSTRUCTURE_FEATURES": {
        "volume_features": ["volume_ratio", "volume_momentum", "vwap_distance"],
        "spread_features": ["bid_ask_spread", "effective_spread"],
        "imbalance_features": ["order_imbalance", "volume_imbalance"]
    }
}

# Signal combination logic
SIGNAL_LOGIC_PARAMS = {
    "AND": {
        "description": "All conditions must be true",
        "min_signals": 2
    },
    "OR": {
        "description": "Any condition must be true",
        "min_signals": 1
    },
    "WEIGHTED": {
        "description": "Weighted combination of signals",
        "default_threshold": 0.6
    },
    "ML_BASED": {
        "description": "ML model determines signal",
        "confidence_threshold": 0.7
    }
}

# Timeframe conversions (to minutes)
TIMEFRAME_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1w": 10080
}

# Session times (IST)
TRADING_SESSIONS = {
    "PRE_MARKET": {"start": "09:00", "end": "09:15"},
    "REGULAR": {"start": "09:15", "end": "15:30"},
    "POST_MARKET": {"start": "15:30", "end": "16:00"},
    "ASIA": {"start": "05:30", "end": "13:30"},
    "LONDON": {"start": "12:30", "end": "21:30"},
    "NEWYORK": {"start": "17:30", "end": "02:30"}
}

# Kill zones for SMC
KILL_ZONES = {
    "ASIA_KZ": {"start": "07:00", "end": "09:00"},
    "LONDON_KZ": {"start": "13:00", "end": "15:00"},
    "NY_KZ": {"start": "19:00", "end": "21:00"},
    "LONDON_CLOSE": {"start": "21:00", "end": "22:00"}
}

# Volume profile settings
VOLUME_PROFILE_SETTINGS = {
    "DEFAULT_BINS": 24,
    "VALUE_AREA_PERCENTAGE": 0.70,
    "POC_SMOOTHING": 3,
    "DELTA_THRESHOLD": 0.6,
    "HVN_THRESHOLD": 1.5,  # High Volume Node
    "LVN_THRESHOLD": 0.5   # Low Volume Node
}

# Risk management defaults
RISK_DEFAULTS = {
    "POSITION_SIZE": 100000,
    "MAX_RISK_PER_TRADE": 0.02,
    "MAX_PORTFOLIO_RISK": 0.06,
    "STOP_LOSS": 0.02,
    "TAKE_PROFIT": 0.03,
    "TRAILING_STOP_ACTIVATION": 0.01,
    "TRAILING_STOP_DISTANCE": 0.005
}

# SQL query templates
QUERY_TEMPLATES = {
    "indicator_base": """
        WITH price_data AS (
            SELECT 
                trade_date,
                trade_time,
                open_price,
                high_price,
                low_price,
                close_price,
                volume,
                underlying_value
            FROM {table_name}
            WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
            AND index_name = '{index_name}'
            ORDER BY trade_date, trade_time
        )
    """,
    
    "indicator_calculation": """
        {indicator_name}_{period} = {calculation}
    """,
    
    "multi_timeframe": """
        WITH tf_{timeframe} AS (
            SELECT 
                DATE_TRUNC('{interval}', trade_time) as bar_time,
                FIRST_VALUE(open_price) as open,
                MAX(high_price) as high,
                MIN(low_price) as low,
                LAST_VALUE(close_price) as close,
                SUM(volume) as volume
            FROM price_data
            GROUP BY bar_time
        )
    """
}

# Error messages
ERROR_MESSAGES = {
    "INVALID_INDICATOR": "Invalid indicator: {indicator}",
    "MISSING_PARAMS": "Missing required parameters for {indicator}: {params}",
    "INVALID_TIMEFRAME": "Invalid timeframe: {timeframe}",
    "NO_DATA": "No data available for the specified period",
    "ML_MODEL_ERROR": "ML model error: {error}",
    "SIGNAL_LOGIC_ERROR": "Signal logic error: {error}",
    "FEATURE_ERROR": "Feature engineering error: {error}"
}

# Performance metrics
PERFORMANCE_METRICS = {
    "RETURNS": ["total_return", "annualized_return", "daily_return"],
    "RISK": ["volatility", "max_drawdown", "var_95", "cvar_95"],
    "RISK_ADJUSTED": ["sharpe_ratio", "sortino_ratio", "calmar_ratio"],
    "TRADING": ["win_rate", "profit_factor", "avg_win_loss_ratio"],
    "ML_SPECIFIC": ["signal_accuracy", "precision", "recall", "f1_score"]
}

# Cache settings
CACHE_SETTINGS = {
    "INDICATOR_TTL": 300,  # 5 minutes
    "ML_PREDICTION_TTL": 60,  # 1 minute
    "FEATURE_TTL": 300,  # 5 minutes
    "MAX_CACHE_SIZE": 10000  # Maximum number of cached items
}

# Column mappings for database
DB_COLUMN_MAPPINGS = {
    "datetime": "trade_date || ' ' || trade_time",
    "open": "open_price",
    "high": "high_price", 
    "low": "low_price",
    "close": "close_price",
    "volume": "volume",
    "open_interest": "open_interest",
    "underlying": "underlying_value",
    "strike": "strike_price",
    "option_type": "option_type",
    "expiry": "expiry_date"
}