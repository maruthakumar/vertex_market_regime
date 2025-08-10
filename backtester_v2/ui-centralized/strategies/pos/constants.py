"""
Constants for POS (Positional) Strategy
"""

# Strategy types
STRATEGY_TYPES = {
    "CALENDAR_SPREAD": "Calendar Spread",
    "IRON_CONDOR": "Iron Condor", 
    "IRON_FLY": "Iron Fly",
    "BUTTERFLY": "Butterfly Spread",
    "STRANGLE": "Strangle",
    "STRADDLE": "Straddle",
    "RATIO_SPREAD": "Ratio Spread",
    "DIAGONAL_SPREAD": "Diagonal Spread",
    "CUSTOM": "Custom Multi-leg"
}

# Maximum limits
MAX_LEGS = 20
MAX_ADJUSTMENTS_PER_LEG = 5
MAX_CONCURRENT_POSITIONS = 10

# Time constants
DEFAULT_ENTRY_TIME = "09:20:00"
DEFAULT_EXIT_TIME = "15:20:00"
ADJUSTMENT_COOLDOWN_MINUTES = 60

# Risk limits
DEFAULT_MAX_PORTFOLIO_RISK = 0.02  # 2%
DEFAULT_MAX_DAILY_LOSS = 0.05  # 5%
DEFAULT_MAX_DRAWDOWN = 0.10  # 10%

# Greek default limits
DEFAULT_GREEK_LIMITS = {
    "max_delta": 100,
    "min_delta": -100,
    "max_gamma": 50,
    "min_gamma": -50,
    "max_theta": 1000,
    "min_theta": -1000,
    "max_vega": 500,
    "min_vega": -500
}

# Transaction costs
DEFAULT_TRANSACTION_COST = 0.0005  # 0.05%
DEFAULT_SLIPPAGE = 0.0001  # 0.01%

# Position sizing methods
POSITION_SIZING_METHODS = {
    "FIXED": "Fixed position size",
    "KELLY": "Kelly criterion",
    "VOLATILITY_BASED": "Volatility-adjusted",
    "RISK_PARITY": "Risk parity",
    "EQUAL_WEIGHT": "Equal weight"
}

# Adjustment triggers
ADJUSTMENT_TRIGGERS = {
    "PRICE_BASED": {
        "description": "Trigger based on underlying price movement",
        "parameters": ["threshold_percentage", "direction"]
    },
    "TIME_BASED": {
        "description": "Trigger based on time to expiry",
        "parameters": ["days_to_expiry", "time_of_day"]
    },
    "GREEK_BASED": {
        "description": "Trigger based on Greek values",
        "parameters": ["greek_type", "threshold_value", "comparison"]
    },
    "PNL_BASED": {
        "description": "Trigger based on P&L",
        "parameters": ["pnl_threshold", "pnl_type"]
    },
    "VOLATILITY_BASED": {
        "description": "Trigger based on volatility changes",
        "parameters": ["vol_threshold", "vol_type"]
    }
}

# Market regimes
MARKET_REGIMES = {
    "BULLISH": {"trend": "up", "volatility": "normal"},
    "BEARISH": {"trend": "down", "volatility": "normal"},
    "NEUTRAL": {"trend": "sideways", "volatility": "normal"},
    "HIGH_VOLATILITY": {"trend": "any", "volatility": "high"},
    "LOW_VOLATILITY": {"trend": "any", "volatility": "low"}
}

# VIX levels
VIX_LEVELS = {
    "LOW": (0, 15),
    "NORMAL": (15, 25),
    "HIGH": (25, 35),
    "EXTREME": (35, 100)
}

# Rebalancing frequencies
REBALANCE_FREQUENCIES = {
    "NEVER": 0,
    "DAILY": 1,
    "WEEKLY": 7,
    "BIWEEKLY": 14,
    "MONTHLY": 30
}

# Strike selection methods
STRIKE_SELECTION_METHODS = {
    "ATM": "At The Money",
    "ITM": "In The Money",
    "OTM": "Out of The Money",
    "STRIKE_PRICE": "Specific Strike Price",
    "DELTA_BASED": "Based on Delta value",
    "PERCENTAGE_BASED": "Percentage from spot"
}

# Common expiry selections
EXPIRY_SELECTIONS = {
    "CURRENT_WEEK": "Current week expiry",
    "NEXT_WEEK": "Next week expiry",
    "CURRENT_MONTH": "Current month expiry",
    "NEXT_MONTH": "Next month expiry",
    "SPECIFIC_DATE": "Specific expiry date"
}

# SQL query templates
QUERY_TEMPLATES = {
    "multi_leg_base": """
        WITH leg_{leg_id} AS (
            SELECT 
                trade_date,
                trade_time,
                strike_price,
                option_type,
                close_price,
                volume,
                open_interest,
                implied_volatility,
                delta,
                gamma,
                theta,
                vega
            FROM {table_name}
            WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
            AND index_name = '{index_name}'
            AND option_type = '{option_type}'
            {strike_filter}
        )
    """,
    
    "greek_calculation": """
        portfolio_delta = SUM(delta * position_size * contract_multiplier),
        portfolio_gamma = SUM(gamma * position_size * contract_multiplier),
        portfolio_theta = SUM(theta * position_size * contract_multiplier),
        portfolio_vega = SUM(vega * position_size * contract_multiplier)
    """,
    
    "adjustment_check": """
        CASE 
            WHEN {trigger_condition} THEN 1
            ELSE 0
        END as adjustment_trigger_{rule_id}
    """
}

# Error messages
ERROR_MESSAGES = {
    "INVALID_LEG_COUNT": "Number of legs must be between 1 and {max_legs}",
    "INVALID_STRIKE_SELECTION": "Invalid strike selection method: {method}",
    "MISSING_DELTA_TARGET": "Delta target required for DELTA_BASED strike selection",
    "INVALID_ADJUSTMENT_TRIGGER": "Invalid adjustment trigger type: {trigger}",
    "GREEK_LIMIT_EXCEEDED": "Greek limit exceeded: {greek} = {value}, limit = {limit}",
    "POSITION_SIZE_EXCEEDED": "Position size exceeds maximum allowed",
    "INSUFFICIENT_DATA": "Insufficient data for date range: {start_date} to {end_date}"
}

# Column mappings for database
DB_COLUMN_MAPPINGS = {
    "trade_date": "trade_date",
    "trade_time": "trade_time",
    "index_name": "index_name",
    "expiry_date": "expiry_date",
    "strike_price": "strike_price",
    "option_type": "option_type",
    "open": "open_price",
    "high": "high_price",
    "low": "low_price",
    "close": "close_price",
    "volume": "volume",
    "open_interest": "open_interest",
    "implied_volatility": "implied_volatility",
    "underlying_price": "underlying_value",
    "delta": "delta",
    "gamma": "gamma",
    "theta": "theta",
    "vega": "vega",
    "rho": "rho"
}